"""Raw Anthropic SDK driver — the no-framework rail.

Runs the two snippets ``https://docs.decimal.ai/sdk/python/frameworks`` documents for
code that calls the provider SDK directly, with no agent framework in between:

* tracing — ``init(anthropic=True)`` (here: its body,
  ``decimalai.providers.instrument(anthropic=True)``), which turns on the
  OpenInference Anthropic instrumentor and routes its OTel spans through
  ``decimalai.otel.DecimalSpanExporter``;
* skills — ``decimalai.anthropic.instrument(enable_skill_loader=True)``, which
  patches ``client.messages.create()`` so the routed skill fragment is
  prepended to ``system``.

The run itself is the ordinary Anthropic tool-use loop: ``messages.create`` →
``stop_reason="tool_use"`` → run the tool → ``messages.create`` with the
``tool_result``. That is what a raw-SDK agent is; nothing about it is
DecimalAI-specific.

**Where the stub sits.** The `anthropic` package has no model object to swap,
so the stub is the Anthropic *service*: a real HTTP server on 127.0.0.1 that
answers ``POST /v1/messages`` with the shared ``stub_script`` as a
Messages-API response, and ``base_url=`` points the real client at it. Every
line of the real `anthropic` SDK still runs — real request building, real
retries, real response models — and the OpenInference instrumentor sees exactly
what it would see in production. No key is used (the funded key has no
balance), no ``api.anthropic.com`` request is made, and the DecimalAI wire
stays a real socket to the probe.

**Why this driver owns its TracerProvider.** ``providers.instrument()`` takes a
``tracer_provider=`` for callers who run their own OTel; this driver passes one
so its exporter is attached to that provider alone. Without it the exporter
lands on the process-global provider — which, in a suite where several drivers
route through OTel in one process, means one run's spans reach another driver's
exporter and every count doubles. Nothing about the traced path changes: same
instrumentor, same exporter, same ``SimpleSpanProcessor``.

**The SDK's own spans go there too.** anthropic >= 1.15 traces its own API calls, on
the provider registered with the OpenTelemetry API unless the client is given one.
Under the documented ``init(anthropic=True)`` that is the provider DecimalAI's
exporter is on, so every call arrives as two spans: the SDK's (model and tokens, no
messages by default) inside OpenInference's (the prompt and the completion). A driver
that left the SDK's spans on the global provider would grade one span where
production exports two, and stayed green while those traces lost their content
(2026-10-20). So a client that has the ``open_telemetry`` option is given this
driver's provider.

**Instrumented once, on purpose.** The OpenInference instrumentor is a
process-wide singleton that binds its tracer the first time it is enabled, so
``agent_name`` is fixed for the process at that moment — a property of the rail,
not a choice made here. The driver still passes each run's name, which is what
the documented API offers.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..delivery import TOOL_LOADED
from . import (
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    FrameworkLimit,
    Shape,
    fanout_threads,
    stub_script,
    tool_result,
    user_message,
)

#: The model id every request names. Anthropic-shaped so nothing in the SDK or
#: the instrumentor treats it as unknown, but not a model that exists.
STUB_MODEL = "claude-conformance-stub"

def _tools(ctx: Ctx) -> List[Dict[str, Any]]:
    """The tool the loop offers, in Anthropic's schema shape."""
    return [{
        "name": ctx.tool_name,
        "description": "Look a value up for the conformance run.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }]


# ── the stub Anthropic service ───────────────────────────────────────────────


def _reply(ctx: Ctx, body: Dict[str, Any]) -> Dict[str, Any]:
    """Map the shared stub script onto a Messages-API response.

    Stateless, so concurrent lanes cannot interleave: the turn is decided by
    what the request carries. A request that already contains a ``tool_result``
    (or that declares no tools at all) gets the final text turn; anything else
    gets the tool-call turn.
    """
    turns = stub_script(ctx, use_tool=bool(body.get("tools")))
    answered = any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for message in body.get("messages") or []
        if isinstance(message, dict) and isinstance(message.get("content"), list)
        for block in message["content"]
    )
    turn = turns[-1] if (answered or len(turns) == 1) else turns[0]

    content: List[Dict[str, Any]] = []
    if turn.tool_call:
        name, args = turn.tool_call
        content.append({
            "type": "tool_use", "id": "toolu_conformance", "name": name, "input": args,
        })
    if turn.content:
        content.append({"type": "text", "text": turn.content})
    return {
        "id": "msg_conformance",
        "type": "message",
        "role": "assistant",
        "model": body.get("model") or STUB_MODEL,
        "stop_reason": "tool_use" if turn.tool_call else "end_turn",
        "stop_sequence": None,
        "content": content,
        "usage": {
            "input_tokens": turn.input_tokens,
            "output_tokens": turn.output_tokens,
        },
    }


def _sse(message: Dict[str, Any]) -> bytes:
    """One Messages-API response as the server-sent events ``stream: true`` gets.

    The event sequence the Messages API documents for streaming: a
    ``message_start`` with no content, each content block as
    start / delta / stop (text as a ``text_delta``, a tool call's input as one
    ``input_json_delta``), then ``message_delta`` with the stop reason and the
    output token count, then ``message_stop``. The SDK's ``MessageStream``
    rebuilds the same ``Message`` the non-streaming ``_reply`` would have
    returned from these, so both paths answer the same script.
    """
    events: List[Tuple[str, Dict[str, Any]]] = [(
        "message_start",
        {"type": "message_start", "message": {
            **message, "content": [], "stop_reason": None,
            "usage": {"input_tokens": message["usage"]["input_tokens"], "output_tokens": 1},
        }},
    )]
    for index, block in enumerate(message["content"]):
        if block["type"] == "tool_use":
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        events += [
            ("content_block_start",
             {"type": "content_block_start", "index": index, "content_block": start}),
            ("content_block_delta",
             {"type": "content_block_delta", "index": index, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": index}),
        ]
    events += [
        ("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": message["usage"]["output_tokens"]},
        }),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    ).encode("utf-8")


def _start_provider(ctx: Ctx, *, fail: bool = False) -> Tuple[Any, str]:
    """A real HTTP Anthropic stand-in for one run. Returns (server, base_url).

    Answers ``stream: true`` requests (``messages.stream()``) as server-sent
    events and everything else as one JSON body — the two ways the Messages API
    answers the same request.
    """

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: Any) -> None:  # noqa: A003 - silence stderr
            pass

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except Exception:
                body = {}
            if fail:
                # 400 rather than 5xx: the SDK does not retry it, so a failing
                # run is one request, one span, one trace.
                status = 400
                payload: Dict[str, Any] = {
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": "conformance: the provider rejected this on purpose",
                    },
                }
            else:
                status, payload = 200, _reply(ctx, body)
            streamed = status == 200 and bool(body.get("stream"))
            data = _sse(payload) if streamed else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header(
                "Content-Type", "text/event-stream" if streamed else "application/json"
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    host, port = server.server_address[:2]
    return server, f"http://{host}:{port}"


# ── the documented snippet ───────────────────────────────────────────────────

_pipeline_lock = threading.Lock()
_pipeline: Any = None


def _instrument(ctx: Ctx) -> None:
    """``init(anthropic=True)``, on a TracerProvider this driver owns.

    Called by every run; only the first one instruments, because the
    OpenInference instrumentor is a process-wide singleton and a second
    ``instrument()`` against an explicit provider would attach a second exporter
    (and double every trace).
    """
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return
        from opentelemetry.sdk.trace import TracerProvider

        from decimalai import providers

        provider = TracerProvider()
        providers.instrument(
            anthropic=True, agent_name=ctx.agent_name, tracer_provider=provider
        )
        _pipeline = provider


def _client(base_url: str) -> Any:
    import inspect

    import anthropic

    kwargs: Dict[str, Any] = {}
    if "open_telemetry" in inspect.signature(anthropic.Anthropic.__init__).parameters:
        # The SDK's own spans, on the provider the exporter is on (see the module
        # docstring): what the global provider gets in production.
        kwargs["open_telemetry"] = {"tracer_provider": _pipeline}
    # A placeholder the SDK accepts; the stub service never looks at it, and no
    # request leaves 127.0.0.1.
    return anthropic.Anthropic(api_key="sk-ant-conformance-stub", base_url=base_url,
                               **kwargs)


def _send(client: Any, *, stream: bool, **kwargs: Any) -> Any:
    """One model request: ``messages.create()``, or ``messages.stream()`` read
    to its final message the way the SDK's helpers document."""
    if not stream:
        return client.messages.create(**kwargs)
    with client.messages.stream(**kwargs) as events:
        return events.get_final_message()


def _loop(
    ctx: Ctx, client: Any, *, tools: bool = True, system: Optional[str] = SYSTEM_PROMPT,
    history: Sequence[Dict[str, Any]] = (), stream: bool = False,
) -> Any:
    """The documented Anthropic tool-use loop, verbatim in shape.

    ``history`` is the conversation so far — earlier turns' user and assistant
    messages, ahead of this turn's user message, which is how a raw-SDK
    application continues a conversation.
    """
    kwargs: Dict[str, Any] = {"model": STUB_MODEL, "max_tokens": 512}
    if system is not None:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = _tools(ctx)

    messages: List[Dict[str, Any]] = [
        *history, {"role": "user", "content": user_message(ctx)},
    ]
    first = _send(client, stream=stream, messages=messages, **kwargs)
    uses = [b for b in first.content if getattr(b, "type", None) == "tool_use"]
    if not uses:
        return first

    results = []
    for use in uses:
        query = (use.input or {}).get("query", "") if isinstance(use.input, dict) else ""
        results.append({
            "type": "tool_result",
            "tool_use_id": use.id,
            "content": tool_result(ctx, str(query)),
        })
    messages = messages + [
        {"role": "assistant", "content": [b.model_dump() for b in first.content]},
        {"role": "user", "content": results},
    ]
    return _send(client, stream=stream, messages=messages, **kwargs)


def _run(
    ctx: Ctx, *, fail: bool = False, tools: bool = True,
    system: Optional[str] = SYSTEM_PROMPT,
    history: Sequence[Dict[str, Any]] = (), stream: bool = False,
) -> Any:
    _instrument(ctx)
    server, base_url = _start_provider(ctx, fail=fail)
    client = _client(base_url)
    try:
        from decimalai import providers

        # The documented run boundary. A provider instrumentor sees one SDK
        # call at a time and emits an unparented root span for each, so without
        # this a two-call tool loop arrives as two unrelated single-span traces.
        # This is the one thing a raw-SDK user must say themselves: there is no
        # run object for a library to hook, so the boundary has to be declared.
        with providers.agent_run(ctx.agent_name):
            return _loop(
                ctx, client, tools=tools, system=system, history=history, stream=stream,
            )
    finally:
        # Close before the server goes away: a keep-alive socket left to the
        # garbage collector raises a ResourceWarning from whichever phase
        # happens to collect it, and C12 reads warnings as evidence that an
        # adapter spoke up. Don't hand it a false one.
        client.close()
        server.shutdown()
        server.server_close()


def run(ctx: Ctx) -> Any:
    """One raw-SDK run: the tool-use loop against the stub service."""
    return _run(ctx)


def run_error(ctx: Ctx) -> Any:
    """The same run with the provider refusing the request."""
    return _run(ctx, fail=True)


def run_degenerate(ctx: Ctx) -> Any:
    """A bare ``messages.create`` — no tools, no system prompt, nothing to declare."""
    return _run(ctx, tools=False, system=None)


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail: the documented ``messages.create()`` patch, then N lanes.

    Runs last, like every process-wide monkeypatch in this suite — the patch has
    no uninstall, so a rail-enabled ``create()`` would otherwise colour the
    phases above.

    In the ``tool_loaded`` delivery cell the driver ASKS for the tool loop
    instead of quietly not asking, so the adapter has to refuse out loud on this
    run — see ``contract.grade_delivery``, which will not accept the N/A without
    that refusal. Asking is not an assertion.
    """
    from decimalai.anthropic import instrument

    instrument(
        enable_skill_loader=True,
        enable_load_skill_tool=ctxs[0].delivery_mode == TOOL_LOADED,
    )
    return fanout_threads(run)(ctxs)


def _rail_on() -> None:
    """The docs' one-line skills setup — ``instrument(enable_skill_loader=True)``,
    with no agent name, exactly as ``frameworks.mdx`` shows it. A no-op once the
    skills phase has patched the client."""
    from decimalai.anthropic import instrument

    instrument(enable_skill_loader=True)


def run_unnamed(ctx: Ctx) -> Any:
    """The rail, with the run named where a raw-SDK run is named: ``agent_run()``.

    There is no agent object on this rail at all. The documented way to say
    whose run this is is the run boundary, ``providers.agent_run(name)``, and
    the documented skills setup names no agent — so this is the docs' setup,
    for an agent the skills phase did not run.
    """
    _rail_on()
    return _run(ctx)


def run_conversation(ctxs: Sequence[Ctx]) -> Any:
    """Two turns: the second ``messages.create()`` loop is sent the first turn's
    user message and the assistant's answer ahead of the new user message."""
    _rail_on()
    first, second = ctxs
    answer = _run(first)
    text = "".join(
        getattr(b, "text", "") or "" for b in getattr(answer, "content", None) or []
    )
    history = [
        {"role": "user", "content": user_message(first)},
        {"role": "assistant", "content": text or "(no answer)"},
    ]
    return _run(second, history=history)


def _stream_shape(ctxs: Sequence[Ctx]) -> Any:
    """The same tool loop, every request sent with ``client.messages.stream()``.

    The Messages API's documented streaming helper. In anthropic 1.x it builds
    its own request and never goes through ``messages.create()``.
    """
    _rail_on()
    return [_run(ctx, stream=True) for ctx in ctxs]


DRIVER = Driver(
    name="anthropic",
    # The docs' capability table has no raw-Anthropic row: this rail is
    # advertised in the prose "No framework at all" section instead. Claiming a
    # slug here would fail the coverage guard's orphan check, and rightly so.
    covers=frozenset(),
    requires=("anthropic", "openinference.instrumentation.anthropic"),
    frameworks=("anthropic", "openinference-instrumentation-anthropic"),
    entrypoint="decimalai.providers.instrument(anthropic=True) / decimalai.anthropic.instrument()",
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    run_unnamed=run_unnamed,
    run_conversation=run_conversation,
    shapes=(
        Shape(
            name="messages_stream",
            doc=(
                "the tool-use loop sent with client.messages.stream() under "
                "decimalai.anthropic.instrument(enable_skill_loader=True)"
            ),
            run=_stream_shape,
            routed=True,
        ),
    ),
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        model_can_load_skill_bodies=False,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
        has_invocation_observer=False,
        reasons={
            "has_invocation_observer": (
                "decimalai/anthropic.py hands its caller no run: instrument() takes no "
                "on_trace, so C15 has no observer to grade. Its traces are assembled "
                "by the OTel exporter, which links a run to an enclosing "
                "decimalai.start_trace() only where an adapter recorded that link when "
                "the run started — decimalai/crewai.py does, this adapter does not — "
                "so C16 is N/A with C15. A known gap: a run wrapped in start_trace() "
                "ships unlinked, and a routed delivery is credited on both traces. Not "
                "a framework limit and not this suite's opinion: test_coverage reads "
                "this flag off instrument()'s own signature, so the commit that gives "
                "it on_trace turns both items on."
            ),
            "model_can_load_skill_bodies": (
                "this rail is prompt-injection only. enable_load_skill_tool is accepted "
                "but DORMANT on this adapter — decimalai/anthropic.py logs "
                "'enable_load_skill_tool is not supported on the anthropic adapter (no "
                "tool loop to route the result); staying on prompt injection. Use "
                "openai_agents or pydantic_ai for the native load_skill tool' — because a "
                "single patched messages.create() cannot route a tool result back "
                "mid-turn. The model therefore has no way to ASK for a body, so the "
                "strongest rung observable here is DELIVERED, and delivery is not "
                "activation. snippets/silent-noops.mdx already says it: 'activation isn't "
                "measurable for bare prompt-injection usage.' C13 still applies and is "
                "graded: with no loader, a delivered body is exactly what is most likely "
                "to be promoted to a fabricated activation."
            ),
        },
        delivery_limits={
            TOOL_LOADED: FrameworkLimit(
                reason=(
                    "This rail is a patched `client.messages.create()` — ONE provider "
                    "request. The tool loop belongs to the caller's own while-loop, "
                    "which the SDK never sees, so there is no turn for a load_skill "
                    "result to be routed back into. The tool-loaded channel does not "
                    "exist here at any setting of any flag; prompt injection is the "
                    "whole rail, which is why the injected cell is graded strictly "
                    "and is not allowed to be N/A."
                ),
                adapter_module="decimalai/anthropic.py",
                refusal_marker=(
                    "enable_load_skill_tool is not supported on the anthropic adapter"
                ),
            ),
        },
    ),
)
