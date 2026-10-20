"""Raw Anthropic SDK driver — the no-framework rail.

Runs the two snippets the docs give for code that calls the Anthropic SDK
directly, with no agent framework in between:

* tracing — ``init(anthropic=True)`` (here: its body,
  ``decimalai.providers.instrument(anthropic=True)``), which turns on the
  OpenInference Anthropic instrumentor and routes its OTel spans through
  ``decimalai.otel.DecimalSpanExporter``, with ``providers.agent_run(name)``
  around the ordinary tool-use loop as the run boundary. The phases up to
  ``concurrent`` run this and nothing else of DecimalAI;
* an agent's prompt and skills — ``decimalai.anthropic.agent_config(name)``, one
  run per user message, handed to the SDK's own Tool Runner::

      support = agent_config(name)
      with support.run(question) as run:
          client.beta.messages.tool_runner(
              model=..., max_tokens=..., system=run.system,
              tools=[lookup, *run.tools],
              messages=[{"role": "user", "content": question}],
          ).until_done()

  ``nested`` and every phase after it run this. ``run.system`` is the saved
  prompt, then the skill list routed on the run's message; ``run.tools`` is
  ``load_skill``, which the Tool Runner calls in its own loop, so a body comes
  back as a ``tool_result`` mid-run. Opening the run opens its trace, filed
  under the DecimalAI agent, and ``agent_config()`` turns on the same tracing as
  ``init(anthropic=True)``. Nothing in the SDK is patched, so every rail phase
  builds its run that way (``Driver.loads_agent_from_platform``: the harness
  puts each phase's agent in the workspace first, as the dashboard would).

The second shapes are the other two loops the module documents: a hand-written
``messages.create()`` loop (and the same loop over ``messages.stream()``) taking
``run.tool_params`` and ``run.tool_result(block)``, and ``async with`` around the async
Tool Runner on ``AsyncAnthropic``.

**Where the stub sits.** The `anthropic` package has no model object to swap,
so the stub is the Anthropic *service*: a real HTTP server on 127.0.0.1 that
answers ``POST /v1/messages`` with the shared ``stub_script`` as a
Messages-API response, and ``base_url=`` points the real client at it. Every
line of the real `anthropic` SDK still runs — real request building, real
retries, real response models, the real Tool Runner — and the OpenInference
instrumentor sees exactly what it would see in production. No key is used, no
``api.anthropic.com`` request is made, and the DecimalAI wire stays a real
socket to the probe.

**Why tracing is on the process-global provider.** ``agent_config()`` turns
tracing on with ``providers.instrument(anthropic=True)``, the global-provider
path. This driver used to pass a ``tracer_provider=`` of its own, so that other
drivers' spans in the same process could not reach its exporter; each driver now
runs in its own process (``isolation.py``), and the two cannot be mixed: an
explicit provider leaves the rail unmarked as instrumented, so ``agent_config()``
would attach a second exporter to the global provider and move ``agent_run()``
there, while the instrumentor (a process-wide singleton, bound once) kept
sending the model spans to the first one. A run would arrive as two halves.

**The SDK's own spans go there too.** anthropic >= 1.15 traces its own API calls, on
the provider registered with the OpenTelemetry API unless the client is given one: here,
as in production, the one DecimalAI's exporter is on. Every call then arrives as two
spans, the SDK's (model and tokens, no messages by default) inside OpenInference's (the
prompt and the completion), and the driver grades what production exports. (While
this driver had a private provider, it gave the client that provider through the
``open_telemetry`` option for this reason; on the global provider nothing needs
passing.)

**Instrumented once, on purpose.** The OpenInference instrumentor binds its
tracer the first time it is enabled, so the exporter's default ``agent_name``
is fixed for the process at that moment — a property of the rail, not a choice
made here. Each run is still named where the docs name it: ``agent_run(name)``
on the tracing half, ``agent_config(name)`` on the rail.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from ..delivery import INJECTED
from . import (
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    Shape,
    StubTurn,
    fanout_threads,
    stub_script,
    tool_result,
    trace_observer,
    user_message,
)

#: The model id every request names. Anthropic-shaped so nothing in the SDK or
#: the instrumentor treats it as unknown, but not a model that exists.
STUB_MODEL = "claude-conformance-stub"

#: A hand-written loop's bound on model calls. The stub answers on the second,
#: so this only stops a script defect from spinning forever.
_MAX_MODEL_CALLS = 4


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


def _runner_tool(ctx: Ctx, *, asynchronous: bool = False) -> Any:
    """The same tool as the Tool Runner takes it: ``@beta_tool`` on a function."""
    from anthropic import beta_async_tool, beta_tool

    if asynchronous:
        async def alookup(query: str) -> str:
            """Look a value up for the conformance run.

            Args:
                query: What to look up.
            """
            return tool_result(ctx, query)

        return beta_async_tool(alookup, name=ctx.tool_name)

    def lookup(query: str) -> str:
        """Look a value up for the conformance run.

        Args:
            query: What to look up.
        """
        return tool_result(ctx, query)

    return beta_tool(lookup, name=ctx.tool_name)


def _skill_script(ctx: Ctx) -> List[StubTurn]:
    """The rail's script: call ``load_skill``, then answer.

    ``load_skill`` is the rail ``agent_config()`` gives a run, and the Tool Runner
    (or the caller's own loop, through ``run.tool_result``) runs it, so the body comes
    back mid-run. Same shape and same reply text as the shared script; only the
    tool called differs, because the tool IS the mechanism under test.

    In the ``injected`` delivery cell the ``load_skill`` tool is switched OFF in
    this process's environment (``DECIMALAI_LOAD_SKILL_TOOL=0``), so the model is
    scripted onto the ordinary conformance tool instead: asking for a tool that
    does not exist would be a driver artifact dressed up as an adapter defect.
    The body has to arrive in ``run.system`` in that cell, which is the point of it.
    """
    turns = stub_script(ctx)
    if not ctx.skills or ctx.delivery_mode == INJECTED:
        return turns
    ask, answer = turns[0], turns[-1]
    return [
        StubTurn(
            tool_call=("load_skill", {"name": ctx.skills[0]["name"]}),
            content=ask.content,
            input_tokens=ask.input_tokens,
            output_tokens=ask.output_tokens,
        ),
        answer,
    ]


# ── the stub Anthropic service ───────────────────────────────────────────────


def _is_result(block: Any) -> bool:
    return isinstance(block, dict) and block.get("type") == "tool_result"


def _rounds(body: Dict[str, Any]) -> int:
    """How many tool results the current question has had back so far.

    Counted from the last user message that is a question rather than a list of
    tool results, so a conversation's earlier turns do not move this turn's
    place in the script.
    """
    rounds = 0
    for message in body.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list) and any(_is_result(b) for b in content):
            rounds += 1
        else:
            rounds = 0
    return rounds


def _question(body: Dict[str, Any]) -> str:
    """The latest user message that is a question, as text."""
    for message in reversed(body.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list) and any(_is_result(b) for b in content):
            continue
        return content if isinstance(content, str) else json.dumps(content, default=str)
    return ""


def _reply(
    lanes: Sequence[Tuple[Ctx, Optional[List[StubTurn]]]], body: Dict[str, Any],
) -> Dict[str, Any]:
    """Map the lane's script onto a Messages-API response.

    Stateless, so concurrent requests cannot interleave: the lane is the one
    whose prompt sentinel the request's latest question carries (the LONGEST,
    because a conversation's second turn's sentinel extends the first's), and
    the turn is decided by how many tool results that question has had back. A
    lane with no script of its own answers the shared one — with no tool call
    when the request declares no tools.
    """
    question = _question(body)
    ctx, script = max(
        lanes,
        key=lambda lane: (lane[0].prompt_sentinel in question, len(lane[0].prompt_sentinel)),
    )
    turns = script or stub_script(ctx, use_tool=bool(body.get("tools")))
    rounds = _rounds(body)
    turn = turns[min(rounds, len(turns) - 1)]
    return _message_payload(
        body.get("model") or STUB_MODEL, turn, f"toolu_conformance_{rounds}",
    )


def _message_payload(model: str, turn: StubTurn, tool_use_id: str) -> Dict[str, Any]:
    """One ``StubTurn`` as a Messages-API ``Message``: a ``tool_use`` block for its
    tool call, a ``text`` block for its content.

    Shared with the journey's stub model (``journey.JourneyModel``), so the two
    stand-ins for ``POST /v1/messages`` cannot disagree about what it returns.
    """
    content: List[Dict[str, Any]] = []
    if turn.tool_call:
        name, args = turn.tool_call
        content.append({"type": "tool_use", "id": tool_use_id, "name": name, "input": args})
    if turn.content:
        content.append({"type": "text", "text": turn.content})
    return {
        "id": "msg_conformance",
        "type": "message",
        "role": "assistant",
        "model": model,
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


def _start_provider(
    lanes: Sequence[Tuple[Ctx, Optional[List[StubTurn]]]], *, fail: bool = False,
) -> Tuple[Any, str]:
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
                status, payload = 200, _reply(lanes, body)
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


# A placeholder the SDK accepts; the stub service never looks at it, and no
# request leaves 127.0.0.1.
_STUB_KEY = "sk-ant-conformance-stub"


@contextmanager
def _service(
    lanes: Sequence[Tuple[Ctx, Optional[List[StubTurn]]]], *, fail: bool = False,
) -> Iterator[Any]:
    """The stub service and a real ``anthropic.Anthropic`` client pointed at it."""
    import anthropic

    server, base_url = _start_provider(lanes, fail=fail)
    client = anthropic.Anthropic(api_key=_STUB_KEY, base_url=base_url)
    try:
        yield client
    finally:
        # Close before the server goes away: a keep-alive socket left to the
        # garbage collector raises a ResourceWarning from whichever phase
        # happens to collect it, and C12 reads warnings as evidence that an
        # adapter spoke up. Don't hand it a false one.
        client.close()
        server.shutdown()
        server.server_close()


# ── the tracing snippet: init(anthropic=True) ────────────────────────────────

_pipeline_lock = threading.Lock()
_instrumented = False


def _instrument(ctx: Ctx) -> None:
    """``init(anthropic=True)``: its body, on the process-global provider.

    Called by every tracing run; only the first one instruments, for the reason
    the module docstring gives.
    """
    global _instrumented
    with _pipeline_lock:
        if _instrumented:
            return
        from decimalai import providers

        providers.instrument(anthropic=True, agent_name=ctx.agent_name)
        _instrumented = True


def _send(client: Any, *, stream: bool, **kwargs: Any) -> Any:
    """One model request: ``messages.create()``, or ``messages.stream()`` read
    to its final message the way the SDK's helpers document."""
    if not stream:
        return client.messages.create(**kwargs)
    with client.messages.stream(**kwargs) as events:
        return events.get_final_message()


def _loop(ctx: Ctx, client: Any, *, tools: bool = True, system: Optional[str] = SYSTEM_PROMPT) -> Any:
    """The documented Anthropic tool-use loop, verbatim in shape."""
    kwargs: Dict[str, Any] = {"model": STUB_MODEL, "max_tokens": 512}
    if system is not None:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = _tools(ctx)

    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_message(ctx)}]
    first = client.messages.create(messages=messages, **kwargs)
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
    return client.messages.create(messages=messages, **kwargs)


def _traced(ctx: Ctx, *, fail: bool = False, tools: bool = True,
            system: Optional[str] = SYSTEM_PROMPT) -> Any:
    _instrument(ctx)
    with _service([(ctx, None)], fail=fail) as client:
        from decimalai import providers

        # The documented run boundary. A provider instrumentor sees one SDK
        # call at a time and emits an unparented root span for each, so without
        # this a two-call tool loop arrives as two unrelated single-span traces.
        # This is the one thing a tracing-only user must say themselves: there
        # is no run object for a library to hook, so the boundary has to be
        # declared. (agent_config()'s run declares it for them.)
        with providers.agent_run(ctx.agent_name):
            return _loop(ctx, client, tools=tools, system=system)


def run(ctx: Ctx) -> Any:
    """One traced run: the tool-use loop against the stub service."""
    return _traced(ctx)


def run_error(ctx: Ctx) -> Any:
    """The same run with the provider refusing the request."""
    return _traced(ctx, fail=True)


def run_degenerate(ctx: Ctx) -> Any:
    """A bare ``messages.create`` — no tools, no system prompt, nothing to declare."""
    return _traced(ctx, tools=False, system=None)


# ── the rail snippet: agent_config() ───────────────────────────────────────────


def _config(ctx: Ctx, configs: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Any:
    """This agent's ``agent_config()`` config. ``configs`` shares one per agent
    name, as a module-level one would be."""
    from decimalai.anthropic import agent_config

    configs = {} if configs is None else configs
    if ctx.agent_name not in configs:
        configs[ctx.agent_name] = agent_config(ctx.agent_name, **kwargs)
    return configs[ctx.agent_name]


def _tool_runner(ctx: Ctx, client: Any, run: Any,
                 history: Sequence[Dict[str, Any]] = ()) -> Any:
    """The documented snippet's model call: the beta Tool Runner, to the end."""
    return client.beta.messages.tool_runner(
        model=STUB_MODEL,
        max_tokens=512,
        system=run.system,
        tools=[_runner_tool(ctx), *run.tools],
        messages=[*history, run.message],
    ).until_done()


def _tool_runner_turn(ctx: Ctx, client: Any, run: Any, history: List[Dict[str, Any]]) -> Any:
    """One turn of a chat on the Tool Runner, copied into ``history`` as the module's docs
    say: the user turn (``run.message``), then every reply and tool-result turn."""
    history.append(run.message)
    runner = client.beta.messages.tool_runner(
        model=STUB_MODEL, max_tokens=512, system=run.system,
        tools=[_runner_tool(ctx), *run.tools], messages=list(history),
    )
    last = None
    for message in runner:
        last = message
        history.append({"role": "assistant", "content": message.content})
        results = runner.generate_tool_call_response()
        if results:
            history.append(results)
    return last


def _one_run(ctx: Ctx, support: Any, *, script: Optional[List[StubTurn]] = None) -> Any:
    """One user message: ``with support.run(question) as run:`` around the Tool Runner."""
    with _service([(ctx, script)]) as client:
        with support.run(user_message(ctx)) as run:
            return _tool_runner(ctx, client, run)


def run_nested(ctx: Ctx) -> Any:
    """One run with the suite's observer as ``agent_config(on_trace=...)``.

    The harness runs this inside a ``decimalai.start_trace()`` of its own; the
    run reads the enclosing trace when it opens, which is where its parent link
    is recorded.
    """
    return _one_run(ctx, _config(ctx, on_trace=trace_observer(ctx)))


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail, N lanes at once.

    The lanes share one agent name and so one config, the way every run of a
    module-level agent does. Each lane asks a different question, so the router
    mints a routing decision per lane and a leaked one is visible.
    """
    configs: Dict[str, Any] = {}
    pairs = [(ctx, _config(ctx, configs)) for ctx in ctxs]

    def _go(pair: Any) -> Any:
        ctx, support = pair
        return _one_run(ctx, support, script=_skill_script(ctx))

    return fanout_threads(_go)(pairs)


def run_unnamed(ctx: Ctx) -> Any:
    """The rail where the only name the run has is the DecimalAI one.

    The Messages API has no agent object at all, so there is nothing for a
    framework to call the agent. What this process does have is an ambient
    name: the tracing phases turned tracing on for another agent, and the
    exporter's default name was fixed then. Here nothing names this run but
    ``agent_config(name)`` — no ``agent_run()``, no ``init(agent_name=...)`` — so
    the trace and the routing call must both carry that name, not the
    process's.
    """
    return _one_run(ctx, _config(ctx), script=_skill_script(ctx))


def run_conversation(ctxs: Sequence[Ctx]) -> Any:
    """Two turns of one ``support.conversation()``.

    The second turn is sent the first turn's whole exchange ahead of its own
    message (``run.message``, which carries any skills new to the conversation),
    which is how a Messages-API application continues a conversation. The
    conversation keeps its ``system`` and tools; each run is still routed on its
    own message.
    """
    first, second = ctxs
    chat = _config(first).conversation()
    lanes = [(ctx, _skill_script(ctx)) for ctx in ctxs]
    history: List[Dict[str, Any]] = []
    with _service(lanes) as client:
        with chat.run(user_message(first)) as run:
            _tool_runner_turn(first, client, run, history)
        with chat.run(user_message(second)) as run:
            return _tool_runner_turn(second, client, run, history)


# ── second documented shapes ─────────────────────────────────────────────────


def _hand_written(ctx: Ctx, client: Any, run: Any, *, stream: bool) -> Any:
    """The module's hand-written loop: ``tools=[*mine, *run.tool_params]``, and
    ``run.tool_result(block)`` for each ``tool_use`` — the ``tool_result`` for
    ``load_skill``, ``None`` for the caller's own tool."""
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_message(ctx)}]
    response = None
    for _ in range(_MAX_MODEL_CALLS):
        response = _send(
            client, stream=stream, model=STUB_MODEL, max_tokens=512,
            system=run.system, tools=[*_tools(ctx), *run.tool_params], messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = run.tool_result(block)
            if result is None:
                query = block.input.get("query", "") if isinstance(block.input, dict) else ""
                result = {
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": tool_result(ctx, str(query)),
                }
            results.append(result)
        messages.append({"role": "user", "content": results})
    return response


def _hand_written_shape(stream: bool) -> Any:
    def _shape(ctxs: Sequence[Ctx]) -> Any:
        out = []
        for ctx in ctxs:
            support = _config(ctx)
            with _service([(ctx, _skill_script(ctx))]) as client:
                with support.run(user_message(ctx)) as run:
                    out.append(_hand_written(ctx, client, run, stream=stream))
        return out

    return _shape


async def _async_run(ctx: Ctx, support: Any) -> Any:
    import anthropic

    server, base_url = _start_provider([(ctx, _skill_script(ctx))])
    client = anthropic.AsyncAnthropic(api_key=_STUB_KEY, base_url=base_url)
    try:
        async with support.run(user_message(ctx)) as run:
            return await client.beta.messages.tool_runner(
                model=STUB_MODEL,
                max_tokens=512,
                system=run.system,
                tools=[_runner_tool(ctx, asynchronous=True), *run.tools],
                messages=[{"role": "user", "content": user_message(ctx)}],
            ).until_done()
    finally:
        await client.close()
        server.shutdown()
        server.server_close()


def _async_shape(ctxs: Sequence[Ctx]) -> Any:
    """``async with support.run(...)`` around the async Tool Runner."""
    return [asyncio.run(_async_run(ctx, _config(ctx))) for ctx in ctxs]


DRIVER = Driver(
    name="anthropic",
    # The capability table's "Anthropic Messages API (with `agent_config(...)`)" row
    # (decimalai/_frameworks.py), which agent_config() earned in 0.23. The tracing
    # half, init(anthropic=True), is advertised in prose and recorded in
    # test_coverage.PROVIDER_FLAG_DRIVERS.
    covers=frozenset({"anthropic-messages-api"}),
    requires=("anthropic", "openinference.instrumentation.anthropic"),
    frameworks=("anthropic", "openinference-instrumentation-anthropic"),
    entrypoint=(
        "decimalai.providers.instrument(anthropic=True) / "
        "decimalai.anthropic.agent_config(name).run(message)"
    ),
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    run_nested=run_nested,
    run_unnamed=run_unnamed,
    run_conversation=run_conversation,
    loads_agent_from_platform=True,
    shapes=(
        Shape(
            name="messages_create_loop",
            doc=(
                "a hand-written client.messages.create() loop inside "
                "agent_config(name).run(message), with tools=[*mine, *run.tool_params] "
                "and run.tool_result(block) answering load_skill"
            ),
            run=_hand_written_shape(stream=False),
            routed=True,
        ),
        Shape(
            name="messages_stream",
            doc=(
                "the same hand-written loop with every request sent through "
                "client.messages.stream(), read to get_final_message()"
            ),
            run=_hand_written_shape(stream=True),
            routed=True,
        ),
        Shape(
            name="async_tool_runner",
            doc=(
                "async with agent_config(name).run(message) around "
                "AsyncAnthropic().beta.messages.tool_runner(tools=[*mine, *run.tools])"
            ),
            run=_async_shape,
            routed=True,
        ),
    ),
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
    ),
)
