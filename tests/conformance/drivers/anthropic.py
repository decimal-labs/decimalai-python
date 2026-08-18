"""Raw Anthropic SDK driver — the no-framework rail.

Runs the two snippets ``decimalai-docs/sdk/python/frameworks.mdx`` documents for
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

from . import (
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
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


def _start_provider(ctx: Ctx, *, fail: bool = False) -> Tuple[Any, str]:
    """A real HTTP Anthropic stand-in for one run. Returns (server, base_url)."""

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
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
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
    import anthropic

    # A placeholder the SDK accepts; the stub service never looks at it, and no
    # request leaves 127.0.0.1.
    return anthropic.Anthropic(api_key="sk-ant-conformance-stub", base_url=base_url)


def _loop(
    ctx: Ctx, client: Any, *, tools: bool = True, system: Optional[str] = SYSTEM_PROMPT
) -> Any:
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


def _run(
    ctx: Ctx, *, fail: bool = False, tools: bool = True,
    system: Optional[str] = SYSTEM_PROMPT,
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
            return _loop(ctx, client, tools=tools, system=system)
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
    """
    from decimalai.anthropic import instrument

    instrument(enable_skill_loader=True)
    return fanout_threads(run)(ctxs)


DRIVER = Driver(
    name="anthropic",
    # The docs' capability table has no raw-Anthropic row: this rail is
    # advertised in the prose "No framework at all" section instead. Claiming a
    # slug here would fail the coverage guard's orphan check, and rightly so.
    covers=frozenset(),
    requires=("anthropic", "openinference.instrumentation.anthropic"),
    entrypoint="decimalai.providers.instrument(anthropic=True) / decimalai.anthropic.instrument()",
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        model_can_load_skill_bodies=False,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
        reasons={
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
    ),
)
