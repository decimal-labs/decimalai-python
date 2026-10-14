"""``AsyncAnthropic`` routes skills on a worker thread, never on the event loop.

The skill router is a synchronous HTTP client: one routing call is a route
request plus a body fetch, each with its own read budget and admission retries.
The async patches used to make that call inline, from the coroutine. Every
other coroutine on the loop (in an async server, every other conversation in
the process) then stood still until the platform answered. Phase 1 had moved
ADK, LangChain, OpenAI Agents and Pydantic AI off the loop. The Anthropic
adapter still routed inline, and Phase 1 had widened its patch from
``AsyncMessages.create`` to ``.stream()`` and ``.parse()`` as well, so three
async entry points blocked the loop.

Like ``test_anthropic_phase1``, these tests drive the REAL ``anthropic 1.13``
``AsyncAnthropic`` client (real request building, real stream managers, real
response parsing) against an ``httpx2.MockTransport``. Nothing leaves the
process. The router is a stub whose call cannot finish until a coroutine on the
caller's loop releases it, so a call made on the loop thread is caught every
time: it waits out its timeout instead of being released.
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Callable, Dict, List, Optional

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from pydantic import BaseModel  # noqa: E402

FRAGMENT = "<<decimalai-routed-skills>> refund-policy"
BASE = "You are a refunds agent."
QUESTION = "how do I refund order 42?"
MODEL = "claude-opus-5-5"
ROUTING_ID = "rt_phase1d"
INJECTED = f"{BASE}\n\n{FRAGMENT}"
# Long enough that a blocked loop is unmistakable, short enough that the red
# run of this file stays cheap.
GATE_TIMEOUT_S = 2.0

#: The SDK methods the loader replaces, per class.
PATCHED = ("create", "stream", "parse")


# ── the fake platform and the fake Anthropic service ────────────────────────


class _GatedRouter:
    """A ``SkillRouter`` stand-in whose HTTP call cannot finish until a
    coroutine on the caller's event loop opens ``gate``.

    It therefore finishes in time only if the loop keeps running while the
    call is in flight. Made on the loop thread, the call waits out
    ``GATE_TIMEOUT_S`` and records ``released=False``.
    """

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.finished = threading.Event()
        self.calls: List[Dict[str, Any]] = []

    def build_prompt_fragment(self, **kwargs: Any):
        from decimalai import skill_router as sr

        self.entered.set()
        try:
            released = self.gate.wait(GATE_TIMEOUT_S)
            self.calls.append({
                **kwargs, "thread": threading.get_ident(), "released": released,
            })
            # The per-call rails the real router writes: a name it offered.
            sr._last_offered_names_ctx.set(["refund-policy"])
            return FRAGMENT, ROUTING_ID
        finally:
            self.finished.set()


async def _open_gate_once_routing(router: _GatedRouter) -> None:
    # Polls with real awaits: it can only see `entered` and open the gate if
    # the loop gets control back while the router call is still running.
    while not router.entered.is_set():
        await asyncio.sleep(0.005)
    router.gate.set()


_MESSAGE = {
    "id": "msg_phase1d",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "content": [{"type": "text", "text": '{"order": 42}'}],
    "usage": {"input_tokens": 3, "output_tokens": 5},
}


def _sse() -> bytes:
    """A minimal, well-formed Messages-API event stream for ``_MESSAGE``."""
    events = [
        ("message_start", {"type": "message_start", "message": {
            **_MESSAGE, "content": [], "stop_reason": None,
        }}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": '{"order": 42}'}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 5}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events
    ).encode()


class _Wire:
    """The fake Anthropic service: records every request body it is sent."""

    def __init__(self) -> None:
        self.bodies: List[Dict[str, Any]] = []

    def __call__(self, request: Any) -> Any:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get("stream"):
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=_sse(),
            )
        return httpx2.Response(200, json=_MESSAGE)

    @property
    def systems(self) -> List[Any]:
        return [body.get("system") for body in self.bodies]


class Order(BaseModel):
    order: int


@pytest.fixture
def sdk(monkeypatch):
    """The loader on the real SDK classes, restored after the test.

    The loader patches CLASS attributes and has no uninstall, so every method
    it may replace is snapshotted through ``monkeypatch`` first.
    """
    from anthropic.resources.messages import AsyncMessages, Messages

    import decimalai.anthropic as da

    for cls in (Messages, AsyncMessages):
        for name in PATCHED:
            monkeypatch.setattr(cls, name, vars(cls)[name])
    monkeypatch.setenv("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")
    monkeypatch.setattr(da, "_skill_loader_installed", False)
    monkeypatch.setattr(da, "_install_agent_name", None)
    router = _GatedRouter()
    monkeypatch.setattr(da, "_skill_router_singleton", router)

    wire = _Wire()
    aclient = anthropic.AsyncAnthropic(
        api_key="sk-ant-test-phase1d",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(wire)),
    )

    class _Sdk:
        pass

    s = _Sdk()
    s.da, s.router, s.wire, s.aclient = da, router, wire, aclient
    s.AsyncMessages = AsyncMessages
    s.install = lambda: da.instrument(enable_skill_loader=True)

    def use(new_router: Any) -> None:
        s.router = new_router
        monkeypatch.setattr(da, "_skill_router_singleton", new_router)

    s.use = use
    try:
        yield s
    finally:
        asyncio.run(aclient.close())


@pytest.fixture
def no_ambient_agent():
    """``_active_agent_name`` is a process-wide ContextVar other tests set
    without resetting; start from "no run is named"."""
    from decimalai.otel import _active_agent_name

    token = _active_agent_name.set(None)
    try:
        yield
    finally:
        _active_agent_name.reset(token)


def _kwargs(question: str = QUESTION) -> Dict[str, Any]:
    return {
        "model": MODEL,
        "max_tokens": 64,
        "system": BASE,
        "messages": [{"role": "user", "content": question}],
    }


# ── every async way to send a message ───────────────────────────────────────


async def _acreate(c):
    return await c.messages.create(**_kwargs())


async def _acreate_stream_true(c):
    return [event async for event in await c.messages.create(**_kwargs(), stream=True)]


async def _astream(c):
    async with c.messages.stream(**_kwargs()) as stream:
        assert isinstance(stream, anthropic.lib.streaming.AsyncMessageStream)
        return await stream.get_final_message()


async def _aparse(c):
    return await c.messages.parse(**_kwargs(), output_format=Order)


async def _araw_response(c):
    return await c.messages.with_raw_response.create(**_kwargs())


async def _astreaming_response(c):
    async with c.messages.with_streaming_response.create(**_kwargs()) as response:
        return await response.read()


ASYNC_PATHS: Dict[str, Callable[[Any], Any]] = {
    "create": _acreate,
    "create(stream=True)": _acreate_stream_true,
    "stream": _astream,
    "parse": _aparse,
    "with_raw_response.create": _araw_response,
    "with_streaming_response.create": _astreaming_response,
}


def _run_releasing_the_gate(sdk, path: str) -> int:
    """Send one message by ``path`` while a coroutine beside it waits to open
    the router's gate. Returns the loop's thread id."""

    async def go() -> int:
        opener = asyncio.create_task(_open_gate_once_routing(sdk.router))
        await ASYNC_PATHS[path](sdk.aclient)
        await opener
        return threading.get_ident()

    return asyncio.run(go())


# ── 1. the loop keeps running while the router is in flight ─────────────────


@pytest.mark.parametrize("path", list(ASYNC_PATHS))
def test_a_slow_router_does_not_block_a_concurrent_coroutine(sdk, path):
    sdk.install()

    loop_thread = _run_releasing_the_gate(sdk, path)

    [call] = sdk.router.calls
    assert call["released"] is True, (
        f"async {path}: the router call held the event loop, so a coroutine "
        "scheduled beside it could not run until the call gave up"
    )
    assert call["thread"] != loop_thread
    # Delivery is unchanged: the skills are in the request, once, after the
    # caller's own prompt, routed on the caller's question.
    assert sdk.wire.systems == [INJECTED]
    assert call["query"] == QUESTION


@pytest.mark.parametrize("path", list(ASYNC_PATHS))
def test_the_run_name_set_by_agent_run_reaches_the_router_off_the_loop(
    sdk, no_ambient_agent, path,
):
    """The worker thread runs in a copy of the caller's context, so the name
    ``agent_run()`` set for the run is the name the call is routed for."""
    from decimalai import providers

    sdk.install()

    async def go() -> int:
        with providers.agent_run("support"):
            opener = asyncio.create_task(_open_gate_once_routing(sdk.router))
            await ASYNC_PATHS[path](sdk.aclient)
            await opener
        return threading.get_ident()

    loop_thread = asyncio.run(go())

    [call] = sdk.router.calls
    assert call["released"] is True and call["thread"] != loop_thread
    assert call["agent_name"] == "support"


def test_two_concurrent_runs_route_at_once_each_for_its_own_agent(sdk, no_ambient_agent):
    """Both routing calls must be in flight together (only possible off the
    loop), and each must be routed for, and delivered to, its own run."""
    from decimalai import providers

    both_routing = threading.Barrier(2, timeout=GATE_TIMEOUT_S)

    class _BarrierRouter:
        def __init__(self) -> None:
            self.calls: List[Dict[str, Any]] = []

        def build_prompt_fragment(self, **kwargs: Any):
            both_routing.wait()
            self.calls.append(kwargs)
            name = kwargs["agent_name"]
            return f"<<skills for {name}>>", f"rt_{name}"

    sdk.use(_BarrierRouter())
    sdk.install()

    async def run(agent: str) -> Optional[str]:
        with providers.agent_run(agent):
            await sdk.aclient.messages.create(**_kwargs(question=f"question from {agent}"))
            return sdk.da.get_current_routing_id()

    async def go():
        return await asyncio.gather(run("support"), run("billing"))

    routing_ids = asyncio.run(go())

    sent = {body["messages"][0]["content"]: body["system"] for body in sdk.wire.bodies}
    assert sent == {
        "question from support": f"{BASE}\n\n<<skills for support>>",
        "question from billing": f"{BASE}\n\n<<skills for billing>>",
    }
    # Each run reads back its own decision, not the other run's.
    assert routing_ids == ["rt_support", "rt_billing"]


# ── 2. the caller's context ends up as if routing had run inline ────────────


@pytest.mark.parametrize("path", ["create", "stream", "parse"])
def test_what_routing_wrote_to_the_context_reaches_the_caller(sdk, path):
    """Routing writes this SDK's context variables: the routing id this
    adapter exposes through ``get_current_routing_id()``, and the router's
    per-call rails, which ``skill_system`` drains. Run on a worker thread those
    writes land in a copy of the context, and are copied back afterwards so the
    awaiting coroutine sees what an inline call would have left."""
    from decimalai import skill_router as sr

    sdk.install()

    async def go():
        # A stale rail left by an earlier call: routing drains it, and the
        # caller must see it drained.
        sr._last_offered_names_ctx.set(["stale-from-an-earlier-call"])
        opener = asyncio.create_task(_open_gate_once_routing(sdk.router))
        await ASYNC_PATHS[path](sdk.aclient)
        await opener
        return sdk.da.get_current_routing_id(), sr._last_offered_names_ctx.get()

    routing_id, offered_rail = asyncio.run(go())

    assert sdk.router.calls[0]["released"] is True
    assert routing_id == ROUTING_ID
    assert not offered_rail


def test_the_worker_files_the_routing_decision_under_the_callers_run(sdk, no_ambient_agent):
    """The run's trace is found from the OTel span current in the caller's
    context. The worker thread must see that span, or the decision is filed
    under no run and the trace says nothing was routed."""
    from opentelemetry.sdk.trace import TracerProvider

    from decimalai import otel

    sdk.install()

    async def go():
        with otel.agent_run("support", tracer_provider=TracerProvider()) as span:
            opener = asyncio.create_task(_open_gate_once_routing(sdk.router))
            await sdk.aclient.messages.create(**_kwargs())
            await opener
            return span.get_span_context().trace_id

    trace_id = asyncio.run(go())

    [call] = sdk.router.calls
    assert call["released"] is True
    assert call["scope"] == f"{trace_id:032x}"
    rail = otel._pop_skill_rail(trace_id)
    assert rail is not None, "the routing decision was filed under no run"
    assert rail["routing_id"] == ROUTING_ID
    assert rail["offered"] == ["refund-policy"]


def test_without_an_asyncio_loop_anyio_carries_the_call(sdk, monkeypatch):
    """anthropic's async client also runs under trio (it is built on anyio),
    where there is no asyncio loop to hand the call to. anyio's worker thread
    takes it there. Exercised here on anyio's asyncio backend, with the asyncio
    branch switched off."""
    monkeypatch.setattr(sdk.da, "_running_asyncio_loop", lambda: None)
    sdk.install()

    async def go():
        opener = asyncio.create_task(_open_gate_once_routing(sdk.router))
        await _acreate(sdk.aclient)
        await opener
        return threading.get_ident(), sdk.da.get_current_routing_id()

    loop_thread, routing_id = asyncio.run(go())

    [call] = sdk.router.calls
    assert call["released"] is True and call["thread"] != loop_thread
    assert sdk.wire.systems == [INJECTED]
    assert routing_id == ROUTING_ID


# ── 3. what must not change ─────────────────────────────────────────────────


@pytest.mark.parametrize("path", list(ASYNC_PATHS))
def test_a_router_that_fails_off_the_loop_sends_the_callers_prompt(sdk, path):
    class _Down:
        def build_prompt_fragment(self, **kwargs: Any):
            raise ConnectionError("router offline")

    sdk.use(_Down())
    sdk.install()

    asyncio.run(ASYNC_PATHS[path](sdk.aclient))

    assert sdk.wire.systems == [BASE]


def test_cancelling_a_call_mid_routing_cancels_at_once_and_sends_nothing(sdk):
    """A cancelled request (a client that hung up, a timeout around the call)
    is cancelled while the router is still in flight, not after it, and the
    message is never sent."""
    sdk.install()

    async def go() -> bool:
        call = asyncio.create_task(_acreate(sdk.aclient))
        while not sdk.router.entered.is_set():
            await asyncio.sleep(0.005)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        cancelled_while_routing = not sdk.router.finished.is_set()
        # Let the abandoned routing call finish before the loop closes.
        sdk.router.gate.set()
        await asyncio.get_running_loop().run_in_executor(
            None, sdk.router.finished.wait, GATE_TIMEOUT_S,
        )
        return cancelled_while_routing

    assert asyncio.run(go()) is True
    assert sdk.wire.bodies == []


def test_an_async_stream_built_on_create_still_injects_once(sdk, monkeypatch):
    """The stream patch routes when the stream is entered, and must keep its
    re-entry guard up until the request is actually sent. If a release builds
    ``stream()`` on ``create(stream=True)`` (openai's SDK does), the inner
    create runs when the manager is entered; with the guard already down it
    would route a second time and put the skills in ``system`` twice."""
    from anthropic.lib.streaming import AsyncMessageStreamManager

    def delegating_stream(self, **kwargs):
        return AsyncMessageStreamManager(self.create(**kwargs, stream=True))

    monkeypatch.setattr(sdk.AsyncMessages, "stream", delegating_stream)
    sdk.install()

    _run_releasing_the_gate(sdk, "stream")

    assert sdk.wire.systems == [INJECTED]
    assert len(sdk.router.calls) == 1


def test_if_the_hand_off_to_a_thread_fails_the_call_still_gets_its_skills(sdk, monkeypatch):
    """An executor that is shutting down refuses new work. Routing then runs
    inline, as it always used to, rather than letting the call go out without
    its skills and with nothing to show for it."""

    class _ShuttingDownLoop:
        def run_in_executor(self, *args: Any, **kwargs: Any):
            raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(sdk.da, "_running_asyncio_loop", lambda: _ShuttingDownLoop())
    sdk.router.gate.set()
    sdk.install()

    async def go():
        await _acreate(sdk.aclient)
        return sdk.da.get_current_routing_id()

    assert asyncio.run(go()) == ROUTING_ID
    assert sdk.wire.systems == [INJECTED]
    assert len(sdk.router.calls) == 1


def test_entering_one_stream_manager_twice_fails_as_the_sdks_does(sdk):
    """The SDK's manager awaits one request coroutine, so a second ``async
    with`` on it raises. The patched manager must not route a second time, or
    send a second request, on the way to that error."""
    sdk.install()
    sdk.router.gate.set()

    async def go():
        manager = sdk.aclient.messages.stream(**_kwargs())
        async with manager as stream:
            await stream.get_final_message()
        with pytest.raises(RuntimeError, match="cannot reuse already awaited coroutine"):
            async with manager:
                pass

    asyncio.run(go())

    assert len(sdk.router.calls) == 1
    assert sdk.wire.systems == [INJECTED]


def test_a_call_missing_a_required_argument_still_raises_typeerror(sdk):
    sdk.install()
    sdk.router.gate.set()

    async def go():
        await sdk.aclient.messages.create(model=MODEL, messages=[{"role": "user", "content": QUESTION}])

    with pytest.raises(TypeError, match="max_tokens"):
        asyncio.run(go())
    assert sdk.wire.bodies == []


@pytest.mark.parametrize("order", ["openinference first", "decimalai first"])
def test_openinference_still_records_the_system_the_model_was_sent(sdk, order):
    """``decimalai.init(anthropic=True)`` turns on OpenInference's Anthropic
    instrumentor next to this loader, and the platform reads delivery from the
    span's rendered input. Moving routing to ``stream()``'s ``__aenter__`` must
    keep the request built where OpenInference wraps it, in either install
    order."""
    pytest.importorskip("openinference.instrumentation.anthropic")
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    sdk.router.gate.set()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AnthropicInstrumentor()
    if order == "openinference first":
        instrumentor.instrument(tracer_provider=provider)
        sdk.install()
    else:
        sdk.install()
        instrumentor.instrument(tracer_provider=provider)
    try:
        async def go():
            await _acreate(sdk.aclient)
            await _astream(sdk.aclient)
            await _aparse(sdk.aclient)

        asyncio.run(go())
    finally:
        instrumentor.uninstrument()

    assert sdk.wire.systems == [INJECTED] * 3
    recorded = [
        dict(span.attributes).get("llm.input_messages.0.message.content")
        for span in exporter.get_finished_spans()
    ]
    assert recorded == [INJECTED] * 3
