"""Every way the Anthropic SDK sends a message gets the routed skills — once.

``decimalai.anthropic.instrument(enable_skill_loader=True)`` used to patch only
``Messages.create`` / ``AsyncMessages.create``. In anthropic 1.13 the
``messages.stream()`` and ``messages.parse()`` helpers each build their own
request with ``self._post`` and never call ``create()``, so a caller who
streamed (the SDK's own recommendation for long outputs) or asked for
structured output got no skills, and nothing said so.

These tests drive the REAL ``anthropic.Anthropic`` / ``AsyncAnthropic``
clients — real request building, real stream managers, real response parsing —
against an ``httpx2.MockTransport``. Nothing leaves the process and no key is
used. The router is a stub that returns a fixed fragment, so the assertion is
about the bytes the SDK would have put on the wire.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest

anthropic = pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

from pydantic import BaseModel  # noqa: E402

FRAGMENT = "<<decimalai-routed-skills>>"
BASE = "You are a refunds agent."
QUESTION = "how do I refund order 42?"
MODEL = "claude-opus-5-5"

#: The real SDK methods the loader is expected to cover, per class.
PATCHED = ("create", "stream", "parse")


class _StubRouter:
    """Stands in for ``SkillRouter``: one fixed fragment, every call recorded."""

    def __init__(self, *, fail: bool = False) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.fail = fail

    def build_prompt_fragment(self, **kwargs: Any):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("router down")
        return FRAGMENT, "rt_phase1"


_MESSAGE = {
    "id": "msg_phase1",
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
    """A clean, restorable install of the loader onto the real SDK classes.

    The loader patches CLASS attributes and has no uninstall, so every method
    it may replace is snapshotted through ``monkeypatch`` first and put back
    after the test.
    """
    import decimalai.anthropic as da
    from anthropic.resources.messages import AsyncMessages, Messages

    for cls in (Messages, AsyncMessages):
        for name in PATCHED:
            monkeypatch.setattr(cls, name, vars(cls)[name])
    monkeypatch.setenv("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")
    monkeypatch.setattr(da, "_skill_loader_installed", False)
    monkeypatch.setattr(da, "_install_agent_name", None)
    router = _StubRouter()
    monkeypatch.setattr(da, "_skill_router_singleton", router)

    wire = _Wire()
    sync_client = anthropic.Anthropic(
        api_key="sk-ant-test-phase1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(wire)),
    )
    async_client = anthropic.AsyncAnthropic(
        api_key="sk-ant-test-phase1",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(wire)),
    )

    class _Sdk:
        pass

    s = _Sdk()
    s.da, s.router, s.wire = da, router, wire
    s.client, s.aclient = sync_client, async_client
    s.Messages, s.AsyncMessages = Messages, AsyncMessages
    s.install = lambda: da.instrument(enable_skill_loader=True)
    try:
        yield s
    finally:
        sync_client.close()
        asyncio.run(async_client.close())


def _kwargs(system: Any = BASE) -> Dict[str, Any]:
    return {
        "model": MODEL,
        "max_tokens": 64,
        "system": system,
        "messages": [{"role": "user", "content": QUESTION}],
    }


# ── the seven ways to send a message ─────────────────────────────────────────


def _create(s):
    return s.client.messages.create(**_kwargs())


def _create_stream_true(s):
    return list(s.client.messages.create(**_kwargs(), stream=True))


def _stream(s):
    with s.client.messages.stream(**_kwargs()) as stream:
        return stream.get_final_message()


def _parse(s):
    return s.client.messages.parse(**_kwargs(), output_format=Order)


def _acreate(s):
    return asyncio.run(s.aclient.messages.create(**_kwargs()))


def _astream(s):
    async def go():
        async with s.aclient.messages.stream(**_kwargs()) as stream:
            return await stream.get_final_message()
    return asyncio.run(go())


def _aparse(s):
    return asyncio.run(s.aclient.messages.parse(**_kwargs(), output_format=Order))


CALL_PATHS = {
    "create": _create,
    "create(stream=True)": _create_stream_true,
    "stream": _stream,
    "parse": _parse,
    "async create": _acreate,
    "async stream": _astream,
    "async parse": _aparse,
}


@pytest.mark.parametrize("path", list(CALL_PATHS))
def test_every_call_path_sends_the_skills_exactly_once(sdk, path):
    sdk.install()

    CALL_PATHS[path](sdk)

    (system,) = sdk.wire.systems
    assert isinstance(system, str), system
    assert system.count(FRAGMENT) == 1, (
        f"{path}: expected the routed skills once in `system`, got {system!r}"
    )
    # The caller's prompt stays first, so it stays in the cacheable prefix.
    assert system.startswith(BASE)
    # Routed on the caller's actual question, not an empty/menu-only query.
    assert [call["query"] for call in sdk.router.calls] == [QUESTION]


def test_stream_keeps_a_block_list_system_in_place(sdk):
    """The block-list shape — and its ``cache_control`` breakpoint — must
    survive the stream path exactly as it does ``create()``."""
    sdk.install()
    blocks = [{"type": "text", "text": BASE, "cache_control": {"type": "ephemeral"}}]

    with sdk.client.messages.stream(**_kwargs(system=blocks)) as stream:
        stream.get_final_message()

    (system,) = sdk.wire.systems
    assert system[0] == blocks[0]
    assert system[1:] == [{"type": "text", "text": FRAGMENT}]


def test_helpers_still_return_what_they_promise(sdk):
    """Patching must not change what the helpers hand back."""
    sdk.install()

    with sdk.client.messages.stream(**_kwargs()) as stream:
        assert isinstance(stream, anthropic.lib.streaming.MessageStream)
        assert stream.get_final_message().id == "msg_phase1"
    assert sdk.client.messages.parse(**_kwargs(), output_format=Order).parsed_output == Order(order=42)
    assert asyncio.run(
        sdk.aclient.messages.parse(**_kwargs(), output_format=Order)
    ).parsed_output == Order(order=42)


@pytest.mark.parametrize("path", ["stream", "parse", "async stream", "async parse"])
def test_router_failure_never_breaks_the_call(sdk, monkeypatch, path):
    """Skill injection is best-effort: a dead router sends the caller's own
    prompt, untouched, and the request still completes."""
    monkeypatch.setattr(sdk.da, "_skill_router_singleton", _StubRouter(fail=True))
    sdk.install()

    CALL_PATHS[path](sdk)

    assert sdk.wire.systems == [BASE]


def test_raw_and_streaming_response_wrappers_still_inject_once(sdk):
    """``with_raw_response`` / ``with_streaming_response`` wrap ``create`` — a
    regression guard that the new patch did not drop or double them."""
    sdk.install()

    sdk.client.messages.with_raw_response.create(**_kwargs())
    with sdk.client.messages.with_streaming_response.create(**_kwargs()) as response:
        response.read()

    assert [s.count(FRAGMENT) for s in sdk.wire.systems] == [1, 1]


def test_installing_twice_does_not_inject_twice(sdk):
    sdk.install()
    sdk.install()

    _stream(sdk)
    _aparse(sdk)

    assert [s.count(FRAGMENT) for s in sdk.wire.systems] == [1, 1]


# ── one patched method calling another ───────────────────────────────────────


def test_a_patched_helper_that_delegates_to_create_injects_once(sdk, monkeypatch):
    """anthropic 1.13's helpers do not call ``create()``, but the pattern is
    common (openai's ``stream()`` is built on ``create(stream=True)``) and the
    SDK could adopt it in any release. If it does, the outer call must inject
    and the inner one must not, or the skills land in ``system`` twice."""
    def delegating_stream(self, **kwargs):
        return self.create(**kwargs, stream=True)

    monkeypatch.setattr(sdk.Messages, "stream", delegating_stream)
    sdk.install()

    list(sdk.client.messages.stream(**_kwargs()))

    (system,) = sdk.wire.systems
    assert system.count(FRAGMENT) == 1, system
    assert len(sdk.router.calls) == 1


def test_an_async_helper_that_awaits_create_injects_once(sdk, monkeypatch):
    async def delegating_parse(self, **kwargs):
        kwargs.pop("output_format", None)
        return await self.create(**kwargs)

    monkeypatch.setattr(sdk.AsyncMessages, "parse", delegating_parse)
    sdk.install()

    asyncio.run(sdk.aclient.messages.parse(**_kwargs(), output_format=Order))

    (system,) = sdk.wire.systems
    assert system.count(FRAGMENT) == 1, system
    assert len(sdk.router.calls) == 1


def test_concurrent_async_calls_each_get_their_own_injection(sdk):
    """The re-entry guard is per call, not process-wide: two calls in flight
    at once on one event loop must both be injected."""
    sdk.install()

    async def go():
        await asyncio.gather(
            sdk.aclient.messages.create(**_kwargs()),
            sdk.aclient.messages.parse(**_kwargs(), output_format=Order),
        )

    asyncio.run(go())

    assert [s.count(FRAGMENT) for s in sdk.wire.systems] == [1, 1]


# ── the docstring ────────────────────────────────────────────────────────────


def test_docstring_does_not_call_agent_skills_beta():
    """Anthropic's Agent Skills went GA on 2026-08-19 (``client.skills`` is a
    non-beta resource in anthropic 1.13)."""
    import decimalai.anthropic as da

    doc = da.__doc__ or ""
    assert "Skills (beta)" not in doc
    assert "stream()" in doc and "parse()" in doc, (
        "the module docstring should say which calls get skills"
    )


# ── whose skills: the run's agent, the one its trace is filed under ──────────
#
# The documented skills setup on this adapter is
# ``instrument(enable_skill_loader=True)`` with no name, and the run is named
# with ``providers.agent_run(name)`` — the name the trace is filed under. The
# patch used to route with the INSTALL name only, so that run was routed for
# no agent at all: the org-wide skill set, never that agent's assignments,
# version pins or restricted skills, while its trace said it had been routed.


@pytest.fixture
def no_ambient_agent():
    """``_active_agent_name`` is a process-wide ContextVar other tests set
    without resetting; start each of these from "no run is named"."""
    from decimalai.otel import _active_agent_name

    token = _active_agent_name.set(None)
    try:
        yield
    finally:
        _active_agent_name.reset(token)


def _routed_names(sdk) -> List[Any]:
    return [call.get("agent_name") for call in sdk.router.calls]


@pytest.mark.parametrize("path", ["create", "stream", "async create", "async parse"])
def test_a_run_named_by_agent_run_routes_for_that_agent(sdk, no_ambient_agent, path):
    from decimalai import providers

    sdk.install()  # the documented setup: no agent_name on the adapter
    with providers.agent_run("support"):
        CALL_PATHS[path](sdk)

    assert _routed_names(sdk) == ["support"], (
        "the run is filed under 'support', so it must be routed for 'support' — "
        "a call that names no agent is offered none of that agent's skills"
    )


def test_the_run_name_outranks_the_install_name(sdk, no_ambient_agent):
    """A process serving two agents names each run with ``agent_run``; the
    adapter's one install name must not route both of them as itself."""
    from decimalai import providers

    sdk.da.instrument(agent_name="support-agent", enable_skill_loader=True)
    with providers.agent_run("billing-agent"):
        _create(sdk)
        _astream(sdk)
    _create(sdk)  # outside any named run: the install name, as before

    assert _routed_names(sdk) == ["billing-agent", "billing-agent", "support-agent"]


def test_no_run_and_no_install_name_still_routes_unscoped(sdk, no_ambient_agent):
    sdk.install()
    _create(sdk)
    assert _routed_names(sdk) == [None]


def test_an_explicit_skill_system_agent_name_is_left_alone(sdk, no_ambient_agent):
    """Callers who build ``system`` themselves pass the name per call; that
    argument is theirs and is forwarded as given."""
    from decimalai import providers

    with providers.agent_run("billing-agent"):
        sdk.da.skill_system(BASE, query=QUESTION, agent_name="refund-bot")

    assert _routed_names(sdk) == ["refund-bot"]
