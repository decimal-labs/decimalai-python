"""Pydantic AI adapter, Phase 1 of the 2026-10-13 integration audit.

Every test here drives the REAL ``pydantic_ai.Agent`` (pydantic-ai-slim 2.56,
the only version the adapter supports) with Pydantic AI's own ``FunctionModel``,
so the agent graph that decides when the skills menu is built, which tools
clash, and which thread a prompt function runs on is the product's, not a fake's.
No network, no keys: the model is a function and the router is a stand-in.

The four defects, each verified by experiment in the 2026-10-13 framework
integration audit:

1. Multi-turn: the menu was a non-dynamic system prompt, which Pydantic AI
   builds only when there is no ``message_history``. Turn 2 of a conversation
   was never routed, reused turn 1's menu, and its trace carried no routing_id.
2. A second toolset exposing ``load_skill`` made EVERY run raise ``UserError``.
3. Routing sent ``ctx.agent.name`` — for an unnamed Agent, the variable name
   Pydantic AI inferred (``"agent"``) — while the trace used
   ``instrument(agent_name=...)``. Agent-assigned skills silently missed.
4. The router's blocking HTTP call ran inside an ``async`` prompt function,
   i.e. on the event loop, stalling every other coroutine on it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import types
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

pytest.importorskip("opentelemetry.sdk.trace")
pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import (  # noqa: E402
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402
from pydantic_ai.toolsets import FunctionToolset  # noqa: E402

from decimalai import providers  # noqa: E402
from decimalai.otel import _active_agent_name  # noqa: E402

os.environ.setdefault("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")

NAMESPACED = "decimalai_load_skill"
SKILL = "refund-policy"
BODY = "## Skill: refund-policy\n\nREFUND-BODY-SENTINEL-6f1e"


# ── stand-ins ────────────────────────────────────────────────────────────────


class _Router:
    """A SkillRouter stand-in: one menu per query, one routing id per call."""

    def __init__(self) -> None:
        self.calls: List[SimpleNamespace] = []
        self.loads: List[str] = []
        self._lock = threading.Lock()

    def build_prompt_fragment(self, query=None, agent_name=None, scope=None, **_kw):
        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        with self._lock:
            self.calls.append(SimpleNamespace(
                query=query, agent_name=agent_name, scope=scope, on_loop=on_loop,
            ))
            n = len(self.calls)
        return (
            f"## Available Skills\n| {SKILL} | MENU-FOR[{query}] |",
            "rt_" + format(n, "024x"),
        )

    def load_skill(self, name, scope=None):
        self.loads.append(name)
        return BODY

    def consume_loaded_names(self, scope=None):
        return []

    def consume_routing_id(self, **_kw):
        return None

    def consume_offered_names(self, **_kw):
        return []

    def consume_delivered_names(self, **_kw):
        return []


def _model_view(messages, info: AgentInfo) -> str:
    """What a real provider is sent as instructions for THIS request.

    System-prompt parts anywhere in the history, plus this request's
    instructions. NOT the ``instructions`` stamped on older requests in the
    history: every Pydantic AI model reads the current request's instruction
    parts and ignores those (``Model._get_instruction_parts``).
    """
    texts = []
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, SystemPromptPart):
                    texts.append(part.content)
    if info.instructions:
        texts.append(info.instructions)
    return "\n".join(texts)


class _Model:
    """A FunctionModel script that records what each request saw."""

    def __init__(self, *, call_tool: Optional[str] = None) -> None:
        self.views: List[str] = []
        self.tool_names: List[List[str]] = []
        self.instruction_parts: List[list] = []
        self.system_parts: List[List[str]] = []
        self.call_tool = call_tool

    def __call__(self, messages, info: AgentInfo) -> ModelResponse:
        self.views.append(_model_view(messages, info))
        self.instruction_parts.append(
            list(info.model_request_parameters.instruction_parts or [])
        )
        self.system_parts.append([
            part.content
            for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, SystemPromptPart)
        ])
        self.tool_names.append(sorted(t.name for t in info.function_tools))
        last = messages[-1]
        returned = isinstance(last, ModelRequest) and any(
            isinstance(p, ToolReturnPart) for p in last.parts
        )
        if self.call_tool and not returned:
            return ModelResponse(parts=[ToolCallPart(self.call_tool, {"name": SKILL})])
        return ModelResponse(parts=[TextPart("done")])

    def model(self) -> FunctionModel:
        return FunctionModel(self)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def pa(monkeypatch):
    """The adapter, un-installed, with the REAL Agent restored after the test.

    ``instrument()`` patches ``Agent.__init__`` and ``Agent.iter`` on the real
    class. Registering both with monkeypatch first is what puts them back, so
    no other test in the session inherits this one's patches.
    """
    import decimalai.pydantic_ai as module

    monkeypatch.setattr(Agent, "__init__", Agent.__init__)
    monkeypatch.setattr(Agent, "iter", Agent.iter)
    for attr, value in (
        ("_skill_loader_installed", False),
        ("_name_capture_installed", False),
        ("_run_scope_installed", False),
        ("_default_agent_name", None),
        ("_load_skill_tool_active", False),
        ("_skill_router_singleton", None),
        ("_tool_clash_warned", False),
    ):
        monkeypatch.setattr(module, attr, value, raising=False)
    memo = getattr(module, "_run_fragments", None)
    if memo is not None:
        monkeypatch.setattr(module, "_run_fragments", type(memo)())
    monkeypatch.delenv("DECIMALAI_LOAD_SKILL_TOOL", raising=False)
    token = _active_agent_name.set(None)
    yield module
    _active_agent_name.reset(token)


@pytest.fixture
def router(pa, monkeypatch):
    stand_in = _Router()
    monkeypatch.setattr(pa, "_skill_router_singleton", stand_in)
    return stand_in


@pytest.fixture
def pipeline():
    """The real DecimalAI exporter, with a mock client catching the traces."""
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    prev = (cfg._config, cfg._client, providers._last_provider)
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }
    tp = TracerProvider()
    providers._ensure_pipeline(None, tp)

    def traces():
        from decimalai._config import _sender

        _sender.flush()
        return [c[0][0] for c in cfg._client.ingest_trace.call_args_list]

    yield types.SimpleNamespace(provider=tp, traces=traces)
    cfg._config, cfg._client, providers._last_provider = prev


# ── 1. every turn of a conversation is routed ────────────────────────────────


class TestMultiTurn:
    def test_turn_two_with_message_history_is_routed_and_traced(
        self, pa, router, pipeline
    ):
        """Two turns, the second carrying the first's history: two routing
        decisions, one per turn, and each turn's trace carries its own id."""
        pa.instrument(enable_skill_loader=True)
        script = _Model()
        agent = Agent(script.model(), name="support")

        first = agent.run_sync("where is my refund?")
        agent.run_sync("and can I change my address?",
                       message_history=first.all_messages())

        assert [c.query for c in router.calls] == [
            "where is my refund?", "and can I change my address?",
        ], "turn 2 was not routed — it reused turn 1's menu"
        routing_ids = [t.routing_id for t in pipeline.traces()]
        assert routing_ids == ["rt_" + format(1, "024x"), "rt_" + format(2, "024x")]

    def test_turn_two_sees_its_own_menu_and_not_the_stale_one(self, pa, router):
        """The menu is rebuilt for the new question, and the old menu is not
        replayed to the model from the history."""
        pa.instrument(enable_skill_loader=True)
        script = _Model()
        agent = Agent(script.model(), name="support", system_prompt="CALLER-PROMPT")

        first = agent.run_sync("q-one")
        agent.run_sync("q-two", message_history=first.all_messages())

        turn_two = script.views[-1]
        assert "MENU-FOR[q-two]" in turn_two
        assert "MENU-FOR[q-one]" not in turn_two
        # The caller's own prompt still leads, ahead of the routed menu.
        assert turn_two.index("CALLER-PROMPT") < turn_two.index("MENU-FOR[q-two]")

    def test_the_callers_text_is_a_stable_prefix_and_the_menu_a_dynamic_tail(
        self, pa, router
    ):
        """What prompt caching rests on, read off the real request parameters.

        The caller's `system_prompt=` is a system part and its `instructions=`
        a STATIC instruction part, byte-identical on every turn. The routed menu
        is one DYNAMIC instruction part after them, so Pydantic AI's provider
        models put any cache breakpoint in front of it (and the OpenAI and
        Anthropic models send instruction parts after the system parts).
        """
        pa.instrument(enable_skill_loader=True)
        script = _Model()
        agent = Agent(script.model(), name="support",
                      system_prompt="CALLER-PROMPT", instructions="CALLER-INSTRUCTIONS")

        first = agent.run_sync("q-one")
        agent.run_sync("q-two", message_history=first.all_messages())

        assert script.system_parts == [["CALLER-PROMPT"], ["CALLER-PROMPT"]]
        for parts, query in zip(script.instruction_parts, ("q-one", "q-two")):
            assert [(p.content, p.dynamic) for p in parts[:-1]] == [
                ("CALLER-INSTRUCTIONS", False),
            ]
            assert f"MENU-FOR[{query}]" in parts[-1].content
            assert parts[-1].dynamic is True

    def test_one_run_routes_once_even_across_tool_steps(self, pa, router, pipeline):
        """A run that calls load_skill makes two model requests. It is ONE
        routing decision: both requests see the same menu, the router is asked
        once, and the trace has that decision's id."""
        pa.instrument(enable_skill_loader=True)
        script = _Model(call_tool="load_skill")
        agent = Agent(script.model(), name="support")

        agent.run_sync("where is my refund?")

        assert len(router.calls) == 1
        assert len(script.views) == 2
        assert script.views[0] == script.views[1]
        assert router.loads == [SKILL]
        (trace,) = pipeline.traces()
        assert trace.routing_id == "rt_" + format(1, "024x")

    def test_a_trimmed_history_is_still_routed(self, pa, router):
        """A history processor that drops the first request (and with it every
        system prompt) must not switch routing off for the rest of the chat."""
        pa.instrument(enable_skill_loader=True)
        script = _Model()
        agent = Agent(script.model(), name="support")

        first = agent.run_sync("q-one")
        trimmed = first.all_messages()[1:]  # the model's reply only
        agent.run_sync("q-two", message_history=trimmed)

        assert [c.query for c in router.calls] == ["q-one", "q-two"]
        assert "MENU-FOR[q-two]" in script.views[-1]


# ── 2. a second load_skill tool does not crash the run ───────────────────────


def _their_load_skill(name: str) -> str:
    """Somebody else's skill loader."""
    return f"THEIR-SKILL[{name}]"


def _theirs() -> FunctionToolset:
    toolset = FunctionToolset()
    toolset.add_function(_their_load_skill, name="load_skill")
    return toolset


class TestToolNameClash:
    def test_a_constructor_toolset_with_load_skill_runs(self, pa, router, caplog):
        pa.instrument(enable_skill_loader=True)
        script = _Model(call_tool=NAMESPACED)
        agent = Agent(script.model(), name="support", toolsets=[_theirs()])

        with caplog.at_level(logging.WARNING, logger="decimalai.pydantic_ai"):
            result = agent.run_sync("where is my refund?")

        assert result.output == "done"
        # Both catalogs are callable: theirs under its own name, ours namespaced.
        assert script.tool_names[0] == sorted(["load_skill", NAMESPACED])
        assert router.loads == [SKILL], "the namespaced tool must reach DecimalAI"
        # The menu tells the model which name is DecimalAI's.
        assert NAMESPACED in script.views[0]
        assert any(NAMESPACED in r.getMessage() for r in caplog.records)

    def test_their_tool_still_answers_under_its_own_name(self, pa, router):
        pa.instrument(enable_skill_loader=True)
        script = _Model(call_tool="load_skill")
        agent = Agent(script.model(), name="support", toolsets=[_theirs()])

        result = agent.run_sync("where is my refund?")

        assert result.output == "done"
        assert router.loads == [], "their load_skill must not be routed to DecimalAI"

    def test_a_run_time_toolset_with_load_skill_runs(self, pa, router):
        pa.instrument(enable_skill_loader=True)
        script = _Model(call_tool=NAMESPACED)
        agent = Agent(script.model(), name="support")

        agent.run_sync("where is my refund?", toolsets=[_theirs()])

        assert router.loads == [SKILL]
        assert NAMESPACED in script.views[0]

    def test_their_own_tool_plain_registered_after_construction(self, pa, router):
        """`@agent.tool_plain def load_skill` on an instrumented Agent used to
        raise at the decorator, before the user's code even ran."""
        pa.instrument(enable_skill_loader=True)
        script = _Model(call_tool=NAMESPACED)
        agent = Agent(script.model(), name="support")

        @agent.tool_plain
        def load_skill(name: str) -> str:  # noqa: F811 - the clash IS the test
            return f"MINE[{name}]"

        agent.run_sync("where is my refund?")
        assert router.loads == [SKILL]

    def test_the_clash_warning_is_logged_once(self, pa, router, caplog):
        pa.instrument(enable_skill_loader=True)
        with caplog.at_level(logging.WARNING, logger="decimalai.pydantic_ai"):
            for _ in range(3):
                Agent(_Model().model(), name="support", toolsets=[_theirs()]).run_sync("q")
        clash = [r for r in caplog.records if NAMESPACED in r.getMessage()]
        assert len(clash) == 1

    def test_without_a_clash_the_tool_is_load_skill(self, pa, router):
        """The common case is unchanged: the tool is `load_skill`, the name the
        menu hint, the conformance driver and the docs all use."""
        from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT

        pa.instrument(enable_skill_loader=True)
        script = _Model(call_tool="load_skill")
        agent = Agent(script.model(), name="support")

        agent.run_sync("where is my refund?")

        assert script.tool_names[0] == ["load_skill"]
        assert router.loads == [SKILL]
        assert LOAD_SKILL_PROMPT_HINT in script.views[0]
        assert NAMESPACED not in script.views[0]


    def test_a_context_without_toolsets_keeps_the_plain_name(self, pa, monkeypatch):
        """Nothing to inspect is not a reason to rename. The hint for a context
        that carries no toolsets names `load_skill`, and does not depend on
        whether `pydantic_ai.toolsets` happens to be importable at that moment."""
        import sys

        from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT

        monkeypatch.setitem(sys.modules, "pydantic_ai.toolsets", None)
        assert pa._exposed_tool_name(SimpleNamespace(agent=None)) == "load_skill"
        assert pa._load_skill_hint(SimpleNamespace(agent=None)) == LOAD_SKILL_PROMPT_HINT


# ── 3. routing and the trace agree on whose run this is ──────────────────────


class TestRoutingName:
    def test_an_unnamed_agent_routes_under_the_bound_name(self, pa, router, pipeline):
        pa.instrument(agent_name="support", enable_skill_loader=True)
        agent = Agent(_Model().model())  # Pydantic AI will infer "agent"

        agent.run_sync("where is my refund?")

        assert [c.agent_name for c in router.calls] == ["support"]
        assert [t.agent_name for t in pipeline.traces()] == ["support"]

    def test_an_explicit_name_still_wins(self, pa, router, pipeline):
        pa.instrument(agent_name="support", enable_skill_loader=True)
        Agent(_Model().model(), name="order-bot").run_sync("q")

        assert [c.agent_name for c in router.calls] == ["order-bot"]
        assert [t.agent_name for t in pipeline.traces()] == ["order-bot"]

    def test_an_explicit_name_wins_without_the_run_scope_too(self, pa, router):
        """`trace_runs=False` must not let the process-wide fallback take a name
        the caller chose."""
        pa.instrument(agent_name="support", enable_skill_loader=True, trace_runs=False)
        Agent(_Model().model(), name="order-bot").run_sync("q")

        assert [c.agent_name for c in router.calls] == ["order-bot"]

    def test_without_a_bound_name_the_inferred_one_is_used(self, pa, router, pipeline):
        """No `agent_name=`, no new opinion: routing and trace both use what
        Pydantic AI inferred, so they still agree."""
        pa.instrument(enable_skill_loader=True)
        agent = Agent(_Model().model())

        agent.run_sync("q")

        assert [c.agent_name for c in router.calls] == ["agent"]
        assert [t.agent_name for t in pipeline.traces()] == ["agent"]


# ── 4. the router's blocking call stays off the event loop ───────────────────


class TestOffTheEventLoop:
    def test_routing_runs_in_a_worker_thread(self, pa, router):
        pa.instrument(enable_skill_loader=True)
        Agent(_Model().model(), name="support").run_sync("q")

        assert [c.on_loop for c in router.calls] == [False], (
            "build_prompt_fragment ran on the event loop's thread"
        )

    def test_concurrent_runs_are_not_serialised_by_the_router(self, pa, monkeypatch):
        """Two runs on one loop, and a router call that cannot finish until the
        OTHER run's router call has started. On the loop, the first call holds
        the loop and the second never starts: the barrier breaks and neither
        run is routed. Off the loop, both are in flight at once."""

        class _RendezvousRouter(_Router):
            barrier = threading.Barrier(2, timeout=3)

            def build_prompt_fragment(self, *a, **kw):
                self.barrier.wait()
                return super().build_prompt_fragment(*a, **kw)

        both_at_once = _RendezvousRouter()
        monkeypatch.setattr(pa, "_skill_router_singleton", both_at_once)
        pa.instrument(enable_skill_loader=True)
        script = _Model()
        agent = Agent(script.model(), name="support")

        async def both():
            await asyncio.gather(agent.run("one"), agent.run("two"))

        asyncio.run(both())

        assert sorted(c.query for c in both_at_once.calls) == ["one", "two"]
        assert all("MENU-FOR[" in view for view in script.views)

    def test_the_routing_id_is_visible_to_the_awaiting_context(self, pa, router):
        """Offloading must not strand the routing id in the worker's copy of the
        context: the coroutine that awaited the prompt function reads it."""
        ctx = SimpleNamespace(prompt="q", agent=SimpleNamespace(name="support"))

        async def go():
            fragment = await pa._skills_system_prompt(ctx)
            return fragment, pa.get_current_routing_id()

        fragment, routing_id = asyncio.run(go())
        assert "MENU-FOR[q]" in fragment
        assert routing_id == "rt_" + format(1, "024x")
