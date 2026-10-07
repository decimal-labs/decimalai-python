"""CrewAI delivers a skill BODY — proven at the model boundary.

Until 2026-10-08 ``decimalai/cli/scaffold.py::NO_PROMPT_SEAM`` listed CrewAI, and
that was true: tracing worked, and nothing in the SDK could put a skill into a
CrewAI prompt. ``decimalai/crewai.py`` now registers a ``before_llm_call`` hook.

The seam, read out of crewai 1.15.20 and 1.6.1 rather than remembered:

* ``crewai.hooks.register_before_llm_call_hook`` is public API from 1.5.0. Every
  agent executor copies the global hook list when it is built and runs it in
  ``utilities/agent_utils.py::_setup_before_llm_call_hooks``, right before
  ``llm.call``.
* ``LLMCallHookContext.messages`` IS the executor's live message list, and
  ``get_llm_response`` sends that same list once the hooks ran.
* The list persists across the iterations of one task, so the block from the
  previous call is still there on the next one — and CrewAI appends user-role
  reflection prompts of its own between calls.

Two layers, because crewai is not in the SDK's ``[dev]`` environment:

* ``TestSeamWithoutCrewAI`` drives the hook against a synthetic context, and the
  rail against a real OTel span and the real exporter, on every run of the suite.
* ``TestRealCrewAIRun`` walks a real ``Crew.kickoff`` with a real ``Agent`` and a
  ``BaseLLM`` subclass that CAPTURES the messages instead of sending them — the
  request is the one CrewAI built, the network is the only thing missing. Its
  ``test_the_synthetic_context_has_not_drifted`` pins the first layer's stand-in
  to the real ``LLMCallHookContext``.

The wire-level verdict — the trace a real crew POSTs carries the routing id, the
offered names and a body in the prompt the model was shown — is the conformance
suite's (``tests/conformance``, C8 / C13 / C14 / D1 for ``crewai``).
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# CrewAI phones home and can prompt interactively on a first run. Off before any
# crewai module loads, exactly as the conformance driver does.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


def _real_crewai_installed() -> bool:
    """Asked at IMPORT time, for the same reason the ADK tests do: Layer 1 puts
    a stub ``crewai.hooks`` in ``sys.modules`` inside a fixture."""
    try:
        return importlib.util.find_spec("crewai.hooks") is not None
    except (ImportError, ValueError):
        return False


HAS_REAL_CREWAI = _real_crewai_installed()

# A menu row could never contain this; only a delivered BODY can.
SENTINEL = "SENTINEL-SKILLBODY-CREWAI-5d1e9b"
CREW_SYSTEM = "You are Support. You are terse.\nYour personal goal is: Resolve the ticket"
QUESTION = "What fee applies to an opened box return?"
TASK_PROMPT = (
    f"\nCurrent Task: {QUESTION}\n\nThis is the expected criteria for your final "
    "answer: the fee\nyou MUST return the actual complete content as the final "
    "answer, not a summary.\n\nBegin! This is VERY important to you, use the tools "
    "available and give your best Final Answer, your job depends on it!\n\nThought:"
)
PREFIX = (
    "== DecimalAI skills ==\n"
    "| refund-policy | how refunds work |\n"
    "| returns-window | when returns close |\n\n"
    "## Skill: refund-policy\n\n"
    f"Opened boxes carry a 23.5% restocking fee. {SENTINEL}\n"
)
TAIL = "Most relevant for this request: refund-policy."
ROUTING_ID = "rt_" + "c" * 24
REFLECTION = (
    "Analyze the tool result. If requirements are met, provide the Final Answer."
)


# ── routers ─────────────────────────────────────────────────


class _SplitRouter:
    """Stand-in for the platform: a constant PREFIX carrying the body and a TAIL
    that varies with the query. Records every call it gets."""

    def __init__(self) -> None:
        self.queries: List[Optional[str]] = []
        self.scopes: List[Optional[str]] = []
        self.agent_names: List[Optional[str]] = []

    def _rails(self) -> None:
        import decimalai.skill_router as sr

        sr._last_offered_names_ctx.set(["refund-policy", "returns-window"])
        sr._last_delivered_names_ctx.set(["refund-policy"])

    def build_prompt_parts(self, query=None, *, agent_name=None, scope=None, **kw):
        self.queries.append(query)
        self.scopes.append(scope)
        self.agent_names.append(agent_name)
        self._rails()
        return PREFIX, TAIL, ROUTING_ID


class _LegacyRouter(_SplitRouter):
    """A router object built before the prefix/tail split existed."""

    build_prompt_parts = None  # not callable -> the adapter must degrade

    def build_prompt_fragment(self, query=None, *, agent_name=None, scope=None, **kw):
        self.queries.append(query)
        self._rails()
        return PREFIX, ROUTING_ID


class _EmptyRouter(_SplitRouter):
    """The platform had nothing to offer for this query."""

    def build_prompt_parts(self, query=None, **kw):
        self.queries.append(query)
        import decimalai.skill_router as sr

        sr._last_offered_names_ctx.set(None)
        sr._last_delivered_names_ctx.set(None)
        return "", "", None


class _ExplodingRouter:
    def build_prompt_parts(self, *a, **k):
        raise RuntimeError("platform down")


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Fresh SDK config, fresh adapter globals, an empty OTel skill rail."""
    import decimalai._config as cfg
    import decimalai.crewai as dc
    from decimalai._config import DecimalConfig
    from decimalai.otel import _active_agent_name, _reset_skill_rails

    saved_cfg = (cfg._config, cfg._client)
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }
    saved = (
        dc._tracing_installed, dc._skill_loader_installed,
        dc._install_agent_name, dc._skill_router_singleton,
    )
    dc._skill_loader_installed = False
    dc._install_agent_name = None
    dc._skill_router_singleton = None
    token = _active_agent_name.set(None)
    _reset_skill_rails()
    yield
    _reset_skill_rails()
    _active_agent_name.reset(token)
    (
        dc._tracing_installed, dc._skill_loader_installed,
        dc._install_agent_name, dc._skill_router_singleton,
    ) = saved
    cfg._config, cfg._client = saved_cfg


@pytest.fixture
def use_router(monkeypatch):
    def _use(router):
        import decimalai.crewai as dc

        monkeypatch.setattr(dc, "_skill_router_singleton", router)
        return router

    return _use


@pytest.fixture
def fake_hooks(monkeypatch):
    """A stand-in ``crewai.hooks`` with CrewAI's registry semantics: one global
    list, append on register, a copy on get."""
    registry: List[Any] = []
    mod = types.ModuleType("crewai.hooks")
    mod.register_before_llm_call_hook = registry.append
    mod.get_before_llm_call_hooks = lambda: list(registry)
    mod.clear_before_llm_call_hooks = registry.clear
    if "crewai" not in sys.modules:
        monkeypatch.setitem(sys.modules, "crewai", types.ModuleType("crewai"))
    monkeypatch.setitem(sys.modules, "crewai.hooks", mod)
    return registry


@pytest.fixture
def otel_run():
    """A live OTel span to attribute rails to — a crew's root, as the trace sees it.

    A LOCAL TracerProvider: setting the global one would leak into every test
    after this, since OTel honours ``set_tracer_provider`` once per process.
    """
    from opentelemetry.sdk.trace import TracerProvider

    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("Crew.kickoff") as span:
        yield span.get_span_context().trace_id


def _context(
    messages: Optional[List[Dict[str, Any]]] = None,
    *,
    description: Optional[str] = QUESTION,
    executor: Any = "executor",
) -> SimpleNamespace:
    """Mirrors ``crewai.hooks.LLMCallHookContext`` on the members the adapter
    reads. ``test_the_synthetic_context_has_not_drifted`` holds it to the real
    class wherever crewai is installed."""
    task = SimpleNamespace(description=description) if description is not None else None
    return SimpleNamespace(
        executor=executor,
        messages=messages if messages is not None else [
            {"role": "system", "content": CREW_SYSTEM},
            {"role": "user", "content": TASK_PROMPT},
        ],
        agent=SimpleNamespace(role="Support"),
        task=task,
        crew=None,
        llm=None,
        iterations=0,
    )


def _hook(context: Any) -> Any:
    from decimalai.crewai import _skill_hook

    return _skill_hook(context)


def _blocks(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [m for m in messages if PREFIX in str(m.get("content"))]


def _flushed_traces() -> List[Any]:
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in cfg._client.ingest_trace.call_args_list]


# ── layer 1: the seam, graded on every run of the suite ─────


class TestSeamWithoutCrewAI:
    def test_the_body_lands_after_crewais_own_system_prompt(self, use_router):
        """CrewAI's system prompt is the STABLE part of the request, so it stays
        first; the routed block follows it, and the task comes after both."""
        use_router(_SplitRouter())
        ctx = _context()
        assert _hook(ctx) is None
        roles = [m["role"] for m in ctx.messages]
        contents = [m["content"] for m in ctx.messages]
        assert roles == ["system", "system", "system", "user"]
        assert contents == [CREW_SYSTEM, PREFIX, TAIL, TASK_PROMPT]
        assert SENTINEL in contents[1]

    def test_a_later_call_of_the_same_task_replaces_the_block(self, use_router):
        """The executor's list PERSISTS across iterations. Inserting again
        without removing would stack a second copy of every body per tool call —
        on a ten-step task, ten copies of the same skill in one request."""
        router = use_router(_SplitRouter())
        ctx = _context()
        _hook(ctx)
        ctx.messages.append({"role": "assistant", "content": "Thought: look it up"})
        ctx.messages.append({"role": "user", "content": REFLECTION})
        _hook(ctx)
        assert len(_blocks(ctx.messages)) == 1, "the skill block was stacked, not replaced"
        assert [m["content"] for m in ctx.messages] == [
            CREW_SYSTEM, PREFIX, TAIL, TASK_PROMPT, "Thought: look it up", REFLECTION,
        ]
        # One task, one question — even though the newest user message is now
        # CrewAI's own reflection prompt.
        assert router.queries == [QUESTION, QUESTION]

    def test_routes_on_the_task_description_not_crewais_wrapper(self, use_router):
        router = use_router(_SplitRouter())
        _hook(_context())
        assert router.queries == [QUESTION]

    def test_without_a_task_routes_on_the_last_user_message(self, use_router):
        """``Agent.kickoff()``: no task, so the ask is the user's message."""
        router = use_router(_SplitRouter())
        ctx = _context(
            [{"role": "system", "content": CREW_SYSTEM},
             {"role": "user", "content": [{"type": "text", "text": QUESTION}]}],
            description=None,
        )
        _hook(ctx)
        assert router.queries == [QUESTION]
        assert len(_blocks(ctx.messages)) == 1

    def test_a_call_with_no_executor_is_left_alone(self, use_router):
        """crewai >= 1.7 dispatches these hooks for calls that are not an agent
        turn too — its output converter, guardrails, planner — with
        ``executor=None``. A skill body in a JSON-conversion prompt is noise, and
        claiming delivery for it would credit a call the agent never made."""
        router = use_router(_SplitRouter())
        ctx = _context(executor=None)
        before = [dict(m) for m in ctx.messages]
        assert _hook(ctx) is None
        assert ctx.messages == before
        assert router.queries == [], "the router was consulted for a non-agent call"

    def test_no_system_prompt_puts_the_block_first_as_a_user_message(self, use_router):
        """``Agent(use_system_prompt=False)`` is CrewAI's switch for models that
        REFUSE a system role. Inserting one would break exactly those calls."""
        use_router(_SplitRouter())
        ctx = _context([{"role": "user", "content": TASK_PROMPT}])
        _hook(ctx)
        assert [m["role"] for m in ctx.messages] == ["user", "user", "user"]
        assert ctx.messages[0]["content"] == PREFIX
        assert ctx.messages[-1]["content"] == TASK_PROMPT

    def test_the_rail_is_recorded_against_the_live_run(self, use_router, otel_run):
        from decimalai.otel import _pop_skill_rail

        use_router(_SplitRouter())
        _hook(_context())
        rail = _pop_skill_rail(otel_run)
        assert rail is not None, "the routing decision was not attributed to the run"
        assert rail["routing_id"] == ROUTING_ID
        assert rail["offered"] == ["refund-policy", "returns-window"]
        assert rail["delivered"] == ["refund-policy"]
        # Delivered is NOT an activation: there is no load_skill tool on this
        # rail, so nothing here can be a model-initiated pull.
        assert rail["loaded"] == []

    def test_routing_is_scoped_to_the_run_and_named_for_the_agent(
        self, use_router, otel_run,
    ):
        """Two concurrent kickoffs share the router singleton; scoping by the
        run's trace id keeps them from sharing a routing decision. And an
        agent-scope skill only resolves if the agent's name is sent."""
        import decimalai.crewai as dc

        router = use_router(_SplitRouter())
        dc._install_agent_name = "support"
        _hook(_context())
        assert router.scopes == [f"{otel_run:032x}"]
        assert router.agent_names == ["support"]

    def test_the_runs_own_agent_name_wins_over_the_installed_one(self, use_router):
        """The name the TRACE is filed under is the run-scoped one the OTel rail
        stamps onto its spans; routing for a different agent than the trace
        names would split one run's skills from its trace."""
        import decimalai.crewai as dc
        from decimalai.otel import agent_run

        router = use_router(_SplitRouter())
        dc._install_agent_name = "support"
        with agent_run("billing"):
            _hook(_context())
        assert router.agent_names == ["billing"]

    def test_outside_a_traced_run_nothing_is_attributed(self, use_router):
        """No live span, no run to own the decision: the block still goes in,
        and the rail is dropped rather than guessed at."""
        from decimalai.otel import _skill_rails

        router = use_router(_SplitRouter())
        ctx = _context()
        _hook(ctx)
        assert len(_blocks(ctx.messages)) == 1
        assert router.scopes == [None]
        assert len(_skill_rails) == 0

    def test_a_router_without_the_split_still_delivers(self, use_router, otel_run):
        """The silent no-op guard: swallowing the AttributeError and returning
        would trace perfectly and inject nothing."""
        from decimalai.otel import _pop_skill_rail

        use_router(_LegacyRouter())
        ctx = _context()
        _hook(ctx)
        assert [m["content"] for m in ctx.messages] == [CREW_SYSTEM, PREFIX, TASK_PROMPT]
        assert _pop_skill_rail(otel_run)["delivered"] == ["refund-policy"]

    def test_an_empty_route_claims_nothing(self, use_router, otel_run):
        from decimalai.otel import _pop_skill_rail

        use_router(_EmptyRouter())
        ctx = _context()
        _hook(ctx)
        assert [m["content"] for m in ctx.messages] == [CREW_SYSTEM, TASK_PROMPT]
        assert _pop_skill_rail(otel_run) is None, "a routing decision was claimed for nothing"

    def test_a_router_that_raises_never_breaks_the_model_call(self, use_router):
        """Returning False from a CrewAI before-hook BLOCKS the call. A routing
        failure must degrade to an unskilled call instead."""
        use_router(_ExplodingRouter())
        ctx = _context()
        assert _hook(ctx) is None
        assert [m["content"] for m in ctx.messages] == [CREW_SYSTEM, TASK_PROMPT]

    def test_a_failed_route_on_a_later_call_leaves_no_stale_block(self, use_router):
        """The previous call's block is removed before routing; if this call
        routes nothing, the model is not shown skills nobody routed for it."""
        import decimalai.crewai as dc

        use_router(_SplitRouter())
        ctx = _context()
        _hook(ctx)
        dc._skill_router_singleton = _ExplodingRouter()
        _hook(ctx)
        assert _blocks(ctx.messages) == []

    def test_bodies_are_on_by_default_because_crewai_has_no_load_skill_tool(
        self, monkeypatch,
    ):
        """No ``load_skill`` tool, so injection is the ONLY body channel — the
        ``has_tool_loop=False`` case ``resolve_inject_body`` answers True for."""
        import decimalai.crewai as dc

        captured: Dict[str, Any] = {}

        class _Router:
            def __init__(self, **kw):
                captured.update(kw)

        monkeypatch.setattr("decimalai.skill_router.SkillRouter", _Router)
        dc._get_skill_router()
        assert captured["inject_body"] is True

    def test_the_inject_body_kill_switch_is_honoured(self, monkeypatch):
        import decimalai._config as cfg
        import decimalai.crewai as dc
        from decimalai._config import DecimalConfig

        cfg._config = DecimalConfig(
            api_key="dai_sk_test", base_url="http://localhost:8000",
            enabled=True, inject_skill_body=False,
        )
        captured: Dict[str, Any] = {}

        class _Router:
            def __init__(self, **kw):
                captured.update(kw)

        monkeypatch.setattr("decimalai.skill_router.SkillRouter", _Router)
        dc._get_skill_router()
        assert captured["inject_body"] is False


class TestTheRealRouterOnThisRail:
    """The real ``SkillRouter``, with only its two HTTP reads stubbed — so the
    contextvar rails, the body budget and the scoped bookkeeping are the
    shipped code, not a stand-in's."""

    @staticmethod
    def _router(monkeypatch):
        from decimalai.skill_router import SkillRouter

        router = SkillRouter(
            api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=True,
        )
        result = {
            "prompt_fragment": "Available skills:\n- refund-policy: how refunds work",
            "routing_id": ROUTING_ID,
            "skills": [{"name": "refund-policy"}],
            "strategy": "semantic",
        }
        monkeypatch.setattr(router, "smart_route", lambda *a, **k: dict(result))
        monkeypatch.setattr(
            router, "get_skill_body",
            lambda name, **k: f"Opened boxes carry a 23.5% fee. {SENTINEL}",
        )
        return router

    def test_body_rail_and_released_scope(self, monkeypatch, use_router, otel_run):
        from decimalai.otel import _pop_skill_rail

        router = use_router(self._router(monkeypatch))
        ctx = _context()
        _hook(ctx)
        block = _blocks_with_sentinel(ctx.messages)
        assert len(block) == 1 and "## Skill: refund-policy" in block[0]["content"]

        rail = _pop_skill_rail(otel_run)
        assert rail["routing_id"] == ROUTING_ID
        assert rail["offered"] == ["refund-policy"]
        assert rail["delivered"] == ["refund-policy"]
        # Read from the per-call contextvars and recorded on the OTel rail, so
        # nothing would ever drain the router's per-run copy. Left behind, every
        # run would hold one of its 4096 slots until eviction started warning.
        assert f"{otel_run:032x}" not in router._scoped_routing_rails


def _blocks_with_sentinel(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [m for m in messages if SENTINEL in str(m.get("content"))]


class TestTheRailReachesTheTrace:
    """The rail the hook records is the one the OTel exporter stamps on the run's
    trace — through a real TracerProvider and the real ``DecimalSpanExporter``."""

    def test_offered_and_delivered_but_never_activated(self, use_router):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from decimalai.otel import DecimalSpanExporter

        use_router(_SplitRouter())
        provider = TracerProvider()
        provider.add_span_processor(
            SimpleSpanProcessor(DecimalSpanExporter(agent_name="support"))
        )
        tracer = provider.get_tracer("test")
        ctx = _context()
        with tracer.start_as_current_span("Crew.kickoff"):
            _hook(ctx)
            attrs: Dict[str, Any] = {
                "openinference.span.kind": "LLM",
                "llm.model_name": "gpt-4o-mini",
                "llm.token_count.prompt": 17,
                "llm.token_count.completion": 5,
            }
            for i, m in enumerate(ctx.messages):
                attrs[f"llm.input_messages.{i}.message.role"] = m["role"]
                attrs[f"llm.input_messages.{i}.message.content"] = m["content"]
            with tracer.start_as_current_span("ChatCompletion", attributes=attrs):
                pass

        traces = _flushed_traces()
        assert len(traces) == 1
        trace = traces[0]
        assert trace.routing_id == ROUTING_ID
        assert trace.skills_offered_in_prompt == ["refund-policy", "returns-window"]
        assert trace.skills_delivered == ["refund-policy"]
        # The prompt-injection rail stops at DELIVERED. The platform turns every
        # active_skills / skills_loaded_by_agent entry into an activation row;
        # stamping the delivered body there would fabricate one (C13).
        assert trace.active_skills == []
        assert trace.skills_loaded_by_agent == []
        rendered = "\n".join(str(c.rendered_input) for c in trace.llm_calls)
        assert SENTINEL in rendered, "the body is not in the prompt the trace says the model saw"


class TestInstrument:
    def test_the_hook_is_registered_once(self, fake_hooks):
        import decimalai.crewai as dc
        from decimalai.crewai import _skill_hook, instrument

        dc._tracing_installed = True  # tracing is not under test here
        instrument(agent_name="support", enable_skill_loader=True)
        instrument(agent_name="support", enable_skill_loader=True)
        assert fake_hooks == [_skill_hook]

    def test_cleared_hooks_come_back_on_the_next_instrument(self, fake_hooks):
        """``crewai.hooks.clear_all_global_hooks()`` in user code would otherwise
        leave a module flag saying "installed" over an empty registry."""
        import decimalai.crewai as dc
        from decimalai.crewai import _skill_hook, instrument

        dc._tracing_installed = True
        instrument(enable_skill_loader=True)
        fake_hooks.clear()
        instrument(enable_skill_loader=True)
        assert fake_hooks == [_skill_hook]

    def test_the_loader_is_off_by_default(self, fake_hooks):
        import decimalai.crewai as dc

        dc._tracing_installed = True
        dc.instrument(agent_name="support")
        assert fake_hooks == []

    def test_load_skill_tool_is_refused_out_loud(self, fake_hooks, caplog):
        """The refusal the conformance FrameworkLimit cites. It must be a WARNING
        on the run that asked — a debug line would let the tool_loaded cell's
        N/A stand on a sentence nobody saw."""
        import decimalai.crewai as dc

        dc._tracing_installed = True
        with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
            dc.instrument(enable_skill_loader=True, enable_load_skill_tool=True)
        assert "enable_load_skill_tool is not supported on the crewai adapter" in caplog.text
        assert fake_hooks == [dc._skill_hook], "the refusal must not cost the injection rail"

    def test_a_crewai_without_the_hook_api_warns_and_does_not_raise(
        self, monkeypatch, caplog,
    ):
        import decimalai.crewai as dc

        monkeypatch.setitem(sys.modules, "crewai.hooks", None)
        with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
            assert dc._install_skill_loader() is False
        assert "crewai>=1.15.3" in caplog.text

    def test_tracing_is_installed_once(self, fake_hooks, monkeypatch):
        """``instrument()`` installs what ``init(crewai=True)`` does — once. A
        second exporter would sit on a provider no instrumentor feeds."""
        import decimalai
        import decimalai.crewai as dc
        import decimalai.otel as otel

        dc._tracing_installed = False
        built: List[Any] = []
        activated: List[Any] = []
        monkeypatch.setattr(otel, "instrument", lambda agent_name=None: built.append(agent_name) or "provider")
        monkeypatch.setattr(
            decimalai, "_activate_crewai_instrumentation",
            lambda provider, agent_name=None: activated.append((provider, agent_name)),
        )
        dc.instrument(agent_name="support")
        dc.instrument(agent_name="support", enable_skill_loader=True)
        assert built == ["support"]
        assert activated == [("provider", "support")]

    def test_init_crewai_counts_as_tracing_installed(self, monkeypatch):
        """``init(crewai=True)`` then ``crewai.instrument(enable_skill_loader=True)``
        must add the loader without building a second exporter."""
        import decimalai
        import decimalai.crewai as dc
        import decimalai.otel as otel

        dc._tracing_installed = False
        monkeypatch.setitem(sys.modules, "openinference.instrumentation.crewai", None)
        decimalai._activate_crewai_instrumentation(MagicMock())
        assert dc._tracing_installed is True
        monkeypatch.setattr(
            otel, "instrument",
            lambda **k: pytest.fail("a second exporter was built"),
        )
        dc._install_tracing("support")

    def test_inits_agent_name_routes_a_crew_on_a_worker_thread(
        self, monkeypatch, use_router,
    ):
        """``init(crewai=True, agent_name=...)`` files the traces under that name,
        so the skills must be routed for it too — including for a crew kicked off
        on a worker thread, which a fresh thread's empty context gives no
        run-scoped name. Without the hand-off the router would see no agent, and
        every agent-scope skill would silently drop out of the menu."""
        import threading

        import decimalai

        router = use_router(_SplitRouter())
        monkeypatch.setitem(sys.modules, "openinference.instrumentation.crewai", None)
        decimalai._activate_crewai_instrumentation(MagicMock(), agent_name="support")

        worker = threading.Thread(target=lambda: _hook(_context()))
        worker.start()
        worker.join()
        assert router.agent_names == ["support"]

    def test_a_later_name_travels_with_the_spans(self, monkeypatch):
        """The exporter's default name is fixed when it is built; a second
        instrument() can only name the runs through the run-scoped var."""
        import decimalai.crewai as dc
        from decimalai.otel import _active_agent_name

        dc._tracing_installed = True
        dc.instrument(agent_name="billing")
        assert _active_agent_name.get() == "billing"


# ── layer 2: the real thing ─────────────────────────────────


@pytest.mark.skipif(
    not HAS_REAL_CREWAI,
    reason="crewai is not an SDK dependency; the seam is still graded by "
           "TestSeamWithoutCrewAI on every run and on the wire by tests/conformance",
)
class TestRealCrewAIRun:
    """A real ``Crew.kickoff`` over a real ``Agent``.

    The model is a ``BaseLLM`` subclass that captures the messages it is handed
    instead of calling a provider. Everything upstream of the wire — the prompt
    builder, the executor, the hook dispatch — is CrewAI's own code.
    """

    @pytest.fixture(autouse=True)
    def _loader(self):
        """Register the real hook in CrewAI's real registry, and take it out
        after, so no other test in the session gets skills injected."""
        from crewai.hooks import unregister_before_llm_call_hook

        import decimalai.crewai as dc

        dc._tracing_installed = True  # no global tracer provider from a unit test
        dc.instrument(agent_name="support", enable_skill_loader=True)
        yield
        unregister_before_llm_call_hook(dc._skill_hook)

    @staticmethod
    def _llm(turns: List[str]):
        from crewai.llms.base_llm import BaseLLM

        seen: List[List[Dict[str, Any]]] = []

        class _CapturingLLM(BaseLLM):
            def call(self, messages, tools=None, callbacks=None, available_functions=None,
                     from_task=None, from_agent=None, response_model=None, **kw):
                seen.append([dict(m) for m in messages])
                return turns[min(len(seen) - 1, len(turns) - 1)]

            async def acall(self, messages, tools=None, callbacks=None, available_functions=None,
                            from_task=None, from_agent=None, response_model=None, **kw):
                return self.call(messages)

            def supports_function_calling(self) -> bool:
                return False  # ReAct text tools: works on every crewai with hooks

            def supports_stop_words(self) -> bool:
                return True

            def get_context_window_size(self) -> int:
                return 8192

        return _CapturingLLM(model="capturing-stub"), seen

    @staticmethod
    def _crew(llm, tools=()):
        from crewai import Agent, Crew, Task

        agent = Agent(
            role="Support", goal="Resolve the ticket", backstory="You are terse.",
            llm=llm, tools=list(tools), verbose=False,
        )
        task = Task(description=QUESTION, expected_output="the fee", agent=agent)
        return Crew(agents=[agent], tasks=[task], verbose=False)

    def test_the_sentinel_is_in_the_request_the_model_was_handed(self, use_router):
        router = use_router(_SplitRouter())
        llm, seen = self._llm(["Thought: I know it\nFinal Answer: 23.5%"])
        out = self._crew(llm).kickoff()

        assert seen, "the model was never called"
        messages = seen[0]
        assert any(SENTINEL in str(m["content"]) for m in messages), (
            "CrewAI built a request with no skill body in it — the seam does not work"
        )
        system = [m["content"] for m in messages if m["role"] == "system"]
        assert system[0].startswith("You are Support"), "CrewAI's own prompt lost first place"
        assert system[1:] == [PREFIX, TAIL]
        assert messages[-1]["role"] == "user" and QUESTION in messages[-1]["content"]
        assert router.queries == [QUESTION]
        assert "23.5%" in str(out)

    def test_a_tool_turn_keeps_one_block_and_the_tasks_question(self, use_router):
        """Two model calls in one task, with CrewAI's own observation and
        reflection turns appended between them."""
        from crewai.tools import tool

        @tool("lookup")
        def lookup(query: str) -> str:
            """Look a value up."""
            return f"value for {query}"

        router = use_router(_SplitRouter())
        llm, seen = self._llm([
            'Thought: I need the tool\nAction: lookup\nAction Input: {"query": "box"}',
            "Thought: I know it\nFinal Answer: 23.5%",
        ])
        self._crew(llm, tools=[lookup]).kickoff()

        assert len(seen) == 2, f"expected a tool round-trip, got {len(seen)} model calls"
        for step, messages in enumerate(seen):
            bodies = [m for m in messages if SENTINEL in str(m["content"])]
            assert len(bodies) == 1, f"model call {step} carried {len(bodies)} skill blocks"
        assert router.queries == [QUESTION, QUESTION], router.queries

    def test_kickoff_async_delivers_too(self, use_router):
        use_router(_SplitRouter())
        llm, seen = self._llm(["Thought: I know it\nFinal Answer: 23.5%"])
        asyncio.run(self._crew(llm).kickoff_async())
        assert seen and any(SENTINEL in str(m["content"]) for m in seen[0])

    def test_agent_kickoff_without_a_crew_routes_on_the_users_message(self, use_router):
        """``Agent.kickoff()``: no crew, no task — the ask is the message."""
        from crewai import Agent

        router = use_router(_SplitRouter())
        llm, seen = self._llm(["Thought: I know it\nFinal Answer: 23.5%"])
        agent = Agent(
            role="Support", goal="Resolve the ticket", backstory="You are terse.",
            llm=llm, verbose=False,
        )
        agent.kickoff(QUESTION)
        assert seen and any(SENTINEL in str(m["content"]) for m in seen[0])
        assert router.queries and all(QUESTION in str(q) for q in router.queries), (
            router.queries
        )

    def test_the_synthetic_context_has_not_drifted(self, use_router):
        """Pin ``_context`` to CrewAI's real ``LLMCallHookContext``: built from an
        executor, its ``messages`` must be the executor's own list, and the
        adapter's insert must land in that list."""
        from crewai.hooks import LLMCallHookContext

        from decimalai.crewai import _skill_hook

        use_router(_SplitRouter())
        live = [{"role": "system", "content": CREW_SYSTEM},
                {"role": "user", "content": TASK_PROMPT}]
        executor = SimpleNamespace(
            messages=live, llm=None, iterations=0,
            agent=SimpleNamespace(role="Support"),
            task=SimpleNamespace(description=QUESTION), crew=None,
        )
        context = LLMCallHookContext(executor)
        assert context.executor is executor
        assert context.messages is live
        assert context.task.description == QUESTION
        assert _skill_hook(context) is None
        assert [m["content"] for m in live] == [CREW_SYSTEM, PREFIX, TAIL, TASK_PROMPT]
