"""Tests for the routing-telemetry rails: routing_id + offered names on adapter traces.

``routing_id`` and ``skills_offered_in_prompt`` shipped NULL on every
langchain / openai-agents trace even when a menu was demonstrably injected
and a skill demonstrably used. Both adapters carried the values on
ContextVars written during prompt assembly — but LangChain dispatches its
callbacks under ``copy_context()`` and the Agents runner copies the context
around the dynamic instructions callable, so the write lands in a child
context the trace-send path never sees. Same reality the loaded-names rail
already works around (tests/test_load_skill_loaded_rail.py); this extends
that instance-state pattern to the routing id and the offered/delivered
name sets, with the contextvars kept as the authoritative source wherever
they DO propagate (the generic tracer).

All network mocked (patch ``router.smart_route`` / ``router.get_menu`` —
same idiom as tests/test_load_skill_loaded_rail.py).
"""

from __future__ import annotations

import contextvars
import importlib
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter


def _module(name: str):
    """Resolve a module through ``sys.modules`` — the namespace the code
    under test actually runs in.

    ``tests/test_openai_agents.py`` pops the ``decimalai*`` entries and
    reimports to exercise the missing-dependency path; conftest's
    ``_restore_decimalai_modules`` puts the original instances back in
    ``sys.modules``, but the *package attribute* keeps pointing at the
    orphaned reimport. So a plain ``import decimalai.openai_agents as oa``
    binds one module instance while the running ``_send_trace`` reads the
    other's globals, and a singleton set on `oa` is invisible to the drain.
    """
    importlib.import_module(name)
    return sys.modules[name]


@pytest.fixture(autouse=True)
def _clean_contextvar_rails():
    """The offered/delivered rails and the body budget are module-level
    ContextVars shared by every test in this thread — clear them around
    each test (same as tests/test_load_skill_loaded_rail.py)."""
    for var in (sr._last_offered_names_ctx, sr._last_delivered_names_ctx, sr._body_budget_ctx):
        var.set(None)
    yield
    for var in (sr._last_offered_names_ctx, sr._last_delivered_names_ctx, sr._body_budget_ctx):
        var.set(None)


def _router(**kw) -> SkillRouter:
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", **kw)


def _route_result(routing_id="rt_1", names=("s1", "s2")):
    """The shape smart_route / get_menu return."""
    return {
        "prompt_fragment": "## Recommended Skills\n| " + " | ".join(names) + " |",
        "routing_id": routing_id,
        "skills": [{"name": n} for n in names],
    }


# ── The rails themselves ─────────────────────────────────────────


class TestRoutingRails:
    def test_build_prompt_fragment_records_the_decision(self):
        router = _router()
        with patch.object(router, "smart_route", return_value=_route_result()):
            _, routing_id = router.build_prompt_fragment(query="q")

        assert routing_id == "rt_1"
        assert router.consume_routing_id() == "rt_1"
        assert router.consume_offered_names() == ["s1", "s2"]
        # Drained — a second consume must not re-emit into the next trace.
        assert router.consume_routing_id() is None
        assert router.consume_offered_names() == []

    def test_full_menu_mode_records_too(self):
        """No query → get_menu, not smart_route. Same rail."""
        router = _router()
        with patch.object(router, "get_menu", return_value=_route_result("rt_menu")):
            router.build_prompt_fragment(query=None)

        assert router.consume_routing_id() == "rt_menu"
        assert router.consume_offered_names() == ["s1", "s2"]

    def test_names_accumulate_and_routing_id_is_last_write(self):
        """A multi-LLM-call turn routes more than once before the adapter
        drains at trace-send."""
        router = _router()
        with patch.object(router, "smart_route", return_value=_route_result("rt_1", ("s1",))):
            router.build_prompt_fragment(query="one")
        with patch.object(router, "smart_route", return_value=_route_result("rt_2", ("s1", "s2"))):
            router.build_prompt_fragment(query="two")

        assert router.consume_offered_names() == ["s1", "s2"]  # deduped, order preserved
        assert router.consume_routing_id() == "rt_2"

    def test_cache_hit_re_arms_the_rails(self):
        """The 30s fragment cache serves the second LLM call of a turn
        without a network hit — the rails must still carry the decision or
        a trace that starts mid-turn ships blank."""
        router = _router()
        with patch.object(router, "smart_route", return_value=_route_result()):
            router.build_prompt_fragment(query="q")
        router.consume_routing_id()
        router.consume_offered_names()

        with patch.object(router, "smart_route", side_effect=AssertionError("cache miss")):
            router.build_prompt_fragment(query="q")

        assert router.consume_routing_id() == "rt_1"
        assert router.consume_offered_names() == ["s1", "s2"]

    def test_delivered_bodies_recorded_separately(self):
        router = _router(inject_body=True, inject_body_top_k=1)
        with patch.object(router, "smart_route", return_value=_route_result()), \
                patch.object(router, "get_skill_body", return_value="BODY"):
            router.build_prompt_fragment(query="q")

        assert router.consume_offered_names() == ["s1", "s2"]
        assert router.consume_delivered_names() == ["s1"]  # top-1 body only

    def test_failed_route_records_nothing(self):
        router = _router()
        with patch.object(
            router, "smart_route", side_effect=sr.SkillRouterError("boom"),
        ):
            assert router.build_prompt_fragment(query="q") == ("", None)

        assert router.consume_routing_id() is None
        assert router.consume_offered_names() == []

    def test_drain_survives_context_isolation(self):
        """The root cause: real frameworks assemble the prompt in a copied
        context, so nothing context-local written there reaches the
        trace-send context — the instance rail must close that gap."""
        router = _router()

        def _route_in_isolated_context():
            with patch.object(router, "smart_route", return_value=_route_result()):
                router.build_prompt_fragment(query="q")

        contextvars.copy_context().run(_route_in_isolated_context)

        # The contextvar rail did not escape the copied context...
        assert sr.consume_last_offered_names() == []
        # ...but the instance rail carries the decision out.
        assert router.consume_routing_id() == "rt_1"
        assert router.consume_offered_names() == ["s1", "s2"]

    def test_rails_are_per_instance(self):
        """An undrained router must not leak its decision into another
        instance's drain (adapters each drain their own singleton)."""
        stale = _router()
        with patch.object(stale, "smart_route", return_value=_route_result()):
            stale.build_prompt_fragment(query="q")

        fresh = _router()
        assert fresh.consume_routing_id() is None
        assert fresh.consume_offered_names() == []


# ── openai_agents drain (trace assembly) ─────────────────────────


class TestOpenAIAgentsDrain:
    @pytest.fixture(autouse=True)
    def _reset_sdk(self, monkeypatch):
        """Mirror tests/test_load_skill_loaded_rail.py — fresh config +
        mocked client, clean router singleton and clean adapter rails.
        Everything goes through monkeypatch so an enabled config does not
        outlive the class: a leftover one turns CallbackHandler auto-send
        on for later test files that build handlers by hand."""
        cfg = _module("decimalai._config")
        oa = _module("decimalai.openai_agents")
        from decimalai._config import DecimalConfig

        client = MagicMock()
        client.register_manifest.return_value = {
            "manifest_id": "test-manifest-id", "status": "active",
        }
        monkeypatch.setattr(cfg, "_config", DecimalConfig(
            api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
        ))
        monkeypatch.setattr(cfg, "_client", client)
        monkeypatch.setattr(oa, "_manifest_id", None)
        monkeypatch.setattr(oa, "_skill_router_singleton", None)
        oa._routing_id_ctx.set(None)
        oa._skills_offered_ctx.set(None)
        oa._skills_delivered_ctx.set(None)
        yield

    def _run_one_trace(self):
        cfg = _module("decimalai._config")
        oa = _module("decimalai.openai_agents")

        processor = oa.DecimalTracingProcessor(agent_name="test-agent")
        mock_trace = MagicMock(trace_id=f"trace_{uuid4().hex[:16]}", name="wf")
        processor.on_trace_start(mock_trace)
        processor.on_trace_end(mock_trace)

        cfg._sender.flush()
        return cfg._client.ingest_trace.call_args[0][0]

    def test_trace_recovers_the_decision_from_a_copied_context(self):
        """The reported failure: the instructions callable runs in a copied
        context, so `_consume_routing_id()` / `_consume_skills_offered()`
        both came back empty at trace-send and the trace shipped NULL."""
        oa = _module("decimalai.openai_agents")

        router = _router()
        oa._skill_router_singleton = router

        def _assemble_prompt_in_isolated_context():
            with patch.object(router, "smart_route", return_value=_route_result()):
                _, routing_id = router.build_prompt_fragment(query="q")
            # What the skill loader callable does, in the SAME copied context.
            oa._set_routing_id(routing_id)
            oa._add_skills_offered(sr.consume_last_offered_names())

        contextvars.copy_context().run(_assemble_prompt_in_isolated_context)

        # Neither contextvar write escaped...
        assert oa._consume_routing_id() is None
        assert oa._consume_skills_offered() == []

        run_trace = self._run_one_trace()
        assert run_trace.routing_id == "rt_1"
        assert run_trace.skills_offered_in_prompt == ["s1", "s2"]
        # Drained — the next trace must not repeat this decision.
        assert router.consume_routing_id() is None
        assert router.consume_offered_names() == []

    def test_offered_is_more_than_the_loaded_names(self):
        """Regression guard: the loaded rail alone made a trace look like
        it offered only what the model pulled. The whole menu is offered."""
        oa = _module("decimalai.openai_agents")

        router = _router()
        oa._skill_router_singleton = router
        with patch.object(router, "smart_route", return_value=_route_result()):
            router.build_prompt_fragment(query="q")
        with patch.object(router, "get_skill_body", return_value="BODY"):
            router.load_skill("s1")

        run_trace = self._run_one_trace()
        assert run_trace.skills_loaded_by_agent == ["s1"]
        assert run_trace.skills_offered_in_prompt == ["s1", "s2"]

    def test_delivered_rail_reaches_the_trace(self):
        oa = _module("decimalai.openai_agents")

        router = _router(inject_body=True, inject_body_top_k=1)
        oa._skill_router_singleton = router
        with patch.object(router, "smart_route", return_value=_route_result()), \
                patch.object(router, "get_skill_body", return_value="BODY"):
            router.build_prompt_fragment(query="q")

        run_trace = self._run_one_trace()
        assert run_trace.skills_delivered == ["s1"]
        assert run_trace.skills_offered_in_prompt == ["s1", "s2"]

    def test_contextvar_still_wins_where_it_propagates(self):
        """Don't regress the paths that already worked: a routing_id that
        DID reach this context takes precedence, and the rail is drained
        anyway so it can't leak into the next trace."""
        oa = _module("decimalai.openai_agents")

        router = _router()
        oa._skill_router_singleton = router
        with patch.object(router, "smart_route", return_value=_route_result("rt_rail")):
            router.build_prompt_fragment(query="q")
        oa._set_routing_id("rt_ctx")

        run_trace = self._run_one_trace()
        assert run_trace.routing_id == "rt_ctx"
        assert router.consume_routing_id() is None

    def test_trace_without_routing_stays_null(self):
        run_trace = self._run_one_trace()
        assert run_trace.routing_id is None
        assert run_trace.skills_offered_in_prompt == []

    def test_router_without_the_rails_never_breaks_the_run(self):
        """A router object from an older SDK has none of the consume_*
        methods — telemetry degrades, the trace still ships."""
        oa = _module("decimalai.openai_agents")
        oa._skill_router_singleton = object()

        run_trace = self._run_one_trace()
        assert run_trace.routing_id is None
        assert run_trace.skills_offered_in_prompt == []

    def test_stray_string_is_not_split_into_letters(self):
        """`set.update("abc")` would add 'a', 'b', 'c' as three skills."""
        oa = _module("decimalai.openai_agents")
        oa._skill_router_singleton = MagicMock(
            consume_routing_id=lambda: 7,          # not a str → dropped
            consume_offered_names=lambda: "abc",   # not a list → dropped
            consume_delivered_names=lambda: ["", "  ", "ok"],
            consume_loaded_names=lambda: [],
        )

        run_trace = self._run_one_trace()
        assert run_trace.routing_id is None
        assert run_trace.skills_offered_in_prompt == ["ok"]


# ── langchain drain (trace assembly) ─────────────────────────────


class TestLangchainDrain:
    @pytest.fixture(autouse=True)
    def _clean_adapter_rails(self):
        lc = _module("decimalai.langchain")

        lc._routing_id_ctx.set(None)
        lc._skills_offered_ctx.set(None)
        lc._skills_delivered_ctx.set(None)
        yield

    def _build_trace(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="lc-agent")
        handler._trace_started_at = datetime.now(timezone.utc)
        with patch("decimalai._config._config", None):
            return handler.build_trace()

    def test_trace_recovers_the_decision_from_a_copied_context(self, monkeypatch):
        """LangChain dispatches callbacks under `copy_context()`, so the
        BaseChatModel patch's contextvar writes never reach build_trace."""
        lc = _module("decimalai.langchain")

        router = _router()
        monkeypatch.setattr(lc, "_skill_router_singleton", router)

        def _inject_in_isolated_context():
            with patch.object(router, "smart_route", return_value=_route_result()):
                _, routing_id = router.build_prompt_fragment(query="q")
            lc._set_routing_id(routing_id)
            lc._add_skills_offered(sr.consume_last_offered_names())

        contextvars.copy_context().run(_inject_in_isolated_context)

        assert lc._consume_routing_id() is None
        assert lc._consume_skills_offered() == []

        trace = self._build_trace()
        assert trace.routing_id == "rt_1"
        assert trace.skills_offered_in_prompt == ["s1", "s2"]
        assert router.consume_routing_id() is None
        assert router.consume_offered_names() == []

    def test_offered_is_more_than_the_loaded_names(self, monkeypatch):
        lc = _module("decimalai.langchain")

        router = _router()
        monkeypatch.setattr(lc, "_skill_router_singleton", router)
        with patch.object(router, "smart_route", return_value=_route_result()):
            router.build_prompt_fragment(query="q")
        with patch.object(router, "get_skill_body", return_value="BODY"):
            router.load_skill("s1")

        trace = self._build_trace()
        assert trace.skills_loaded_by_agent == ["s1"]
        assert trace.skills_offered_in_prompt == ["s1", "s2"]

    def test_contextvar_still_wins_where_it_propagates(self, monkeypatch):
        lc = _module("decimalai.langchain")

        router = _router()
        monkeypatch.setattr(lc, "_skill_router_singleton", router)
        with patch.object(router, "smart_route", return_value=_route_result("rt_rail")):
            router.build_prompt_fragment(query="q")
        lc._set_routing_id("rt_ctx")

        trace = self._build_trace()
        assert trace.routing_id == "rt_ctx"
        assert router.consume_routing_id() is None

    def test_trace_without_routing_stays_null(self, monkeypatch):
        lc = _module("decimalai.langchain")

        monkeypatch.setattr(lc, "_skill_router_singleton", None)
        trace = self._build_trace()
        assert trace.routing_id is None
        assert trace.skills_offered_in_prompt == []
