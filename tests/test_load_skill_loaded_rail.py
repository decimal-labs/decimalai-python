"""Tests for the loaded-names rail: load_skill serves recorded on adapter traces.

Under the openai_agents / langchain adapters there is no generic
``@decimalai.trace`` context, so ``generic.log_skill_loaded`` raises and
``load_skill`` used to swallow the failure at debug level — the DB then
showed ``skills_loaded_by_agent = null`` even when the model verifiably
received the body mid-run. The fix records each served body on the router
instance (a contextvar cannot work here: the framework's tool executor
runs in a copied context that never propagates back to the trace-send
path — the same reality behind the ``_last_budget`` fallback), and each
adapter's trace assembly drains its router singleton via
``consume_loaded_names()`` into the trace payload.

Also covers the SkillRouter base_url resolution (explicit →
DECIMAL_BASE_URL → default) — a directly-constructed router used to
hardcode the public host and silently split non-default deployments.

All network mocked (patch ``router.get_skill_body`` — same idiom as
tests/test_load_skill_budget.py).
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter


@pytest.fixture(autouse=True)
def _fresh_body_budget():
    """The body budget lives in a module-level ContextVar shared by every
    test in this thread — clear it around each test (same as
    tests/test_load_skill_budget.py)."""
    sr._body_budget_ctx.set(None)
    yield
    sr._body_budget_ctx.set(None)


def _router(**kw) -> SkillRouter:
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", **kw)


# ── The rail itself ──────────────────────────────────────────────


class TestLoadedNamesRail:
    def test_load_skill_records_without_a_generic_trace(self):
        """The adapter case: no @decimalai.trace context, so the generic
        telemetry call raises — the rail must still carry the name."""
        router = _router()
        with patch.object(router, "get_skill_body", return_value="BODY"):
            out = router.load_skill("code-review")

        assert out == "## Skill: code-review\n\nBODY"
        assert router.consume_loaded_names() == ["code-review"]
        # Drained — a second consume must not re-emit.
        assert router.consume_loaded_names() == []

    def test_repeat_load_recorded_once(self):
        router = _router()
        with patch.object(router, "get_skill_body", return_value="BODY"):
            router.load_skill("code-review")
            router.load_skill("code-review")  # dedup re-load is free
            router.load_skill("pdf-extract")

        assert router.consume_loaded_names() == ["code-review", "pdf-extract"]

    def test_refused_and_not_found_loads_not_recorded(self):
        router = _router(max_loaded_bodies=1)
        with patch.object(
            router, "get_skill_body",
            side_effect=lambda n, **kw: None if n == "ghost" else "BODY",
        ):
            router.load_skill("ghost")  # not found — no body served
            router.load_skill("a")
            refusal = router.load_skill("b")  # count budget exhausted

        assert "budget exhausted" in refusal
        assert router.consume_loaded_names() == ["a"]

    def test_drain_survives_context_isolation(self):
        """Real frameworks execute the tool in a copied context (asyncio
        tasks), so nothing context-local written inside load_skill reaches
        the trace-send context — the instance rail must close that gap."""
        router = _router()

        def _load_in_isolated_context():
            with patch.object(router, "get_skill_body", return_value="BODY"):
                router.load_skill("code-review")

        contextvars.copy_context().run(_load_in_isolated_context)

        assert router.consume_loaded_names() == ["code-review"]

    def test_rails_are_per_instance(self):
        """An undrained router must not leak its loads into another
        instance's drain (adapters each drain their own singleton)."""
        stale = _router()
        with patch.object(stale, "get_skill_body", return_value="BODY"):
            stale.load_skill("stale-skill")

        fresh = _router()
        assert fresh.consume_loaded_names() == []


# ── openai_agents drain (trace assembly) ─────────────────────────


class TestOpenAIAgentsDrain:
    @pytest.fixture(autouse=True)
    def _reset_sdk(self, monkeypatch):
        """Mirror tests/test_openai_agents.py — fresh config + mocked client,
        plus a clean router singleton (the drain reads it)."""
        import decimalai._config as cfg
        import decimalai.openai_agents as oa
        from decimalai._config import DecimalConfig

        cfg._config = DecimalConfig(
            api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
        )
        cfg._client = MagicMock()
        cfg._client.register_manifest.return_value = {
            "manifest_id": "test-manifest-id", "status": "active",
        }
        oa._manifest_id = None
        monkeypatch.setattr(oa, "_skill_router_singleton", None)
        yield

    def _run_one_trace(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"
        mock_trace = MagicMock(trace_id=trace_id, name="wf")
        processor.on_trace_start(mock_trace)
        processor.on_trace_end(mock_trace)

        import decimalai._config as cfg
        from decimalai._config import _sender
        _sender.flush()
        return cfg._client.ingest_trace.call_args[0][0]

    def test_send_trace_drains_loaded_rail(self):
        import decimalai.openai_agents as oa

        router = _router()
        oa._skill_router_singleton = router
        with patch.object(router, "get_skill_body", return_value="BODY"):
            router.load_skill("code-review")

        run_trace = self._run_one_trace()
        assert run_trace.skills_loaded_by_agent == ["code-review"]
        # Loaded implies offered + delivered (same ladder as log_skill_loaded).
        assert "code-review" in run_trace.skills_delivered
        assert "code-review" in run_trace.skills_offered_in_prompt
        # Drained — the next trace must not repeat the load.
        assert router.consume_loaded_names() == []

    def test_trace_without_loads_stays_empty(self):
        run_trace = self._run_one_trace()
        assert run_trace.skills_loaded_by_agent == []


# ── langchain drain (trace assembly) ─────────────────────────────


class TestLangchainDrain:
    def test_build_trace_drains_loaded_rail(self, monkeypatch):
        import decimalai.langchain as lc
        from decimalai.langchain import CallbackHandler

        router = _router()
        monkeypatch.setattr(lc, "_skill_router_singleton", router)
        with patch.object(router, "get_skill_body", return_value="BODY"):
            router.load_skill("code-review")

        handler = CallbackHandler(agent_name="lc-agent")
        handler._trace_started_at = datetime.now(timezone.utc)
        with patch("decimalai._config._config", None):
            trace = handler.build_trace()

        assert trace.skills_loaded_by_agent == ["code-review"]
        assert "code-review" in trace.skills_delivered
        assert "code-review" in trace.skills_offered_in_prompt
        # Rail drained — the next handler must not see the same load.
        assert router.consume_loaded_names() == []


# ── base_url resolution ──────────────────────────────────────────


class TestBaseUrlResolution:
    def test_default_is_public_host(self, monkeypatch):
        monkeypatch.delenv("DECIMAL_BASE_URL", raising=False)
        router = SkillRouter(api_key="dai_sk_test")
        assert router.base_url == "https://api.decimal.ai"

    def test_env_var_honored(self, monkeypatch):
        monkeypatch.setenv("DECIMAL_BASE_URL", "http://localhost:9999/")
        router = SkillRouter(api_key="dai_sk_test")
        assert router.base_url == "http://localhost:9999"

    def test_explicit_argument_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("DECIMAL_BASE_URL", "http://localhost:9999")
        router = SkillRouter(api_key="dai_sk_test", base_url="http://example.test/")
        assert router.base_url == "http://example.test"
