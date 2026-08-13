"""Tests for native-tracer routing_id stamping.

The framework adapters (langchain / anthropic / pydantic_ai) already stamp
a SkillRouter `routing_id` onto the RunTrace so the backend can correlate
which skills were offered with which were activated. These tests cover the
parity fix for the native `@decimalai.trace` / `start_trace` path:

  - TraceContext.set_routing_id() populates build_trace().routing_id
  - module-level decimalai.set_routing_id() works inside a traced function
  - calling it with no active trace is a safe no-op (does NOT raise)
  - build_prompt_fragment auto-stamps the ACTIVE generic trace with
    routing_id + offered (and delivered) names — including on cache hits
"""

from unittest.mock import MagicMock, patch

import pytest

import decimalai
from decimalai.generic import TraceContext


def _build_with_mock_config(ctx: TraceContext):
    """Build a RunTrace from a TraceContext with the config dependency mocked.

    Mirrors the pattern in test_skill_activation_pipeline.py — build_trace()
    reaches into _config._get_config() for the project name.
    """
    with patch("decimalai._config._get_config") as mock_get_config:
        mock_cfg = MagicMock()
        mock_cfg.project = "test"
        mock_get_config.return_value = mock_cfg
        return ctx.build_trace()


class TestTraceContextRoutingId:
    def test_default_routing_id_is_none(self):
        """A fresh trace has no routing_id."""
        ctx = TraceContext(agent_name="test-agent")
        assert ctx._routing_id is None
        trace = _build_with_mock_config(ctx)
        assert trace.routing_id is None

    def test_set_routing_id_stamps_built_trace(self):
        """set_routing_id() populates build_trace().routing_id."""
        ctx = TraceContext(agent_name="test-agent")
        ctx.set_routing_id("rt_123")
        trace = _build_with_mock_config(ctx)
        assert trace.routing_id == "rt_123"

    def test_set_routing_id_last_write_wins(self):
        """A later set_routing_id() overrides the earlier value."""
        ctx = TraceContext(agent_name="test-agent")
        ctx.set_routing_id("rt_old")
        ctx.set_routing_id("rt_new")
        trace = _build_with_mock_config(ctx)
        assert trace.routing_id == "rt_new"


class TestModuleLevelSetRoutingId:
    def test_set_routing_id_inside_traced_function(self):
        """decimalai.set_routing_id() inside a @trace fn stamps the built trace."""
        captured = {}

        @decimalai.trace(agent_name="routing-agent", auto_send=False)
        def run_agent(query):
            decimalai.set_routing_id("rt_123")
            # Grab the active context so we can assert on the trace it builds.
            from decimalai.generic import _get_current_trace
            captured["ctx"] = _get_current_trace()
            return "done"

        assert run_agent("hi") == "done"
        trace = _build_with_mock_config(captured["ctx"])
        assert trace.routing_id == "rt_123"

    def test_set_routing_id_no_active_trace_is_noop(self):
        """Calling decimalai.set_routing_id() with no active trace is a safe no-op."""
        from decimalai.generic import _get_current_trace

        # Sanity: nothing active outside a trace context.
        assert _get_current_trace() is None
        # Must NOT raise (unlike log_skill_activation, routing is optional).
        decimalai.set_routing_id("rt_orphan")

    def test_set_routing_id_is_exported(self):
        """set_routing_id is part of the public top-level API."""
        assert hasattr(decimalai, "set_routing_id")
        assert "set_routing_id" in decimalai.__all__


class TestBuildPromptFragmentAutoStamp:
    """build_prompt_fragment stamps the active generic trace itself,
    so the raw-loop quickstart needs neither decimalai.set_routing_id nor
    decimalai.log_skill_offered (both stay public)."""

    STUB_SKILLS = [{"name": "stub", "hash": "sha256:stub"}]  # skip disk discovery

    @pytest.fixture(autouse=True)
    def _fresh_router_ctx(self):
        from decimalai import skill_router as sr
        sr._last_offered_names_ctx.set(None)
        sr._last_delivered_names_ctx.set(None)
        sr._body_budget_ctx.set(None)
        yield
        sr._last_offered_names_ctx.set(None)
        sr._last_delivered_names_ctx.set(None)
        sr._body_budget_ctx.set(None)

    def _router(self, **kw):
        from decimalai.skill_router import SkillRouter
        return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", **kw)

    def test_fresh_call_stamps_routing_id_and_offered(self):
        router = self._router()
        route = {"prompt_fragment": "MENU", "routing_id": "rt_a5",
                 "skills": [{"name": "code-review"}]}
        with patch.object(router, "smart_route", return_value=route):
            with decimalai.start_trace(
                agent_name="raw-loop", auto_send=False, skills=self.STUB_SKILLS,
            ) as ctx:
                router.build_prompt_fragment(query="review my PR")

        trace = _build_with_mock_config(ctx)
        assert trace.routing_id == "rt_a5"
        assert trace.skills_offered_in_prompt == ["code-review"]
        # Menu row only — nothing delivered, nothing activated.
        assert trace.skills_delivered == []
        assert trace.active_skills == []

    def test_inject_body_stamps_delivered(self):
        router = self._router(inject_body=True)
        route = {"prompt_fragment": "MENU", "routing_id": "rt_a5",
                 "skills": [{"name": "code-review"}]}
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(router, "get_skill_body", return_value="BODY_K"):
            with decimalai.start_trace(
                agent_name="raw-loop", auto_send=False, skills=self.STUB_SKILLS,
            ) as ctx:
                router.build_prompt_fragment(query="review my PR")

        trace = _build_with_mock_config(ctx)
        assert trace.skills_delivered == ["code-review"]
        assert trace.skills_offered_in_prompt == ["code-review"]
        assert trace.active_skills == []  # delivery is not activation

    def test_cache_hit_stamps_a_new_trace(self):
        """A second turn served from the 30s fragment cache still stamps
        the (new) active trace — pre-fix the hit returned early and the
        second trace lost routing_id + offered names."""
        router = self._router()
        route = {"prompt_fragment": "MENU", "routing_id": "rt_hit",
                 "skills": [{"name": "code-review"}]}
        with patch.object(router, "smart_route", return_value=route) as mock_route:
            with decimalai.start_trace(
                agent_name="turn-1", auto_send=False, skills=self.STUB_SKILLS,
            ):
                router.build_prompt_fragment(query="same q")

            with decimalai.start_trace(
                agent_name="turn-2", auto_send=False, skills=self.STUB_SKILLS,
            ) as ctx2:
                router.build_prompt_fragment(query="same q")
            assert mock_route.call_count == 1  # second call was a cache hit

        trace2 = _build_with_mock_config(ctx2)
        assert trace2.routing_id == "rt_hit"
        assert trace2.skills_offered_in_prompt == ["code-review"]

    def test_no_active_trace_is_noop(self):
        from decimalai.generic import _get_current_trace
        assert _get_current_trace() is None

        router = self._router()
        route = {"prompt_fragment": "MENU", "routing_id": "rt_x",
                 "skills": [{"name": "a"}]}
        with patch.object(router, "smart_route", return_value=route):
            fragment, routing_id = router.build_prompt_fragment(query="q")
        assert (fragment, routing_id) == ("MENU", "rt_x")
