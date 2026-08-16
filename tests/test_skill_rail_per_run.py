"""The skills rail reaches the trace, and it reaches the RIGHT trace.

What is being covered
--------------------
The exporter had no path for skill-rail metadata at all: ``routing_id``,
``skills_offered_in_prompt``, ``skills_delivered`` and ``skills_loaded_by_agent``
were never set on a trace assembled by ``DecimalSpanExporter``, so the raw
Anthropic and Pydantic AI rails could not close the offered→activated join that
skill effectiveness is computed from.

Why the concurrency case is the centrepiece
-------------------------------------------
The obvious implementation — drain the router's ``consume_offered_names()`` /
``consume_routing_id()`` at trace-send — is *clear-on-read state on a
process-global singleton*. Under a concurrent fanout the first trace to send
takes every lane's names and the rest get ``[]``, and a trace can report a
routing id minted for somebody else's query. That bug has already happened in
this codebase, more than once, on more than one rail.

The conformance matrix cannot catch it: every lane there is offered the SAME two
skills, so an implementation that hands every trace the union of everything
still satisfies a set-equality assertion. Only a test that gives two in-flight
runs DIFFERENT skills can tell "each run got its own" apart from "everyone got
everything". That is what ``TestTwoRunsInFlight`` does.

These tests drive the real OpenTelemetry SDK for the same reason
``test_provider_run_scope.py`` does: the seam under test is OTel *context* —
which span is current when the router is called, and which trace_id the span
lands in — and a mocked tracer has no context to get wrong.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from decimalai import providers
from decimalai.otel import (
    _active_agent_name,
    _reset_skill_rails,
    _skill_rails,
    current_run_key,
    record_skill_rail,
)

pytest.importorskip("opentelemetry.sdk.trace")


# ── plumbing (mirrors tests/test_provider_run_scope.py) ──────────────────────


@pytest.fixture(autouse=True)
def _sdk_enabled():
    """A configured, mocked SDK — the exporter refuses to assemble without one."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    prev_config, prev_client = cfg._config, cfg._client
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }
    yield cfg
    cfg._config, cfg._client = prev_config, prev_client


@pytest.fixture(autouse=True)
def _clean_context():
    token = _active_agent_name.set(None)
    yield
    _active_agent_name.reset(token)


@pytest.fixture(autouse=True)
def _clean_module_state():
    prev = (providers._pipeline_provider, providers._last_provider,
            set(providers._instrumented))
    _reset_skill_rails()
    yield
    _reset_skill_rails()
    (providers._pipeline_provider, providers._last_provider) = prev[0], prev[1]
    providers._instrumented.clear()
    providers._instrumented.update(prev[2])


class _Pipeline:
    """A caller-owned TracerProvider wired exactly as ``_ensure_pipeline`` wires one."""

    def __init__(self, agent_name: str | None = None) -> None:
        from opentelemetry.sdk.trace import TracerProvider

        self.provider = TracerProvider()
        providers._ensure_pipeline(agent_name, self.provider)
        self.tracer = self.provider.get_tracer("test")

    def traces(self, cfg: Any) -> List[Any]:
        from decimalai._config import _sender

        _sender.flush()
        return [c[0][0] for c in cfg._client.ingest_trace.call_args_list]

    def llm_span(self, name: str = "ChatCompletion") -> None:
        """One span shaped like what a provider instrumentor emits."""
        with self.tracer.start_as_current_span(name, attributes={
            "gen_ai.request.model": "stub-model-1",
            "gen_ai.system": "openai",
            "gen_ai.usage.input_tokens": 11,
            "gen_ai.usage.output_tokens": 3,
        }):
            pass


def _by_agent(traces: List[Any]) -> Dict[str, Any]:
    return {t.agent_name: t for t in traces}


# ── the rail reaches the trace at all ────────────────────────────────────────


class TestRailReachesTheTrace:
    def test_routing_id_and_offered_names_land_on_the_run(self, _sdk_enabled):
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            record_skill_rail(
                routing_id="rt_aaa",
                offered=["alpha", "beta"],
                prompt_text="menu: alpha, beta",
            )
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.routing_id == "rt_aaa"
        assert trace.skills_offered_in_prompt == ["alpha", "beta"]

    def test_loaded_implies_delivered_but_not_offered_in_prompt(self, _sdk_enabled):
        """A loaded body was DELIVERED. It is not evidence it was in the prompt.

        `skills_offered_in_prompt` is a claim about what the model was shown, and
        `record_skill_rail` enforces it by dropping any offered/delivered name
        absent from the rendered fragment. `loaded` arrives from a tool call and
        never passes that filter — so folding it into `offered` re-imported
        through the back door exactly the unverified claim the filter rejects,
        and did so ONLY in the case where there was no evidence: a real menu
        injection records `offered` itself, which makes the union redundant
        everywhere except where it is wrong.

        Note this call passes no prompt_text and no offered names — all we truly
        know is that a body was served.
        """
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            record_skill_rail(routing_id="rt_ladder", loaded=["gamma"])
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.skills_loaded_by_agent == ["gamma"]
        assert trace.skills_delivered == ["gamma"]
        assert trace.skills_offered_in_prompt == [], (
            "a tool-call load was reported as having appeared in the prompt, "
            "which nothing verified"
        )

    def test_a_name_offered_in_the_prompt_and_then_loaded_appears_in_both(self, _sdk_enabled):
        """The other direction, so the rule above cannot hide a real offer.

        This is the shape a real run has: the router injects a menu (recording
        what it offered, checked against the fragment it rendered), and the agent
        then pulls one of those bodies.
        """
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            record_skill_rail(
                routing_id="rt_real",
                offered=["gamma"],
                prompt_text="Available skills:\n- gamma: does a thing",
            )
            record_skill_rail(routing_id="rt_real", loaded=["gamma"])
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.skills_offered_in_prompt == ["gamma"]
        assert trace.skills_delivered == ["gamma"]
        assert trace.skills_loaded_by_agent == ["gamma"]

    def test_first_routing_decision_of_a_run_wins(self, _sdk_enabled):
        """A multi-call turn routes more than once. The run is attributed to the
        decision it started with, not to whichever call happened to be last."""
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            record_skill_rail(routing_id="rt_first", offered=["alpha"])
            record_skill_rail(routing_id="rt_second", offered=["beta"])
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.routing_id == "rt_first"
        # Names still accumulate across the turn — both were genuinely offered.
        assert trace.skills_offered_in_prompt == ["alpha", "beta"]


# ── the centrepiece: two runs in flight ──────────────────────────────────────


class TestTwoRunsInFlight:
    def test_interleaved_runs_keep_their_own_rails(self, _sdk_enabled):
        """Two runs, DIFFERENT offered sets, interleaved on ONE thread.

        Interleaved on purpose: run A routes, then run B routes, then A's span
        closes, then B's. Any implementation that drains a shared slot at
        trace-send hands A's trace B's names (or nothing at all). Only a rail
        keyed by the run itself survives this.
        """
        pipe = _Pipeline()

        run_a = providers.agent_run("agent-a", tracer_provider=pipe.provider)
        run_a.__enter__()
        record_skill_rail(
            routing_id="rt_a", offered=["alpha"], prompt_text="menu: alpha",
        )
        pipe.llm_span()
        run_a.__exit__(None, None, None)

        run_b = providers.agent_run("agent-b", tracer_provider=pipe.provider)
        run_b.__enter__()
        record_skill_rail(
            routing_id="rt_b", offered=["beta"], prompt_text="menu: beta",
        )
        pipe.llm_span()
        run_b.__exit__(None, None, None)

        traces = _by_agent(pipe.traces(_sdk_enabled))
        assert set(traces) == {"agent-a", "agent-b"}
        assert traces["agent-a"].routing_id == "rt_a"
        assert traces["agent-a"].skills_offered_in_prompt == ["alpha"]
        assert traces["agent-b"].routing_id == "rt_b"
        assert traces["agent-b"].skills_offered_in_prompt == ["beta"]
        # Neither inherited the other's — stated explicitly, because a union
        # implementation passes every set-equality check the matrix makes.
        assert "beta" not in traces["agent-a"].skills_offered_in_prompt
        assert "alpha" not in traces["agent-b"].skills_offered_in_prompt

    def test_eight_concurrent_threads_each_carry_only_their_own(self, _sdk_enabled):
        """The fanout shape the conformance matrix runs, but with each lane
        offered its OWN skill so a union cannot masquerade as a pass."""
        pipe = _Pipeline()
        lanes = [f"lane{i}" for i in range(8)]
        start = threading.Barrier(len(lanes))

        def _lane(i: int) -> None:
            name = lanes[i]
            with providers.agent_run(name, tracer_provider=pipe.provider):
                # Every lane routes before any lane finishes — the window in
                # which a shared slot gets overwritten.
                start.wait(timeout=10)
                record_skill_rail(
                    routing_id=f"rt_{name}",
                    offered=[f"skill-{name}"],
                    prompt_text=f"menu: skill-{name}",
                )
                pipe.llm_span()

        threads = [threading.Thread(target=_lane, args=(i,)) for i in range(len(lanes))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        traces = _by_agent(pipe.traces(_sdk_enabled))
        assert set(traces) == set(lanes)
        for name in lanes:
            trace = traces[name]
            assert trace.routing_id == f"rt_{name}", (
                f"{name} reports {trace.routing_id!r} — a routing decision crossed runs"
            )
            assert trace.skills_offered_in_prompt == [f"skill-{name}"], (
                f"{name} carries {trace.skills_offered_in_prompt} — not its own"
            )
        # And every run got a DISTINCT decision, which is what stops one routing
        # decision being double-counted across eight runs.
        ids = [t.routing_id for t in traces.values()]
        assert len(set(ids)) == len(ids)

    def test_a_rail_is_consumed_once(self, _sdk_enabled):
        """Popped, never peeked: a second trace under the same run must not be
        handed the routing decision the first one already claimed."""
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            key = current_run_key()
            record_skill_rail(routing_id="rt_once", offered=["alpha"])
            pipe.llm_span()

        assert key not in _skill_rails


# ── absence, never a guess ───────────────────────────────────────────────────


class TestNoRunToAttributeTo:
    def test_recording_outside_a_run_is_dropped(self, _sdk_enabled):
        """No ``agent_run`` → no owner. The metadata is discarded rather than
        parked in a shared slot for the next trace to pick up."""
        assert current_run_key() is None
        assert record_skill_rail(routing_id="rt_orphan", offered=["alpha"]) is False
        assert not _skill_rails

    def test_the_trace_ships_with_the_fields_empty(self, _sdk_enabled):
        """The drop path, proved end to end: an un-scoped routing decision must
        not reappear on the next trace that happens to be assembled."""
        pipe = _Pipeline(agent_name="agent-a")

        record_skill_rail(routing_id="rt_orphan", offered=["alpha"])

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.routing_id is None
        assert trace.skills_offered_in_prompt == []
        assert trace.skills_delivered == []
        assert trace.skills_loaded_by_agent == []


# ── never claim a name the prompt does not carry ─────────────────────────────


class TestPromptTextFilter:
    def test_a_name_missing_from_the_fragment_is_dropped_and_logged(
        self, _sdk_enabled, caplog,
    ):
        """The router derives its offered list and its prompt fragment from two
        different keys of one payload, and nothing cross-checks them. Copying
        the claim through unchecked would put a name on the trace that the model
        was never shown."""
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            with caplog.at_level(logging.WARNING, logger="decimalai.otel"):
                record_skill_rail(
                    routing_id="rt_partial",
                    offered=["alpha", "ghost"],
                    prompt_text="menu row for alpha only",
                )
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.skills_offered_in_prompt == ["alpha"]
        assert "ghost" in caplog.text
        assert "does not contain it" in caplog.text

    def test_the_filter_cannot_manufacture_a_pass(self, _sdk_enabled):
        """Filtering removes a false claim; it never invents a true one. With no
        name in the fragment the rail records nothing at all."""
        pipe = _Pipeline(agent_name="agent-a")

        with providers.agent_run("agent-a", tracer_provider=pipe.provider):
            record_skill_rail(
                routing_id="rt_none", offered=["ghost"], prompt_text="no names here",
            )
            pipe.llm_span()

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.skills_offered_in_prompt == []
        # The routing decision still happened and is still reported — the run was
        # routed, it just surfaced nothing the fragment carried.
        assert trace.routing_id == "rt_none"


# ── bounded ──────────────────────────────────────────────────────────────────


class TestBounded:
    def test_rails_for_runs_that_never_finish_cannot_pile_up(self, _sdk_enabled):
        """A run whose root span never reaches the exporter leaves its rail
        behind. Bounded LRU, oldest evicted — the same discipline as the
        pending-span buffer."""
        from decimalai.otel import _SKILL_RAIL_MAX

        pipe = _Pipeline(agent_name="agent-a")
        for i in range(_SKILL_RAIL_MAX + 50):
            span = pipe.tracer.start_span(f"orphan-{i}")
            from opentelemetry import trace as trace_api
            with trace_api.use_span(span, end_on_exit=False):
                record_skill_rail(routing_id=f"rt_{i}")
        assert len(_skill_rails) <= _SKILL_RAIL_MAX


# ── the router's per-run loaded-names rail ───────────────────────────────────


class TestScopedLoadedNames:
    """``load_skill``'s docstring has always promised the loaded-names rail is
    kept per scope. Until recently ``scope`` reached only the turn budget."""

    def _router(self, **kwargs: Any) -> Any:
        from decimalai.skill_router import SkillRouter

        router = SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000",
                             **kwargs)
        router.get_skill_body = MagicMock(return_value="the body")  # type: ignore[method-assign]
        return router

    def test_two_runs_do_not_see_each_others_loads(self):
        router = self._router()

        router.load_skill("alpha", scope="run-a")
        router.load_skill("beta", scope="run-b")

        assert router.consume_loaded_names(scope="run-a") == ["alpha"]
        assert router.consume_loaded_names(scope="run-b") == ["beta"]

    def test_a_scoped_drain_is_consumed_once(self):
        router = self._router()
        router.load_skill("alpha", scope="run-a")

        assert router.consume_loaded_names(scope="run-a") == ["alpha"]
        assert router.consume_loaded_names(scope="run-a") == []

    def test_an_unscoped_drain_still_sees_the_shared_slot(self):
        """The legacy path other adapters still use is untouched."""
        router = self._router()
        router.load_skill("alpha", scope="run-a")

        assert router.consume_loaded_names() == ["alpha"]

    def test_a_refused_load_is_not_recorded(self):
        """``load_skill`` never raises — a budget refusal comes back as an
        ordinary string. Recording it would claim a body the model never got."""
        router = self._router(max_loaded_bodies=1)

        first = router.load_skill("alpha", scope="run-a")
        second = router.load_skill("beta", scope="run-a")

        assert first.startswith("## Skill: alpha")
        assert not second.startswith("## Skill: beta")
        assert router.consume_loaded_names(scope="run-a") == ["alpha"]

    def test_a_missing_skill_is_not_recorded(self):
        router = self._router()
        router.get_skill_body = MagicMock(return_value=None)  # type: ignore[method-assign]

        out = router.load_skill("nope", scope="run-a")

        assert "no skill named" in out
        assert router.consume_loaded_names(scope="run-a") == []


# ── the fragment cache no longer pools concurrent runs ───────────────────────


class TestFragmentCacheIsPerRun:
    """Two runs must not share one routing decision.

    These tests mock `_request` — the HTTP layer — and NOT `get_menu`. That
    distinction is the whole point of the class. An earlier version stubbed
    `router.get_menu` directly, which passed against the broken code: it
    replaced the very layer that collides. `get_menu` keeps its OWN cache
    (`_menu_cache`), the cached entry carries the platform's `routing_id`, and
    on the full-menu path (`query=None`) that cache is what two concurrent runs
    used to share. Mocking above it tests the fix that was not made.
    """

    def _router(self, responses: Any) -> Any:
        """A router whose only stub is the transport, so every cache is real."""
        from decimalai.skill_router import SkillRouter

        router = SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")
        router._request = MagicMock(side_effect=responses)  # type: ignore[method-assign]
        return router

    def test_two_scopes_get_two_routing_decisions(self):
        """Full-menu adapters route with ``query=None`` under one agent name, so
        before ``scope`` joined the cache key every concurrent run of that agent
        collided on ONE slot — and shared the routing_id parked in it."""
        minted = iter([
            {"prompt_fragment": "menu", "routing_id": "rt_a", "skills": []},
            {"prompt_fragment": "menu", "routing_id": "rt_b", "skills": []},
        ])
        router = self._router(lambda *a, **kw: next(minted))

        _, first = router.build_prompt_fragment(query=None, scope="run-a")
        _, second = router.build_prompt_fragment(query=None, scope="run-b")

        assert first == "rt_a"
        assert second == "rt_b", (
            "the second run was handed the first run's routing_id — two runs "
            "claiming one routing decision undercounts offers in the "
            "effectiveness join"
        )

    def test_one_run_still_reuses_its_own_slot(self):
        """The cache still does its job within a run — this is a partition, not
        a disable, so routing-call volume per run is unchanged."""
        router = self._router(
            lambda *a, **kw: {"prompt_fragment": "menu", "routing_id": "rt_a", "skills": []}
        )

        router.build_prompt_fragment(query=None, scope="run-a")
        router.build_prompt_fragment(query=None, scope="run-a")

        assert router._request.call_count == 1, (
            f"a second call inside ONE run re-fetched: {router._request.call_count}"
        )

    def test_unscoped_callers_keep_the_old_single_slot(self):
        """langchain, openai-agents and every unscoped caller pass no scope, so
        their key and their call volume are exactly what they were."""
        router = self._router(
            lambda *a, **kw: {"prompt_fragment": "menu", "routing_id": "rt_a", "skills": []}
        )

        router.build_prompt_fragment(query=None)
        router.build_prompt_fragment(query=None)

        assert router._request.call_count == 1


# ── the adapters record only what they injected ──────────────────────────────


class TestAnthropicClaimsOnlyWhatItInjected:
    def test_an_empty_fragment_claims_no_routing_decision(self, _sdk_enabled):
        """The guard that used to be in the wrong order: ``_set_routing_id`` sat
        ABOVE the empty-fragment check, so a call that injected nothing still
        reported that the model had been routed."""
        import decimalai.anthropic as da

        pipe = _Pipeline(agent_name="agent-a")
        router = MagicMock()
        router.build_prompt_fragment.return_value = ("", "rt_empty")
        prev, da._skill_router_singleton = da._skill_router_singleton, router
        try:
            with providers.agent_run("agent-a", tracer_provider=pipe.provider):
                out = da.skill_system("base prompt", query="hello")
                assert out == "base prompt"
                assert not _skill_rails
                pipe.llm_span()
        finally:
            da._skill_router_singleton = prev

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.routing_id is None
        assert trace.skills_offered_in_prompt == []

    def test_an_injected_fragment_is_recorded_against_this_run(self, _sdk_enabled):
        import decimalai.anthropic as da
        from decimalai import skill_router as sr

        pipe = _Pipeline(agent_name="agent-a")
        router = MagicMock()

        def _build(**kwargs: Any) -> Any:
            # What the real router does one statement before the adapter drains.
            sr._last_offered_names_ctx.set(["alpha", "beta"])
            return "Available skills:\n- alpha\n- beta", "rt_live"

        router.build_prompt_fragment.side_effect = _build
        prev, da._skill_router_singleton = da._skill_router_singleton, router
        try:
            with providers.agent_run("agent-a", tracer_provider=pipe.provider):
                out = da.skill_system("base prompt", query="hello")
                assert "alpha" in out and "base prompt" in out
                pipe.llm_span()
        finally:
            da._skill_router_singleton = prev
            sr._last_offered_names_ctx.set(None)

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.routing_id == "rt_live"
        assert trace.skills_offered_in_prompt == ["alpha", "beta"]

    def test_without_agent_run_the_skills_still_reach_the_model(self, _sdk_enabled):
        """A raw-SDK user who skips ``agent_run()`` still gets skills injected —
        the rail degrades to silence, it does not break the call.

        The docs already require ``agent_run()`` on the raw provider rails, and
        frame it as being about trace SHAPE. It is now also what makes the
        skills rail attributable: without it there is no run to file the routing
        decision under, so the fields stay empty rather than being guessed onto
        somebody else's trace.
        """
        import decimalai.anthropic as da
        from decimalai import skill_router as sr

        pipe = _Pipeline(agent_name="agent-a")
        router = MagicMock()

        def _build(**kwargs: Any) -> Any:
            sr._last_offered_names_ctx.set(["alpha"])
            return "Available skills:\n- alpha", "rt_unscoped"

        router.build_prompt_fragment.side_effect = _build
        prev, da._skill_router_singleton = da._skill_router_singleton, router
        try:
            # No agent_run anywhere.
            out = da.skill_system("base prompt", query="hello")
            assert "alpha" in out and "base prompt" in out
            assert not _skill_rails
            with providers.agent_run("agent-a", tracer_provider=pipe.provider):
                pipe.llm_span()
        finally:
            da._skill_router_singleton = prev
            sr._last_offered_names_ctx.set(None)

        (trace,) = pipe.traces(_sdk_enabled)
        assert trace.routing_id is None
        assert trace.skills_offered_in_prompt == []

    def test_the_run_scope_is_threaded_into_the_router(self, _sdk_enabled):
        """The adapter must name its run to the router, or two concurrent runs
        share a fragment-cache slot and therefore a routing decision."""
        import decimalai.anthropic as da

        pipe = _Pipeline(agent_name="agent-a")
        router = MagicMock()
        router.build_prompt_fragment.return_value = ("frag", "rt_x")
        prev, da._skill_router_singleton = da._skill_router_singleton, router
        try:
            with providers.agent_run("agent-a", tracer_provider=pipe.provider):
                expected = f"{current_run_key():032x}"
                da.skill_system(None, query="hello")
        finally:
            da._skill_router_singleton = prev

        assert router.build_prompt_fragment.call_args.kwargs["scope"] == expected
