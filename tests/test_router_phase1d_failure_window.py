"""Phase 1d: a failed route held for an UNSCOPED caller is not shared for 30 s.

``build_prompt_fragment`` caches a slow failed route (a timeout, an admission
wait) so the other model calls of the turn do not each pay it again. The cache
key names the run only when the caller passes ``scope=``. LangChain never does,
and neither does a hand-written loop, so for them one slot per (query, agent)
serves the whole process: one timeout during a brownout left every other
conversation asking the same thing (a chat UI's suggested-prompt button sends
identical first messages) without skills for the full 30 s TTL.

The router cannot tell "the next model call of this turn" from "another
conversation" for an unscoped caller: both arrive with the same query and
agent, and LangChain's ambient run id names each model call, not the turn
(``test_router_phase1d_langchain_failure_window.py`` runs real graphs). So the
narrowest fix inside the router is time. An unscoped failure is reused for
``_UNSCOPED_FAILURE_TTL_S`` (5 s), which covers the model call that follows the
failed route, and not for the 30 s a success is kept. A caller that names its
run keeps the failure for the turn, and no other run is ever served it.

Both properties are tested: the turn does not retry inside the window, and a
conversation that starts after it routes again.
"""

from __future__ import annotations

from typing import Any, List

import httpx
import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter

BASE = "http://platform.test"
QUERY = "refund an opened box"
ROUTE = {
    "strategy": "smart_routing",
    "skills": [{"name": "refund-policy", "description": "Refund rules"}],
    "prompt_fragment": "## Recommended Skills\n| refund-policy | Refund rules |",
    "routing_id": "rt_" + "a" * 24,
}


class _Clock:
    """Stands in for ``time`` inside ``skill_router`` (both the monotonic clock
    the elapsed time is measured on and the wall clock the cache expires on)."""

    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return 1_700_000_000.0 + self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name):
        import time

        return getattr(time, name)


class _Platform:
    """/route answers each outcome in ``script`` once, then 200 with ``ROUTE``.

    "timeout" raises the real ``httpx.ReadTimeout`` after ``timeout_cost_s`` of
    fake time, as the 5 s read budget would.
    """

    def __init__(self, clock: _Clock, script: List[Any], timeout_cost_s: float = 5.0) -> None:
        self.clock, self.script, self.timeout_cost_s = clock, list(script), timeout_cost_s
        self.routes = 0

    def __call__(self, method, url, **kw):
        assert url.endswith("/api/v1/skills/route"), url
        self.routes += 1
        if self.script:
            outcome = self.script.pop(0)
            if outcome == "timeout":
                self.clock.now += self.timeout_cost_s
                raise httpx.ReadTimeout("slow")
            return httpx.Response(outcome, json={"detail": f"forced {outcome}"})
        return httpx.Response(200, json=ROUTE)


@pytest.fixture(autouse=True)
def _no_menu_fallback(monkeypatch):
    """These tests pin how long a FAILED route is held. Since 2026-10-20 a slow
    route is first answered by the agent's menu (test_route_menu_fallback.py),
    which this fake platform does not serve; off here, so the rule is tested alone."""
    monkeypatch.setattr(sr, "_MENU_FALLBACK_REASONS", frozenset())


@pytest.fixture(autouse=True)
def clock(monkeypatch) -> _Clock:
    monkeypatch.setattr(sr, "_hot_path_breakers", sr._HotPathBreakers())
    fake = _Clock()
    monkeypatch.setattr(sr, "time", fake)
    return fake


def _install(monkeypatch, clock, script, **kw) -> _Platform:
    platform = _Platform(clock, script, **kw)
    monkeypatch.setattr(httpx, "request", platform)
    return platform


def _router(**kw) -> SkillRouter:
    return SkillRouter(api_key="dai_sk_test", base_url=BASE, **kw)


def _skills(result) -> bool:
    return "refund-policy" in result[0]


class TestAnUnscopedFailureIsHeldForTheTurnOnly:
    def test_another_conversation_a_few_seconds_later_routes_again(self, monkeypatch, clock):
        """The defect: this conversation used to be served the first one's
        failure until the 30 s TTL ran out."""
        platform = _install(monkeypatch, clock, ["timeout"])
        router = _router()
        assert router.build_prompt_fragment(QUERY, agent_name="support") == ("", None)

        clock.now += 6.0  # another user presses the same suggested prompt
        second = router.build_prompt_fragment(QUERY, agent_name="support")

        assert _skills(second) and second[1] == ROUTE["routing_id"]
        assert platform.routes == 2

    @pytest.mark.parametrize("seconds_later", [0.5, 2.0, 4.9])
    def test_the_rest_of_the_turn_does_not_retry_inside_the_window(
        self, monkeypatch, clock, seconds_later,
    ):
        """The property the cache exists for: the next model call of the turn,
        which comes right after the failed route, does not pay the timeout again."""
        platform = _install(monkeypatch, clock, ["timeout"] * 5)
        router = _router()
        router.build_prompt_fragment(QUERY, agent_name="support")

        clock.now += seconds_later
        assert router.build_prompt_fragment(QUERY, agent_name="support") == ("", None)
        assert platform.routes == 1

    def test_a_long_turn_retries_at_most_once_per_window(self, monkeypatch, clock):
        """Ten model calls 2 s apart against an endpoint that keeps timing out.
        Each attempt costs 5 s and buys a 5 s window, so the turn makes about one
        attempt per 10 s (and the breaker's half-open probe is one of them), not
        one per model call."""
        platform = _install(monkeypatch, clock, ["timeout"] * 50)
        router = _router()
        for _ in range(10):
            assert router.build_prompt_fragment(QUERY, agent_name="support") == ("", None)
            clock.now += 2.0
        assert platform.routes <= 4, f"{platform.routes} routes for 10 model calls"

    def test_the_window_does_not_grow_with_a_raised_read_budget(self, monkeypatch, clock):
        """The window is about the gap between a turn's model calls, which a
        longer timeout does not change. Scaling it with the budget would give a
        30 s budget back its 30 s of shared failure."""
        monkeypatch.setenv(sr._HOT_PATH_READ_ENV, "20")
        platform = _install(monkeypatch, clock, ["timeout"], timeout_cost_s=20.0)
        router = _router()
        router.build_prompt_fragment(QUERY, agent_name="support")
        clock.now += 6.0
        assert _skills(router.build_prompt_fragment(QUERY, agent_name="support"))
        assert platform.routes == 2

    def test_never_longer_than_the_routers_own_cache_ttl(self, monkeypatch, clock):
        platform = _install(monkeypatch, clock, ["timeout"])
        router = _router(fragment_cache_ttl=2.0)
        router.build_prompt_fragment(QUERY, agent_name="support")
        clock.now += 2.5
        assert _skills(router.build_prompt_fragment(QUERY, agent_name="support"))
        assert platform.routes == 2

    def test_a_slow_admission_failure_is_held_by_the_same_window(self, monkeypatch, clock):
        """A 429 whose Retry-After waits ran out is the other slow failure."""

        class _Admission(_Platform):
            def __call__(self, method, url, **kw):
                self.routes += 1
                if self.script:
                    self.script.pop(0)
                    return httpx.Response(429, json={"detail": "busy"},
                                          headers={"Retry-After": "2"})
                return httpx.Response(200, json=ROUTE)

        platform = _Admission(clock, [429] * 3)
        monkeypatch.setattr(httpx, "request", platform)
        router = _router()
        assert router.build_prompt_fragment(QUERY, agent_name="support") == ("", None)
        attempts = platform.routes
        clock.now += 1.0
        router.build_prompt_fragment(QUERY, agent_name="support")
        assert platform.routes == attempts, "the turn paid the admission wait twice"
        clock.now += 5.0
        assert _skills(router.build_prompt_fragment(QUERY, agent_name="support"))

    def test_a_success_is_still_kept_for_the_full_ttl(self, monkeypatch, clock):
        """Only the failure's window changed. Sharing a success across unscoped
        runs is today's behaviour and routing-call volume."""
        platform = _install(monkeypatch, clock, [])
        router = _router()
        router.build_prompt_fragment(QUERY, agent_name="support")
        clock.now += 25.0
        assert _skills(router.build_prompt_fragment(QUERY, agent_name="support"))
        assert platform.routes == 1


class TestAScopedRunKeepsItsFailureForTheTurn:
    """A caller that names its run (pydantic_ai, ADK, CrewAI, anthropic, the
    OpenAI Agents runner) cannot serve its failure to anyone else, so the turn
    keeps it for the cache TTL and no other run waits for it."""

    def test_the_run_does_not_retry_after_the_unscoped_window(self, monkeypatch, clock):
        platform = _install(monkeypatch, clock, ["timeout"] * 5)
        router = _router()
        router.build_prompt_fragment(QUERY, scope="run-1")
        clock.now += 20.0
        assert router.build_prompt_fragment(QUERY, scope="run-1") == ("", None)
        assert platform.routes == 1

    def test_another_run_routes_at_once(self, monkeypatch, clock):
        platform = _install(monkeypatch, clock, ["timeout"])
        router = _router()
        router.build_prompt_fragment(QUERY, scope="run-1")
        assert _skills(router.build_prompt_fragment(QUERY, scope="run-2"))
        assert platform.routes == 2

    def test_an_unscoped_caller_is_not_served_a_scoped_runs_failure(self, monkeypatch, clock):
        platform = _install(monkeypatch, clock, ["timeout"])
        router = _router()
        router.build_prompt_fragment(QUERY, scope="run-1")
        assert _skills(router.build_prompt_fragment(QUERY))
        assert platform.routes == 2
