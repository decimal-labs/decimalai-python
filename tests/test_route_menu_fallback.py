"""A failed route serves the agent's menu, unranked, instead of no skills (2026-10-20).

On 2026-10-18 the OpenAI tutorial's agent answered without its procedures after a
route call timed out, and nothing but the trace said so. The agent's own menu is
the cheap endpoint (0.08 s on prod while routes took 1.4-7.8 s), so after a slow
or erroring route one request for it serves the agent's skills.

What must hold, each pinned below through the real `_request`:
  * only a slow route (timeout, 504) or one that errored (500) earns the extra
    request; a platform shedding load (429, 503), an unreachable one, or an open
    breaker gets no second request;
  * the fallback has its own 2 s read budget and never waits out a Retry-After;
  * a menu that fails too leaves today's empty menu, cached as a slow failure;
  * after a fallback, routes from that host and key pause for 30 s, then resume.
"""
from __future__ import annotations

import httpx
import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import (
    SkillRouter,
    _HotPathBreakers,
    consume_last_offered_names,
    routing_status,
)

BASE = "http://127.0.0.1:8001"
ROUTING_ID = "rt_" + "a1" * 12
MENU = {
    "strategy": "full_menu",
    "skills": [{"name": "refund-policy", "description": "Refund rules", "version": 2}],
    "prompt_fragment": "## Available Skills\n\n| Skill | Description |\n|-------|-------------|\n"
                       "| refund-policy | Refund rules |\n",
    "stable_menu": "## Available Skills\n\n| Skill | Description |\n|-------|-------------|\n"
                   "| refund-policy | Refund rules |\n",
    "menu_instruction": "When a skill above applies, follow its instructions.",
    "routing_hint": "",
    "stable_menu_skills": ["refund-policy"],
    "skill_count": 1,
    "rows_total": 1,
    "truncated": False,
    "query_ranked": False,
    "routing_id": ROUTING_ID,
}


@pytest.fixture(autouse=True)
def _fresh_breakers(monkeypatch):
    monkeypatch.setattr(sr, "_hot_path_breakers", _HotPathBreakers())


class _Platform:
    """Answers route and menu calls as told, and records every request."""

    def __init__(self, route, menu=None):
        self.route, self.menu, self.calls = route, menu, []

    def __call__(self, method, url, *, params=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url.removeprefix(BASE), timeout, params))
        answer = self.route if url.endswith("/api/v1/skills/route") else self.menu
        if isinstance(answer, Exception):
            raise answer
        return answer

    def paths(self):
        return [(method, path) for method, path, _, _ in self.calls]


def _ok(body):
    return httpx.Response(200, json=body)


def _router():
    return SkillRouter(api_key="dai_sk_fixture", base_url=BASE, agent_name="support")


ROUTE = ("POST", "/api/v1/skills/route")
MENU_CALL = ("GET", "/api/v1/skills/menu")


@pytest.mark.parametrize("route_failure", [
    httpx.ReadTimeout("read timed out"),
    httpx.Response(500, json={"detail": "tsquery stack too small"}),
    httpx.Response(504, json={"detail": "gateway timeout"}),
], ids=["timeout", "http_500", "http_504"])
def test_a_slow_or_erroring_route_serves_the_agents_menu(monkeypatch, route_failure):
    platform = _Platform(route_failure, _ok(MENU))
    monkeypatch.setattr(sr.httpx, "request", platform)

    fragment, routing_id = _router().build_prompt_fragment("how do refunds work?")

    assert platform.paths() == [ROUTE, MENU_CALL]
    assert "refund-policy" in fragment
    assert routing_id == ROUTING_ID, "the run names the menu's own decision"
    assert consume_last_offered_names() == ["refund-policy"]
    _, _, timeout, params = platform.calls[1]
    assert timeout.read == sr._MENU_FALLBACK_READ_S
    assert params == {"agent_name": "support"}
    status = routing_status()
    assert status.menu_fallbacks == 1
    assert status.healthy, "the turn had its skills"


def test_the_load_agent_call_gets_the_menu_as_its_prefix(monkeypatch):
    """`load_agent()` routes through `build_prompt_parts`; its first run used to
    record "skills unavailable" here."""
    monkeypatch.setattr(sr.httpx, "request", _Platform(httpx.ReadTimeout("slow"), _ok(MENU)))

    prefix, _tail, routing_id = _router().build_prompt_parts(
        "I want my money back", agent_name="support", scope="run-1",
    )

    assert "refund-policy" in prefix and routing_id == ROUTING_ID


@pytest.mark.parametrize("route_failure", [
    httpx.Response(429, json={"detail": "no available instance"}),
    httpx.Response(503, json={"detail": "service unavailable"}),
    httpx.ConnectError("connection refused"),
], ids=["http_429", "http_503", "transport"])
def test_a_shedding_or_unreachable_platform_gets_no_second_request(monkeypatch, route_failure):
    platform = _Platform(route_failure, _ok(MENU))
    monkeypatch.setattr(sr.httpx, "request", platform)

    fragment, routing_id = _router().build_prompt_fragment("how do refunds work?")

    assert platform.paths() == [ROUTE]
    assert (fragment, routing_id) == ("", None)
    assert routing_status().menu_fallbacks == 0


def test_an_open_breaker_skips_the_route_and_the_menu(monkeypatch):
    platform = _Platform(httpx.ConnectError("connection refused"), _ok(MENU))
    monkeypatch.setattr(sr.httpx, "request", platform)
    router = _router()
    for i in range(3):  # the breaker's threshold
        router.build_prompt_fragment(f"question {i}")
    assert routing_status().breaker_open
    platform.calls.clear()

    assert router.build_prompt_fragment("one more question") == ("", None)
    assert platform.calls == []


def test_a_menu_that_fails_too_leaves_the_empty_menu_cached_as_slow(monkeypatch):
    platform = _Platform(httpx.ReadTimeout("slow route"), httpx.ReadTimeout("slow menu"))
    monkeypatch.setattr(sr.httpx, "request", platform)
    router = _router()

    assert router.build_prompt_fragment("how do refunds work?") == ("", None)
    assert platform.paths() == [ROUTE, MENU_CALL]
    assert routing_status().menu_fallbacks == 0

    # A slow failure is kept for the turn's next model call, as before.
    platform.calls.clear()
    assert router.build_prompt_fragment("how do refunds work?") == ("", None)
    assert platform.calls == []


def test_the_fallback_never_waits_out_a_retry_after(monkeypatch):
    platform = _Platform(
        httpx.ReadTimeout("slow route"),
        httpx.Response(429, headers={"Retry-After": "1"}, json={"detail": "busy"}),
    )
    monkeypatch.setattr(sr.httpx, "request", platform)

    assert _router().build_prompt_fragment("how do refunds work?") == ("", None)
    assert platform.paths() == [ROUTE, MENU_CALL], "one menu request, no admission retry"


def test_after_a_fallback_the_route_pauses_and_turns_go_straight_to_the_menu(monkeypatch):
    platform = _Platform(httpx.ReadTimeout("slow route"), _ok(MENU))
    monkeypatch.setattr(sr.httpx, "request", platform)
    router = _router()
    router.build_prompt_fragment("how do refunds work?")
    platform.calls.clear()

    fragment, routing_id = router.build_prompt_fragment("a different question")

    assert platform.paths() == [MENU_CALL], "no second route timeout to pay"
    assert "refund-policy" in fragment and routing_id == ROUTING_ID
    assert routing_status().menu_fallbacks == 2
    # Another key is not paused: the pause is the failing key's.
    other = SkillRouter(api_key="dai_sk_other", base_url=BASE, agent_name="support")
    platform.calls.clear()
    other.build_prompt_fragment("how do refunds work?")
    assert platform.paths()[0] == ROUTE


def test_when_the_pause_ends_the_route_is_tried_again(monkeypatch):
    monkeypatch.setattr(sr, "_ROUTE_PAUSE_S", 0.0)
    platform = _Platform(httpx.ReadTimeout("slow route"), _ok(MENU))
    monkeypatch.setattr(sr.httpx, "request", platform)
    router = _router()
    router.build_prompt_fragment("how do refunds work?")

    platform.route = _ok({**MENU, "strategy": "smart_routing", "query_ranked": True,
                          "routing_id": "rt_" + "b2" * 12})
    platform.calls.clear()
    _, routing_id = router.build_prompt_fragment("a different question")

    assert platform.paths() == [ROUTE]
    assert routing_id == "rt_" + "b2" * 12
