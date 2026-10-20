"""Phase 1c: the body-injection gate must not cost a body the SDK cannot get wrong.

The gate (audit R2) injects a body only for a result the platform ranked by
THIS query. A small workspace is answered with its whole menu
(``strategy: "full_menu"``), and ``query_ranked`` there is a word-overlap test
against the top row: "I want my money back for order 42" shares no word with
``refund-policy: Refund rules`` and comes back ``query_ranked: false``, the
same as "thanks!". On the four adapters that had no ``load_skill`` tool then
(langchain, adk, crewai, anthropic; ADK and Anthropic got one from
``agent_config()`` in 0.20 and 0.21) prompt injection was the only way a body
reached the model, so the model was shown a menu row it could not open, and
nothing said so.

The SDK cannot tell a reworded question from small talk. What it can tell is
when there is no choice to get wrong:

* **The agent has one skill.** ``rows_total == 1`` on a ``full_menu`` result
  is the platform saying the menu IS this agent's whole eligible set and it is
  one skill. Its body is the only body any turn could get, so it is injected
  (as it was before the gate). The cost is that "thanks!" gets it too.
* **The developer named it.** ``priority_skills`` is shared guidance (reply
  format, house style) the developer reserved a body slot for. It applies to
  every reply, so it is injected whenever the platform offered it, ranked or
  not. It is still never injected when it was NOT offered.

Everywhere else the gate holds: in a workspace with several skills, the top
row of an unranked menu is the best-rated skill, not the relevant one, and the
gate keeps that body out of unrelated turns.

The transport is a stubbed ``httpx.request`` so the real request code runs.
"""

from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import unquote

import httpx
import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter, _HotPathBreakers, consume_last_delivered_names

BASE = "http://platform.test"
RID = "rt_" + "c" * 24
BODIES = {
    "refund-policy": "Opened boxes carry a 23.5% restocking fee.",
    "tone-guide": "Be warm and brief.",
    "shipping-faq": "Orders ship in 2 days.",
}
REWORDED = "I want my money back for order 42"


@pytest.fixture(autouse=True)
def _fresh_breaker_and_budget(monkeypatch):
    monkeypatch.setattr(sr, "_hot_path_breakers", _HotPathBreakers())
    token = sr._body_budget_ctx.set(None)
    yield
    sr._body_budget_ctx.reset(token)


def menu(names: List[str], *, rows_total: Any = "auto", **over) -> Dict[str, Any]:
    """A ``full_menu`` answer the way the platform's ``get_menu`` sends it."""
    rows = "\n".join(f"| {n} | {n} rules |" for n in names)
    out: Dict[str, Any] = {
        "strategy": "full_menu",
        "skills": [{"name": n, "description": f"{n} rules"} for n in names],
        "skill_count": len(names),
        "prompt_fragment": f"## Available Skills\n{rows}",
        "stable_menu": f"## Available Skills\n{rows}",
        "menu_instruction": "Follow a skill that applies.",
        "routing_hint": "",
        "stable_menu_skills": list(names),
        "query_ranked": False,
        "truncated": False,
        "routing_id": RID,
    }
    if rows_total == "auto":
        out["rows_total"] = len(names)
    elif rows_total is not None:
        out["rows_total"] = rows_total
    out.update(over)
    return out


class FakePlatform:
    def __init__(self, route: Dict[str, Any]):
        self.route = route
        self.calls: List[Dict[str, Any]] = []

    def bodies_fetched(self) -> List[str]:
        return [
            unquote(c["url"][len(f"{BASE}/api/v1/skills/"):-len("/body")])
            for c in self.calls if c["url"].endswith("/body")
        ]

    def __call__(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        if url.endswith("/api/v1/skills/route") or url.endswith("/api/v1/skills/menu"):
            return httpx.Response(200, json=self.route)
        if url.endswith("/body"):
            name = unquote(url[len(f"{BASE}/api/v1/skills/"):-len("/body")])
            if name not in BODIES:
                return httpx.Response(404, json={"detail": "Skill not found"})
            body = BODIES[name]
            return httpx.Response(200, json={
                "name": name, "version": 1, "body": body, "truncated": False,
                "total_chars": len(body), "content_hash": "cd" * 32,
                "content_hash_matches_body": True,
            })
        raise AssertionError(f"unexpected request {method} {url}")


def _install(monkeypatch, route) -> FakePlatform:
    platform = FakePlatform(route)
    monkeypatch.setattr(httpx, "request", platform)
    return platform


def _router(**kw) -> SkillRouter:
    # inject_body=True is what the four injection-only adapters resolve to.
    return SkillRouter(api_key="dai_sk_test", base_url=BASE, inject_body=True, **kw)


def _parts(router: SkillRouter, query: str) -> str:
    """The langchain / ADK entry point: the prefix is where bodies ride."""
    prefix, _tail, _rid = router.build_prompt_parts(query, scope="run")
    return prefix


def _fragment(router: SkillRouter, query: str) -> str:
    """The crewai entry point."""
    fragment, _rid = router.build_prompt_fragment(query, scope="run")
    return fragment


ENTRY_POINTS = {"build_prompt_parts": _parts, "build_prompt_fragment": _fragment}


# ── an agent with one skill: there is no wrong body to inject ─────────────


class TestTheAgentsOnlySkillIsDelivered:
    @pytest.mark.parametrize("entry", list(ENTRY_POINTS))
    @pytest.mark.parametrize("query", [
        REWORDED, "can I return this?", "this product broke, I'd like to send it back",
    ])
    def test_a_reworded_question_gets_the_body(self, monkeypatch, entry, query):
        platform = _install(monkeypatch, menu(["refund-policy"]))

        text = ENTRY_POINTS[entry](_router(), query)

        assert f"## Skill: refund-policy\n\n{BODIES['refund-policy']}" in text, (
            "the agent's one skill was shown as a menu row the model cannot open"
        )
        assert consume_last_delivered_names() == ["refund-policy"]
        assert platform.bodies_fetched() == ["refund-policy"]

    def test_small_talk_gets_it_too_which_is_the_cost(self, monkeypatch):
        """The SDK cannot tell "thanks!" from a reworded question; with one
        skill there is only one body either could get. Pinned so the trade-off
        is visible, not accidental."""
        _install(monkeypatch, menu(["refund-policy"]))
        assert "## Skill: refund-policy" in _fragment(_router(), "thanks!")

    @pytest.mark.parametrize("route", [
        # A retrieval fallback in a big workspace serves a capped slice of it:
        # one row out of forty is the best-rated one, not the only one.
        pytest.param(menu(["refund-policy"], rows_total=40, truncated=True, degraded=True,
                          degraded_reason="no_retrieval_candidates"),
                     id="fallback-one-row-of-forty"),
        # A platform that does not say how big the eligible set is.
        pytest.param(menu(["refund-policy"], rows_total=None), id="size-not-stated"),
        # Hybrid retrieval picked one row of a large set and says the order
        # ignored the query.
        pytest.param(menu(["refund-policy"], strategy="smart_routing"), id="smart_routing-unranked"),
    ])
    def test_one_row_that_is_not_the_whole_set_stays_menu_only(self, monkeypatch, route):
        platform = _install(monkeypatch, route)
        assert "## Skill:" not in _fragment(_router(), REWORDED)
        assert platform.bodies_fetched() == []


# ── several skills: the gate still keeps the wrong body out ───────────────


class TestSeveralSkillsStillGetTheMenuOnly:
    @pytest.mark.parametrize("entry", list(ENTRY_POINTS))
    @pytest.mark.parametrize("query", ["thanks!", REWORDED])
    def test_an_unranked_menu_injects_no_body(self, monkeypatch, entry, query):
        """The top row is the best-rated skill (shipping-faq here), so before
        the gate "I want my money back" got the SHIPPING body."""
        platform = _install(monkeypatch, menu(["shipping-faq", "refund-policy"]))

        text = ENTRY_POINTS[entry](_router(), query)

        assert "## Skill:" not in text
        assert consume_last_delivered_names() == []
        assert platform.bodies_fetched() == []

    def test_a_ranked_menu_still_injects_its_top_row(self, monkeypatch):
        _install(monkeypatch, menu(["refund-policy", "shipping-faq"], query_ranked=True))
        text = _fragment(_router(), "refund an opened box")
        assert "## Skill: refund-policy" in text
        assert "## Skill: shipping-faq" not in text


# ── priority_skills: the developer reserved the slot ──────────────────────


class TestPrioritySkillsAreDeliveredRankedOrNot:
    @pytest.mark.parametrize("entry", list(ENTRY_POINTS))
    @pytest.mark.parametrize("query", ["thanks!", REWORDED])
    def test_an_offered_priority_skill_is_injected_on_an_unranked_turn(
        self, monkeypatch, entry, query,
    ):
        """The support scaffold reserves ``support-reply-format``-style house
        guidance with ``priority_skills``; it applies to every reply."""
        platform = _install(monkeypatch, menu(["shipping-faq", "refund-policy", "tone-guide"]))
        router = _router(priority_skills=["tone-guide"], inject_body_top_k=3)

        text = ENTRY_POINTS[entry](router, query)

        assert f"## Skill: tone-guide\n\n{BODIES['tone-guide']}" in text
        # Only the reserved slot: the unranked top row is still kept out.
        assert "## Skill: shipping-faq" not in text
        assert "## Skill: refund-policy" not in text
        assert consume_last_delivered_names() == ["tone-guide"]
        assert platform.bodies_fetched() == ["tone-guide"]

    def test_a_priority_skill_the_platform_did_not_offer_is_never_injected(self, monkeypatch):
        """Never widen access: a name the platform did not offer this agent
        stays out, whatever the developer listed."""
        platform = _install(monkeypatch, menu(["shipping-faq", "refund-policy"]))
        router = _router(priority_skills=["tone-guide"], inject_body_top_k=3)

        assert "## Skill:" not in _fragment(router, "thanks!")
        assert platform.bodies_fetched() == []

    def test_a_ranked_turn_keeps_priority_first_then_the_ranked_rows(self, monkeypatch):
        _install(monkeypatch, menu(["refund-policy", "shipping-faq", "tone-guide"], query_ranked=True))
        router = _router(priority_skills=["tone-guide"], inject_body_top_k=2)

        _fragment(router, "refund an opened box")

        assert consume_last_delivered_names() == ["tone-guide", "refund-policy"]

    def test_no_query_still_injects_nothing(self, monkeypatch):
        """The menu-only path (no query, or strategy='menu') never injected a
        body and does not start now."""
        platform = _install(monkeypatch, menu(["tone-guide"]))
        router = _router(priority_skills=["tone-guide"])
        fragment, _ = router.build_prompt_fragment(None, scope="run")
        assert "## Skill:" not in fragment
        assert platform.bodies_fetched() == []
