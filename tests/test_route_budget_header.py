"""Route requests tell the platform how long they will wait (2026-10-28).

The platform lets a route's query embedding wait up to the caller's budget minus the rest of the
route (backend services/llm/embeddings.route_budget) instead of a fixed 4 s, which was sized for
this SDK's 5 s default. So every route request carries ``X-Decimal-Route-Budget``: the read budget
that request actually uses. Pinned through the real ``_request``:

  * the default budget is sent as 5, and DECIMALAI_SKILL_ROUTE_TIMEOUT_S raises it;
  * the value matches the read timeout of every attempt, including a connect retry;
  * only the route carries it.
"""
from __future__ import annotations

import httpx
import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter, _HotPathBreakers

BASE = "http://127.0.0.1:8001"
ROUTE = {"skills": [], "strategy": "semantic"}


@pytest.fixture(autouse=True)
def _fresh_breakers(monkeypatch):
    monkeypatch.setattr(sr, "_hot_path_breakers", _HotPathBreakers())
    monkeypatch.delenv(sr._HOT_PATH_READ_ENV, raising=False)


class _Platform:
    """A stubbed ``httpx.request`` that plays its answers in order and records each attempt's
    path, read timeout and route-budget header."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, method, url, *, params=None, json=None, headers=None, timeout=None):
        self.calls.append((url.removeprefix(BASE), timeout.read,
                           (headers or {}).get(sr._ROUTE_BUDGET_HEADER)))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _router():
    return SkillRouter(api_key="dai_sk_fixture", base_url=BASE, agent_name="support")


def test_a_route_sends_the_default_budget(monkeypatch):
    platform = _Platform(httpx.Response(200, json=ROUTE))
    monkeypatch.setattr(sr.httpx, "request", platform)

    _router().smart_route("how do refunds work?")

    assert platform.calls == [("/api/v1/skills/route", 5.0, "5")]


def test_a_raised_budget_is_what_the_route_sends_and_waits(monkeypatch):
    monkeypatch.setenv(sr._HOT_PATH_READ_ENV, "10")
    platform = _Platform(httpx.ConnectError("syn lost"), httpx.Response(200, json=ROUTE))
    monkeypatch.setattr(sr.httpx, "request", platform)

    _router().smart_route("how do refunds work?")

    assert platform.calls == [("/api/v1/skills/route", 10.0, "10")] * 2


def test_only_the_route_carries_it(monkeypatch):
    platform = _Platform(httpx.Response(200, json={"skills": []}))
    monkeypatch.setattr(sr.httpx, "request", platform)

    _router().get_menu()

    assert [(path, header) for path, _, header in platform.calls] == [("/api/v1/skills/menu", None)]
