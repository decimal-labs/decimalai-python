"""One retry per transport failure kind, and a shared connection pool (2026-10-20).

On 2026-10-20 the OpenAI tutorial's `load_skill` failed in 1.21 s with `(timeout)`
while the backend served other requests normally: the SDK's 1 s connect budget,
spent by one lost packet, and no second attempt. The model was told the skill was
unavailable and answered without it. What must hold, each pinned through the
real `_request`:

  * a connection that could not be opened is retried once, for any method
    (nothing was sent);
  * a dropped connection or a read timeout is retried once only for requests
    safe to replay — GETs, and endpoints documented idempotent — and a read
    timeout never for the routing request;
  * a hot read's retry has a short read budget, and the menu fallback reads once;
  * each kind gets one spare attempt per call, and a call that fails both
    attempts counts once for the breaker;
  * calls share one pooled client, rebuilt in a forked child, and a patched
    ``httpx.request`` (how test suites stand in for the platform) still wins.
"""
from __future__ import annotations

import os

import httpx
import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter, SkillRouterError, _HotPathBreakers, routing_status

BASE = "http://127.0.0.1:8001"
BODY_PATH = "/api/v1/skills/refund-policy/body"
BODY = {
    "name": "refund-policy",
    "body": "# Refunds\n\nAsk for the order number first.",
    "content_hash": "97a3cb982d1a",
    "version": 2,
    "truncated": False,
}


@pytest.fixture(autouse=True)
def _fresh_breakers(monkeypatch):
    monkeypatch.setattr(sr, "_hot_path_breakers", _HotPathBreakers())


class _Platform:
    """A stubbed ``httpx.request`` that plays its answers in order and records
    every attempt."""

    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, method, url, *, params=None, json=None, headers=None, timeout=None):
        self.calls.append((method, url.removeprefix(BASE), timeout))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    def paths(self):
        return [(method, path) for method, path, _ in self.calls]


def _ok(body):
    return httpx.Response(200, json=body)


def _router():
    return SkillRouter(api_key="dai_sk_fixture", base_url=BASE, agent_name="support")


def test_the_hot_path_connect_budget_survives_a_resent_syn():
    timeout = sr._hot_path_timeout()
    assert timeout.connect == 2.0, "1 s is the OS's own SYN retransmit delay"
    assert timeout.read == sr._DEFAULT_HOT_PATH_READ_S


@pytest.mark.parametrize("failure", [
    httpx.ConnectTimeout("handshake timed out"),
    httpx.ConnectError("connection refused"),
], ids=["connect_timeout", "connect_error"])
def test_a_body_that_could_not_connect_once_still_loads(monkeypatch, failure):
    """The 2026-10-20 failure, replayed: the first connection fails, the skill loads."""
    platform = _Platform(failure, _ok(BODY))
    monkeypatch.setattr(sr.httpx, "request", platform)

    body = _router().get_skill_body("refund-policy")

    assert body == BODY["body"]
    assert platform.paths() == [("GET", BODY_PATH)] * 2
    assert sr._body_hash_tls.unavailable is None, "load_skill must not report the skill unavailable"
    status = routing_status()
    assert status.healthy and status.consecutive_failures == 0


@pytest.mark.parametrize("call, expected", [
    (lambda r: r.create_skill("refund-policy", "Refund rules", "# Refunds"), ("POST", "/api/v1/skills")),
    (lambda r: r.update_skill("skill-1", description="Refund rules"), ("PUT", "/api/v1/skills/skill-1")),
], ids=["post", "put"])
def test_a_write_that_could_not_connect_is_retried(monkeypatch, call, expected):
    """Nothing reached the platform, so even a write is safe to send again."""
    platform = _Platform(httpx.ConnectError("connection refused"), _ok({"id": "skill-1"}))
    monkeypatch.setattr(sr.httpx, "request", platform)

    assert call(_router())["id"] == "skill-1"
    assert platform.paths() == [expected] * 2


@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout("read timed out"),
    httpx.RemoteProtocolError("Server disconnected without sending a response."),
    httpx.ReadError("connection reset by peer"),
], ids=["read_timeout", "remote_protocol", "read_error"])
def test_a_write_the_platform_may_have_received_is_never_replayed(monkeypatch, failure):
    platform = _Platform(failure, _ok({"id": "skill-1"}))
    monkeypatch.setattr(sr.httpx, "request", platform)

    with pytest.raises(SkillRouterError) as raised:
        _router().create_skill("refund-policy", "Refund rules", "# Refunds")

    assert platform.paths() == [("POST", "/api/v1/skills")]
    assert raised.value.__cause__ is failure


def test_a_route_read_timeout_is_not_retried(monkeypatch):
    """A slow route is the platform's own latency; the menu fallback covers it."""
    platform = _Platform(httpx.ReadTimeout("slow route"))
    monkeypatch.setattr(sr.httpx, "request", platform)

    result = _router().smart_route("how do refunds work?")

    assert result["degraded_reason"] == "timeout"
    assert platform.paths() == [("POST", "/api/v1/skills/route")]


def test_a_route_whose_connection_dropped_is_retried(monkeypatch):
    """The routing request is replay-safe (it is retried on a 429 already); a
    dropped connection costs no platform time, unlike a slow read."""
    route = {"skills": [], "prompt_fragment": "", "strategy": "smart_routing"}
    platform = _Platform(httpx.RemoteProtocolError("Server disconnected"), _ok(route))
    monkeypatch.setattr(sr.httpx, "request", platform)

    assert _router().smart_route("how do refunds work?")["strategy"] == "smart_routing"
    assert platform.paths() == [("POST", "/api/v1/skills/route")] * 2


def test_a_hot_read_timeout_retries_once_on_a_short_budget(monkeypatch):
    platform = _Platform(httpx.ReadTimeout("stalled"), _ok(BODY))
    monkeypatch.setattr(sr.httpx, "request", platform)

    assert _router().get_skill_body("refund-policy") == BODY["body"]

    first, second = (timeout for _, _, timeout in platform.calls)
    assert first.read == sr._DEFAULT_HOT_PATH_READ_S
    assert second.read == sr._HOT_PATH_READ_RETRY_S
    assert second.connect == first.connect == sr._HOT_PATH_CONNECT_S


def test_a_short_read_budget_caps_the_retry(monkeypatch):
    """An operator who lowered the hot read budget below the retry's gets theirs."""
    monkeypatch.setenv(sr._HOT_PATH_READ_ENV, "1.5")
    platform = _Platform(httpx.ReadTimeout("stalled"), _ok(BODY))
    monkeypatch.setattr(sr.httpx, "request", platform)

    _router().get_skill_body("refund-policy")

    assert [timeout.read for _, _, timeout in platform.calls] == [1.5, 1.5]


def test_the_menu_fallback_reads_once(monkeypatch):
    """A failed route may cost a turn at most `_MENU_FALLBACK_READ_S` more."""
    platform = _Platform(httpx.ReadTimeout("slow route"), httpx.ReadTimeout("slow menu"))
    monkeypatch.setattr(sr.httpx, "request", platform)

    fragment, _ = _router().build_prompt_fragment("how do refunds work?")

    assert fragment == ""
    assert platform.paths() == [("POST", "/api/v1/skills/route"), ("GET", "/api/v1/skills/menu")]


def test_an_idempotent_assign_is_retried_after_a_read_timeout(monkeypatch):
    """The setup script's 2026-10-20 failure: assign waited out the 30 s cold budget."""
    reply = {"status": "ok", "action": "updated", "subscription_id": "sub-1", "pinned_version_id": None}
    platform = _Platform(httpx.ReadTimeout("The read operation timed out"), _ok(reply))
    monkeypatch.setattr(sr.httpx, "request", platform)

    assert _router().assign_skill("tutorial-order-support", "skill-1") == reply

    path = "/api/v1/agents/tutorial-order-support/skills/skill-1"
    assert platform.paths() == [("POST", path)] * 2
    first, second = (timeout for _, _, timeout in platform.calls)
    assert first == sr._COLD_PATH_TIMEOUT
    assert second.read == sr._COLD_PATH_TIMEOUT


def test_each_failure_kind_gets_one_spare_attempt(monkeypatch):
    platform = _Platform(httpx.ConnectError("refused"), httpx.ReadTimeout("stalled"), _ok(BODY))
    monkeypatch.setattr(sr.httpx, "request", platform)

    assert _router().get_skill_body("refund-policy") == BODY["body"]
    assert len(platform.calls) == 3


def test_a_body_that_fails_both_attempts_counts_once(monkeypatch):
    platform = _Platform(httpx.ConnectTimeout("handshake"), httpx.ConnectTimeout("handshake"))
    monkeypatch.setattr(sr.httpx, "request", platform)

    assert _router().get_skill_body("refund-policy") is None

    assert len(platform.calls) == 2
    assert sr._body_hash_tls.unavailable == "timeout"
    status = routing_status()
    assert status.consecutive_failures == 1
    assert status.timeouts == 1
    assert not status.breaker_open


# ── the shared pool ───────────────────────────────────────────


@pytest.fixture
def pooled(monkeypatch):
    """The real `_send` path: `httpx.request` unpatched, the pool on a mock transport."""
    assert httpx.request is httpx._api.request, "another fixture patched httpx.request"
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json=BODY)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(sr, "_platform_http", client)
    monkeypatch.setattr(sr, "_platform_http_pid", os.getpid())
    return client, seen


def test_calls_share_one_pooled_client(pooled):
    client, seen = pooled
    router = _router()

    router.get_skill_body("refund-policy")
    router.get_skill_body("refund-policy")

    assert seen == [("GET", BODY_PATH)] * 2
    assert sr._platform_client() is client


def test_a_forked_child_gets_its_own_client(pooled, monkeypatch):
    client, _ = pooled
    monkeypatch.setattr(sr, "_platform_http_pid", os.getpid() + 1)  # as inherited from a parent

    fresh = sr._platform_client()

    assert fresh is not client
    assert sr._platform_client() is fresh
    fresh.close()


def test_a_patched_httpx_request_wins_over_the_pool(pooled, monkeypatch):
    """Suites (here and in other repos) stub `httpx.request`; the pool must not
    route around them to a real network."""
    _, seen = pooled
    platform = _Platform(_ok(BODY))
    monkeypatch.setattr(sr.httpx, "request", platform)

    assert _router().get_skill_body("refund-policy") == BODY["body"]
    assert platform.paths() == [("GET", BODY_PATH)]
    assert seen == []


def test_the_pool_keeps_no_cookies(monkeypatch):
    """One pool serves every router and key in a process; a cookie set on one
    key's response must not ride along on the next request."""
    sent_cookies = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent_cookies.append(request.headers.get("cookie"))
        return httpx.Response(200, json=BODY, headers={"Set-Cookie": "session=key-a; Path=/"})

    monkeypatch.setattr(sr, "_platform_http", None)
    fresh = sr._platform_client()
    monkeypatch.setattr(fresh, "_transport", httpx.MockTransport(handler))

    SkillRouter(api_key="dai_sk_key_a", base_url=BASE).get_skill_body("refund-policy")
    SkillRouter(api_key="dai_sk_key_b", base_url=BASE).get_skill_body("refund-policy")

    assert sent_cookies == [None, None]
    fresh.close()
