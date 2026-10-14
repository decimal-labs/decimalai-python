"""Phase 1d: a circuit breaker pauses only the routing its failures are about.

The hot-path breaker was ONE object for the whole process. Until Phase 1 only
transport failures fed it (a timeout, a refused connection), and those are a
property of the host, not of any API key. Phase 1 rightly started counting
401, 402, 403, 429 and 5xx as failures too, because each leaves a turn without
skills. On a process-wide breaker that meant three 401s from one router's
revoked key paused skill routing for every other router in the process,
healthy keys included, and the ERROR log told the operator to raise the read
timeout, which does nothing for a 401.

Now:

* an HTTP failure (401/402/403/429/5xx, a 404 from /route, a 200 that is not
  JSON) opens the breaker of that (base URL, API key) only;
* a transport failure still pauses every key on the same HOST, and nothing on
  another host: the host did not answer, which no key can change, and sharing
  it is what stops N routers each paying the timeout during an outage;
* ``routing_status()`` still reports every key's failures, so a process with
  one broken tenant reads unhealthy while that tenant's agents run without
  skills.

The transport is a stubbed ``httpx.request``, so the real ``SkillRouter._request``
runs; nothing touches a network.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import httpx
import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter, routing_status

HOST_A = "http://platform-a.test"
HOST_B = "http://platform-b.test"
GOOD, OTHER, BAD = "dai_sk_good", "dai_sk_other", "dai_sk_revoked"

ROUTE = {
    "strategy": "smart_routing",
    "skills": [{"name": "refund-policy", "description": "Refund rules"}],
    "prompt_fragment": "## Recommended Skills\n| refund-policy | Refund rules |",
    "routing_id": "rt_" + "a" * 24,
}


class _Clock:
    """Stands in for ``time`` inside ``skill_router``: both clocks are ours."""

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


class Platform:
    """A stubbed ``httpx.request``.

    ``key_status`` answers every call made with that key with that status.
    ``down`` is the set of hosts whose calls time out. ``script`` is a list of
    outcomes ("timeout" or a status) consumed one per call before either.
    """

    def __init__(self) -> None:
        self.key_status: Dict[str, int] = {}
        self.down: set = set()
        self.script: List[Any] = []
        self.calls: List[Tuple[str, str]] = []

    def __call__(self, method, url, **kw):
        key = (kw.get("headers") or {}).get("Authorization", "").removeprefix("Bearer ")
        self.calls.append((url, key))
        if self.script:
            outcome = self.script.pop(0)
            if outcome == "timeout":
                raise httpx.ReadTimeout("slow")
            return httpx.Response(outcome, json={"detail": f"forced {outcome}"})
        if urlsplit(url).hostname in self.down:
            raise httpx.ConnectTimeout("no route to host")
        if key in self.key_status:
            code = self.key_status[key]
            return httpx.Response(code, json={"detail": f"forced {code}"})
        if url.endswith("/api/v1/skills/route"):
            return httpx.Response(200, json=ROUTE)
        raise AssertionError(f"unexpected request {method} {url}")

    def made(self, key: str, host: Optional[str] = None) -> int:
        return sum(
            1 for url, k in self.calls
            if k == key and (host is None or url.startswith(host))
        )


@pytest.fixture(autouse=True)
def platform(monkeypatch) -> Platform:
    monkeypatch.setattr(sr, "_hot_path_breakers", sr._HotPathBreakers())
    fake = Platform()
    monkeypatch.setattr(httpx, "request", fake)
    return fake


@pytest.fixture
def clock(monkeypatch) -> _Clock:
    fake = _Clock()
    monkeypatch.setattr(sr, "time", fake)
    return fake


def _router(key: str, base_url: str = HOST_A) -> SkillRouter:
    return SkillRouter(api_key=key, base_url=base_url)


def _open_by_status(router: SkillRouter, platform: Platform, code: int) -> None:
    platform.key_status[router.api_key] = code
    for i in range(3):
        assert router.smart_route(f"question {i}")["degraded_reason"] == f"http_{code}"
    assert router.smart_route("one more")["degraded_reason"] == "circuit_open"


# ── one key's failed responses pause only that key ───────────────────────


class TestAFailingKeyPausesOnlyItself:
    @pytest.mark.parametrize("code", [401, 402, 403, 429, 500, 503])
    def test_another_key_on_the_same_platform_keeps_routing(self, platform, code):
        bad, good = _router(BAD), _router(GOOD)
        _open_by_status(bad, platform, code)
        made_by_bad = platform.made(BAD)

        out = good.smart_route("refund an opened box")

        assert out.get("routing_id") == ROUTE["routing_id"], (
            f"a healthy key got {out.get('degraded_reason')!r} after another key's HTTP {code}s"
        )
        assert platform.made(GOOD) == 1
        assert platform.made(BAD) == made_by_bad, "the failing key is still paused"

    def test_a_router_built_with_the_corrected_key_routes_at_once(self, platform):
        """The fix-the-key path: `init(api_key=B)` drops the routers and builds
        new ones with B. It used to wait out the cooldown A's 401s started."""
        _open_by_status(_router(BAD), platform, 401)
        fixed = _router(GOOD)
        assert fixed.build_prompt_fragment("refund an opened box", scope="run")[1] == ROUTE["routing_id"]

    def test_two_routers_with_one_key_share_its_breaker(self, platform):
        """A revoked key fails identically wherever it is used: a second router
        holding it (another adapter's singleton, a rebuilt one) is paused too."""
        _open_by_status(_router(BAD), platform, 401)
        before = platform.made(BAD)
        assert _router(BAD).smart_route("q")["degraded_reason"] == "circuit_open"
        assert platform.made(BAD) == before

    def test_the_same_key_against_another_base_url_is_its_own_breaker(self, platform):
        _open_by_status(_router(BAD, HOST_A), platform, 429)
        platform.key_status.pop(BAD)
        assert _router(BAD, HOST_B).smart_route("q").get("routing_id") == ROUTE["routing_id"]

    def test_an_answer_for_one_key_does_not_close_another_keys_breaker(self, platform):
        bad, good = _router(BAD), _router(GOOD)
        _open_by_status(bad, platform, 401)
        good.smart_route("q")
        assert bad.smart_route("q")["degraded_reason"] == "circuit_open"


# ── transport failures: shared by every key on that host, and only there ─


class TestTransportFailuresArePerHost:
    def test_a_dead_host_pauses_every_key_on_it(self, platform):
        """The host did not answer, which no key changes. Sharing this is what
        keeps N routers from each paying the timeout during an outage."""
        platform.down.add(urlsplit(HOST_A).netloc)
        first = _router(GOOD)
        for i in range(3):
            assert first.smart_route(f"q{i}")["degraded_reason"] == "timeout"

        out = _router(OTHER).smart_route("q")

        assert out["degraded_reason"] == "circuit_open"
        assert platform.made(OTHER) == 0, "a second key paid the timeout again"

    def test_a_dead_host_does_not_pause_another_host(self, platform):
        platform.down.add(urlsplit(HOST_A).netloc)
        for i in range(3):
            _router(GOOD, HOST_A).smart_route(f"q{i}")
        assert _router(GOOD, HOST_A).smart_route("q")["degraded_reason"] == "circuit_open"

        out = _router(GOOD, HOST_B).smart_route("q")

        assert out.get("routing_id") == ROUTE["routing_id"]

    def test_any_answer_proves_the_host_is_reachable(self, platform):
        """A 401 is an answer: the host is up. Two timeouts, a 401 and one more
        timeout are not three transport failures in a row."""
        platform.script = ["timeout", "timeout", 401, "timeout"]
        router = _router(GOOD)
        reasons = [router.smart_route(f"q{i}")["degraded_reason"] for i in range(4)]
        assert reasons == ["timeout", "timeout", "http_401", "timeout"]
        assert router.smart_route("q").get("routing_id") == ROUTE["routing_id"], (
            "the host breaker opened on failures an answer had interrupted"
        )


# ── routing_status() still sees every key ───────────────────────────────


class TestRoutingStatusSeesEveryKey:
    def test_a_failing_key_reads_unhealthy_while_another_key_succeeds(self, platform):
        """Agents on the failing key run without skills. Another key's success
        does not change that, so it must not reset the process's health."""
        platform.key_status[BAD] = 401
        _router(BAD).smart_route("q")
        _router(GOOD).smart_route("q")

        st = routing_status()

        assert st.healthy is False
        assert st.consecutive_failures == 1
        assert st.http_errors == 1 and st.last_status_code == 401
        assert st.breaker_open is False

    def test_an_open_key_breaker_is_reported_open(self, platform):
        _open_by_status(_router(BAD), platform, 401)
        _router(GOOD).smart_route("q")
        st = routing_status()
        assert st.breaker_open is True and st.opens == 1
        assert st.suppressed == 1
        assert st.healthy is False

    def test_counters_are_totals_over_every_breaker(self, platform):
        platform.script = ["timeout"]
        _router(GOOD, HOST_B).smart_route("q")
        platform.key_status[BAD] = 503
        _router(BAD).smart_route("q")
        _router(BAD).smart_route("q")
        st = routing_status()
        assert (st.timeouts, st.http_errors) == (1, 2)

    def test_a_key_the_process_stopped_using_stops_counting(self, platform, clock):
        """`init(api_key=B)` after A was revoked: nothing routes with A again, so
        A's last failure says nothing about the agents running now."""
        _open_by_status(_router(BAD), platform, 401)
        good = _router(GOOD)
        good.smart_route("q")
        assert routing_status().healthy is False  # A was failing a moment ago

        clock.now += 61.0
        good.smart_route("q")

        st = routing_status()
        assert st.healthy is True, "a key nothing routes with any more kept the process unhealthy"
        assert st.breaker_open is False
        assert st.http_errors == 3, "the totals keep A's failures"

    def test_a_failing_key_left_idle_stays_unhealthy(self, platform, clock):
        """The only key in the process is not superseded by being idle: its next
        turn will fail the same way."""
        platform.key_status[BAD] = 401
        _router(BAD).smart_route("q")
        clock.now += 3600.0
        assert routing_status().healthy is False

    def test_a_dead_host_the_process_moved_off_stops_counting(self, platform, clock):
        platform.down.add(urlsplit(HOST_A).netloc)
        for i in range(3):
            _router(GOOD, HOST_A).smart_route(f"q{i}")
        moved = _router(GOOD, HOST_B)
        moved.smart_route("q")
        clock.now += 61.0
        moved.smart_route("q")
        assert routing_status().healthy is True

    def test_the_registry_is_bounded_and_the_totals_survive_eviction(self, platform, monkeypatch):
        monkeypatch.setattr(sr, "_hot_path_breakers", sr._HotPathBreakers(max_breakers=8))
        for i in range(40):
            key = f"dai_sk_tenant{i}"
            platform.key_status[key] = 503
            _router(key).smart_route("q")
        assert len(sr._hot_path_breakers._breakers) <= 8
        assert routing_status().http_errors == 40


# ── the ERROR line names the cause it can do something about ──────────────


class TestTheOpenLogFitsTheCause:
    def _open_lines(self, caplog) -> List[str]:
        return [r.getMessage() for r in caplog.records
                if r.levelno == logging.ERROR and "pausing" in r.getMessage()]

    def test_a_revoked_key_is_not_told_to_raise_the_timeout(self, platform, caplog):
        caplog.set_level(logging.ERROR, logger="decimalai.skill_router")
        _open_by_status(_router(BAD), platform, 401)
        (line,) = self._open_lines(caplog)
        assert sr._HOT_PATH_READ_ENV not in line
        assert "401" in line and "API key" in line
        assert "other keys" in line
        assert BAD not in line, "the key itself must never reach a log"

    @pytest.mark.parametrize("code,hint", [(402, "plan"), (429, "rate"), (503, "server error")])
    def test_each_status_gets_its_own_hint(self, platform, caplog, code, hint):
        caplog.set_level(logging.ERROR, logger="decimalai.skill_router")
        _open_by_status(_router(BAD), platform, code)
        (line,) = self._open_lines(caplog)
        assert hint in line.lower()
        assert sr._HOT_PATH_READ_ENV not in line

    def test_credentials_in_the_base_url_never_reach_the_log(self, platform, caplog):
        caplog.set_level(logging.WARNING, logger="decimalai.skill_router")
        # Assembled here, so this public file carries no URL-with-credentials
        # literal for a secret scanner to flag. The values are fake.
        credentialed = "http://" + ":".join(("proxyuser", "proxysecret")) + "@platform-a.test"
        router = _router(BAD, credentialed)
        _open_by_status(router, platform, 401)
        platform.key_status.pop(BAD)
        platform.down.add(urlsplit(HOST_A).netloc)
        other = _router(GOOD, credentialed)
        for i in range(4):
            other.smart_route(f"q{i}")
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "pausing" in text and "platform-a.test" in text
        assert "proxysecret" not in text

    def test_a_dead_host_still_points_at_the_timeout(self, platform, caplog):
        caplog.set_level(logging.ERROR, logger="decimalai.skill_router")
        platform.down.add(urlsplit(HOST_A).netloc)
        for i in range(3):
            _router(GOOD).smart_route(f"q{i}")
        (line,) = self._open_lines(caplog)
        assert sr._HOT_PATH_READ_ENV in line
        assert "platform-a.test" in line
        assert GOOD not in line
