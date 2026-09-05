"""The SDK must not be load-bearing for the customer's agent.

Every test here corresponds to a failure observed by RUNNING the shipped code on
2026-08-28, when `decimalai-docs/guides/trust-and-exit.mdx` already promised
"turn DecimalAI off and your agent still works" and "your agent never waits on
our API to answer your users". Both were false.

These are ratchets. Each was seen to go RED with the old behaviour restored.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

import decimalai
from decimalai._agent import _reset_prompt_cache
from decimalai.skill_router import (
    _COLD_PATH_TIMEOUT,
    _DEFAULT_HOT_PATH_READ_S,
    _HOT_PATH_READ_CEILING_S,
    _HOT_PATH_READ_ENV,
    _HOT_PATH_READ_FLOOR_S,
    SkillRouter,
    _CircuitBreaker,
    _hot_path_read_budget,
    _hot_path_timeout,
    _is_hot_path,
    routing_status,
)

UNREACHABLE = "http://127.0.0.1:9"  # discard port: refuses immediately


def _serve(handler_cls):
    """A throwaway server on an ephemeral port.

    ThreadingHTTPServer with daemon threads, and `server_close()` rather than
    `shutdown()`, because httpx holds the connection alive: a blocking shutdown
    waits for the handler thread, which is either sleeping (the hang test) or
    parked reading the next keep-alive request. That cost 90s across this file.
    """
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def _stop(srv):
    srv.server_close()


class TestLoadAgentSurvivesAnOutage:
    """`load_agent()` runs at import in a worker's boot path. An unreachable
    platform used to mean the customer's agent could not start at all."""

    def test_a_fallback_lets_the_agent_boot_with_the_platform_down(self):
        _reset_prompt_cache()
        decimalai.init(api_key="dai_sk_t", base_url=UNREACHABLE, verify=False)
        cfg = decimalai.load_agent("bot", fallback="You are a returns agent.")
        assert cfg.system_prompt == "You are a returns agent."
        assert cfg.is_fallback is True

    def test_without_a_fallback_it_still_fails_closed(self):
        """Fail-closed is deliberate: substituting a prompt nobody wrote makes
        the agent follow invented instructions and look fine doing it."""
        _reset_prompt_cache()
        decimalai.init(api_key="dai_sk_t", base_url=UNREACHABLE, verify=False)
        with pytest.raises(Exception):
            decimalai.load_agent("bot")

    def test_a_previously_read_prompt_is_served_when_the_platform_goes_away(self):
        class Ok(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = (
                    b'{"agent_name":"bot","system_prompt":"LIVE",'
                    b'"version_number":3,"content_hash":"h","version_mode":"latest"}'
                )
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        _reset_prompt_cache()
        srv, url = _serve(Ok)
        try:
            decimalai.init(api_key="dai_sk_t", base_url=url, verify=False)
            live = decimalai.load_agent("bot")
            assert live.system_prompt == "LIVE"
            assert live.is_fallback is False
        finally:
            _stop(srv)

        stale = decimalai.load_agent("bot")
        assert stale.system_prompt == "LIVE"
        assert stale.is_fallback is True
        assert stale.stale_age_seconds is not None

    def test_a_404_still_raises_even_with_a_fallback(self):
        """A reachable platform saying "no such agent" is a real answer, not an
        outage. Serving a fallback there would hide a deleted or misspelled agent."""
        class NotFound(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(404)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        _reset_prompt_cache()
        srv, url = _serve(NotFound)
        try:
            decimalai.init(api_key="dai_sk_t", base_url=url, verify=False)
            with pytest.raises(Exception):
                decimalai.load_agent("bot", fallback="invented")
        finally:
            _stop(srv)


class TestTheHotPathCannotHoldAUserTurn:
    def test_route_and_menu_are_hot_paths_and_everything_else_is_not(self):
        assert _is_hot_path("/api/v1/skills/route")
        assert _is_hot_path("/api/v1/skills/menu")
        assert not _is_hot_path("/api/v1/skills/x/body")
        assert not _is_hot_path("/api/v1/skills/x/publish")

    def test_the_hot_path_read_budget_is_seconds_not_the_cold_30(self):
        """A 30s budget inside a user's turn turns a platform brownout into an
        outage of the CUSTOMER's product. Measured at 30.3s per turn before."""
        assert _hot_path_timeout().read is not None
        assert _hot_path_timeout().read <= 10.0
        assert _COLD_PATH_TIMEOUT == 30.0

    def test_the_budget_clears_what_a_healthy_platform_actually_serves(self):
        """The OTHER half of the contract, and the half that was missing.

        Shipped at 2.0s in 0.12.0. Prod serves /skills/route at p95 1.39s when
        healthy (backend at cpu=2, same load), so 2.0s left 1.4x of headroom —
        inside ordinary variance. The fleet went from delivering a skill body in
        69.4% of sessions to 3.1% in the hour it restarted onto that release,
        with `no_skill_offered` going 7.9% -> 77.6%.

        A budget that fails on a healthy platform is not fail-fast, it is off.
        2.0 must not be a legal default again.
        """
        assert _DEFAULT_HOT_PATH_READ_S >= 3.0, (
            "the default must clear the measured healthy p95 (1.39s) with real "
            "margin — see the fleet delivery collapse of 2026-09-01"
        )
        assert _DEFAULT_HOT_PATH_READ_S < _COLD_PATH_TIMEOUT

    def test_the_budget_is_configurable_and_clamped(self, monkeypatch):
        """An operator can tune it without a release; garbage cannot break a turn."""
        monkeypatch.setenv(_HOT_PATH_READ_ENV, "7.5")
        assert _hot_path_read_budget() == 7.5
        assert _hot_path_timeout().read == 7.5

        # Clamped at both ends: below the floor cannot succeed against a warm
        # backend, above the ceiling is the 30s stall we removed.
        monkeypatch.setenv(_HOT_PATH_READ_ENV, "0.01")
        assert _hot_path_read_budget() == _HOT_PATH_READ_FLOOR_S
        monkeypatch.setenv(_HOT_PATH_READ_ENV, "600")
        assert _hot_path_read_budget() == _HOT_PATH_READ_CEILING_S

        # Garbage and nonsense fall back rather than raising — this runs inside
        # the customer's turn, so a bad env var must not be what breaks it.
        for bad in ("", "abc", "-1", "0"):
            monkeypatch.setenv(_HOT_PATH_READ_ENV, bad)
            assert _hot_path_read_budget() == _DEFAULT_HOT_PATH_READ_S

    def test_the_budget_is_read_per_call_not_frozen_at_import(self, monkeypatch):
        """A module constant captures whatever the env was at first import,
        which in a test suite is whichever test ran first."""
        monkeypatch.setenv(_HOT_PATH_READ_ENV, "9")
        assert _hot_path_timeout().read == 9.0
        monkeypatch.setenv(_HOT_PATH_READ_ENV, "4")
        assert _hot_path_timeout().read == 4.0

    def test_a_hung_platform_returns_in_seconds_and_the_turn_proceeds(self):
        class Hang(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                time.sleep(30)

        srv, url = _serve(Hang)
        try:
            router = SkillRouter(api_key="k", base_url=url)
            start = time.monotonic()
            out = router.smart_route("hello")  # fails OPEN
            elapsed = time.monotonic() - start
        finally:
            _stop(srv)
        assert elapsed < 8.0, f"a hung platform held the turn for {elapsed:.1f}s"
        assert out["skills"] == []

    def test_the_breaker_opens_and_stops_paying_the_timeout(self):
        breaker = _CircuitBreaker(threshold=3, cooldown_s=30.0)
        assert breaker.is_open() is False
        for _ in range(3):
            breaker.record_failure()
        assert breaker.is_open() is True
        breaker.record_success()
        assert breaker.is_open() is False

    def test_the_breaker_reopens_the_gate_after_the_cooldown(self):
        breaker = _CircuitBreaker(threshold=1, cooldown_s=0.0)
        breaker.record_failure()
        assert breaker.is_open() is False  # cooldown elapsed -> half-open

    def test_a_blip_costs_the_base_cooldown_not_the_ceiling(self):
        """It was a flat 30s, so three slow calls bought a guaranteed half
        minute of skill-less answers — and on a platform whose latency is
        bursty, one slow minute suppressed routing through the next one."""
        breaker = _CircuitBreaker(threshold=1, cooldown_s=5.0, max_cooldown_s=30.0)
        breaker.record_failure()
        assert breaker._current_cooldown() == 5.0

    def test_repeated_opens_back_off_and_a_success_resets_the_ladder(self):
        """A real outage must still back all the way off, or the breaker is
        just a slower retry loop."""
        breaker = _CircuitBreaker(threshold=1, cooldown_s=5.0, max_cooldown_s=30.0)
        seen = []
        for _ in range(5):
            breaker.record_failure()
            seen.append(breaker._current_cooldown())
            breaker._opened_at = None  # force half-open without sleeping
        assert seen == [5.0, 10.0, 20.0, 30.0, 30.0], seen

        breaker.record_success()
        breaker.record_failure()
        assert breaker._current_cooldown() == 5.0, "one success must reset the ladder"


class TestRoutingDegradationIsVisible:
    """The 2026-09-01 failure ran for 21 hours across 93 agents with every
    health signal green, because a hot-path timeout degrades to an empty menu
    and a 200. Nothing reported that agents were answering without skills."""

    def test_a_healthy_process_reports_healthy(self):
        from decimalai import skill_router as sr

        sr._hot_path_breaker = _CircuitBreaker()
        assert routing_status().healthy is True
        assert routing_status().breaker_open is False

    def test_a_timeout_is_counted_and_flips_healthy(self):
        from decimalai import skill_router as sr

        sr._hot_path_breaker = _CircuitBreaker(threshold=3)
        sr._hot_path_breaker.record_failure(httpx.ReadTimeout("too slow"))
        st = routing_status()
        assert st.healthy is False, "a timeout must not read as healthy"
        assert st.timeouts == 1
        assert "ReadTimeout" in (st.last_error or "")

    def test_the_open_breaker_is_reported_with_a_count(self):
        from decimalai import skill_router as sr

        sr._hot_path_breaker = _CircuitBreaker(threshold=2, cooldown_s=60.0)
        for _ in range(2):
            sr._hot_path_breaker.record_failure(httpx.ConnectError("down"))
        st = routing_status()
        assert st.breaker_open is True
        assert st.opens == 1
        assert st.healthy is False

    def test_a_success_restores_healthy(self):
        from decimalai import skill_router as sr

        sr._hot_path_breaker = _CircuitBreaker(threshold=2)
        sr._hot_path_breaker.record_failure(httpx.ConnectError("down"))
        sr._hot_path_breaker.record_success()
        st = routing_status()
        assert st.healthy is True
        # The cumulative counter is NOT reset — a health check wants "has this
        # ever degraded", not a gauge a recovery quietly zeroes.
        assert st.timeouts == 1


class TestInjectBodyResolvesPerAdapter:
    """The 2026-08-28 defect: a global `False` default meant adapters with no
    tool loop handed the model skill titles it had no mechanism to read."""

    def _cfg(self, **kw):
        from decimalai._config import DecimalConfig

        return DecimalConfig(api_key="k", **kw)

    def test_an_adapter_with_no_tool_loop_injects(self):
        assert self._cfg().resolve_inject_body(has_tool_loop=False) is True

    def test_an_adapter_with_a_tool_loop_does_not_double_deliver(self):
        assert self._cfg().resolve_inject_body(has_tool_loop=True) is False

    @pytest.mark.parametrize("explicit", [True, False])
    def test_an_explicit_setting_always_wins(self, explicit):
        cfg = self._cfg(inject_skill_body=explicit)
        assert cfg.resolve_inject_body(has_tool_loop=True) is explicit
        assert cfg.resolve_inject_body(has_tool_loop=False) is explicit

    def test_unset_is_none_so_it_is_distinguishable_from_false(self):
        assert self._cfg().inject_skill_body is None


class TestRepeatLoadSkillTerminates:
    """A repeat load returning the body again left the tool loop unbounded:
    4/4 runs of the shipped openai-agents scaffold died with MaxTurnsExceeded."""

    def test_a_repeat_load_returns_a_stop_instruction_not_the_body(self):
        from decimalai.skill_router import _BodyLoadBudget

        budget = _BodyLoadBudget(max_bodies=5, token_budget=10_000, deadline_s=30.0)
        assert budget.check("policy") is None
        budget.record("policy", 10)
        repeat = budget.check("policy")
        assert repeat is not None
        assert "already loaded" in repeat
        assert "budget exhausted" not in repeat  # it is free, just terminal


class TestDotenvResolvesFromTheProjectNotSitePackages:
    def test_it_uses_the_working_directory(self):
        """`load_dotenv()` with no path resolves from _config.py inside
        site-packages, so whether a project's .env is read was decided by where
        the venv happened to sit: fine on a laptop, silent in every container."""
        import inspect

        from decimalai import _config

        src = inspect.getsource(_config)
        assert "find_dotenv(usecwd=True)" in src


class TestManifestRegistrationDoesNotRepeatItself:
    """`POST /api/v1/manifests` is 17.9% of ALL backend traffic and ~99.4% of it
    dedupes server-side (measured on prod, 2026-09-03: ~50,800 registrations a
    day resolving to 200 distinct hashes).

    The tracker kept a SINGLE hash slot, so an oscillating snapshot re-sent a
    manifest the server already had.
    """

    def _snap(self, h):
        from decimalai.schema.manifest import ManifestSnapshot

        s = ManifestSnapshot(agent_name="bot")
        s.manifest_hash = h
        return s

    def test_an_oscillating_snapshot_registers_each_hash_once(self):
        from decimalai.schema.manifest import ManifestTracker

        t = ManifestTracker()
        assert t.check_and_update(self._snap("a")) is True
        assert t.check_and_update(self._snap("b")) is True
        # Back to a shape this process ALREADY registered. A single slot sent
        # this one; remembering every hash does not.
        assert t.check_and_update(self._snap("a")) is False
        assert t.check_and_update(self._snap("b")) is False

    def test_a_genuinely_new_hash_still_registers(self):
        """The guard must not become a mute button. An agent that discovers a
        tool mid-run HAS a new manifest and the platform must be told — that is
        the product working, not waste."""
        from decimalai.schema.manifest import ManifestTracker

        t = ManifestTracker()
        assert t.check_and_update(self._snap("a")) is True
        assert t.check_and_update(self._snap("a")) is False
        assert t.check_and_update(self._snap("c")) is True

    def test_the_memory_is_bounded(self):
        """A long-lived worker must not accumulate one entry per config change
        for the life of the process."""
        from decimalai.schema.manifest import ManifestTracker

        t = ManifestTracker()
        for i in range(ManifestTracker._MAX_REMEMBERED + 40):
            t.check_and_update(self._snap(f"h{i}"))
        assert len(t._seen_hashes) <= ManifestTracker._MAX_REMEMBERED
        # And it evicts the OLDEST, so the recent shapes stay suppressed.
        assert t.check_and_update(self._snap(f"h{ManifestTracker._MAX_REMEMBERED + 39}")) is False

    def test_last_manifest_still_reports_what_was_most_recently_offered(self):
        """Callers read `last_manifest` after the check. Suppressing a repeat
        must not leave them looking at a stale shape."""
        from decimalai.schema.manifest import ManifestTracker

        t = ManifestTracker()
        t.check_and_update(self._snap("a"))
        t.check_and_update(self._snap("b"))
        t.check_and_update(self._snap("a"))
        assert t.last_hash == "a"
        assert t.last_manifest is not None and t.last_manifest.manifest_hash == "a"

    def test_reset_forgets_everything_so_a_failed_registration_retries(self):
        """`reset()` is the ROLLBACK the caller uses when registration fails.

        The first version of the remembered-hash set did not clear here, and a
        hash surviving the rollback suppresses the retry — leaving the next trace
        citing a manifest the backend never stored, which is the
        "manifest_id does not exist" 400 this area exists to avoid. Caught by the
        existing failure-retry tests; pinned here so the reason is written down.
        """
        from decimalai.schema.manifest import ManifestTracker

        t = ManifestTracker()
        assert t.check_and_update(self._snap("a")) is True
        assert t.check_and_update(self._snap("a")) is False
        t.reset()
        assert t.check_and_update(self._snap("a")) is True, \
            "reset() left a hash remembered — a failed registration would never retry"


class TestAnEmptyMenuNamesItsFailure:
    """The fleet filed every empty menu as `no_skill_offered` with an empty detail;
    58% of them were Cloud Run edge aborts it could not tell from a slow read."""

    def _router(self):
        from decimalai import skill_router as sr
        from decimalai.skill_router import SkillRouter

        sr._hot_path_breaker = _CircuitBreaker()
        return SkillRouter(api_key="dai_sk_t", base_url="http://127.0.0.1:9")

    def test_a_timeout_is_named(self, monkeypatch):
        def boom(*a, **k):
            raise httpx.ReadTimeout("slow")
        monkeypatch.setattr(httpx, "request", boom)
        assert self._router().smart_route("q")["degraded_reason"] == "timeout"

    def test_an_edge_abort_429_is_named_by_status(self, monkeypatch):
        class _Resp:
            status_code = 429
            headers: dict = {}
            text = "The request was aborted because there was no available instance."
            def json(self):
                return {"detail": self.text}
        monkeypatch.setattr(httpx, "request", lambda *a, **k: _Resp())
        assert self._router().smart_route("q")["degraded_reason"] == "http_429"

    def test_an_open_breaker_is_named(self):
        from decimalai import skill_router as sr

        r = self._router()
        sr._hot_path_breaker = _CircuitBreaker(threshold=1, cooldown_s=30.0)
        sr._hot_path_breaker.record_failure()
        assert r.smart_route("q")["degraded_reason"] == "circuit_open"
