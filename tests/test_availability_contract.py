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
    _HOT_PATH_TIMEOUT,
    SkillRouter,
    _CircuitBreaker,
    _is_hot_path,
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
        assert _HOT_PATH_TIMEOUT.read is not None
        assert _HOT_PATH_TIMEOUT.read <= 5.0
        assert _COLD_PATH_TIMEOUT == 30.0

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
