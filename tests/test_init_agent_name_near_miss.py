"""Near-miss agent-name warning at init()."""
import logging
import pytest
from decimalai import _edit_distance_within, _warn_on_near_miss_agent_name


class TestEditDistance:
    @pytest.mark.parametrize("a,b,ok", [
        ("refund-bot", "refund_bot", True),    # the real-world typo
        ("refund-bot", "refund-bots", True),
        ("refundbot",  "refund-bot", True),
        ("refund-bot", "refund-bot", True),
        ("refund-bot", "billing-agent", False),
        ("a", "abcdefgh", False),
    ])
    def test_bounded(self, a, b, ok):
        assert _edit_distance_within(a, b, 2) is ok


class TestWarning:
    def _run(self, monkeypatch, caplog, names, agent_name):
        import decimalai as d
        import json, io

        class _Resp:
            status = 200
            def read(self): return json.dumps(
                {"agents": [{"agent_name": n} for n in names]}).encode()
            def __enter__(self): return self
            def __exit__(self, *a): return False

        monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            _warn_on_near_miss_agent_name(
                base_url="http://x", api_key="k",
                agent_name=agent_name, timeout=1.0)
        return caplog.text

    def test_typo_warns_and_names_the_suggestion(self, monkeypatch, caplog):
        out = self._run(monkeypatch, caplog, ["refund-bot", "billing"], "refund_bot")
        assert "refund_bot" in out and "refund-bot" in out

    def test_exact_match_is_silent(self, monkeypatch, caplog):
        assert self._run(monkeypatch, caplog, ["refund-bot"], "refund-bot") == ""

    def test_genuinely_new_name_is_silent(self, monkeypatch, caplog):
        """The normal first-run case must not warn, or the signal is noise."""
        assert self._run(monkeypatch, caplog, ["billing", "support"], "refund-bot") == ""

    def test_empty_workspace_is_silent(self, monkeypatch, caplog):
        assert self._run(monkeypatch, caplog, [], "refund-bot") == ""

    def test_network_failure_is_silent(self, monkeypatch, caplog):
        def boom(*a, **k): raise OSError("no route to host")
        monkeypatch.setattr("urllib.request.urlopen", boom)
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            _warn_on_near_miss_agent_name(
                base_url="http://x", api_key="k", agent_name="refund_bot", timeout=1.0)
        assert caplog.text == ""
