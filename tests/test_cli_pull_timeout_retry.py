"""Tests for the anonymous pull path's timeout budget + single retry.

`decimalai skills pull` talks to the registry anonymously — plain
`httpx.get`, no client, no connection reuse — and a cold prod instance
has served first requests in 9-28s. The old flat 20s timeout killed
pulls the server went on to complete. Every pull-path request now
routes through `_pull_get`, which budgets `_PULL_HTTP_TIMEOUT` per
attempt and retries exactly once on timeout.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from decimalai.cli.main import _PULL_HTTP_TIMEOUT, _pull_get, cli


def _resp(payload, status_code=200):
    resp = MagicMock(status_code=status_code)
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


_SEARCH = {"items": [{"id": "sk-timeout-001", "name": "slow-skill"}]}
_DETAIL = {
    "id": "sk-timeout-001",
    "name": "slow-skill",
    "body_markdown": "# Slow\n\nbody.",
    "latest_version_number": 1,
}


class TestPullGetRetry:
    """The `_pull_get` transport helper in isolation."""

    def test_retries_once_on_timeout(self):
        good = _resp({})
        with patch(
            "httpx.get", side_effect=[httpx.ReadTimeout("cold start"), good]
        ) as get:
            resp = _pull_get("https://api.example.test/api/v1/registry/skills")
        assert resp is good
        assert get.call_count == 2
        # Both attempts carry the enlarged budget (the old flat 20s was
        # inside the measured 9-28s cold-start band).
        for call in get.call_args_list:
            assert call.kwargs["timeout"] == _PULL_HTTP_TIMEOUT

    def test_second_timeout_propagates(self):
        with patch(
            "httpx.get",
            side_effect=[httpx.ReadTimeout("cold"), httpx.ReadTimeout("still cold")],
        ) as get:
            with pytest.raises(httpx.ReadTimeout):
                _pull_get("https://api.example.test/api/v1/registry/skills")
        # Exactly one retry — never a loop against a dead backend.
        assert get.call_count == 2

    def test_connect_timeout_also_retried(self):
        # ConnectTimeout subclasses TimeoutException; the retry covers the
        # whole timeout family, not just read timeouts.
        good = _resp({})
        with patch(
            "httpx.get", side_effect=[httpx.ConnectTimeout("handshake"), good]
        ) as get:
            assert _pull_get("https://api.example.test/x") is good
        assert get.call_count == 2

    def test_non_timeout_transport_error_not_retried(self):
        # A refused connection fails the same way twice — retrying only
        # delays the error message.
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")) as get:
            with pytest.raises(httpx.ConnectError):
                _pull_get("https://api.example.test/x")
        assert get.call_count == 1


class TestPullCommandColdStart:
    """`skills pull` end-to-end with a timeout injected mid-sequence."""

    def test_pull_survives_one_cold_start_timeout(self, tmp_path):
        with patch(
            "httpx.get",
            side_effect=[
                httpx.ReadTimeout("cold start"),  # search, first attempt
                _resp(_SEARCH),                   # search, retry
                _resp(_DETAIL),                   # detail
                _resp({}, status_code=404),       # eval — none authored
            ],
        ):
            result = CliRunner().invoke(
                cli, ["skills", "pull", "slow-skill", "--out", str(tmp_path)]
            )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "slow-skill" / "SKILL.md").exists()

    def test_pull_fails_cleanly_when_retry_also_times_out(self, tmp_path):
        with patch(
            "httpx.get",
            side_effect=[httpx.ReadTimeout("cold"), httpx.ReadTimeout("dead")],
        ):
            result = CliRunner().invoke(
                cli, ["skills", "pull", "slow-skill", "--out", str(tmp_path)]
            )
        assert result.exit_code == 1
        assert "Registry lookup failed" in result.output
