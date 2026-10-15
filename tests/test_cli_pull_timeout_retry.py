"""Tests for the anonymous pull path's timeout budget + single retry.

`decimalai skills pull` talks to the registry anonymously, and a cold prod
instance has served first requests in 9-28s. The old flat 20s timeout killed
pulls the server went on to complete. The bundle comes through the skills
engine's registry client (30s per request, one retry on timeout); the reads
beside it (eval.yaml, the quality line) go through `_pull_get`, which budgets
`_PULL_HTTP_TIMEOUT` per attempt and retries exactly once on timeout.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from decimalai.cli.main import cli
from decimalai.cli.skills_pm import _PULL_HTTP_TIMEOUT, _pull_get
from decimalai.skillpkg.client import DEFAULT_TIMEOUT, RegistryClient
from tests.fakes.fake_registry import BASE_URL, FakeRegistry


def _resp(payload, status_code=200):
    resp = MagicMock(status_code=status_code)
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


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
    """`skills pull` end-to-end with a timeout injected on the bundle request."""

    @pytest.fixture
    def registry(self, monkeypatch):
        """The fake registry in process: every client the CLI builds sends through it."""
        registry = FakeRegistry()
        registry.add_skill("slow-skill", body="# Slow\n\nbody.\n")
        transport = registry.transport()
        real = RegistryClient.from_env

        def from_env(*args, **kwargs):
            kwargs.setdefault("transport", transport)
            kwargs.setdefault("sleep", lambda _s: None)
            return real(*args, **kwargs)

        monkeypatch.setattr(RegistryClient, "from_env", from_env)
        return registry

    def _pull(self, tmp_path):
        return CliRunner().invoke(cli, ["skills", "pull", "slow-skill", "--out", str(tmp_path),
                                        "--no-evals", "--base-url", BASE_URL])

    def test_the_bundle_request_runs_on_the_cold_budget(self, registry, tmp_path):
        result = self._pull(tmp_path)
        assert result.exit_code == 0, result.output
        (call,) = registry.calls("registry_bundle")
        assert call.timeout["read"] == DEFAULT_TIMEOUT == _PULL_HTTP_TIMEOUT

    def test_pull_survives_one_cold_start_timeout(self, registry, tmp_path):
        registry.fault("registry_bundle", "timeout")
        result = self._pull(tmp_path)
        assert result.exit_code == 0, result.output
        assert (tmp_path / "slow-skill" / "SKILL.md").exists()
        assert len(registry.calls("registry_bundle")) == 2

    def test_pull_fails_cleanly_when_retry_also_times_out(self, registry, tmp_path):
        registry.fault("registry_bundle", "timeout", "timeout")
        result = self._pull(tmp_path)
        assert result.exit_code == 8
        assert "did not answer within 30s" in result.output
        assert "Traceback" not in result.output
        assert not (tmp_path / "slow-skill").exists()
