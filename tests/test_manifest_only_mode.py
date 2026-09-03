"""Tests for DECIMALAI_MODE=manifest_only and flush_manifest_for_ci()."""

from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

import decimalai
import decimalai._config as _cfg
from decimalai._config import _is_manifest_only


@pytest.fixture(autouse=True)
def reset_config():
    """Reset SDK global state between tests."""
    yield
    _cfg._config = None
    _cfg._client = None


@pytest.fixture
def clean_env(monkeypatch):
    """Strip DECIMAL_*/GITHUB_* env vars and force hermetic SDK init."""
    for k in list(os.environ.keys()):
        if k.startswith("DECIMAL") or k.startswith("GITHUB_"):
            monkeypatch.delenv(k, raising=False)
    # Otherwise init() runs its default health probe against the configured backend, which has no listener in tests.
    monkeypatch.setenv("DECIMALAI_SKIP_VERIFY", "1")
    return monkeypatch


# ─────────────────────────────────────────────────────────────────────
# Mode detection
# ─────────────────────────────────────────────────────────────────────


class TestManifestOnlyModeDetection:
    """init() reads DECIMALAI_MODE env var and sets the mode flag accordingly."""

    def test_default_mode_is_not_manifest_only(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        decimalai.init()
        assert _is_manifest_only() is False

    def test_manifest_only_env_var_sets_flag(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "manifest_only")
        decimalai.init()
        assert _is_manifest_only() is True

    def test_manifest_only_is_case_insensitive(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "MANIFEST_ONLY")
        decimalai.init()
        assert _is_manifest_only() is True

    def test_unrecognized_mode_value_is_not_manifest_only(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "something_else")
        decimalai.init()
        assert _is_manifest_only() is False

    def test_empty_mode_value_is_not_manifest_only(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "")
        decimalai.init()
        assert _is_manifest_only() is False


# ─────────────────────────────────────────────────────────────────────
# PR context extraction
# ─────────────────────────────────────────────────────────────────────


class TestReadGithubPrContext:
    """_read_github_pr_context derives PR metadata from GitHub Actions env vars."""

    def test_returns_empty_dict_when_no_repo_set(self, clean_env):
        ctx = decimalai._read_github_pr_context()
        assert ctx == {}

    def test_returns_full_context_for_pr_event(self, clean_env):
        clean_env.setenv("GITHUB_REPOSITORY", "acme/support-bot")
        clean_env.setenv("GITHUB_REF", "refs/pull/42/merge")
        clean_env.setenv("GITHUB_HEAD_REF", "fix/refund-prompt")
        clean_env.setenv("GITHUB_SHA", "abc123def456")

        ctx = decimalai._read_github_pr_context()
        assert ctx["repo"] == "acme/support-bot"
        assert ctx["pr_number"] == 42
        assert ctx["branch"] == "fix/refund-prompt"
        assert ctx["commit_sha"] == "abc123def456"

    def test_pr_number_is_none_for_non_pr_event(self, clean_env):
        """Push events to main don't have refs/pull/N — pr_number is None."""
        clean_env.setenv("GITHUB_REPOSITORY", "acme/support-bot")
        clean_env.setenv("GITHUB_REF", "refs/heads/main")
        clean_env.setenv("GITHUB_REF_NAME", "main")

        ctx = decimalai._read_github_pr_context()
        assert ctx["repo"] == "acme/support-bot"
        assert ctx["pr_number"] is None
        assert ctx["branch"] == "main"

    def test_invalid_pr_ref_does_not_raise(self, clean_env):
        """Garbled refs shouldn't crash the SDK."""
        clean_env.setenv("GITHUB_REPOSITORY", "acme/support-bot")
        clean_env.setenv("GITHUB_REF", "refs/pull/not-a-number/merge")

        ctx = decimalai._read_github_pr_context()
        assert ctx["pr_number"] is None


# ─────────────────────────────────────────────────────────────────────
# Manifest ID writing
# ─────────────────────────────────────────────────────────────────────


class TestWriteManifestIdForCi:
    """_write_manifest_id_for_ci writes the manifest_id to the right place."""

    def test_explicit_output_path_takes_precedence(self, clean_env, tmp_path):
        explicit = tmp_path / "out.txt"
        result = decimalai._write_manifest_id_for_ci(
            "mfst_abc123", str(explicit)
        )
        assert result == str(explicit)
        assert explicit.read_text() == "mfst_abc123"

    def test_github_output_used_when_set(self, clean_env, tmp_path):
        gh_out = tmp_path / "github_output"
        gh_out.write_text("")  # GitHub initializes this
        clean_env.setenv("GITHUB_OUTPUT", str(gh_out))

        result = decimalai._write_manifest_id_for_ci("mfst_abc123", None)
        assert result == str(gh_out)
        # GitHub Actions output format: key=value
        assert "decimal_manifest_id=mfst_abc123" in gh_out.read_text()

    def test_github_output_appends_does_not_overwrite(self, clean_env, tmp_path):
        gh_out = tmp_path / "github_output"
        gh_out.write_text("other_step_output=foo\n")
        clean_env.setenv("GITHUB_OUTPUT", str(gh_out))

        decimalai._write_manifest_id_for_ci("mfst_abc123", None)
        contents = gh_out.read_text()
        assert "other_step_output=foo" in contents
        assert "decimal_manifest_id=mfst_abc123" in contents

    def test_fallback_writes_to_local_file(self, clean_env, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = decimalai._write_manifest_id_for_ci("mfst_abc123", None)
        assert result == "decimal_manifest_id.txt"
        assert (tmp_path / "decimal_manifest_id.txt").read_text() == "mfst_abc123"


# ─────────────────────────────────────────────────────────────────────
# flush_manifest_for_ci end-to-end
# ─────────────────────────────────────────────────────────────────────


class TestTraceIngestionBouncer:
    """The HTTP client refuses to send traces when in manifest_only mode.

    This is the "bouncer" that makes the mode actually mean something —
    framework integrations (LangChain/OpenAI Agents/etc.) can fire callbacks
    freely; the client just won't POST anything to /api/v1/traces.

    These tests are the safety net for the most embarrassing class of bug:
    a customer's CI run accidentally pollutes their production trace store
    because their init script invoked the agent during build.
    """

    def _make_client(self):
        from decimalai._client import DecimalAIClient
        return DecimalAIClient(
            api_key="dai_sk_test",
            project="test",
            base_url="http://localhost:8000",
        )

    def test_ingest_trace_is_noop_in_manifest_only(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "manifest_only")
        decimalai.init()

        client = self._make_client()
        # Use a minimal RunTrace-like mock so we don't depend on the schema
        from unittest.mock import MagicMock
        fake_trace = MagicMock()
        fake_trace.id = "trace_xyz"

        # Patch the underlying HTTP to ensure we'd see if a request went out
        with patch.object(client, "_request_with_retry") as request_mock:
            result = client.ingest_trace(fake_trace)

        assert result == {"status": "skipped", "reason": "manifest_only_mode"}
        request_mock.assert_not_called()

    def test_ingest_traces_batch_is_noop_in_manifest_only(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "manifest_only")
        decimalai.init()

        client = self._make_client()
        from unittest.mock import MagicMock
        traces = [MagicMock(), MagicMock(), MagicMock()]

        with patch.object(client, "_request_with_retry") as request_mock:
            result = client.ingest_traces_batch(traces)

        assert result["status"] == "skipped"
        assert result["skipped_count"] == 3
        request_mock.assert_not_called()

    def test_ingest_raw_trace_is_noop_in_manifest_only(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "manifest_only")
        decimalai.init()

        client = self._make_client()
        with patch.object(client, "_request_with_retry") as request_mock:
            result = client.ingest_raw_trace({"agent_name": "x", "status": "success"})

        assert result["status"] == "skipped"
        request_mock.assert_not_called()

    def test_buffer_trace_is_noop_in_manifest_only(self, clean_env):
        """The buffer must also short-circuit — otherwise framework integrations
        could fill memory with traces that get dropped at flush time."""
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "manifest_only")
        decimalai.init()

        client = self._make_client()
        from unittest.mock import MagicMock
        fake_trace = MagicMock()

        # Push 100 traces; under normal mode this would trigger auto-flush at 50
        for _ in range(100):
            client.buffer_trace(fake_trace)

        # Buffer should remain empty — we never queued anything
        assert client._trace_buffer == []

    def test_normal_mode_still_sends_traces(self, clean_env):
        """Sanity check: when not in manifest_only mode, traces are sent."""
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        # Explicitly DO NOT set DECIMALAI_MODE
        decimalai.init()

        client = self._make_client()
        from unittest.mock import MagicMock
        fake_trace = MagicMock()
        fake_trace.id = "trace_xyz"
        fake_trace.model_dump.return_value = {"id": "trace_xyz"}

        with patch.object(client, "_request_with_retry") as request_mock:
            request_mock.return_value.json.return_value = {"ok": True}
            client.ingest_trace(fake_trace)

        # In normal mode, the HTTP call happens
        request_mock.assert_called_once()

    def test_manifest_registration_still_works_in_manifest_only_mode(self, clean_env):
        """Critical: while traces are blocked, MANIFEST registration must still
        work in manifest_only mode — that's the whole point of the mode."""
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "manifest_only")
        decimalai.init()

        client = self._make_client()
        from unittest.mock import MagicMock
        manifest = MagicMock()
        manifest.id = "mfst_xyz"
        manifest.manifest_hash = "h_abc"
        manifest.model_dump.return_value = {"id": "mfst_xyz"}

        # Patch the TRANSPORT (`_http.request`), not `_http.post`.
        #
        # This used to patch `_http.post` with the comment "register_manifest
        # uses _http.post directly" — which pinned the implementation, not the
        # behaviour. When `register_manifest` moved onto the `_request_with_retry`
        # ladder (so a 429 from someone else's CI no longer fails their build),
        # this test broke with a real ConnectError while the behaviour it claims
        # to protect was perfectly intact. `_http.request` is the seam BELOW the
        # retry logic, so the assertions below now hold across either routing —
        # and `test_strict_manifest_warning.py` already mocks at this level.
        with patch.object(client._http, "request") as req_mock:
            req_mock.return_value.status_code = 200
            req_mock.return_value.is_success = True
            req_mock.return_value.json.return_value = {"manifest_id": "mfst_xyz"}
            req_mock.return_value.raise_for_status = lambda: None
            client.register_manifest(manifest)

        # The manifest endpoint WAS called — only traces are blocked, not manifests
        req_mock.assert_called_once()
        method, url = req_mock.call_args[0][0], req_mock.call_args[0][1]
        assert method == "POST"
        assert "/api/v1/manifests" in url

    def test_manifest_registration_retries_a_429_instead_of_failing_the_build(
        self, clean_env
    ):
        """`flush_manifest_for_ci` runs in other people's CI. A 429 must not end it.

        `register_manifest` used a bare `_http.post`, which has no retry ladder,
        so a single rate-limited response surfaced as an unhandled
        DecimalAPIError on the FIRST call of a CI run. decimal-labs's own
        regression-check dogfood job failed exactly that way on two consecutive
        runs (2026-09-03) while the backend was shedding load.

        Fails against the pre-fix bare-post implementation.
        """
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "manifest_only")
        decimalai.init()

        client = self._make_client()
        from unittest.mock import MagicMock

        manifest = MagicMock()
        manifest.id = "mfst_xyz"
        manifest.manifest_hash = "h_abc"
        manifest.model_dump.return_value = {"id": "mfst_xyz"}

        throttled = MagicMock()
        throttled.status_code = 429
        throttled.is_success = False
        throttled.headers = {"Retry-After": "0"}

        accepted = MagicMock()
        accepted.status_code = 200
        accepted.is_success = True
        accepted.json.return_value = {"manifest_id": "mfst_xyz"}
        accepted.raise_for_status = lambda: None

        with patch.object(
            client._http, "request", side_effect=[throttled, accepted]
        ) as req_mock:
            result = client.register_manifest(manifest)

        assert req_mock.call_count == 2, (
            "a 429 on manifest registration must be retried, not raised — a bare "
            "post fails the caller's build on its first request"
        )
        assert result["manifest_id"] == "mfst_xyz"


class TestFlushManifestForCi:
    """flush_manifest_for_ci wires everything together."""

    def test_uploads_manifest_and_writes_id(self, clean_env, tmp_path, monkeypatch):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        clean_env.setenv("DECIMALAI_MODE", "manifest_only")
        clean_env.setenv("GITHUB_REPOSITORY", "acme/support-bot")
        clean_env.setenv("GITHUB_REF", "refs/pull/7/merge")
        clean_env.setenv("GITHUB_HEAD_REF", "fix/refund")
        clean_env.setenv("GITHUB_SHA", "abc123")
        gh_out = tmp_path / "github_output"
        gh_out.write_text("")
        clean_env.setenv("GITHUB_OUTPUT", str(gh_out))

        decimalai.init()

        # Mock the client so we don't need a real backend
        mock_client = MagicMock()
        mock_client.register_manifest.return_value = {
            "manifest_id": "mfst_xyz",
            "manifest_hash": "h_abc",
        }
        _cfg._client = mock_client

        result = decimalai.flush_manifest_for_ci(
            agent_name="support-agent",
            tools=[{"name": "search", "schema": {"type": "object"}}],
            prompts={"system": "You are helpful."},
            models={"default": {"provider": "openai", "model": "gpt-4o"}},
        )

        assert result["manifest_id"] == "mfst_xyz"
        assert result["pr_context"]["repo"] == "acme/support-bot"
        assert result["pr_context"]["pr_number"] == 7
        assert result["output_path"] == str(gh_out)
        assert "decimal_manifest_id=mfst_xyz" in gh_out.read_text()

        # Confirm the upload was actually attempted
        assert mock_client.register_manifest.called

    def test_raises_if_registration_returns_no_manifest_id(self, clean_env):
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        decimalai.init()

        mock_client = MagicMock()
        mock_client.register_manifest.return_value = {"status": "error"}
        _cfg._client = mock_client

        with pytest.raises(RuntimeError, match="manifest_id"):
            decimalai.flush_manifest_for_ci(
                agent_name="support-agent",
                tools=[{"name": "x"}],
            )

    def test_works_outside_github_context(self, clean_env, tmp_path, monkeypatch):
        """If not running under GitHub Actions, falls back to local file."""
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        decimalai.init()
        monkeypatch.chdir(tmp_path)

        mock_client = MagicMock()
        mock_client.register_manifest.return_value = {"manifest_id": "mfst_local"}
        _cfg._client = mock_client

        result = decimalai.flush_manifest_for_ci(
            agent_name="support-agent",
            tools=[{"name": "x"}],
        )

        assert result["pr_context"] == {}  # no GitHub context
        assert result["output_path"] == "decimal_manifest_id.txt"
        assert (tmp_path / "decimal_manifest_id.txt").read_text() == "mfst_local"

    def test_extracts_from_langchain_chain(self, clean_env, tmp_path, monkeypatch):
        """Pass a LangChain chain object; introspection auto-fills tools/prompts/models."""
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        decimalai.init()
        monkeypatch.chdir(tmp_path)

        from langchain_core.prompts import PromptTemplate
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        class SearchArgs(BaseModel):
            query: str

        search = StructuredTool.from_function(
            func=lambda query: query,
            name="search_docs",
            description="search",
            args_schema=SearchArgs,
        )

        class FakeLLM:
            model_name = "gpt-4o"
            temperature = 0.1

        class FakeChain:
            def __init__(self):
                self.tools = [search]
                self.llm = FakeLLM()
                self.prompt = PromptTemplate.from_template("Be concise.")

        captured = {}
        mock_client = MagicMock()

        def capture_register(snapshot):
            # snapshot is the ManifestSnapshot pydantic model
            captured["agent_name"] = snapshot.agent_name
            captured["components"] = [
                (c.component_type, c.component_name) for c in snapshot.components
            ]
            return {"manifest_id": "mfst_lc"}

        mock_client.register_manifest.side_effect = capture_register
        _cfg._client = mock_client

        result = decimalai.flush_manifest_for_ci(
            agent_name="support-agent",
            chain=FakeChain(),
        )

        assert result["manifest_id"] == "mfst_lc"
        assert captured["agent_name"] == "support-agent"
        # The introspection found the tool, the model, and the prompt
        component_names = {name for _, name in captured["components"]}
        component_types = {ctype for ctype, _ in captured["components"]}
        assert "search_docs" in component_names
        assert "tool" in component_types
        assert "model" in component_types
        assert "prompt" in component_types

    def test_explicit_args_override_introspection(self, clean_env, tmp_path, monkeypatch):
        """If both chain and explicit tools are provided, explicit wins.

        This is the escape hatch when introspection picks up the wrong thing
        and the customer wants to specify explicitly anyway.
        """
        clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
        decimalai.init()
        monkeypatch.chdir(tmp_path)

        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel

        class A(BaseModel):
            x: str

        chain_tool = StructuredTool.from_function(
            func=lambda x: x, name="from_chain", description="x", args_schema=A,
        )

        class FakeChain:
            tools = [chain_tool]
            llm = None
            prompt = None

        captured = {}

        def capture(snapshot):
            captured["names"] = [c.component_name for c in snapshot.components if c.component_type == "tool"]
            return {"manifest_id": "mfst_x"}

        mock_client = MagicMock()
        mock_client.register_manifest.side_effect = capture
        _cfg._client = mock_client

        decimalai.flush_manifest_for_ci(
            agent_name="x",
            chain=FakeChain(),
            tools=[{"name": "explicit_tool", "schema": {}}],
        )

        assert "explicit_tool" in captured["names"]
        assert "from_chain" not in captured["names"]
