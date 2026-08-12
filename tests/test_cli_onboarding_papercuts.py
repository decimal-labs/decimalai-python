"""CLI onboarding papercuts — the small first-run failures that read as
"the tool is broken" when the real cause is a key, a payload key name, or a
truncated help example.

* Init error triage: `init` with a rejected key must say "Invalid API key" +
  the settings URL — NOT "Connection failed" + an MDN link. "Connection failed"
  is reserved for transport errors (DNS/refused/timeout).
* Shared no-key error: the message used by the demo doors and every keyed
  command points at the settings page, like init's no-key error does.
* Version column: `skills list` reads the version from
  `latest_version.version_number` (the key the endpoint actually serializes)
  instead of printing "v?".
* Bundled attachments: `skills pull` downloads the attachments the body
  references via the public registry attachment endpoints.
* Help papercuts: unmangled pull examples, complete one-liners, and a
  "start here" hint on the bare invocation.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner

from decimalai.cli.main import cli


def _no_keys(monkeypatch):
    for k in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


# ── init triage: auth rejection vs transport failure ──


class TestInitErrorTriage:
    def _client_with_status(self, status):
        client = MagicMock()
        resp = MagicMock()
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status} error",
            request=MagicMock(),
            response=MagicMock(status_code=status),
        )
        client._http.get.return_value = resp
        return client

    def test_401_says_invalid_key_not_connection_failed(self):
        client = self._client_with_status(401)
        with patch("decimalai._client.DecimalAIClient", return_value=client):
            result = CliRunner().invoke(cli, [
                "init", "--api-key", "dai_sk_bad",
                "--base-url", "https://api.decimal.ai", "--no-test-trace",
            ])
        assert result.exit_code == 1
        assert "Invalid API key" in result.output
        assert "https://app.decimal.ai/settings" in result.output
        assert "Connection failed" not in result.output

    def test_403_takes_the_same_branch(self):
        client = self._client_with_status(403)
        with patch("decimalai._client.DecimalAIClient", return_value=client):
            result = CliRunner().invoke(cli, [
                "init", "--api-key", "dai_sk_bad",
                "--base-url", "https://api.decimal.ai", "--no-test-trace",
            ])
        assert result.exit_code == 1
        assert "Invalid API key" in result.output

    def test_other_http_status_is_named_not_connection_failed(self):
        client = self._client_with_status(500)
        with patch("decimalai._client.DecimalAIClient", return_value=client):
            result = CliRunner().invoke(cli, [
                "init", "--api-key", "dai_sk_x",
                "--base-url", "https://api.decimal.ai", "--no-test-trace",
            ])
        assert result.exit_code == 1
        assert "HTTP 500" in result.output
        assert "Connection failed" not in result.output

    def test_connection_refused_still_says_connection_failed(self):
        client = MagicMock()
        client._http.get.side_effect = httpx.ConnectError("connection refused")
        with patch("decimalai._client.DecimalAIClient", return_value=client):
            result = CliRunner().invoke(cli, [
                "init", "--api-key", "dai_sk_x",
                "--base-url", "http://localhost:9",  # nothing listens here
                "--no-test-trace",
            ])
        assert result.exit_code == 1
        assert "Connection failed" in result.output
        assert "Invalid API key" not in result.output


class TestSdkVerifyHelper:
    """The SDK-side probe (`_verify_backend_at_init`) mirrors the triage."""

    def test_401_message_names_the_settings_page(self):
        import urllib.error
        from email.message import Message

        from decimalai import _verify_backend_at_init
        from decimalai._config import DecimalConfigError

        err = urllib.error.HTTPError(
            "https://api.decimal.ai/api/v1/auth/verify", 401, "Unauthorized",
            Message(), None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
            try:
                _verify_backend_at_init(
                    base_url="https://api.decimal.ai",
                    api_key="dai_sk_bad",
                    timeout=1.0,
                )
                raise AssertionError("expected DecimalConfigError")
            except DecimalConfigError as exc:
                assert "Invalid API key" in str(exc)
                assert "https://app.decimal.ai/settings" in str(exc)


# ── shared no-key error points at settings ─────────────────


class TestNoKeySettingsHint:
    def test_demo_skills_no_key_error_names_settings_url(self, monkeypatch):
        _no_keys(monkeypatch)
        result = CliRunner().invoke(cli, [
            "demo", "skills", "--base-url", "https://api.decimal.ai",
        ])
        assert result.exit_code == 1
        assert "Get one at https://app.decimal.ai/settings" in result.output

    def test_local_base_url_maps_to_local_dashboard(self, monkeypatch):
        _no_keys(monkeypatch)
        result = CliRunner().invoke(cli, [
            "demo", "regression", "--base-url", "http://localhost:8000",
        ])
        assert result.exit_code == 1
        assert "Get one at http://localhost:3000/settings" in result.output


# ── skills list version column ──────────────────────────────


class TestSkillsListVersion:
    def _client(self, skills):
        client = MagicMock()
        resp = MagicMock()
        resp.json.return_value = {"skills": skills}
        resp.raise_for_status = MagicMock()
        client._http.get.return_value = resp
        return client

    def test_reads_latest_version_version_number(self):
        client = self._client([
            {"name": "pdf", "latest_version": {"version_number": 3}},
            {"name": "code-review", "latest_version": {"version_number": 1}},
        ])
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "skills", "list", "--api-key", "k",
                "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 0, result.output
        assert "v3" in result.output
        assert "v1" in result.output
        assert "v?" not in result.output

    def test_missing_version_still_prints_placeholder(self):
        client = self._client([{"name": "pdf"}])
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "skills", "list", "--api-key", "k",
                "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 0, result.output
        assert "v?" in result.output


# ── pull delivers bundled attachments ───────────────────────


def _resp(payload=None, status=200):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    if status >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status}", request=MagicMock(), response=MagicMock(status_code=status),
        )
    else:
        r.raise_for_status = MagicMock()
    return r


class TestPullAttachments:
    DETAIL = {
        "id": "sk1",
        "name": "pdf",
        "url_slug": "pdf",
        "description": "PDF skill",
        "body_markdown": "# PDF\n\nRun scripts/extract.py first.",
        "latest_version_number": 1,
        "attachment_count": 1,
        "effectiveness": {},
        "benchmark_summary": {},
    }

    def _fake_get(self, atts_list_resp, att_detail_resp):
        def fake_get(url, **kwargs):
            if url.endswith("/api/v1/registry/skills"):
                return _resp({"items": [{"id": "sk1", "name": "pdf"}]})
            if url.endswith("/registry/skills/sk1"):
                return _resp(self.DETAIL)
            if url.endswith("/sk1/attachments"):
                return atts_list_resp
            if url.endswith("/sk1/attachments/a1"):
                return att_detail_resp
            if url.endswith("/sk1/eval"):
                return _resp(status=404)
            raise AssertionError(f"unexpected GET {url}")
        return fake_get

    def test_attachments_written_next_to_skill_md(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake = self._fake_get(
            _resp({"attachments": [{"id": "a1", "file_path": "scripts/extract.py"}]}),
            _resp({"id": "a1", "file_path": "scripts/extract.py",
                   "content_text": "print('extract')"}),
        )
        with patch("httpx.get", side_effect=fake):
            result = CliRunner().invoke(cli, ["skills", "pull", "pdf"])
        assert result.exit_code == 0, result.output
        script = tmp_path / "pdf" / "scripts" / "extract.py"
        assert script.read_text() == "print('extract')"
        assert "1 bundled file(s)" in result.output
        assert "not included in pull" not in result.output

    def test_undeliverable_attachments_are_named_not_silent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake = self._fake_get(_resp(status=500), _resp(status=500))
        with patch("httpx.get", side_effect=fake):
            result = CliRunner().invoke(cli, ["skills", "pull", "pdf"])
        assert result.exit_code == 0, result.output
        assert "references 1 bundled file(s) not included in pull" in result.output

    def test_traversal_attachment_path_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fake = self._fake_get(
            _resp({"attachments": [
                {"id": "a1", "file_path": "../../evil.sh", "content_text": "boom"},
            ]}),
            _resp({}),
        )
        with patch("httpx.get", side_effect=fake):
            result = CliRunner().invoke(cli, ["skills", "pull", "pdf"])
        assert result.exit_code == 0, result.output
        assert not (tmp_path.parent / "evil.sh").exists()
        assert not (tmp_path.parent.parent / "evil.sh").exists()
        assert "skipping bundled file" in result.output


# ── click help papercuts ────────────────────────────────────


class TestHelpPapercuts:
    def test_pull_examples_render_unmangled(self):
        result = CliRunner().invoke(cli, ["skills", "pull", "--help"])
        assert result.exit_code == 0
        # Pre-fix, click re-wrapped the 2nd/3rd example paragraphs into
        # "# Write to a specific path     $ decimalai skills pull pdf --out".
        assert "$ decimalai skills pull pdf --out ./agents/skills/" in result.output
        assert "$ decimalai skills pull pdf --stdout" in result.output

    def test_top_level_one_liners_do_not_truncate(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Check dataset compatibility after a manifest change." in result.output
        assert "Manage evaluators — deterministic checks and LLM judges." in result.output
        assert "Preview and apply mechanical trace repairs." in result.output

    def test_bare_invocation_prints_start_hint(self):
        result = CliRunner().invoke(cli, [])
        # Click prints the group help on a bare invocation and exits 2 —
        # that exit code is kept as-is; the hint rides the help epilog.
        assert result.exit_code == 2
        assert "Start with: decimalai init" in result.output

    def test_help_flag_also_shows_start_hint(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Start with: decimalai init" in result.output
