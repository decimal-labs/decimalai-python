"""Tests for the `decimalai regression-check` CLI command."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from decimalai.cli.main import (
    _render_regression_github_annotations,
    _render_regression_terminal,
    _resolve_candidate_manifest_id,
    _should_fail,
    cli,
)


# ─────────────────────────────────────────────────────────────────────
# Pure function tests
# ─────────────────────────────────────────────────────────────────────


class TestShouldFail:
    """_should_fail maps verdict + threshold to a binary exit decision."""

    def test_high_verdict_with_high_threshold_fails(self):
        assert _should_fail("high_risk", "high") is True

    def test_high_verdict_with_medium_threshold_fails(self):
        assert _should_fail("high_risk", "medium") is True

    def test_high_verdict_with_none_threshold_does_not_fail(self):
        assert _should_fail("high_risk", "none") is False

    def test_medium_verdict_with_high_threshold_does_not_fail(self):
        assert _should_fail("medium_risk", "high") is False

    def test_medium_verdict_with_medium_threshold_fails(self):
        assert _should_fail("medium_risk", "medium") is True

    def test_low_verdict_never_fails_at_default_thresholds(self):
        assert _should_fail("low_risk", "high") is False
        assert _should_fail("low_risk", "medium") is False

    def test_no_change_never_fails(self):
        assert _should_fail("no_change", "high") is False

    def test_first_run_never_fails(self):
        assert _should_fail("first_run", "high") is False
        assert _should_fail("first_run", "medium") is False


class TestResolveCandidateManifestId:
    """_resolve_candidate_manifest_id reads the ID from the right place."""

    def test_returns_none_when_nothing_set(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        for k in ("GITHUB_OUTPUT",):
            monkeypatch.delenv(k, raising=False)
        assert _resolve_candidate_manifest_id() is None

    def test_reads_from_github_output_when_set(self, tmp_path, monkeypatch):
        gh_out = tmp_path / "github_output"
        gh_out.write_text(
            "other_step=foo\n"
            "decimal_manifest_id=mfst_xyz\n"
            "another_step=bar\n"
        )
        monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
        assert _resolve_candidate_manifest_id() == "mfst_xyz"

    def test_reads_from_local_file_when_no_github_output(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        Path(tmp_path / "decimal_manifest_id.txt").write_text("mfst_local")
        assert _resolve_candidate_manifest_id() == "mfst_local"

    def test_github_output_takes_precedence_over_local_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        Path(tmp_path / "decimal_manifest_id.txt").write_text("mfst_local")
        gh_out = tmp_path / "github_output"
        gh_out.write_text("decimal_manifest_id=mfst_gh\n")
        monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
        assert _resolve_candidate_manifest_id() == "mfst_gh"


# ─────────────────────────────────────────────────────────────────────
# Output rendering tests
# ─────────────────────────────────────────────────────────────────────


class TestRenderTerminal:
    """Terminal output renders the impact report in a human-readable form."""

    def _sample_report(self, verdict="high_risk", **overrides):
        report = {
            "id": "rc_123",
            "verdict": verdict,
            "verdict_message": "247 traces will break.",
            "high_risk_count": 247,
            "medium_risk_count": 89,
            "low_risk_count": 1254,
            "total_traces_analyzed": 2002,
            "diff_summary": {
                "changes": [
                    {"type": "tool_removed", "name": "compare_competitors", "severity": "high"},
                ],
            },
        }
        report.update(overrides)
        return report

    def test_terminal_output_includes_verdict_and_counts(self, capsys):
        _render_regression_terminal(self._sample_report(), "support-agent", "https://api.decimal.ai")
        out = capsys.readouterr().out
        assert "247" in out
        assert "support-agent" in out
        assert "HIGH RISK" in out

    def test_first_run_renders_friendly_message(self, capsys):
        report = {
            "verdict": "first_run",
            "verdict_message": "First run for this agent. Recorded baseline.",
        }
        _render_regression_terminal(report, "support-agent", "https://api.decimal.ai")
        out = capsys.readouterr().out
        assert "First run" in out
        # Should NOT show the impact bars on first run
        assert "HIGH RISK" not in out

    def test_dashboard_link_uses_app_subdomain(self, capsys):
        _render_regression_terminal(
            self._sample_report(), "support-agent", "https://api.decimal.ai"
        )
        out = capsys.readouterr().out
        assert "https://app.decimal.ai" in out
        assert "support-agent/regression/rc_123" in out

    def test_terminal_shows_grade_and_training_data_policy(self, capsys):
        # Parity with the PR comment + dashboard: graded backends carry
        # detail.grade + detail.policy on model/prompt changes.
        report = self._sample_report(diff_summary={"changes": [
            {"type": "model_changed", "name": "default", "severity": "medium",
             "detail": {"grade": "moderate", "change_kind": "version_bump",
                        "old_model": "gpt-4o-2024-05-13", "new_model": "gpt-4o-mini",
                        "policy": {"name": "default", "disposition": "flag", "implies": "warn"}}},
            {"type": "prompt_section_rewritten", "name": "system", "severity": "medium",
             "detail": {"grade": "major", "diff_pct": 88.1,
                        "policy": {"name": "default", "disposition": "replay", "implies": "warn"}}},
        ]})
        _render_regression_terminal(report, "support-agent", "https://api.decimal.ai")
        out = capsys.readouterr().out
        # grade inline on the change lines
        assert "moderate" in out and "version bump" in out
        assert "gpt-4o-2024-05-13 → gpt-4o-mini" in out
        assert "major" in out and "88.1% changed" in out
        # separate training-data policy block
        assert "Training-data policy (default)" in out
        assert "→ flag" in out and "→ replay" in out

    def test_terminal_omits_policy_block_on_older_backend(self, capsys):
        # No detail on the change (pre-graded backend) → no grade, no policy
        # block, no crash.
        _render_regression_terminal(
            self._sample_report(), "support-agent", "https://api.decimal.ai"
        )
        out = capsys.readouterr().out
        assert "Training-data policy" not in out


class TestRenderGithubAnnotations:
    """GitHub Actions annotations format: ::level title=...::message."""

    def _sample(self, verdict="high_risk", message="247 traces"):
        return {
            "verdict": verdict, "verdict_message": message,
            "high_risk_count": 247, "medium_risk_count": 0, "low_risk_count": 0,
        }

    def test_high_risk_emits_error_annotation(self, capsys):
        _render_regression_github_annotations(self._sample("high_risk"), "support-agent")
        out = capsys.readouterr().out
        assert out.startswith("::error")
        assert "support-agent" in out
        assert "247" in out

    def test_medium_risk_emits_warning_annotation(self, capsys):
        _render_regression_github_annotations(self._sample("medium_risk", "may behave"), "x")
        out = capsys.readouterr().out
        assert out.startswith("::warning")

    def test_first_run_emits_notice(self, capsys):
        _render_regression_github_annotations(self._sample("first_run", "First run"), "x")
        out = capsys.readouterr().out
        assert out.startswith("::notice")
        assert "First run" in out

    def test_no_change_emits_notice(self, capsys):
        _render_regression_github_annotations(self._sample("no_change", "Safe"), "x")
        out = capsys.readouterr().out
        assert out.startswith("::notice")


# ─────────────────────────────────────────────────────────────────────
# CLI command end-to-end (using Click's CliRunner)
# ─────────────────────────────────────────────────────────────────────


class TestRegressionCheckCommand:
    """End-to-end tests of the `decimal regression-check` command."""

    def _patch_client(self, mock_response):
        """Returns a patcher that replaces _make_client with a mock returning mock_response."""
        mock_client = MagicMock()
        mock_client.run_regression_check.return_value = mock_response
        return patch(
            "decimalai.cli.main._make_client",
            return_value=mock_client,
        )

    def test_missing_agent_name_errors(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["regression-check", "--candidate-manifest-id", "x", "--api-key", "k"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "agent-name" in result.output.lower()

    def test_no_manifest_id_and_no_file_errors(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "regression-check", "--agent-name", "support-agent", "--api-key", "k",
        ])
        assert result.exit_code == 2
        assert "candidate manifest" in result.output.lower()

    def test_high_risk_verdict_exits_nonzero_by_default(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        with self._patch_client({
            "verdict": "high_risk",
            "verdict_message": "247 traces will break.",
            "high_risk_count": 247, "medium_risk_count": 0, "low_risk_count": 0,
            "total_traces_analyzed": 2002,
            "diff_summary": {"changes": []},
        }):
            result = runner.invoke(cli, [
                "regression-check",
                "--agent-name", "support-agent",
                "--candidate-manifest-id", "mfst_x",
                "--api-key", "k",
            ])
        assert result.exit_code == 1  # default fail-on=high
        assert "HIGH RISK" in result.output

    def test_high_risk_with_fail_on_none_exits_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        with self._patch_client({
            "verdict": "high_risk",
            "verdict_message": "247 will break.",
            "high_risk_count": 247, "medium_risk_count": 0, "low_risk_count": 0,
            "total_traces_analyzed": 2002, "diff_summary": {"changes": []},
        }):
            result = runner.invoke(cli, [
                "regression-check",
                "--agent-name", "support-agent",
                "--candidate-manifest-id", "mfst_x",
                "--fail-on", "none",
                "--api-key", "k",
            ])
        assert result.exit_code == 0

    def test_no_change_verdict_exits_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        with self._patch_client({
            "verdict": "no_change",
            "verdict_message": "No structural impact.",
            "high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 0,
            "total_traces_analyzed": 100, "diff_summary": {"changes": []},
        }):
            result = runner.invoke(cli, [
                "regression-check",
                "--agent-name", "support-agent",
                "--candidate-manifest-id", "mfst_x",
                "--api-key", "k",
            ])
        assert result.exit_code == 0

    def test_first_run_verdict_exits_zero_and_shows_friendly_message(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        with self._patch_client({
            "verdict": "first_run",
            "verdict_message": "First run for this agent.",
            "high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 0,
            "total_traces_analyzed": 0, "diff_summary": {"first_run": True},
        }):
            result = runner.invoke(cli, [
                "regression-check",
                "--agent-name", "support-agent",
                "--candidate-manifest-id", "mfst_x",
                "--api-key", "k",
            ])
        assert result.exit_code == 0
        assert "First run" in result.output

    def test_json_format_outputs_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        sample = {
            "verdict": "medium_risk",
            "verdict_message": "5 may behave differently",
            "high_risk_count": 0, "medium_risk_count": 5, "low_risk_count": 0,
            "total_traces_analyzed": 100, "diff_summary": {"changes": []},
        }
        with self._patch_client(sample):
            result = runner.invoke(cli, [
                "regression-check",
                "--agent-name", "support-agent",
                "--candidate-manifest-id", "mfst_x",
                "--format", "json",
                "--fail-on", "high",  # so we get exit 0
                "--api-key", "k",
            ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["verdict"] == "medium_risk"

    def test_github_format_emits_warning_annotation_for_medium_risk(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        with self._patch_client({
            "verdict": "medium_risk",
            "verdict_message": "5 may behave differently",
            "high_risk_count": 0, "medium_risk_count": 5, "low_risk_count": 0,
            "total_traces_analyzed": 100, "diff_summary": {"changes": []},
        }):
            result = runner.invoke(cli, [
                "regression-check",
                "--agent-name", "support-agent",
                "--candidate-manifest-id", "mfst_x",
                "--format", "github",
                "--fail-on", "high",
                "--api-key", "k",
            ])
        assert result.exit_code == 0
        assert "::warning" in result.output

    def test_reads_candidate_id_from_local_file_if_arg_omitted(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # GITHUB_OUTPUT takes priority in _resolve_candidate_manifest_id, so
        # clear it to force the file fallback (CI always sets this var).
        monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
        Path(tmp_path / "decimal_manifest_id.txt").write_text("mfst_from_file")
        runner = CliRunner()
        with self._patch_client({
            "verdict": "no_change",
            "verdict_message": "Safe",
            "high_risk_count": 0, "medium_risk_count": 0, "low_risk_count": 0,
            "total_traces_analyzed": 0, "diff_summary": {"changes": []},
        }) as mock_make_client:
            result = runner.invoke(cli, [
                "regression-check",
                "--agent-name", "support-agent",
                "--api-key", "k",
            ])
            mock_client = mock_make_client.return_value
            mock_client.run_regression_check.assert_called_once()
            kwargs = mock_client.run_regression_check.call_args.kwargs
            assert kwargs["candidate_manifest_id"] == "mfst_from_file"
        assert result.exit_code == 0

    def test_api_error_exits_with_code_2(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mock_client = MagicMock()
        mock_client.run_regression_check.side_effect = RuntimeError("connection refused")
        runner = CliRunner()
        with patch("decimalai.cli.main._make_client", return_value=mock_client):
            result = runner.invoke(cli, [
                "regression-check",
                "--agent-name", "support-agent",
                "--candidate-manifest-id", "mfst_x",
                "--api-key", "k",
            ])
        assert result.exit_code == 2
        assert "failed" in result.output.lower()
