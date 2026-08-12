"""`decimalai skills push` — upload a local skillevaluation results.json.

Network is mocked at ``_make_client`` (same convention as
test_skills_status_cli): these lock command wiring — endpoint, posted
body, name resolution, skipped-case rendering, error surfaces — not the
backend's import semantics (covered platform-side in
test_benchmark_import.py).
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from decimalai.cli.main import cli


def _results_doc(skill_name="gdpr-pii-classifier"):
    return {
        "format": "skillevaluation/test-run-result@v1",
        "pass_rate": {"with_skill": 0.8, "without_skill": 0.4, "delta_pts": 40.0},
        "errors": 0,
        "cases_aggregated": 5,
        "cases_skipped_apples_oranges": 0,
        "total_cases": 5,
        "verdict": "mixed",
        "skill": {"name": skill_name},
        "cases": [
            {"case_name": "alpha", "outcome": "flip_to_pass",
             "with_skill": {"passed": True, "task_attempted": True, "errored": False},
             "without_skill": {"passed": False, "task_attempted": True, "errored": False}},
        ],
    }


def _mock_response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


def _write(doc, path="results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


class TestSkillsPush:
    def test_registered_under_skills(self):
        result = CliRunner().invoke(cli, ["skills", "--help"])
        assert result.exit_code == 0
        assert "push" in result.output

    def test_posts_document_to_import_endpoint(self):
        client = MagicMock()
        client._http.post.return_value = _mock_response({
            "run_id": "r1", "overall_verdict": "mixed", "verified": False,
            "total_cases": 5, "imported_cases": 5, "skipped_cases": [],
        })
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write(_results_doc())
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "push", "results.json",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])

        assert result.exit_code == 0, result.output
        url, = client._http.post.call_args.args
        assert url == "/api/v1/skills/gdpr-pii-classifier/benchmark/import"
        posted = client._http.post.call_args.kwargs["json"]
        assert posted["format"] == "skillevaluation/test-run-result@v1"
        assert "UNVERIFIED" in result.output
        assert "5/5" in result.output

    def test_skill_name_flag_overrides_document(self):
        client = MagicMock()
        client._http.post.return_value = _mock_response({
            "run_id": "r1", "overall_verdict": "pass", "verified": False,
            "total_cases": 1, "imported_cases": 1, "skipped_cases": [],
        })
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write(_results_doc(skill_name="from-doc"))
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "push", "results.json", "--skill", "explicit-name",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
        assert result.exit_code == 0, result.output
        url, = client._http.post.call_args.args
        assert "explicit-name" in url

    def test_missing_skill_name_is_actionable(self):
        doc = _results_doc()
        del doc["skill"]
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write(doc)
            result = runner.invoke(cli, [
                "skills", "push", "results.json",
                "--api-key", "k", "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 1
        assert "--skill" in result.output

    def test_renders_skipped_cases_with_sync_hint(self):
        client = MagicMock()
        client._http.post.return_value = _mock_response({
            "run_id": "r1", "overall_verdict": "mixed", "verified": False,
            "total_cases": 5, "imported_cases": 3,
            "skipped_cases": ["new_case_a", "new_case_b"],
        })
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write(_results_doc())
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "push", "results.json",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
        assert result.exit_code == 0, result.output
        assert "new_case_a" in result.output
        assert "skills sync" in result.output

    def test_404_points_at_sync_first(self):
        """Pushing before syncing (the natural first-touch mistake) must
        explain the missing step, not dump a traceback."""
        client = MagicMock()
        client._http.post.return_value = _mock_response(
            {"detail": "Skill not found"}, status_code=404,
        )
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write(_results_doc())
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "push", "results.json",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
        assert result.exit_code == 1
        assert "doesn't exist on the platform yet" in result.output
        assert "skills sync" in result.output
        assert "Traceback" not in result.output

    def test_422_rejection_surfaces_detail(self):
        client = MagicMock()
        client._http.post.return_value = _mock_response(
            {"detail": "verdict must be one of pass/fail/mixed/error"},
            status_code=422,
        )
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write(_results_doc())
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "push", "results.json",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
        assert result.exit_code == 1
        assert "rejected" in result.output
        assert "verdict must be" in result.output

    def test_unreadable_file_is_actionable(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("results.json", "w", encoding="utf-8") as f:
                f.write("{not json")
            result = runner.invoke(cli, [
                "skills", "push", "results.json",
                "--api-key", "k", "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 1
        assert "could not read" in result.output


_EVAL_YAML_V2 = """\
cases:
  - name: email_is_pii
    prompt: "Classify these fields: email, ip_address."
    expectations:
      - "The response classifies email as PII"
  - name: name_is_pii
    prompt: "Classify these fields: name, age."
    expectations:
      - "The response classifies name as PII"
"""


def _write_skill_dir(path="skills/s", eval_yaml=None):
    os.makedirs(path)
    with open(os.path.join(path, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write("---\nname: s\n---\n\n# Body\nLong enough body for the command.")
    if eval_yaml is not None:
        with open(os.path.join(path, "eval.yaml"), "w", encoding="utf-8") as f:
            f.write(eval_yaml)


def _benchmark_client():
    """Mock client for a full sync → run round-trip."""
    client = MagicMock()
    sync_resp = _mock_response({"results": []})
    run_resp = _mock_response({
        "passed_cases": 2, "total_cases": 2, "overall_verdict": "pass",
        "aggregate_metrics": {}, "results": [],
    })
    client._http.post.side_effect = [sync_resp, run_resp]
    return client


class TestBenchmarkRuns:
    """`--runs N` — a run-level parameter on the hosted run endpoint: re-run
    the whole suite N times, uniformly, aggregated by MEAN (ADR-0007). It does
    NOT modify the synced eval.yaml. The old per-case pass^k `--trials` flag is
    removed and redirects here."""

    def _invoke(self, runner, extra_args, client=None):
        args = [
            "skills", "benchmark", "skills/s",
            "--api-key", "k", "--base-url", "http://localhost:8000",
            *extra_args,
        ]
        if client is None:
            return runner.invoke(cli, args)
        with patch("decimalai.cli.main._make_client", return_value=client):
            return runner.invoke(cli, args)

    # ── removed --trials redirects to --runs ─────────────────

    def test_trials_removed_points_to_runs(self):
        """A stale script still passing --trials must fail loudly with the
        exact replacement — never silently ignored (would mislabel the run)."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, ["--trials", "3"])
        assert result.exit_code == 1
        assert "--trials has been removed" in result.output
        assert "--runs 3" in result.output  # echoes the value they asked for

    def test_trials_removed_before_any_network(self):
        """The redirect fires before the client is even constructed, so no
        sync/run is attempted."""
        client = MagicMock()
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, ["--trials", "3"], client=client)
        assert result.exit_code == 1
        assert client._http.post.call_count == 0

    # ── --runs flag validation ───────────────────────────────

    def test_rejects_zero(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, ["--runs", "0"])
        assert result.exit_code == 1
        assert "between 1 and 10" in result.output

    def test_rejects_over_platform_cap(self):
        """11 > _BENCHMARK_MAX_RUNS (10) — reject here rather than let the
        server clamp silently."""
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, ["--runs", "11"])
        assert result.exit_code == 1
        assert "between 1 and 10" in result.output

    def test_rejects_non_integer(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, ["--runs", "three"])
        assert result.exit_code == 2  # click type error
        assert "not a valid integer" in result.output

    def test_runs_without_eval_yaml_is_fine(self):
        """--runs is a run-level query param, independent of the eval.yaml —
        it works even with no eval.yaml present (unlike the old --trials)."""
        client = _benchmark_client()
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=None)
            result = self._invoke(runner, ["--runs", "3"], client=client)
        assert result.exit_code == 0, result.output
        run_call = client._http.post.call_args_list[-1]
        assert run_call.args[0].endswith("/benchmark/run")
        assert run_call.kwargs["params"]["runs"] == 3

    # ── --runs is a query param, not an eval.yaml rewrite ────

    def test_runs_passed_as_query_param(self):
        client = _benchmark_client()
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, ["--runs", "3"], client=client)
        assert result.exit_code == 0, result.output
        run_call = client._http.post.call_args_list[-1]
        assert run_call.args[0].endswith("/benchmark/run")
        assert run_call.kwargs["params"]["runs"] == 3
        # The uploaded eval.yaml is the file's exact text — --runs never
        # touches it (no parse/re-dump round-trip).
        uploaded = client._http.post.call_args_list[0].kwargs["json"]["skills"][0]
        assert uploaded["eval_yaml_text"] == _EVAL_YAML_V2

    def test_no_runs_flag_omits_param(self):
        """No --runs → the benchmark/run call carries no `runs` param (the
        server defaults to 1)."""
        client = _benchmark_client()
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, [], client=client)
        assert result.exit_code == 0, result.output
        run_call = client._http.post.call_args_list[-1]
        assert "runs" not in run_call.kwargs["params"]

    def test_local_eval_yaml_untouched(self):
        client = _benchmark_client()
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, ["--runs", "3"], client=client)
            with open("skills/s/eval.yaml", encoding="utf-8") as f:
                on_disk = f.read()
        assert result.exit_code == 0, result.output
        assert on_disk == _EVAL_YAML_V2

    # ── sync-failure honesty ─────────────────────────────────

    def test_sync_failure_still_continues(self):
        """The degrade path is unchanged: a sync failure continues against the
        remote version (--runs doesn't depend on the synced eval.yaml)."""
        client = MagicMock()
        run_resp = _mock_response({
            "passed_cases": 1, "total_cases": 1, "overall_verdict": "pass",
            "aggregate_metrics": {}, "results": [],
        })
        client._http.post.side_effect = [RuntimeError("boom"), run_resp]
        runner = CliRunner()
        with runner.isolated_filesystem():
            _write_skill_dir(eval_yaml=_EVAL_YAML_V2)
            result = self._invoke(runner, ["--runs", "3"], client=client)
        assert result.exit_code == 0, result.output
        assert "continuing with existing remote version" in result.output


class TestBenchmarkQuotaHint:
    def test_429_mentions_local_runner_and_upgrade(self, tmp_path):
        client = MagicMock()
        # First post = skills/sync (succeeds); second = benchmark/run (429).
        sync_resp = _mock_response({"results": []})
        quota_resp = _mock_response(
            {"detail": {"error": "limit_exceeded", "feature": "benchmark_cases",
                        "used": 100, "limit": 100, "plan": "free",
                        "upgrade_url": "https://app.decimal.ai/billing"}},
            status_code=429,
        )
        client._http.post.side_effect = [sync_resp, quota_resp]
        runner = CliRunner()
        with runner.isolated_filesystem():
            import os
            os.makedirs("skills/s")
            with open("skills/s/SKILL.md", "w", encoding="utf-8") as f:
                f.write("---\nname: s\n---\n\n# Body\nLong enough body for the command.")
            with patch("decimalai.cli.main._make_client", return_value=client):
                result = runner.invoke(cli, [
                    "skills", "benchmark", "skills/s",
                    "--api-key", "k", "--base-url", "http://localhost:8000",
                ])
        assert result.exit_code == 1
        assert "benchmark_cases limit reached" in result.output
        assert "skillevaluation[runner]" in result.output
        assert "https://app.decimal.ai/billing" in result.output
