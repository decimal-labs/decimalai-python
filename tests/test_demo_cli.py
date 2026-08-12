"""Tests for the `decimalai demo` harness — the shared one-command sandbox.

Two seeds (regression + skills), one teardown (`demo reset`). These lock the
CLI surface and the printed deep links a new user follows to the "wow":

  * `demo regression` seeds the v1→v2 agent, runs the regression check, and
    prints the impact-report URL.
  * `demo skills` seeds the ranked registry and prints the registry + top-skill
    links.
  * `demo reset` calls the shared cleanup endpoint that wipes BOTH demos.

Network is mocked at `_make_client` — these assert command wiring (the right
endpoint, the `force` flag, the right URL), not backend behavior (covered by
the platform-side `test_demo_*` suites).
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from decimalai.cli.main import _demo_web_url, cli


def _mock_response(payload):
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status = MagicMock()
    return resp


def _patched_client():
    """A MagicMock standing in for DecimalAIClient with the surface the
    demo commands touch: `_http.post/delete`, `run_regression_check`, `close`."""
    return MagicMock()


# ── Registration / help ────────────────────────────────────


class TestDemoHelp:
    def test_demo_group_in_top_level_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "demo" in result.output

    def test_demo_lists_three_subcommands(self):
        result = CliRunner().invoke(cli, ["demo", "--help"])
        assert result.exit_code == 0
        for sub in ("regression", "skills", "reset"):
            assert sub in result.output

    def test_reset_flag_documented(self):
        result = CliRunner().invoke(cli, ["demo", "skills", "--help"])
        assert result.exit_code == 0
        assert "--reset" in result.output and "--no-reset" in result.output


# ── Web-URL derivation ─────────────────────────────────────


class TestDemoWebUrl:
    def test_explicit_web_wins(self):
        assert _demo_web_url("https://api.decimal.ai", "https://x.test/") == "https://x.test"

    def test_localhost_pairs_with_3000(self):
        assert _demo_web_url("http://localhost:8000", None) == "http://localhost:3000"
        assert _demo_web_url("http://127.0.0.1:8000", None) == "http://localhost:3000"

    def test_hosted_api_maps_to_app(self):
        assert _demo_web_url("https://api.decimal.ai", None) == "https://app.decimal.ai"


# ── demo regression ────────────────────────────────────────


class TestDemoRegression:
    def test_requires_api_key(self, monkeypatch):
        for k in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = CliRunner().invoke(cli, ["demo", "regression", "--base-url", "http://localhost:8000"])
        assert result.exit_code == 1

    def test_happy_path_prints_impact_report_url(self):
        client = _patched_client()
        client._http.post.return_value = _mock_response({
            "agent_name": "[Demo] support-agent",
            "status": "created",
            "traces": 10,
            "v1_manifest_id": "v1aaaaaaaaaa",
            "v2_manifest_id": "v2bbbbbbbbbb",
        })
        client.run_regression_check.return_value = {
            "id": "rc_123",
            "verdict": "high_risk",
            "verdict_message": "2 traces will break",
            "total_traces_analyzed": 10,
            "high_risk_count": 2,
            "medium_risk_count": 1,
            "low_risk_count": 0,
        }
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "demo", "regression",
                "--api-key", "dai_sk_test", "--base-url", "http://localhost:8000",
            ])

        assert result.exit_code == 0, result.output
        # Seeded against the impact endpoint, with force=True (reset default).
        path, kwargs = client._http.post.call_args[0][0], client._http.post.call_args.kwargs
        assert path == "/api/v1/demo/seed-impact"
        assert kwargs["params"] == {"force": True}
        # Ran the check on the returned candidate manifest.
        assert client.run_regression_check.call_args.kwargs["candidate_manifest_id"] == "v2bbbbbbbbbb"
        # Printed the localhost-paired deep link to the run.
        assert "http://localhost:3000/agents/%5BDemo%5D%20support-agent/impact-reports/rc_123" in result.output

    def test_no_reset_passes_force_false(self):
        client = _patched_client()
        client._http.post.return_value = _mock_response({
            "agent_name": "[Demo] support-agent", "status": "created",
            "v2_manifest_id": "v2xxxxxxxxxx", "traces": 10,
        })
        client.run_regression_check.return_value = {"id": "rc_9", "verdict": "low_risk"}
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "demo", "regression", "--no-reset",
                "--api-key", "k", "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 0, result.output
        assert client._http.post.call_args.kwargs["params"] == {"force": False}

    def test_already_exists_skips_check_and_points_to_agent(self):
        client = _patched_client()
        client._http.post.return_value = _mock_response({
            "agent_name": "[Demo] support-agent",
            "status": "already_exists",
            "message": "Demo data already exists.",
        })
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "demo", "regression", "--no-reset",
                "--api-key", "k", "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 0, result.output
        client.run_regression_check.assert_not_called()
        assert "already exists" in result.output.lower()
        assert "--reset" in result.output


# ── demo skills ────────────────────────────────────────────


class TestDemoSkills:
    def test_happy_path_prints_registry_and_top_skill(self):
        client = _patched_client()
        client._http.post.return_value = _mock_response({
            "status": "created",
            "skill_names": ["[Demo] code-reviewer", "[Demo] sql-optimizer"],
            "top_skill_id": "skill_top_42",
            "traces": 36,
            "daily_stats_rows": 120,
        })
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "demo", "skills",
                "--api-key", "k", "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 0, result.output
        assert client._http.post.call_args[0][0] == "/api/v1/demo/seed-skills"
        assert client._http.post.call_args.kwargs["params"] == {"force": True}
        # 2026-06-01 — under the Skills Registry rename, demo links point
        # to /skills (browse) and /skills/<id-or-name> (detail). The
        # detail link falls back to the id when the seed payload doesn't
        # ship a top_skill_name.
        assert "http://localhost:3000/skills" in result.output
        assert "http://localhost:3000/skills/skill_top_42" in result.output

    def test_web_override_used_for_links(self):
        client = _patched_client()
        client._http.post.return_value = _mock_response({
            "status": "created", "skill_names": [], "top_skill_id": "s1",
            "traces": 1, "daily_stats_rows": 1,
        })
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "demo", "skills", "--web", "https://demo.example",
                "--api-key", "k", "--base-url", "https://api.decimal.ai",
            ])
        assert result.exit_code == 0, result.output
        assert "https://demo.example/skills" in result.output

    def test_already_exists_points_to_registry_without_deep_link(self):
        client = _patched_client()
        client._http.post.return_value = _mock_response({
            "status": "already_exists",
            "message": "Demo skills already exist.",
        })
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "demo", "skills", "--no-reset",
                "--api-key", "k", "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 0, result.output
        assert "http://localhost:3000/skills" in result.output
        # The already-exists branch reaches the Registry browse only —
        # no deep link to a specific skill (we don't know which top skill
        # the seed picked previously).
        # Note: under the rename the per-skill link is /skills/<name>, so
        # the negative assertion now checks against that pattern.
        assert "skills/skill_top_" not in result.output


# ── demo reset (shared teardown) ───────────────────────────


class TestDemoReset:
    def test_reports_both_demos_counts(self):
        client = _patched_client()
        client._http.delete.return_value = _mock_response({
            "status": "cleaned",
            "manifests": 2,
            "traces": 10,
            "skills": {"skills": 3, "traces": 36},
        })
        with patch("decimalai.cli.main._make_client", return_value=client):
            result = CliRunner().invoke(cli, [
                "demo", "reset",
                "--api-key", "k", "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 0, result.output
        assert client._http.delete.call_args[0][0] == "/api/v1/demo/cleanup"
        assert "2 manifest(s)" in result.output and "10 trace(s)" in result.output
        assert "3 skill(s)" in result.output

    def test_requires_api_key(self, monkeypatch):
        for k in ("DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
            monkeypatch.delenv(k, raising=False)
        result = CliRunner().invoke(cli, ["demo", "reset", "--base-url", "http://localhost:8000"])
        assert result.exit_code == 1


# ── accurate seed wording + slug URL ────────────────────────────────


class TestDemoSkillsWordingAndSlug:
    def _invoke(self, payload):
        client = _patched_client()
        client._http.post.return_value = _mock_response(payload)
        with patch("decimalai.cli.main._make_client", return_value=client):
            return CliRunner().invoke(cli, [
                "demo", "skills",
                "--api-key", "k", "--base-url", "http://localhost:8000",
            ])

    def test_banner_says_demo_skills_into_workspace_not_public(self):
        result = self._invoke({
            "status": "created", "skill_names": [], "top_skill_id": "s1",
            "traces": 1, "daily_stats_rows": 1,
        })
        assert result.exit_code == 0, result.output
        # The seeds are org-scoped (verified: other orgs see 0 items) —
        # the old "3 public skills" banner was wrong on the load-bearing word.
        assert "Seeding 3 demo skills into your workspace" in result.output
        assert "public skills" not in result.output

    def test_top_skill_link_uses_url_slug_when_present(self):
        result = self._invoke({
            "status": "created",
            "skill_names": ["[Demo] code-reviewer"],
            "top_skill_id": "skill_top_42",
            "top_skill_name": "[Demo] code-reviewer",
            "top_skill_slug": "demo-code-reviewer",
            "traces": 36, "daily_stats_rows": 120,
        })
        assert result.exit_code == 0, result.output
        assert "http://localhost:3000/skills/demo-code-reviewer" in result.output
        # The display name is not a URL — never interpolate it.
        assert "/skills/[Demo] code-reviewer" not in result.output

    def test_top_skill_link_falls_back_to_id_never_display_name(self):
        result = self._invoke({
            "status": "created",
            "skill_names": ["[Demo] code-reviewer"],
            "top_skill_id": "skill_top_42",
            "top_skill_name": "[Demo] code-reviewer",
            "traces": 36, "daily_stats_rows": 120,
        })
        assert result.exit_code == 0, result.output
        assert "http://localhost:3000/skills/skill_top_42" in result.output
        assert "/skills/[Demo] code-reviewer" not in result.output
