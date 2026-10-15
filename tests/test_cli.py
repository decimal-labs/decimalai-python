"""Tests for the CLI commands."""

from click.testing import CliRunner

import decimalai
from decimalai.cli.main import cli


class TestCli:
    """Test CLI structure and basic invocations."""

    def test_cli_version(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert decimalai.__version__ in result.output

    def test_cli_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "traces" in result.output

    def test_traces_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["traces", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output

    def test_traces_list_requires_api_key(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["traces", "list"])
        assert result.exit_code == 1

    def test_traces_show_requires_api_key(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["traces", "show", "some-id"])
        assert result.exit_code == 1

    def test_compat_check_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compat-check", "--help"])
        assert result.exit_code == 0
        assert "--agent-name" in result.output
        assert "--format" in result.output
        assert "table" in result.output
        assert "json" in result.output
        assert "github" in result.output

    def test_compat_check_requires_agent_name(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compat-check"])
        assert result.exit_code != 0
        assert "agent-name" in result.output.lower() or "missing" in result.output.lower()

    def test_compat_check_requires_api_key(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compat-check", "--agent-name", "test-agent"])
        assert result.exit_code == 1

    def test_compat_check_in_help_output(self):
        """Verify compat-check appears in the top-level CLI help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "compat-check" in result.output


class TestRepairCli:
    """The `decimalai repair` command group."""

    def test_repair_help_lists_subcommands(self):
        result = CliRunner().invoke(cli, ["repair", "--help"])
        assert result.exit_code == 0
        assert "preview" in result.output
        assert "apply" in result.output

    def test_repair_in_top_level_help(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "repair" in result.output

    def test_repair_preview_help(self):
        result = CliRunner().invoke(cli, ["repair", "preview", "--help"])
        assert result.exit_code == 0
        assert "--old-manifest-id" in result.output
        assert "--new-manifest-id" in result.output

    def test_repair_preview_requires_manifest_ids(self):
        result = CliRunner().invoke(cli, ["repair", "preview"])
        assert result.exit_code != 0
        assert "missing" in result.output.lower() or "manifest" in result.output.lower()

    def test_repair_preview_requires_api_key(self):
        result = CliRunner().invoke(
            cli, ["repair", "preview", "--old-manifest-id", "m1", "--new-manifest-id", "m2"]
        )
        assert result.exit_code == 1

    def test_repair_preview_renders_rules(self):
        from unittest.mock import patch

        from decimalai._client import DecimalAIClient

        canned = {
            "rules": [
                {
                    "rule_type": "tool_rename",
                    "component_name": "search",
                    "confidence": "high",
                    "details": {"from": "search", "to": "web_search"},
                }
            ],
            "previews": [],
            "total_eligible": 1,
        }
        with patch.object(DecimalAIClient, "repair_preview", return_value=canned):
            result = CliRunner().invoke(cli, [
                "repair", "preview",
                "--old-manifest-id", "m1", "--new-manifest-id", "m2",
                "--api-key", "dai_sk_test", "--base-url", "http://localhost:8000",
            ])
        assert result.exit_code == 0, result.output
        assert "[0]" in result.output
        assert "tool_rename" in result.output


def _quality(registry, detail):
    """Answer `pull`'s two registry reads beside the bundle: no eval.yaml, then the detail
    the quality line comes from (the fake answers neither route itself)."""
    import httpx

    registry.fault("other", httpx.Response(404, json={"detail": "Not Found"}),
                   httpx.Response(200, json=detail))


class TestSkillsPullCli:
    """Tests for `decimalai skills pull <ref>` — the no-auth public registry pull.

    Key invariant: this command must NOT require an API key. Anonymous
    developers can copy a skill into ./<install-name>/SKILL.md as a first
    interaction with DecimalAI. It writes the registry's own bytes, through the
    anonymous bundle route (tests/fakes/fake_registry.py).
    """

    def test_help_describes_no_signup(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["skills", "pull", "--help"])
        assert result.exit_code == 0
        # The "no signup required" framing is the whole point — pin it.
        assert "no signup" in result.output.lower()
        assert "--out" in result.output
        assert "--stdout" in result.output

    def _pull(self, tmp_path, name, detail=None, *extra):
        from tests.fakes.fake_registry import FakeRegistry

        registry = FakeRegistry()
        registry.add_skill(name, body=f"# {name}\n\nDo a code review.\n")
        _quality(registry, {"id": "x", "name": name, **(detail or {})})
        with registry.serve() as url:
            return registry, CliRunner().invoke(
                cli, ["skills", "pull", name, "--out", str(tmp_path), "--base-url", url, *extra])

    def test_pull_writes_skill_md_to_disk(self, tmp_path):
        _, result = self._pull(tmp_path, "code-review", {
            # SkillScore v2 shape: composite + decomposition flags + live pass,
            # with the benchmark lift in its own summary block.
            "effectiveness": {
                "skill_score": 0.87,
                "score_components": {"legs": 2, "provisional": False},
                "avg_pass_rate": 0.96,
            },
            "benchmark_summary": {
                "pass_rate_delta_pts": 60.0,
                "tokens_delta_pct": -43.0,
                "turns_delta_pct": -30.0,
            },
        })
        assert result.exit_code == 0, result.output
        written = tmp_path / "code-review" / "SKILL.md"
        assert written.exists(), f"missing {written}"
        body = written.read_text()
        assert body.startswith("---\nname: code-review\n")
        assert "Do a code review" in body
        # Quality line — composite · lift · live pass, on one line.
        assert "SkillScore 87 · +60 pts vs no skill · 96% live pass" in result.output
        # Full evidence (2 legs) → no provisional marker.
        assert "(provisional)" not in result.output
        # Efficiency headline is tokens-first and no longer repeats the lift.
        assert "Efficiency vs no skill: -43% tokens · -30% turns" in result.output
        assert "pts pass rate" not in result.output
        # Sends users to the canonical /skills/<name> URL post-pull, and to `add`.
        assert "/skills/code-review" in result.output
        assert "decimalai skills add code-review" in result.output

    def test_pull_quality_line_marks_provisional(self, tmp_path):
        """One evidence leg → the score renders `(provisional)` so a pulled
        skill never overstates one-signal evidence."""
        _, result = self._pull(tmp_path, "prov-skill", {
            "effectiveness": {
                "skill_score": 0.97,
                "score_components": {"legs": 1, "provisional": True},
            },
        })
        assert result.exit_code == 0, result.output
        assert "SkillScore 97 (provisional)" in result.output
        # No live/lift evidence → quality line is the score alone.
        assert "live pass" not in result.output
        assert "vs no skill" not in result.output

    def test_pull_quality_line_falls_back_to_frozen_alias(self, tmp_path):
        """Older backends without `skill_score` still surface the composite
        via the frozen `avg_effectiveness` alias."""
        _, result = self._pull(tmp_path, "alias-skill", {
            "effectiveness": {"avg_effectiveness": 0.72},
        })
        assert result.exit_code == 0, result.output
        assert "SkillScore 72" in result.output

    def test_pull_no_evidence_omits_quality_line(self, tmp_path):
        """No score, no benchmark, no live pass → the pull stays clean
        (no quality line at all) instead of printing empty scaffolding."""
        _, result = self._pull(tmp_path, "new-skill")
        assert result.exit_code == 0, result.output
        assert "SkillScore" not in result.output
        assert "Efficiency vs no skill" not in result.output
        # The pull confirmation + next-steps funnel still print.
        assert "✓ Pulled new-skill v1" in result.output
        assert "/skills/new-skill" in result.output

    def test_pull_stdout_mode_prints_body(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _, result = self._pull(tmp_path / "unused", "pdf", None, "--stdout")
        assert result.exit_code == 0, result.output
        assert result.stdout.startswith("---\nname: pdf\n")
        assert "Do a code review" in result.stdout
        assert not (tmp_path / "unused").exists() and not (tmp_path / "pdf").exists()

    def test_pull_exits_7_when_skill_not_found(self, tmp_path):
        from tests.fakes.fake_registry import FakeRegistry

        registry = FakeRegistry()
        with registry.serve() as url:
            result = CliRunner().invoke(cli, ["skills", "pull", "no-such-skill",
                                              "--out", str(tmp_path), "--base-url", url])
        assert result.exit_code == 7
        assert "no-such-skill" in result.output
        assert "no public skill" in result.output

    def test_pull_does_not_require_or_send_an_api_key(self, tmp_path, monkeypatch):
        """A key in the environment is never sent: pull is the anonymous door."""
        monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_should_not_travel")
        registry, result = self._pull(tmp_path, "x", None, "--stdout")
        assert result.exit_code == 0, result.output
        assert "api key" not in result.output.lower()
        assert registry.requests
        assert not any(r.headers.get("authorization") for r in registry.requests)


class TestTracesImportWireFormat:
    """`decimalai traces import` — the migration journey's documented trigger.

    THE REGRESSION THIS EXISTS TO CATCH (2026-09-05). The JSONL branch posted the
    file's bytes as a raw body with `Content-Type: application/x-ndjson`, while
    the endpoint is declared `file: UploadFile = File(...)`,
    `agent_name: … = Form(…)` — multipart only. Every invocation answered 422 and
    died in `raise_for_status()`, including the exact command the endpoint's own
    OpenAPI sample prints and the public LangSmith/Braintrust migration guide
    gives.

    The second half is subtler and is why "just send files=" was not enough:
    `sdk_headers` pins `Content-Type: application/json` on the shared client, and
    httpx only `setdefault`s the multipart content-type it computes — so the
    pinned JSON type wins, the boundary never ships, and the server sees no form
    fields. These assert the WIRE FORMAT (multipart, with a boundary, carrying
    the file and the agent_name part), because a response-shape assertion passes
    against either encoding as soon as a mock is involved.
    """

    def _run(self, args, captured):
        from unittest.mock import MagicMock, patch
        from click.testing import CliRunner

        resp = MagicMock(status_code=200)
        resp.json.return_value = {"imported_count": 2, "error_count": 0}

        real_client = __import__("httpx").Client

        class RecordingClient(real_client):
            def post(self, url, **kwargs):
                captured.append({"url": url, **kwargs, "headers": dict(self.headers)})
                return resp

        with patch("httpx.Client", RecordingClient):
            return CliRunner().invoke(cli, args)

    def test_jsonl_is_sent_as_multipart_not_a_raw_body(self, tmp_path):
        f = tmp_path / "legacy.jsonl"
        f.write_text('{"input": "a", "output": "b"}\n')
        captured: list = []
        result = self._run(
            ["traces", "import", str(f), "--agent-name", "acme-bot",
             "--api-key", "dai_sk_x", "--base-url", "http://localhost:8000"],
            captured,
        )
        assert result.exit_code == 0, result.output
        upload = [c for c in captured if "files" in c]
        assert upload, f"JSONL import sent no multipart request; calls were {captured}"
        call = upload[-1]
        assert call["url"] == "/api/v1/traces/import"
        assert "file" in call["files"]
        # The agent_name must ride as a FORM field — the server rejects the
        # upload without it unless every row carries its own.
        assert call["data"]["agent_name"] == "acme-bot"
        # And the client must NOT be pinning a JSON content-type, or httpx's
        # multipart boundary never reaches the server.
        assert not any(k.lower() == "content-type" for k in call["headers"]), (
            "the upload client still pins a Content-Type; httpx setdefault()s the "
            f"multipart one, so the boundary is dropped: {call['headers']}"
        )
        assert "content" not in call, "the file must not be sent as a raw body"

    def test_a_server_error_prints_what_the_server_said(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from click.testing import CliRunner
        import httpx

        f = tmp_path / "legacy.jsonl"
        f.write_text('{"input": "a", "output": "b"}\n')
        resp = MagicMock(status_code=400)
        resp.json.return_value = {"detail": "No agent_name provided."}
        resp.text = '{"detail": "No agent_name provided."}'

        real_client = httpx.Client

        class ErrClient(real_client):
            def post(self, url, **kwargs):
                return resp

        with patch("httpx.Client", ErrClient):
            result = CliRunner().invoke(
                cli,
                ["traces", "import", str(f),
                 "--api-key", "dai_sk_x", "--base-url", "http://localhost:8000"],
            )
        # A user error must read as a sentence, not an httpx traceback.
        assert result.exit_code != 0
        assert "No agent_name provided." in result.output
        assert "Traceback" not in result.output
