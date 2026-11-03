"""CLI output and options that named something the product does not do."""

from __future__ import annotations

import re
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from decimalai.cli.main import cli

FULL_ID = "1f6c1b52-8c1e-4f6a-9d2b-3a7e5c0d9e11"


def _run(*args, client=None):
    client = client or MagicMock()
    with patch("decimalai.cli.main._make_client", return_value=client):
        result = CliRunner().invoke(cli, [*args, "--api-key", "k"])
    return result, client


class TestListsPrintIdsOtherCommandsAccept:
    def test_traces_list(self):
        client = MagicMock()
        client.list_traces.return_value = {"traces": [{"id": FULL_ID, "status": "success", "agent_name": "a"}]}
        result, _ = _run("traces", "list", client=client)
        assert FULL_ID in result.output

    def test_evaluators_list(self):
        client = MagicMock()
        client.list_evaluators.return_value = {"evaluators": [{"id": FULL_ID, "name": "n", "eval_type": "llm_judge"}]}
        result, _ = _run("evaluators", "list", client=client)
        assert FULL_ID in result.output

    def test_evaluators_add(self):
        client = MagicMock()
        client.add_evaluator.return_value = {"id": FULL_ID, "name": "n"}
        result, _ = _run("evaluators", "add", "--template", "helpfulness", client=client)
        assert FULL_ID in result.output

    def test_manifests_list(self):
        client = MagicMock()
        client.list_manifests.return_value = {"manifests": [{"id": FULL_ID, "agent_name": "a"}]}
        result, _ = _run("manifests", "list", client=client)
        assert FULL_ID in result.output

    def test_datasets_list(self):
        client = MagicMock()
        client.list_datasets.return_value = {"datasets": [{"id": FULL_ID, "name": "n", "dataset_type": "sft"}]}
        result, _ = _run("datasets", "list", client=client)
        assert FULL_ID in result.output

    def test_datasets_show_lists_whole_version_ids(self):
        client = MagicMock()
        client.get_dataset.return_value = {"id": "d", "name": "n", "versions": [{"id": FULL_ID, "version_number": 1}]}
        result, _ = _run("datasets", "show", "d", client=client)
        assert FULL_ID in result.output


class TestDatasetFiltersReachTheServer:
    def test_list_sends_agent_name(self):
        client = MagicMock()
        client.list_datasets.return_value = {"datasets": []}
        _run("datasets", "list", "--agent-name", "support", "--limit", "5", client=client)
        client.list_datasets.assert_called_once_with(limit=5, agent_name="support")

    def test_client_list_datasets_puts_agent_name_in_the_query(self):
        from decimalai._client import DecimalAIClient

        client = DecimalAIClient.__new__(DecimalAIClient)
        client._http = MagicMock()
        client._http.get.return_value = MagicMock(status_code=200, json=lambda: {"datasets": []})
        client.list_datasets(limit=5, agent_name="support")
        assert client._http.get.call_args.kwargs["params"]["agent_name"] == "support"

    def test_build_sends_the_field_the_server_reads(self):
        client = MagicMock()
        client.build_dataset.return_value = {"version_id": "v", "row_count": 1}
        _run("datasets", "build", "d", "--verdict", "keep", client=client)
        client.build_dataset.assert_called_once_with("d", filters={"allowed_verdicts": ["keep"]})


class TestEvaluatorTemplates:
    def test_help_names_a_template_that_exists(self):
        result = CliRunner().invoke(cli, ["evaluators", "add", "--help"])
        assert "helpfulness_judge" not in result.output
        assert "--template helpfulness " in result.output

    def test_a_template_keeps_its_own_category(self):
        client = MagicMock()
        client.add_evaluator.return_value = {"id": "x"}
        _run("evaluators", "add", "--template", "helpfulness", client=client)
        assert client.add_evaluator.call_args.kwargs["category"] is None

    def test_an_explicit_category_still_wins_over_the_template(self):
        client = MagicMock()
        client.add_evaluator.return_value = {"id": "x"}
        _run("evaluators", "add", "--template", "helpfulness", "--category", "safety", client=client)
        assert client.add_evaluator.call_args.kwargs["category"] == "safety"

    def test_a_custom_judge_keeps_the_default_category(self):
        client = MagicMock()
        client.add_evaluator.return_value = {"id": "x"}
        _run("evaluators", "add", "--name", "tone", "--type", "llm_judge", "--rubric", "r {output}", client=client)
        assert client.add_evaluator.call_args.kwargs["category"] == "quality"

    def test_the_client_omits_a_category_it_was_not_given(self):
        from decimalai._client import DecimalAIClient

        client = DecimalAIClient.__new__(DecimalAIClient)
        client._request_with_retry = MagicMock()
        client.add_evaluator(template_id="helpfulness", category=None)
        assert "category" not in client._request_with_retry.call_args.kwargs["json"]


class TestPublishDoesNotVerify:
    def test_push_help_does_not_promise_a_run_on_publish(self):
        result = CliRunner().invoke(cli, ["skills", "push", "--help"])
        text = re.sub(r"\s+", " ", result.output)
        assert "verification run" not in text
        assert "Publishing does not run one" in text

    def test_push_output_does_not_promise_a_run_on_publish(self, tmp_path):
        results = tmp_path / "results.json"
        results.write_text('{"skill": "s", "cases": []}')
        response = MagicMock(status_code=200)
        response.json.return_value = {"overall_verdict": "pass", "name": "s"}
        client = MagicMock()
        client._http.post.return_value = response
        result, _ = _run("skills", "push", str(results), "--skill", "s", client=client)
        assert "verification run included" not in result.output
        assert "decimalai skills benchmark" in result.output


class TestRepairApplyPointsSomewhereReal:
    def test_it_prints_no_dashboard_page_that_does_not_exist(self):
        client = MagicMock()
        client.repair_apply.return_value = {"batch_id": "b1", "status": "completed", "total_episodes": 2, "repaired_count": 2}
        result, _ = _run("repair", "apply", "--old-manifest-id", "o", "--new-manifest-id", "n", client=client)
        assert result.exit_code == 0, result.output
        assert "/repair/" not in result.output
        assert "b1" in result.output


class TestSkillScoreWording:
    def test_pull_quality_line_does_not_say_provisional(self):
        from decimalai.cli import skills_pm

        response = MagicMock(status_code=200)
        response.json.return_value = {
            "effectiveness": {"skill_score": 0.97, "score_components": {"legs": 1, "provisional": True}},
        }
        with patch.object(skills_pm, "_pull_get", return_value=response):
            result = CliRunner().invoke(_quality_command(skills_pm), [])
        assert "SkillScore 97" in result.output
        assert "provisional" not in result.output


def _quality_command(skills_pm):
    import click

    @click.command()
    def show():
        skills_pm._print_quality("https://example.test/x")

    return show
