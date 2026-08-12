"""`decimalai replay run BATCH_ID` must iterate the batch's tasks and
submit a per-task result (so the batch advances), NOT replay the whole agent.
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from decimalai.cli.main import cli


def _batch():
    return {
        "id": "b1",
        "agent_name": "a",
        "tasks": [
            {"id": "rt1", "status": "pending",
             "task_input": {"user_input": "hi", "agent_name": "a"}},
            {"id": "rt2", "status": "completed",
             "task_input": {"user_input": "x", "agent_name": "a"}},
        ],
    }


def test_replay_run_is_batch_aware():
    client = MagicMock()
    client.get_replay_batch.return_value = _batch()
    client.submit_replay_result.return_value = {"eval_verdict": "keep"}

    with patch("decimalai._config._get_client", return_value=client), \
         patch("decimalai.replay.tasks.load_agent_fn", return_value=lambda x: "out"), \
         patch("decimalai.replay.tasks._latest_trace_id", return_value="base"), \
         patch("decimalai.replay.tasks._wait_for_new_trace", return_value="new-trace"), \
         patch("decimalai.init"):
        result = CliRunner().invoke(cli, [
            "replay", "run", "b1", "--agent-fn", "mod:fn",
            "--api-key", "dai_sk_test", "--base-url", "http://localhost:8000",
        ])

    assert result.exit_code == 0, result.output
    # Only the ONE pending task is submitted, keyed by its task_id (not the agent).
    client.submit_replay_result.assert_called_once()
    kwargs = client.submit_replay_result.call_args.kwargs
    assert kwargs["task_id"] == "rt1"
    assert kwargs["replayed_trace_id"] == "new-trace"


def test_replay_run_dry_run_submits_nothing():
    client = MagicMock()
    client.get_replay_batch.return_value = _batch()

    with patch("decimalai._config._get_client", return_value=client), \
         patch("decimalai.replay.tasks.load_agent_fn", return_value=lambda x: "out"), \
         patch("decimalai.init"):
        result = CliRunner().invoke(cli, [
            "replay", "run", "b1", "--agent-fn", "mod:fn", "--dry-run",
            "--api-key", "dai_sk_test", "--base-url", "http://localhost:8000",
        ])

    assert result.exit_code == 0, result.output
    assert "1 pending task" in result.output
    client.submit_replay_result.assert_not_called()
