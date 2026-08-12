"""Lock in: replay links the trace IT produced, not a stale/concurrent one.

Deep-audit finding (sdk-data): replay.run() slept 200ms then took
``list_traces(limit=1)[0]`` and assumed it was the trace just produced.
Any concurrent trace for the same agent, or an async export slower than
the sleep, made it link the WRONG trace — silently corrupting eval scores.

The fix captures the most-recent trace id BEFORE running the agent and
accepts only a trace NEWER than that baseline.

No backend — the client is faked.
"""

from unittest.mock import MagicMock, patch

import decimalai.replay.tasks as tasks
from decimalai.replay.tasks import (
    ReplayPrompt,
    _latest_trace_id,
    _wait_for_new_trace,
)


class _FakeClient:
    """Returns whatever trace ids the test stages via ``next_ids``."""

    def __init__(self, id_sequence):
        # id_sequence: list of trace ids returned on successive list_traces
        self._seq = list(id_sequence)
        self.calls = 0

    def list_traces(self, limit=1, agent_name=None, **kw):
        idx = min(self.calls, len(self._seq) - 1)
        self.calls += 1
        tid = self._seq[idx]
        traces = [{"id": tid}] if tid is not None else []
        return {"traces": traces}


def test_wait_for_new_trace_rejects_stale_baseline():
    """If the most-recent trace is still the baseline (export lagged),
    _wait_for_new_trace keeps polling and only returns a NEWER id.
    """
    # First two polls still see the old trace, then the new one lands.
    client = _FakeClient(["old", "old", "new"])
    result = _wait_for_new_trace(
        client, "agent", baseline_trace_id="old", attempts=5, interval_s=0.0
    )
    assert result == "new", "should wait past the stale baseline for the new trace"


def test_wait_for_new_trace_times_out_rather_than_mislinking():
    """If no newer trace ever appears, return None instead of linking the
    stale baseline trace (the bug was linking the wrong trace anyway).
    """
    client = _FakeClient(["old"])
    result = _wait_for_new_trace(
        client, "agent", baseline_trace_id="old", attempts=3, interval_s=0.0
    )
    assert result is None, "must NOT return the stale baseline trace id"


def test_wait_for_new_trace_accepts_when_no_baseline():
    """First-ever trace (no baseline) is accepted immediately."""
    client = _FakeClient(["t1"])
    result = _wait_for_new_trace(
        client, "agent", baseline_trace_id=None, attempts=3, interval_s=0.0
    )
    assert result == "t1"


def test_run_links_new_trace_not_concurrent_one():
    """End-to-end: a concurrent trace for the same agent is present as the
    baseline; replay must link the trace the agent_fn just produced, NOT
    the pre-existing concurrent one.
    """
    prompt = ReplayPrompt(
        trace_id="original-123",
        user_input="hello",
        agent_name="support-agent",
    )

    # Baseline poll (before agent_fn) sees a concurrent trace "concurrent".
    # After agent_fn, polls return the freshly-produced "replay-fresh".
    fake_client = _FakeClient(["concurrent", "replay-fresh"])

    captured_link = {}

    def fake_link(original_id, replayed_id):
        captured_link["original"] = original_id
        captured_link["replayed"] = replayed_id
        return {"eval_score": 1.0, "eval_verdict": "pass"}

    with patch.object(tasks, "get_prompts", return_value=[prompt]), \
         patch.object(tasks, "link", side_effect=fake_link), \
         patch("decimalai._config._get_client", return_value=fake_client), \
         patch("decimalai.replay.tasks._wait_for_new_trace") as mock_wait:
        # Force the helper to behave deterministically: it must be handed the
        # concurrent baseline and return the fresh trace.
        mock_wait.return_value = "replay-fresh"

        results = tasks.run(agent_fn=lambda x: "out", agent_name="support-agent")

        # The baseline captured before agent_fn must be the concurrent trace.
        _, kwargs = None, None
        call = mock_wait.call_args
        # baseline is positional arg #3 (client, agent_name, baseline)
        baseline_passed = call.args[2] if len(call.args) >= 3 else call.kwargs.get("baseline_trace_id")
        assert baseline_passed == "concurrent", (
            "the pre-agent baseline should be the concurrent trace id"
        )

    assert captured_link["replayed"] == "replay-fresh", (
        "replay must link the freshly-produced trace, not the concurrent one"
    )
    assert results.completed == 1
