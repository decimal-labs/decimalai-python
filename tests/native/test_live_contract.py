"""Every live-check item is seen to fail, and the rate and the provider wall are graded as
the contract says."""
from __future__ import annotations

import copy

import pytest

from . import column, live, live_contract as contract
from .column_contract import V1


def _run(**over):
    run = {"traces": [{"active_skills": [dict(V1)]}],
           "calls": [{"name": "load_skill", "args": {"skill_name": column.TARGET}}],
           "answer": "Under our policy, 31-60 days gets store credit, so $100 store credit.",
           "error": None, "received": [], "target_loads": 1}
    run.update(over)
    return run


def test_a_good_trial_passes():
    assert contract.grade_trial("google-adk", _run()) == []


@pytest.mark.parametrize("item,defect", [
    ("L1", {"calls": [{"name": "load_skill", "args": {"skill_name": "other"}}]}),
    ("L1", {"calls": []}),
    ("L2", {"traces": [{"active_skills": [dict(V1, version_id="other")]}]}),
    ("L2", {"traces": [{"active_skills": []}]}),
    ("L3", {"answer": "You can get a full refund."}),
])
def test_each_defect_fails_its_item(item, defect):
    failed = contract.grade_trial("google-adk", _run(**defect))
    assert [f.split(":")[0] for f in failed] == [item]


def test_the_load_tool_is_each_frameworks_own():
    deep = _run(calls=[{"name": "read_file",
                        "args": {"file_path": f"/p/.agents/skills/{column.TARGET}/SKILL.md"}}])
    assert contract.grade_trial("deepagents", deep) == []
    assert contract.grade_trial("google-adk", deep)[0].startswith("L1")


def test_two_of_three_trials_pass_a_framework():
    bad = _run(answer="no idea")
    assert contract.grade("crewai", {"runs": [_run(), _run(), bad]})[0] == "pass"
    assert contract.grade("crewai", {"runs": [_run(), bad, bad]})[0] == "fail"


def test_a_provider_wall_is_a_skip_never_a_pass():
    wall = _run(error="RateLimitError: insufficient_quota")
    assert contract.grade("crewai", {"runs": [wall, wall, wall]})[0] == "skip"
    assert live.provider_wall("Your credit balance is too low")
    # One refused trial does not count against the framework, nor for it.
    assert contract.grade("crewai", {"runs": [wall, _run(), _run()]})[0] == "pass"
    assert contract.grade("crewai", {"runs": [wall, _run(), _run(answer="no")]})[0] == "fail"
