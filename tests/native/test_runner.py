"""The native scenario runner: a child per scenario, tracing on, crashes surface."""
from __future__ import annotations

import pytest

from ._runner import ScenarioError, run_scenario


def test_a_scenario_runs_in_the_project_with_tracing_on(native_project, monkeypatch):
    monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_from_the_developer")
    got = run_scenario("tests.native._scenario_selftest:echo", {"x": 1}, cwd=native_project.root)
    assert got["cwd"] == str(native_project.root)
    assert got["ctx"] == {"x": 1}
    assert got["tracing_off"] is None
    assert got["decimal_env"] == []


def test_a_crash_is_a_failure_not_a_skip(native_project):
    with pytest.raises(ScenarioError, match="scenario failed on purpose"):
        run_scenario("tests.native._scenario_selftest:boom", {}, cwd=native_project.root)
