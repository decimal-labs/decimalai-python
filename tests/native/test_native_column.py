"""The native conformance column: NC1-NC6 for every framework whose own loader reads
``.agents/skills``, graded on the traces DecimalAI sent.

Each framework's scenario (``_scenario_<framework>.py:column``) runs once per session in a
child process of its own, in the test environment that installs it. A framework graded in
the other environment is skipped with a reason naming it; a declared N/A must still fail,
so the ledger shrinks the day the gap closes. Collected by default: CI's floors jobs run it
at the lowest declared versions, which is what "at the floor versions" means.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from tests.native_loaders import ledger

from . import column_contract as contract
from ._runner import run_scenario
from .conftest import missing_imports

#: framework (its LOADER_ENVS name) -> its scenario module under tests/native.
FRAMEWORKS: Dict[str, str] = {
    "claude-agent-sdk": "_scenario_claude_agent_sdk",
    "google-adk": "_scenario_adk",
    "crewai": "_scenario_crewai",
    "openai-agents": "_scenario_openai_agents",
    "deepagents": "_scenario_deepagents",
    "agent-framework": "_scenario_agent_framework",
    "pydantic-ai-harness": "_scenario_pydantic_ai_harness",
    "strands-agents": "_scenario_strands",
}

CELLS = [(fw, item) for fw in FRAMEWORKS for item in contract.ITEMS]

_results: Dict[str, Any] = {}


@pytest.fixture
def column_result(native_project, request):
    framework = request.node.callspec.params["framework"]
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(framework):
        ledger.skip_declared("elsewhere", f"{framework}:column",
                             f"{framework}'s native column")
    # Absent here: allowed only when no test environment is named (CI's `.[dev]` floors
    # job, a bare local run); under DECIMAL_TEST_ENV it fails, as everywhere in this suite.
    missing = missing_imports(framework)
    if missing:
        ledger.skip_declared("unavailable", f"{framework}:column",
                             f"{framework} is not installed (no {', '.join(missing)})")
    if framework not in _results:
        _results[framework] = run_scenario(
            f"tests.native.{FRAMEWORKS[framework]}:column", {}, cwd=native_project.root)
    return _results[framework]


@pytest.mark.parametrize(("framework", "item"), CELLS, ids=[f"{f}:{i}" for f, i in CELLS])
def test_native_column_cell(column_result, framework, item):
    reason = contract.declared_na(framework, item)
    if reason is None:
        contract.grade(item, column_result)
        return
    try:
        contract.grade(item, column_result)
    except AssertionError:
        pytest.xfail(f"declared N/A: {reason}")
    pytest.fail(f"{framework}:{item} passes but is declared N/A ({reason}): remove it from "
                "column_contract.DECLARED_NA")


def test_the_na_ledger_stays_within_its_budget():
    assert len(contract.DECLARED_NA) <= contract.NA_BUDGET, (
        f"{len(contract.DECLARED_NA)} declared N/A cells, budget {contract.NA_BUDGET}")


def test_every_framework_the_native_loader_suite_grades_has_a_column():
    from tests.conformance.environments import LOADER_ENVS

    assert set(FRAMEWORKS) == set(LOADER_ENVS)
