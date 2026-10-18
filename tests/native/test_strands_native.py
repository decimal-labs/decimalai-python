"""Strands Agents in native mode: ``decimalai.strands.skills()`` on a real ``Agent``, graded
on the traces the probe received.

The framework runs in a child process per scenario (``_scenario_strands.py``):
``init(otel=True)`` installs a global tracer provider, Strands' tracer is a process-wide
singleton that reads its settings once, and these tests ride the default unit run. The
observer's decisions are tested here, in-process, on stand-ins for Strands' objects.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace
from typing import Any, Dict, List

from skillevaluation import skillfmt

import decimalai
from decimalai import strands as observed
from decimalai._native import BY_STAMP
from tests.native_loaders import ledger, oracle

from ._runner import run_scenario
from .conftest import V2_SENTINEL

FRAMEWORK = "strands-agents"
SC = oracle.SCENARIO
_results: Dict[str, Any] = {}


def _scenario(native_project, name: str) -> Dict[str, Any]:
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "Strands native mode")
    if importlib.util.find_spec("strands") is None:
        ledger.skip_declared("unavailable", f"{FRAMEWORK}:native",
                             "strands-agents is not importable")
    if name not in _results:
        _results[name] = run_scenario(f"tests.native._scenario_strands:{name}", {},
                                      cwd=native_project.root)
    return _results[name]


def _target(trace: Dict[str, Any], expected: Dict[str, str]) -> List[Dict[str, Any]]:
    return [e for e in trace.get("active_skills") or []
            if isinstance(e, dict) and e.get("name") == expected["name"]]


def _only_trace(run: Dict[str, Any]) -> Dict[str, Any]:
    assert not run["error"], run["error"]
    assert len(run["traces"]) == 1, f"one run, {len(run['traces'])} traces"
    return run["traces"][0]


# ── the run, end to end ──────────────────────────────────────────────────────


def test_a_load_is_one_activation_with_the_installed_identity(native_project):
    run = _scenario(native_project, "column")["runs"][0]
    trace = _only_trace(run)
    expected = native_project.expected()
    assert run["target_loads"] == 2  # the model loaded it twice ...
    assert _target(trace, expected) == [expected]  # ... and the trace counts it once
    assert expected["name"] in trace["skills_loaded_by_agent"]
    assert expected["name"] in trace["skills_offered_in_prompt"]
    assert trace["agent_name"] == "native-strands"
    assert len(trace["llm_calls"]) == 5  # the run's model calls ride along


def test_after_an_update_the_agents_v1_is_what_the_model_got_and_what_the_trace_says(
        native_project):
    result = _scenario(native_project, "column")
    v1 = native_project.expected()
    received = "\n".join(result["runs"][1]["received"])
    # AgentSkills reads an agent's skills once (init_agent) and keeps them, so the same
    # agent keeps serving v1 after `skills update` wrote v2. If this flips, Strands changed;
    # the activation must still follow what was received.
    assert native_project.sentinel() in received and V2_SENTINEL not in received
    trace = _only_trace(result["runs"][1])
    assert _target(trace, v1) == [v1]


def test_the_resource_read_and_the_script_run_work_through_the_helper(native_project):
    run = _scenario(native_project, "column")["runs"][0]
    received = "\n".join(run["received"])
    assert SC.resource_marker in received  # file_editor read it, at the Location given
    assert SC.script_out in received  # shell ran it there


def test_the_helper_is_the_agent_skills_plugin_and_lists_the_installed_skills(native_project):
    observed_ = _scenario(native_project, "listing")["observed"]
    assert observed_["is_agent_skills"] and observed_["plugin_name"] == "agent_skills"
    installed = sorted(native_project.lock["skills"])
    assert observed_["available"] == installed
    for install_name in installed:
        assert f"<name>{install_name}</name>" in observed_["system_prompt"]


def test_inside_the_users_own_trace_the_activation_lands_on_it(native_project):
    run = _scenario(native_project, "user_trace")["runs"][0]
    assert not run["error"], run["error"]
    expected = native_project.expected()
    mine = [t for t in run["traces"] if t.get("agent_name") == "users-own-trace"]
    assert len(mine) == 1, [t.get("agent_name") for t in run["traces"]]
    assert _target(mine[0], expected) == [expected]


# ── the observer's decisions, in-process ─────────────────────────────────────


def _parsed(native_project) -> SimpleNamespace:
    """The target as Strands parses it: name, body (instructions), metadata, folder."""
    folder = native_project.skills_dir / SC.target
    frontmatter, body = skillfmt.parse_skill_md_text((folder / "SKILL.md").read_text())
    return SimpleNamespace(name=SC.target, instructions=body.strip(),
                           metadata=frontmatter["metadata"], path=folder)


def _event(skill_name: str, text: str, status: str = "success") -> SimpleNamespace:
    return SimpleNamespace(
        tool_use={"name": observed.SKILLS_TOOL, "toolUseId": "t1",
                  "input": {"skill_name": skill_name}},
        result={"toolUseId": "t1", "status": status, "content": [{"text": text}]},
        agent=object())


def _loads_after(native_project, *events) -> List[Any]:
    skill = _parsed(native_project)
    plugin = SimpleNamespace(get_available_skills=lambda agent: [skill])
    with decimalai.start_trace(agent_name="observer-unit", auto_send=False) as ctx:
        for event in events:
            observed._observe_load(plugin, event)
        return ctx._native_loads().activations


def test_the_body_strands_served_is_resolved_by_the_parsed_stamp(native_project):
    skill = _parsed(native_project)
    answer = f"{skill.instructions}\n\n---\nLocation: {skill.path}/SKILL.md"
    [got] = _loads_after(native_project, _event(SC.target, answer))
    assert (got.entry(), got.how) == (native_project.expected(), BY_STAMP)


def test_nothing_is_recorded_when_the_body_did_not_reach_the_model(native_project):
    assert _loads_after(
        native_project,
        _event("acme-no-such-skill", "Skill 'acme-no-such-skill' not found."),
        _event(SC.target, "rewritten by another hook"),
        _event(SC.target, _parsed(native_project).instructions, status="error"),
    ) == []
