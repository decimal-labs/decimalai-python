"""crewai native mode: CrewAI's own ``Agent(skills=...)`` loads the installed skills, and
the trace DecimalAI sends names the version the model received.

The scenario (``_scenario_crewai.native``) runs once, in a child process: a real
``Crew.kickoff()``, ``Agent(skills=decimalai.crewai.skills(),
tools=decimalai.crewai.skill_tools())``, a scripted ``BaseLLM``, and
``decimalai.init(crewai=True)`` pointed at a probe. Run 1 loads the target twice,
reads its resource and runs its script; then the target is updated to version 2 on
disk; run 2 loads it again with the same agent object. Then an agent whose skill is
ACTIVATED (always-on: no tool call, only ``SkillUsedEvent``), and a control agent with
no skills of its own, which shows the router is on at all.

CrewAI is traced through OpenTelemetry, so the carrier of a load is the run's
OpenTelemetry rail. The ``post_tool_call`` hook runs on the run's thread and CrewAI
runs an event handler in a copy of the emitting context, so both see that run as
current (measured: the same trace id as the run's spans); these tests prove each
load reaches the trace the exporter sends.

The pointer helpers' tests run in this process: they only read files and run the
fixture's script.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Any, Dict, List

import pytest

from tests.conformance.probe import MENU_INSTRUCTION
from tests.native_loaders import ledger, oracle

from ._runner import run_scenario
from ._scenario_crewai import ROUTER_SENTINEL, ROUTER_SKILL
from .conftest import V2_SENTINEL

FRAMEWORK = "crewai"

# Read by crewai at import (the scenario child gets the same from its runner).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


@pytest.fixture(autouse=True)
def _crewai_is_graded_here():
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "CrewAI native mode")


_result: Dict[str, Any] = {}


@pytest.fixture
def scenario(native_project) -> Dict[str, Any]:
    """The scenario's result, run once per session (it costs a child process)."""
    if not _result:
        _result.update(run_scenario("tests.native._scenario_crewai:native", {},
                                    cwd=native_project.root))
    return _result


def _run(scenario, i: int) -> Dict[str, Any]:
    run = scenario["runs"][i]
    assert not run["error"], run["error"]
    assert len(run["traces"]) == 1, f"run {i + 1} sent {len(run['traces'])} traces"
    return run


def _trace(scenario, i: int) -> Dict[str, Any]:
    return _run(scenario, i)["traces"][0]


def _calls(scenario, agent: str) -> List[Dict[str, Any]]:
    return [c for c in scenario["calls"] if c["agent"] == agent]


def _wire(native_project, install_name: str) -> str:
    return native_project.expected(install_name)["name"]


def test_a_load_is_an_activation_of_the_installed_version(scenario, native_project):
    trace = _trace(scenario, 0)
    want = native_project.expected()
    assert trace["active_skills"] == [want]
    assert want["name"] in trace["skills_loaded_by_agent"]


def test_the_catalog_is_recorded_as_offered_under_the_wire_names(scenario, native_project):
    offered = _trace(scenario, 0)["skills_offered_in_prompt"]
    assert sorted(offered) == sorted(_wire(native_project, n) for n in native_project.lock["skills"])


def test_the_same_skill_loaded_twice_in_one_run_is_one_activation(scenario, native_project):
    run = _run(scenario, 0)
    bodies = [t for t in run["received"] if native_project.sentinel() in t]
    assert len(bodies) == 2, "the model did not receive the body twice"
    assert [e["name"] for e in run["traces"][0]["active_skills"]] == [
        native_project.expected()["name"]]


def test_after_an_update_the_activation_names_the_version_the_model_received(
        scenario, native_project):
    """CrewAI parsed the frontmatter when the agent was built, but reads the body from
    disk on every load: the same agent delivers version 2 after ``skills update``, with
    version 1's stamp still in its parsed frontmatter. The activation goes by the body
    the model got, so it names version 2."""
    v1 = native_project.expected()
    received = "\n".join(_run(scenario, 1)["received"])
    got_v1, got_v2 = native_project.sentinel() in received, V2_SENTINEL in received
    assert (got_v1, got_v2) == (False, True), "CrewAI no longer re-reads the body on load"
    v2 = scenario["notes"]["v2_expected"]
    assert v2["version_id"] != v1["version_id"]
    assert _trace(scenario, 1)["active_skills"] == [v2]


def test_an_always_on_skill_is_recorded_from_skill_used_event(scenario):
    """No tool call delivers an activated skill; ``SkillUsedEvent`` announces it, and
    its handler runs on CrewAI's handler pool. The activation still lands on the
    trace of the run that used it."""
    run = _run(scenario, 2)
    assert V2_SENTINEL in run["received"][0]
    assert run["traces"][0]["active_skills"] == [scenario["notes"]["v2_expected"]]


def test_the_router_stays_out_of_agents_with_crewai_skills(scenario):
    for agent in ("Support", "Desk"):
        calls = _calls(scenario, agent)
        assert calls, f"{agent} made no model call"
        for call in calls:
            for text in (ROUTER_SENTINEL, MENU_INSTRUCTION, ROUTER_SKILL["name"]):
                assert text not in call["system"], f"the router injected {text!r} for {agent}"
    for i in (0, 1, 2):
        trace = _trace(scenario, i)
        assert trace.get("routing_id") is None
        assert ROUTER_SKILL["name"] not in (trace.get("skills_delivered") or [])
    said = [l["message"] for l in scenario["logs"] if l["level"] == "INFO"
            and "does not inject" in l["message"]]
    assert len(said) == 2, said  # once per agent, though Support made seven calls
    assert "'Support'" in said[0] and "load_skill" in said[0]
    assert "'Desk'" in said[1] and "always-on" in said[1]


def test_the_router_still_serves_an_agent_without_crewai_skills(scenario):
    """The control: without it, the test above would pass on a router that is off."""
    (call,) = _calls(scenario, "Plain")
    assert ROUTER_SENTINEL in call["system"] and MENU_INSTRUCTION in call["system"]
    assert _trace(scenario, 3)["routing_id"]


def test_no_warning_when_only_crewai_served_the_skills(scenario):
    assert not [l for l in scenario["logs"] if l["level"] == "WARNING"]


def test_the_resource_and_the_script_ran_through_the_skill_tools(scenario):
    received = _run(scenario, 0)["received"]
    assert received[2] == oracle.FEES
    assert received[3].startswith(oracle.SCENARIO.script_out + "\n")
    assert received[3].endswith("[exit status 0]")


# ── the pointer helpers, in this process ────────────────────


def test_skills_points_crewai_at_the_installed_skills(native_project):
    from crewai.skills.loader import discover_skills

    import decimalai.crewai

    (root,) = decimalai.crewai.skills()
    assert root == native_project.skills_dir
    assert sorted(s.name for s in discover_skills(root)) == sorted(native_project.lock["skills"])


@pytest.fixture
def tools(native_project):
    import decimalai.crewai

    return {t.name: t for t in decimalai.crewai.skill_tools()}


def test_skill_tools_read_a_resource_and_run_a_script(tools, native_project):
    sc = oracle.SCENARIO
    assert set(tools) == {"load_skill_resource", "run_skill_script"}
    assert tools["load_skill_resource"].run(skill_name=sc.target, file_path=sc.resource) == oracle.FEES
    out = tools["run_skill_script"].run(skill_name=sc.target, file_path=sc.script,
                                        args=list(sc.script_args))
    assert out == f"{sc.script_out}\n\n[exit status 0]"
    # CrewAI's tool cache must not answer a later call with an earlier file.
    assert not tools["load_skill_resource"].cache_function({}, "x")
    assert not tools["run_skill_script"].cache_function({}, "x")


@pytest.mark.parametrize("skill, path", [
    ("acme-refund-policy", "../acme-returns-desk/SKILL.md"),   # another skill
    ("acme-refund-policy", "SKILL.md"),                        # not a resource folder
    ("acme-refund-policy", "/etc/hosts"),                      # absolute, outside
    ("acme-refund-policy", "references/../../../.decimal/skills.lock"),
    ("..", "skills.lock"),
    ("no-such-skill", "references/fees.md"),
])
def test_skill_tools_stay_inside_an_installed_skill(tools, skill, path):
    for name in tools:
        got = tools[name].run(skill_name=skill, file_path=path)
        assert got.startswith("Error: "), (name, skill, path, got[:200])


def test_a_symlink_out_of_the_skill_is_refused(tools, native_project, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("not a skill file")
    link = native_project.skills_dir / native_project.target / "references" / "leak.md"
    link.symlink_to(outside)
    got = tools["load_skill_resource"].run(skill_name=native_project.target,
                                           file_path="references/leak.md")
    assert got.startswith("Error: ") and "not a skill file" not in got


def test_skill_tools_without_scripts(native_project):
    import decimalai.crewai

    assert [t.name for t in decimalai.crewai.skill_tools(scripts=False)] == ["load_skill_resource"]


def test_the_body_is_read_back_out_of_crewais_own_rendering(native_project):
    from crewai.skills.loader import activate_skill, discover_skills, format_skill_context

    from decimalai.crewai import _body_in_skill_context

    (target,) = [s for s in discover_skills(native_project.skills_dir)
                 if s.name == native_project.target]
    activated = activate_skill(target)
    text = format_skill_context(activated, label=target.name)
    assert _body_in_skill_context(text, target.name, target) == activated.instructions
    assert _body_in_skill_context("Skill 'x' is not available.", target.name, target) is None


def test_a_native_load_in_a_routed_run_warns_once(monkeypatch, caplog):
    import decimalai.crewai as dc

    monkeypatch.setattr(dc, "_routed_runs", OrderedDict({7: None}))
    monkeypatch.setattr(dc, "_native_skills_warned", False)
    with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
        dc._warn_if_two_catalogs(8, "a")   # a run the router never served
        dc._warn_if_two_catalogs(7, "a")
        dc._warn_if_two_catalogs(7, "b")
    found = [r.getMessage() for r in caplog.records if r.name == "decimalai.crewai"]
    assert len(found) == 1 and "two skill catalogs" in found[0]
