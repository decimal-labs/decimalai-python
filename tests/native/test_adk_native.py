"""google-adk native mode: ADK's own ``SkillToolset`` loads the installed skills, and the
trace DecimalAI sends names the version the model received.

The scenario (``_scenario_adk.native``) runs once, in a child process: the real ADK
runner, ``SkillToolset(skills=decimalai.adk.skills())``, a scripted model, and
``decimalai.adk.DecimalAIPlugin`` in the App, pointed at a probe. The toolset agent also
carries ``decimalai.adk.agent_config()``'s instruction and tools. Run 1 lists, loads the
target twice, reads its resource and runs its script; then the target is updated to
version 2 on disk; run 2 loads it again with the same agent object. A third run, an
agent with ``agent_config()``'s pieces and no toolset, is the control that shows
``agent_config()`` lists skills at all. Everything below grades what the probe received
and what the scripted model was handed.

The pointer helper's tests run in this process: they only read files.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from tests.conformance.probe import MENU_INSTRUCTION
from tests.native_loaders import ledger

from ._runner import run_scenario
from ._scenario_adk import ROUTER_SENTINEL, ROUTER_SKILL
from .conftest import V2_SENTINEL

FRAMEWORK = "google-adk"


@pytest.fixture(autouse=True)
def _adk_is_graded_here():
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "ADK native mode")


_result: Dict[str, Any] = {}


@pytest.fixture
def scenario(native_project) -> Dict[str, Any]:
    """The scenario's result, run once per session (it costs a child process)."""
    if not _result:
        _result.update(run_scenario("tests.native._scenario_adk:native", {},
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


def test_a_load_is_an_activation_of_the_installed_version(scenario, native_project):
    trace = _trace(scenario, 0)
    want = native_project.expected()
    assert trace["active_skills"] == [want]
    assert want["name"] in trace["skills_loaded_by_agent"]
    # Listed by list_skills, under the same (wire) name as the activation.
    assert want["name"] in trace["skills_offered_in_prompt"]
    assert native_project.target not in trace["skills_offered_in_prompt"]


def test_the_same_skill_loaded_twice_in_one_run_is_one_activation(scenario, native_project):
    run = _run(scenario, 0)
    bodies = [t for t in run["received"] if native_project.sentinel() in t]
    assert len(bodies) == 2, "the model did not receive the body twice"
    assert [e["name"] for e in run["traces"][0]["active_skills"]] == [
        native_project.expected()["name"]]


def test_after_an_update_the_activation_names_the_version_the_model_received(
        scenario, native_project):
    """ADK parsed the folders when ``decimalai.adk.skills()`` ran, so the agent built
    before ``skills update`` keeps serving version 1; the activation says version 1,
    although the folder and the lock now say version 2."""
    v1 = native_project.expected()
    received = "\n".join(_run(scenario, 1)["received"])
    got_v1, got_v2 = native_project.sentinel() in received, V2_SENTINEL in received
    assert (got_v1, got_v2) == (True, False), "ADK no longer serves the body it parsed"
    trace = _trace(scenario, 1)
    assert trace["active_skills"] == [v1]
    assert scenario["notes"]["v2_expected"]["version_id"] != v1["version_id"]


def test_load_agent_gives_the_agent_with_a_skill_toolset_no_decimal_skills(scenario):
    native = _calls(scenario, "support")
    assert native, "the native agent made no model call"
    for call in native:
        for text in (ROUTER_SENTINEL, MENU_INSTRUCTION, ROUTER_SKILL["name"]):
            assert text not in call["system"], f"agent_config() put {text!r} in the request"
    for i in (0, 1):
        trace = _trace(scenario, i)
        assert trace.get("routing_id") is None
        assert ROUTER_SKILL["name"] not in (trace.get("skills_delivered") or [])
    said = [l for l in scenario["logs"] if l["level"] == "WARNING"
            and "SkillToolset" in l["message"]]
    assert len(said) == 1 and "(support)" in said[0]["message"], said


def test_load_agent_still_lists_skills_for_an_agent_without_a_toolset(scenario):
    """The control: without it, the test above would pass on an agent config that lists nothing."""
    (call,) = _calls(scenario, "plain")
    assert ROUTER_SKILL["name"] in call["system"] and MENU_INSTRUCTION in call["system"]
    assert ROUTER_SENTINEL not in call["system"], "a body arrives by load_skill, not in the prompt"
    assert _trace(scenario, 2)["routing_id"]


def test_no_two_catalogs_warning_when_only_the_toolset_served(scenario):
    assert not [l for l in scenario["logs"] if "two skill catalogs" in l["message"]]


# ── the pointer helper ──────────────────────────────────────


def test_skills_lists_every_installed_skill_and_adk_offers_them(scenario, native_project):
    import decimalai.adk

    installed = sorted(native_project.lock["skills"])
    assert sorted(s.name for s in decimalai.adk.skills()) == installed
    listing = _run(scenario, 0)["received"][0]
    for name in installed:
        assert f"<name>\\n{name}\\n</name>" in listing, f"list_skills did not offer {name}"


def test_skills_skips_a_bad_folder_with_one_warning(native_project, caplog):
    from google.adk.skills import load_skills_from_dir

    import decimalai.adk

    bad = native_project.skills_dir / "zz-broken"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: not-the-folder\ndescription: x\n---\nbody\n")
    with pytest.raises(ValueError):
        load_skills_from_dir(native_project.skills_dir)  # ADK alone loses every skill

    with caplog.at_level(logging.WARNING, logger="decimalai.adk"):
        got = decimalai.adk.skills()
    assert sorted(s.name for s in got) == sorted(native_project.lock["skills"])
    warned = [r.getMessage() for r in caplog.records
              if r.name == "decimalai.adk" and r.levelno == logging.WARNING]
    assert len(warned) == 1 and "'zz-broken'" in warned[0], warned


def test_the_catalog_parser_reads_adks_own_rendering():
    from google.adk.skills import Frontmatter
    from google.adk.skills.prompt import format_skills_as_xml

    from decimalai.adk import _catalog_names

    xml = format_skills_as_xml([Frontmatter(name="a-b", description="d"),
                                Frontmatter(name="c", description="e <f>")])
    assert _catalog_names(xml) == ["a-b", "c"]
    assert _catalog_names("no listing here") == []
