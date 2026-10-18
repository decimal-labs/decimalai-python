"""OpenAI Agents SDK native mode: the sandbox's own ``Skills`` capability loads the skill,
the adapter's hook turns the load into an activation, and the router stays out of it.

The scenarios (``_scenario_openai_agents``) run in a child: a ``SandboxAgent`` with
``sandbox_skills()`` in a local Unix sandbox, the SDK's ``ScriptedModel``,
``decimalai.init(openai_agents=True)`` pointed at a probe whose router holds two skills of
its own (so an injection would show). ``native``: run 1 loads the target, reads its
``SKILL.md``, loads it again, reads its resource and runs its script; then
``decimalai skills update`` replaces it with version 2 and run 2, with the same agent,
loads and reads it again. ``two_catalogs``: the capability is hidden from the adapter, so
the router injects and the sandbox loads in the same run.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from decimalai.skillpkg.identity import ManagedSkill
from tests.native_loaders import ledger

from ._runner import run_scenario
from .conftest import V2_SENTINEL

FRAMEWORK = "openai-agents"
_results: Dict[str, Any] = {}


def _scenario(native_project, name: str) -> Dict[str, Any]:
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "OpenAI Agents native mode")
    if name not in _results:
        _results[name] = run_scenario(f"tests.native._scenario_openai_agents:{name}", {},
                                      cwd=native_project.root)
    return _results[name]


@pytest.fixture
def result(native_project):
    return _scenario(native_project, "native")


@pytest.fixture
def two_catalogs(native_project):
    return _scenario(native_project, "two_catalogs")


def _run(result: Dict[str, Any], i: int):
    run = result["runs"][i]
    assert run["error"] is None, run["error"]
    (trace,) = run["traces"]
    return run, trace


def _wire(lock: dict, install_name: str) -> str:
    e = lock["skills"][install_name]
    return ManagedSkill(install_name=install_name, registry_name=e["name"],
                        skill_id=e["skill_id"], version_id=e["version_id"],
                        body_sha256=e["body_sha256"], path="").wire_name


def test_a_load_is_an_activation_of_the_installed_version(result, native_project):
    _, trace = _run(result, 0)
    want = native_project.expected()
    assert trace["active_skills"] == [want]
    assert want["name"] in trace["skills_loaded_by_agent"]
    assert want["name"] in trace["skills_delivered"]


def test_two_loads_in_a_run_are_one_activation(result, native_project):
    run, trace = _run(result, 0)
    assert run["target_loads"] == 2
    statuses = [o for o in result["notes"]["outputs"][0] if "'status'" in o]
    assert ["'loaded'" in statuses[0], "'already_loaded'" in statuses[1]] == [True, True]
    assert trace["active_skills"] == [native_project.expected()]


def test_the_listing_names_every_installed_skill(result, native_project):
    """The sandbox lists the user-only skill to the model too (it ignores
    ``disable-model-invocation``: a ledgered upstream gap), so it is offered."""
    _, trace = _run(result, 0)
    assert set(trace["skills_offered_in_prompt"]) == {
        _wire(native_project.lock, n) for n in native_project.lock["skills"]}


def test_after_skills_update_the_activation_names_the_version_the_model_received(
        result, native_project):
    """The sandbox delivers v2: each run opens a fresh sandbox session, and
    ``load_skill`` copies the folder from disk into it at load time. (A session kept
    across runs would keep its v1 copy; the hook reads that copy, so it would say v1.)"""
    run, trace = _run(result, 1)
    received = "\n".join(run["received"])
    assert V2_SENTINEL in received and native_project.sentinel() not in received
    v2 = native_project.update_to_v2()
    assert trace["active_skills"] == [v2]


def test_router_injection_is_off_for_an_agent_with_the_sandbox_skills_capability(result):
    notes = result["notes"]
    assert notes["route_calls"] == 0
    for system in notes["system"]:
        assert "## Skills" in system  # the sandbox's own listing
        assert not any(name in system for name in notes["router_skills"])
        assert "decimalai_load_skill" not in system
    # Exactly the sandbox's tools: its shell and its own load_skill, none of ours.
    assert notes["tools"] == [["exec_command", "load_skill", "write_stdin"]] * 2
    said = [line for line in notes["logs"] if "sandbox Skills capability" in line["message"]]
    assert [line["level"] for line in said] == ["INFO"]


def test_a_native_load_on_a_run_the_router_injected_into_warns_once(two_catalogs,
                                                                    native_project):
    for i, system in enumerate(two_catalogs["notes"]["system"]):
        assert "conformance-skill-alpha" in system and "## Skills" in system
        _, trace = _run(two_catalogs, i)
        assert trace["active_skills"] == [native_project.expected()]
    warned = [line for line in two_catalogs["notes"]["logs"]
              if "two skill catalogs reached the model" in line["message"]]
    assert [line["level"] for line in warned] == ["WARNING"]


def test_the_pointer_helper_lists_the_installed_skills(result, native_project):
    from decimalai.openai_agents import sandbox_skills

    listed = sandbox_skills().lazy_from.list_skill_metadata(skills_path=".agents")
    assert sorted(m.name for m in listed) == sorted(native_project.lock["skills"])
    # ...and the model was shown each of them, at the path load_skill stages it under.
    system = result["notes"]["system"][0]
    for name in native_project.lock["skills"]:
        assert f"- {name}: " in system and f"(file: .agents/{name})" in system
