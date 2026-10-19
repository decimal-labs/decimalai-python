"""Claude Agent SDK native mode: Claude Code's own plugin loader delivers the skill, the
adapter's stream hook turns the delivery into an activation, and the probe receives it.

The scenario (``_scenario_claude_agent_sdk.native``) runs in a child: the real bundled CLI
against the local Messages stub, ``decimalai.init(claude_agent_sdk=True)`` pointed at a
probe, the project loaded through ``skills_plugin()``. Run 1 loads the target with
``args``, loads it again, reads its resource and runs its script; then
``decimalai skills update`` replaces it with version 2 and run 2 loads it once more.

The last tests drive the hook in-process with synthetic messages: they patch nothing
global, so they need no child.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from decimalai import _native
from decimalai.skillpkg.identity import ManagedSkill
from tests.native_loaders import ledger

from ._runner import run_scenario
from .conftest import V2_SENTINEL

FRAMEWORK = "claude-agent-sdk"
PLUGIN = "decimalai-skills"
_results: Dict[str, Any] = {}


@pytest.fixture
def result(native_project):
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "Claude Agent SDK native mode")
    if "native" not in _results:
        _results["native"] = run_scenario(
            "tests.native._scenario_claude_agent_sdk:native", {}, cwd=native_project.root)
    return _results["native"]


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


def test_two_loads_in_a_run_are_one_activation_and_args_are_not_the_body(result, native_project):
    run, trace = _run(result, 0)
    assert run["target_loads"] == 2
    # The first load passed args, which the CLI appends to the body; the second one
    # delivered the body again (a re-invocation). Both reached the model...
    first, second = run["received"][:2]
    assert "ARGUMENTS: 100 45" in first and native_project.sentinel() in first
    assert native_project.sentinel() in second
    # ...and the run reports one activation, verified, so the args were not hashed.
    assert [e["name"] for e in trace["active_skills"]] == [native_project.expected()["name"]]
    assert trace["active_skills"][0]["version_id"] == native_project.expected()["version_id"]


def test_the_listing_names_the_installed_skills_the_model_can_invoke(result, native_project):
    _, trace = _run(result, 0)
    lock = native_project.lock["skills"]
    model_invocable = {_wire(native_project.lock, n) for n, e in lock.items()
                       if e.get("invocation") != "user"}
    # Claude Code keeps a user-only skill from the model, so it is not "offered";
    # Claude Code's own skills and other plugins' are not the platform's to attribute.
    assert set(trace["skills_offered_in_prompt"]) == model_invocable
    assert _wire(native_project.lock, "acme-returns-desk") not in model_invocable


def test_after_skills_update_the_activation_names_the_version_the_model_received(
        result, native_project):
    """Claude Code delivers v2: each ``query()`` starts a CLI of its own, which reads the
    skill folder when the model loads it. (A cached v1 would have to be attributed v1;
    the resolver's seen-hash step is what does that, see ``test_native_observer``.)"""
    run, trace = _run(result, 1)
    received = "\n".join(run["received"])
    assert V2_SENTINEL in received and native_project.sentinel() not in received
    v2 = native_project.update_to_v2()
    assert trace["active_skills"] == [v2]
    assert result["notes"]["v2"] == v2


def test_the_pointer_helper_lists_the_installed_skills(result, native_project):
    from decimalai.claude_agent_sdk import skills_plugin

    assert skills_plugin() == {"type": "local", "path": str(native_project.skills_dir.parent)}
    notes = result["notes"]
    ours = [p for p in notes["plugins"] or [] if p.get("name") == PLUGIN]
    assert [p["path"] for p in ours] == [notes["plugin_path"]], notes["plugins"]
    assert not notes["plugin_errors"]
    listed = {n.split(":", 1)[1] for n in notes["init_skills"] if n.startswith(PLUGIN + ":")}
    assert listed == set(native_project.lock["skills"])


# ── the stream hook, in-process ─────────────────────────────────────────────


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ToolUseBlock:
    def __init__(self, id: str, name: str, input: dict) -> None:  # noqa: A002
        self.id, self.name, self.input = id, name, input


class ToolResultBlock:
    def __init__(self, tool_use_id: str, content: Any) -> None:
        self.tool_use_id, self.content, self.is_error = tool_use_id, content, None


class AssistantMessage:
    def __init__(self, content: List[Any]) -> None:
        self.content, self.model, self.usage = content, "claude", None


class UserMessage:
    def __init__(self, content: Any) -> None:
        self.content = content


def _feed(*messages: Any):
    from decimalai import claude_agent_sdk as cas

    state = cas._RunState(agent_name="t", parent_trace_id=None,
                          user_input="q", options=None)
    for m in messages:
        cas._ingest_message(state, m)
    return state.native.activations


def _delivery(native_project, body_extra: str = "") -> str:
    from .conftest import body_of

    folder = native_project.skills_dir / native_project.target
    body = body_of((folder / "SKILL.md").read_bytes())
    return f"Base directory for this skill: {folder}\n\n{body}{body_extra}"


def _skill_call(i: str, **inp: Any) -> AssistantMessage:
    return AssistantMessage([ToolUseBlock(i, "Skill", {"skill": f"{PLUGIN}:acme-refund-policy",
                                                       **inp})])


def test_a_call_the_cli_answers_already_loaded_is_not_a_load(native_project):
    got = _feed(_skill_call("t1"), UserMessage([ToolResultBlock("t1", "Launching skill")]),
                UserMessage([TextBlock("Skill /decimalai-skills:acme-refund-policy is "
                                       "already loaded above; instructions unchanged.")]))
    assert got == []


def test_a_body_in_string_content_or_in_the_tool_result_is_read(native_project):
    want = native_project.expected()
    as_string = _feed(_skill_call("t1"), UserMessage(_delivery(native_project)))
    in_result = _feed(_skill_call("t2"),
                      UserMessage([ToolResultBlock("t2", _delivery(native_project))]))
    for got in (as_string, in_result):
        assert [(a.entry(), a.how) for a in got] == [(want, _native.BY_LOCK)]


def test_args_from_a_call_the_stream_did_not_show_are_still_taken_off(native_project):
    got = _feed(UserMessage([TextBlock(_delivery(native_project, "\n\n\nARGUMENTS: 7 days"))]))
    assert [a.entry() for a in got] == [native_project.expected()]


def test_a_body_that_is_no_known_version_names_the_skill_but_no_version(native_project):
    got = _feed(_skill_call("t1"),
                UserMessage([TextBlock(_delivery(native_project, "\nEdited by hand."))]))
    (activation,) = got
    assert activation.how == _native.UNVERIFIED and activation.version_id is None
    assert activation.skill_id == native_project.expected()["skill_id"]
