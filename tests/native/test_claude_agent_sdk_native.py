"""Claude Agent SDK native mode: Claude Code's own plugin loader delivers the skill, the
adapter's stream hook turns the delivery into an activation, and the probe receives it.

The scenario (``_scenario_claude_agent_sdk.native``) runs in a child: the real bundled CLI
against the local Messages stub, ``decimalai.init(claude_agent_sdk=True)`` pointed at a
probe, the project loaded through ``skills_plugin()``. Run 1 loads the target with
``args``, loads it again, reads its resource and runs its script; then
``decimalai skills update`` replaces it with version 2 and run 2 loads it once more.

The last tests drive the hook in-process with synthetic messages: they patch nothing
global, so they need no child.

The CLI rewrites every body it delivers (it fills ``$ARGUMENTS``, ``$0``, ``$1``, ... and
``${CLAUDE_SKILL_DIR}``), so a body with placeholders reaches the model as bytes no version
hashes to. ``rewritten`` loads such skills through the real CLI, and the replays after it feed
the hook the CLI's recorded output (``tests/fixtures/claude_skill_args_2_1_299.json``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from decimalai import _native
from decimalai.skillpkg import identity
from decimalai.skillpkg.identity import ManagedSkill
from tests.native_loaders import ledger, oracle

from ._runner import run_scenario
from .conftest import V2_SENTINEL, NativeProject

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


# ── bodies the CLI rewrites ─────────────────────────────────────────────────

FIXTURE = json.loads((Path(__file__).resolve().parents[1] / "fixtures"
                      / "claude_skill_args_2_1_299.json").read_text())
RECORDED = {c["case"]: c for c in FIXTURE["renders"]}


@dataclasses.dataclass(frozen=True)
class _Row(oracle.SkillRow):
    """An oracle skill with a body of its own and, optionally, an ``arguments`` line."""

    body: str = ""
    arguments: str = ""

    @property
    def stored_body(self) -> str:
        return self.body


def _row(i: int, name: str, body: str, arguments: str = "") -> _Row:
    base = oracle._row(0x40 + i, name=name, title=name, sentinel=f"SENTINEL-REWRITE-{i}",
                       allowed_tools=(), invocation_mode="model", variant="authored",
                       public=False, pin="latest")
    return _Row(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                body=body, arguments=arguments)


#: ``$12`` is a price, but the 13th word of a long args fills it.
LATE_FEES = ("# Late fees\n\nA late fee is $12, waived once a year. Refunds over $100 need a "
             "manager.\nThe customer wrote: \"$ARGUMENTS\". The fee table is in "
             "${CLAUDE_SKILL_DIR}/references.\n")
LATE_ARGS = "Write a reply to a customer who paid on time but was charged a late fee"
CITY_GUIDE = ("# City guide\n\nRecommend three sights in $city for session ${CLAUDE_SESSION_ID}"
              " of ${CLAUDE_PROJECT_DIR}.\n")
REFUND_TOTAL = "# Refund total\n\nAdd \\$5 to every refund. Asked for: [$ARGUMENTS].\n"
PLAIN = "# Plain\n\nNo placeholders in this one.\n"
ROWS = (
    _row(0, "acme/late-fees", LATE_FEES),
    _row(1, "acme/city-guide", CITY_GUIDE, "arguments: [city]"),
    _row(2, "acme/refund-total", REFUND_TOTAL),
    _row(3, "acme/plain", PLAIN),
    _row(4, "acme/full-placeholders", RECORDED["full_args"]["body"],
         RECORDED["full_args"]["frontmatter"].strip()),
)
LOADS = [("acme-late-fees", LATE_ARGS), ("acme-city-guide", "Lisbon"),
         ("acme-refund-total", None), ("acme-plain", "!urgent refund")]


@pytest.fixture
def placeholder_project(tmp_path, monkeypatch) -> NativeProject:
    root = (tmp_path / "project").resolve()
    root.mkdir()
    (root / ".git").mkdir()
    lock = oracle.write(root, ROWS)
    for row in ROWS:
        if row.arguments:
            path = root / oracle.ROOT / oracle.install_name_for(row.name, row.skill_id) / "SKILL.md"
            text = path.read_text()
            path.write_text(text.replace("\nmetadata:\n", f"\n{row.arguments}\nmetadata:\n", 1))
    monkeypatch.chdir(root)
    identity.clear_cache()
    _native._reset_seen()
    yield NativeProject(root=root, lock=lock)
    identity.clear_cache()
    _native._reset_seen()


@pytest.fixture
def rewritten(placeholder_project):
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "Claude Agent SDK native mode")
    return run_scenario("tests.native._scenario_claude_agent_sdk:rewritten", {"loads": LOADS},
                        cwd=placeholder_project.root)


def test_bodies_the_cli_rewrote_are_activations_of_the_installed_versions(
        rewritten, placeholder_project):
    run = rewritten["runs"][0]
    assert run["error"] is None, run["error"]
    (trace,) = run["traces"]
    received = "\n".join(run["received"])
    skill_dir = placeholder_project.skills_dir / "acme-late-fees"
    # The CLI rewrote each body, so none of them is a locked body byte for byte...
    assert "A late fee is charged, waived once a year. Refunds over $100" in received
    assert f'The customer wrote: "{LATE_ARGS}". The fee table is in {skill_dir}/' in received
    assert "Recommend three sights in Lisbon for session " in received
    assert f" of {placeholder_project.root}." in received
    assert "Add $5 to every refund. Asked for: []." in received
    assert "ARGUMENTS: \\!urgent refund" in received
    # ...and each activation still names the installed version exactly.
    by_name = sorted(trace["active_skills"], key=lambda e: e["name"])
    want = sorted((placeholder_project.expected(n) for n, _ in LOADS), key=lambda e: e["name"])
    assert by_name == want


class SystemMessage:
    def __init__(self, data: dict) -> None:
        self.subtype, self.data = "init", data


def _replay(project: NativeProject, case: str, *, args: Optional[str] = "recorded",
            shown: bool = True, install_name: str = "acme-full-placeholders"):
    """The hook fed what the CLI delivered for ``case`` (with this project's paths), after a
    ``Skill`` call with the recorded args (or ``args``), or after no call it saw."""
    recorded = RECORDED[case]
    skill_dir = project.skills_dir / install_name
    plugin_root = project.root / ".agents"
    delivered = (recorded["delivered"].replace("<SKILL_DIR>", str(skill_dir))
                 .replace("<PLUGIN_ROOT>", str(plugin_root))
                 .replace("<PROJECT_DIR>", str(project.root))
                 .replace("<SESSION_ID>", "session-1"))
    init = SystemMessage({"session_id": "session-1", "cwd": str(project.root),
                          "plugins": [{"name": PLUGIN, "path": str(plugin_root)}]})
    call_args = recorded["args"] if args == "recorded" else args
    inp = {"skill": f"{PLUGIN}:{install_name}"}
    if call_args is not None:
        inp["args"] = call_args
    messages = [init]
    if shown:
        messages += [AssistantMessage([ToolUseBlock("t1", "Skill", inp)]),
                     UserMessage([ToolResultBlock("t1", "Launching skill")])]
    messages.append(UserMessage([TextBlock(
        f"Base directory for this skill: {skill_dir}\n{delivered}")]))
    return _feed(*messages)


def test_a_recorded_rewrite_with_args_names_the_version(placeholder_project):
    (got,) = _replay(placeholder_project, "full_args")
    assert (got.entry(), got.how) == (
        placeholder_project.expected("acme-full-placeholders"), _native.BY_RENDER)


def test_a_recorded_rewrite_without_args_names_the_version(placeholder_project):
    want = placeholder_project.expected("acme-full-placeholders")
    for shown in (True, False):
        (got,) = _replay(placeholder_project, "full_absent", shown=shown)
        assert (got.entry(), got.how) == (want, _native.BY_RENDER)
        _native._reset_seen()


def test_a_rewrite_that_does_not_fit_the_calls_args_names_no_version(placeholder_project):
    # The body the CLI renders for no args, after a call that passed some: no guess.
    (got,) = _replay(placeholder_project, "full_absent", args="Paris tomorrow")
    assert (got.how, got.version_id) == (_native.UNVERIFIED, None)
    assert got.skill_id == placeholder_project.expected("acme-full-placeholders")["skill_id"]


def test_after_the_lock_moved_on_an_old_rewrite_is_not_the_new_version(placeholder_project):
    name = "acme-full-placeholders"
    skill_md = placeholder_project.skills_dir / name / "SKILL.md"
    skill_md.write_text(skill_md.read_text().replace("Fee: $12.", "Fee: $15."))
    lock_path = placeholder_project.root / ".decimal" / "skills.lock"
    lock = json.loads(lock_path.read_text())
    from skillevaluation import skillfmt

    entry = lock["skills"][name]
    entry["version_id"] = "00000000-0000-0000-0000-00000000beef"
    entry["body_sha256"] = skillfmt.body_sha256(skillfmt.split_frontmatter(
        skill_md.read_text())[1])
    lock_path.write_text(json.dumps(lock))
    identity.clear_cache()
    (got,) = _replay(placeholder_project, "full_args")
    assert (got.how, got.version_id) == (_native.UNVERIFIED, None)


def test_a_body_edited_after_install_is_no_template(placeholder_project):
    skill_md = placeholder_project.skills_dir / "acme-full-placeholders" / "SKILL.md"
    skill_md.write_text(skill_md.read_text().replace("Fee: $12.", "Fee: $15."))
    (got,) = _replay(placeholder_project, "full_args")
    assert (got.how, got.version_id) == (_native.UNVERIFIED, None)


def test_appended_args_with_an_escaped_bang_are_taken_off(placeholder_project):
    plain = placeholder_project.skills_dir / "acme-plain"
    delivered = (f"Base directory for this skill: {plain}\n\n"
                 + PLAIN.strip() + "\n\n\nARGUMENTS: \\!urgent refund")
    got = _feed(_skill_call_to("acme-plain", "t1", args="!urgent refund"),
                UserMessage([TextBlock(delivered)]))
    assert [(a.entry(), a.how) for a in got] == [
        (placeholder_project.expected("acme-plain"), _native.BY_LOCK)]


def _skill_call_to(install_name: str, i: str, **inp: Any) -> AssistantMessage:
    return AssistantMessage([ToolUseBlock(i, "Skill", {"skill": f"{PLUGIN}:{install_name}",
                                                       **inp})])
