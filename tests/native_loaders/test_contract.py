"""Synthetic cases for the contract's own checks. No framework, no child process.

A check that has only ever seen healthy folders and healthy loaders has never been
seen to fail. Each case here hands ``contract.py`` one defect it must catch (and the
healthy baseline it must pass), so weakening the check turns this file red even on a
day every loader is clean:

* N8's static half reads every installed ``SKILL.md``. The runtime half loads only
  the target, so for every other skill (a user-only skill a user invokes by slash
  command, say) the static half is the only guard. Claude Code treats ``Bash``,
  ``Bash(*)``, ``Bash(*:*)`` and a comma-damaged ``Bash,`` alike: unrestricted.
* N3 must catch a delivery cut short after the sentinel line.
* N9 must see the plugin load when the documented snippet runs from a subdirectory
  of the project.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from . import contract, oracle, source
from .oracle import SCENARIO

OUTSIDE = "python3 -c 'print(6*7)'"


def _project(tmp_path: Path):
    project = source.graded_project(tmp_path / "project")
    return project, contract.expectation_from_lock(project)


def _foreign(ex: contract.Expectation) -> str:
    return next(n for n, e in sorted(ex.skills.items()) if e["render"]["variant"] == "foreign")


def _set_allowed_tools(ex: contract.Expectation, name: str, value: str) -> None:
    """Write ``allowed-tools: <value>`` into ``name``'s frontmatter, in its key slot."""
    md = ex.project / ex.root / name / "SKILL.md"
    head, sep, body = md.read_text(encoding="utf-8")[4:].partition("\n---\n")
    lines = [ln for ln in head.split("\n") if not ln.startswith("allowed-tools:")]
    at = next(i for i, ln in enumerate(lines)
              if ln.startswith(("disable-model-invocation:", "metadata:")))
    lines.insert(at, f"allowed-tools: {oracle.yaml_quote(value)}")
    md.write_text("---\n" + "\n".join(lines) + sep + body, encoding="utf-8")


def _n8_runtime_ok(ex: contract.Expectation) -> dict:
    """An observation whose runtime half of N8 passes, so only the static half decides."""
    foreign = _foreign(ex)
    scoped = f"python3 /p/.agents/skills/{SCENARIO.target}/{SCENARIO.script} 100 45"
    foreign_cmd = f"python3 /p/.agents/skills/{foreign}/{SCENARIO.script} 100 45"
    return {
        "loader": "claude-agent-sdk",
        "script_stdout": SCENARIO.script_out,
        "claude": {"scoped_command": scoped, "outside_command": OUTSIDE, "foreign": foreign,
                   "foreign_loaded": True, "foreign_command": foreign_cmd,
                   "commands_denied": [OUTSIDE, foreign_cmd]},
    }


# ── N8, the static half ──────────────────────────────────────────────────────


def test_n8_passes_the_oracle_as_written(tmp_path):
    _, ex = _project(tmp_path)
    v = contract.n8(_n8_runtime_ok(ex), ex)
    assert v.ok, v.detail


USER_ONLY = SCENARIO.user_only
TARGET = SCENARIO.target
FOREIGN = "<foreign>"

#: (skill, allowed-tools as rendered, a phrase the failure must carry)
N8_DEFECTS = [
    (USER_ONLY, "Bash", "bare Bash"),
    (USER_ONLY, "Read Bash(*)", "bare Bash"),
    (USER_ONLY, "Read Bash(*:*)", "bare Bash"),
    (USER_ONLY, "Read Bash( * : * )", "bare Bash"),
    (USER_ONLY, "Read, Bash,", "bare Bash"),
    (USER_ONLY, "Read,Bash", "bare Bash"),
    (USER_ONLY, "Read, Bash,", "comma"),
    (USER_ONLY, "Read, Grep,", "comma"),
    (TARGET, "Read, Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/compute_refund.py *)", "comma"),
    (FOREIGN, "Read Bash(*)", "bare Bash"),
    (FOREIGN, "Read, Grep", "comma"),
    (FOREIGN, "Read WebFetch", "(foreign) grants ['WebFetch']"),
]


@pytest.mark.parametrize(("skill", "value", "phrase"), N8_DEFECTS,
                         ids=[f"{s}={v!r}:{p}" for s, v, p in N8_DEFECTS])
def test_n8_fails_unrestricted_or_damaged_allowed_tools(tmp_path, skill, value, phrase):
    _, ex = _project(tmp_path)
    name = _foreign(ex) if skill == FOREIGN else skill
    _set_allowed_tools(ex, name, value)
    v = contract.n8(_n8_runtime_ok(ex), ex)
    assert not v.ok, f"N8 passed allowed-tools: {value!r} on {name}"
    assert phrase in v.detail and name in v.detail, v.detail


N8_HEALTHY = [
    (USER_ONLY, "Read Grep Glob"),
    (USER_ONLY, "Bash(git log *)"),
    (FOREIGN, "Read Grep Glob"),
]


@pytest.mark.parametrize(("skill", "value"), N8_HEALTHY, ids=[f"{s}={v!r}" for s, v in N8_HEALTHY])
def test_n8_passes_scoped_and_read_only_allowed_tools(tmp_path, skill, value):
    _, ex = _project(tmp_path)
    _set_allowed_tools(ex, _foreign(ex) if skill == FOREIGN else skill, value)
    v = contract.n8(_n8_runtime_ok(ex), ex)
    assert v.ok, v.detail


#: Inputs the contract's tokenizer must split exactly as the format module does.
TOKENIZER_CASES = [
    "Read", "Read Bash", "Read, Bash,", "Read,Bash", "Read,  Grep,\tGlob",
    "Bash(*)", "Bash(*:*)", "Bash( * : * )", "Bash(git log *)", "Bash(git log, *) Read",
    "Read Bash(git log", "Read Read", "Bash(python3 ${CLAUDE_SKILL_DIR}/scripts/x.py *)",
    "Bash()", "Bash(*.py)", "WebFetch mcp__x__y", "",
]


def test_the_tokenizer_agrees_with_skillfmt():
    """The contract mirrors ``skillevaluation.skillfmt`` (it grades the renderer, so it
    does not import it); this keeps the two from drifting wherever both are installed."""
    skillfmt = pytest.importorskip("skillevaluation.skillfmt")
    for value in TOKENIZER_CASES:
        want = skillfmt.tokenize_allowed_tools(value)
        assert contract.allowed_tools_tokens(value) == want, value
        assert [contract.is_bare_bash(t) for t in want] == [skillfmt.is_bare_bash(t) for t in want], value


# ── N3 ───────────────────────────────────────────────────────────────────────


def _target_body() -> str:
    row = next(r for r in oracle.ROWS if oracle.install_name_for(r.name, r.skill_id) == TARGET)
    return row.stored_body.strip()


def _lock_expectation() -> contract.Expectation:
    return contract.expectation_for(Path("/nonexistent-project"), oracle.build().lock)


def test_n3_passes_the_whole_body_inside_the_loaders_wrapper():
    body = _target_body()
    obs = {"loader": "google-adk", "body": body,
           "body_text": f"<skill name={TARGET!r}>\n{body}\n</skill>\nResources: references/"}
    v = contract.n3(obs, _lock_expectation())
    assert v.ok, v.detail


def test_n3_fails_a_delivery_cut_short_after_the_sentinel():
    """The loader parsed the whole body, but the model got only its first three lines."""
    body = _target_body()
    first_three = "\n".join(body.split("\n")[:3])
    assert oracle.sentinel_of(TARGET) in first_three
    obs = {"loader": "google-adk", "body": body, "body_text": first_three}
    v = contract.n3(obs, _lock_expectation())
    assert not v.ok, "N3 passed a delivery that stops after the sentinel line"
    assert "verbatim" in v.detail, v.detail


# ── N9 ───────────────────────────────────────────────────────────────────────


def _n9_obs(project: Path, ex: contract.Expectation, **subdirectory) -> dict:
    agents = str(project / ".agents")
    plugin = {"name": oracle.PLUGIN_NAME, "path": agents, "source": f"{oracle.PLUGIN_NAME}@inline",
              "version": f"lock-{oracle.lock_hash(ex.root, ex.skills)[:12]}"}
    skills = [f"{oracle.PLUGIN_NAME}:{n}" for n in ex.names]
    sub = {"cwd": str(project / "src"), "plugin_root": agents, "plugin": plugin,
           "plugins": [plugin], "plugin_errors": [], "skills": skills}
    sub.update(subdirectory)
    return {"loader": "claude-agent-sdk", "claude": {
        "plugin": plugin, "plugins": [plugin], "plugin_errors": [], "plugin_root": agents,
        "skills": skills, "from_subdirectory": sub}}


def test_n9_passes_the_plugin_loaded_from_the_root_and_from_a_subdirectory(tmp_path):
    project, ex = _project(tmp_path)
    v = contract.n9(_n9_obs(project, ex), ex)
    assert v.ok, v.detail


def test_n9_fails_when_the_snippet_never_ran_from_a_subdirectory(tmp_path):
    project, ex = _project(tmp_path)
    obs = _n9_obs(project, ex)
    del obs["claude"]["from_subdirectory"]
    v = contract.n9(obs, ex)
    assert not v.ok and "subdirectory" in v.detail, v.detail


def test_n9_fails_when_nothing_loads_from_a_subdirectory(tmp_path):
    """What a relative ``--plugin-dir .agents`` gives from a subdirectory: no plugin,
    exit 0, one ``path-not-found`` entry in ``init.plugin_errors`` and nothing else."""
    project, ex = _project(tmp_path)
    obs = _n9_obs(project, ex, plugin=None, plugins=[], skills=[],
                  plugin_errors=[{"type": "path-not-found", "path": str(project / "src" / ".agents")}])
    v = contract.n9(obs, ex)
    assert not v.ok and "subdirectory" in v.detail and "path-not-found" in v.detail, v.detail


def test_n9_fails_when_the_subdirectory_run_is_the_project_root(tmp_path):
    project, ex = _project(tmp_path)
    v = contract.n9(_n9_obs(project, ex, cwd=str(project)), ex)
    assert not v.ok and "not a subdirectory" in v.detail, v.detail


def test_n9_fails_when_a_subdirectory_run_lists_other_skills(tmp_path):
    project, ex = _project(tmp_path)
    v = contract.n9(_n9_obs(project, ex, skills=[f"{oracle.PLUGIN_NAME}:{ex.names[0]}"]), ex)
    assert not v.ok and "subdirectory" in v.detail, v.detail


def test_the_subdirectory_case_leaves_no_trace_in_the_graded_project(graded):
    """The Claude child makes its subdirectory and removes it again. The producer check
    in ``test_native_loaders.py`` compares files, so an empty directory left behind
    would go unnoticed there."""
    here = graded.graded_here()
    if here:
        graded.observation(here[0])  # starts every child this environment grades
    top = sorted(p.name for p in graded.project.iterdir())
    assert top == [".agents", ".decimal"], f"after the children ran, the project holds {top}"
