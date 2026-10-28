"""Pydantic AI harness native mode: a ``load_capability`` of a skill is an activation of the
version delivered, the router leaves an Agent with ``Skills`` alone, and the pointer
helper's ``Skills`` lists the installed skills.

The framework runs for real (pydantic-ai-harness on pydantic-ai-slim) with a scripted
``FunctionModel``, in a child process per scenario (``_scenario_pydantic_ai_harness``),
against a probe; the assertions read the traces the probe received. The cases at the
bottom are pure functions of the observer.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, List

import pytest

from tests.native_loaders import ledger

from ._runner import run_scenario
from .conftest import V2_SENTINEL

FRAMEWORK = "pydantic-ai-harness"
_results: Dict[str, Any] = {}


def _scenario(name: str, project) -> Dict[str, Any]:
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "the harness's native mode")
    if importlib.util.find_spec("pydantic_ai_harness") is None:
        ledger.skip_declared("unavailable", f"{FRAMEWORK}:native",
                             "pydantic-ai-harness is not installed")
    if name not in _results:
        _results[name] = run_scenario(f"tests.native._scenario_pydantic_ai_harness:{name}", {},
                                      cwd=project.root)
    return _results[name]


def _trace(result: Dict[str, Any], run: int) -> Dict[str, Any]:
    entry = result["runs"][run]
    assert not entry["error"], entry["error"]
    assert len(entry["traces"]) == 1, f"run {run + 1} sent {len(entry['traces'])} traces"
    return entry["traces"][0]


def _target_entries(trace: Dict[str, Any], names) -> List[Dict[str, Any]]:
    return [e for e in trace.get("active_skills") or []
            if isinstance(e, dict) and e.get("name") in names]


def test_a_load_is_an_activation_of_the_installed_version(native_project):
    """The harness keeps none of the frontmatter's metadata, so no stamp reaches the
    model: the identity comes from the delivered body's hash in the lock."""
    result = _scenario("column", native_project)
    want = native_project.expected()
    trace = _trace(result, 0)
    assert _target_entries(trace, {want["name"], native_project.target}) == [want]
    assert want["name"] in trace["skills_loaded_by_agent"]


def test_the_observer_reads_what_load_capability_returned(native_project):
    result = _scenario("column", native_project)
    first = result["runs"][0]["received"][0]
    assert first.startswith(f"# Skill: {native_project.target}\n\nSkill directory: `"), first[:200]
    assert native_project.sentinel() in first


def test_a_second_load_in_one_run_is_refused_and_counted_once(native_project):
    """The harness refuses a second ``load_capability`` of an active skill (no body: a
    retry prompt before Pydantic AI 2.63, a tool result since), and the trace counts the
    skill once."""
    result = _scenario("column", native_project)
    run = result["runs"][0]
    assert run["target_loads"] == 2
    assert "is already active" in run["received"][1]
    want = native_project.expected()
    assert len(_target_entries(_trace(result, 0), {want["name"], native_project.target})) == 1


def test_after_an_update_the_activation_names_the_version_the_model_read(native_project):
    """The harness reads its skill libraries at the start of every run, so the run after
    the update, on the same Agent, delivers version 2."""
    result = _scenario("column", native_project)
    received = "\n".join(result["runs"][1]["received"])
    assert V2_SENTINEL in received and native_project.sentinel() not in received
    v2 = native_project.update_to_v2()
    assert _target_entries(_trace(result, 1), {v2["name"], native_project.target}) == [v2]


def test_the_catalog_the_model_was_shown_is_offered(native_project):
    result = _scenario("column", native_project)
    offered = set(_trace(result, 0)["skills_offered_in_prompt"])
    # Since 0.57 the harness keeps a user-only skill out of the model's catalog.
    listed = {native_project.expected(name)["name"]
              for name, entry in native_project.lock["skills"].items()
              if entry["invocation"] != "user"}
    assert offered == listed, (offered, listed)


def test_the_router_leaves_an_agent_with_skills_alone(native_project):
    result = _scenario("checks", native_project)
    assert result["errors"] == [None, None, None]
    for label in ("observed", "plain_skills"):
        seen = result[label]
        assert "load_skill" not in seen["tools"] and "load_capability" in seen["tools"], seen
        assert "conformance-skill-alpha" not in seen["instructions"], seen["instructions"][-400:]
    observed, plain_skills, no_skills = (r[0] for r in result["traces"])
    assert observed.get("routing_id") is None and plain_skills.get("routing_id") is None
    # The control: the same router, in the same process, serves an Agent without Skills.
    assert "load_skill" in result["no_skills"]["tools"]
    assert "conformance-skill-alpha" in result["no_skills"]["instructions"]
    assert no_skills.get("routing_id")
    said = [line for line in result["logs"] if "adds neither its skill list" in line]
    # Once for a Skills whose loads are reported, once for a plain one, which says how.
    assert len(said) == 2 and all(line.startswith("WARNING") for line in said), result["logs"]
    assert "Each skill the model loads through it is reported" in said[0]
    assert "decimalai.pydantic_ai_harness.skills(" in said[1]


def test_the_pointer_helper_lists_the_installed_skills(native_project):
    result = _scenario("checks", native_project)
    catalog = result["observed"]["instructions"]
    for name, entry in native_project.lock["skills"].items():
        if entry["invocation"] == "user":   # honoured since 0.57: kept from the model
            assert f"- {name}: " not in catalog, catalog[-600:]
        else:
            assert f"- {name}: " in catalog, catalog[-600:]


# ── the observer's reading of a rendered skill ───────────────────────────────


def test_the_rendered_heading_is_cut_back_to_the_body(native_project):
    from decimalai import _native
    from decimalai.pydantic_ai_harness import _body

    from .conftest import body_of

    raw = (native_project.skills_dir / native_project.target / "SKILL.md").read_bytes()
    body = body_of(raw).strip()
    folder = str(native_project.skills_dir / native_project.target)
    rendered = (f"# Skill: {native_project.target}\n\nSkill directory: `{folder}`. Relative "
                f"paths in this skill resolve against it.\n\n{body}")
    got, directory = _body(native_project.target, rendered)
    assert (got, directory) == (body, folder)
    activation = _native.resolve(_native.Load(name=native_project.target, body=got,
                                              path=directory))
    assert (activation.entry(), activation.how) == (native_project.expected(), _native.BY_LOCK)


def test_a_rendering_it_does_not_know_is_reported_unverified(native_project):
    from decimalai import _native
    from decimalai.pydantic_ai_harness import _body

    text = f"## {native_project.target}\n\nsomething else entirely"
    got, directory = _body(native_project.target, text)
    assert (got, directory) == (text, None)
    activation = _native.resolve(_native.Load(name=native_project.target, body=got))
    assert activation.how == _native.UNVERIFIED and activation.version_id is None
    assert activation.skill_id == native_project.expected()["skill_id"]


@pytest.mark.parametrize("result", [
    {"instructions": "x"},
    type("ToolReturn", (), {"return_value": {"instructions": "x"}})(),
])
def test_both_shapes_of_a_load_result_are_read(result):
    from decimalai.pydantic_ai_harness import _instructions

    assert _instructions(result) == "x"


def test_a_loaded_text_in_the_history_is_reported_once_for_its_own_run(monkeypatch):
    """Pydantic AI 2.63+ states a loaded skill's text as an instruction change in the history.
    Each one of this run is reported once, however many responses follow it; another run's
    (a continued conversation's history) and another capability's are not."""
    from types import SimpleNamespace

    from decimalai import _native
    from decimalai import pydantic_ai_harness as harness

    reported = []
    monkeypatch.setattr(_native, "record", lambda load, **_: reported.append(load.body))

    def request(run_id, *parts):
        return SimpleNamespace(kind="request", run_id=run_id, parts=list(parts))

    def change(capability, content):
        return SimpleNamespace(part_kind="instruction-delta", id=f"capability:{capability}",
                               content=content)

    observer = SimpleNamespace(ids=frozenset({"acme"}), reported=set(), loads=None)
    ctx = SimpleNamespace(run_id="run-2", messages=[
        request("run-1", change("acme", "# Skill: acme\n\nOLD RUN")),
        request("run-2", change("other", "# Skill: other\n\nNOT A SKILL")),
        request("run-2", change("acme", "# Skill: acme\n\nBODY")),
    ])
    harness._observe_instruction_changes(observer, ctx)
    harness._observe_instruction_changes(observer, ctx)  # the next response's view
    assert reported == ["BODY"]
