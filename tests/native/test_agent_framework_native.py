"""Microsoft Agent Framework in native mode: ``decimalai.agent_framework.skills()`` on a real
``Agent``, graded on the traces the probe received.

The framework runs in a child process per scenario (``_scenario_agent_framework.py``):
``init(otel=True)`` installs a global tracer provider and Agent Framework's telemetry
settings are process-wide, and these tests ride the default unit run. The helpers that need
no framework are tested here, in-process.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, List

import pytest

from decimalai import _native
from decimalai import agent_framework as maf
from decimalai._native import BY_STAMP, Load
from tests.native_loaders import ledger, oracle

from ._runner import run_scenario
from .conftest import V2_SENTINEL

FRAMEWORK = "agent-framework"
SC = oracle.SCENARIO
_results: Dict[str, Any] = {}


def _scenario(native_project, name: str) -> Dict[str, Any]:
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "Agent Framework native mode")
    if importlib.util.find_spec("agent_framework") is None:
        ledger.skip_declared("unavailable", f"{FRAMEWORK}:native",
                             "agent-framework-core is not importable")
    if name not in _results:
        _results[name] = run_scenario(f"tests.native._scenario_agent_framework:{name}", {},
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
    assert trace["agent_name"] == "native-maf"
    assert len(trace["llm_calls"]) == 5  # the run's model calls ride along


def test_after_an_update_the_cached_v1_is_what_the_model_got_and_what_the_trace_says(
        native_project):
    result = _scenario(native_project, "column")
    v1 = native_project.expected()
    received = "\n".join(result["runs"][1]["received"])
    # Agent Framework caches a provider's skills (and each skill's content) on first use,
    # so the same agent keeps serving v1 after `skills update` wrote v2. If this flips, the
    # framework changed its default; the activation must still follow what was received.
    assert native_project.sentinel() in received and V2_SENTINEL not in received
    trace = _only_trace(result["runs"][1])
    assert _target(trace, v1) == [v1]
    assert result["notes"]["v2"]["version_id"] != v1["version_id"]


def test_the_resource_read_and_the_script_run_work_through_the_helper(native_project):
    run = _scenario(native_project, "column")["runs"][0]
    received = "\n".join(run["received"])
    assert SC.resource_marker in received  # read_skill_resource answered the file
    assert SC.script_out in received  # run_skill_script ran it with the helper's runner


def test_the_helper_lists_the_installed_skills(native_project):
    result = _scenario(native_project, "default_approval")
    instructions = result["observed"]["instructions"]
    for install_name in native_project.lock["skills"]:
        assert f"<name>{install_name}</name>" in instructions
    trace = _only_trace(result["runs"][0])
    assert native_project.expected()["name"] in trace["skills_offered_in_prompt"]


def test_by_default_reads_run_unasked_and_a_script_waits_for_approval(native_project):
    observed = _scenario(native_project, "default_approval")["observed"]
    assert {"load_skill", "read_skill_resource", "run_skill_script"} <= set(
        observed["tools_offered"])
    results = "\n".join(observed["results"])
    assert native_project.sentinel() in results and SC.resource_marker in results
    assert observed["approval_requests"] == ["run_skill_script"]
    assert SC.script_out not in results


def test_inside_the_users_own_trace_the_activation_lands_on_it(native_project):
    run = _scenario(native_project, "user_trace")["runs"][0]
    assert not run["error"], run["error"]
    expected = native_project.expected()
    mine = [t for t in run["traces"] if t.get("agent_name") == "users-own-trace"]
    assert len(mine) == 1, [t.get("agent_name") for t in run["traces"]]
    assert _target(mine[0], expected) == [expected]


# ── the helpers, in-process ──────────────────────────────────────────────────


def _maf_answer(skill_md: str) -> str:
    """What FileSkill.get_content returns: the raw SKILL.md, then its two blocks."""
    return (f"{skill_md}\n\n<available_resources>\n  <resource name=\"{SC.resource}\"/>\n"
            f"</available_resources>\n\n<available_scripts>\n  <script name=\"{SC.script}\"/>\n"
            "</available_scripts>")


def test_the_delivered_skill_md_is_the_answer_without_agent_frameworks_blocks(native_project):
    skill_md = (native_project.skills_dir / SC.target / "SKILL.md").read_text()
    assert maf._skill_md(_maf_answer(skill_md)) == skill_md
    got = _native.resolve(Load(framework=maf.FRAMEWORK, name=SC.target,
                               text=maf._skill_md(_maf_answer(skill_md))))
    assert (got.entry(), got.how) == (native_project.expected(), BY_STAMP)
    # Hashing the answer as it came would make the body unverifiable.
    whole = _native.resolve(Load(framework=maf.FRAMEWORK, name=SC.target,
                                 text=_maf_answer(skill_md)))
    assert whole.version_id is None


def test_the_listing_names_come_from_the_providers_own_instructions():
    text = ("<available_skills>\n  <skill>\n    <name>acme-refund-policy</name>\n"
            "    <description>x &amp; y</description>\n  </skill>\n</available_skills>")
    assert maf._listed_names([text, 42]) == ["acme-refund-policy"]


@pytest.mark.parametrize(("args", "argv"), [
    (None, []),
    (["100", "45"], ["100", "45"]),
    ({"amount": 100, "days": 45}, ["--amount", "100", "--days", "45"]),
    ({"verbose": True, "dry": False, "x": None, "tag": ["a", "b"]},
     ["--verbose", "--tag", "a", "--tag", "b"]),
])
def test_the_runner_passes_the_models_arguments_as_cli_arguments(args, argv):
    assert maf._argv(args) == argv


def _run_script(path, args=None, **kw) -> str:
    import asyncio
    from types import SimpleNamespace

    script = SimpleNamespace(full_path=str(path), name=path.name)
    return asyncio.run(maf.script_runner(**kw)(None, script, args))


def test_the_runner_answers_with_stdout_or_with_what_went_wrong(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("import sys\nprint('ran', *sys.argv[1:])\n")
    assert _run_script(ok, ["100", "45"]) == "ran 100 45\n"
    bad = tmp_path / "bad.py"
    bad.write_text("import sys\nprint('partial')\nsys.exit('boom')\n")
    answer = _run_script(bad)
    assert "exited with status 1" in answer and "partial" in answer and "boom" in answer
    slow = tmp_path / "slow.py"
    slow.write_text("import time\ntime.sleep(30)\n")
    assert "did not finish within 0.5s" in _run_script(slow, timeout=0.5)
    text = tmp_path / "notes.txt"
    text.write_text("not a script")
    assert "cannot be run" in _run_script(text)
