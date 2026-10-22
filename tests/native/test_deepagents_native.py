"""Deep Agents native mode: a ``read_file`` of a SKILL.md is an activation of the version
read, DecimalAI puts nothing into the agent's prompt (``init(langchain=True)`` traces only),
and the pointer helper shows the skills.

The framework runs for real (deepagents, LangGraph underneath, traced by the LangChain
adapter) with a scripted model, in a child process per scenario
(``_scenario_deepagents``), against a probe; the assertions read the traces the probe
received. The handler-level cases at the bottom drive ``CallbackHandler`` directly and
touch no process-wide state.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, List

import pytest

from tests.native_loaders import ledger

from ._runner import run_scenario
from .conftest import V2_SENTINEL

FRAMEWORK = "deepagents"
_results: Dict[str, Any] = {}


def _scenario(name: str, project) -> Dict[str, Any]:
    from tests.conformance.environments import loader_graded_elsewhere

    if loader_graded_elsewhere(FRAMEWORK):
        ledger.skip_declared("elsewhere", f"{FRAMEWORK}:native", "Deep Agents native mode")
    if importlib.util.find_spec("deepagents") is None:
        ledger.skip_declared("unavailable", f"{FRAMEWORK}:native", "deepagents is not installed")
    if name not in _results:
        _results[name] = run_scenario(f"tests.native._scenario_deepagents:{name}", {},
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


def test_a_skill_md_read_is_an_activation_of_the_installed_version(native_project):
    result = _scenario("column", native_project)
    want = native_project.expected()
    trace = _trace(result, 0)
    assert _target_entries(trace, {want["name"], native_project.target}) == [want]
    assert want["name"] in trace["skills_loaded_by_agent"]


def test_the_hook_reads_what_read_file_returned(native_project):
    """The format the adapter strips: a status header, then the file verbatim, with no
    line numbers. If deepagents changes it, this says so before an activation degrades."""
    result = _scenario("column", native_project)
    head = result["notes"]["delivered_head"][0]
    assert head == "@@ lines 1-25 of 25 @@\n---\nname: acme-refund-policy", head


def test_reading_it_twice_in_one_run_is_one_activation(native_project):
    result = _scenario("column", native_project)
    assert result["runs"][0]["target_loads"] == 2
    want = native_project.expected()
    assert len(_target_entries(_trace(result, 0), {want["name"], native_project.target})) == 1


def test_after_an_update_the_activation_names_the_version_the_model_read(native_project):
    """Deep Agents keeps the listing it parsed in the thread (run 2 continues run 1's
    conversation), but ``read_file`` reads the file again: the model gets version 2."""
    result = _scenario("column", native_project)
    v1 = native_project.expected()
    received = "\n".join(result["runs"][1]["received"])
    assert V2_SENTINEL in received and native_project.sentinel() not in received
    v2 = native_project.update_to_v2()
    assert _target_entries(_trace(result, 1), {v2["name"], native_project.target}) == [v2]
    assert v2["version_id"] != v1["version_id"]


def test_the_listing_the_model_was_shown_is_offered(native_project):
    result = _scenario("column", native_project)
    offered = set(_trace(result, 0)["skills_offered_in_prompt"])
    listed = {native_project.expected(name)["name"] for name in native_project.lock["skills"]}
    assert offered == listed, (offered, listed)


def _told(warned: List[str], words: str) -> List[str]:
    return [line for line in warned if words in line]


def test_init_puts_nothing_in_a_deep_agent_and_tells_only_a_plain_graph(native_project):
    """``init(langchain=True)`` traces only since 0.25: no DecimalAI text reaches either
    model and no run is routed. The plain graph is the run told so, once, and the Deep
    Agents agent is not: its skills come from its own middleware, as documented."""
    result = _scenario("checks", native_project)
    assert result["errors"] == [None] * 5
    assert result["deepagents"]["injected"] == [] and result["plain"]["injected"] == []
    assert [runs[0].get("routing_id") for runs in result["traces"][:3]] == [None] * 3
    assert _told(result["deepagents"]["warned"], "traces only") == []
    told = _told(result["plain"]["warned"], "traces only")
    assert len(told) == 1 and "agent_config(name).middleware" in told[0], result["logs"]


def test_agent_config_beside_deep_agents_skills_warns_once(native_project):
    """Deep Agents' skills middleware and ``agent_config()``'s in one agent show the model
    two skill lists. Deep Agents runs a user's middleware before its skills middleware, so
    ``skills_metadata`` is not in the state at our ``before_agent``: the check is made at the
    model call. Said once per config, on the first run."""
    result = _scenario("checks", native_project)
    first, second = result["both"]["warned"]
    assert len(_told(first, "two skill lists")) == 1, first
    assert _told(second, "two skill lists") == [], second
    assert result["both"]["prompt_shown"], "the agent's prompt is not in the call"
    assert result["both"]["listed"], "Deep Agents' own list is not in the call"


def test_a_pinned_skill_is_an_activation_too(native_project):
    """``pinned_skills=[...]`` delivers the body without a ``read_file``: the middleware
    appends it to the conversation, frontmatter cut, so the lock names the version."""
    result = _scenario("checks", native_project)
    assert len(result["pinned"]["messages"]) == 1, result["pinned"]
    want = native_project.expected()
    trace = result["traces"][2][0]
    assert _target_entries(trace, {want["name"], native_project.target}) == [want]


def test_the_pointer_helper_lists_the_installed_skills(native_project):
    result = _scenario("checks", native_project)
    assert result["deepagents"]["listed"] == sorted(native_project.lock["skills"])
    assert result["deepagents"]["backend"] == "FilesystemBackend"


# ── the handler, driven directly ─────────────────────────────────────────────


def _drive_read(handler, *, path: str, output: Any) -> None:
    from uuid import uuid4

    from langchain_core.messages import ToolMessage

    root, node, tool = uuid4(), uuid4(), uuid4()
    handler.on_chain_start({}, {"messages": []}, run_id=root, name="LangGraph")
    handler.on_chain_start({}, {}, run_id=node, parent_run_id=root,
                           name="SkillsMiddleware.before_agent")
    handler.on_chain_end({}, run_id=node, parent_run_id=root)
    handler.on_tool_start({"name": "read_file"}, str({"file_path": path}), run_id=tool,
                          parent_run_id=root, inputs={"file_path": path})
    handler.on_tool_end(ToolMessage(content=output, tool_call_id="c1", name="read_file"),
                        run_id=tool, parent_run_id=root)


def _built(handler) -> Dict[str, Any]:
    """The open run's trace, as it would be sent."""
    return handler.build_trace(next(iter(handler._runs.values()))).model_dump(mode="json")


def _skill_md(native_project) -> tuple:
    path = native_project.skills_dir / native_project.target / "SKILL.md"
    return str(path), path.read_text()


def test_a_partial_read_is_reported_unverified(native_project):
    """A window that stops short of the end delivers part of the body: the skill is
    known (stamp and path), the version is not, and the trace says exactly that."""
    pytest.importorskip("langchain_core")
    from decimalai.langchain import CallbackHandler

    path, text = _skill_md(native_project)
    shown = "\n".join(text.splitlines()[:20])
    handler = CallbackHandler(agent_name="partial", auto_send=False)
    _drive_read(handler, path=path, output=f"@@ lines 1-20 of 25 | next offset 20 @@\n{shown}")
    want = native_project.expected()
    entries = _target_entries(_built(handler), {want["name"]})
    assert len(entries) == 1
    assert entries[0]["skill_id"] == want["skill_id"] and "version_id" not in entries[0]
    assert entries[0]["body_sha256"] != want["body_sha256"]
