"""Deep Agents native mode: a ``read_file`` of a SKILL.md is an activation of the version
read, the router stays out of the agent's prompt, and the pointer helper shows the skills.

The framework runs for real (deepagents, LangGraph underneath, traced by the LangChain
adapter) with a scripted model, in a child process per scenario
(``_scenario_deepagents``), against a probe; the assertions read the traces the probe
received. The handler-level cases at the bottom drive ``CallbackHandler`` directly and
touch no process-wide state.
"""

from __future__ import annotations

import importlib.util
import logging
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


def test_the_router_stays_out_of_a_deep_agent_and_serves_a_plain_graph(native_project):
    result = _scenario("checks", native_project)
    assert result["errors"] == [None, None, None]
    assert result["deepagents"]["injected"] == []
    deep, plain = result["traces"][0][0], result["traces"][1][0]
    assert deep.get("routing_id") is None
    # The control: the same router, in the same process, injects into a graph without
    # the skills middleware, so the empty list above is the rule, not a dead router.
    assert result["plain"]["injected"], "the router injected nothing anywhere"
    assert plain.get("routing_id")
    said = [line for line in result["logs"] if "does not inject its menu" in line]
    assert len(said) == 1 and said[0].startswith("INFO"), result["logs"]


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


def _drive_read(handler, *, path: str, output: Any, router_injected: bool = False) -> None:
    from uuid import uuid4

    from langchain_core.messages import ToolMessage

    root, node, tool = uuid4(), uuid4(), uuid4()
    handler.on_chain_start({}, {"messages": []}, run_id=root, name="LangGraph")
    handler.on_chain_start({}, {}, run_id=node, parent_run_id=root,
                           name="SkillsMiddleware.before_agent")
    handler.on_chain_end({}, run_id=node, parent_run_id=root)
    handler._runs[root].router_injected = router_injected
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


def test_a_native_load_on_a_routed_run_warns_once(native_project, monkeypatch, caplog):
    pytest.importorskip("langchain_core")
    import decimalai.langchain as lc

    monkeypatch.setattr(lc, "_native_double_catalog_warned", False)
    path, text = _skill_md(native_project)
    with caplog.at_level(logging.WARNING, logger="decimalai.langchain"):
        for _ in range(2):
            handler = lc.CallbackHandler(agent_name="routed", auto_send=False)
            _drive_read(handler, path=path, output=f"@@ lines 1-25 of 25 @@\n{text}",
                        router_injected=True)
            assert _target_entries(_built(handler), {native_project.expected()["name"]}) == [
                native_project.expected()]
    warned = [r for r in caplog.records if "two catalogs" in r.getMessage()]
    assert len(warned) == 1
