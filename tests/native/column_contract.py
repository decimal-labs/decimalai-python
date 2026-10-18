"""The native column's contract: the only place its assertions live.

One cell per (framework, item), graded from the scenario result ``column.py`` describes:

======  =======================================================================================
NC1     run 1's trace lists the target among the skills the model was offered
NC2     run 1's trace activates the target with the installed version's exact identity
        (wire name, skill id, version id, body hash)
NC3     the model loaded the target twice in run 1, and the trace counts it once
NC4     a tool call in run 1 read the target's resource through the framework, and its result
        holds the resource's bytes
NC5     a tool call in run 1 ran the target's script through the framework, and its result
        holds the script's output
NC6     after ``skills update``, run 2's activation names the version whose body the model
        actually received (v1 from a loader that cached it, v2 from one that re-read it)
======  =======================================================================================

The expected identities come from the lock (``oracle.build``), never from the trace or the
probe. An item a framework cannot satisfy at these versions is declared in
:data:`DECLARED_NA` with its reason; a declared item that starts passing fails, so the
ledger cannot outlive the gap it excuses.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from decimalai.skillpkg.identity import ManagedSkill
from tests.native import column
from tests.native.conftest import V2_SENTINEL, _rows_v2
from tests.native_loaders import oracle

ITEMS = ("NC1", "NC2", "NC3", "NC4", "NC5", "NC6")

#: (framework, item) -> why it cannot apply at the frameworks' current versions. Each entry
#: is debt with a named way out.
DECLARED_NA: Dict[Tuple[str, str], str] = {}
#: The ledger may not grow without anyone noticing.
NA_BUDGET = 4


def expected(lock: dict, install_name: str = column.TARGET) -> Dict[str, str]:
    e = lock["skills"][install_name]
    skill = ManagedSkill(install_name=install_name, registry_name=e["name"],
                         skill_id=e["skill_id"], version_id=e["version_id"],
                         body_sha256=e["body_sha256"], path="")
    return {"name": skill.wire_name, "skill_id": e["skill_id"],
            "version_id": e["version_id"], "body_sha256": e["body_sha256"]}


V1 = expected(oracle.build().lock)
V2 = expected(oracle.build(_rows_v2()).lock)
V1_SENTINEL = oracle.sentinel_of(column.TARGET)
#: Names a trace may report the target by: its wire name, or the install name before the
#: client translates it.
TARGET_NAMES = {V1["name"], column.TARGET}


def _run(result: Dict[str, Any], i: int) -> Dict[str, Any]:
    runs = result.get("runs") or []
    assert len(runs) > i, f"the scenario made {len(runs)} run(s); the column needs run {i + 1}"
    run = runs[i]
    assert not run.get("error"), f"run {i + 1} raised: {run['error']}"
    assert run.get("traces"), f"run {i + 1} sent no trace to the probe"
    return run


def _target_entries(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [e for t in run["traces"] for e in (t.get("active_skills") or [])
            if isinstance(e, dict) and e.get("name") in TARGET_NAMES]


def _identity(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {k: entry.get(k) for k in ("name", "skill_id", "version_id", "body_sha256")}


def _tool_records(run: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """``(what went in, what came back)`` for every tool call and tool span in the run."""
    for trace in run["traces"]:
        for call in trace.get("llm_calls") or []:
            for tool in call.get("tool_calls") or []:
                yield (f"{tool.get('tool_name')} {json.dumps(tool.get('args'), default=str)}",
                       json.dumps(tool.get("result"), default=str))
        for span in trace.get("spans") or []:
            if str(span.get("span_type", "")).lower() == "tool":
                yield (f"{span.get('name')} {span.get('input_preview') or ''}",
                       str(span.get("output_preview") or ""))


def _found_tool(run: Dict[str, Any], needle_in: str, needle_out: str) -> bool:
    return any(needle_in in i and needle_out in o for i, o in _tool_records(run))


def grade(item: str, result: Dict[str, Any]) -> None:
    """Assert one item for one framework's scenario result."""
    if item == "NC1":
        run = _run(result, 0)
        offered = {n for t in run["traces"] for n in (t.get("skills_offered_in_prompt") or [])}
        assert offered & TARGET_NAMES, (
            f"run 1 never lists {sorted(TARGET_NAMES)} as offered; offered: {sorted(offered)}")
    elif item == "NC2":
        run = _run(result, 0)
        entries = _target_entries(run)
        assert entries, (f"run 1 recorded no activation of the target; active_skills: "
                         f"{[t.get('active_skills') for t in run['traces']]}")
        assert [_identity(e) for e in entries] == [V1], (
            f"run 1's activation is {[_identity(e) for e in entries]}, not the installed "
            f"version {V1}")
    elif item == "NC3":
        run = _run(result, 0)
        assert run.get("target_loads", 0) >= 2, (
            f"the scenario loaded the target {run.get('target_loads')} time(s) in run 1; "
            "the item needs two loads to grade")
        entries = _target_entries(run)
        assert len(entries) == 1, (
            f"two loads in one run gave {len(entries)} activation entries: {entries}")
    elif item == "NC4":
        run = _run(result, 0)
        assert _found_tool(run, "fees.md", column.RESOURCE_MARKER), (
            f"no tool call in run 1 read {column.RESOURCE} and returned its bytes; "
            f"tool calls: {[i[:160] for i, _ in _tool_records(run)]}")
    elif item == "NC5":
        run = _run(result, 0)
        assert _found_tool(run, "compute_refund", column.SCRIPT_OUT), (
            f"no tool call in run 1 ran {column.SCRIPT} and returned {column.SCRIPT_OUT!r}; "
            f"tool calls: {[i[:160] for i, _ in _tool_records(run)]}")
    elif item == "NC6":
        run = _run(result, 1)
        text = "\n".join(run.get("received") or [])
        got_v1, got_v2 = V1_SENTINEL in text, V2_SENTINEL in text
        assert got_v1 != got_v2, (
            f"run 2's model received {'both' if got_v1 else 'neither'} version's body "
            "(sentinels); the item needs exactly one load of the updated skill")
        want = V1 if got_v1 else V2
        entries = _target_entries(run)
        assert [_identity(e) for e in entries] == [want], (
            f"after the update the model received {'v1' if got_v1 else 'v2'}, but run 2 "
            f"recorded {[_identity(e) for e in entries]}; expected {want}")
    else:
        raise ValueError(f"no item {item!r}; known: {ITEMS}")


def declared_na(framework: str, item: str) -> Optional[str]:
    return DECLARED_NA.get((framework, item))
