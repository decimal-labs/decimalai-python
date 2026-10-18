"""The live native check's contract: the only place its assertions live.

Per trial of one framework, with a REAL model:

==  ======================================================================================
L1  the model called the framework's own load tool for the target skill
L2  the trace DecimalAI sent carries the activation with the installed version's exact
    identity (wire name, skill id, version id, body hash)
L3  the answer carries the skill's policy (:data:`live.ANSWER_FACT`)
==  ======================================================================================

A framework passes when at least :data:`PASS_TRIALS` of its trials pass all three. A trial
the provider refused for billing or rate is not counted; if every trial was refused, the
framework is skipped, never passed.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Tuple

from tests.native import column
from tests.native import live
from tests.native.column_contract import V1, _identity, _target_entries

PASS_TRIALS = 2

#: framework -> (its native load tool, a test of that call's arguments for the target).
def _names_target(args: Any) -> bool:
    text = args if isinstance(args, str) else json.dumps(args, default=str)
    return column.TARGET in text


LOAD_TOOLS: Dict[str, Tuple[Tuple[str, ...], Callable[[Any], bool]]] = {
    "claude-agent-sdk": (("Skill",), _names_target),
    "google-adk": (("load_skill",), _names_target),
    "crewai": (("load_skill",), _names_target),
    "openai-agents": (("load_skill",), _names_target),
    "deepagents": (("read_file",), lambda a: _names_target(a) and "SKILL.md" in json.dumps(a)),
    "agent-framework": (("load_skill",), _names_target),
    "pydantic-ai-harness": (("load_capability",), _names_target),
    "strands-agents": (("skills",), _names_target),
}


def grade_trial(framework: str, run: Dict[str, Any]) -> List[str]:
    """The items one trial failed (empty: it passed)."""
    failed = []
    names, targets = LOAD_TOOLS[framework]
    if not any(c.get("name") in names and targets(c.get("args")) for c in run.get("calls") or []):
        failed.append(f"L1: no {'/'.join(names)} call for {column.TARGET}; calls: "
                      f"{[c.get('name') for c in run.get('calls') or []]}")
    entries = _target_entries(run) if run.get("traces") else []
    if [_identity(e) for e in entries] != [V1]:
        failed.append(f"L2: activation {[_identity(e) for e in entries]} is not {V1}")
    answer = (run.get("answer") or "").lower()
    if live.ANSWER_FACT not in answer:
        failed.append(f"L3: the answer does not say {live.ANSWER_FACT!r}: {answer[:200]!r}")
    return failed


def grade(framework: str, result: Dict[str, Any]) -> Tuple[str, List[str]]:
    """``("pass" | "fail" | "skip", one line per trial)``."""
    lines, passed, counted = [], 0, 0
    for i, run in enumerate(result.get("runs") or [], 1):
        if live.provider_wall(run.get("error")):
            lines.append(f"trial {i}: provider refused ({run['error'][:120]})")
            continue
        counted += 1
        failed = ([f"raised: {run['error']}"] if run.get("error") else []) + (
            [] if run.get("error") else grade_trial(framework, run))
        passed += not failed
        lines.append(f"trial {i}: {'pass' if not failed else 'FAIL ' + '; '.join(failed)}")
    if counted == 0:
        return "skip", lines
    return ("pass" if passed >= min(PASS_TRIALS, counted) else "fail"), lines
