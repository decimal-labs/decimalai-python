"""Every native-column check is seen to fail.

A synthetic scenario result that does everything right passes all six items; each defect
below breaks exactly the item it names. A check weakened on a day every framework is clean
would otherwise go unnoticed.
"""
from __future__ import annotations

import copy
from typing import Any, Callable, Dict

import pytest

from . import column, column_contract as contract
from .conftest import V2_SENTINEL


def _trace(entry: Dict[str, str], *, offered=True, tools=True) -> Dict[str, Any]:
    calls = []
    if tools:
        calls = [{"tool_calls": [
            {"tool_name": "read", "args": {"path": f"x/{column.RESOURCE}"},
             "result": f"fees for {column.RESOURCE_MARKER}"},
            {"tool_name": "run", "args": {"cmd": f"python3 {column.SCRIPT} 100 45"},
             "result": column.SCRIPT_OUT + "\n"},
        ]}]
    return {"active_skills": [dict(entry)],
            "skills_offered_in_prompt": [contract.V1["name"]] if offered else [],
            "llm_calls": calls, "spans": []}


def good() -> Dict[str, Any]:
    return {"framework": "synthetic", "versions": {}, "na": {}, "notes": {}, "runs": [
        {"traces": [_trace(contract.V1)], "received": [contract.V1_SENTINEL],
         "target_loads": 2, "error": None, "route_calls": 0},
        {"traces": [_trace(contract.V2, tools=False)], "received": [V2_SENTINEL],
         "target_loads": 1, "error": None, "route_calls": 0},
    ]}


def _set(path, value) -> Callable[[Dict[str, Any]], None]:
    def apply(result):
        node = result
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
    return apply


DEFECTS = {
    "NC1": _set(["runs", 0, "traces", 0, "skills_offered_in_prompt"], []),
    "NC2": _set(["runs", 0, "traces", 0, "active_skills"],
                [dict(contract.V1, version_id=contract.V2["version_id"])]),
    "NC3": _set(["runs", 0, "traces", 0, "active_skills"], [contract.V1, contract.V1]),
    "NC4": _set(["runs", 0, "traces", 0, "llm_calls", 0, "tool_calls", 0, "result"], "denied"),
    "NC5": _set(["runs", 0, "traces", 0, "llm_calls", 0, "tool_calls", 1, "result"], "error"),
    # The model got v2 but the trace says v1: the file-re-read misattribution, inverted.
    "NC6": _set(["runs", 1, "traces", 0, "active_skills"], [contract.V1]),
    "NC7": _set(["runs", 0, "route_calls"], 1),
}


def test_nc7_also_fails_on_a_routing_id():
    result = copy.deepcopy(good())
    result["runs"][1]["traces"][0]["routing_id"] = "rt_" + "a" * 24
    with pytest.raises(AssertionError, match="routing ids"):
        contract.grade("NC7", result)


@pytest.mark.parametrize("item", contract.ITEMS)
def test_a_correct_run_passes(item):
    contract.grade(item, good())


@pytest.mark.parametrize("item", contract.ITEMS)
def test_each_defect_fails_its_item(item):
    result = copy.deepcopy(good())
    DEFECTS[item](result)
    with pytest.raises(AssertionError):
        contract.grade(item, result)


def test_nc6_follows_what_the_model_received_not_the_disk():
    # A loader that cached v1: the model received v1 after the update, and v1 is right.
    result = copy.deepcopy(good())
    result["runs"][1]["received"] = [contract.V1_SENTINEL]
    result["runs"][1]["traces"][0]["active_skills"] = [dict(contract.V1)]
    contract.grade("NC6", result)


def test_nc3_needs_the_scenario_to_load_twice():
    result = copy.deepcopy(good())
    result["runs"][0]["target_loads"] = 1
    with pytest.raises(AssertionError, match="two loads"):
        contract.grade("NC3", result)
