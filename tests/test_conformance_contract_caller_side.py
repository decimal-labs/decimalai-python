"""Unit tests for the conformance contract's caller-side items (C15 / C16).

They run in the DEFAULT suite, like ``test_conformance_contract_activation.py``,
for the same reason: a green matrix cell does not prove an item bites. Three
adapters pass C15 and C16 today, so nothing in the matrix ever shows them a
broken observer or a missing link. These feed each item the broken shapes
directly — an observer never called, called twice, handed a trace that never
shipped, handed the live trace instead of a copy, whose exception reached the
run; a run trace with no parent, the wrong parent, a parent that never shipped —
and assert it goes red on each, then green on the shape every observer adapter
produces.

Nothing here imports a framework. The inputs are the dicts the probe records off
the wire and the entries ``drivers.trace_observer`` records, so a shape that
passes here would pass in the matrix.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from tests.conformance import contract
from tests.conformance.drivers import (
    OBSERVER_FAILURE,
    OBSERVER_TAMPER,
    ObserverError,
    observed_mark,
    observed_since,
    trace_observer,
)
from tests.conformance.harness import Observation, Phase
from tests.conformance.probe import Probe, Recorded

ENCLOSING = str(uuid.uuid4())
RUN = str(uuid.uuid4())
RUN_TYPE = contract.OBSERVED_TRACE_TYPE


def _trace(tid: str, *, parent: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    out = {
        "id": tid,
        "parent_trace_id": parent,
        "agent_name": "conformance-agent",
        "status": "success",
        "routing_id": None,
        "skills_delivered": [],
        "user_input_preview": "Please look up the sentinel.",
        "spans": [],
        "llm_calls": [],
    }
    out.update(extra)
    return out


def _seen(trace: Dict[str, Any], type_: str = RUN_TYPE) -> Dict[str, Any]:
    """What ``drivers.trace_observer`` records for one call."""
    return {"lane": 0, "type": type_, "trace": dict(trace)}


def _obs(
    wire: List[Dict[str, Any]],
    observed: List[Dict[str, Any]],
    *,
    enclosing: Optional[List[str]] = None,
    exception: Optional[BaseException] = None,
    ran: bool = True,
) -> Observation:
    requests = [
        Recorded(seq=i, method="POST", path="/api/v1/traces", query={}, body=t, status=200)
        for i, t in enumerate(wire, start=1)
    ]
    phase = Phase(
        name="nested", ctxs=[], requests=requests, exception=exception,
        observed=observed, enclosing_trace_ids=[ENCLOSING] if enclosing is None else enclosing,
        ran=ran, na_reason=None if ran else "the nested driver hook is not implemented",
    )
    return Observation(driver=None, probe=Probe(), ctx=None, phases={"nested": phase})


def _good() -> tuple:
    run = _trace(RUN, parent=ENCLOSING)
    return [_trace(ENCLOSING), run], [_seen(run)]


def _c15(*args: Any, **kw: Any) -> contract.Result:
    return contract.c15_invocation_observer(_obs(*args, **kw))


def _c16(*args: Any, **kw: Any) -> contract.Result:
    return contract.c16_parent_link(_obs(*args, **kw))


# ── C15: the observer ────────────────────────────────────────────────────────


def test_c15_passes_on_the_shape_every_observer_adapter_produces() -> None:
    wire, observed = _good()
    result = _c15(wire, observed)
    assert result.status == contract.PASS, result.message


def test_c15_fails_when_the_observer_is_never_called() -> None:
    wire, _ = _good()
    result = _c15(wire, [])
    assert result.status == contract.FAIL
    assert "never called" in result.message


def test_c15_fails_when_one_run_is_handed_over_twice() -> None:
    wire, observed = _good()
    result = _c15(wire, observed * 2)
    assert result.status == contract.FAIL
    assert "2 times" in result.message


def test_c15_fails_when_the_observed_trace_never_shipped() -> None:
    """The failure shape of an observer whose exception killed the export."""
    _, observed = _good()
    result = _c15([_trace(ENCLOSING)], observed)
    assert result.status == contract.FAIL
    assert "shipped no trace of its own" in result.message

    other = _trace(str(uuid.uuid4()), parent=ENCLOSING)
    result = _c15([_trace(ENCLOSING), other], observed)
    assert result.status == contract.FAIL
    assert "never received" in result.message


def test_c15_fails_when_a_shipped_run_was_never_handed_over() -> None:
    wire, observed = _good()
    second = _trace(str(uuid.uuid4()), parent=ENCLOSING)
    result = _c15(wire + [second], observed)
    assert result.status == contract.FAIL
    assert "never handed to the observer" in result.message


@pytest.mark.parametrize("field,value", [
    ("parent_trace_id", None),
    ("agent_name", "someone-else"),
    ("status", "error"),
    ("skills_delivered", ["refund-policy"]),
])
def test_c15_fails_when_the_observed_copy_disagrees_with_the_wire(field: str, value: Any) -> None:
    """The caller records the run from the observer; a copy taken before the
    parent was set (say) records a different run from the one that shipped."""
    wire, observed = _good()
    observed[0]["trace"][field] = value
    result = _c15(wire, observed)
    assert result.status == contract.FAIL
    assert field in result.message


def test_c15_fails_when_the_observer_was_handed_the_live_trace() -> None:
    wire, observed = _good()
    wire[1]["skills_delivered"] = [OBSERVER_TAMPER]
    observed[0]["trace"]["skills_delivered"] = [OBSERVER_TAMPER]
    result = _c15(wire, observed)
    assert result.status == contract.FAIL
    assert "not a copy" in result.message


def test_c15_fails_when_the_observers_exception_reached_the_run() -> None:
    wire, observed = _good()
    result = _c15(wire, observed, exception=ObserverError(OBSERVER_FAILURE))
    assert result.status == contract.FAIL
    assert "cost the caller's run nothing" in result.message


def test_c15_fails_when_the_argument_is_not_a_run_trace() -> None:
    wire, observed = _good()
    observed[0]["type"] = "builtins.str"
    result = _c15(wire, observed)
    assert result.status == contract.FAIL
    assert "builtins.str" in result.message


def test_c15_fails_rather_than_passing_a_phase_that_never_ran() -> None:
    result = _c15([], [], ran=False)
    assert result.status == contract.FAIL


# ── C16: the parent link ─────────────────────────────────────────────────────


def test_c16_passes_when_the_run_is_filed_under_the_enclosing_trace() -> None:
    wire, observed = _good()
    result = _c16(wire, observed)
    assert result.status == contract.PASS, result.message
    assert ENCLOSING in result.message


@pytest.mark.parametrize("parent", [None, str(uuid.uuid4())])
def test_c16_fails_when_the_run_is_unlinked_or_linked_elsewhere(parent: Optional[str]) -> None:
    run = _trace(RUN, parent=parent)
    result = _c16([_trace(ENCLOSING), run], [_seen(run)])
    assert result.status == contract.FAIL
    assert "credited twice" in result.message


def test_c16_fails_when_the_enclosing_trace_never_shipped() -> None:
    run = _trace(RUN, parent=ENCLOSING)
    result = _c16([run], [_seen(run)])
    assert result.status == contract.FAIL
    assert "never reached the wire" in result.message


def test_c16_fails_when_the_run_shipped_nothing_of_its_own() -> None:
    result = _c16([_trace(ENCLOSING)], [])
    assert result.status == contract.FAIL
    assert "nothing to link" in result.message


def test_c16_blames_the_harness_when_no_enclosing_trace_was_recorded() -> None:
    wire, observed = _good()
    result = _c16(wire, observed, enclosing=[])
    assert result.status == contract.FAIL
    assert "harness defect" in result.message


def test_the_flag_gates_both_items_and_needs_a_reason_and_a_hook() -> None:
    """One hand-over, one flag: an adapter with an observer is graded on the
    link the same day. False needs a printed reason, True needs ``run_nested``."""
    from tests.conformance.drivers import CAPABILITY_ITEMS, Capabilities, Driver

    assert CAPABILITY_ITEMS["has_invocation_observer"] == ("C15", "C16")
    none = Capabilities(
        has_invocation_observer=False, reasons={"has_invocation_observer": "no on_trace"},
    )
    assert none.na_reason("C15") == none.na_reason("C16") == "no on_trace"
    with pytest.raises(ValueError):
        Capabilities(has_invocation_observer=False)
    with pytest.raises(ValueError, match="run_nested"):
        Driver(
            name="x", covers=frozenset(), requires=(), entrypoint="x", run=lambda c: None,
            capabilities=Capabilities(
                has_skills_rail=False, supports_concurrency=False,
                supports_error_path=False, supports_degenerate=False,
                reasons={f: "n/a" for f in (
                    "has_skills_rail", "supports_concurrency",
                    "supports_error_path", "supports_degenerate",
                )},
            ),
        )


# ── the suite's observer ─────────────────────────────────────────────────────


def test_the_suite_observer_records_then_edits_then_raises() -> None:
    """Each step is a C15 clause: what it was handed (recorded before anything
    else), whether that was a copy (the edit), and isolation (the raise)."""
    from decimalai.schema.trace import RunTrace

    ctx = SimpleNamespace(lane=3)
    trace = RunTrace(agent_name="conformance-agent", parent_trace_id=ENCLOSING)
    cursor = observed_mark()
    with pytest.raises(ObserverError, match=OBSERVER_FAILURE):
        trace_observer(ctx)(trace)  # type: ignore[arg-type]
    [entry] = observed_since(cursor)
    assert entry["lane"] == 3 and entry["type"] == RUN_TYPE
    assert entry["trace"]["id"] == str(trace.id)
    assert entry["trace"]["parent_trace_id"] == ENCLOSING
    assert OBSERVER_TAMPER not in str(entry["trace"]), "recorded after the edit"
    assert OBSERVER_TAMPER in trace.skills_delivered, "the edit did not happen"
