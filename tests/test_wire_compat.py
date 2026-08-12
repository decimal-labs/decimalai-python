"""Wire-snapshot forward/backward compatibility for the SDK.

The SDK is the contract surface between customer code and the DecimalAI
backend. If we accidentally tighten the response shape (e.g., make a
previously-optional field required, or rename a key), customer code at
older or newer SDK versions starts failing.

This test pins three snapshots of the regression-check wire shape and
verifies the current SDK can consume each one:

  - v0.1: minimal, pre-pr_context era. No diff_summary, no pr_context.
  - v0.2: adds pr_context + diff_summary.
  - current: full shape with impacts[].
  - with-future-fields: current shape + unknown fields the backend
    might add later. SDK MUST pass them through (forward compat).

The SDK historically returns the regression-check payload as
`Dict[str, Any]` (see `Client.run_regression_check` /
`Client.get_regression_check`), so extra fields are trivially accepted;
the test locks that contract in so a future SDK refactor (e.g.,
switching to a strict dataclass) doesn't silently break customers
running older backends or hitting newer ones.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pytest


FIXTURES = Path(__file__).parent / "fixtures" / "wire"


def _load(name: str) -> Dict[str, Any]:
    return json.loads((FIXTURES / name).read_text())


# ─────────────────────────────────────────────────────────────────────
# Backward compat: older payloads still decode
# ─────────────────────────────────────────────────────────────────────


def test_v01_minimal_snapshot_decodes():
    """v0.1 had no pr_context, no impacts[], diff_summary could be null.

    The SDK must not blow up on these; `.get()` returns None for absent
    keys, and customer code is expected to handle that.
    """
    payload = _load("regression_check_v01.json")
    resp = payload

    assert resp["id"] == "rc_v01_a"
    assert resp["verdict"] == "high_risk"
    # The keys that didn't exist in v0.1 should be absent — the SDK
    # MUST NOT have injected defaults for them.
    assert "pr_context" not in resp
    assert "impacts" not in resp
    assert resp["diff_summary"] is None


def test_v02_with_pr_context_decodes():
    """v0.2 added pr_context + diff_summary.changes."""
    payload = _load("regression_check_v02_with_pr_context.json")
    resp = payload

    assert resp["id"] == "rc_v02_b"
    assert resp["pr_context"]["pr_number"] == 42
    assert resp["diff_summary"]["total_changes"] == 2
    change_types = {c["type"] for c in resp["diff_summary"]["changes"]}
    assert change_types == {"tool_removed", "prompt_section_rewritten"}
    # impacts[] still absent in v0.2.
    assert "impacts" not in resp


# ─────────────────────────────────────────────────────────────────────
# Current shape
# ─────────────────────────────────────────────────────────────────────


def test_current_full_shape_decodes():
    """The current shape: impacts[] + pr_context + diff_summary."""
    payload = _load("regression_check_current.json")
    resp = payload

    assert resp["verdict"] == "high_risk"
    assert resp["high_risk_count"] == 247
    assert len(resp["impacts"]) == 1
    impact = resp["impacts"][0]
    assert impact["surface_change_type"] == "tool_removed"
    assert impact["affected_trace_count"] == 247


# ─────────────────────────────────────────────────────────────────────
# Forward compat: unknown fields must pass through unchanged
# ─────────────────────────────────────────────────────────────────────


def test_unknown_fields_pass_through_unchanged():
    """If the backend adds a field the SDK has never heard of, the SDK
    MUST surface it transparently — never silently strip it. Customer
    code that knows about the new field needs to be able to read it.
    """
    payload = _load("regression_check_with_future_fields.json")
    resp = payload

    # Known fields still work.
    assert resp["verdict"] == "medium_risk"

    # Unknown fields are still there.
    assert resp["future_field_a"] == "this field does not exist in v0.x"  # type: ignore[typeddict-item]
    assert resp["future_field_b"] == {"nested": "value"}  # type: ignore[typeddict-item]
    assert resp["future_field_c"] == [1, 2, 3]  # type: ignore[typeddict-item]
    assert resp["auto_replay_budget_remaining_cents"] == 999  # type: ignore[typeddict-item]


# ─────────────────────────────────────────────────────────────────────
# Negative compat: required scalar keys are always present
# ─────────────────────────────────────────────────────────────────────


REQUIRED_KEYS = (
    "id",
    "agent_name",
    "baseline_manifest_id",
    "candidate_manifest_id",
    "status",
    "verdict",
    "high_risk_count",
    "medium_risk_count",
    "low_risk_count",
    "total_traces_analyzed",
    "created_at",
)


@pytest.mark.parametrize(
    "snapshot_name",
    [
        "regression_check_v01.json",
        "regression_check_v02_with_pr_context.json",
        "regression_check_current.json",
        "regression_check_with_future_fields.json",
    ],
)
def test_every_snapshot_has_the_required_scalar_keys(snapshot_name):
    """Across all supported versions, these keys must exist. If we
    ever drop one, customer dashboards break — this test ensures we
    notice before shipping.
    """
    payload = _load(snapshot_name)
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    assert not missing, (
        f"{snapshot_name} is missing required keys {missing}; "
        f"this would break customers."
    )


# ─────────────────────────────────────────────────────────────────────
# Contract: the RegressionCheckResponse TypedDict must COVER the full
# backend wire shape.
#
# An earlier rewrite corrected the phantom keys but under-shot the real
# shape — it omitted 9 fields the backend actually emits, so a
# typed caller reading `result["impacts"]` / `["error_message"]` got a
# false type error. There was no test guarding it, which is how the
# incomplete fix shipped. This pins the authoritative key-set so the
# TypedDict can't silently fall behind the wire again.
# ─────────────────────────────────────────────────────────────────────

# Source of truth: the backend's regression-check serializer
# `_serialize_regression_check` (the dict it builds, plus `impacts`
# which it adds when include_impacts=True — the create/get/run paths
# the SDK hits all pass include_impacts=True). KEEP IN SYNC with that
# serializer; if the backend adds a field here, add it to the TypedDict.
BACKEND_WIRE_KEYS = frozenset({
    "id",
    "agent_name",
    "baseline_manifest_id",
    "candidate_manifest_id",
    "status",
    "verdict",
    "verdict_message",
    "human_summary",
    "high_risk_count",
    "medium_risk_count",
    "low_risk_count",
    "total_traces_analyzed",
    "diff_summary",
    "pr_context",
    "error_message",
    "created_at",
    "eval_verdict",
    "eval_breakdown",
    "downstream_impact",
    "call_replay",
    "source",
    "baseline_manifest_label",
    "baseline_manifest_hash",
    "candidate_manifest_label",
    "candidate_manifest_hash",
    "human_decision",
    "impacts",
})


def test_typeddict_covers_full_backend_wire_shape():
    """Every field the backend serializer emits must be declared on the
    SDK's RegressionCheckResponse TypedDict. A typed customer reading a
    real wire field should never get a false typeddict-item error.
    """
    from decimalai._responses import RegressionCheckResponse

    declared = set(RegressionCheckResponse.__annotations__)
    missing = BACKEND_WIRE_KEYS - declared
    assert not missing, (
        f"RegressionCheckResponse is missing backend wire fields {sorted(missing)}; "
        f"a typed caller reading them gets a false type error. Add them to "
        f"_responses.py (source of truth: regression.py _serialize_regression_check)."
    )
