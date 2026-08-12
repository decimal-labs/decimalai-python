"""Typed response shapes for `DecimalAIClient` methods.

These are `TypedDict` classes — runtime-identical to `Dict[str, Any]` so all
existing callers that do ``result["manifest_id"]`` keep working unchanged, but
type checkers (mypy / pyright) can now flag missing keys and typos.

The classes mirror the backend's JSON response shapes. When the backend
evolves, update these definitions and the SDK's type information stays
in sync.

Why not `pydantic.BaseModel`? Two reasons: (1) it would break dict-style
access for the SDK's existing user base, and (2) the SDK is meant to be a
thin HTTP wrapper, not a re-validation layer over a service that already
validates with pydantic on the server side.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict

# ── Auth ────────────────────────────────────────────────────


class VerifyAuthResponse(TypedDict, total=False):
    """Response of ``GET /api/v1/auth/verify``.

    Mirrors the live backend shape (verified 2026-06-15 against the running
    service). The backend returns ``status``/``api_key_valid``/``scope``/
    ``project_id``/``workspace_id``/``permissions``/
    ``require_manifest_on_ingest``/``message`` — it does *not* return
    ``user_id``, ``org_id``, or ``plan``, so those must not be declared here
    (a ``cast()`` to this TypedDict previously told type checkers that
    ``result["plan"]`` was valid when it would ``KeyError`` at runtime).
    """

    status: str
    api_key_valid: bool
    scope: str
    project_id: Optional[str]
    workspace_id: Optional[str]
    permissions: Dict[str, Any]
    require_manifest_on_ingest: bool
    message: str


# ── Trace ingestion ─────────────────────────────────────────


class IngestionSuccess(TypedDict):
    """Successful ingestion of one or many traces.

    Single-trace (POST /traces) echoes ``id``/``trace_id``/``status``/``agent_name``/
    ``spans``/``llm_calls``. Batch (POST /traces/batch) returns ``count``/``trace_ids``
    and an optional per-item ``errors`` list.
    These keys are the REAL wire shape. An earlier version of this TypedDict
    declared imported_count/skipped_duplicates/failed, which the backend never
    emitted — a typed caller reading them would KeyError. TypedDict is
    runtime-identical to Dict[str, Any], so correcting it changed type hints
    only, never behavior.
    """

    id: NotRequired[str]
    trace_id: NotRequired[str]
    status: NotRequired[Literal["ok", "success"]]
    agent_name: NotRequired[str]
    spans: NotRequired[int]
    llm_calls: NotRequired[int]
    # batch (POST /traces/batch)
    count: NotRequired[int]
    trace_ids: NotRequired[list[str]]
    errors: NotRequired[list[dict]]


class IngestionSkipped(TypedDict):
    """Ingestion no-op response when the SDK is in ``manifest_only`` mode."""

    status: Literal["skipped"]
    reason: Literal["manifest_only_mode"]
    skipped_count: NotRequired[int]


IngestionResult = Union[IngestionSuccess, IngestionSkipped]


# ── Traces ──────────────────────────────────────────────────


class TraceListItem(TypedDict, total=False):
    """One row of `GET /api/v1/traces`."""

    id: str
    agent_name: Optional[str]
    manifest_id: Optional[str]
    parent_trace_id: Optional[str]
    status: str
    source_type: str
    started_at: Optional[str]
    ended_at: Optional[str]
    user_input_preview: Optional[str]
    final_output_preview: Optional[str]
    eval_verdict: Optional[str]
    eval_score: Optional[float]
    created_at: Optional[str]


class TraceListResponse(TypedDict):
    traces: List[TraceListItem]
    total: int
    limit: int
    offset: int


class TraceDetailResponse(TraceListItem, total=False):
    """Single-trace response from `GET /api/v1/traces/{id}`. Inherits list fields,
    adds spans / llm_calls / active_skills / eval_breakdown."""

    spans: List[Dict[str, Any]]
    llm_calls: List[Dict[str, Any]]
    active_skills: List[Dict[str, Any]]
    eval_breakdown: Optional[Dict[str, Any]]
    eval_details_json: Optional[Dict[str, Any]]
    error_message: Optional[str]


# ── Agents ──────────────────────────────────────────────────


class AgentSummary(TypedDict, total=False):
    agent_name: str
    trace_count: int
    last_trace_at: Optional[str]
    unevaluated_count: int
    is_subagent: bool
    latest_manifest: Optional[Dict[str, Any]]


class AgentListResponse(TypedDict, total=False):
    agents: List[AgentSummary]
    parent_child_map: Dict[str, List[str]]


# ── Manifests ───────────────────────────────────────────────


class ManifestRegistrationResponse(TypedDict, total=False):
    status: Literal["ok"]
    manifest_id: str
    manifest_hash: str
    version_label: Optional[str]
    is_new: bool
    components: int
    compatibility_report_id: Optional[str]


class ManifestSummary(TypedDict, total=False):
    id: str
    project_id: Optional[str]
    version_label: Optional[str]
    manifest_hash: str
    parent_manifest_id: Optional[str]
    status: str
    detection_source: Optional[str]
    agent_name: Optional[str]
    components_count: int
    created_at: Optional[str]


class ManifestListResponse(TypedDict):
    manifests: List[ManifestSummary]
    total: int
    limit: int
    offset: int


class ManifestDiff(TypedDict, total=False):
    """The inner structural diff (agentversion ``ManifestDiff`` shape)."""
    old_manifest_id: str
    new_manifest_id: str
    changed_surfaces: List[Dict[str, Any]]
    summary: Dict[str, Any]


class ManifestDiffResponse(TypedDict, total=False):
    # GET /api/v1/manifests/{id}/diff returns an ENVELOPE
    # {"diff": <ManifestDiff | None>, "message"?: str, "verdict"?: str}
    # — NOT the diff fields at top level. `diff` is None on a self-compare
    # (verdict="self_comparison") and when there is no parent (message=...). A
    # typed caller reading result["changed_surfaces"] would KeyError; the real
    # path is result["diff"]["changed_surfaces"]. Type-only (TypedDict == Dict at
    # runtime); no behavior change. Same defect class as the 2026-06-20
    # RegressionCheckResponse correction below.
    diff: Optional[ManifestDiff]
    message: str
    verdict: str


# ── Regression checks ───────────────────────────────────────


class RegressionCheckResponse(TypedDict, total=False):
    # These keys are the REAL wire shape emitted by the backend's
    # _serialize_regression_check (and match the GitHub Action's api.ts type). An
    # earlier version declared old_manifest_id/new_manifest_id/impact_summary/
    # pr_url/completed_at, none of which were ever sent — a typed caller reading
    # them would KeyError. Correcting it was type-only (TypedDict == Dict at
    # runtime); no behavior change.
    id: str
    agent_name: Optional[str]
    baseline_manifest_id: Optional[str]
    candidate_manifest_id: Optional[str]
    status: str
    verdict: Optional[str]
    verdict_message: Optional[str]
    human_summary: Optional[str]
    high_risk_count: Optional[int]
    medium_risk_count: Optional[int]
    low_risk_count: Optional[int]
    total_traces_analyzed: Optional[int]
    diff_summary: Optional[Dict[str, Any]]
    pr_context: Optional[Dict[str, Any]]
    eval_verdict: Optional[str]
    eval_breakdown: Optional[Dict[str, Any]]
    downstream_impact: Optional[Dict[str, Any]]
    created_at: Optional[str]
    # An earlier rewrite dropped the phantom keys but under-shot the real
    # shape. These 9 are also emitted by the backend's
    # _serialize_regression_check and reach the SDK on the
    # create/get/run paths. `impacts` is present only when the backend
    # serializes with include_impacts=True (create/get/run all do).
    error_message: Optional[str]
    source: Optional[str]
    call_replay: Optional[Dict[str, Any]]
    human_decision: Optional[str]
    baseline_manifest_label: Optional[str]
    baseline_manifest_hash: Optional[str]
    candidate_manifest_label: Optional[str]
    candidate_manifest_hash: Optional[str]
    impacts: Optional[List[Dict[str, Any]]]


class RegressionCheckListResponse(TypedDict):
    regression_checks: List[RegressionCheckResponse]
    total: int
    limit: int
    offset: int
