"""Closed metadata contract shared with the public SDK evidence exporter."""

import hashlib
import json
import math
import re
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

KINDS = frozenset(
    {
        "runtime.boot",
        "runtime.heartbeat",
        "runtime.stopping",
        "runtime.exited",
        "runtime.module",
        "supervisor.started",
        "supervisor.completed",
        "http.attempt",
        "http.response",
        "quality.page_failed",
        "quality.page_invalid",
        "quality.page_success",
        "quality.checkpoint_saved",
        "quality.checkpoint_failed",
        "quality.checkpoint_processed",
        "identity.row",
        "identity.turn",
        "paid.saved",
        "paid.readback",
        "adapter.invocation",
        "lifecycle.baseline",
        "lifecycle.readback",
        "window.sealed",
        "stream.closed",
        "collection.gap",
    }
)
FIELDS = frozenset(
    {
        "lane",
        "owner_digest",
        "cursor_digest",
        "next_cursor_digest",
        "tail_cursor_digest",
        "digest_scheme",
        "before_retry_due",
        "after_retry_due",
        "before_saved_at",
        "after_saved_at",
        "before_failed_request",
        "after_failed_request",
        "cursor_present",
        "next_cursor_present",
        "cursor_advanced",
        "cursor_progress",
        "cursor_state",
        "returned_rows",
        "invalid_shape",
        "attempt",
        "status",
        "elapsed_seconds",
        "retry_selected",
        "timeout_phase",
        "retry_after",
        "write_completed",
        "trace_id",
        "parent_trace_id",
        "model_call_ids",
        "witnesses",
        "witnesses_complete",
        "framework",
        "skill_id",
        "version_id",
        "skill_hash",
        "source_skill_id",
        "name_digest",
        "phase",
        "terminal",
        "created",
        "previous_due",
        "retry_due",
        "observed_at",
        "receipt_ids",
        "receipt_id",
        "result_hash",
        "provider",
        "model",
        "prompt_hash",
        "rubric_hash",
        "value",
        "verdict",
        "assessed_sources",
        "before",
        "after",
        "ingress",
        "terminal_service",
        "read_attempts",
        "read_limit",
        "seconds_limit",
        "batch_available",
        "batch_reads",
        "batch_traces",
        "pass_id",
        "attempt_id",
        "cycle",
        "child_pid",
        "supervisor_pid",
        "supervisor_head",
        "producer_head",
        "started_at",
        "finished_at",
        "exit_code",
        "stop_reason",
        "left",
        "admission_cap_seconds",
        "deadline_ts",
        "pass_outcome",
        "assessment_attempted",
        "assessment_valid",
        "assessed",
        "graded",
        "published",
        "publication_pending",
        "publication_unchecked",
        "publication_read_errors",
        "confirmation_reserve_seconds",
        "summary_hash",
        "module",
        "artifact_hash",
        "capture_class",
        "release_id",
        "git_head",
        "versions",
        "writer_guard",
        "observer_failures",
        "before_observer_failures",
        "after_observer_failures",
        "spool_pending",
        "spool_bytes",
        "scope_complete",
        "final_seq",
        "ranges",
        "expected_events",
        "agent_id",
        "fence_at",
        "source_floors",
        "raw_counts",
        "reason",
        "count",
    }
)
NESTED = (
    frozenset(
        {
            "request_id",
            "published_at",
            "canonical_receipts_complete",
            "next_after_id",
            "stream_id",
            "first",
            "last",
            "pending",
            "oldest_created",
            "oldest_due",
            "receipt",
            "identity",
            "uses_all_time",
            "distinct_users_all_time",
            "traces",
            "manifests",
            "llm_calls",
            "spans",
            "prompts",
            "skills",
            "python",
            "sdk",
            "adk",
            "genai",
            "pydantic",
        }
    )
    | FIELDS
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def metadata(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        raise ValueError("metadata nesting limit")
    if isinstance(value, dict):
        if len(value) > 70 or any(k not in NESTED for k in value):
            raise ValueError("unsupported metadata field")
        return {k: metadata(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 100:
            raise ValueError("metadata list limit")
        return [metadata(v, depth + 1) for v in value]
    if value is None or type(value) in (bool, int):
        if type(value) is int and not -(2**63) < value < 2**63:
            raise ValueError("integer limit")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("finite numbers required")
        return value
    if (
        isinstance(value, str)
        and len(value) <= 160
        and re.fullmatch(r"[A-Za-z0-9_.:+/ -]*", value)
    ):
        # No free text, authorization values, raw bodies or paths.
        if any(
            word in value.lower()
            for word in (
                "dai_sk_",
                "bearer ",
                "api_key",
                "/users/",
                "/home/",
                "sk-",
                "aiza",
                "ghp_",
                "gsk_",
                "ya29.",
            )
        ):
            raise ValueError("sensitive metadata")
        return value
    raise ValueError("unsupported metadata value")


class EvidenceEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(default=1, ge=1, le=1)
    event_id: UUID
    seq: int = Field(ge=1, le=2**63 - 1)
    boot_id: UUID
    kind: str = Field(max_length=64)
    occurred_at: datetime
    boot_elapsed_ms: int = Field(ge=0)
    operation_id: str | None = Field(
        default=None, max_length=128, pattern=r"^[A-Za-z0-9_.:+-]+$"
    )
    correlation: dict[str, str] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("timezone required")
        return value

    @field_validator("kind")
    @classmethod
    def kind_known(cls, value: str) -> str:
        if value not in KINDS:
            raise ValueError("unsupported event kind")
        return value

    @field_validator("correlation")
    @classmethod
    def correlation_safe(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) - {"request_id", "trace_id", "parent_trace_id"} or any(
            len(v) > 128 or not re.fullmatch(r"[A-Za-z0-9_.:+-]+", v)
            for v in value.values()
        ):
            raise ValueError("invalid correlation")
        for item in value.values():
            metadata(item)
        return value

    @field_validator("operation_id")
    @classmethod
    def operation_safe(cls, value: str | None) -> str | None:
        if value is not None:
            metadata(value)
        return value

    @field_validator("payload")
    @classmethod
    def payload_safe(cls, value: dict[str, Any]) -> dict[str, Any]:
        if set(value) - FIELDS:
            raise ValueError("unsupported payload field")
        metadata(value)
        for key in (
            "trace_id",
            "parent_trace_id",
            "lane",
            "owner_digest",
            "cursor_digest",
            "module",
            "artifact_hash",
            "capture_class",
            "release_id",
            "attempt_id",
            "framework",
            "source_skill_id",
            "provider",
            "model",
            "prompt_hash",
            "rubric_hash",
            "result_hash",
            "agent_id",
            "fence_at",
        ):
            if (
                key in value
                and value[key] is not None
                and not isinstance(value[key], str)
            ):
                raise ValueError("identifier metadata requires strings")
        for key in ("ranges", "witnesses", "source_floors"):
            if (
                key in value
                and value[key] is not None
                and (
                    not isinstance(value[key], list)
                    or any(not isinstance(row, dict) for row in value[key])
                )
            ):
                raise ValueError("structured metadata requires object rows")
        for key in ("before", "after"):
            if key in value and (
                not isinstance(value[key], dict)
                or set(value[key]) != {"receipt", "identity"}
                or any(not isinstance(row, dict) for row in value[key].values())
            ):
                raise ValueError("queue snapshot shape")
        for key in ("terminal_service", "ingress", "raw_counts", "versions"):
            if key in value and not isinstance(value[key], dict):
                raise ValueError("object metadata required")
        for key in ("model_call_ids", "assessed_sources", "receipt_ids"):
            if key in value and (
                not isinstance(value[key], list)
                or any(not isinstance(row, str) for row in value[key])
            ):
                raise ValueError("identifier list required")
        if len(canonical(value)) > 7000:
            raise ValueError("event payload too large")
        return value


class EvidenceBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int = Field(default=1, ge=1, le=1)
    batch_id: UUID
    stream_id: UUID
    events: list[EvidenceEventInput] = Field(min_length=1, max_length=100)
