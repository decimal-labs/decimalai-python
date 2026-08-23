"""Trace schema models for the DecimalAI SDK.

These Pydantic models define the SDK-side view of trace data
that gets sent to the platform backend.

Trace entities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .common import CallRole, FinishReason, SourceType, SpanType, Status


class ToolCallRecord(BaseModel):
    """An individual tool invocation within an LLM call."""

    id: UUID = Field(default_factory=uuid4)
    tool_name: str
    tool_version: Optional[str] = None
    status: Status = Status.SUCCESS
    args: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Any] = None
    latency_ms: Optional[int] = None


class LlmCallRecord(BaseModel):
    """A single LLM invocation with full fidelity.

    This is the most important artifact for SFT derivation.
    """

    id: UUID = Field(default_factory=uuid4)
    span_id: Optional[UUID] = None
    agent_name: Optional[str] = None
    call_role: CallRole = CallRole.OTHER
    provider: Optional[str] = None
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = None
    # The exact rendered request and response
    rendered_input: Optional[List[Dict[str, Any]]] = None
    output: Optional[Dict[str, Any]] = None
    # Tool calls triggered by this LLM call
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    finish_reason: Optional[FinishReason] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    # ── Prompt-cache split ────────────────────────────────────────────────
    # Carried as the provider reported them, NEVER folded into
    # ``input_tokens``. DecimalAI injects a per-query skill menu at the front
    # of the system prompt, and varying bytes at position zero defeat every
    # provider's prefix cache for everything behind them — so whether a
    # prompt stayed cacheable is a number DecimalAI has to be able to show.
    # A folded total cannot answer it (that fold is what the Claude adapter
    # used to do, and it is why the signal did not exist).
    #
    # ``None`` and ``0`` are DIFFERENT FACTS and both round-trip to the
    # platform intact:
    #     None  the provider did not report it (no cache concept, older
    #           adapter, usage block absent)
    #     0     the provider reported it, and it was zero — a cache MISS
    # Leave the field unset rather than defaulting it to 0; "we never
    # measured" must not masquerade as "we measured a cold cache".
    #
    # The two providers relate these to ``input_tokens`` in OPPOSITE ways.
    # Do not sum them without checking ``provider``:
    #   Anthropic  ``input_tokens`` is the UNCACHED REMAINDER, so both fields
    #              are ADDITIONAL to it.
    #                cache_read_tokens     = usage.cache_read_input_tokens
    #                cache_creation_tokens = usage.cache_creation_input_tokens
    #                effective prompt = input + cache_read + cache_creation
    #   OpenAI     ``prompt_tokens`` ALREADY INCLUDES the cached part, so
    #              cache_read_tokens is a SUBSET of ``input_tokens`` and
    #              cache_creation_tokens stays None (the auto-cache has no
    #              separately-reported creation step).
    #                cache_read_tokens = usage.prompt_tokens_details
    #                                         .cached_tokens
    #                effective prompt = input_tokens (already whole)
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    latency_ms: Optional[int] = None
    cost_usd: Optional[float] = None
    # Streaming support
    streaming: bool = False
    streaming_token_count: Optional[int] = None
    # Multi-modal content type
    content_type: str = "text"  # text | image | audio | multimodal
    # Structured output schema
    response_format: Optional[Dict[str, Any]] = None
    status: Status = Status.SUCCESS
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class TraceSpan(BaseModel):
    """A span within a trace. Nests via parent_span_id."""

    id: UUID = Field(default_factory=uuid4)
    parent_span_id: Optional[UUID] = None
    span_type: SpanType = SpanType.OTHER
    name: str = ""
    status: Status = Status.SUCCESS
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    input_preview: Optional[str] = None
    output_preview: Optional[str] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)


class RunTrace(BaseModel):
    """Top-level execution record. One per agent invocation.

    Sent from SDK to platform via POST /api/v1/traces.
    """

    id: UUID = Field(default_factory=uuid4)
    project: Optional[str] = None
    manifest_id: Optional[str] = None
    session_id: Optional[str] = None
    agent_name: Optional[str] = None
    status: Status = Status.SUCCESS
    source_type: SourceType = SourceType.PRODUCTION
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    user_input_preview: Optional[str] = None
    final_output_preview: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    # Multi-agent: link this trace to a parent orchestrator trace.
    # When set, the platform will propagate error status to the parent
    # and include this trace in the parent's child_summary.
    parent_trace_id: Optional[str] = None
    # SkillRouter: the `rt_<24-hex>` ID of the routing decision that
    # surfaced skills for this trace. When set, the platform joins
    # `routing_decision × trace_skill_activation` on this column to
    # answer offered-vs-activated — the primary skill-quality metric.
    # Populated by framework adapters that call
    # `SkillRouter.build_prompt_fragment()` at prompt-assembly time.
    routing_id: Optional[str] = None
    # Nested data — sent together
    spans: List[TraceSpan] = Field(default_factory=list)
    llm_calls: List[LlmCallRecord] = Field(default_factory=list)
    # Client-side eval scores — run before upload
    eval_scores: List[Dict[str, Any]] = Field(default_factory=list)
    # Skills active during this trace — first-class field
    # Each entry: {"name": "code-review", "hash": "sha256:..."} or just {"name": "code-review"}
    active_skills: List[Dict[str, Any]] = Field(default_factory=list)
    # Skill discovery telemetry. ``offered_in_prompt`` lists
    # names whose descriptions were in the agent's system prompt
    # registry. ``loaded_by_agent`` lists names whose SKILL.md the agent
    # actually read. Both feed the Skill Rater's "discoverability gap"
    # metric — high gap means relevant skills went unread.
    skills_offered_in_prompt: List[str] = Field(default_factory=list)
    skills_loaded_by_agent: List[str] = Field(default_factory=list)
    # Names whose full BODY reached the model (Router body injection or a
    # load_skill serve) — between offered (menu row) and
    # activated. Delivered implies offered, never implies activation.
    skills_delivered: List[str] = Field(default_factory=list)
    # Session aggregation — multi-turn context
    session_metadata: Dict[str, Any] = Field(default_factory=dict)
    turn_index: Optional[int] = None

