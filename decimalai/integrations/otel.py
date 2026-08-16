"""OpenTelemetry integration for DecimalAI.

Provides a ``DecimalSpanExporter`` that receives OTel spans, maps them
to DecimalAI ``RunTrace``/``LlmCallRecord`` models, and sends them to
the DecimalAI backend via the background sender.

Maps OpenTelemetry GenAI semantic convention attributes to DecimalAI
fields for interoperability with the broader OTel ecosystem.

Usage::

    from decimalai.integrations.otel import install_otel

    install_otel(agent_name="my-agent")

    # Then use any OTel-instrumented library — spans are auto-captured
    from opentelemetry import trace
    tracer = trace.get_tracer("my-agent")
    with tracer.start_as_current_span("handle_request") as span:
        span.set_attribute("gen_ai.request.model", "gpt-4o")
        ...

Requires the OpenTelemetry SDK (a core dependency of decimalai).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

# Shared with the manifest-capable exporter so both rails read the message
# attributes identically and register manifests by the same rules. Pure-stdlib
# module, no cycle.
from ..otel import (
    _extract_system_prompt,
    _ManifestRegistry,
    _messages_from_attrs,
    _submit_or_send_inline,
)

logger = logging.getLogger("decimalai.integrations.otel")

# ── OTel GenAI semantic convention attribute keys ─────────────────
# See: https://opentelemetry.io/docs/specs/semconv/gen-ai/

OTEL_GEN_AI_SYSTEM = "gen_ai.system"
OTEL_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
OTEL_GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
OTEL_GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
OTEL_GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
OTEL_GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
OTEL_GEN_AI_RESPONSE_FINISH_REASON = "gen_ai.response.finish_reasons"
OTEL_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"

# Agent-level attributes (emitted by CrewAI, AutoGen/AG2)
OTEL_GEN_AI_AGENT_NAME = "gen_ai.agent.name"
OTEL_GEN_AI_AGENT_ID = "gen_ai.agent.id"
OTEL_GEN_AI_TOOL_NAME = "gen_ai.tool.name"

# Content attributes (various instrumentors use different keys)
# We check all of them to maximise compatibility.
_INPUT_ATTR_KEYS = [
    "gen_ai.content.prompt",       # GenAI semconv
    "gen_ai.prompt",               # OpenLLMetry / Traceloop
    "input",                       # OpenInference
    "input.value",                 # OpenInference v2
    "llm.input_messages",          # LangSmith-style
    "gen_ai.tool.call.arguments",  # GenAI semconv tool span (AG2)
]
_OUTPUT_ATTR_KEYS = [
    "gen_ai.content.completion",   # GenAI semconv
    "gen_ai.completion",           # OpenLLMetry / Traceloop
    "output",                      # OpenInference
    "output.value",                # OpenInference v2
    "llm.output_messages",         # LangSmith-style
    "gen_ai.tool.call.result",     # GenAI semconv tool span (AG2)
]

# DecimalAI-specific attributes users can set on spans
DECIMAL_AGENT_NAME = "decimal.agent_name"
DECIMAL_COST_USD = "decimal.cost_usd"
DECIMAL_SPAN_TYPE = "decimal.span_type"


def _get_otel_sdk():
    """Lazy-import OTel SDK with helpful error message."""
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
        from opentelemetry.sdk.trace.export import (
            SpanExporter,
            SpanExportResult,
        )
        return otel_trace, TracerProvider, ReadableSpan, SpanExporter, SpanExportResult
    except ImportError:
        raise ImportError(
            "OpenTelemetry SDK is required for OTel integration but is missing "
            "(it ships as a core dependency of decimalai). "
            "Reinstall with: pip install decimalai"
        )


class DecimalSpanExporter:
    """OTel SpanExporter that maps spans to DecimalAI traces.

    Collects spans from a single trace into a RunTrace, extracting
    GenAI semantic convention attributes into LlmCallRecord fields.
    """

    def __init__(
        self,
        agent_name: Optional[str] = None,
    ):
        self.agent_name = agent_name
        # Buffer spans by trace_id to assemble into RunTrace
        self._trace_buffers: Dict[str, List[Any]] = {}
        # Every trace this exporter produced used to go out with no
        # manifest_id, so under require_manifest_on_ingest (the default, and on
        # in production) the backend 400'd 100% of them — install_otel() looked
        # wired up and delivered nothing. Shares the manifest-capable exporter's
        # registry so both rails accumulate and version identically.
        self._manifests = _ManifestRegistry()

    def export(self, spans: Sequence[Any]) -> Any:
        """Export a batch of OTel spans to DecimalAI."""
        _, _, _, _, SpanExportResult = _get_otel_sdk()

        from .. import _config

        if not _config._is_enabled():
            return SpanExportResult.SUCCESS

        try:
            client = _config._get_client()
        except Exception:
            return SpanExportResult.FAILURE

        # Group spans by trace_id
        grouped: Dict[str, List[Any]] = {}
        for span in spans:
            trace_id = format(span.context.trace_id, "032x")
            grouped.setdefault(trace_id, []).append(span)

        for trace_id, trace_spans in grouped.items():
            try:
                self._export_trace(trace_id, trace_spans, client)
            except Exception:
                logger.exception("Failed to export OTel trace %s", trace_id[:12])

        return SpanExportResult.SUCCESS

    def _export_trace(
        self,
        trace_id: str,
        spans: List[Any],
        client: Any,
    ) -> None:
        """Convert a batch of OTel spans to a RunTrace and send."""
        from ..schema.common import FinishReason, SpanType, Status
        from ..schema.trace import LlmCallRecord, RunTrace, TraceSpan

        llm_calls: List[LlmCallRecord] = []
        trace_spans: List[TraceSpan] = []
        agent_name = self.agent_name
        earliest_start = None
        latest_end = None
        user_input = None
        final_output = None
        # Manifest accumulators for this trace; folded into the per-agent
        # cumulative view by _ManifestRegistry below.
        seen_model: Optional[Dict[str, Any]] = None
        seen_tools: Dict[str, Dict[str, Any]] = {}
        seen_prompts: Dict[str, str] = {}

        # ── Keep the shape the framework emitted ──
        # OTel names a span's parent by a 64-bit id; a TraceSpan is identified
        # by a UUID we mint. So every span's UUID has to exist BEFORE any span
        # is built — children are routinely exported ahead of their parents
        # (they end first) and a link resolved lazily would simply be dropped.
        # Without this every span went out with parent_span_id unset and the
        # whole trace arrived as a flat list of same-level roots. Same fix, and
        # same reason, as in :mod:`decimalai.otel`.
        span_uuids: Dict[int, Any] = {}
        for span in spans:
            sid = _span_id_of(span)
            if sid is not None and sid not in span_uuids:
                span_uuids[sid] = uuid4()

        for span in spans:
            attrs = dict(span.attributes or {})
            # This span's identity, and its parent's — the latter only when the
            # parent is one of the spans in this trace. A parent that never
            # reached this exporter stays unset rather than becoming a dangling
            # pointer.
            span_uuid = span_uuids.get(_span_id_of(span)) or uuid4()
            parent_uuid = span_uuids.get(_parent_span_id_of(span))

            # Detect agent name from span attributes.
            # Priority: decimal.agent_name > gen_ai.agent.name
            if not agent_name:
                agent_name = (
                    attrs.get(DECIMAL_AGENT_NAME)
                    or attrs.get(OTEL_GEN_AI_AGENT_NAME)
                )
                if agent_name:
                    agent_name = str(agent_name)

            # Extract timestamps
            start_time = _ns_to_datetime(span.start_time) if span.start_time else None
            end_time = _ns_to_datetime(span.end_time) if span.end_time else None

            if start_time:
                if earliest_start is None or start_time < earliest_start:
                    earliest_start = start_time
            if end_time:
                if latest_end is None or end_time > latest_end:
                    latest_end = end_time

            # Determine span status
            status = Status.SUCCESS
            if hasattr(span, 'status') and span.status:
                status_code = getattr(span.status, 'status_code', None)
                if status_code and str(status_code).endswith('ERROR'):
                    status = Status.ERROR

            # Check if this is a GenAI/LLM span
            model_name = attrs.get(OTEL_GEN_AI_REQUEST_MODEL)
            operation = attrs.get(OTEL_GEN_AI_OPERATION_NAME, "")

            # ── Classify the span ───────────────────────────────
            tool_name = attrs.get(OTEL_GEN_AI_TOOL_NAME)
            is_tool_span = bool(tool_name) or "execute_tool" in str(operation).lower()

            if model_name or ("gen_ai" in str(operation).lower() and not is_tool_span):
                # ── LLM call span ──
                input_tokens = attrs.get(OTEL_GEN_AI_USAGE_INPUT_TOKENS)
                output_tokens = attrs.get(OTEL_GEN_AI_USAGE_OUTPUT_TOKENS)

                finish_reason = None
                raw_fr = attrs.get(OTEL_GEN_AI_RESPONSE_FINISH_REASON)
                if raw_fr:
                    fr_str = raw_fr[0] if isinstance(raw_fr, (list, tuple)) else str(raw_fr)
                    try:
                        finish_reason = FinishReason(fr_str)
                    except ValueError:
                        finish_reason = None

                latency_ms = None
                if start_time and end_time:
                    latency_ms = int((end_time - start_time).total_seconds() * 1000)

                cost_usd = attrs.get(DECIMAL_COST_USD)

                llm_calls.append(LlmCallRecord(
                    # Which span this call happened in, so a consumer can put
                    # it back in the waterfall instead of guessing by timestamp.
                    span_id=span_uuid,
                    model_name=str(model_name) if model_name else None,
                    provider=str(attrs.get(OTEL_GEN_AI_SYSTEM, "")),
                    # The FULL rendered request/response — the previews below
                    # are display strings, these are the SFT artifact.
                    rendered_input=_rendered_input_from_attrs(attrs),
                    output=_output_message_from_attrs(attrs),
                    input_tokens=int(input_tokens) if input_tokens else None,
                    output_tokens=int(output_tokens) if output_tokens else None,
                    temperature=float(attrs.get(OTEL_GEN_AI_REQUEST_TEMPERATURE, 0)) if attrs.get(OTEL_GEN_AI_REQUEST_TEMPERATURE) else None,
                    latency_ms=latency_ms,
                    cost_usd=float(cost_usd) if cost_usd else None,
                    finish_reason=finish_reason,
                    status=status,
                    started_at=start_time,
                    ended_at=end_time,
                ))

                # Manifest auto-detection: first model wins, first system
                # prompt wins (a later, dynamically-built one is per-run
                # content and would flip the manifest hash mid-trace).
                if seen_model is None and model_name:
                    seen_model = {
                        "provider": str(attrs.get(OTEL_GEN_AI_SYSTEM) or "") or None,
                        "model": str(model_name),
                        "temperature": (
                            float(attrs[OTEL_GEN_AI_REQUEST_TEMPERATURE])
                            if attrs.get(OTEL_GEN_AI_REQUEST_TEMPERATURE) is not None
                            else None
                        ),
                        "max_tokens": (
                            int(attrs[OTEL_GEN_AI_REQUEST_MAX_TOKENS])
                            if attrs.get(OTEL_GEN_AI_REQUEST_MAX_TOKENS) is not None
                            else None
                        ),
                    }
                if "system" not in seen_prompts:
                    system_prompt = _extract_system_prompt(attrs)
                    if system_prompt:
                        seen_prompts["system"] = system_prompt

                # Try to extract user input from this LLM span
                if not user_input:
                    user_input = _extract_preview(attrs, "input")
                # Always update final_output (last LLM span wins)
                output_content = _extract_preview(attrs, "output")
                if output_content:
                    final_output = output_content

            elif is_tool_span:
                # ── Tool execution span (CrewAI / AutoGen) ──
                resolved_name = str(tool_name or span.name or "unknown_tool")
                seen_tools.setdefault(resolved_name, {"name": resolved_name})
                tool_input = _extract_preview(attrs, "input") or ""
                tool_output = _extract_preview(attrs, "output") or ""

                trace_spans.append(TraceSpan(
                    id=span_uuid,
                    parent_span_id=parent_uuid,
                    span_type=SpanType.TOOL,
                    name=resolved_name,
                    status=status,
                    started_at=start_time,
                    ended_at=end_time,
                    input_preview=tool_input[:200] if tool_input else None,
                    output_preview=tool_output[:200] if tool_output else None,
                ))
            else:
                # ── Regular span ──
                span_type_str = attrs.get(DECIMAL_SPAN_TYPE, "")

                # Auto-classify from span name / operation for known frameworks
                if not span_type_str:
                    span_name_lower = (span.name or "").lower()
                    if "invoke_agent" in span_name_lower or "agent" in span_name_lower:
                        span_type_str = "agent"
                    elif "retriev" in span_name_lower:
                        span_type_str = "retrieval"
                    elif "guardrail" in span_name_lower:
                        span_type_str = "guardrail"
                    elif "handoff" in span_name_lower:
                        span_type_str = "handoff"
                    elif "planning" in span_name_lower or "plan" in span_name_lower:
                        span_type_str = "planning"
                    else:
                        span_type_str = "other"

                try:
                    span_type = SpanType(span_type_str)
                except ValueError:
                    span_type = SpanType.OTHER

                span_input = _extract_preview(attrs, "input")
                span_output = _extract_preview(attrs, "output")

                trace_spans.append(TraceSpan(
                    id=span_uuid,
                    parent_span_id=parent_uuid,
                    span_type=span_type,
                    name=span.name or "unknown",
                    status=status,
                    started_at=start_time,
                    ended_at=end_time,
                    input_preview=(span_input or "")[:200] or None,
                    output_preview=(span_output or "")[:200] or None,
                ))

                # Root-level agent spans often carry final input/output
                if span_type == SpanType.AGENT:
                    if not user_input and span_input:
                        user_input = span_input[:500]
                    if span_output:
                        final_output = span_output[:500]

        # Register/refresh this agent's manifest before assembling the trace —
        # the backend rejects a trace with no manifest_id, and a run that
        # observed no model/tool/prompt still gets one (see _ManifestRegistry).
        resolved_agent_name = agent_name or "unknown"
        manifest_id = self._manifests.manifest_id_for(
            resolved_agent_name,
            model=seen_model,
            tools=seen_tools,
            prompts=seen_prompts,
        )

        # Assemble RunTrace
        trace = RunTrace(
            id=uuid4(),
            agent_name=resolved_agent_name,
            manifest_id=manifest_id,
            status=Status.SUCCESS if all(
                s.status == Status.SUCCESS for s in trace_spans
            ) and all(
                lc.status == Status.SUCCESS for lc in llm_calls
            ) else Status.ERROR,
            source_type="production",
            started_at=earliest_start,
            ended_at=latest_end,
            user_input_preview=user_input,
            final_output_preview=final_output,
            spans=trace_spans,
            llm_calls=llm_calls,
        )

        # Send via background sender (falling back to an inline POST when the
        # interpreter is already tearing the thread pool down).
        _submit_or_send_inline(client, trace)
        logger.debug(
            "Exported OTel trace %s → RunTrace %s (%d spans, %d llm calls)",
            trace_id[:12], trace.id, len(trace_spans), len(llm_calls),
        )

    def shutdown(self) -> None:
        """Called when the TracerProvider shuts down."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush any buffered spans."""
        return True


def install_otel(
    agent_name: Optional[str] = None,
    provider: Optional[Any] = None,
) -> Any:
    """Install DecimalAI as an OTel span exporter.

    Creates a TracerProvider with a DecimalSpanExporter (or adds it to
    an existing provider) and sets it as the global tracer provider.

    Args:
        agent_name: Default agent name for all traces.
        provider: Existing TracerProvider to add the exporter to.
            If None, creates a new one and sets it as global.

    Returns:
        The TracerProvider being used.

    Example::

        from decimalai.integrations.otel import install_otel

        # Standalone
        install_otel(agent_name="my-agent")

        # With existing provider
        from opentelemetry.sdk.trace import TracerProvider
        provider = TracerProvider()
        install_otel(agent_name="my-agent", provider=provider)
    """
    otel_trace, TracerProvider, _, _, _ = _get_otel_sdk()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    exporter = DecimalSpanExporter(agent_name=agent_name)

    if provider is None:
        provider = TracerProvider()
        otel_trace.set_tracer_provider(provider)

    provider.add_span_processor(SimpleSpanProcessor(exporter))

    logger.info(
        "DecimalAI OTel exporter installed (agent_name=%s, provider=%s)",
        agent_name, type(provider).__name__,
    )
    return provider


# ── Helpers ──────────────────────────────────────────────────────


def _extract_content(
    attrs: Dict[str, Any], keys: List[str], max_len: Optional[int] = 500
) -> Optional[str]:
    """Try multiple attribute keys to extract text content from a span.

    Different OTEL instrumentors use different attribute names for
    prompt/completion content. We check all known variants.
    """
    for key in keys:
        val = attrs.get(key)
        if val:
            text = str(val)
            if text and text != "None":
                return text if max_len is None else text[:max_len]
    return None


def _extract_preview(attrs: Dict[str, Any], direction: str) -> Optional[str]:
    """Content for one direction, message-aware.

    OpenInference splits each message across ``…{i}.message.role`` and
    ``…{i}.message.content`` keys, none of which the flat lookup above can
    reach — so the message contents are read first, in index order.
    """
    messages = _messages_from_attrs(attrs, direction)
    if messages:
        joined = "\n".join(m["content"] for m in messages if m["content"])
        if joined:
            return joined[:500]
    keys = _INPUT_ATTR_KEYS if direction == "input" else _OUTPUT_ATTR_KEYS
    return _extract_content(attrs, keys)


def _rendered_input_from_attrs(
    attrs: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """The rendered request for ``LlmCallRecord.rendered_input`` — untruncated."""
    messages = _messages_from_attrs(attrs, "input")
    if messages:
        return messages
    content = _extract_content(attrs, _INPUT_ATTR_KEYS, max_len=None)
    if content is None:
        return None
    return [{"role": "user", "content": content}]


def _output_message_from_attrs(attrs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The response message for ``LlmCallRecord.output`` — untruncated."""
    messages = _messages_from_attrs(attrs, "output")
    if messages:
        return messages[0]
    content = _extract_content(attrs, _OUTPUT_ATTR_KEYS, max_len=None)
    if content is None:
        return None
    return {"role": "assistant", "content": content}


def _span_id_of(span: Any) -> Optional[int]:
    """The span's own 64-bit OTel id, or None if it has no context."""
    ctx = getattr(span, "context", None)
    return getattr(ctx, "span_id", None) if ctx else None


def _parent_span_id_of(span: Any) -> Optional[int]:
    """The id of the span's parent, or None for a root span."""
    parent = getattr(span, "parent", None)
    if not parent:
        return None
    sid = getattr(parent, "span_id", None)
    return sid if sid else None  # 0 is "no parent", same as absent


def _ns_to_datetime(ns: int) -> datetime:
    """Convert nanosecond timestamp to datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
