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
]
_OUTPUT_ATTR_KEYS = [
    "gen_ai.content.completion",   # GenAI semconv
    "gen_ai.completion",           # OpenLLMetry / Traceloop
    "output",                      # OpenInference
    "output.value",                # OpenInference v2
    "llm.output_messages",         # LangSmith-style
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
        from .. import _config
        from ..schema.common import FinishReason, SpanType, Status
        from ..schema.trace import LlmCallRecord, RunTrace, TraceSpan

        llm_calls: List[LlmCallRecord] = []
        trace_spans: List[TraceSpan] = []
        agent_name = self.agent_name
        earliest_start = None
        latest_end = None
        user_input = None
        final_output = None

        for span in spans:
            attrs = dict(span.attributes or {})

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
                    model_name=str(model_name) if model_name else None,
                    provider=str(attrs.get(OTEL_GEN_AI_SYSTEM, "")),
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

                # Try to extract user input from this LLM span
                if not user_input:
                    user_input = _extract_content(attrs, _INPUT_ATTR_KEYS)
                # Always update final_output (last LLM span wins)
                output_content = _extract_content(attrs, _OUTPUT_ATTR_KEYS)
                if output_content:
                    final_output = output_content

            elif is_tool_span:
                # ── Tool execution span (CrewAI / AutoGen) ──
                resolved_name = str(tool_name or span.name or "unknown_tool")
                tool_input = _extract_content(attrs, _INPUT_ATTR_KEYS) or ""
                tool_output = _extract_content(attrs, _OUTPUT_ATTR_KEYS) or ""

                trace_spans.append(TraceSpan(
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

                span_input = _extract_content(attrs, _INPUT_ATTR_KEYS)
                span_output = _extract_content(attrs, _OUTPUT_ATTR_KEYS)

                trace_spans.append(TraceSpan(
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

        # Assemble RunTrace
        trace = RunTrace(
            id=uuid4(),
            agent_name=agent_name or "unknown",
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

        # Send via background sender
        _config._sender.submit(client.ingest_trace, trace)
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


def _extract_content(attrs: Dict[str, Any], keys: List[str]) -> Optional[str]:
    """Try multiple attribute keys to extract text content from a span.

    Different OTEL instrumentors use different attribute names for
    prompt/completion content. We check all known variants.
    """
    for key in keys:
        val = attrs.get(key)
        if val:
            text = str(val)
            if text and text != "None":
                return text[:500]
    return None


def _ns_to_datetime(ns: int) -> datetime:
    """Convert nanosecond timestamp to datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
