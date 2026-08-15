"""OpenTelemetry SpanExporter for DecimalAI.

Routes OTEL spans from any OTEL-native framework (CrewAI, AutoGen, Haystack,
Semantic Kernel, Google ADK, etc.) into the DecimalAI backend.

Simple path (global, 3 lines)::

    import decimalai
    decimalai.init()

    from decimalai.otel import instrument
    instrument()  # all OTEL-instrumented calls are now traced

Manual path (custom TracerProvider)::

    from decimalai.otel import DecimalSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(DecimalSpanExporter()))
    trace_api.set_tracer_provider(provider)
"""

from __future__ import annotations

import warnings

import json
import logging
import re
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from .schema.common import FinishReason, SpanType, Status
from .schema.manifest import ManifestTracker, extract_from_config
from .schema.trace import LlmCallRecord, RunTrace, ToolCallRecord, TraceSpan

logger = logging.getLogger("decimalai.otel")

# ── GenAI Semantic Convention attribute keys ──────────────────
# https://opentelemetry.io/docs/specs/semconv/gen-ai/

_GENAI_SYSTEM = "gen_ai.system"
_GENAI_MODEL = "gen_ai.request.model"
_GENAI_TEMPERATURE = "gen_ai.request.temperature"
_GENAI_MAX_TOKENS = "gen_ai.request.max_tokens"
_GENAI_TOP_P = "gen_ai.request.top_p"
_GENAI_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_GENAI_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_GENAI_FINISH_REASON = "gen_ai.response.finish_reasons"

# Fallback attribute keys used by some frameworks. `llm.model_name` is the
# OpenInference convention (Arize Phoenix instrumentations — CrewAI, LlamaIndex,
# etc.); without it those spans carry no model and never become LLM calls.
_ALT_MODEL_KEYS = ("llm.request.model", "llm.model", "llm.model_name", "model")
_ALT_PROVIDER_KEYS = ("llm.system", "llm.provider", "ai.provider")
_ALT_INPUT_TOKEN_KEYS = ("llm.usage.prompt_tokens", "llm.token_count.prompt")
_ALT_OUTPUT_TOKEN_KEYS = (
    "llm.usage.completion_tokens",
    "llm.token_count.completion",
)

# gen_ai.operation.name values that are never themselves an LLM request.
# Agent frameworks (AG2 among them) stamp gen_ai.request.model onto their
# agent/conversation/tool spans as metadata; without this gate each such
# span would become a phantom LlmCallRecord.
_NON_LLM_OPERATIONS = frozenset(
    {"invoke_agent", "create_agent", "conversation", "execute_tool"}
)


def instrument(
    agent_name: Optional[str] = None,
    *,
    service_name: str = "decimal-agent",
    skills: Optional[List[Dict[str, Any]]] = None,
    skill_dirs: Optional[List[str]] = None,
    prompts: Optional[Dict[str, str]] = None,
) -> Any:
    """Install DecimalAI as an OpenTelemetry span exporter.

    Sets up a ``TracerProvider`` with a ``BatchSpanProcessor`` that
    sends completed spans to the DecimalAI backend.  Works with any
    framework that emits OTEL spans (CrewAI, AutoGen, Haystack, etc.).

    Args:
        agent_name: Default agent name. If None, auto-detected from
            the root span's ``service.name`` resource attribute.
        service_name: OTEL service name for the resource.
            Defaults to ``"decimal-agent"``.
        prompts: Optional explicit static prompt templates
            (e.g. ``{"system": "..."}``). When set, these are recorded in the
            manifest instead of the rendered system prompt auto-harvested from
            spans — use this when the rendered prompt carries per-run content
            (RAG chunks, dates) that would otherwise flip the manifest hash.

    Returns:
        The ``TracerProvider`` the exporter was installed on (also set as
        the global tracer provider). Callers that need to activate an
        instrumentor against this exact provider (e.g. the CrewAI / AG2
        activation in :func:`decimalai.init`) should pass it explicitly
        rather than rely on the global — OTEL honors
        ``set_tracer_provider`` only once per process.

    Raises:
        ImportError: If ``opentelemetry-sdk`` is not installed.

    Example::

        import decimalai
        decimalai.init()

        from decimalai.otel import instrument
        instrument()

        # Any OTEL-instrumented code now sends traces to DecimalAI
    """
    try:
        from opentelemetry import trace as trace_api
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        raise ImportError(
            "opentelemetry-sdk is required for instrument() but is missing "
            "(it ships as a core dependency of decimalai). "
            "Reinstall with: pip install decimalai"
        )

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    # Resolve skills (auto-discover or explicit)
    resolved_skills = skills
    if not resolved_skills:
        try:
            from .skills import discover_skills
            resolved_skills = discover_skills(skill_dirs) or None
        except Exception:
            logger.debug("Skill auto-discovery failed", exc_info=True)

    exporter = DecimalSpanExporter(
        agent_name=agent_name, skills=resolved_skills, prompts=prompts
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)

    logger.info(
        "DecimalAI OTEL exporter installed (agent_name=%s, service=%s)",
        agent_name,
        service_name,
    )
    return provider


class DecimalSpanExporter:
    """OpenTelemetry ``SpanExporter`` that routes spans to DecimalAI.

    Implements the OTEL ``SpanExporter`` protocol via duck typing.
    Spans are grouped by ``trace_id``, converted to ``RunTrace`` objects,
    and sent via the ``BackgroundSender``.

    Args:
        agent_name: Default agent name for all traces. If ``None``,
            auto-detected from the root span's service name or
            the first span with an agent-like name.
    """

    def __init__(
        self,
        agent_name: Optional[str] = None,
        skills: Optional[List[Dict[str, Any]]] = None,
        prompts: Optional[Dict[str, str]] = None,
    ):
        self.default_agent_name = agent_name
        self._skills = skills
        # Explicit static prompt templates ({"system": ...}); when set, these
        # win over the rendered system prompt auto-harvested from spans.
        self._explicit_prompts = prompts
        # Manifest tracking state
        self._manifest_tracker = ManifestTracker()
        self._manifest_id: Optional[str] = None
        self._manifest_lock = threading.Lock()
        # Spans of one trace can arrive across multiple export() batches, so
        # buffer them by trace_id and finalize once the root span shows up.
        self._pending: Dict[int, List[Any]] = defaultdict(list)
        self._pending_lock = threading.Lock()

    def export(self, spans: Sequence[Any]) -> Any:
        """Export a batch of OTEL spans, grouped into DecimalAI traces.

        Args:
            spans: Sequence of ``ReadableSpan`` objects from the OTEL SDK.

        Returns:
            ``SpanExportResult.SUCCESS`` on success.
        """
        try:
            from opentelemetry.sdk.trace.export import SpanExportResult
        except ImportError:
            # If somehow called without OTEL installed, just succeed silently
            return None

        if not spans:
            return SpanExportResult.SUCCESS

        # A trace's spans can arrive across several export() calls — the
        # BatchSpanProcessor flushes on a timer (5s by default), so any agent
        # run longer than that delay is delivered in pieces. Buffer spans by
        # trace_id and only finalize a trace once its root span (parent is
        # None) arrives. The root always ends last, so by the time it shows up
        # every child has been buffered. Without this, a single agent run
        # fragments into one DecimalAI trace per batch.
        ready: List[int] = []
        with self._pending_lock:
            for span in spans:
                tid = _get_trace_id(span)
                self._pending[tid].append(span)
                if _get_parent_span_id(span) is None:
                    ready.append(tid)
            groups = [
                (tid, self._pending.pop(tid))
                for tid in ready
                if tid in self._pending
            ]
            # A trace whose ROOT span never reaches this
            # exporter (root owned by another tracer, sampled out, or the process
            # killed mid-run) would buffer its children in _pending forever — an
            # unbounded memory leak in a long-lived agent host. Cap the buffer and
            # drop the oldest traces (FIFO; defaultdict preserves insertion order).
            _MAX_PENDING_TRACES = 1000
            while len(self._pending) > _MAX_PENDING_TRACES:
                self._pending.pop(next(iter(self._pending)), None)

        for tid, group in groups:
            self._finalize_trace(tid, group)

        return SpanExportResult.SUCCESS

    def _finalize_trace(self, tid: int, group: List[Any]) -> None:
        """Assemble one trace_id's buffered spans into a RunTrace and send it."""
        try:
            result = self._assemble_trace(group)
            if result is not None:
                run_trace, seen_model, seen_tools, seen_prompts = result
                agent_name = run_trace.agent_name or "otel-agent"
                self._maybe_register_manifest(
                    agent_name, seen_model, seen_tools, seen_prompts
                )
                # Stamp the just-registered manifest onto the trace. The trace
                # was assembled before registration ran, so its manifest_id is
                # still stale (None on the first export) — without this the
                # backend rejects it under require_manifest_on_ingest.
                run_trace.manifest_id = self._manifest_id
                self._send(run_trace)
        except Exception:
            logger.exception(
                "Failed to assemble trace from %d spans (trace_id=%s)",
                len(group),
                hex(tid),
            )

    def _flush_pending(self) -> None:
        """Finalize every buffered trace, whether or not its root arrived.

        The last-chance path for force_flush/shutdown: emits traces whose root
        span never showed up (malformed trace, or the process exiting mid-run).
        Finalized traces are popped from the buffer, so this is idempotent.
        """
        with self._pending_lock:
            groups = list(self._pending.items())
            self._pending.clear()
        for tid, group in groups:
            self._finalize_trace(tid, group)

    def shutdown(self) -> None:
        """Flush any spans still buffered, then clean up."""
        self._flush_pending()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Flush buffered traces out to the backend."""
        self._flush_pending()
        return True

    # ── Trace assembly ─────────────────────────────────────

    def _assemble_trace(self, otel_spans: List[Any]) -> Optional[tuple]:
        """Convert a group of OTEL spans (same trace_id) to a RunTrace.

        Returns:
            Tuple of (RunTrace, seen_model, seen_tools, seen_prompts) or None.
        """
        from . import _config

        if not _config._is_enabled():
            return None

        config = _config._config
        trace_spans: List[TraceSpan] = []
        llm_calls: List[LlmCallRecord] = []
        agent_name = self.default_agent_name
        root_started: Optional[datetime] = None
        root_ended: Optional[datetime] = None
        user_input: Optional[str] = None
        final_output: Optional[str] = None
        # Whether the root span supplied the trace-level previews; if so its
        # values win over the per-LLM-span fallbacks below.
        root_set_input = False
        root_set_output = False
        trace_status = Status.SUCCESS
        # Manifest auto-detection accumulators
        seen_model: Optional[Dict[str, Any]] = None
        seen_tools: Dict[str, Dict[str, Any]] = {}
        seen_prompts: Dict[str, str] = {}
        # Skill tracking — merge from OTEL span attributes + auto-detection
        active_skills: Dict[str, Optional[str]] = {}

        for otel_span in otel_spans:
            attrs = _get_attributes(otel_span)
            name = _get_span_name(otel_span)
            parent_id = _get_parent_span_id(otel_span)
            span_id_str = _get_span_id(otel_span)
            started_at = _ns_to_datetime(getattr(otel_span, "start_time", None))
            ended_at = _ns_to_datetime(getattr(otel_span, "end_time", None))

            # Track root span timing
            if parent_id is None:
                root_started = started_at
                root_ended = ended_at
                # Auto-detect agent name from root span
                if not agent_name:
                    agent_name = _extract_service_name(otel_span) or name
                # Prefer the root span's own input/output for the trace-level
                # previews — the root span is the agent's overall turn.
                root_input = _preview_from_attrs(attrs, "input")
                if root_input is not None:
                    user_input = root_input
                    root_set_input = True
                root_output = _preview_from_attrs(attrs, "output")
                if root_output is not None:
                    final_output = root_output
                    root_set_output = True

            # Check OTEL status
            otel_status = getattr(otel_span, "status", None)
            if otel_status and hasattr(otel_status, "status_code"):
                status_code_name = str(otel_status.status_code)
                if "ERROR" in status_code_name:
                    trace_status = Status.ERROR

            # Preserve active_skills from external OTEL span attributes
            span_skills = attrs.get("decimal.active_skills") or attrs.get("active_skills")
            if span_skills and isinstance(span_skills, (list, tuple)):
                for entry in span_skills:
                    if isinstance(entry, str) and entry not in active_skills:
                        active_skills[entry] = None
                    elif isinstance(entry, dict):
                        sname = entry.get("name", "")
                        if sname and sname not in active_skills:
                            active_skills[sname] = entry.get("hash")

            # Determine if this is an LLM span. An explicit non-LLM
            # gen_ai.operation.name wins over the model attribute — see
            # _NON_LLM_OPERATIONS.
            model = _get_first(attrs, _GENAI_MODEL, *_ALT_MODEL_KEYS)
            operation = str(attrs.get("gen_ai.operation.name") or "").lower()
            if operation in _NON_LLM_OPERATIONS:
                model = None

            if model:
                # This is an LLM call — create LlmCallRecord
                llm_call = self._make_llm_call(
                    attrs, name, model, started_at, ended_at, otel_span
                )
                llm_calls.append(llm_call)

                # Harvest tools for manifest auto-detection. Frameworks that
                # inline tool calls in the LLM span (OpenInference — CrewAI,
                # LlamaIndex, …) emit no dedicated tool spans, so the declared
                # tool set and the tools actually invoked both live here.
                for tname in _extract_declared_tools(attrs):
                    seen_tools.setdefault(tname, {"name": tname})
                for tc in llm_call.tool_calls:
                    seen_tools.setdefault(tc.tool_name, {"name": tc.tool_name})

                # Accumulate model for manifest auto-detection
                if seen_model is None:
                    provider = _get_first(attrs, _GENAI_SYSTEM, *_ALT_PROVIDER_KEYS)
                    if not provider:
                        provider = _infer_provider(model)
                    seen_model = {
                        "provider": provider,
                        "model": model,
                        "temperature": _get_float(attrs, _GENAI_TEMPERATURE),
                        "max_tokens": _get_int(attrs, _GENAI_MAX_TOKENS),
                    }

                # Harvest the system prompt for the manifest.
                # Capture only the FIRST system prompt per trace: it's the
                # RENDERED prompt, so re-capturing a later (dynamically-built)
                # one would flip the manifest hash mid-trace. Pass an explicit
                # static template via install(prompts=...) to override.
                if "system" not in seen_prompts:
                    sys_prompt = _extract_system_prompt(attrs)
                    if sys_prompt:
                        seen_prompts["system"] = sys_prompt

                # Also create a wrapper TraceSpan
                trace_span = TraceSpan(
                    id=uuid4(),
                    parent_span_id=None,
                    span_type=SpanType.LLM,
                    name=f"llm:{model}",
                    status=llm_call.status,
                    started_at=started_at,
                    ended_at=ended_at,
                    input_preview=_preview_from_attrs(attrs, "input"),
                    output_preview=_preview_from_attrs(attrs, "output"),
                )
                trace_spans.append(trace_span)

                # Fallback trace-level previews when the root span carried
                # none: first LLM call's input, last LLM call's output. Root
                # values (when present) take precedence.
                if not root_set_input and user_input is None:
                    user_input = trace_span.input_preview
                if not root_set_output:
                    llm_output = trace_span.output_preview
                    if llm_output is not None:
                        final_output = llm_output
            else:
                # Non-LLM span — classify by name/kind
                span_type = _classify_span(name, attrs)
                span_status = Status.SUCCESS
                if otel_status and hasattr(otel_status, "status_code"):
                    if "ERROR" in str(otel_status.status_code):
                        span_status = Status.ERROR

                # Accumulate tools for manifest auto-detection
                if span_type == SpanType.TOOL and name not in seen_tools:
                    seen_tools[name] = {"name": name}

                trace_span = TraceSpan(
                    id=uuid4(),
                    parent_span_id=None,
                    span_type=span_type,
                    name=name,
                    status=span_status,
                    started_at=started_at,
                    ended_at=ended_at,
                    input_preview=_preview_from_attrs(attrs, "input"),
                    output_preview=_preview_from_attrs(attrs, "output"),
                )
                trace_spans.append(trace_span)

        if not trace_spans and not llm_calls:
            return None

        # Auto-detect skills from SDK registry (if installed)
        if self._skills and llm_calls:
            try:
                from .skills import detect_skill_activations
                for call in llm_calls:
                    if not call.rendered_input:
                        continue
                    detected = detect_skill_activations(
                        call.rendered_input, self._skills
                    )
                    for skill_name in detected:
                        if skill_name not in active_skills:
                            registry_hash = next(
                                (s.get("hash") for s in self._skills
                                 if s.get("name") == skill_name),
                                None,
                            )
                            active_skills[skill_name] = registry_hash
            except Exception:
                logger.debug("Skill auto-detection failed", exc_info=True)

        # Build active_skills list
        active_skills_list: List[Dict[str, Any]] = []
        for sname, shash in active_skills.items():
            entry: Dict[str, Any] = {"name": sname}
            if shash:
                entry["hash"] = shash
            active_skills_list.append(entry)

        now = datetime.now(timezone.utc)

        return RunTrace(
            id=uuid4(),
            project=config.project if config else None,
            agent_name=agent_name or "otel-agent",
            status=trace_status,
            source_type="production",
            started_at=root_started or now,
            ended_at=root_ended or now,
            user_input_preview=user_input,
            final_output_preview=final_output,
            spans=trace_spans,
            llm_calls=llm_calls,
            active_skills=active_skills_list,
            manifest_id=self._manifest_id,
        ), seen_model, seen_tools, seen_prompts

    def _make_llm_call(
        self,
        attrs: Dict[str, Any],
        name: str,
        model: str,
        started_at: Optional[datetime],
        ended_at: Optional[datetime],
        otel_span: Any,
    ) -> LlmCallRecord:
        """Build an LlmCallRecord from OTEL span attributes."""
        provider = _get_first(attrs, _GENAI_SYSTEM, *_ALT_PROVIDER_KEYS)
        if not provider:
            provider = _infer_provider(model)

        temperature = _get_float(attrs, _GENAI_TEMPERATURE)
        max_tokens = _get_int(attrs, _GENAI_MAX_TOKENS)
        input_tokens = _get_int(
            attrs, _GENAI_INPUT_TOKENS, *_ALT_INPUT_TOKEN_KEYS
        )
        output_tokens = _get_int(
            attrs, _GENAI_OUTPUT_TOKENS, *_ALT_OUTPUT_TOKEN_KEYS
        )

        latency_ms = None
        if started_at and ended_at:
            latency_ms = int((ended_at - started_at).total_seconds() * 1000)

        # Check for errors
        otel_status = getattr(otel_span, "status", None)
        status = Status.SUCCESS
        finish_reason = FinishReason.STOP
        if otel_status and hasattr(otel_status, "status_code"):
            if "ERROR" in str(otel_status.status_code):
                status = Status.ERROR
                finish_reason = FinishReason.ERROR

        # Try to extract finish reason from attributes
        finish_reasons = attrs.get(_GENAI_FINISH_REASON)
        if finish_reasons:
            if isinstance(finish_reasons, (list, tuple)) and finish_reasons:
                fr_str = str(finish_reasons[0]).lower()
            else:
                fr_str = str(finish_reasons).lower()
            if "stop" in fr_str:
                finish_reason = FinishReason.STOP
            elif "length" in fr_str or "max" in fr_str:
                finish_reason = FinishReason.LENGTH
            elif "tool" in fr_str or "function" in fr_str:
                finish_reason = FinishReason.TOOL_CALLS

        return LlmCallRecord(
            id=uuid4(),
            span_id=None,
            agent_name=self.default_agent_name,
            provider=provider,
            model_name=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
            # The FULL rendered request/response, not the 200-char previews —
            # these are what SFT derivation reads.
            rendered_input=_rendered_input_from_attrs(attrs),
            output=_output_message_from_attrs(attrs),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            tool_calls=_extract_openinference_tool_calls(attrs),
        )

    def _send(self, trace: RunTrace) -> None:
        """Send a trace via the background sender."""
        from . import _config

        if not _config._is_enabled():
            return

        try:
            client = _config._get_client()
            _config._sender.submit(client.ingest_trace, trace)
            logger.debug(
                "Queued OTEL trace %s (%d spans, %d llm_calls, manifest=%s) for agent %s",
                trace.id,
                len(trace.spans),
                len(trace.llm_calls),
                trace.manifest_id or "none",
                trace.agent_name,
            )
        except Exception:
            logger.exception("Failed to queue OTEL trace %s", trace.id)

    def _maybe_register_manifest(
        self,
        agent_name: str,
        seen_model: Optional[Dict[str, Any]],
        seen_tools: Dict[str, Dict[str, Any]],
        seen_prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        """Register manifest from accumulated OTel span data.

        Thread-safe via _manifest_lock. Only registers if the
        manifest hash has changed since last registration.
        """
        from . import _config

        if not _config._is_enabled():
            return

        tools = list(seen_tools.values()) if seen_tools else None
        models = {"default": seen_model} if seen_model else None
        # An explicit static template (install(prompts=...)) wins over the
        # auto-harvested rendered prompt — see the rendered-vs-template note in
        # _assemble_trace.
        prompts = self._explicit_prompts or (seen_prompts or None)

        if not tools and not models and not self._skills and not prompts:
            return

        snapshot = extract_from_config(
            agent_name=agent_name,
            tools=tools,
            models=models,
            prompts=prompts,
            skills=self._skills,
        )

        with self._manifest_lock:
            if not self._manifest_tracker.check_and_update(snapshot):
                return  # Same hash — already registered

            try:
                client = _config._get_client()
                result = client.register_manifest(snapshot)
                self._manifest_id = result.get("manifest_id", snapshot.id)
                logger.info(
                    "Registered manifest %s from OTel spans (hash=%s, components=%d)",
                    self._manifest_id,
                    snapshot.manifest_hash[:12],
                    len(snapshot.components),
                )
            except Exception:
                logger.warning("Failed to register manifest from OTel spans", exc_info=True)
                self._manifest_id = snapshot.id


# ── Utilities ──────────────────────────────────────────────


def _get_trace_id(span: Any) -> int:
    """Extract the trace_id from an OTEL span as int."""
    ctx = getattr(span, "context", None)
    if ctx:
        return getattr(ctx, "trace_id", 0)
    return 0


def _get_span_id(span: Any) -> Optional[str]:
    """Extract span_id as a hex string."""
    ctx = getattr(span, "context", None)
    if ctx:
        sid = getattr(ctx, "span_id", None)
        if sid is not None:
            return format(sid, "016x")
    return None


def _get_parent_span_id(span: Any) -> Optional[str]:
    """Extract parent span_id as a hex string, or None for root."""
    parent = getattr(span, "parent", None)
    if parent:
        sid = getattr(parent, "span_id", None)
        if sid is not None and sid != 0:
            return format(sid, "016x")
    return None


def _get_span_name(span: Any) -> str:
    """Extract the span name."""
    return getattr(span, "name", "unknown") or "unknown"


def _get_attributes(span: Any) -> Dict[str, Any]:
    """Extract attributes dict from an OTEL span."""
    attrs = getattr(span, "attributes", None)
    if attrs is None:
        return {}
    if isinstance(attrs, dict):
        return attrs
    # BoundedAttributes or similar — convert to dict
    try:
        return dict(attrs)
    except Exception:
        return {}


def _extract_service_name(span: Any) -> Optional[str]:
    """Extract service.name from the span's resource."""
    resource = getattr(span, "resource", None)
    if resource:
        res_attrs = getattr(resource, "attributes", {})
        if isinstance(res_attrs, dict):
            return res_attrs.get("service.name")
        try:
            return dict(res_attrs).get("service.name")
        except Exception:
            pass
    return None


def _ns_to_datetime(ns: Optional[int]) -> Optional[datetime]:
    """Convert nanosecond timestamp to datetime."""
    if ns is None or ns == 0:
        return None
    try:
        return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    except (ValueError, OverflowError, OSError):
        return None


def _get_first(attrs: Dict[str, Any], *keys: str) -> Optional[str]:
    """Return the first non-None value for the given keys."""
    for key in keys:
        val = attrs.get(key)
        if val is not None:
            return str(val)
    return None


def _get_int(attrs: Dict[str, Any], *keys: str) -> Optional[int]:
    """Return the first non-None integer value for the given keys."""
    for key in keys:
        val = attrs.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return None


def _get_float(attrs: Dict[str, Any], *keys: str) -> Optional[float]:
    """Return the first non-None float value for the given keys."""
    for key in keys:
        val = attrs.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return None


# OpenInference inlines tool calls inside LLM message attributes rather than
# emitting dedicated tool spans, so they must be harvested from the LLM span.
_OI_TOOLCALL_NAME_RE = re.compile(
    r"^llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.function\.name$"
)
_OI_TOOLCALL_ARGS_RE = re.compile(
    r"^llm\.output_messages\.(\d+)\.message\.tool_calls\.(\d+)\.tool_call\.function\.arguments$"
)


def _extract_openinference_tool_calls(attrs: Dict[str, Any]) -> List[ToolCallRecord]:
    """Pull tool calls the model made in this step from OpenInference
    ``llm.output_messages.*.message.tool_calls.*`` attributes."""
    found: Dict[tuple, Dict[str, Any]] = {}
    for key, val in attrs.items():
        m = _OI_TOOLCALL_NAME_RE.match(key)
        if m:
            found.setdefault(m.groups(), {})["name"] = str(val)
            continue
        m = _OI_TOOLCALL_ARGS_RE.match(key)
        if m:
            found.setdefault(m.groups(), {})["args"] = val
    records: List[ToolCallRecord] = []
    for _, info in sorted(found.items()):
        name = info.get("name")
        if not name:
            continue
        args = info.get("args")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {"raw": args}
        if not isinstance(args, dict):
            args = {}
        records.append(ToolCallRecord(tool_name=name, args=args))
    return records


def _extract_declared_tools(attrs: Dict[str, Any]) -> List[str]:
    """Pull declared tool names from OpenInference ``llm.tools.*.tool.json_schema``
    attributes — the tool set the agent was given, i.e. the manifest's tools."""
    names: List[str] = []
    for key, val in attrs.items():
        if not (key.startswith("llm.tools.") and key.endswith(".tool.json_schema")):
            continue
        schema = val
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except (ValueError, TypeError):
                continue
        if isinstance(schema, dict):
            tname = schema.get("name")
            if tname:
                names.append(str(tname))
    return names


# OpenInference carries the chat messages on the LLM span as
# llm.{input,output}_messages.{i}.message.role / .content — the same attribute
# namespace the inline tool calls above use (CrewAI, LlamaIndex/Phoenix, …).
# The GenAI semconv has an indexed spelling of its own
# (gen_ai.prompt.{i}.role / .content, gen_ai.completion.{i}...).
_MSG_KEY_RES = {
    "input": (
        re.compile(r"^llm\.input_messages\.(\d+)\.message\.(role|content)$"),
        re.compile(r"^gen_ai\.prompt\.(\d+)\.(role|content)$"),
    ),
    "output": (
        re.compile(r"^llm\.output_messages\.(\d+)\.message\.(role|content)$"),
        re.compile(r"^gen_ai\.completion\.(\d+)\.(role|content)$"),
    ),
}

# A multi-part (multi-modal) OpenInference message puts its text under
# …{i}.message.contents.{j}.message_content.text instead of …{i}.message.content.
_MSG_PART_RES = {
    "input": re.compile(
        r"^llm\.input_messages\.(\d+)\.message\.contents\.(\d+)\.message_content\.text$"
    ),
    "output": re.compile(
        r"^llm\.output_messages\.(\d+)\.message\.contents\.(\d+)\.message_content\.text$"
    ),
}

# Every key in the indexed-message namespaces above, content and metadata
# alike. _content_from_attrs skips them: role and tool_call keys are metadata,
# and a substring scan would otherwise return the ROLE ("system"/"assistant")
# as the preview — the whole namespace belongs to _messages_from_attrs.
_INDEXED_MSG_NAMESPACE_RE = re.compile(
    r"^(llm\.(input|output)_messages|gen_ai\.(prompt|completion))\.\d+\."
)

_DEFAULT_ROLE = {"input": "user", "output": "assistant"}


def _messages_from_attrs(
    attrs: Dict[str, Any], direction: str
) -> Optional[List[Dict[str, Any]]]:
    """Rebuild one direction's chat messages from indexed span attributes.

    Frameworks that follow OpenInference (CrewAI, LlamaIndex/Phoenix, …) split
    each message across ``…{i}.message.role`` and ``…{i}.message.content``
    keys. Returns them in index order as ``{"role", "content"}`` dicts — the
    shape the other adapters normalize to — or None when the span carries no
    indexed messages.

    A message whose content is absent (an assistant turn that only made tool
    calls) is kept with an empty content so turn order survives; its tool calls
    are carried separately on ``LlmCallRecord.tool_calls``.
    """
    found: Dict[int, Dict[str, str]] = {}
    parts: Dict[int, Dict[int, str]] = {}
    for key, val in attrs.items():
        part = _MSG_PART_RES[direction].match(key)
        if part:
            parts.setdefault(int(part.group(1)), {})[int(part.group(2))] = str(val)
            continue
        for pattern in _MSG_KEY_RES[direction]:
            m = pattern.match(key)
            if m:
                found.setdefault(int(m.group(1)), {})[m.group(2)] = str(val)
                break
    for idx, by_part in parts.items():
        found.setdefault(idx, {}).setdefault(
            "content", "".join(by_part[j] for j in sorted(by_part))
        )
    if not found:
        return None
    return [
        {
            "role": found[idx].get("role") or _DEFAULT_ROLE[direction],
            "content": found[idx].get("content", ""),
        }
        for idx in sorted(found)
    ]


def _rendered_input_from_attrs(
    attrs: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """The rendered request for ``LlmCallRecord.rendered_input``.

    Indexed messages when the span carries them; otherwise a single user
    message wrapping whatever prompt content is there (a bare
    ``gen_ai.prompt``/``input.value`` string) — the same fallback the other
    adapters apply to non-message input.
    """
    messages = _messages_from_attrs(attrs, "input")
    if messages:
        return messages
    content = _content_from_attrs(attrs, "input")
    if content is None:
        return None
    return [{"role": "user", "content": content}]


def _output_message_from_attrs(attrs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The response message for ``LlmCallRecord.output``."""
    messages = _messages_from_attrs(attrs, "output")
    if messages:
        return messages[0]
    content = _content_from_attrs(attrs, "output")
    if content is None:
        return None
    return {"role": "assistant", "content": content}


def _extract_system_prompt(attrs: Dict[str, Any]) -> Optional[str]:
    """Pull the system/developer prompt from an LLM span's attributes.

    Prefers the OpenInference role-indexed input messages; falls back to the
    GenAI-semconv ``gen_ai.system_instructions`` / ``gen_ai.prompt`` keys.
    Returns the RENDERED prompt (callers should capture only the first per
    trace — see _assemble_trace).
    """
    for message in _messages_from_attrs(attrs, "input") or ():
        if message["role"].lower() in ("system", "developer") and message["content"]:
            return message["content"]
    # GenAI-semconv fallback.
    for key in ("gen_ai.system_instructions", "gen_ai.prompt"):
        val = attrs.get(key)
        if val:
            return str(val)
    return None


def _classify_span(name: str, attrs: Dict[str, Any]) -> SpanType:
    """Classify an OTEL span into a DecimalAI SpanType."""
    # Honor an explicit span-kind attribute when a framework supplies one
    # (OpenInference's `openinference.span.kind`, GenAI semconv's
    # `gen_ai.operation.name`) — more reliable than guessing from the name.
    explicit = str(
        attrs.get("openinference.span.kind")
        or attrs.get("gen_ai.operation.name")
        or ""
    ).lower()
    if explicit:
        if "tool" in explicit or "function" in explicit:
            return SpanType.TOOL
        if "agent" in explicit:
            return SpanType.AGENT
        if "retriev" in explicit or "rerank" in explicit:
            return SpanType.RETRIEVAL
        if "chain" in explicit:
            return SpanType.OTHER
        if "llm" in explicit or "chat" in explicit:
            return SpanType.LLM

    name_lower = name.lower()
    if "tool" in name_lower or "function" in name_lower:
        return SpanType.TOOL
    if "agent" in name_lower or "crew" in name_lower:
        return SpanType.AGENT
    if "retriev" in name_lower or "search" in name_lower or "rag" in name_lower:
        return SpanType.RETRIEVAL
    if "chain" in name_lower or "pipeline" in name_lower or "task" in name_lower:
        return SpanType.OTHER
    if "llm" in name_lower or "chat" in name_lower or "generat" in name_lower:
        return SpanType.LLM
    return SpanType.OTHER


def _content_from_attrs(attrs: Dict[str, Any], direction: str) -> Optional[str]:
    """Extract one direction's content from unstructured span attributes."""
    # Patterns are direction-specific: ``gen_ai.prompt`` is an input-side key
    # and ``gen_ai.completion`` output-side, so neither may serve the other
    # direction (an output preview must never surface the prompt).
    if direction == "input":
        key_patterns = (
            "gen_ai.input", "llm.input", "input", "gen_ai.prompt",
            # AG2 tool spans; the key contains neither "input" nor "prompt".
            "gen_ai.tool.call.arguments",
        )
    else:
        key_patterns = (
            "gen_ai.output", "llm.output", "output", "gen_ai.completion",
            "gen_ai.tool.call.result",
        )
    for key_pattern in key_patterns:
        for key, val in attrs.items():
            key_lower = key.lower()
            # Token counts carry the direction as a substring too
            # (gen_ai.usage.input_tokens, llm.usage.completion_tokens) —
            # they are counts, not content, and must never become previews.
            if "token" in key_lower or "usage" in key_lower:
                continue
            if _INDEXED_MSG_NAMESPACE_RE.match(key_lower):
                continue
            if key_pattern in key_lower:
                return str(val)
    return None


def _preview_from_attrs(
    attrs: Dict[str, Any], direction: str, max_len: int = 200
) -> Optional[str]:
    """Extract a preview string from span attributes."""
    messages = _messages_from_attrs(attrs, direction)
    if messages:
        joined = "\n".join(m["content"] for m in messages if m["content"])
        if joined:
            return joined[:max_len]
    content = _content_from_attrs(attrs, direction)
    return content[:max_len] if content is not None else None


def _infer_provider(model: Optional[str]) -> Optional[str]:
    """Infer provider from model name."""
    if not model:
        return None
    m = model.lower()
    if "gpt" in m or "o1" in m or "o3" in m or "davinci" in m:
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "google"
    if "mistral" in m or "mixtral" in m:
        return "mistral"
    if "llama" in m:
        return "meta"
    if "command" in m or "coral" in m:
        return "cohere"
    return None


# ── Deprecated: install() ────────────────────────────────────────────────────
#
# Renamed to `instrument()` 2026-08-11. "install" was doing double duty across
# this SDK: here it turned on TRACING for a framework, while
# `SkillRouter.install()` added a SKILL to a workspace. Two unrelated actions
# under one word, in one package — and the skill sense is the one users arrive
# with, because it is what every extension marketplace means by install.
#
# Behaviour is unchanged and this alias is not going away soon; it warns so the
# docs and the code agree on one name.
def install(*args, **kwargs):  # pragma: no cover - thin deprecation shim
    warnings.warn(
        "decimalai.otel.install() is deprecated; use "
        "decimalai.otel.instrument() instead. It turns on tracing for otel "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
