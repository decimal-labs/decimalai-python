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

import contextlib
import contextvars
import json
import logging
import re
import threading
import warnings
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence
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
# `gen_ai.provider.name` is what current GenAI semconv calls the provider
# (`gen_ai.system` is the older spelling); AG2 emits only the new one, so
# without it every AG2 span fell through to guessing from the model id.
_ALT_PROVIDER_KEYS = (
    "gen_ai.provider.name", "llm.system", "llm.provider", "ai.provider",
)
_ALT_INPUT_TOKEN_KEYS = ("llm.usage.prompt_tokens", "llm.token_count.prompt")
# OpenInference reports the finish reason under its OWN key, not the OTel
# semantic-convention one. Reading only `gen_ai.response.finish_reasons` meant
# every OpenInference-instrumented call (CrewAI, raw OpenAI, raw Anthropic) fell
# through to the STOP default — so a span that plainly said
# `llm.finish_reason = tool_calls` produced a record claiming the model stopped
# normally. A default that contradicts its own source is worse than an absent
# value: downstream it is indistinguishable from a real observation.
#: Key suffixes that describe content rather than being it. Checked as an
#: endswith() so `input.value` and `gen_ai.tool.call.arguments` still match.
_NON_CONTENT_KEY_SUFFIXES = (
    ".mime_type", ".type", ".encoding", ".format", ".content_type",
)

_ALT_FINISH_REASON_KEYS = (
    "llm.finish_reason",
    "llm.response.finish_reason",
    "finish_reason",
)
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

# Version label for the manifest a run registers when it observed no model, no
# tool and no prompt. See _ManifestRegistry for why it is labelled rather than
# left to the backend's v1/v2/v3 auto-increment.
_UNDECLARED_LABEL = "undeclared"

# The span attribute that names the agent a span belongs to. Callers may set it
# themselves on any span; :class:`_AgentNameStamper` sets it automatically from
# the agent active in the current context. The older
# ``decimalai.integrations.otel`` exporter has always honoured this key — this
# one did not, so a user who followed that documented escape hatch got their
# name silently dropped.
_DECIMAL_AGENT_NAME = "decimal.agent_name"

# The agent whose run is executing in THIS context (thread, or asyncio task
# descended from it). A ContextVar rather than a module global on purpose: eight
# concurrent runs of eight different agents in one process each need their own
# answer, and a plain global would hand all eight whichever name was written
# last. Unset outside any instrumented context, in which case the exporter falls
# back to the name it was constructed with — the old behaviour, unchanged for
# the single-agent process that is the common case.
_active_agent_name: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar(
    "decimalai_otel_active_agent", default=None
)


# ── Skill rail ────────────────────────────────────────────────
# What the router decided for a run, held until that run's trace is assembled.
#
# Keyed by the run's OTel **trace_id**, not by a ContextVar and not by the
# exporter instance. Both alternatives were rejected for reasons that have
# already bitten this codebase:
#
#   * a ContextVar cannot be read at export time — under BatchSpanProcessor the
#     trace is assembled on a worker thread that never saw the caller's context,
#     so the read returns None and the rail silently empties;
#   * the router's own clear-on-read rails (`consume_offered_names()` et al) are
#     process-global, so under eight concurrent lanes the first trace to send
#     takes everyone's names and lanes two through eight get `[]`.
#
# The trace_id is the one identifier that is both readable on the calling thread
# at routing time and carried on the span at export time, so it is the join key.
#
# Module level, not an exporter attribute: a process can hold two
# DecimalSpanExporters (a `providers` pipeline plus `otel.instrument()`), and an
# instance-owned store would strand every entry written while the other one was
# the live exporter.
_SKILL_RAIL_MAX = 256
_skill_rails: "OrderedDict[int, Dict[str, Any]]" = OrderedDict()
_skill_rails_lock = threading.Lock()


def current_run_key() -> Optional[int]:
    """The live OTel trace_id, or None when there is no run to attribute to.

    None is a real answer, not a failure: it means nothing has declared a run
    boundary here (no ``agent_run``, tracing off, an invalid span context). The
    caller must then DROP its metadata rather than guess an owner — see
    :func:`record_skill_rail`.
    """
    try:
        from opentelemetry import trace as trace_api
    except ImportError:
        return None
    try:
        span_ctx = trace_api.get_current_span().get_span_context()
    except Exception:
        return None
    if not getattr(span_ctx, "is_valid", False):
        return None
    tid = getattr(span_ctx, "trace_id", 0)
    return tid or None


def record_skill_rail(
    *,
    routing_id: Optional[str] = None,
    offered: Optional[Sequence[str]] = None,
    delivered: Optional[Sequence[str]] = None,
    loaded: Optional[Sequence[str]] = None,
    prompt_text: Optional[str] = None,
) -> bool:
    """Attribute one routing/loading fact to the run whose trace is live NOW.

    Args:
        routing_id: the ``rt_…`` id the platform minted for this decision.
            FIRST write wins for a run — the first decision is the one the run
            is attributed to, and a later turn must not overwrite it.
        offered: names whose menu row was put in the prompt.
        delivered: names whose BODY was put in the prompt.
        loaded: names whose body reached the model as a tool result.
        prompt_text: the text that was actually injected. When given, ``offered``
            and ``delivered`` are filtered down to names that genuinely appear
            in it — the router derives its offered list and its prompt fragment
            from two different keys of one payload and nothing cross-checks
            them, so recording the claim unfiltered would import that gap into
            the trace. A dropped name is logged at WARNING, by name.

    Returns:
        True when the fact was attributed to a run, False when there was no run
        to attribute it to and it was therefore dropped. A caller that gets
        False must not retry against some other slot.
    """
    key = current_run_key()
    if key is None:
        return False

    def _kept(names: Optional[Sequence[str]], kind: str) -> List[str]:
        vals = [n for n in (names or []) if isinstance(n, str) and n]
        if prompt_text is None:
            return vals
        keep = []
        for n in vals:
            if n in prompt_text:
                keep.append(n)
            else:
                logger.warning(
                    "skill rail: the router %s %r but the rendered fragment does "
                    "not contain it — dropping it rather than claiming the model "
                    "was shown it",
                    kind, n,
                )
        return keep

    kept_offered = _kept(offered, "offered")
    kept_delivered = _kept(delivered, "delivered")
    kept_loaded = [n for n in (loaded or []) if isinstance(n, str) and n]

    with _skill_rails_lock:
        rail = _skill_rails.get(key)
        if rail is None:
            rail = {
                "routing_id": None,
                "offered": [],
                "delivered": [],
                "loaded": [],
            }
            _skill_rails[key] = rail
        if routing_id and not rail["routing_id"]:
            rail["routing_id"] = routing_id
        for field, incoming in (
            ("offered", kept_offered),
            ("delivered", kept_delivered),
            ("loaded", kept_loaded),
        ):
            bucket = rail[field]
            for name in incoming:
                if name not in bucket:
                    bucket.append(name)
        # A run whose root span never reaches the exporter would sit here
        # forever. Same discipline as the pending-span buffer: bounded, oldest
        # evicted first.
        while len(_skill_rails) > _SKILL_RAIL_MAX:
            _skill_rails.popitem(last=False)
    return True


def _pop_skill_rail(tid: int) -> Optional[Dict[str, Any]]:
    """Take this run's rail, removing it. Pop, never peek — a rail read twice
    would hand one routing decision to two traces."""
    with _skill_rails_lock:
        return _skill_rails.pop(tid, None)


def _reset_skill_rails() -> None:
    """Drop every buffered rail. Test seam only."""
    with _skill_rails_lock:
        _skill_rails.clear()


class _AgentNameStamper:
    """A span processor that records, at span START, whose agent the span is.

    Start, not export: by export time the span is sitting on the
    ``BatchSpanProcessor`` worker thread, which has no idea which run produced
    it. At start we are still on the caller's thread, so the context can answer.

    Implements the OTel ``SpanProcessor`` protocol by duck typing, the same way
    :class:`DecimalSpanExporter` implements ``SpanExporter`` — so this module
    still imports with no OTel SDK present, and so the SDK can stay mocked in
    tests. That means implementing the protocol **in full**, private hooks
    included: the SDK calls ``_on_ending`` from inside ``span.end()``, and a
    stand-in missing it raises there, killing the span and every span above it.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            name = _active_agent_name.get()
            if not name:
                return
            existing = getattr(span, "attributes", None) or {}
            if existing.get(_DECIMAL_AGENT_NAME):
                return  # an explicit name on the span outranks the context
            span.set_attribute(_DECIMAL_AGENT_NAME, name)
        except Exception:  # pragma: no cover - stamping must never break a span
            logger.debug("Could not stamp the agent name onto a span", exc_info=True)

    def _on_ending(self, span: Any) -> None:
        return None

    def on_end(self, span: Any) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# ── the run scope ────────────────────────────────────────────────────────────

#: Name of the span :func:`agent_run` opens. Deliberately generic and stable:
#: it is the only span DecimalAI itself contributes to a trace, and the UI keys
#: waterfall grouping off names.
RUN_SPAN_NAME = "agent.run"

#: GenAI semconv marks an agent-invocation span with this operation name. The
#: exporter reads it two ways: ``_classify_span`` files the span as an AGENT
#: span, and ``_NON_LLM_OPERATIONS`` keeps it out of ``llm_calls`` even if some
#: framework decorates it with a model attribute.
_GENAI_OPERATION = "gen_ai.operation.name"
_GENAI_AGENT_NAME = "gen_ai.agent.name"
_INVOKE_AGENT = "invoke_agent"


@contextlib.contextmanager
def agent_run(
    agent_name: Optional[str] = None,
    *,
    span_name: str = RUN_SPAN_NAME,
    tracer_provider: Any = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[Any]:
    """Wrap one logical agent run in a real parent span, named for its agent.

    This exists because a *provider* instrumentor (OpenInference's openai /
    anthropic / google-genai) has no idea what a run is. It sees one SDK call,
    emits one span, and — with nothing above it — that span is a ROOT: its own
    OTel ``trace_id``, its own DecimalAI trace. So the ordinary tool-use loop,
    which is at least two provider calls, arrives as N unrelated one-span traces
    with no key to group them by. Pydantic AI, which does no tracing of its own
    and rides entirely on the provider instrumentor, was observed fragmenting a
    single ``agent.run_sync()`` that way.

    Opening a span for the duration of the run fixes both halves at once:

    * **Structure.** The provider spans start inside this span's context, so
      OTel parents them under it — same ``trace_id``, real ``parent_span_id``.
      :class:`DecimalSpanExporter` buffers by trace and finalizes when a
      parentless root arrives, and now there is one. Note what this does NOT
      do: it adds the parent that genuinely wraps the calls and nothing else.
      Steps the framework never emitted (a tool span derived from an LLM span's
      attributes, say) stay absent, because inventing them would put a step in
      the waterfall that never ran.
    * **Identity.** ``agent_name`` is published to :data:`_active_agent_name`
      for the duration, so :class:`_AgentNameStamper` stamps it on every span
      the run produces and the exporter files the trace under the right agent.
      A ContextVar, so eight concurrent runs of eight agents each get their own
      answer — which the exporter's process-wide ``default_agent_name``, fixed
      when the exporter was built, cannot give.

    Nesting is safe: an inner ``agent_run`` becomes a child span like any other,
    it does not start a second trace.

    Args:
        agent_name: Whose run this is. ``None`` leaves the surrounding context's
            answer (or the exporter's default) in place.
        span_name: Span name. Defaults to :data:`RUN_SPAN_NAME`.
        tracer_provider: The ``TracerProvider`` to open the span on. Defaults to
            the process-global one. Pass the provider the DecimalAI exporter is
            attached to when it is not the global — otherwise the parent span
            never reaches the exporter and the children stay orphaned.
        attributes: Extra span attributes.

    Yields:
        The OTel span, or ``None`` when the OTel SDK is unavailable (the name is
        still scoped, so any other rail reading the context still sees it).

    Example::

        import decimalai
        decimalai.init(anthropic=True)

        with decimalai.providers.agent_run("support-bot"):
            first = client.messages.create(...)      # one trace,
            second = client.messages.create(...)     # two nested LLM calls
    """
    token = _active_agent_name.set(agent_name) if agent_name else None
    try:
        try:
            from opentelemetry import trace as trace_api
        except ImportError:  # pragma: no cover - OTel ships as a core dep
            logger.debug("agent_run(): no OpenTelemetry SDK; scoping the name only")
            yield None
            return

        span_attrs: Dict[str, Any] = {_GENAI_OPERATION: _INVOKE_AGENT}
        if agent_name:
            # Both spellings on purpose: `gen_ai.agent.name` is the semconv key
            # other backends read, `decimal.agent_name` is the one THIS exporter
            # reads (and setting it here means the trace is named correctly even
            # on a pipeline that has no _AgentNameStamper installed).
            span_attrs[_GENAI_AGENT_NAME] = agent_name
            span_attrs[_DECIMAL_AGENT_NAME] = agent_name
        if attributes:
            span_attrs.update(attributes)

        provider = tracer_provider or trace_api.get_tracer_provider()
        tracer = provider.get_tracer("decimalai")
        # start_as_current_span records the exception and sets ERROR status by
        # default, which is what C10 ("a failing run produces exactly ONE trace,
        # marked errored") needs — the raising run still emits its one trace.
        with tracer.start_as_current_span(span_name, attributes=span_attrs) as span:
            yield span
    finally:
        if token is not None:
            _active_agent_name.reset(token)


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
    # shutdown_on_exit=False + our own earlier hook: the SDK's default exit
    # flush runs too late to reach the background sender. See
    # _register_flush_atexit.
    provider = TracerProvider(resource=resource, shutdown_on_exit=False)

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
    # Whose run is this? Recorded in the CONTEXT, not just on the exporter,
    # because only the first call in a process gets to own the pipeline: OTel
    # honours ``set_tracer_provider`` once, and an instrumentor binds its tracer
    # the first time it is enabled. So the provider built by a SECOND
    # instrument() call never sees a span, and before this the second agent's
    # traces were silently filed under the first agent's name. The stamper below
    # goes onto THIS provider; the one already installed on the first provider
    # reads the same ContextVar, so it picks the new name up too.
    if agent_name:
        _active_agent_name.set(agent_name)
    provider.add_span_processor(_AgentNameStamper())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace_api.set_tracer_provider(provider)
    _register_flush_atexit(provider)

    logger.info(
        "DecimalAI OTEL exporter installed (agent_name=%s, service=%s)",
        agent_name,
        service_name,
    )
    return provider


class _ManifestRegistry:
    """Per-agent manifest state for an OTel exporter — always has an id to give.

    Two rules, both learned the hard way:

    **Always declare something.** The backend requires ``manifest_id`` on
    ingest (``require_manifest_on_ingest``, on by default and on in
    production), so an exporter that registers a manifest only when it
    happened to see a model/tool/prompt loses 100% of the traces that saw
    none — every Microsoft AutoGen run (its runtime spans carry no GenAI
    attributes at all) and every AG2/CrewAI turn that neither called a tool
    nor made a model call. Those traces 400'd. So a run with nothing to
    declare still registers: a snapshot with ZERO components, labelled
    ``undeclared``.

    Zero components is the honest encoding of "nothing to declare": every
    contract surface is ABSENT, which is what the diff engine reads as
    "not declared". The tempting alternative — synthesizing placeholder
    components (a model of ``provider="unknown"``, an empty tool registry) —
    writes a false claim into the contract AND makes the first real
    observation diff as ``provider: 'unknown' → 'openai'``, i.e. major /
    breaking. Absence says nothing; a placeholder says something wrong.

    **Never un-declare.** The accumulators are cumulative per agent, not per
    trace. A manifest describes the AGENT, not one turn: a turn where the
    model didn't call ``search`` has not REMOVED ``search``. Before this,
    ``seen_tools`` was rebuilt from each trace in isolation, so two identical
    turns of an unchanged AG2 agent (which declares tools only by *executing*
    them) registered two manifest versions and the platform reported
    ``tool_registry breaking/major "search removed"`` with a ``replay``
    decision — a fabricated breaking change on an agent nobody touched.
    Accumulating monotonically means the snapshot can grow (a genuine
    addition, which the diff calls minor/non-breaking) but never shrink.

    Same reason a "nothing observed" trace that follows a populated one
    reuses the last manifest instead of registering the empty snapshot: an
    empty snapshot ON TOP of a populated one is exactly the "everything was
    removed" diff, the worst version-history poisoning available.

    And when nothing has ever been observed for an agent that the workspace
    ALREADY has an active manifest for (a redeploy, another rail, an explicit
    ``decimalai.register_manifest``), the existing manifest is adopted rather
    than a fresh ``undeclared`` version minted on top of it — "nothing to
    declare" means "running under the contract already declared".
    """

    def __init__(
        self,
        skills: Optional[List[Dict[str, Any]]] = None,
        prompts: Optional[Dict[str, str]] = None,
    ):
        self._skills = skills
        # Explicit static prompt templates ({"system": ...}); when set, these
        # win over the rendered system prompt auto-harvested from spans.
        self._explicit_prompts = prompts
        self._lock = threading.Lock()
        self._by_agent: Dict[str, "_AgentManifestState"] = {}

    def manifest_id_for(
        self,
        agent_name: str,
        *,
        model: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Dict[str, Any]]] = None,
        prompts: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """Fold one trace's observations in and return the manifest id to stamp.

        Returns None only when the SDK is disabled or registration failed
        without even a synthetic id to fall back on.
        """
        from . import _config

        if not _config._is_enabled():
            return None

        with self._lock:
            state = self._by_agent.get(agent_name)
            if state is None:
                state = _AgentManifestState()
                self._by_agent[agent_name] = state

            # Monotonic merge — first observation of each component wins, and
            # nothing is ever dropped. See the class docstring.
            for tool_name, tool in (tools or {}).items():
                state.tools.setdefault(tool_name, tool)
            if state.model is None and model:
                state.model = model
            for prompt_name, prompt_text in (prompts or {}).items():
                state.prompts.setdefault(prompt_name, prompt_text)

            declared_tools = list(state.tools.values()) or None
            declared_models = {"default": state.model} if state.model else None
            declared_prompts = self._explicit_prompts or (state.prompts or None)
            undeclared = not (
                declared_tools or declared_models or declared_prompts or self._skills
            )

            if undeclared:
                if state.manifest_id is not None:
                    # Already have an id (adopted, or an earlier undeclared
                    # registration). Re-registering the empty snapshot would be
                    # a no-op at best; keep the id.
                    return state.manifest_id
                if not state.adoption_checked:
                    state.adoption_checked = True
                    adopted = self._adopt_active_manifest(agent_name)
                    if adopted:
                        state.manifest_id = adopted
                        logger.info(
                            "Nothing to declare for agent %s — reusing its active "
                            "manifest %s rather than registering an empty version",
                            agent_name, adopted,
                        )
                        return adopted

            snapshot = extract_from_config(
                agent_name=agent_name,
                tools=declared_tools,
                models=declared_models,
                prompts=declared_prompts,
                skills=self._skills,
                # A named label instead of the backend's v1/v2/v3 auto-increment,
                # so the timeline reads "undeclared → v1" (first declaration)
                # rather than "v1 → v2" (which reads as a change to a contract
                # that was never declared). Labels with no `vN` token don't
                # participate in auto-increment, so this doesn't consume v1.
                version_label=_UNDECLARED_LABEL if undeclared else None,
            )

            if not state.tracker.check_and_update(snapshot):
                return state.manifest_id  # Same hash — already registered

            try:
                client = _config._get_client()
                result = client.register_manifest(snapshot)
                state.manifest_id = result.get("manifest_id", snapshot.id)
                logger.info(
                    "Registered manifest %s from OTel spans (agent=%s, hash=%s, "
                    "components=%d)",
                    state.manifest_id,
                    agent_name,
                    snapshot.manifest_hash[:12],
                    len(snapshot.components),
                )
            except Exception as exc:
                logger.warning(
                    "Failed to register manifest from OTel spans for agent %s",
                    agent_name, exc_info=True,
                )
                _config._sender.record_manifest_error(exc)
                # Forget the hash so the NEXT trace retries. The fallback id
                # below is a client-side uuid the backend has never seen, and
                # it rejects an unknown manifest_id the same as a missing one —
                # so caching it would turn one blip (a restart, a network hiccup)
                # into permanent trace loss for the rest of the process.
                state.tracker.reset()
                state.manifest_id = snapshot.id
            return state.manifest_id

    def _adopt_active_manifest(self, agent_name: str) -> Optional[str]:
        """The id of ``agent_name``'s already-active manifest, if it has one."""
        from . import _config

        try:
            client = _config._get_client()
            resp = client.list_manifests(limit=20, agent_name=agent_name)
            for manifest in (resp or {}).get("manifests") or []:
                if (
                    manifest.get("status") == "active"
                    and manifest.get("agent_name") == agent_name
                    and manifest.get("id")
                ):
                    return str(manifest["id"])
        except Exception:
            # Best-effort: a lookup failure (or an unexpected response shape)
            # just means we register the undeclared snapshot instead.
            logger.debug(
                "Could not look up an existing manifest for %s", agent_name,
                exc_info=True,
            )
        return None


class _AgentManifestState:
    """One agent's cumulative manifest view inside a :class:`_ManifestRegistry`."""

    __slots__ = ("tracker", "manifest_id", "tools", "model", "prompts", "adoption_checked")

    def __init__(self) -> None:
        self.tracker = ManifestTracker()
        self.manifest_id: Optional[str] = None
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.model: Optional[Dict[str, Any]] = None
        self.prompts: Dict[str, str] = {}
        self.adoption_checked = False


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
        # win over the rendered system prompt auto-harvested from spans. The
        # registry below applies that precedence — kept here as the record of
        # what this exporter was constructed with.
        self._explicit_prompts = prompts
        # Manifest tracking state — per agent name, cumulative. One process can
        # export traces for several agents (the name is auto-detected from each
        # root span), and the manifest hash does NOT include the agent name, so
        # a single shared tracker would hand agent B the id it minted for A.
        self._manifests = _ManifestRegistry(skills=skills, prompts=prompts)
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
                # Stamp the manifest onto the trace AFTER registration — the
                # trace was assembled before it ran, and without an id the
                # backend rejects the trace under require_manifest_on_ingest.
                # There is always an id to stamp now, including for a run that
                # observed no model/tool/prompt (see _ManifestRegistry).
                run_trace.manifest_id = self._maybe_register_manifest(
                    agent_name, seen_model, seen_tools, seen_prompts
                )
                # The skills rail, if this run had one. Deliberately AFTER the
                # manifest stamp and outside _assemble_trace: four already
                # declared fields assigned on a finished object, so nothing
                # about trace shape, identity or manifest registration can move.
                rail = _pop_skill_rail(tid)
                if rail:
                    # A body the agent pulled with load_skill WAS delivered —
                    # it reached the model — so `loaded` folds into `delivered`.
                    #
                    # It deliberately does NOT fold into `offered`. That field is
                    # `skills_offered_in_prompt`: a claim that the name appeared
                    # in the prompt the model was shown, which is why
                    # record_skill_rail filters offered/delivered against the
                    # rendered fragment and drops anything absent. `loaded`
                    # arrives from a tool call and never passes that filter, so
                    # unioning it in here re-imported through the back door
                    # exactly the unverified claim the filter exists to reject.
                    # The ladder loaded ⊆ delivered still holds; loaded ⊆ offered
                    # does not, and should not — a skill loaded by name we cannot
                    # find in the fragment is a real signal, not a rounding error.
                    loaded = set(rail["loaded"])
                    delivered = set(rail["delivered"]) | loaded
                    offered = set(rail["offered"]) | set(rail["delivered"])
                    run_trace.routing_id = rail["routing_id"]
                    run_trace.skills_offered_in_prompt = sorted(offered)
                    run_trace.skills_delivered = sorted(delivered)
                    run_trace.skills_loaded_by_agent = sorted(loaded)
                # Disk skills the SDK did not inject. After the rail merge, so
                # the rail's own names are excluded rather than re-inferred,
                # and so the assignments above cannot clobber the result.
                self._infer_skill_rungs(run_trace)
                self._send(run_trace)
        except Exception:
            logger.exception(
                "Failed to assemble trace from %d spans (trace_id=%s)",
                len(group),
                hex(tid),
            )
        finally:
            # This run's rail must not outlive this run, on the FAILURE path
            # too. The pop above only runs when assembly succeeded AND the run
            # had a rail, so a run whose assembly raised used to leave its entry
            # in the process-global _skill_rails store forever. That store is
            # capped at _SKILL_RAIL_MAX with oldest-first eviction, so the leak
            # is bounded in memory but NOT harmless: every stranded entry
            # occupies a slot and pushes a live run's rail out early, which
            # loses that run's routing_id. Popping again here is free — the pop
            # above already removed it, so this is a no-op on the happy path.
            _pop_skill_rail(tid)

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

    def _infer_skill_rungs(self, run_trace: RunTrace) -> None:
        """Infer OFFERED / DELIVERED for disk skills the SDK did not inject.

        ``self._skills`` is disk-derived (``install(skills=...)`` else
        ``discover_skills()``), so it describes skills a harness may have put
        in the prompt itself. Prompt text establishes that the skill was put in
        front of the model, never that the model reached for it — so
        ``active_skills`` is untouched here. On this rail activation arrives
        only through a ``decimal.active_skills`` span attribute (an explicit
        declaration by the emitting framework) or a ``record_skill_rail``
        ``loaded`` entry.

        Called AFTER the rail merge in ``_finalize_trace``, for two reasons:
        the precedence rule needs this run's router-accounted names, and the
        merge assigns ``skills_delivered`` outright, so an earlier write would
        be discarded.
        """
        if not self._skills or not run_trace.llm_calls:
            return
        try:
            from .skills import infer_prompt_rungs

            # Passed split for readability; infer_prompt_rungs pools them —
            # see the trade-off note there on why suppression is blanket.
            router_offered = set(run_trace.skills_offered_in_prompt)
            router_delivered = set(run_trace.skills_delivered) | set(run_trace.skills_loaded_by_agent)
            offered, delivered = infer_prompt_rungs(
                (call.rendered_input for call in run_trace.llm_calls),
                self._skills,
                router_offered=router_offered,
                router_delivered=router_delivered,
            )
            if not offered and not delivered:
                return
            # NO delivered->offered fold here: see the note on the same site
            # in langchain.py. Tier-2 matches a BODY whose name may never
            # appear in the prompt, so folding would assert a menu row that was
            # never shown.
            run_trace.skills_offered_in_prompt = sorted(
                set(run_trace.skills_offered_in_prompt) | set(offered)
            )
            run_trace.skills_delivered = sorted(
                set(run_trace.skills_delivered) | set(delivered)
            )
        except Exception:
            logger.debug("Skill prompt-presence inference failed", exc_info=True)

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

        # ── Preserve the shape the framework actually emitted ──
        # A TraceSpan is identified by a UUID we mint, while OTel identifies a
        # span by a 64-bit id and names its parent by that id. So the parent
        # link can only be carried across if every span's UUID is known BEFORE
        # any span is built — a child is frequently exported ahead of its
        # parent (children end first, and BatchSpanProcessor preserves that
        # order), so resolving lazily during the loop would drop exactly the
        # links that matter. Mint the whole id map up front, then translate.
        #
        # This used to be `parent_span_id=None` on every span, which flattened
        # every trace on the OTel rail — CrewAI, AG2, Microsoft AutoGen,
        # raw-provider instrumentors and hand-rolled OTel alike — into a list
        # of same-level siblings. The waterfall then showed an 8-span agent run
        # as eight unrelated roots: no nesting, no "this LLM call happened
        # inside that tool", and no way to tell a retry from a sub-agent.
        span_uuids: Dict[str, Any] = {}
        for otel_span in otel_spans:
            sid = _get_span_id(otel_span)
            if sid is not None and sid not in span_uuids:
                span_uuids[sid] = uuid4()

        # ── One model call, two instrumentors ──
        # Instrumentors stack. CrewAI <= 1.15 routes through LiteLLM, which
        # calls the `openai` SDK, and the documented install enables an
        # instrumentor for BOTH — so a single completion emits an
        # `openai ChatCompletion` span nested inside a `litellm completion`
        # span, each carrying the same model and the same token counts. Recorded
        # naively that is one call reported twice: doubled token counts, doubled
        # cost, and nothing in the contract catches it because every field on
        # both records is individually valid.
        #
        # A real model call is a LEAF. An LLM span with an LLM span inside it is
        # a wrapper around that call, so the wrapper is dropped and the
        # innermost span — the one closest to the wire — is kept. This also
        # stays correct if a wrapper genuinely issues several calls: it has
        # several LLM children, and those children are the calls.
        llm_span_ids: set = set()
        llm_parent_ids: set = set()
        for otel_span in otel_spans:
            _a = dict(getattr(otel_span, "attributes", None) or {})
            _op = str(_a.get("gen_ai.operation.name") or "").lower()
            if _op in _NON_LLM_OPERATIONS:
                continue
            if not _get_first(_a, _GENAI_MODEL, *_ALT_MODEL_KEYS):
                continue
            _sid = _get_span_id(otel_span)
            if _sid is not None:
                llm_span_ids.add(_sid)
            _pid = _get_parent_span_id(otel_span)
            if _pid is not None:
                llm_parent_ids.add(_pid)
        wrapper_llm_span_ids = llm_span_ids & llm_parent_ids

        # ── Whose run was this? ──
        # The exporter's own ``default_agent_name`` is fixed when the exporter
        # is BUILT, and on this rail that is once per process: OTel honours
        # ``set_tracer_provider`` once, and every instrumentor binds its tracer
        # the first time it is enabled. So a second ``instrument(agent_name=B)``
        # in the same process changes nothing — B's spans still reach A's
        # exporter and every one of B's traces is filed under A. The name has to
        # travel WITH the span, which is what ``decimal.agent_name`` is for
        # (set by the caller directly, or stamped at span start by
        # :class:`_AgentNameStamper` from the agent active in that context).
        # The root span's answer wins; any span's beats the process-wide guess.
        root_declared: Optional[str] = None
        any_declared: Optional[str] = None
        for otel_span in otel_spans:
            declared = _get_attributes(otel_span).get(_DECIMAL_AGENT_NAME)
            if not isinstance(declared, str) or not declared:
                continue
            if any_declared is None:
                any_declared = declared
            if root_declared is None and _get_parent_span_id(otel_span) is None:
                root_declared = declared
        if root_declared or any_declared:
            agent_name = root_declared or any_declared

        for otel_span in otel_spans:
            attrs = _get_attributes(otel_span)
            name = _get_span_name(otel_span)
            parent_id = _get_parent_span_id(otel_span)
            span_id_str = _get_span_id(otel_span)
            started_at = _ns_to_datetime(getattr(otel_span, "start_time", None))
            ended_at = _ns_to_datetime(getattr(otel_span, "end_time", None))
            # This span's identity, and its parent's — but only when the parent
            # is one of the spans in this trace. A parent that never reached
            # this exporter (owned by another tracer, sampled out, or dropped
            # from an over-full buffer) stays unset rather than becoming a
            # dangling pointer the UI cannot resolve.
            span_uuid = span_uuids.get(span_id_str) if span_id_str else None
            if span_uuid is None:
                span_uuid = uuid4()
            parent_uuid = span_uuids.get(parent_id) if parent_id else None

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

            if model and _get_span_id(otel_span) in wrapper_llm_span_ids:
                # A wrapper around another LLM span, not a second call to the
                # model. Skipped so the nested instrumentor pair above counts
                # once. It is still a span in the waterfall; it just is not an
                # llm_call.
                model = None

            if model:
                # This is an LLM call — create LlmCallRecord
                llm_call = self._make_llm_call(
                    attrs, name, model, started_at, ended_at, otel_span,
                    span_uuid=span_uuid, agent_name=agent_name,
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
                    id=span_uuid,
                    parent_span_id=parent_uuid,
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
                #
                # The ASK is the last USER message of the rendered request, not
                # the whole request joined and truncated — see
                # _ask_from_rendered_input. The joined preview stays as the
                # fallback's fallback, for a span that carries prompt content
                # but no messages to pick from.
                if not root_set_input and user_input is None:
                    user_input = (
                        _ask_from_rendered_input(llm_call.rendered_input)
                        or trace_span.input_preview
                    )
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

                # Accumulate tools for manifest auto-detection. Prefer the
                # declared tool name over the span name: AG2 names its tool
                # spans "execute_tool <fn>", which would put the operation
                # prefix into the manifest's tool registry and re-key every
                # tool if the framework ever renames its spans.
                if span_type == SpanType.TOOL:
                    tool_name = str(
                        attrs.get("gen_ai.tool.name") or attrs.get("tool.name") or name
                    )
                    seen_tools.setdefault(tool_name, {"name": tool_name})

                trace_span = TraceSpan(
                    id=span_uuid,
                    parent_span_id=parent_uuid,
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

        # No prompt-matching here. Inferring the offered/delivered rungs is
        # done by `_infer_skill_rungs` in the caller, STRICTLY AFTER the run's
        # skill rail is merged — this method returns before the rail is popped,
        # and the merge assigns `skills_delivered` outright, so anything
        # written from in here would be silently discarded on every run that
        # had a rail. `active_skills` below therefore holds only what a
        # `decimal.active_skills` span attribute explicitly declared.

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
            # Stamped by _finalize_trace once the manifest for this agent has
            # been registered — the caller knows the agent name by then.
            manifest_id=None,
        ), seen_model, seen_tools, seen_prompts

    def _make_llm_call(
        self,
        attrs: Dict[str, Any],
        name: str,
        model: str,
        started_at: Optional[datetime],
        ended_at: Optional[datetime],
        otel_span: Any,
        span_uuid: Optional[Any] = None,
        agent_name: Optional[str] = None,
    ) -> LlmCallRecord:
        """Build an LlmCallRecord from OTEL span attributes.

        ``span_uuid`` is the id of the :class:`TraceSpan` this call is the
        wrapper for, so a consumer can put the call back where it happened in
        the waterfall instead of guessing by timestamp. ``agent_name`` is the
        name the enclosing trace resolved to, so a call cannot claim a
        different agent than the trace it is part of.
        """
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
        if not finish_reasons:
            for _k in _ALT_FINISH_REASON_KEYS:
                if attrs.get(_k):
                    finish_reasons = attrs[_k]
                    break
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
            span_id=span_uuid,
            agent_name=agent_name or self.default_agent_name,
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
            tool_calls=_extract_tool_calls(attrs),
        )

    def _send(self, trace: RunTrace) -> None:
        """Send a trace via the background sender."""
        from . import _config

        if not _config._is_enabled():
            return

        try:
            client = _config._get_client()
            _submit_or_send_inline(client, trace)
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
    ) -> Optional[str]:
        """Fold this trace's observations into the agent's manifest.

        Returns the manifest id to stamp on the trace. Always registers
        something — a run that observed nothing still needs an id or the
        backend rejects its trace. Delegates to :class:`_ManifestRegistry`,
        which owns the accumulate-never-shrink rules.
        """
        return self._manifests.manifest_id_for(
            agent_name,
            model=seen_model,
            tools=seen_tools,
            # An explicit static template (instrument(prompts=...)) wins over
            # the auto-harvested rendered prompt — see the rendered-vs-template
            # note in _assemble_trace; the registry applies that precedence.
            prompts=seen_prompts,
        )


# ── Utilities ──────────────────────────────────────────────


def _submit_or_send_inline(client: Any, trace: RunTrace) -> None:
    """Queue a trace on the background sender, falling back to a direct POST.

    The background sender is a ``ThreadPoolExecutor``, and CPython runs
    ``threading._shutdown()`` — which is where the executor's own exit hook
    refuses further work — BEFORE ordinary ``atexit`` callbacks. So any flush
    that happens from a plain ``atexit`` handler (a user's own
    ``TracerProvider``, which defaults to ``shutdown_on_exit=True``) hits
    ``RuntimeError: cannot schedule new futures after interpreter shutdown``
    and the trace is lost. :func:`_register_flush_atexit` moves our own
    provider's flush earlier; this covers the paths we don't own by sending
    the trace on the calling thread instead of dropping it.
    """
    from . import _config

    try:
        _config._sender.submit(client.ingest_trace, trace)
        return
    except RuntimeError:
        logger.debug(
            "Background sender unavailable (interpreter shutting down) — "
            "sending trace %s inline", trace.id,
        )
    try:
        client.ingest_trace(trace)
    except Exception as exc:
        _config._sender._record_failure(exc, str(trace.id))
        raise
    _config._sender._record_success()


def _register_flush_atexit(provider: Any) -> None:
    """Flush ``provider`` early enough that the background sender is still alive.

    ``TracerProvider(shutdown_on_exit=True)`` registers its shutdown as an
    ordinary ``atexit`` callback, and those run AFTER
    ``threading._shutdown()`` has already stopped the thread pool the SDK
    sends on — so a plain script (no explicit ``flush()``) exported zero
    traces on the whole community rail. ``threading._register_atexit`` runs
    during ``threading._shutdown()``, in reverse registration order, so a hook
    registered here runs before the pool's own — the flush lands while it can
    still be queued and drained.

    Falls back to plain ``atexit`` on a runtime without the private hook; the
    inline-send fallback in :func:`_submit_or_send_inline` still saves the
    trace there.
    """
    import atexit

    def _flush() -> None:
        try:
            provider.shutdown()
        except Exception:  # pragma: no cover - never crash on the way out
            logger.debug("TracerProvider shutdown failed", exc_info=True)

    register = getattr(threading, "_register_atexit", None)
    if register is not None:
        register(_flush)
    else:  # pragma: no cover - CPython < 3.9 / alternative runtimes
        atexit.register(_flush)


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


def _semconv_tool_calls(attrs: Dict[str, Any]) -> List[ToolCallRecord]:
    """Tool calls carried as ``tool_call`` parts in ``gen_ai.output.messages``.

    The GenAI-semconv counterpart of the OpenInference indexed keys below —
    without it an AG2 assistant turn's tool calls reached neither
    ``LlmCallRecord.tool_calls`` nor the manifest's tool registry.
    """
    records: List[ToolCallRecord] = []
    raw = attrs.get("gen_ai.output.messages")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return records
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return records
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        for part in entry.get("parts") or ():
            if not isinstance(part, dict) or part.get("type") != "tool_call":
                continue
            name = part.get("name")
            if not name:
                continue
            args = part.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = {"raw": args}
            records.append(
                ToolCallRecord(
                    # The outcome is NOT observed here. These records are read
                    # off the LLM span, which the instrumentor emits when the
                    # model REQUESTS a tool — before the tool runs, and carrying
                    # no result. ToolCallRecord defaults to SUCCESS, so leaving
                    # it produced "this tool ran and succeeded" for a call that
                    # may never have executed or may have raised. RUNNING is the
                    # honest state for "requested, outcome unseen"; a rail that
                    # genuinely observes the result sets SUCCESS or ERROR.
                    tool_name=str(name),
                    status=Status.RUNNING,
                    args=args if isinstance(args, dict) else {},
                )
            )
    return records


def _extract_tool_calls(attrs: Dict[str, Any]) -> List[ToolCallRecord]:
    """Pull tool calls the model made in this step from OpenInference
    ``llm.output_messages.*.message.tool_calls.*`` attributes, or from the
    GenAI-semconv ``gen_ai.output.messages`` array."""
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
        # Requested, not observed — see the note on the sibling extractor above.
        records.append(ToolCallRecord(tool_name=name, status=Status.RUNNING, args=args))
    return records or _semconv_tool_calls(attrs)


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

# Current GenAI semconv carries the whole conversation as ONE attribute holding
# a JSON array — `gen_ai.input.messages` / `gen_ai.output.messages`, each entry
# `{"role", "parts": [...]}` — rather than the indexed keys above. AG2 emits
# this shape (both the agent spans and, with capture_messages on, the chat
# span); reading only the indexed spelling made every AG2 preview the raw JSON.
_SEMCONV_MSG_KEYS = {
    "input": ("gen_ai.input.messages",),
    "output": ("gen_ai.output.messages",),
}


def _text_from_parts(parts: Any) -> str:
    """Flatten a GenAI-semconv message ``parts`` list to plain text.

    Text parts contribute their content; a tool-call part contributes a compact
    ``name(arguments)`` rendering and a tool result its response, so a turn that
    only called tools still previews as something a human can read instead of
    an empty string.
    """
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, (list, tuple)):
        return ""
    chunks: List[str] = []
    for part in parts:
        if isinstance(part, str):
            chunks.append(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "tool_call":
            args = part.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args, default=str) if args else ""
            chunks.append(f"{part.get('name', 'tool')}({args})")
        elif ptype == "tool_call_response":
            chunks.append(str(part.get("response", "")))
        else:
            content = part.get("content") or part.get("text")
            if content is not None:
                chunks.append(str(content))
    return "\n".join(c for c in chunks if c)


def _semconv_messages_from_attrs(
    attrs: Dict[str, Any], direction: str
) -> Optional[List[Dict[str, Any]]]:
    """Rebuild one direction's messages from the GenAI-semconv JSON attribute."""
    for key in _SEMCONV_MSG_KEYS[direction]:
        raw = attrs.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (ValueError, TypeError):
                continue
        if isinstance(raw, dict):
            raw = [raw]
        if not isinstance(raw, (list, tuple)) or not raw:
            continue
        messages: List[Dict[str, Any]] = []
        for entry in raw:
            if isinstance(entry, str):
                messages.append({"role": _DEFAULT_ROLE[direction], "content": entry})
            elif isinstance(entry, dict):
                messages.append({
                    "role": str(entry.get("role") or _DEFAULT_ROLE[direction]),
                    "content": _text_from_parts(
                        entry.get("parts", entry.get("content", ""))
                    ),
                })
        if messages:
            return messages
    return None


def _messages_from_attrs(
    attrs: Dict[str, Any], direction: str
) -> Optional[List[Dict[str, Any]]]:
    """Rebuild one direction's chat messages from span attributes.

    Two wire shapes, both supported. Frameworks that follow OpenInference
    (CrewAI, LlamaIndex/Phoenix, …) split each message across
    ``…{i}.message.role`` and ``…{i}.message.content`` keys; frameworks on
    current GenAI semconv (AG2 among them) put the whole array in one
    ``gen_ai.{input,output}.messages`` JSON attribute. Returns them in order as
    ``{"role", "content"}`` dicts — the shape the other adapters normalize to —
    or None when the span carries neither.

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
        return _semconv_messages_from_attrs(attrs, direction)
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


def _ask_from_rendered_input(
    rendered_input: Optional[Sequence[Any]], max_len: int = 200
) -> Optional[str]:
    """The run's ASK — the LAST USER message, not the whole request joined.

    ``user_input_preview`` is a trace-level field meaning "what was asked", and
    the fallback that fills it from an LLM span used to take
    ``_preview_from_attrs(attrs, "input")``: every message joined with newlines
    and cut at ``max_len``. Two things are wrong with that, and only the second
    is about the cap:

    * a rendered chat request LEADS with the system preamble, so once that
      preamble passes the cap the preview is 100% system prompt and 0% user
      question. Measured on CrewAI: a 167-char preamble puts the ask at index
      198 of the join, leaving two of its characters inside a 200-char preview;
    * on turn six of a conversation the join is the entire history, so the
      current ask is drowned in it however large the cap is.

    The rule here is ported verbatim from :func:`decimalai.openai_agents.
    _query_from_input_items`, which already made this exact call for this exact
    field and wrote down why: "The last user message IS the current ask …
    routing on the concatenation would drown the current ask in history." Its
    fallback is ported too — the last message with any text, for a request that
    carries only assistant/tool context and no user turn at all.
    ``decimalai.langchain`` previews ``call.rendered_input[-1]["content"]`` for
    the same field. The OTel exporter was the outlier, not these.

    Returns None when there is nothing to read, so the caller can fall back
    again — to the joined preview, for a span carrying prompt text but no
    messages to choose between.
    """
    entries = [e for e in (rendered_input or []) if isinstance(e, dict)]
    if not entries:
        return None

    def _role(i: int) -> str:
        return str(entries[i].get("role") or "").lower()

    def _text(i: int) -> str:
        c = entries[i].get("content")
        return c if isinstance(c, str) else ""

    fallback: Optional[str] = None
    for i in range(len(entries) - 1, -1, -1):
        content = _text(i)
        if not content.strip():
            continue
        if _role(i) == "user":
            # NOT every role=="user" entry is something a person asked. Anthropic
            # (and anything following its Messages shape) renders a TOOL RESULT
            # as a user turn, so the last user message in a tool loop is machine
            # output — taking it blindly put a tool result in a field that means
            # "what was asked". Measured: it changed user_input_preview on 20
            # anthropic and 19 pydantic-ai traces.
            #
            # The tell is in the turn before it. An assistant message that only
            # made tool calls is kept with EMPTY content precisely so turn order
            # survives (see _messages_from_attrs), so a user turn preceded by a
            # contentless assistant turn is the result coming back — while a
            # genuine follow-up in a chat is preceded by an assistant turn that
            # actually said something. That keeps the ask right in all three
            # shapes: [system, user], the tool loop, and a multi-turn
            # conversation where the LAST user turn really is the current ask.
            if i > 0 and _role(i - 1) == "assistant" and not _text(i - 1).strip():
                continue
            return content[:max_len]
        if fallback is None:
            fallback = content[:max_len]
    return fallback


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
            # Metadata ABOUT the content is not the content. The scan matches
            # key names by substring, so `input.mime_type` matches the "input"
            # pattern — and once blank attributes are skipped, a span carrying
            # `input.value=""` alongside `input.mime_type="application/json"`
            # falls through to the mime type and reports it as the prompt.
            # A preview reading "application/json" is indistinguishable
            # downstream from a model that was really shown that string.
            if key_lower.endswith(_NON_CONTENT_KEY_SUFFIXES):
                continue
            if _INDEXED_MSG_NAMESPACE_RE.match(key_lower):
                continue
            if key_pattern in key_lower:
                text = str(val)
                # A blank attribute is not content, and must not be reported as
                # if it were. CrewAI's kickoff span carries ``crew_inputs=''``
                # whenever the crew was started without an inputs dict — the
                # common shape — and that key matches the "input" pattern. A
                # bare `return str(val)` therefore answered "" for a span that
                # simply carries no input, which is a DIFFERENT claim from
                # "there is none": the empty string is a value, so
                # _assemble_trace's `if root_input is not None` accepted it,
                # set the trace preview to "" and suppressed the LLM fallback
                # that had the real prompt. Keep scanning instead — a later key
                # (or a later pattern) may hold the actual content.
                if not text.strip():
                    continue
                return text
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
