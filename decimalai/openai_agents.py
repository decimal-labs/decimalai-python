"""OpenAI Agents SDK integration.

Traces all agent runs, LLM generations, tool calls, handoffs, and guardrails
and sends them to the DecimalAI backend.

Simple path (global, 3 lines)::

    import decimalai
    decimalai.init()

    from decimalai.openai_agents import instrument
    instrument()  # all Agent runs are now traced

Custom path (exclusive — replaces default OpenAI tracing)::

    from decimalai.openai_agents import instrument
    instrument(exclusive=True)
"""

from __future__ import annotations

import warnings

import logging
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import NAMESPACE_OID, UUID, uuid4, uuid5


def _coerce_span_id(raw: Any) -> Optional[UUID]:
    """OpenAI Agents emits span_ids like 'span_<hex>' / 'trace_<hex>', but
    our `TraceSpan` schema requires UUIDs. Map non-UUID strings to a
    deterministic UUID5 so parent/child relationships stay consistent
    within a trace.

    Returns None for None input. Raises only on truly malformed input.
    """
    if raw is None:
        return None
    if isinstance(raw, UUID):
        return raw
    s = str(raw)
    try:
        return UUID(s)
    except (ValueError, AttributeError):
        return uuid5(NAMESPACE_OID, s)

from .schema.common import FinishReason, SpanType, Status
from .schema.manifest import ManifestTracker, extract_from_config
from .schema.trace import LlmCallRecord, RunTrace, TraceSpan

logger = logging.getLogger("decimalai.openai_agents")

# ── Global manifest state ──────────────────────────────────
_manifest_tracker = ManifestTracker()
_manifest_id: Optional[str] = None  # Set after first successful registration
_manifest_lock = threading.Lock()  # Thread safety for manifest registration

# ── Thread-local parent trace context ──────────────────────
# Enables multi-agent linking: set a parent trace ID and all subsequent
# traces in this thread will include it as parent_trace_id.
_parent_ctx = threading.local()


def set_parent_trace(parent_trace_id: str) -> None:
    """Set the parent trace ID for multi-agent linking.

    All traces sent from this thread will include ``parent_trace_id``
    in their payload, linking them as children of the parent orchestrator.

    Example::

        from decimalai.openai_agents import set_parent_trace, clear_parent_trace

        # After the orchestrator trace completes:
        set_parent_trace(orchestrator_trace_id)

        # Run sub-agent — its trace will be linked to the parent
        result = Runner.run_sync(sub_agent, task)

        clear_parent_trace()
    """
    _parent_ctx.parent_trace_id = parent_trace_id


def get_parent_trace() -> Optional[str]:
    """Get the current parent trace ID, or None if not set."""
    return getattr(_parent_ctx, "parent_trace_id", None)


def clear_parent_trace() -> None:
    """Clear the parent trace ID context."""
    _parent_ctx.parent_trace_id = None


# ── SkillRouter routing-id context ──────────────────────────
# Populated by the dynamic instructions callable (see _install_skill_loader).
# Read by _send_trace when assembling the RunTrace. ContextVar (not threading.local)
# so asyncio tasks get isolated copies — important for parallel Runner.run_async
# under a single thread.
_routing_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "decimalai_skill_router_routing_id", default=None,
)


def _set_routing_id(routing_id: Optional[str]) -> None:
    _routing_id_ctx.set(routing_id)


def _consume_routing_id() -> Optional[str]:
    """Read + clear the current routing_id."""
    rid = _routing_id_ctx.get()
    if rid is not None:
        _routing_id_ctx.set(None)
    return rid


# Skill Rater discovery telemetry context. The skill loader
# stamps offered names here (drained from `router.consume_last_offered_names`)
# and `_send_trace` drains them into RunTrace.skills_offered_in_prompt.
# Set type instead of list so multi-LLM-call turns dedupe naturally.
_skills_offered_ctx: ContextVar[Optional[set]] = ContextVar(
    "decimalai_skills_offered_in_prompt", default=None,
)


def _add_skills_offered(names: List[str]) -> None:
    """Add offered skill names to the current trace's running set."""
    if not names:
        return
    existing = _skills_offered_ctx.get() or set()
    existing.update(n for n in names if isinstance(n, str) and n.strip())
    _skills_offered_ctx.set(existing)


def _consume_skills_offered() -> List[str]:
    """Read + clear the current trace's offered-names set."""
    s = _skills_offered_ctx.get()
    if not s:
        return []
    _skills_offered_ctx.set(None)
    return sorted(s)


# 'delivered' = the full skill body reached the model (the Router's body
# injection). Same mirror-rail pattern as the offered names above.
_skills_delivered_ctx: ContextVar[Optional[set]] = ContextVar(
    "decimalai_skills_delivered", default=None,
)


def _add_skills_delivered(names: List[str]) -> None:
    """Add delivered skill names to the current trace's running set."""
    if not names:
        return
    existing = _skills_delivered_ctx.get() or set()
    existing.update(n for n in names if isinstance(n, str) and n.strip())
    _skills_delivered_ctx.set(existing)


def _consume_skills_delivered() -> List[str]:
    """Read + clear the current trace's delivered-names set."""
    s = _skills_delivered_ctx.get()
    if not s:
        return []
    _skills_delivered_ctx.set(None)
    return sorted(s)


# ── SkillRouter dynamic loader ──────────────────────────────
# When `install(enable_skill_loader=True)` runs, we monkey-patch
# `agents.Agent.__init__` so every Agent created afterwards has its
# string `instructions` wrapped into a callable that prepends skill
# content fetched from the platform per-run. User-supplied callables
# pass through untouched (their judgment wins).

_skill_loader_installed = False
_skill_router_singleton: Any = None


def _get_skill_router() -> Any:
    """Lazily construct a SkillRouter using the SDK's global config."""
    global _skill_router_singleton
    if _skill_router_singleton is not None:
        return _skill_router_singleton
    try:
        from ._config import _get_config
        from .skill_router import SkillRouter
        config = _get_config()
        _skill_router_singleton = SkillRouter(
            api_key=config.api_key,
            base_url=config.base_url,
            inject_body=getattr(config, "inject_skill_body", False),
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter singleton init failed", exc_info=True)
        return None


def _extract_query(ctx: Any) -> Optional[str]:
    """Best-effort: pull the user's input out of a RunContextWrapper.

    The Agents SDK doesn't formally expose this on a stable attribute,
    so we probe several known shapes. If nothing matches we fall back
    to full-menu mode (query=None) — still useful, just no semantic
    routing for that call.
    """
    for attr in ("input", "user_input", "query"):
        val = getattr(ctx, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    inner = getattr(ctx, "context", None)
    if inner is not None:
        for attr in ("input", "user_input", "query"):
            val = getattr(inner, attr, None)
            if isinstance(val, str) and val.strip():
                return val
    return None


def _load_skill_tool_enabled() -> bool:
    """Config gate for the load_skill tool (kill switch:
    DECIMALAI_LOAD_SKILL_TOOL=0 / init(load_skill_tool=False))."""
    try:
        from ._config import _get_config
        return bool(getattr(_get_config(), "load_skill_tool", True))
    except Exception:
        return True


def _handle_load_skill(name: str) -> str:
    """Tool callback — always returns a string the model can act on."""
    router = _get_skill_router()
    if router is None:
        return "load_skill error: skill router unavailable."
    try:
        return router.load_skill(name)
    except Exception:
        logger.debug("load_skill handler failed (non-fatal)", exc_info=True)
        return f"load_skill error: could not load {name!r} (transient error)."


def _make_load_skill_tool() -> Any:
    """Build the native load_skill FunctionTool (the progressive-disclosure path).

    The OpenAI Agents SDK owns its tool loop, so the tool result is routed
    back mid-turn for free — this adapter ships the tool live (langchain /
    anthropic patch a non-loop layer and stay prompt-injection)."""
    try:
        from agents import function_tool
    except Exception:
        return None
    from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION

    def load_skill(name: str) -> str:
        return _handle_load_skill(name)

    load_skill.__doc__ = LOAD_SKILL_TOOL_DESCRIPTION
    try:
        return function_tool(load_skill)
    except Exception:
        logger.debug("function_tool(load_skill) failed (non-fatal)", exc_info=True)
        return None


def _agent_has_load_skill_tool(agent: Any) -> bool:
    return any(
        getattr(t, "name", None) == "load_skill"
        for t in (getattr(agent, "tools", None) or [])
    )


def _make_skill_aware_instructions(base: str):
    """Return a sync callable usable as `Agent.instructions`."""

    def instructions_fn(ctx: Any, agent: Any) -> str:
        try:
            router = _get_skill_router()
            if router is None:
                return base
            fragment, routing_id = router.build_prompt_fragment(
                query=_extract_query(ctx),
                agent_name=getattr(agent, "name", None),
            )
            if routing_id:
                _set_routing_id(routing_id)
            # Pull the names the Router offered for this call and
            # accumulate against the active trace.
            from .skill_router import (
                consume_last_delivered_names,
                consume_last_offered_names,
            )
            offered = consume_last_offered_names()
            if offered:
                _add_skills_offered(offered)
            # Names whose BODY the Router injected count as delivered.
            delivered = consume_last_delivered_names()
            if delivered:
                _add_skills_delivered(delivered)
            if not fragment:
                return base
            # Tell the model how bodies arrive — only when this
            # agent actually has the tool. The server fragment keeps the
            # activation-statement instruction unchanged (Stage-M parity).
            if _agent_has_load_skill_tool(agent):
                from .skill_router import LOAD_SKILL_PROMPT_HINT
                fragment = f"{fragment}\n{LOAD_SKILL_PROMPT_HINT}"
            return f"{fragment}\n\n{base}".strip() if base else fragment
        except Exception:
            # Never break a run because of a Router hiccup.
            logger.debug("Skill loader callable failed (non-fatal)", exc_info=True)
            return base

    return instructions_fn


def _install_skill_loader() -> None:
    """Monkey-patch `agents.Agent.__init__` so new Agents auto-load skills.

    Idempotent — safe to call multiple times. Only wraps string-typed
    `instructions`; if a user passed their own callable, we leave it
    alone (their judgment > ours). Also registers the load_skill tool on
    every new Agent unless config disables it.
    """
    global _skill_loader_installed
    if _skill_loader_installed:
        return
    try:
        from agents import Agent
    except ImportError:
        logger.warning(
            "enable_skill_loader=True but openai-agents not installed; "
            "skipping skill loader install. "
            "Install it with: pip install \"decimalai[openai-agents]\""
        )
        return

    original_init = Agent.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        current = getattr(self, "instructions", None)
        if current is None or isinstance(current, str):
            base = current or ""
            self.instructions = _make_skill_aware_instructions(base)
        # callables left untouched
        # Register load_skill so surfaced descriptions are
        # executable — an agent that can see a skill can pull its body.
        if _load_skill_tool_enabled():
            try:
                tools = getattr(self, "tools", None)
                if isinstance(tools, list) and not _agent_has_load_skill_tool(self):
                    tool = _make_load_skill_tool()
                    if tool is not None:
                        tools.append(tool)
            except Exception:
                logger.debug(
                    "load_skill tool registration failed (non-fatal)", exc_info=True
                )

    Agent.__init__ = patched_init  # type: ignore[method-assign]
    _skill_loader_installed = True
    logger.info("DecimalAI SkillRouter loader installed (OpenAI Agents)")

# ── Type aliases for duck-typing against the OpenAI Agents SDK ──
# We avoid hard imports so the module loads even without openai-agents installed.
# At runtime, the actual Trace and Span objects are passed to us by the SDK.


def _introspect_agent(agent: Any) -> Dict[str, Any]:
    """Introspect an OpenAI Agents SDK Agent object for manifest data.

    Extracts tools (with full JSON schemas if available), model,
    instructions, and handoffs from the Agent object.

    Args:
        agent: An ``agents.Agent`` instance.

    Returns:
        Dict with keys: tools, prompts, models, subagents.
    """
    result: Dict[str, Any] = {}

    # Extract tools with full schemas
    agent_tools = getattr(agent, "tools", None) or []
    if agent_tools:
        tools_list = []
        for t in agent_tools:
            tool_entry: Dict[str, Any] = {"name": getattr(t, "name", str(t))}
            # FunctionTool has params_json_schema
            schema = getattr(t, "params_json_schema", None)
            if schema:
                tool_entry["schema"] = schema
            # Also try input_json_schema (some versions)
            if not schema:
                schema = getattr(t, "input_json_schema", None)
                if schema:
                    tool_entry["schema"] = schema
            tools_list.append(tool_entry)
        result["tools"] = tools_list

    # Extract instructions as prompt
    instructions = getattr(agent, "instructions", None)
    if instructions and isinstance(instructions, str):
        result["prompts"] = {"system": instructions}

    # Extract model — resolve the model NAME whether `agent.model` is a bare
    # string ("gpt-5-mini") or a Model instance (e.g. an
    # OpenAIChatCompletionsModel pointed at a non-OpenAI provider). str() on a
    # Model instance yields a useless object repr, so pull `.model`/`.name`.
    model = getattr(agent, "model", None)
    if model is not None:
        if isinstance(model, str):
            model_name = model
        else:
            _n = getattr(model, "model", None) or getattr(model, "name", None)
            model_name = _n if isinstance(_n, str) else ""
        if model_name:
            result["models"] = {"default": {
                "provider": _infer_provider(model_name),
                "model": model_name,
            }}

    # Extract handoffs as subagents
    handoffs = getattr(agent, "handoffs", None) or []
    if handoffs:
        subagents = []
        for h in handoffs:
            h_name = getattr(h, "name", None) or getattr(h, "agent_name", str(h))
            subagents.append({"name": h_name})
        result["subagents"] = subagents

    return result


def instrument(
    agent_name: Optional[str] = None,
    *,
    agent: Any = None,
    exclusive: bool = False,
    skills: Optional[List[Dict[str, Any]]] = None,
    skill_dirs: Optional[List[str]] = None,
    enable_skill_loader: bool = False,
    disk_sync: Optional[bool] = None,
) -> None:
    """Register DecimalAI as a trace processor for OpenAI Agents SDK.

    After calling ``install()``, every ``Runner.run()`` / ``Runner.run_sync()``
    call will be automatically traced and sent to the DecimalAI backend.

    Args:
        agent_name: Default agent name for all traces. If None, the name
            is auto-detected from the root agent span.
        agent: Optional ``agents.Agent`` instance. If provided, DecimalAI
            will introspect it for full tool schemas, instructions, model,
            and handoffs to register a manifest immediately. This gives
            the highest fidelity manifest (full JSON schemas for tools).
            If omitted, manifest is auto-detected from span data at trace
            time (tool names only, no schemas).
        exclusive: If True, replaces ALL existing trace processors (including
            OpenAI's default). If False (default), adds alongside existing
            processors so traces still go to the OpenAI dashboard too.

    Raises:
        ImportError: If ``openai-agents`` is not installed.

    Example::

        import decimalai
        decimalai.init()

        from decimalai.openai_agents import instrument
        from agents import Agent, Runner

        agent = Agent(name="my-agent", instructions="You are helpful.")
        instrument(agent=agent)  # full tool schema introspection

        result = Runner.run_sync(agent, "Hello!")
        # Trace auto-captured and sent to DecimalAI
    """
    global _manifest_id

    try:
        from agents.tracing import (
            add_trace_processor,
            set_trace_processors,
        )
    except ImportError:
        raise ImportError(
            "openai-agents is required for instrument() but is not installed. "
            "Install the OpenAI Agents extra with: "
            "pip install \"decimalai[openai-agents]\""
        )

    # Router authority (skill_authority): when None, derive disk_sync from config
    # — router-authoritative installs (loader active) default to NOT mirroring
    # skills to disk, so a native skill-loading runtime can't also load them and
    # double-inject. An explicit disk_sync=... always wins.
    if disk_sync is None:
        try:
            from ._config import _get_config
            disk_sync = _get_config().resolve_disk_sync(loader_active=enable_skill_loader)
        except Exception:
            disk_sync = True

    # Resolve agent name from Agent object if not provided
    if agent_name is None and agent is not None:
        agent_name = getattr(agent, "name", None)

    # Resolve skills (auto-discover or explicit). Skip discovery entirely
    # when disk_sync is False — the caller is signaling that the platform
    # is the only source/sink and the SDK shouldn't read local SKILL.md.
    resolved_skills: Optional[List[Dict[str, Any]]] = skills
    if disk_sync and not resolved_skills:
        try:
            from .skills import discover_skills
            resolved_skills = discover_skills(skill_dirs) or None
        except Exception:
            logger.debug("Skill auto-discovery failed", exc_info=True)

    # Sync discovered skills to platform for observability + pull platform-
    # only skills back to disk for IDE/runtime use. Both paths are gated on
    # disk_sync=True. When False, the SDK treats the platform as the sole
    # source of truth (useful for pure-Python stacks that don't use a
    # disk-loading runtime, or to suppress duplicate injection when the
    # SkillRouter loader is the active injector).
    if disk_sync:
        try:
            from ._config import _get_config, _sender

            config = _get_config()
            local_names = {s["name"] for s in resolved_skills} if resolved_skills else set()

            if resolved_skills:
                def _sync_skills_background():
                    try:
                        from .skill_router import SkillRouter

                        router = SkillRouter(
                            api_key=config.api_key,
                            base_url=config.base_url,
                        )
                        try:
                            from ._install import get_install_identity
                            _ident = get_install_identity()
                        except Exception:
                            _ident = {}
                        from .skills import _with_local_timestamps
                        result = router.sync_skills(
                            _with_local_timestamps(resolved_skills),
                            # Don't blind-clobber a teammate's newer dashboard
                            # edit on agent boot; record this checkout's baseline.
                            conflict_policy="newer_wins",
                            install_id=_ident.get("install_id"),
                            install_label=_ident.get("install_label"),
                        )
                        synced = result.get("synced", 0)
                        created = result.get("created", 0)
                        logger.info(
                            "Skills synced to platform: %d synced, %d created",
                            synced,
                            created,
                        )
                    except Exception:
                        logger.debug("Background skill sync failed", exc_info=True)

                _sender.submit(_sync_skills_background)

            # Bidirectional sync: pull platform-only skills to disk.
            def _pull_missing_background():
                try:
                    from .disk_export import AGENT_PATHS
                    from .skill_router import SkillRouter

                    router = SkillRouter(
                        api_key=config.api_key,
                        base_url=config.base_url,
                    )
                    # A trace's agent_name is a free-form label, not a disk
                    # runtime key — only honor names that are real runtimes,
                    # else export_to_disk raises "Unknown agent" on every
                    # install whose agent has a custom name.
                    target_agent = agent_name if agent_name in AGENT_PATHS else "universal"
                    result = router.pull_missing(
                        local_skill_names=local_names,
                        agents=[target_agent],
                        disk_wins=False,  # platform wins by default
                    )
                    pulled = result.get("pulled", 0)
                    updated = result.get("updated", 0)
                    if pulled or updated:
                        logger.info(
                            "Pulled %d new skills from platform, updated %d",
                            pulled,
                            updated,
                        )
                except Exception:
                    logger.debug("Background skill pull failed (non-fatal)", exc_info=True)

            _sender.submit(_pull_missing_background)
        except Exception:
            logger.debug("Skill sync setup failed (non-fatal)", exc_info=True)

    processor = DecimalTracingProcessor(
        agent_name=agent_name,
        skills_registry=resolved_skills,
    )

    if exclusive:
        set_trace_processors([processor])
    else:
        add_trace_processor(processor)

    # If an Agent object was provided, introspect it for manifest
    if agent is not None:
        _register_manifest_from_agent(agent, agent_name, resolved_skills)

    # SkillRouter dynamic loader — opt-in. When enabled, Agent.__init__
    # is monkey-patched so newly created Agents have their string
    # `instructions` wrapped into a per-run callable that prepends
    # skill content from the platform. See _install_skill_loader().
    if enable_skill_loader:
        from .skill_router import _warn_if_disk_runtime_detected
        _warn_if_disk_runtime_detected("openai_agents")
        _install_skill_loader()

    logger.info(
        "DecimalAI OpenAI Agents tracing installed (agent_name=%s, exclusive=%s, agent_introspected=%s, skill_loader=%s, disk_sync=%s)",
        agent_name,
        exclusive,
        agent is not None,
        enable_skill_loader,
        disk_sync,
    )


def _register_manifest_from_agent(
    agent: Any,
    agent_name: Optional[str],
    skills: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Register a manifest by introspecting an Agent object at install time."""
    global _manifest_id
    from . import _config

    if not _config._is_enabled():
        return

    try:
        data = _introspect_agent(agent)
        resolved_name = agent_name or getattr(agent, "name", "unknown")

        snapshot = extract_from_config(
            agent_name=resolved_name,
            tools=data.get("tools"),
            prompts=data.get("prompts"),
            models=data.get("models"),
            subagents=data.get("subagents"),
            skills=skills,
        )

        with _manifest_lock:
            if not _manifest_tracker.check_and_update(snapshot):
                return  # Same hash — already registered

            client = _config._get_client()
            result = client.register_manifest(snapshot)
            _manifest_id = result.get("manifest_id", snapshot.id)
            logger.info(
                "Registered manifest %s from Agent introspection (hash=%s, components=%d)",
                _manifest_id,
                snapshot.manifest_hash[:12],
                len(snapshot.components),
            )
    except Exception:
        logger.warning("Failed to register manifest from Agent introspection", exc_info=True)


class _TraceAccumulator:
    """Per-trace state that accumulates spans as they complete."""

    def __init__(self, trace_id: str, trace_name: str):
        self.trace_id = trace_id
        self.trace_name = trace_name
        self.started_at: Optional[datetime] = None
        self.spans: List[TraceSpan] = []
        self.llm_calls: List[LlmCallRecord] = []
        self.agent_name: Optional[str] = None
        self.user_input_preview: Optional[str] = None
        self.final_output_preview: Optional[str] = None
        self.status = Status.SUCCESS
        self.error_message: Optional[str] = None
        # Manifest auto-detection accumulators
        self.seen_tools: Dict[str, Dict[str, Any]] = {}  # name -> {name, ...}
        self.seen_model: Optional[Dict[str, Any]] = None
        self.seen_handoffs: List[str] = []
        self.seen_guardrails: List[str] = []
        self.active_skills: Dict[str, Optional[str]] = {}
        # SkillRouter: routing decision that surfaced the skills for this
        # trace. Stamped at trace-end from `_consume_routing_id()`, so
        # `routing_decision × trace_skill_activation` joins can close.
        self.routing_id: Optional[str] = None
        # Skill Rater discovery telemetry. ``skills_offered_in_prompt``
        # is auto-populated by the dynamic-instructions callable from the
        # Router's offered set; ``skills_loaded_by_agent`` is a manual hook
        # for callers who want to annotate that the agent read a skill's
        # body (use `decimalai.log_skill_loaded`).
        self.skills_offered_in_prompt: set[str] = set()
        self.skills_loaded_by_agent: set[str] = set()
        # Bodies that reached the model (Router body injection) —
        # between offered and activated; never implies activation.
        self.skills_delivered: set[str] = set()


class DecimalTracingProcessor:
    """OpenAI Agents SDK TracingProcessor that captures traces for DecimalAI.

    Implements the ``TracingProcessor`` protocol via duck typing (no
    inheritance required — the SDK checks methods, not class hierarchy).

    Thread-safe: each trace gets its own ``_TraceAccumulator`` keyed by
    ``trace_id``, protected by a lock.
    """

    def __init__(self, agent_name: Optional[str] = None, skills_registry: Optional[List[Dict[str, Any]]] = None):
        self.default_agent_name = agent_name
        self._skills_registry = skills_registry or []
        self._traces: Dict[str, _TraceAccumulator] = {}
        self._lock = threading.Lock()

    # ── TracingProcessor protocol ──────────────────────────

    def on_trace_start(self, trace: Any) -> None:
        """Called when a new trace begins execution."""
        trace_id = getattr(trace, "trace_id", None) or str(uuid4())
        trace_name = getattr(trace, "name", "") or "trace"

        acc = _TraceAccumulator(trace_id=trace_id, trace_name=trace_name)
        acc.started_at = datetime.now(timezone.utc)

        with self._lock:
            self._traces[trace_id] = acc

    def on_trace_end(self, trace: Any) -> None:
        """Called when a trace completes — assembles and sends the RunTrace."""
        trace_id = getattr(trace, "trace_id", None)
        if not trace_id:
            return

        with self._lock:
            acc = self._traces.pop(trace_id, None)

        if acc is None:
            logger.warning("on_trace_end for unknown trace %s", trace_id)
            return

        self._send_trace(acc)

    def on_span_start(self, span: Any) -> None:
        """Called when a new span begins — we just note the start time."""
        # Most data is available at span end, so this is a lightweight hook.
        pass

    def on_span_end(self, span: Any) -> None:
        """Called when a span completes — extract data based on span type."""
        trace_id = getattr(span, "trace_id", None)
        if not trace_id:
            return

        with self._lock:
            acc = self._traces.get(trace_id)

        if acc is None:
            return

        span_data = getattr(span, "span_data", None)
        span_type = getattr(span_data, "type", None) if span_data else None

        if span_type == "generation":
            self._handle_generation(span, span_data, acc)
        elif span_type == "function":
            self._handle_function(span, span_data, acc)
        elif span_type == "agent":
            self._handle_agent(span, span_data, acc)
        elif span_type == "response":
            self._handle_response(span, span_data, acc)
        elif span_type == "handoff":
            self._handle_handoff(span, span_data, acc)
        elif span_type == "guardrail":
            self._handle_guardrail(span, span_data, acc)
        elif span_type == "custom":
            self._handle_custom(span, span_data, acc)
        else:
            # Unknown span type — create a generic TraceSpan
            self._handle_generic(span, span_data, acc)

    def shutdown(self) -> None:
        """Flush any remaining traces and clean up."""
        with self._lock:
            remaining = list(self._traces.values())
            self._traces.clear()

        for acc in remaining:
            self._send_trace(acc)

    def force_flush(self) -> None:
        """Force-flush queued data (no-op — we send immediately on trace_end)."""
        pass

    # ── Span handlers ─────────────────────────────────────

    def _handle_generation(
        self, span: Any, span_data: Any, acc: _TraceAccumulator
    ) -> None:
        """Map a GenerationSpanData to LlmCallRecord."""
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        # Extract model and usage from span_data
        model = getattr(span_data, "model", None)
        model_config = getattr(span_data, "model_config", None) or {}
        usage = getattr(span_data, "usage", None) or {}

        # Accumulate model info for manifest auto-detection
        if model and acc.seen_model is None:
            acc.seen_model = {
                "provider": _infer_provider(model),
                "model": model,
                "temperature": model_config.get("temperature") if isinstance(model_config, dict) else None,
                "max_tokens": model_config.get("max_tokens") if isinstance(model_config, dict) else None,
            }

        # Extract input/output
        raw_input = getattr(span_data, "input", None)
        raw_output = getattr(span_data, "output", None)

        # Convert input to rendered_input format
        rendered_input = None
        if raw_input:
            rendered_input = _normalize_messages(raw_input)

        # Convert output
        output_dict = None
        if raw_output:
            if isinstance(raw_output, (list, tuple)):
                # Sequence of message dicts
                output_dict = {"messages": list(raw_output)}
            elif isinstance(raw_output, dict):
                output_dict = raw_output
            else:
                output_dict = {"content": str(raw_output)}

        # Determine provider from model name
        provider = _infer_provider(model)

        # Parse token usage
        input_tokens = usage.get("input_tokens") or usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens") or usage.get("completion_tokens")

        # Latency
        latency_ms = None
        if started_at and ended_at:
            latency_ms = int((ended_at - started_at).total_seconds() * 1000)

        # Error handling
        span_error = getattr(span, "error", None)
        status = Status.SUCCESS
        finish_reason = FinishReason.STOP
        if span_error:
            status = Status.ERROR
            finish_reason = FinishReason.ERROR
            if output_dict is None:
                error_msg = span_error.get("message", "") if isinstance(span_error, dict) else str(span_error)
                output_dict = {"error": error_msg[:500]}

        call = LlmCallRecord(
            id=uuid4(),
            span_id=parent_id,
            agent_name=acc.agent_name or self.default_agent_name,
            provider=provider,
            model_name=model,
            temperature=model_config.get("temperature") if isinstance(model_config, dict) else None,
            max_output_tokens=model_config.get("max_tokens") if isinstance(model_config, dict) else None,
            rendered_input=rendered_input,
            output=output_dict,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
        )
        acc.llm_calls.append(call)

        # Also create a wrapper span
        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.LLM,
            name=f"generation:{model or 'unknown'}",
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            input_preview=_preview(raw_input),
            output_preview=_preview(raw_output),
        )
        acc.spans.append(trace_span)

    def _handle_function(
        self, span: Any, span_data: Any, acc: _TraceAccumulator
    ) -> None:
        """Map a FunctionSpanData to ToolCallRecord + TraceSpan."""
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        func_name = getattr(span_data, "name", "unknown")
        func_input = getattr(span_data, "input", None)
        func_output = getattr(span_data, "output", None)

        span_error = getattr(span, "error", None)
        status = Status.ERROR if span_error else Status.SUCCESS

        latency_ms = None
        if started_at and ended_at:
            latency_ms = int((ended_at - started_at).total_seconds() * 1000)

        # Create TraceSpan
        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.TOOL,
            name=func_name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            input_preview=str(func_input)[:200] if func_input else None,
            output_preview=str(func_output)[:200] if func_output else None,
        )
        acc.spans.append(trace_span)

    def _handle_agent(
        self, span: Any, span_data: Any, acc: _TraceAccumulator
    ) -> None:
        """Map an AgentSpanData to TraceSpan and auto-detect agent name."""
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        agent_span_name = getattr(span_data, "name", "agent")

        # Auto-detect agent name from the first (root-ish) agent span
        if acc.agent_name is None and agent_span_name:
            acc.agent_name = agent_span_name

        span_error = getattr(span, "error", None)
        status = Status.ERROR if span_error else Status.SUCCESS

        # Accumulate tools for manifest (names only from span data)
        span_tools = getattr(span_data, "tools", None) or []
        for tool_name in span_tools:
            name_str = str(tool_name)
            if name_str not in acc.seen_tools:
                acc.seen_tools[name_str] = {"name": name_str}

        # Accumulate handoffs as subagent references
        span_handoffs = getattr(span_data, "handoffs", None) or []
        for h in span_handoffs:
            h_name = str(h)
            if h_name not in acc.seen_handoffs:
                acc.seen_handoffs.append(h_name)

        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.AGENT,
            name=agent_span_name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            attributes={
                "tools": span_tools,
                "handoffs": span_handoffs,
                "output_type": getattr(span_data, "output_type", None),
            },
        )
        acc.spans.append(trace_span)

    def _handle_response(
        self, span: Any, span_data: Any, acc: _TraceAccumulator
    ) -> None:
        """Map a ResponseSpanData (Responses API call) to TraceSpan."""
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        span_error = getattr(span, "error", None)
        status = Status.ERROR if span_error else Status.SUCCESS

        response_obj = getattr(span_data, "response", None)
        response_id = None
        if response_obj:
            response_id = getattr(response_obj, "id", None)

        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.OTHER,
            name="response",
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            attributes={"response_id": response_id} if response_id else {},
        )
        acc.spans.append(trace_span)

        # Capture final output from response
        if response_obj and hasattr(response_obj, "output"):
            acc.final_output_preview = _preview(response_obj.output)

    def _handle_handoff(
        self, span: Any, span_data: Any, acc: _TraceAccumulator
    ) -> None:
        """Map a HandoffSpanData to a TraceSpan."""
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        from_agent = getattr(span_data, "from_agent", None)
        to_agent = getattr(span_data, "to_agent", None)

        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.OTHER,
            name=f"handoff:{from_agent}->{to_agent}",
            status=Status.SUCCESS,
            started_at=started_at,
            ended_at=ended_at,
            attributes={"from_agent": from_agent, "to_agent": to_agent},
        )
        acc.spans.append(trace_span)

    def _handle_guardrail(
        self, span: Any, span_data: Any, acc: _TraceAccumulator
    ) -> None:
        """Map a GuardrailSpanData to a TraceSpan."""
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        guardrail_name = getattr(span_data, "name", "guardrail")
        triggered = getattr(span_data, "triggered", False)

        # Accumulate guardrail for manifest
        if guardrail_name not in acc.seen_guardrails:
            acc.seen_guardrails.append(guardrail_name)

        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.OTHER,
            name=f"guardrail:{guardrail_name}",
            status=Status.SUCCESS,
            started_at=started_at,
            ended_at=ended_at,
            attributes={"triggered": triggered},
        )
        acc.spans.append(trace_span)

    def _handle_custom(
        self, span: Any, span_data: Any, acc: _TraceAccumulator
    ) -> None:
        """Map a CustomSpanData to a TraceSpan."""
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        custom_name = getattr(span_data, "name", "custom")
        custom_data = getattr(span_data, "data", {})

        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.OTHER,
            name=custom_name,
            status=Status.SUCCESS,
            started_at=started_at,
            ended_at=ended_at,
            attributes=custom_data if isinstance(custom_data, dict) else {},
        )
        acc.spans.append(trace_span)

    def _handle_generic(
        self, span: Any, span_data: Any, acc: _TraceAccumulator
    ) -> None:
        """Fallback handler for unknown span types."""
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        span_type_str = getattr(span_data, "type", "unknown") if span_data else "unknown"
        span_error = getattr(span, "error", None)
        status = Status.ERROR if span_error else Status.SUCCESS

        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.OTHER,
            name=span_type_str,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
        )
        acc.spans.append(trace_span)

    # ── Send trace ─────────────────────────────────────────

    def _send_trace(self, acc: _TraceAccumulator) -> None:
        """Assemble a RunTrace from the accumulator and send it."""
        from . import _config

        if not _config._is_enabled():
            logger.debug("Tracing disabled, skipping send")
            return

        agent_name = acc.agent_name or self.default_agent_name or acc.trace_name
        ended_at = datetime.now(timezone.utc)

        # Auto-register manifest from span data (fallback if no Agent was passed)
        self._maybe_register_manifest(acc, agent_name)

        # Auto-detect skill activations from LLM calls
        self._detect_skills(acc)

        config = _config._config

        # Build active_skills list
        active_skills_list: List[Dict[str, Any]] = []
        for name, h in acc.active_skills.items():
            entry: Dict[str, Any] = {"name": name}
            if h:
                entry["hash"] = h
            active_skills_list.append(entry)

        # SkillRouter: consume the routing_id set by the dynamic
        # instructions callable. We read at trace-end (not start)
        # because the instructions callable fires AFTER on_trace_start.
        if acc.routing_id is None:
            acc.routing_id = _consume_routing_id()

        # Drain the per-trace offered-names contextvar populated
        # by the skill loader callable. Merge with any direct
        # `log_skill_offered` calls already accumulated on the accumulator.
        drained_offered = _consume_skills_offered()
        if drained_offered:
            acc.skills_offered_in_prompt.update(drained_offered)

        # Drain the delivered-names contextvar (Router body injection).
        drained_delivered = _consume_skills_delivered()
        if drained_delivered:
            acc.skills_delivered.update(drained_delivered)
            acc.skills_offered_in_prompt.update(drained_delivered)  # delivered implies offered

        trace = RunTrace(
            id=uuid4(),
            project=config.project if config else None,
            agent_name=agent_name,
            status=acc.status,
            source_type="production",
            started_at=acc.started_at or ended_at,
            ended_at=ended_at,
            user_input_preview=acc.user_input_preview,
            final_output_preview=acc.final_output_preview,
            spans=acc.spans,
            llm_calls=acc.llm_calls,
            active_skills=active_skills_list,
            manifest_id=_manifest_id,
            parent_trace_id=get_parent_trace(),
            routing_id=acc.routing_id,
            # Sorted for deterministic output (tests, diffs).
            skills_offered_in_prompt=sorted(acc.skills_offered_in_prompt),
            skills_loaded_by_agent=sorted(acc.skills_loaded_by_agent),
            skills_delivered=sorted(acc.skills_delivered),
        )

        try:
            client = _config._get_client()
            _config._sender.submit(client.ingest_trace, trace)
            logger.debug(
                "Queued trace %s (%d spans, %d llm_calls, %d active_skills, manifest=%s) for agent %s",
                trace.id,
                len(trace.spans),
                len(trace.llm_calls),
                len(trace.active_skills),
                trace.manifest_id or "none",
                agent_name,
            )
        except Exception:
            logger.exception("Failed to queue trace %s", trace.id)

    def _detect_skills(self, acc: _TraceAccumulator) -> None:
        """Auto-detect skill activations from LLM call content."""
        if not self._skills_registry or not acc.llm_calls:
            return

        try:
            from .skills import detect_skill_activations
            for call in acc.llm_calls:
                if not call.rendered_input:
                    continue
                detected = detect_skill_activations(
                    call.rendered_input, self._skills_registry
                )
                for skill_name in detected:
                    if skill_name not in acc.active_skills:
                        registry_hash = next(
                            (s.get("hash") for s in self._skills_registry
                             if s.get("name") == skill_name),
                            None,
                        )
                        acc.active_skills[skill_name] = registry_hash
        except Exception:
            logger.debug("Skill auto-detection failed", exc_info=True)

    def _maybe_register_manifest(
        self, acc: _TraceAccumulator, agent_name: str
    ) -> None:
        """Extract and register manifest from accumulated span data.

        This is the fallback path when no Agent object was passed to instrument().
        Thread-safe via _manifest_lock.
        """
        global _manifest_id
        from . import _config

        if not _config._is_enabled():
            return

        # If manifest was already registered via Agent introspection, skip
        if _manifest_id is not None:
            return

        # Need at least tools or model to register
        tools = list(acc.seen_tools.values()) if acc.seen_tools else None
        models = {"default": acc.seen_model} if acc.seen_model else None
        subagents = [{"name": h} for h in acc.seen_handoffs] if acc.seen_handoffs else None

        if not tools and not models:
            return

        snapshot = extract_from_config(
            agent_name=agent_name,
            tools=tools,
            models=models,
            subagents=subagents,
        )

        with _manifest_lock:
            if not _manifest_tracker.check_and_update(snapshot):
                return  # Same hash — already registered

            try:
                client = _config._get_client()
                result = client.register_manifest(snapshot)
                _manifest_id = result.get("manifest_id", snapshot.id)
                logger.info(
                    "Registered manifest %s from span data (hash=%s, components=%d)",
                    _manifest_id,
                    snapshot.manifest_hash[:12],
                    len(snapshot.components),
                )
            except Exception:
                logger.warning("Failed to register manifest from spans", exc_info=True)
                _manifest_id = snapshot.id


# ── Utilities ──────────────────────────────────────────────


def _parse_iso(val: Optional[str]) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp string to datetime, or return None."""
    if not val:
        return None
    try:
        # Handle ISO timestamps with or without timezone
        if val.endswith("Z"):
            val = val[:-1] + "+00:00"
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return None


def _preview(obj: Any, max_len: int = 200) -> Optional[str]:
    """Create a preview string from an arbitrary object."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj[:max_len]
    if isinstance(obj, (list, tuple)):
        # For message lists, show first message content
        if obj and isinstance(obj[0], dict):
            content = obj[0].get("content", str(obj[0]))
            return str(content)[:max_len]
        return str(obj)[:max_len]
    if isinstance(obj, dict):
        content = obj.get("content", str(obj))
        return str(content)[:max_len]
    return str(obj)[:max_len]


def _normalize_messages(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize message input to the DecimalAI rendered_input format."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        result = []
        for item in raw:
            if isinstance(item, dict):
                result.append({
                    "role": item.get("role", "user"),
                    "content": str(item.get("content", "")),
                })
            else:
                result.append({"role": "user", "content": str(item)})
        return result
    return [{"role": "user", "content": str(raw)}]


def _infer_provider(model: Optional[str]) -> Optional[str]:
    """Infer the provider from a model name."""
    if not model:
        return None
    model_lower = model.lower()
    if "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    if "claude" in model_lower:
        return "anthropic"
    if "gemini" in model_lower:
        return "google"
    if "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    if "llama" in model_lower:
        return "meta"
    # Default for OpenAI Agents SDK — it's OpenAI
    return "openai"


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
        "decimalai.openai_agents.install() is deprecated; use "
        "decimalai.openai_agents.instrument() instead. It turns on tracing for openai_agents "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
