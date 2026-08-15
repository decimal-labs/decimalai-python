"""LangChain / LangGraph integration.

Simple path (global, 2 lines)::

    import decimalai
    decimalai.init()

    from decimalai.langchain import instrument
    instrument()  # all LangChain calls are now traced

Custom path (per-call control)::

    from decimalai.langchain import CallbackHandler
    agent.invoke(input, config={"callbacks": [CallbackHandler(agent_name="my-agent")]})
"""

from __future__ import annotations

import warnings

import logging
import threading
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

# The one module-level framework import in this file, and it is deliberate: a base
# class cannot be chosen lazily the way every other langchain import here is. The
# try/except preserves the guarantee that a core-only install stays importable —
# without langchain-core the handler simply subclasses object and keeps working by
# duck typing, which is how it has always worked at runtime.
#
# Why bother, given duck typing works: langchain-core annotates
# ``Callbacks = list[BaseCallbackHandler] | ...``, so passing this handler to
# ``config={"callbacks": [...]}`` is a type error under mypy and pyright even though
# it runs correctly. Verified on langchain-core 1.5.3: there is no isinstance check
# in the callback manager, and traces are byte-identical either way.
try:  # pragma: no cover - import shape depends on the installed extras
    from langchain_core.callbacks import BaseCallbackHandler as _CallbackBase
except Exception:  # noqa: BLE001 - any import failure means "not installed"
    _CallbackBase = object  # type: ignore[assignment,misc]

from .integrations._lc_compat import (
    extract_message_content,
    extract_model_name,
    extract_output_dict,
    extract_provider,
    extract_token_usage,
    has_tool_calls,
    normalize_role,
)
from .schema.common import FinishReason, SpanType, Status
from .schema.manifest import ManifestTracker, extract_from_config
from .schema.trace import LlmCallRecord, RunTrace, ToolCallRecord, TraceSpan

logger = logging.getLogger("decimalai.langchain")


# Surface the FIRST occurrence of each silent-failure category at WARNING
# level (one-shot per process), then drop subsequent occurrences back to
# DEBUG. The categories below are all user-actionable problems (missing
# SKILL.md, network down, wrong API key, etc.) that the prior implementation
# hid at DEBUG level — invisible at the default WARNING root logger level.
# Staying quiet hides real misconfiguration, so we surface
# once-per-category by default.
_warned_once: set[str] = set()


def _warn_once_then_debug(category: str, message: str) -> None:
    """First call per `category` logs at WARNING with exc_info; subsequent
    calls in the same process log at DEBUG. Use for non-fatal SDK-internal
    failures where a single warning is enough to flag misconfiguration."""
    if category not in _warned_once:
        _warned_once.add(category)
        logger.warning(message, exc_info=True)
    else:
        logger.debug(message, exc_info=True)


# Internal LangChain chain types that add noise without value
_SKIP_CHAIN_TYPES = frozenset({
    "RunnableSequence", "RunnableLambda", "RunnableParallel",
    "RunnablePassthrough", "RunnableBranch", "RunnableWithFallbacks",
    "RunnableAssign", "RunnablePick", "RunnableEach",
})

# ContextVar for global callback registration
_decimal_callback_var: ContextVar[Optional[CallbackHandler]] = ContextVar(
    "decimal_langchain_callback", default=None
)

_installed = False

# Global manifest state — shared across all handler instances
_manifest_tracker = ManifestTracker()
_manifest_id: Optional[str] = None  # Set after first successful registration
_manifest_lock = threading.Lock()  # Thread safety for manifest registration
_explicit_manifest_config: Optional[Dict[str, Any]] = None  # From instrument() kwargs

# Global eval state — populated by instrument()
_evals: List[Any] = []  # List of DecimalEval instances
_builtin_evals_enabled: bool = True

# ── SkillRouter routing-id context ──────────────────────────
# Populated by the BaseChatModel.invoke monkey-patch (see
# _install_skill_loader). Read by CallbackHandler.build_trace when
# assembling the RunTrace, so the platform's
# `routing_decision × trace_skill_activation` join can close.
_routing_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "decimalai_skill_router_routing_id_langchain", default=None,
)


def _set_routing_id(routing_id: Optional[str]) -> None:
    _routing_id_ctx.set(routing_id)


def _consume_routing_id() -> Optional[str]:
    """Read + clear the current routing_id."""
    rid = _routing_id_ctx.get()
    if rid is not None:
        _routing_id_ctx.set(None)
    return rid


# Skill Rater discovery telemetry. Same pattern as routing_id:
# the BaseChatModel.invoke patch stamps offered names here (drained from
# `router.consume_last_offered_names`); build_trace drains into the
# RunTrace.skills_offered_in_prompt field. Set type so multi-LLM-call
# turns dedupe naturally.
_skills_offered_ctx: ContextVar[Optional[set]] = ContextVar(
    "decimalai_skills_offered_in_prompt_langchain", default=None,
)


def _add_skills_offered(names: List[str]) -> None:
    if not names:
        return
    existing = _skills_offered_ctx.get() or set()
    existing.update(n for n in names if isinstance(n, str) and n.strip())
    _skills_offered_ctx.set(existing)


def _consume_skills_offered() -> List[str]:
    s = _skills_offered_ctx.get()
    if not s:
        return []
    _skills_offered_ctx.set(None)
    return sorted(s)


# 'delivered' = the full skill body reached the model (the Router's body
# injection). Same mirror-rail pattern as the offered names above.
_skills_delivered_ctx: ContextVar[Optional[set]] = ContextVar(
    "decimalai_skills_delivered_langchain", default=None,
)


def _add_skills_delivered(names: List[str]) -> None:
    if not names:
        return
    existing = _skills_delivered_ctx.get() or set()
    existing.update(n for n in names if isinstance(n, str) and n.strip())
    _skills_delivered_ctx.set(existing)


def _consume_skills_delivered() -> List[str]:
    s = _skills_delivered_ctx.get()
    if not s:
        return []
    _skills_delivered_ctx.set(None)
    return sorted(s)


def _clean_names(names: Any) -> List[str]:
    """Keep only non-blank strings — same filter as `_add_skills_offered`."""
    if not isinstance(names, (list, tuple, set)):
        return []
    return [n for n in names if isinstance(n, str) and n.strip()]


def _drain_router_rails() -> tuple[Optional[str], List[str], List[str], List[str]]:
    """Drain the Router singleton's instance rails —
    ``(routing_id, offered, delivered, loaded)``.

    The contextvars above stay authoritative wherever they propagate, but
    under LangChain they don't: the BaseChatModel patch writes them inside
    the runnable's context, and LangChain dispatches this handler's
    callbacks under `copy_context()`, so every rail came back empty on a
    run that demonstrably injected a menu. The Router carries the same
    values as instance state; `build_trace` unions both. Known cost (same
    as `consume_loaded_names`): concurrent runs sharing one router
    singleton drain into whichever trace sends first.
    """
    if _skill_router_singleton is None:
        return None, [], [], []
    try:
        routing_id = _skill_router_singleton.consume_routing_id()
        offered = _skill_router_singleton.consume_offered_names()
        delivered = _skill_router_singleton.consume_delivered_names()
        loaded = _skill_router_singleton.consume_loaded_names()
    except Exception:
        # A router object from an older SDK carries none of these methods.
        logger.debug("router-rail drain failed (non-fatal)", exc_info=True)
        return None, [], [], []
    # Sanitize here rather than at the merge: a stray string would
    # otherwise be added one letter at a time as three "skills".
    return (
        routing_id if isinstance(routing_id, str) else None,
        _clean_names(offered),
        _clean_names(delivered),
        _clean_names(loaded),
    )


# ── SkillRouter dynamic loader ──────────────────────────────
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


def _extract_query_from_messages(messages: Any) -> Optional[str]:
    """Walk a LangChain message list backward to find the most recent
    human/user message content. Used to drive semantic routing.

    Returns None when the input shape is unrecognized — the caller
    degrades to full-menu mode (no semantic routing) in that case.
    """
    if isinstance(messages, str):
        return messages.strip() or None
    if not isinstance(messages, list):
        return None
    for msg in reversed(messages):
        # LangChain BaseMessage instances expose .type and .content
        role = getattr(msg, "type", None) or getattr(msg, "role", None)
        if role in ("human", "user"):
            content = getattr(msg, "content", None) or (
                msg.get("content") if isinstance(msg, dict) else None
            )
            if isinstance(content, str) and content.strip():
                return content
    return None


def _inject_skills_into_input(input_value: Any) -> Any:
    """Prepend a SkillRouter-built system message to a chat model's input.

    Accepts the three common input shapes LangChain's BaseChatModel.invoke
    sees: a string, a list of messages (BaseMessage / dict), or a
    PromptValue. Falls through unchanged on anything else.
    """
    router = _get_skill_router()
    if router is None:
        return input_value

    try:
        from langchain_core.messages import SystemMessage
    except ImportError:
        return input_value

    query = _extract_query_from_messages(input_value)
    try:
        fragment, routing_id = router.build_prompt_fragment(query=query)
    except Exception:
        logger.debug("build_prompt_fragment failed (non-fatal)", exc_info=True)
        return input_value

    if routing_id:
        _set_routing_id(routing_id)
    # Pull the names the Router offered for this call and
    # accumulate against the active trace.
    from .skill_router import consume_last_delivered_names, consume_last_offered_names
    offered = consume_last_offered_names()
    if offered:
        _add_skills_offered(offered)
    # Names whose BODY the Router injected count as delivered.
    delivered = consume_last_delivered_names()
    if delivered:
        _add_skills_delivered(delivered)
    if not fragment:
        return input_value

    sys_msg = SystemMessage(content=fragment)
    if isinstance(input_value, list):
        return [sys_msg, *input_value]
    if isinstance(input_value, str):
        # Convert string → [system_with_skills, human_with_query]
        from langchain_core.messages import HumanMessage
        return [sys_msg, HumanMessage(content=input_value)]
    # Unrecognized shape (PromptValue, etc.) — leave untouched.
    return input_value


def _install_skill_loader() -> None:
    """Monkey-patch `BaseChatModel.invoke`/`ainvoke` to inject skills.

    Idempotent. Any exception inside the injection path falls back to
    the original input — a Router failure never breaks a model call.
    """
    global _skill_loader_installed
    if _skill_loader_installed:
        return
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
    except ImportError:
        logger.warning(
            "enable_skill_loader=True but langchain-core not installed; "
            "skipping skill loader install. "
            "Install it with: pip install \"decimalai[langchain]\""
        )
        return

    original_invoke = BaseChatModel.invoke
    original_ainvoke = BaseChatModel.ainvoke

    def patched_invoke(self, input, config=None, *, stop=None, **kwargs):
        try:
            input = _inject_skills_into_input(input)
        except Exception:
            logger.debug("Skill injection failed (non-fatal)", exc_info=True)
        return original_invoke(self, input, config=config, stop=stop, **kwargs)

    async def patched_ainvoke(self, input, config=None, *, stop=None, **kwargs):
        try:
            input = _inject_skills_into_input(input)
        except Exception:
            logger.debug("Skill injection failed (non-fatal)", exc_info=True)
        return await original_ainvoke(self, input, config=config, stop=stop, **kwargs)

    BaseChatModel.invoke = patched_invoke  # type: ignore[method-assign]
    BaseChatModel.ainvoke = patched_ainvoke  # type: ignore[method-assign]
    _skill_loader_installed = True
    logger.info("DecimalAI SkillRouter loader installed (LangChain)")

# Global instrument() config — stored for per-invocation handler creation
_install_agent_name: Optional[str] = None

# Last-resort trace label. The ingest API requires a trace-level
# `agent_name`; a trace that ships None is rejected with
# TRACE_VALIDATION_FAILED and lost. Named after the adapter, matching the
# defaults the other adapters already carry ("llamaindex-agent",
# "otel-agent", "claude-agent").
DEFAULT_AGENT_NAME = "langchain-agent"


def instrument(
    agent_name: Optional[str] = None,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    prompts: Optional[Dict[str, str]] = None,
    models: Optional[Dict[str, Dict[str, Any]]] = None,
    skills: Optional[List[Dict[str, Any]]] = None,
    skill_dirs: Optional[List[str]] = None,
    evals: Optional[List[Any]] = None,
    builtin_evals: bool = True,
    enable_skill_loader: bool = False,
    enable_load_skill_tool: bool = False,
    disk_sync: Optional[bool] = None,
) -> None:
    """Register DecimalAI tracing globally for all LangChain calls.

    After calling ``install()``, every ``chain.invoke()``, ``agent.invoke()``,
    and ``llm.invoke()`` call will be automatically traced and sent to the
    DecimalAI backend. No per-call ``config={"callbacks": [...]}`` needed.

    Uses LangChain's ``register_configure_hook`` — the same mechanism
    that LangSmith uses for its built-in tracing.

    Args:
        agent_name: Default agent name for all traces. If None, the name
            is auto-detected from the chain/agent being executed, and
            falls back to ``"langchain-agent"`` when the executed
            runnable carries no usable name.
        tools: Explicit tool descriptors for manifest hashing. If omitted,
            tools are auto-detected from LangChain bind_tools.
        prompts: Explicit prompt dict {"system": "..."}. If omitted,
            prompts are auto-extracted from ChatPromptTemplate.
        models: Explicit model config dict. If omitted, model is
            auto-detected from LLM invocations.

    Raises:
        ImportError: If ``langchain-core`` is not installed.

    Example::

        import decimalai
        decimalai.init()

        from decimalai.langchain import instrument
        instrument()

        # All LangChain calls are now traced — no callbacks needed
        agent.invoke({"input": "What is AAPL?"})
    """
    global _installed, _explicit_manifest_config, _evals, _builtin_evals_enabled
    global _install_agent_name

    if _installed:
        # Repeat calls do not reconfigure tracing, but the skill loader is an
        # independent, idempotent monkey-patch — honor it here so
        # `decimalai.init(langchain=True)` followed by
        # `instrument(enable_skill_loader=True)` installs the loader instead
        # of silently dropping it.
        if enable_skill_loader and not _skill_loader_installed:
            from .skill_router import _warn_if_disk_runtime_detected
            _warn_if_disk_runtime_detected("langchain")
            _install_skill_loader()
        # Any other configuration on a repeat call is dropped — say so at
        # WARNING instead of the old silent DEBUG no-op.
        ignored = [
            arg_name for arg_name, value in (
                ("tools", tools),
                ("prompts", prompts),
                ("models", models),
                ("skills", skills),
                ("skill_dirs", skill_dirs),
                ("evals", evals),
                ("disk_sync", disk_sync),
            ) if value is not None
        ]
        if agent_name is not None and agent_name != _install_agent_name:
            ignored.insert(0, "agent_name")
        if not builtin_evals:
            ignored.append("builtin_evals")
        if enable_load_skill_tool:
            ignored.append("enable_load_skill_tool")
        if ignored:
            logger.warning(
                "DecimalAI LangChain tracing already installed; ignoring %s "
                "from this instrument() call — repeat calls do not "
                "reconfigure tracing.",
                ", ".join(ignored),
            )
        else:
            logger.debug("DecimalAI LangChain tracing already installed")
        return

    try:
        from langchain_core.tracers.context import register_configure_hook
    except ImportError:
        raise ImportError(
            "langchain-core is required for instrument() but is not installed. "
            "Install the LangChain extra with: pip install \"decimalai[langchain]\""
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

    # Store explicit manifest config if provided. When disk_sync=False,
    # skip the disk-reading auto-discovery — the caller is signaling that
    # the platform is the only source of skills.
    if tools or prompts or models or skills:
        # Auto-discover skills from SKILL.md files if not explicitly provided
        resolved_skills = skills
        if disk_sync and not resolved_skills:
            try:
                from .skills import discover_skills
                resolved_skills = discover_skills(skill_dirs) or None
            except Exception:
                _warn_once_then_debug("skill_auto_discovery", "Skill auto-discovery failed")
        _explicit_manifest_config = {
            "tools": tools,
            "prompts": prompts,
            "models": models,
            "skills": resolved_skills,
        }
    elif skill_dirs and disk_sync:
        # Only skill_dirs specified — still try auto-discovery
        try:
            from .skills import discover_skills
            discovered = discover_skills(skill_dirs)
            if discovered:
                _explicit_manifest_config = {"skills": discovered}
        except Exception:
            _warn_once_then_debug("skill_auto_discovery", "Skill auto-discovery failed")

    # Store eval configuration
    _evals = list(evals or [])
    _builtin_evals_enabled = builtin_evals
    _install_agent_name = agent_name

    # Register SDK-defined evals with the platform so they appear in the UI
    # before any traces have been sent. User-supplied evals only — we exclude
    # `builtin=True` evaluators because those are platform-side concepts.
    user_evals = [e for e in _evals if not getattr(e, "builtin", False)]
    if user_evals:
        try:
            from ._config import _get_client, _sender

            def _register_evals_background():
                try:
                    client = _get_client()
                    client.register_evals(user_evals, agent_name=agent_name)
                except Exception:
                    _warn_once_then_debug(
                        "eval_registration_background",
                        "Background SDK eval registration failed (non-fatal)",
                    )

            _sender.submit(_register_evals_background)
        except Exception:
            _warn_once_then_debug(
                "eval_registration_schedule",
                "Could not schedule SDK eval registration (non-fatal)",
            )

    # Collect all discovered skills for sync (from explicit config or
    # standalone discovery). All disk-reading paths are gated on disk_sync.
    _all_discovered = None
    if _explicit_manifest_config and _explicit_manifest_config.get("skills"):
        _all_discovered = _explicit_manifest_config["skills"]

    if disk_sync and not _all_discovered and not skills and not skill_dirs:
        try:
            from .skills import discover_skills
            _all_discovered = discover_skills() or None
        except Exception:
            _warn_once_then_debug("skill_auto_discovery_standalone", "Standalone skill auto-discovery failed")

    # Sync to platform + pull from platform — both gated on disk_sync.
    # disk_sync=False = platform is the sole source/sink.
    if disk_sync:
        try:
            from ._config import _get_config, _sender

            config = _get_config()
            local_names = {s["name"] for s in _all_discovered} if _all_discovered else set()

            if _all_discovered:
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
                            _with_local_timestamps(_all_discovered),
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
                        _warn_once_then_debug("skill_sync_background", "Background skill sync failed")

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
                    _warn_once_then_debug("skill_pull_background", "Background skill pull failed (non-fatal)")

            _sender.submit(_pull_missing_background)
        except Exception:
            _warn_once_then_debug("skill_sync_setup", "Skill sync setup failed (non-fatal)")

    handler = CallbackHandler(agent_name=agent_name, auto_send=True)
    _decimal_callback_var.set(handler)
    register_configure_hook(_decimal_callback_var, inheritable=True)

    # SkillRouter dynamic loader — opt-in. When enabled,
    # BaseChatModel.invoke/ainvoke get monkey-patched so a SystemMessage
    # carrying the platform-routed skills is prepended on every call.
    #
    # enable_load_skill_tool is accepted but DORMANT here:
    # this adapter patches chat-model invoke/ainvoke — a non-loop
    # layer that cannot route a tool result back mid-turn. Skills stay
    # prompt-injected (inject_skill_body now trims + budgets bodies); the
    # live load_skill tool ships on openai_agents and pydantic_ai.
    if enable_load_skill_tool:
        logger.warning(
            "enable_load_skill_tool is not supported on the langchain adapter "
            "(invoke-layer patch, no tool loop); staying on prompt injection. "
            "Use openai_agents or pydantic_ai for the native load_skill tool."
        )
    if enable_skill_loader:
        from .skill_router import _warn_if_disk_runtime_detected
        _warn_if_disk_runtime_detected("langchain")
        _install_skill_loader()

    _installed = True
    logger.info(
        "DecimalAI LangChain tracing installed globally (agent_name=%s, skill_loader=%s, disk_sync=%s)",
        agent_name,
        enable_skill_loader,
        disk_sync,
    )



class CallbackHandler(_CallbackBase):
    """LangChain/LangGraph callback handler that captures traces for DecimalAI.

    Subclasses ``BaseCallbackHandler`` when langchain-core is installed, and falls
    back to duck typing when it is not.

    The ``ignore_*`` / ``raise_error`` / ``run_inline`` flags below are CLASS
    attributes, not assignments in ``__init__``. On the real base class they are
    read-only properties, so assigning them per instance raises AttributeError the
    moment langchain-core is present — which is exactly the case this subclassing
    exists to support.
    Traces are auto-sent to the DecimalAI backend when the root chain
    completes (requires ``decimalai.init()`` to have been called).

    Args:
        agent_name: Name of the agent (shown in the UI). If None, it is
            auto-detected from the root chain, then from the global
            ``instrument(agent_name=...)``, then ``"langchain-agent"``.
        auto_send: If True (default), automatically send the trace when
            the root chain finishes. Set False for manual control.
        session_id: Optional session grouping.
    """

    # LangChain BaseCallbackHandler protocol flags. Class-level on purpose — see
    # the note in the class docstring.
    raise_error: bool = False
    run_inline: bool = False
    ignore_llm: bool = False
    ignore_retry: bool = True
    ignore_chain: bool = False
    ignore_agent: bool = False
    ignore_retriever: bool = True
    ignore_chat_model: bool = False
    ignore_custom_event: bool = True

    def __init__(
        self,
        agent_name: Optional[str] = None,
        auto_send: bool = True,
        session_id: Optional[str] = None,
        project: Optional[str] = None,
        parent_trace_id: Optional[str] = None,
        subagents: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            parent_trace_id: When this agent runs as a sub-agent of another,
                pass the parent agent's trace ID. The backend uses this to
                link the child trace to the parent for the multi-agent
                debugging surfaces — Subagent Health, Delegation Analytics,
                Topology Graph.
            subagents: When this agent has sub-agents it delegates to,
                declare them here as [{"name": "child-agent-name"}]. The
                names are registered as subagent components in this agent's
                manifest, which lets the backend resolve `is_subagent=True`
                on the child agents.
        """
        self.agent_name = agent_name
        self.auto_send = auto_send
        self.session_id = session_id
        self.project = project
        self.parent_trace_id = parent_trace_id
        self.subagents = list(subagents) if subagents else None

        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all trace-building state."""
        self._trace_id: UUID = uuid4()
        self._spans: Dict[UUID, TraceSpan] = {}
        self._llm_calls: Dict[UUID, LlmCallRecord] = {}
        self._tool_calls: Dict[UUID, ToolCallRecord] = {}
        self._span_stack: List[UUID] = []
        self._trace_started_at: Optional[datetime] = None
        self._user_input_preview: Optional[str] = None
        self._final_output_preview: Optional[str] = None
        self._root_run_id: Optional[UUID] = None
        # Manifest auto-detection accumulators
        self._seen_tools: Dict[str, Dict[str, Any]] = {}  # name -> {schema, ...}
        self._seen_model: Optional[Dict[str, Any]] = None
        self._seen_prompts: Dict[str, str] = {}  # role -> text
        self._seen_output_contract: Optional[Dict[str, Any]] = None
        self._streaming_buffers: Dict[UUID, List[str]] = {}  # token buffers for streaming  # response_format
        # Skill activation tracking
        self._active_skills: Dict[str, Optional[str]] = {}
        # Skill Rater discovery telemetry. `_skills_offered_in_prompt`
        # is auto-populated from the Router's offered set via the
        # BaseChatModel.invoke patch; `_skills_loaded_by_agent` is a manual
        # hook (use `decimalai.log_skill_loaded` or the explicit
        # `CallbackHandler.log_skill_loaded` method below).
        self._skills_offered_in_prompt: set[str] = set()
        self._skills_loaded_by_agent: set[str] = set()
        # Bodies that reached the model (Router body injection) —
        # between offered and activated; never implies activation.
        self._skills_delivered: set[str] = set()

    def log_skill_offered(self, *, names: List[str]) -> None:
        """Manually record skills that were offered in the system prompt."""
        for name in names:
            if isinstance(name, str) and name.strip():
                self._skills_offered_in_prompt.add(name.strip())

    def log_skill_delivered(self, *, names: List[str]) -> None:
        """Record skills whose full body reached the model. Implies offered."""
        for name in names:
            if isinstance(name, str) and name.strip():
                n = name.strip()
                self._skills_delivered.add(n)
                self._skills_offered_in_prompt.add(n)

    def log_skill_loaded(self, *, name: str) -> None:
        """Manually record that the agent read a skill's body. Implies offered+delivered."""
        if isinstance(name, str) and name.strip():
            n = name.strip()
            self._skills_loaded_by_agent.add(n)
            self._skills_offered_in_prompt.add(n)
            self._skills_delivered.add(n)

    # ── Chain lifecycle (spans) ─────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain or graph node starts."""
        serialized = serialized or {}

        # Extract name — prefer kwargs["name"] (LangChain's `run_name` override
        # from with_config) over serialized type. Without this precedence,
        # `(prompt | llm).with_config(run_name="X")` is silently filtered out
        # by the SKIP list below because the serialized id resolves to
        # "RunnableSequence". User intent: run_name wins.
        name = kwargs.get("name", "") or serialized.get("name", "")
        if not name:
            id_list = serialized.get("id", [])
            name = id_list[-1] if id_list else ""

        # Skip noisy internal LangChain wrappers
        if any(name.startswith(skip) for skip in _SKIP_CHAIN_TYPES):
            return

        if not name:
            name = "chain"

        span_id = run_id or uuid4()
        parent_id = parent_run_id if parent_run_id in self._spans else None

        # Track the root run so state resets for each root invocation
        # This ensures concurrent invocations don't interleave spans
        if self._root_run_id is None:
            if parent_run_id is None or parent_run_id not in self._spans:
                # This is a new root invocation — reset all state
                old_agent = self.agent_name
                old_session = self.session_id
                self._reset_state()
                self.agent_name = old_agent
                self.session_id = old_session
            self._root_run_id = span_id

        # Auto-detect span type
        span_type = SpanType.AGENT
        name_lower = name.lower()
        if "tool" in name_lower:
            span_type = SpanType.TOOL
        elif "prompt" in name_lower or "template" in name_lower:
            span_type = SpanType.OTHER

        span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=span_type,
            name=str(name),
            status=Status.RUNNING,
            started_at=datetime.now(timezone.utc),
            input_preview=_preview(inputs),
        )
        self._spans[span_id] = span
        self._span_stack.append(span_id)

        # Capture first input as trace-level preview
        if self._trace_started_at is None:
            self._trace_started_at = span.started_at
            self._user_input_preview = _preview(inputs)

        # Auto-detect agent_name from root chain if not explicitly set
        if self.agent_name is None and parent_run_id is None:
            detected = str(name)
            if detected and detected not in ('chain', 'unknown'):
                self.agent_name = detected

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain or graph node ends."""
        span_id = run_id
        if span_id and span_id in self._spans:
            span = self._spans[span_id]
            span.status = Status.SUCCESS
            span.ended_at = datetime.now(timezone.utc)
            span.output_preview = _preview(outputs)

        if self._span_stack and self._span_stack[-1] == span_id:
            self._span_stack.pop()

        self._final_output_preview = _preview(outputs)

        # Auto-send at the true outermost end — the only callback that
        # carries parent_run_id=None. Matching on `span_id ==
        # self._root_run_id` broke on langchain-core 1.5.x, which reuses the
        # root run_id for child steps (ChatPromptTemplate's events arrive
        # with run_id == parent_run_id == root), so the old check fired at
        # the PROMPT step's end and sent the trace before the LLM call
        # existed. The emptiness guard keeps all-skipped runs (e.g. a bare
        # RunnablePassthrough) from sending empty traces.
        if parent_run_id is None and self.auto_send and (self._spans or self._llm_calls):
            self._auto_send()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        span_id = run_id
        if span_id and span_id in self._spans:
            span = self._spans[span_id]
            span.status = Status.ERROR
            span.ended_at = datetime.now(timezone.utc)
            span.output_preview = str(error)[:200]

        if self._span_stack and self._span_stack[-1] == span_id:
            self._span_stack.pop()

        # Auto-send on the outermost error too — same parent_run_id=None
        # boundary as on_chain_end (see the note there).
        if parent_run_id is None and self.auto_send and (self._spans or self._llm_calls):
            self._auto_send()

    # ── Sub-agent resolution ───────────────────────────────

    def _agent_name_or_default(self) -> str:
        """The name this handler's records ship under — never None.

        `on_chain_start`'s auto-detection cannot fire when the root
        runnable is one of `_SKIP_CHAIN_TYPES` — `prompt | llm` is a
        RunnableSequence, so the callback returns before the detection
        block — nor for a bare `llm.invoke()`, which emits no chain
        callback at all. Both left `agent_name` None all the way into the
        payload, and the ingest API rejects that: every LCEL trace from
        `init(langchain=True)` was dropped with TRACE_VALIDATION_FAILED.

        Resolution is deliberately NOT written back to `self.agent_name`:
        that field staying None is what lets a later root chain with a real
        name still be auto-detected.
        """
        return self.agent_name or _install_agent_name or DEFAULT_AGENT_NAME

    def _resolve_agent_name(self, parent_run_id: Optional[UUID] = None) -> Optional[str]:
        """Resolve the agent name for an LLM call by walking parent spans.

        Finds the nearest ancestor span of type AGENT and returns its name.
        Falls back to ``_agent_name_or_default()`` if no agent-type parent
        is found, preserving backward compatibility for the single-agent
        case.

        The walk is bounded. If `_spans` ever contains a parent_span_id
        cycle (malformed `parent_run_id` from LangChain, or a custom
        callback corrupting `_spans`), the old `while span_id and
        span_id in self._spans` looped forever, blocking the LangChain
        dispatcher thread. Two guards: a `seen` set so we break on
        cycles, and a hop cap of 32 — agent topologies don't go deeper
        than ~10 in practice.
        """
        span_id = parent_run_id
        seen: set[UUID] = set()
        for _ in range(32):
            if not span_id or span_id in seen or span_id not in self._spans:
                break
            seen.add(span_id)
            span = self._spans[span_id]
            if span.span_type == SpanType.AGENT and span.name:
                return span.name
            span_id = span.parent_span_id
        return self._agent_name_or_default()

    # ── Agent lifecycle (no-op, suppresses warnings) ────────

    def on_agent_action(self, action: Any, *, run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        pass

    def on_agent_finish(self, finish: Any, *, run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        pass

    # ── LLM lifecycle ──────────────────────────────────────

    def on_llm_new_token(self, token: str, *, run_id: Optional[UUID] = None, **kwargs: Any) -> None:
        """Buffer streaming tokens per LLM call."""
        if run_id is None:
            return
        if run_id not in self._streaming_buffers:
            self._streaming_buffers[run_id] = []
        self._streaming_buffers[run_id].append(token)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call starts."""
        call_id = run_id or uuid4()
        params = invocation_params or {}

        rendered_input = [{"role": "user", "content": p} for p in prompts]

        call = LlmCallRecord(
            id=call_id,
            span_id=parent_run_id,
            agent_name=self._resolve_agent_name(parent_run_id),
            provider=extract_provider(params, serialized),
            model_name=extract_model_name(params),
            temperature=params.get("temperature"),
            max_output_tokens=params.get("max_tokens"),
            rendered_input=rendered_input,
            status=Status.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        self._llm_calls[call_id] = call

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        invocation_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model starts."""
        call_id = run_id or uuid4()
        params = invocation_params or {}

        rendered_input = []
        for msg_list in messages:
            for msg in msg_list:
                rendered_input.append({
                    "role": normalize_role(msg),
                    "content": extract_message_content(msg),
                })

        call = LlmCallRecord(
            id=call_id,
            span_id=parent_run_id,
            agent_name=self._resolve_agent_name(parent_run_id),
            provider=extract_provider(params, serialized),
            model_name=extract_model_name(params),
            temperature=params.get("temperature"),
            max_output_tokens=params.get("max_tokens"),
            rendered_input=rendered_input,
            status=Status.RUNNING,
            started_at=datetime.now(timezone.utc),
        )

        # Detect multi-modal content from messages
        content_type = "text"
        for msg_list in messages:
            for msg in msg_list:
                content = extract_message_content(msg)
                if isinstance(content, list):
                    # Multi-part message (e.g. [{"type": "text", ...}, {"type": "image_url", ...}])
                    part_types = {p.get("type", "text") if isinstance(p, dict) else "text" for p in content}
                    if "image_url" in part_types or "image" in part_types:
                        content_type = "image" if part_types == {"image_url"} or part_types == {"image"} else "multimodal"
                    elif "input_audio" in part_types or "audio" in part_types:
                        content_type = "audio" if len(part_types) == 1 else "multimodal"
        call.content_type = content_type

        # Capture response_format from invocation params (structured output)
        resp_fmt = params.get("response_format")
        if resp_fmt and isinstance(resp_fmt, dict):
            call.response_format = resp_fmt
        elif resp_fmt and hasattr(resp_fmt, "model_json_schema"):
            # Pydantic model class — capture its JSON schema
            try:
                call.response_format = {
                    "type": "json_schema",
                    "json_schema": resp_fmt.model_json_schema(),
                }
            except Exception:
                pass

        self._llm_calls[call_id] = call

        # Auto-detect model and prompts for manifest
        provider = extract_provider(params, serialized)
        model_name = extract_model_name(params)
        if model_name and not self._seen_model:
            self._seen_model = {
                "provider": provider or "unknown",
                "model": model_name,
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                "top_p": params.get("top_p"),
            }

        # Capture structured output schema (response_format) as output_contract
        resp_fmt = params.get("response_format")
        if resp_fmt and isinstance(resp_fmt, dict) and not self._seen_output_contract:
            self._seen_output_contract = resp_fmt
        # Capture system/human prompts from chat messages (auto-detection)
        # NOTE: This captures the RENDERED prompt, not the template.
        # If prompts include dynamic content (RAG chunks, dates, few-shot
        # examples), the hash will change on every run, causing false drift.
        # Users with dynamic prompts should use:
        #   install(prompts={"system": "Your static template text..."})
        for msg_list in messages:
            for msg in msg_list:
                role = normalize_role(msg)
                content = extract_message_content(msg)
                if role in ("system", "developer") and content and role not in self._seen_prompts:
                    self._seen_prompts[role] = content
                elif role in ("system", "developer") and content and role in self._seen_prompts:
                    # Prompt changed within the same trace — likely dynamic
                    if content != self._seen_prompts[role]:
                        logger.warning(
                            "Auto-detected %s prompt changed within trace. "
                            "If your prompts include dynamic content (RAG, dates, "
                            "few-shot), pass static templates via install(prompts=...) "
                            "to avoid false manifest drift.",
                            role,
                        )

        # Auto-detect skill activations from system prompts
        if self._seen_prompts:
            self._detect_skills_from_prompts()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call ends."""
        call_id = run_id
        if not call_id or call_id not in self._llm_calls:
            return

        call = self._llm_calls[call_id]
        call.status = Status.SUCCESS
        call.ended_at = datetime.now(timezone.utc)
        if call.started_at:
            call.latency_ms = int(
                (call.ended_at - call.started_at).total_seconds() * 1000
            )

        output = extract_output_dict(response)
        if output:
            call.output = output

        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0][0] if response.generations[0] else None
            if gen and hasattr(gen, "message"):
                if has_tool_calls(gen.message):
                    call.finish_reason = FinishReason.TOOL_CALLS
                else:
                    call.finish_reason = FinishReason.STOP

        input_tokens, output_tokens = extract_token_usage(response)
        call.input_tokens = input_tokens
        call.output_tokens = output_tokens

        # Handle streaming buffer — join buffered tokens
        if call_id in self._streaming_buffers:
            tokens = self._streaming_buffers.pop(call_id)
            if tokens:
                call.streaming = True
                call.streaming_token_count = len(tokens)
                # If output wasn't set from response, use joined tokens
                if not call.output or not call.output.get("content"):
                    call.output = call.output or {}
                    call.output["streaming_content"] = "".join(tokens)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call errors."""
        call_id = run_id
        if call_id and call_id in self._llm_calls:
            call = self._llm_calls[call_id]
            call.status = Status.ERROR
            call.ended_at = datetime.now(timezone.utc)
            call.output = {"error": str(error)[:500]}
            call.finish_reason = FinishReason.ERROR
        # Drain the streaming buffer for this call so a mid-stream
        # error doesn't leak the buffered tokens for the lifetime of the
        # handler. `on_llm_end` already pops on the happy path; the error
        # path used to skip the pop, which was a multi-hour leak in
        # long-lived processes that hit many streaming failures (flaky
        # provider, timeout, content filter).
        if call_id is not None:
            self._streaming_buffers.pop(call_id, None)

    # ── Tool lifecycle ─────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts."""
        tool_id = run_id or uuid4()
        tool_name = serialized.get("name", "unknown")

        span = TraceSpan(
            id=tool_id,
            parent_span_id=parent_run_id,
            span_type=SpanType.TOOL,
            name=str(tool_name),
            status=Status.RUNNING,
            started_at=datetime.now(timezone.utc),
            input_preview=str(input_str)[:200],
        )
        self._spans[tool_id] = span

        tool_call = ToolCallRecord(
            id=tool_id,
            tool_name=str(tool_name),
            args={"input": input_str} if isinstance(input_str, str) else {},
            status=Status.RUNNING,
        )
        self._tool_calls[tool_id] = tool_call

        # Auto-detect tools for manifest.
        # Include `description` so that manifest diffs and tool-impact
        # analysis see description changes as manifest deltas. Without it,
        # rewriting a tool's description — "Search the web" → "Search the
        # corporate intranet" — produces NO manifest signal, even though
        # it can completely change how the model uses the tool.
        if str(tool_name) not in self._seen_tools:
            self._seen_tools[str(tool_name)] = {
                "name": str(tool_name),
                "description": serialized.get("description", ""),
                "schema": serialized.get("schema"),
            }

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool ends."""
        tool_id = run_id
        if tool_id and tool_id in self._spans:
            span = self._spans[tool_id]
            span.status = Status.SUCCESS
            span.ended_at = datetime.now(timezone.utc)
            span.output_preview = str(output)[:200]

        if tool_id and tool_id in self._tool_calls:
            tc = self._tool_calls[tool_id]
            tc.status = Status.SUCCESS
            tc.result = str(output)[:1000]

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        tool_id = run_id
        if tool_id and tool_id in self._spans:
            span = self._spans[tool_id]
            span.status = Status.ERROR
            span.ended_at = datetime.now(timezone.utc)

        if tool_id and tool_id in self._tool_calls:
            tc = self._tool_calls[tool_id]
            tc.status = Status.ERROR

    # ── Retriever + text (no-op) ───────────────────────────

    def on_retriever_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_retriever_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_retriever_error(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_text(self, *args: Any, **kwargs: Any) -> None:
        pass

    # ── Build & send ───────────────────────────────────────

    def build_trace(self) -> RunTrace:
        """Assemble the collected spans and LLM calls into a RunTrace."""
        from . import _config

        config = _config._config

        # Attach tool calls to their parent LLM calls
        for tc in self._tool_calls.values():
            tool_span = self._spans.get(tc.id)
            if tool_span and tool_span.parent_span_id:
                for lc in self._llm_calls.values():
                    if lc.span_id == tool_span.parent_span_id:
                        lc.tool_calls.append(tc)
                        break

        # Build active_skills list
        active_skills_list: List[Dict[str, Any]] = []
        for name, h in self._active_skills.items():
            entry: Dict[str, Any] = {"name": name}
            if h:
                entry["hash"] = h
            active_skills_list.append(entry)

        # Drain the Router's instance rails first, unconditionally — see
        # `_drain_router_rails`. Unconditional because an undrained rail
        # leaks into the NEXT trace.
        rail_routing_id, rail_offered, rail_delivered, rail_loaded = _drain_router_rails()

        # Drain the offered-names contextvar populated by the
        # BaseChatModel patch; merge with any direct log_skill_offered calls.
        drained_offered = _consume_skills_offered()
        if drained_offered:
            for n in drained_offered:
                self._skills_offered_in_prompt.add(n)
        for n in rail_offered:
            self._skills_offered_in_prompt.add(n)

        # Drain the delivered-names contextvar (Router body injection).
        drained_delivered = _consume_skills_delivered()
        if drained_delivered:
            for n in drained_delivered:
                self._skills_delivered.add(n)
                self._skills_offered_in_prompt.add(n)  # delivered implies offered
        for n in rail_delivered:
            self._skills_delivered.add(n)
            self._skills_offered_in_prompt.add(n)

        # Bodies served by the singleton's `load_skill(...)` — this adapter
        # registers no native load_skill tool, but a user-supplied tool
        # that calls it still lands its serves here.
        for n in rail_loaded:
            self.log_skill_loaded(name=n)

        # Derive trace status from collected spans/LLM calls: if any errored,
        # the run errored. on_chain_error/on_llm_error mark these ERROR.
        trace_status = Status.SUCCESS
        if any(s.status == Status.ERROR for s in self._spans.values()) or any(
            lc.status == Status.ERROR for lc in self._llm_calls.values()
        ):
            trace_status = Status.ERROR

        return RunTrace(
            id=self._trace_id,
            project=config.project if config else None,
            agent_name=self._agent_name_or_default(),
            session_id=self.session_id,
            parent_trace_id=self.parent_trace_id,
            status=trace_status,
            source_type="production",
            started_at=self._trace_started_at,
            ended_at=datetime.now(timezone.utc),
            user_input_preview=self._user_input_preview,
            final_output_preview=self._final_output_preview,
            spans=list(self._spans.values()),
            llm_calls=list(self._llm_calls.values()),
            active_skills=active_skills_list,
            manifest_id=_manifest_id,
            # SkillRouter: stamp the routing_id set by the BaseChatModel
            # monkey-patch so the offered-vs-activated join can close. The
            # rail is the fallback: LangChain runs callbacks under
            # `copy_context()`, so the patch's contextvar write never
            # reaches this build.
            routing_id=_consume_routing_id() or rail_routing_id,
            # Skill Rater discovery telemetry. Sorted for
            # deterministic output (tests, diffs).
            skills_offered_in_prompt=sorted(self._skills_offered_in_prompt),
            skills_loaded_by_agent=sorted(self._skills_loaded_by_agent),
            skills_delivered=sorted(self._skills_delivered),
        )

    def get_trace(self) -> RunTrace:
        """Build the trace and reset for the next invocation."""
        trace = self.build_trace()
        self._reset_state()
        return trace

    def get_trace_id(self) -> str:
        """Return this handler's current trace ID.

        Useful when an orchestrator's tool needs to spawn a child agent
        and pass `parent_trace_id` into the child's CallbackHandler so
        the backend can link the resulting traces.
        """
        return str(self._trace_id)

    def _auto_send(self) -> None:
        """Send the trace via the background sender (called on root chain end)."""
        from . import _config

        if not _config._is_enabled():
            logger.debug("Tracing disabled, skipping auto-send")
            return

        # Auto-register manifest on first trace (or when agent changes)
        self._maybe_register_manifest()

        try:
            client = _config._get_client()
            trace = self.get_trace()

            # Run evals before sending
            eval_scores = self._run_evals(trace)
            if eval_scores:
                # Attach eval scores to trace payload
                trace.eval_scores = eval_scores

            # Use background sender for non-blocking send
            _config._sender.submit(client.ingest_trace, trace)
            logger.debug(
                "Queued trace %s (%d spans, %d llm_calls, %d eval_scores, manifest=%s)",
                trace.id,
                len(trace.spans),
                len(trace.llm_calls),
                len(eval_scores),
                trace.manifest_id or "none",
            )
        except Exception:
            logger.exception("Failed to queue trace %s", self._trace_id)
            self._reset_state()
    def _run_evals(self, trace: RunTrace) -> List[Dict[str, Any]]:
        """Run all registered evals against the trace."""
        from .evals import DecimalEval, run_evals, trace_to_trace_data
        from .evals.builtin import BUILTIN_EVALS

        all_evals: List[DecimalEval] = []

        # Add built-in evals if enabled
        if _builtin_evals_enabled:
            all_evals.extend(BUILTIN_EVALS)

        # Add user-defined evals
        all_evals.extend(_evals)

        if not all_evals:
            return []

        try:
            trace_data = trace_to_trace_data(trace)
            scores = run_evals(trace_data, all_evals)
            if scores:
                logger.debug("Ran %d evals, produced %d scores", len(all_evals), len(scores))
            return scores
        except Exception as e:
            logger.warning("Eval execution failed: %s", e)
            return []

    def _detect_skills_from_prompts(self) -> None:
        """Auto-detect skill activations from system/developer prompts.

        Uses the global skills registry (from instrument()) to match
        skill references in the rendered system prompt text.
        """
        skills_registry = (_explicit_manifest_config or {}).get("skills")
        if not skills_registry:
            return

        try:
            from .skills import detect_skill_activations
            # Build system text from seen prompts
            system_text = "\n".join(self._seen_prompts.values())
            if not system_text:
                return

            detected = detect_skill_activations(
                [{"role": "system", "content": system_text}],
                skills_registry,
            )
            for skill_name in detected:
                if skill_name not in self._active_skills:
                    registry_hash = next(
                        (s.get("hash") for s in skills_registry
                         if s.get("name") == skill_name),
                        None,
                    )
                    self._active_skills[skill_name] = registry_hash
        except Exception:
            _warn_once_then_debug(
                "skill_activation_detection",
                "Skill auto-detection from prompts failed",
            )

    def _maybe_register_manifest(self) -> None:
        """Extract and register manifest if not already done.

        Thread-safe via _manifest_lock.
        """
        global _manifest_id

        from . import _config
        if not _config._is_enabled():
            return

        # Use explicit config from instrument() if provided
        tools = None
        prompts = None
        models = None

        if _explicit_manifest_config:
            tools = _explicit_manifest_config.get("tools")
            prompts = _explicit_manifest_config.get("prompts")
            models = _explicit_manifest_config.get("models")

        # Fall back to auto-detected values
        if not tools and self._seen_tools:
            tools = list(self._seen_tools.values())
        if not prompts and self._seen_prompts:
            prompts = dict(self._seen_prompts)
        if not models and self._seen_model:
            models = {"default": self._seen_model}

        # Include output contract if detected
        output_contract = None
        if self._seen_output_contract:
            output_contract = self._seen_output_contract

        # Need at least something to register
        skills = (_explicit_manifest_config or {}).get("skills")
        subagents = self.subagents
        if not tools and not prompts and not models and not skills and not subagents:
            return

        # Same label the trace ships under, or the manifest lands on a
        # different agent than the traces that reference its manifest_id.
        agent_name = self._agent_name_or_default()
        snapshot = extract_from_config(
            agent_name=agent_name,
            tools=tools,
            prompts=prompts,
            models=models,
            subagents=subagents,
            output_schema=output_contract,
            skills=skills,
        )

        # Thread-safe manifest registration
        with _manifest_lock:
            # Check if manifest changed
            if not _manifest_tracker.check_and_update(snapshot):
                return  # Same hash — already registered

            try:
                client = _config._get_client()
                result = client.register_manifest(snapshot)
                _manifest_id = result.get("manifest_id", snapshot.id)
                logger.info(
                    "Registered manifest %s (hash=%s, components=%d)",
                    _manifest_id,
                    snapshot.manifest_hash[:12],
                    len(snapshot.components),
                )
            except Exception:
                logger.warning("Failed to register manifest, continuing without", exc_info=True)
                # Still use local ID so traces get tagged
                _manifest_id = snapshot.id

    # ── Backwards compatibility ────────────────────────────

    def reset(self) -> None:
        """Reset state (alias for backwards compat)."""
        self._reset_state()

    def get_completed_trace(self) -> RunTrace:
        """Alias for get_trace() (backwards compat)."""
        return self.get_trace()


def _preview(obj: Any, max_len: int = 2000) -> str:
    """Create a JSON-friendly preview string from any LangChain object.

    Why this exists: prior versions used `str(obj)`, which for LangChain
    objects produces Python `repr()` output like
    ``{'messages': [HumanMessage(content='...')]}``. The dashboard then has
    no way to pretty-print that because it's not valid JSON.

    Strategy:
      1. If the value is a dict containing 'messages' (the typical
         LangChain/LangGraph callback shape), extract the latest message's
         content text — that's what the user actually wants to read.
      2. Otherwise, try JSON serialization with a permissive default that
         coerces unknown objects via their ``dict()`` / ``__dict__`` / repr.
      3. As a last resort, fall back to ``str(obj)``.

    Truncates to ``max_len`` chars (default 2000 — the dashboard's
    input_preview / output_preview columns are TEXT, not bounded varchar).
    """
    import json as _json

    if obj is None:
        return ""

    # Shape: {"messages": [...]} → return the latest message's readable content
    if isinstance(obj, dict) and isinstance(obj.get("messages"), list) and obj["messages"]:
        last = obj["messages"][-1]
        # LangChain BaseMessage has .content; dict messages have ["content"]
        content = getattr(last, "content", None)
        if content is None and isinstance(last, dict):
            content = last.get("content")
        if content is not None:
            # If content is itself a list of content blocks (multimodal), join them
            if isinstance(content, list):
                parts = [str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content]
                content = "\n".join(parts)
            text = str(content).strip()
            if text:
                return text[:max_len]

    # Try JSON with a coercive default
    def _coerce(o: Any) -> Any:
        if hasattr(o, "model_dump"):
            try:
                return o.model_dump()
            except Exception:
                pass
        if hasattr(o, "dict") and callable(o.dict):
            try:
                return o.dict()
            except Exception:
                pass
        if hasattr(o, "__dict__"):
            return o.__dict__
        return repr(o)

    try:
        s = _json.dumps(obj, default=_coerce, ensure_ascii=False)
        return s[:max_len]
    except Exception:
        s = str(obj)
        return s[:max_len]


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
        "decimalai.langchain.install() is deprecated; use "
        "decimalai.langchain.instrument() instead. It turns on tracing for langchain "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
