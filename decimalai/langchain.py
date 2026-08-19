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

import logging
import threading
import warnings
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
    extract_tool_call_names,
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

# ContextVar for global callback registration. Defined at the bottom of this
# module, once `CallbackHandler` exists — its DEFAULT has to be the global
# handler, and a ContextVar's default is fixed at construction. See
# `_publish_handler` for why that matters (it is the whole of the
# worker-thread bug).

_installed = False

# Global manifest state — shared across all handler instances
_manifest_tracker = ManifestTracker()
_manifest_id: Optional[str] = None  # Most recent successful registration
_manifest_lock = threading.Lock()  # Thread safety for manifest registration

# Per-AGENT manifest state. `_manifest_id` alone is a single process-global
# slot, and `_manifest_tracker` a single hash slot whose hash does NOT
# include the agent name — so in a process running two agents, the second
# agent's traces were stamped with the FIRST agent's manifest, and two agents
# with the same structure deduped against each other so the second never
# registered at all. Both are keyed by agent_name here; `_manifest_id` is kept
# as the last-registered value for callers (and tests) that read it.
_manifest_ids: Dict[str, str] = {}
_manifest_hashes: Dict[str, str] = {}
_explicit_manifest_config: Optional[Dict[str, Any]] = None  # From instrument() kwargs
# Agent names we have already asked the platform about on the
# "nothing to declare" path (see `_adopt_active_manifest`). One probe per
# agent per process, hit or miss.
_manifest_adoption_probed: set[str] = set()

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


# One model call's identity, as a Router rail scope key. Minted by
# `_open_call_rails` and read back by `_capture_call_rails` in the same
# Context, so it names the call rather than merely being recent.
#
# It exists for the one case LangChain's own run ids cannot cover: a bare
# `llm.invoke()` injects its skills BEFORE the callback manager has minted a
# run id, so `_ambient_run_scope` has nothing else to answer with, and a
# routing decision filed under no owner at all is one a concurrent run could
# claim.
_call_scope_ctx: ContextVar[Optional[str]] = ContextVar(
    "decimalai_skill_router_call_scope_langchain", default=None,
)


def _open_call_rails() -> tuple:
    """Start ONE model call's skills rails, and return the reset tokens.

    Every rail above is a ContextVar the injection writes and the handler
    reads back inside the same model call (see
    `CallbackHandler._capture_call_rails`). Scoping them to the call — set on
    the way in, `reset()` on the way out — is what makes the read
    attributable rather than merely recent:

      * nothing can be READ IN from an earlier call that happened to run on
        this thread (a pooled worker keeps its Context between tasks), and
      * nothing LEAKS OUT to whatever runs next, so a later trace cannot pick
        up a routing decision that was never made for it.

    Both directions were real: the rails were previously drained off the
    Router singleton at trace-build time, which is process-global state, and
    concurrent runs took each other's.
    """
    return (
        _routing_id_ctx.set(None),
        _skills_offered_ctx.set(None),
        _skills_delivered_ctx.set(None),
        _call_scope_ctx.set(uuid4().hex),
    )


def _close_call_rails(tokens: tuple) -> None:
    """End the model call started by `_open_call_rails`."""
    for var, token in zip(
        (_routing_id_ctx, _skills_offered_ctx, _skills_delivered_ctx,
         _call_scope_ctx),
        tokens,
    ):
        try:
            var.reset(token)
        except ValueError:  # pragma: no cover - token from another Context
            var.set(None)


def _registered_manifest_id(result: Any, snapshot: Any) -> str:
    """The id to stamp on traces after a register call.

    Guards the response shape. `manifest_id` goes straight into
    `RunTrace.manifest_id`, which pydantic types as `Optional[str]`, so a
    response that omits the key or answers with a non-string used to raise
    inside `build_trace` — and that exception is caught one frame up as
    "failed to queue trace", i.e. the run is lost with a misleading reason.
    Falling back to the snapshot's own id keeps traces flowing.
    """
    manifest_id = result.get("manifest_id") if isinstance(result, dict) else None
    if isinstance(manifest_id, str) and manifest_id:
        return manifest_id
    return str(snapshot.id)


def _forget_manifests_if_tracker_reset() -> None:
    """Honour a caller that swapped in a fresh ``ManifestTracker``.

    Resetting the tracker is how a test fixture (and `reset()`) says "forget
    what this process has registered". The per-agent maps have to hear it too,
    or the reset is a no-op for every agent already in them. Caller holds
    ``_manifest_lock``.
    """
    if _manifest_tracker.last_hash is None and (_manifest_ids or _manifest_hashes):
        _manifest_ids.clear()
        _manifest_hashes.clear()


def _register_snapshot(agent_name: str, snapshot: Any) -> Optional[str]:
    """Register one agent's manifest snapshot, returning the id to stamp.

    Dedup is keyed by (agent, hash), never by hash alone: two agents in one
    process routinely have the SAME structure (same model, same tools, a
    different name), and a single-slot tracker made the second one's snapshot
    look like a repeat — it was never registered, so its traces had no
    manifest of their own to carry.
    """
    global _manifest_id

    from . import _config

    with _manifest_lock:
        _forget_manifests_if_tracker_reset()
        known = _manifest_ids.get(agent_name)
        if known and _manifest_hashes.get(agent_name) == snapshot.manifest_hash:
            return known  # Same agent, same structure — already registered.
        # Keep the legacy single slot warm: it is this module's public-ish
        # "what did we last register" surface, and `reset()` on it is what
        # `_forget_manifests_if_tracker_reset` listens for.
        _manifest_tracker.check_and_update(snapshot)
        try:
            client = _config._get_client()
            result = client.register_manifest(snapshot)
            manifest_id = _registered_manifest_id(result, snapshot)
            logger.info(
                "Registered manifest %s for %s (hash=%s, components=%d)",
                manifest_id,
                agent_name,
                snapshot.manifest_hash[:12],
                len(snapshot.components),
            )
        except Exception:
            logger.warning(
                "Failed to register manifest for %s, continuing without",
                agent_name, exc_info=True,
            )
            # Still tag traces with the local id rather than shipping none.
            manifest_id = str(snapshot.id)
        _manifest_ids[agent_name] = manifest_id
        _manifest_hashes[agent_name] = snapshot.manifest_hash
        _manifest_id = manifest_id
        return manifest_id


def _adopt_active_manifest(agent_name: str) -> Optional[str]:
    """Return the id of the agent's currently-active manifest, if any.

    A process that starts up with nothing to declare (a worker that only
    runs pure-Python chains, a re-deployed replica) must not register an
    empty manifest over the contract a sibling process already declared —
    the diff would read the absent surfaces as deletions. Asking the
    platform is what makes the "never regress" rule hold across processes
    and restarts, not just within one.

    Best-effort: any failure returns None and the caller falls back to
    registering the placeholder.
    """
    from . import _config

    try:
        client = _config._get_client()
        resp = client.list_manifests(limit=5, agent_name=agent_name)
        rows = resp.get("manifests") if isinstance(resp, dict) else None
        for row in rows or []:
            if isinstance(row, dict) and row.get("status") == "active" and row.get("id"):
                return str(row["id"])
    except Exception:
        logger.debug(
            "Could not look up an existing manifest for %s (non-fatal)",
            agent_name, exc_info=True,
        )
    return None


def _clean_names(names: Any) -> List[str]:
    """Keep only non-blank strings — same filter as `_add_skills_offered`."""
    if not isinstance(names, (list, tuple, set)):
        return []
    return [n for n in names if isinstance(n, str) and n.strip()]


# Resolved once. `_ambient_run_scope` runs on every routing call and every
# body load, so the import lookup does not belong inside it. `False` means
# "looked and it isn't there" — distinct from `None`, which is the unresolved
# state.
_lc_config_var_cache: Any = None


def _lc_config_var() -> Any:
    """LangChain's own per-runnable config ContextVar, or None."""
    global _lc_config_var_cache
    if _lc_config_var_cache is None:
        try:
            from langchain_core.runnables.config import var_child_runnable_config
            _lc_config_var_cache = var_child_runnable_config
        except Exception:  # pragma: no cover - langchain-core not installed
            _lc_config_var_cache = False
    return _lc_config_var_cache or None


def _ambient_run_scope() -> Optional[str]:
    """The LangChain run executing right now, as a Router rail scope key.

    This is the write half of the per-run fix. The Router calls it (see
    ``skill_router.register_ambient_scope_resolver``) whenever it is about
    to file a routing decision or a body load and the caller passed no
    ``scope=`` of its own — which on this adapter is always, because
    LangChain registers no native ``load_skill`` tool and the call arrives
    from arbitrary user tool code.

    The key is LangChain's OWN run identity, read off the config ContextVar
    LangChain itself maintains for the currently-executing runnable. That
    matters twice over: it is set inside the runnable's context, which is
    exactly where the injection and the tool body run, and every id it can
    answer with is one this handler has already seen as a callback ``run_id``
    or ``parent_run_id`` — so ``_drain_scoped_router_rails`` can recognise it
    as a member of THIS run rather than having to trust that nobody else was
    writing at the same time.

    It is deliberately NOT the root run id: resolving a root would need a
    registry of live handlers to look the id up in, and there isn't one. The
    reader closes that gap instead, by draining every member id it owns.

    When LangChain has no run id yet — a bare ``llm.invoke()`` injects before
    the callback manager mints one — the per-call token `_open_call_rails`
    put in this Context answers instead, and `_capture_call_rails` records it
    on the run it belongs to. Preferring the LangChain id where one exists is
    deliberate: it is the identity the handler indexes runs by, so it needs
    no second bookkeeping step to be recognised.

    None means "no LangChain run and no model call on the stack" — Router
    traffic from outside a run entirely, which no trace can honestly claim.
    """
    config_var = _lc_config_var()
    if config_var is not None:
        config = config_var.get()
        if isinstance(config, dict):
            parent_run_id = getattr(config.get("callbacks"), "parent_run_id", None)
            if parent_run_id is not None:
                return str(parent_run_id)
    return _call_scope_ctx.get()


try:  # pragma: no cover - exercised by every scoped-rail test
    from .skill_router import (
        register_ambient_scope_resolver as _register_scope_resolver,
    )

    _register_scope_resolver(_ambient_run_scope)
except Exception:  # noqa: BLE001 - a Router-less install still traces fine
    logger.debug("could not register the LangChain ambient scope resolver",
                 exc_info=True)


def _drain_unscoped_rails_for(
    scopes: "set[str]",
) -> tuple[Optional[str], List[str], List[str], List[str]]:
    """Atomic, ownership-required version of :func:`_drain_router_rails`.

    Falls back to the old unconditional drain only for a router object from an
    older SDK that has no such method — there is no ownership record to consult
    there, so the pre-existing behaviour is all that is available.
    """
    if _skill_router_singleton is None:
        return None, [], [], []
    owned = getattr(_skill_router_singleton, "drain_unscoped_rails_for", None)
    if owned is None:
        return _drain_router_rails()
    try:
        return owned(scopes)
    except Exception:
        logger.debug("Ownership-checked rail drain failed", exc_info=True)
        return None, [], [], []


def _drain_router_rails() -> tuple[Optional[str], List[str], List[str], List[str]]:
    """Drain the Router singleton's UNSCOPED instance rails —
    ``(routing_id, offered, delivered, loaded)``.

    The contextvars above stay authoritative wherever they propagate, but
    under LangChain they don't: the BaseChatModel patch writes them inside
    the runnable's context, and LangChain dispatches this handler's
    callbacks under `copy_context()`, so every rail came back empty on a
    run that demonstrably injected a menu. The Router carries the same
    values as instance state, which is why this drain exists at all.

    These rails are process-wide and clear-on-read, so what comes back here
    is NOT attributable on its own: under concurrency it is a mix of every
    live run, and the first trace to drain took every lane's names. Callers
    must therefore treat the return value as a last resort behind
    `_drain_scoped_router_rails`, AND must first ask
    `_unscoped_rail_owners()` whether anything on it was written by a run
    other than theirs. The drain itself stays unconditional either way — an
    undrained rail leaks forward into the NEXT trace instead.
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


def _run_scopes(state: "_RunState") -> "set[str]":
    """Every Router rail scope this run owns.

    Two kinds, and both are provable rather than inferred: the LangChain run
    ids the handler itself indexed (``member_ids``), and the per-model-call
    tokens `_capture_call_rails` recorded from inside the call (``rail_scopes``).
    """
    return {str(m) for m in state.member_ids} | set(state.rail_scopes)


def _drain_scoped_router_rails(
    scopes: "set[str]",
) -> tuple[Optional[str], List[str], List[str], List[str]]:
    """Drain the Router rails filed under THIS run's scopes.

    ``_ambient_run_scope`` files each write under whichever LangChain run was
    executing — the agent node for a routing decision, the tools node for a
    body load — and every id it can answer with reaches this handler as a
    callback ``run_id`` or ``parent_run_id``. Draining the whole set is
    therefore a complete read of this run and, by construction, of nothing
    else: another run's writes are filed under scopes this run never owned.

    Pop, never peek — the same rule `otel._pop_skill_rail` states: a rail read
    twice would hand one routing decision to two traces.

    Returns empty on a router that predates the scoped rails (or a caller's
    stand-in), so the unscoped path below still answers for it.
    """
    router = _skill_router_singleton
    if router is None or not scopes:
        return None, [], [], []
    routing_id: Optional[str] = None
    offered: List[str] = []
    delivered: List[str] = []
    loaded: List[str] = []
    for scope in scopes:
        try:
            rid = router.consume_routing_id(scope=scope)
            offered.extend(_clean_names(router.consume_offered_names(scope=scope)))
            delivered.extend(_clean_names(router.consume_delivered_names(scope=scope)))
            loaded.extend(_clean_names(router.consume_loaded_names(scope=scope)))
        except TypeError:
            # A router from an older SDK takes no `scope` at all. Nothing is
            # scoped on it, so there is nothing here to drain.
            return None, [], [], []
        except Exception:
            logger.debug("scoped router-rail drain failed (non-fatal)", exc_info=True)
            continue
        if routing_id is None and isinstance(rid, str) and rid:
            routing_id = rid
    return routing_id, offered, delivered, loaded


def _discard_scoped_router_rails(state: "_RunState") -> None:
    """Release the Router rails of a run that will never build a trace.

    Reached from the eviction path, `_reset_state`, and the `finally` in
    `_close_run` (so an errored run is covered too). Without it a run whose
    end callback never arrived would leave its entries on the process-wide
    singleton until the LRU pushed them out — one dict entry per abandoned
    run, which is the shape of a slow leak in a long-lived server.
    """
    router = _skill_router_singleton
    if router is None:
        return
    for scope in _run_scopes(state):
        try:
            router.consume_routing_id(scope=scope)
            router.consume_offered_names(scope=scope)
            router.consume_delivered_names(scope=scope)
            router.consume_loaded_names(scope=scope)
        except Exception:
            return  # an older router keeps nothing scoped to release


def _unscoped_rail_owners() -> Optional["set[str]"]:
    """Which runs wrote the Router's currently-undrained UNSCOPED content.

    This is what makes keeping the unscoped fallback safe rather than merely
    convenient. The rails themselves carry no provenance — that is the whole
    defect — but the Router now records the ambient scope of every unscoped
    write, so a reader can ask "is any of this somebody else's?" instead of
    guessing.

    Returns None when the router cannot answer (an older SDK, or a caller's
    stand-in): those carry no scoped rails either, so there is no information
    to do better with and the caller keeps the pre-scope behaviour.

    A write with no ambient scope at all contributes no owner. That case is
    Router traffic from outside any LangChain model call or runnable — the
    generic tracer, another adapter, a `load_skill` at import time — and it
    stays claimable exactly as it is today, because no run identity exists to
    prefer over any other.
    """
    router = _skill_router_singleton
    if router is None:
        return set()
    peek = getattr(router, "unscoped_rail_owners", None)
    if not callable(peek):
        return None
    try:
        owners = peek()
    except Exception:
        logger.debug("unscoped rail-owner peek failed (non-fatal)", exc_info=True)
        return None
    if not isinstance(owners, (list, tuple, set)):
        return None
    return {o for o in owners if isinstance(o, str) and o}


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


def _as_message_list(input_value: Any) -> Optional[List[Any]]:
    """Normalize a `BaseChatModel.invoke` input to a message list, or None.

    None means "this shape cannot carry an injected system message" — the
    caller must then leave the input alone AND consult no Router, because
    the mere act of calling `build_prompt_fragment` mints a routing_id and
    fills the Router's offered/delivered rails, which `build_trace` later
    stamps onto the trace. Doing that on a call we cannot inject into is
    how an LCEL run came to claim a routing_id and 30 offered skill names
    while the model provably saw neither.
    """
    if isinstance(input_value, list):
        return list(input_value)
    if isinstance(input_value, str):
        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            return None
        return [HumanMessage(content=input_value)]
    # PromptValue (ChatPromptValue / StringPromptValue) — the shape every
    # `prompt | llm` LCEL chain hands the model. It converts losslessly to
    # messages, which is exactly what `invoke` does with it internally, so
    # injecting here is the same call the model would have received.
    to_messages = getattr(input_value, "to_messages", None)
    if callable(to_messages):
        try:
            messages = to_messages()
        except Exception:
            logger.debug("PromptValue.to_messages() failed (non-fatal)", exc_info=True)
            return None
        if isinstance(messages, list):
            return list(messages)
    return None


def _inject_skills_into_input(input_value: Any) -> Any:
    """Prepend a SkillRouter-built system message to a chat model's input.

    Accepts the three input shapes LangChain's BaseChatModel.invoke sees: a
    string, a list of messages (BaseMessage / dict), and a PromptValue.
    Falls through unchanged — and without consulting the Router at all — on
    anything else, so the trace never claims a routing decision that did not
    reach the model.
    """
    # Shape dispatch FIRST. This used to run last, after the Router had
    # already been consulted and its routing_id + offered names stamped on
    # the trace — so every `prompt | llm` chain (a PromptValue, which the old
    # code could not inject into) reported a full skill menu the model never
    # saw.
    messages = _as_message_list(input_value)
    if messages is None:
        return input_value

    router = _get_skill_router()
    if router is None:
        return input_value

    try:
        from langchain_core.messages import SystemMessage
    except ImportError:
        return input_value

    query = _extract_query_from_messages(messages)
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

    return [SystemMessage(content=fragment), *messages]


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
        tokens = _open_call_rails()
        try:
            try:
                input = _inject_skills_into_input(input)
            except Exception:
                logger.debug("Skill injection failed (non-fatal)", exc_info=True)
            return original_invoke(self, input, config=config, stop=stop, **kwargs)
        finally:
            _close_call_rails(tokens)

    async def patched_ainvoke(self, input, config=None, *, stop=None, **kwargs):
        tokens = _open_call_rails()
        try:
            try:
                input = _inject_skills_into_input(input)
            except Exception:
                logger.debug("Skill injection failed (non-fatal)", exc_info=True)
            return await original_ainvoke(self, input, config=config, stop=stop, **kwargs)
        finally:
            _close_call_rails(tokens)

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


def _publish_handler(
    agent_name: Optional[str], register_configure_hook: Any,
) -> CallbackHandler:
    """Make the module's global handler the process-wide LangChain callback.

    The published handler is the ContextVar's DEFAULT, not merely its value
    in the installing context — which is the entire threading fix.
    LangChain's configure hook installs the handler only when
    `var.get()` is non-None, and in a Context that never had a value set
    `.get()` can only answer with the default. A worker started with
    `threading.Thread` gets exactly such a fresh empty Context, so against
    the old `default=None` var four chains on four threads produced ZERO
    traces and not one warning. `instrument()` used to `.set(handler)` in
    the calling context, which reaches `copy_context()` children (the
    `.batch` executor, asyncio tasks) but never a plain thread.

    `handle_class=CallbackHandler` is the other half, and it is not
    optional — publishing by default without it makes duplicate tracing
    WORSE. With `handle_class=None` LangChain dedupes the global handler by
    object IDENTITY, so a caller who runs `instrument()` and ALSO passes
    `config={"callbacks": [CallbackHandler(...)]}` gets both handlers, and
    both ship a trace. Because a span's id IS the LangChain run_id, the two
    traces carry identical span ids; the backend's `_insertable_rows`
    id-dedup keeps the first and stores the second with zero spans and zero
    llm_calls — a phantom empty trace on the agent's timeline for every
    single run. Deduping by TYPE means an explicitly-passed handler
    suppresses the global one, which is what "per-call control" always
    meant.
    """
    _global_handler.agent_name = agent_name
    _global_handler.auto_send = True
    _global_handler.reset()
    _decimal_callback_var.set(_global_handler)
    register_configure_hook(
        _decimal_callback_var, True, handle_class=CallbackHandler,
    )
    return _global_handler


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
    # Captured before the derivation below overwrites it: `disk_sync=True`
    # passed by hand is a request, the derived default is only a guess.
    _disk_sync_explicit = disk_sync is True
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
                    # Reading and uploading local skills is free; WRITING them
                    # creates a directory in the user's repo. Only mirror when
                    # something will actually read the files back.
                    from .skill_router import should_auto_pull_to_disk
                    allowed, why = should_auto_pull_to_disk(
                        target_agent, explicitly_requested=_disk_sync_explicit
                    )
                    if not allowed:
                        logger.debug("Skipping skill pull to disk: %s", why)
                        return
                    logger.debug("Pulling skills to disk: %s", why)
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

    _publish_handler(agent_name, register_configure_hook)

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


# Ceiling on root runs one handler tracks at once. Reached only when a root
# chain starts and never ends; see `_new_run_state`.
_MAX_LIVE_RUNS = 256

# Sort key for records that never got a timestamp.
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


class _RunState:
    """Everything one root run needs to build its own trace.

    Every field here used to be a single slot on the handler, which was
    survivable only while runs were strictly serial. ``instrument()``
    publishes exactly ONE handler process-wide, and LangChain copies the
    context for parallel work — ``ContextThreadPoolExecutor`` for
    ``chain.batch([...])``, the event loop for ``asyncio.gather(ainvoke)``
    — so three concurrent runs shared one set of slots. ``on_chain_start``
    only reset when ``_root_run_id`` was None, so runs 2 and 3 appended
    into run 1's dicts, and the first outermost ``on_chain_end`` shipped
    the still-open sibling spans (the backend answers ``spans[N]:
    'ended_at' is required``, a 400) and wiped the state the other two had
    yet to use. Net: 0 of 3 runs persisted. Keyed per root run, the same
    batch produces three complete traces.
    """

    __slots__ = (
        "root_run_id", "member_ids", "rail_scopes", "opened_at", "is_leaf_root",
        "agent_hint",
        "detected_agent_name", "trace_id", "spans", "llm_calls", "tool_calls",
        "tool_requests",
        "span_stack", "trace_started_at", "user_input_preview",
        "final_output_preview", "seen_tools", "seen_model", "seen_prompts",
        "seen_output_contract", "streaming_buffers", "active_skills",
        "skills_offered_in_prompt", "skills_loaded_by_agent", "skills_delivered",
        "routing_id",
    )

    def __init__(
        self,
        root_run_id: Optional[UUID] = None,
        *,
        is_leaf_root: bool = False,
    ) -> None:
        self.root_run_id = root_run_id
        # Every run_id (root, nested chain, LLM call, tool) that belongs to
        # this run, so the reverse `_root_of` index can be pruned on close.
        self.member_ids: set[UUID] = {root_run_id} if root_run_id else set()
        # Router rail scopes this run owns that are NOT LangChain run ids: the
        # per-model-call token `_open_call_rails` mints. A bare `llm.invoke()`
        # injects skills before LangChain has minted any run id at all, so
        # without this its routing decision would be filed under no owner and
        # a concurrent run could claim it. Recorded by `_capture_call_rails`,
        # which runs inside that same model call's Context.
        self.rail_scopes: set[str] = set()
        self.opened_at: datetime = datetime.now(timezone.utc)
        # True when the root IS a leaf callback (a model call or a tool) —
        # a bare `llm.invoke()` emits no chain callbacks at all, so nothing
        # else will ever close or send this run.
        self.is_leaf_root = is_leaf_root
        self.agent_hint: Optional[str] = None
        # The name auto-detected from THIS run's root chain. Per-run because
        # the detection used to be written back onto `handler.agent_name`,
        # where it stuck for the life of the process: `instrument()` publishes
        # ONE handler, so in a process serving several agents every run after
        # the first was filed under the first agent's name — three
        # differently-named parallel chains all landed on one agent and the
        # other two had no traces at all.
        self.detected_agent_name: Optional[str] = None

        self.trace_id: UUID = uuid4()
        self.spans: Dict[UUID, TraceSpan] = {}
        self.llm_calls: Dict[UUID, LlmCallRecord] = {}
        self.tool_calls: Dict[UUID, ToolCallRecord] = {}
        # llm_call id -> tool names that call asked for, in order. Used to
        # attach tool records across a graph's node boundary.
        self.tool_requests: Dict[UUID, List[str]] = {}
        self.span_stack: List[UUID] = []
        self.trace_started_at: Optional[datetime] = None
        self.user_input_preview: Optional[str] = None
        self.final_output_preview: Optional[str] = None
        # Manifest auto-detection accumulators
        self.seen_tools: Dict[str, Dict[str, Any]] = {}  # name -> {schema, ...}
        self.seen_model: Optional[Dict[str, Any]] = None
        self.seen_prompts: Dict[str, str] = {}  # role -> text
        self.seen_output_contract: Optional[Dict[str, Any]] = None  # response_format
        self.streaming_buffers: Dict[UUID, List[str]] = {}  # token buffers
        # Skill activation tracking
        self.active_skills: Dict[str, Optional[str]] = {}
        # Skill Rater discovery telemetry. `skills_offered_in_prompt` is
        # auto-populated from the Router's offered set via the
        # BaseChatModel.invoke patch; `skills_loaded_by_agent` is a manual
        # hook (use `decimalai.log_skill_loaded` or the explicit
        # `CallbackHandler.log_skill_loaded` method).
        self.skills_offered_in_prompt: set[str] = set()
        self.skills_loaded_by_agent: set[str] = set()
        # Bodies that reached the model (Router body injection) — between
        # offered and activated; never implies activation.
        self.skills_delivered: set[str] = set()
        # The routing decision THIS run was given, captured off the
        # BaseChatModel patch's contextvar inside the model call it belongs
        # to (see `_capture_call_rails`). Per-run because the Router
        # singleton's instance rail is process-wide: eight concurrent lanes
        # drained whichever decision the router had minted last, so a trace
        # reported a routing_id the router had made for somebody else's
        # prompt — and two traces reported the same one.
        self.routing_id: Optional[str] = None


def _bind_current_run_fields(cls: type) -> type:
    """Republish `_RunState` fields as `handler._<field>`, read and write.

    `handler._spans`, `handler._seen_model`, `handler._trace_started_at` and
    the rest were plain attributes before trace state moved per-run. They are
    load-bearing for manual `auto_send=False` use, so keep them — pointed at
    the most recently started run.
    """
    def _make(name: str) -> property:
        def getter(self: Any) -> Any:
            return getattr(self._current, name)

        def setter(self: Any, value: Any) -> None:
            setattr(self._current, name, value)

        return property(getter, setter)

    for field in cls._CURRENT_FIELDS:  # type: ignore[attr-defined]
        setattr(cls, f"_{field}", _make(field))
    return cls


@_bind_current_run_fields
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
        self._explicit_agent_name: Optional[str] = agent_name
        self.auto_send = auto_send
        self.session_id = session_id
        self.project = project
        self.parent_trace_id = parent_trace_id
        self.subagents = list(subagents) if subagents else None

        # Live root runs, keyed by their root run_id, plus the map from
        # every run_id we have seen to the root that owns it. Guarded by an
        # RLock: `.batch()` dispatches a run's children on LangChain's
        # ContextThreadPoolExecutor, so these dicts are written from several
        # threads at once.
        self._runs: Dict[UUID, _RunState] = {}
        self._root_of: Dict[UUID, UUID] = {}
        self._state_lock = threading.RLock()
        self._reset_state()

    # ── Agent name ─────────────────────────────────────────

    @property
    def agent_name(self) -> Optional[str]:
        """The name this handler is filing under right now.

        Reads the name given at construction (or by ``instrument()``), and
        falls back to the one auto-detected from the CURRENT run's root
        chain. The auto-detected half is a read-through rather than a stored
        value, which is the fix for the identity defect: it used to be
        assigned onto the handler by `on_chain_start`, and `instrument()`
        publishes exactly ONE handler process-wide, so the first chain's name
        became every later chain's name — a process serving three
        differently-named agents filed all three under the first one, and the
        other two had no traces of their own on the platform at all.
        """
        if self._explicit_agent_name is not None:
            return self._explicit_agent_name
        state = getattr(self, "_current", None)
        return state.detected_agent_name if state is not None else None

    @agent_name.setter
    def agent_name(self, value: Optional[str]) -> None:
        self._explicit_agent_name = value

    # ── Per-run state ──────────────────────────────────────
    #
    # Everything a trace is built from lives on a `_RunState` keyed by root
    # run, never on the handler. `self._current` is the most recently
    # STARTED run and backs the legacy `handler._spans` / `handler._seen_model`
    # / … attributes below, which are part of this class's public-ish surface
    # (manual `auto_send=False` use, and tests).

    def _reset_state(self) -> None:
        """Drop every in-flight run and start from one fresh, detached state."""
        with self._state_lock:
            dropped = list(self._runs.values())
            self._runs.clear()
            self._root_of.clear()
            self._current: _RunState = _RunState()
        # A discarded run never builds a trace, so nothing else will ever
        # drain the Router rails filed under its ids. Release them here or
        # they sit on the singleton until the LRU pushes them out.
        for state in dropped:
            _discard_scoped_router_rails(state)

    def _new_run_state(self, root_run_id: UUID, *, is_leaf_root: bool = False) -> _RunState:
        """Open a state for a new root run and make it the current one."""
        state = _RunState(root_run_id, is_leaf_root=is_leaf_root)
        evicted: Optional[_RunState] = None
        with self._state_lock:
            if len(self._runs) >= _MAX_LIVE_RUNS:
                # A root whose end callback never arrived (a hard kill inside
                # a node, a framework that swallows the error) would otherwise
                # pin its state forever. Evict the oldest so a long-lived
                # process cannot grow without bound.
                oldest = min(self._runs.values(), key=lambda s: s.opened_at)
                self._forget_run(oldest)
                evicted = oldest
                _warn_once_then_debug(
                    "run_state_evicted",
                    f"Dropping the oldest of {_MAX_LIVE_RUNS} in-flight LangChain "
                    f"runs — a root chain started but never ended, so its trace "
                    f"is lost. Trace {oldest.trace_id} ({oldest.agent_hint or 'unnamed'}).",
                )
            self._runs[root_run_id] = state
            self._root_of[root_run_id] = root_run_id
            state.member_ids.add(root_run_id)
            self._current = state
        if evicted is not None:
            # Same reasoning as `_reset_state`: an evicted run's trace is
            # already lost, so its Router rails have no reader left.
            _discard_scoped_router_rails(evicted)
        return state

    def _forget_run(self, state: _RunState) -> None:
        """Remove a run's bookkeeping. Caller holds `_state_lock`."""
        if state.root_run_id is not None:
            self._runs.pop(state.root_run_id, None)
        for member in state.member_ids:
            if self._root_of.get(member) == state.root_run_id:
                self._root_of.pop(member, None)

    def _state_for(
        self,
        run_id: Optional[UUID],
        parent_run_id: Optional[UUID] = None,
    ) -> Optional[_RunState]:
        """Resolve the run state that owns `run_id`, or None if it is gone.

        None is a normal outcome, not an error: it is what a duplicate
        callback delivery looks like after the run has already been shipped
        and popped, which is precisely how double instrumentation used to
        emit a second, empty trace.
        """
        with self._state_lock:
            for candidate in (run_id, parent_run_id):
                if candidate is None:
                    continue
                root = self._root_of.get(candidate)
                if root is not None:
                    state = self._runs.get(root)
                    if state is not None:
                        if run_id is not None and run_id not in state.member_ids:
                            state.member_ids.add(run_id)
                            self._root_of[run_id] = root
                        return state
            return None

    # Legacy single-slot attributes, delegated to the current run state so
    # `handler._spans[...]`, `handler._active_skills = {...}` and friends keep
    # working for manual (auto_send=False) callers and tests.
    _CURRENT_FIELDS = (
        "trace_id", "spans", "llm_calls", "tool_calls", "span_stack",
        "trace_started_at", "user_input_preview", "final_output_preview",
        "root_run_id", "seen_tools", "seen_model", "seen_prompts",
        "seen_output_contract", "streaming_buffers", "active_skills",
        "skills_offered_in_prompt", "skills_loaded_by_agent", "skills_delivered",
    )

    def _capture_call_rails(self, state: _RunState) -> None:
        """Attach the skills rails of the model call that is starting NOW.

        Called from `on_llm_start` / `on_chat_model_start`, which LangChain
        dispatches from INSIDE the patched `BaseChatModel.invoke` — the same
        Context the injection wrote its rails into, and the only place where
        "this routing decision" and "this run" are provably the same thing.

        The rails are read, not consumed: `_close_call_rails` clears them
        when the model call returns, so a value cannot outlive the call it
        describes. Reading them here is what replaced draining the Router
        singleton at trace-build time — that drain ran long after the call,
        against state every concurrent run shared, and handed traces routing
        decisions made for other runs' prompts.
        """
        # Claim this model call's rail scope for the run. `_ambient_run_scope`
        # falls back to this token when LangChain has no run id to give (a
        # bare `llm.invoke()` injects before one exists), and recording it
        # here — inside the call, in the Context that minted it — is what lets
        # `_drain_scoped_router_rails` recognise the resulting write as ours.
        call_scope = _call_scope_ctx.get()
        if call_scope:
            state.rail_scopes.add(call_scope)

        routing_id = _routing_id_ctx.get()
        if routing_id and state.routing_id is None:
            # First decision of the run wins: a multi-turn run routes once
            # per model call against the same query, and the trace carries
            # one routing_id.
            state.routing_id = routing_id
        for name in _skills_offered_ctx.get() or ():
            if isinstance(name, str) and name.strip():
                state.skills_offered_in_prompt.add(name.strip())
        for name in _skills_delivered_ctx.get() or ():
            if isinstance(name, str) and name.strip():
                state.skills_delivered.add(name.strip())
                state.skills_offered_in_prompt.add(name.strip())

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

        span_id = run_id or uuid4()

        # Claim this run for a trace BEFORE the skip check. A skipped wrapper
        # is still a real node in the callback tree — `prompt | llm` with no
        # `run_name` is a RunnableSequence, i.e. the outermost run of the
        # whole invocation — and its children arrive carrying its run_id as
        # their parent. Returning early without recording it left every child
        # looking parentless, which is how a second concurrent run used to be
        # mistaken for a continuation of the first.
        state = self._state_for(run_id, parent_run_id)
        is_new_root = state is None
        if state is None:
            state = self._new_run_state(span_id)
            state.agent_hint = str(name) or None

        # Skip noisy internal LangChain wrappers
        if any(name.startswith(skip) for skip in _SKIP_CHAIN_TYPES):
            return

        if not name:
            name = "chain"

        parent_id = parent_run_id if parent_run_id in state.spans else None

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
        state.spans[span_id] = span
        state.span_stack.append(span_id)

        # Capture first input as trace-level preview
        if state.trace_started_at is None:
            state.trace_started_at = span.started_at
            state.user_input_preview = _preview(inputs)

        # Auto-detect agent_name from root chain if not explicitly set.
        # Recorded on the RUN, never written back onto `self.agent_name`:
        # one handler serves the whole process, so a name learned here used
        # to become every later run's name too.
        if self._explicit_agent_name is None and (parent_run_id is None or is_new_root):
            detected = str(name)
            if detected and detected not in ('chain', 'unknown'):
                if state.detected_agent_name is None:
                    state.detected_agent_name = detected

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain or graph node ends."""
        state = self._state_for(run_id, parent_run_id)
        if state is None:
            # Already shipped and popped. The common cause is the same
            # callback delivered twice (two handlers installed, or a
            # re-registered configure hook) — which used to run the send path
            # a second time against wiped state and put a phantom EMPTY trace
            # in the dashboard, whose spans then collided by id with the real
            # trace's.
            return

        span_id = run_id
        if span_id and span_id in state.spans:
            span = state.spans[span_id]
            span.status = Status.SUCCESS
            span.ended_at = datetime.now(timezone.utc)
            span.output_preview = _preview(outputs)

        if state.span_stack and state.span_stack[-1] == span_id:
            state.span_stack.pop()

        state.final_output_preview = _preview(outputs)

        # Auto-send at the true outermost end — the only callback that
        # carries parent_run_id=None. Matching on `span_id ==
        # self._root_run_id` broke on langchain-core 1.5.x, which reuses the
        # root run_id for child steps (ChatPromptTemplate's events arrive
        # with run_id == parent_run_id == root), so the old check fired at
        # the PROMPT step's end and sent the trace before the LLM call
        # existed. The emptiness guard keeps all-skipped runs (e.g. a bare
        # RunnablePassthrough) from sending empty traces.
        if parent_run_id is None:
            self._close_run(state)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors."""
        state = self._state_for(run_id, parent_run_id)
        if state is None:
            return

        span_id = run_id
        if span_id and span_id in state.spans:
            span = state.spans[span_id]
            span.status = Status.ERROR
            span.ended_at = datetime.now(timezone.utc)
            span.output_preview = str(error)[:200]

        if state.span_stack and state.span_stack[-1] == span_id:
            state.span_stack.pop()

        # Auto-send on the outermost error too — same parent_run_id=None
        # boundary as on_chain_end (see the note there).
        if parent_run_id is None:
            self._close_run(state)

    def _close_run(self, state: _RunState) -> None:
        """Retire one root run: stop routing callbacks to it, then send it."""
        with self._state_lock:
            self._forget_run(state)
        try:
            if self.auto_send and (state.spans or state.llm_calls):
                self._auto_send(state)
        finally:
            # `build_trace` is what drains this run's Router rails, and on an
            # auto-sending handler it is the ONLY reader this run will ever
            # get — so any path that reaches here without building a trace
            # (tracing switched off, a run with nothing to send, `_auto_send`
            # failing before it assembles, `on_chain_error` routing here)
            # leaves a per-run entry behind on the process-wide singleton.
            # Idempotent: after a build there is nothing left to release.
            #
            # Guarded on `auto_send` because a manual (`auto_send=False`)
            # caller builds the trace HERSELF, after the chain returns —
            # releasing the rails here would take them away before she asks.
            if self.auto_send:
                _discard_scoped_router_rails(state)

    # ── Sub-agent resolution ───────────────────────────────

    def _agent_name_or_default(self, state: Optional[_RunState] = None) -> str:
        """The name ONE RUN's records ship under — never None.

        `on_chain_start`'s auto-detection cannot fire when the root
        runnable is one of `_SKIP_CHAIN_TYPES` — `prompt | llm` is a
        RunnableSequence, so the callback returns before the detection
        block — nor for a bare `llm.invoke()`, which emits no chain
        callback at all. Both left `agent_name` None all the way into the
        payload, and the ingest API rejects that: every LCEL trace from
        `init(langchain=True)` was dropped with TRACE_VALIDATION_FAILED.

        Resolution is deliberately NOT written back to `self.agent_name`,
        and the auto-detected name lives on the RUN rather than on the
        handler: that is what lets the next root chain be named on its own
        merits instead of inheriting the first one's name for the life of
        the process.
        """
        state = state if state is not None else self._current
        detected = state.detected_agent_name if state is not None else None
        return (
            self._explicit_agent_name
            or detected
            or _install_agent_name
            or DEFAULT_AGENT_NAME
        )

    def _resolve_agent_name(
        self,
        parent_run_id: Optional[UUID] = None,
        state: Optional[_RunState] = None,
    ) -> Optional[str]:
        """Resolve the agent name for an LLM call by walking parent spans.

        Finds the nearest ancestor span of type AGENT and returns its name.
        Falls back to ``_agent_name_or_default()`` if no agent-type parent
        is found, preserving backward compatibility for the single-agent
        case.

        The walk is bounded. If the run's spans ever contain a
        parent_span_id cycle (malformed `parent_run_id` from LangChain, or a
        custom callback corrupting them), the old `while span_id and
        span_id in self._spans` looped forever, blocking the LangChain
        dispatcher thread. Two guards: a `seen` set so we break on
        cycles, and a hop cap of 32 — agent topologies don't go deeper
        than ~10 in practice.
        """
        spans = (state or self._current).spans
        span_id = parent_run_id
        seen: set[UUID] = set()
        for _ in range(32):
            if not span_id or span_id in seen or span_id not in spans:
                break
            seen.add(span_id)
            span = spans[span_id]
            if span.span_type == SpanType.AGENT and span.name:
                return span.name
            span_id = span.parent_span_id
        return self._agent_name_or_default(state)

    # ── LLM / tool run routing ─────────────────────────────

    def _state_for_leaf(
        self,
        run_id: Optional[UUID],
        parent_run_id: Optional[UUID],
    ) -> _RunState:
        """Resolve (or open) the run state owning an LLM or tool callback.

        An LLM callback with no resolvable parent means a bare
        `llm.invoke()` — no chain callbacks are emitted for it at all. That
        used to leave the record stranded in the handler's one set of slots:
        no trace was ever sent for it, and the stranded record was then
        shipped inside whatever unrelated run next reached the send path
        (observed: a bare call's prompt surfacing in a later
        `RunnableLambda` trace). It now gets a root of its own.
        """
        state = self._state_for(run_id, parent_run_id)
        if state is not None:
            return state
        return self._new_run_state(run_id or uuid4(), is_leaf_root=True)

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
        state = self._state_for(run_id, kwargs.get("parent_run_id"))
        if state is None:
            return
        if run_id not in state.streaming_buffers:
            state.streaming_buffers[run_id] = []
        state.streaming_buffers[run_id].append(token)

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
        """Called when a (completion, non-chat) LLM call starts."""
        call_id = run_id or uuid4()
        params = invocation_params or {}
        state = self._state_for_leaf(call_id, parent_run_id)
        self._capture_call_rails(state)

        rendered_input = [{"role": "user", "content": p} for p in prompts]

        provider = extract_provider(params, serialized)
        model_name = extract_model_name(params)
        call = LlmCallRecord(
            id=call_id,
            span_id=parent_run_id,
            agent_name=self._resolve_agent_name(parent_run_id, state),
            provider=provider,
            model_name=model_name,
            temperature=params.get("temperature"),
            max_output_tokens=params.get("max_tokens"),
            rendered_input=rendered_input,
            status=Status.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        state.llm_calls[call_id] = call

        # A completion model is still a declared model. Only `on_chat_model_
        # start` used to record one, so a non-chat LLM produced an empty
        # manifest and every one of its traces 400'd on the manifest gate.
        if model_name and not state.seen_model:
            state.seen_model = {
                "provider": provider or "unknown",
                "model": model_name,
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                "top_p": params.get("top_p"),
            }

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
        state = self._state_for_leaf(call_id, parent_run_id)
        self._capture_call_rails(state)

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
            agent_name=self._resolve_agent_name(parent_run_id, state),
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

        state.llm_calls[call_id] = call

        # Auto-detect model and prompts for manifest
        provider = extract_provider(params, serialized)
        model_name = extract_model_name(params)
        if model_name and not state.seen_model:
            state.seen_model = {
                "provider": provider or "unknown",
                "model": model_name,
                "temperature": params.get("temperature"),
                "max_tokens": params.get("max_tokens"),
                "top_p": params.get("top_p"),
            }

        # Capture structured output schema (response_format) as output_contract
        resp_fmt = params.get("response_format")
        if resp_fmt and isinstance(resp_fmt, dict) and not state.seen_output_contract:
            state.seen_output_contract = resp_fmt
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
                if role in ("system", "developer") and content and role not in state.seen_prompts:
                    state.seen_prompts[role] = content
                elif role in ("system", "developer") and content and role in state.seen_prompts:
                    # Prompt changed within the same trace — likely dynamic
                    if content != state.seen_prompts[role]:
                        logger.warning(
                            "Auto-detected %s prompt changed within trace. "
                            "If your prompts include dynamic content (RAG, dates, "
                            "few-shot), pass static templates via install(prompts=...) "
                            "to avoid false manifest drift.",
                            role,
                        )

        # Skill rungs are NOT inferred here. This fires at
        # on_chat_model_start, before `_capture_call_rails` and the trace-build
        # rail merge have finished naming what the ROUTER accounted for on this
        # run — and the precedence rule needs that set complete, or a skill the
        # router only offered gets re-inferred from its own menu row. The
        # inference runs once in `build_trace`, over `state.seen_prompts`,
        # which is fully populated by then.

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call ends."""
        call_id = run_id
        state = self._state_for(call_id, parent_run_id)
        if not call_id or state is None or call_id not in state.llm_calls:
            return

        call = state.llm_calls[call_id]
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
                requested = extract_tool_call_names(gen.message)
                if requested:
                    call.finish_reason = FinishReason.TOOL_CALLS
                    # Remember WHICH tools this turn asked for — `build_trace`
                    # needs it to attach the resulting ToolCallRecords across a
                    # graph node boundary. See the note there.
                    state.tool_requests[call_id] = requested
                else:
                    call.finish_reason = FinishReason.STOP

        input_tokens, output_tokens = extract_token_usage(response)
        call.input_tokens = input_tokens
        call.output_tokens = output_tokens

        # Handle streaming buffer — join buffered tokens
        if call_id in state.streaming_buffers:
            tokens = state.streaming_buffers.pop(call_id)
            if tokens:
                call.streaming = True
                call.streaming_token_count = len(tokens)
                # If output wasn't set from response, use joined tokens
                if not call.output or not call.output.get("content"):
                    call.output = call.output or {}
                    call.output["streaming_content"] = "".join(tokens)

        self._close_if_leaf_root(state, call_id)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call errors."""
        call_id = run_id
        state = self._state_for(call_id, parent_run_id)
        if state is None:
            return
        if call_id and call_id in state.llm_calls:
            call = state.llm_calls[call_id]
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
            state.streaming_buffers.pop(call_id, None)

        self._close_if_leaf_root(state, call_id)

    def _close_if_leaf_root(
        self, state: _RunState, run_id: Optional[UUID],
    ) -> None:
        """Ship a run whose ROOT is this leaf callback.

        A bare `llm.invoke()` (and, rarely, a tool invoked outside any
        chain) emits no chain callbacks at all, so no `on_chain_end` is
        coming to close or send it — and leaving it open would also pin the
        state until the in-flight cap evicts it.
        """
        if not state.is_leaf_root or run_id is None or state.root_run_id != run_id:
            return
        if state.trace_started_at is None:
            call = state.llm_calls.get(run_id)
            span = state.spans.get(run_id)
            started = (call.started_at if call else None) or (
                span.started_at if span else None
            )
            state.trace_started_at = started or state.opened_at
            if call and call.rendered_input:
                state.user_input_preview = _preview(
                    call.rendered_input[-1].get("content")
                )
            elif span:
                state.user_input_preview = span.input_preview
        self._close_run(state)

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
        state = self._state_for_leaf(tool_id, parent_run_id)

        span = TraceSpan(
            id=tool_id,
            parent_span_id=parent_run_id,
            span_type=SpanType.TOOL,
            name=str(tool_name),
            status=Status.RUNNING,
            started_at=datetime.now(timezone.utc),
            input_preview=str(input_str)[:200],
        )
        state.spans[tool_id] = span

        tool_call = ToolCallRecord(
            id=tool_id,
            tool_name=str(tool_name),
            args={"input": input_str} if isinstance(input_str, str) else {},
            status=Status.RUNNING,
        )
        state.tool_calls[tool_id] = tool_call

        # Auto-detect tools for manifest.
        # Include `description` so that manifest diffs and tool-impact
        # analysis see description changes as manifest deltas. Without it,
        # rewriting a tool's description — "Search the web" → "Search the
        # corporate intranet" — produces NO manifest signal, even though
        # it can completely change how the model uses the tool.
        if str(tool_name) not in state.seen_tools:
            state.seen_tools[str(tool_name)] = {
                "name": str(tool_name),
                "description": serialized.get("description", ""),
                "schema": serialized.get("schema"),
            }

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool ends."""
        tool_id = run_id
        state = self._state_for(tool_id, parent_run_id)
        if state is None:
            return
        if tool_id and tool_id in state.spans:
            span = state.spans[tool_id]
            span.status = Status.SUCCESS
            span.ended_at = datetime.now(timezone.utc)
            span.output_preview = str(output)[:200]

        if tool_id and tool_id in state.tool_calls:
            tc = state.tool_calls[tool_id]
            tc.status = Status.SUCCESS
            tc.result = str(output)[:1000]

        self._close_if_leaf_root(state, tool_id)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Optional[UUID] = None,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool errors."""
        tool_id = run_id
        state = self._state_for(tool_id, parent_run_id)
        if state is None:
            return
        if tool_id and tool_id in state.spans:
            span = state.spans[tool_id]
            span.status = Status.ERROR
            span.ended_at = datetime.now(timezone.utc)

        if tool_id and tool_id in state.tool_calls:
            tc = state.tool_calls[tool_id]
            tc.status = Status.ERROR

        self._close_if_leaf_root(state, tool_id)

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

    def build_trace(self, state: Optional[_RunState] = None) -> RunTrace:
        """Assemble one run's spans and LLM calls into a RunTrace."""
        from . import _config

        config = _config._config
        state = state or self._current

        self._attach_tool_calls(state)

        # Build active_skills list
        active_skills_list: List[Dict[str, Any]] = []
        for name, h in state.active_skills.items():
            entry: Dict[str, Any] = {"name": name}
            if h:
                entry["hash"] = h
            active_skills_list.append(entry)

        # ── the skills rails ────────────────────────────────
        # Three sources, in strict precedence order, and every one of them is
        # drained whether or not it is used — an undrained rail leaks forward
        # into the NEXT trace instead of merely being missing from this one.
        #
        # 1. What this RUN captured for itself. `_capture_call_rails` reads
        #    the routing decision inside the model call it was made for, so it
        #    is attributable by construction.
        # 2. The Router rails filed under THIS run's scopes. A load from a
        #    user-supplied tool cannot reach (1) — LangChain dispatches
        #    callbacks under `copy_context()`, so the tool's ContextVar writes
        #    are invisible here — but it CAN be filed under the LangChain run
        #    id that was executing, which is one of ours. See
        #    `_drain_scoped_router_rails`.
        # 3. The Router's UNSCOPED rails, and only once their provenance has
        #    been checked. They are process-wide and clear-on-read: unioning
        #    them unconditionally is what let one run's `load_skill` be
        #    reported as ANOTHER run's activation, and what put a concurrent
        #    lane's routing_id (and, in one measured trace, 34 offered names
        #    against a prompt that offered one) onto a trace. If any of the
        #    content was written by a run that is not this one, the values are
        #    dropped on the floor: a number that might belong to somebody else
        #    is worse than no number, and activation is the rung the product
        #    claim rests on.
        scopes = _run_scopes(state)
        (
            scoped_routing_id, scoped_offered, scoped_delivered, scoped_loaded,
        ) = _drain_scoped_router_rails(scopes)

        # One atomic ask-and-take. The previous peek-then-drain pair had two
        # ways of putting another run's data on this trace: a write landing
        # between the two calls was taken while the snapshot still said clear,
        # and an UNOWNED write (a `load_skill` from a plain thread, outside any
        # run the resolver can name) produced an empty owner set, which
        # `owners - scopes` reads as "nothing contradicts me" rather than
        # "nobody vouches for this". Ownership is required now, and rails that
        # are not ours are left in place for the run that earned them instead of
        # being drained and thrown away.
        (
            rail_routing_id,
            rail_offered,
            rail_delivered,
            rail_loaded,
        ) = _drain_unscoped_rails_for(scopes)

        # Drain the contextvars too — a `log_skill_*` call or an injection
        # that happened outside the patched invoke lands there, and an
        # undrained contextvar leaks into the next trace in this context.
        for n in _consume_skills_offered():
            state.skills_offered_in_prompt.add(n)
        for n in _consume_skills_delivered():
            state.skills_delivered.add(n)
            state.skills_offered_in_prompt.add(n)  # delivered implies offered
        ctx_routing_id = _consume_routing_id()

        if not state.skills_offered_in_prompt:
            for n in scoped_offered or rail_offered:
                state.skills_offered_in_prompt.add(n)
        if not state.skills_delivered:
            for n in scoped_delivered or rail_delivered:
                state.skills_delivered.add(n)
                state.skills_offered_in_prompt.add(n)

        # Bodies served by the singleton's `load_skill(...)` — this adapter
        # registers no native load_skill tool, but a user-supplied tool
        # that calls it still lands its serves here.
        if not state.skills_loaded_by_agent:
            for n in scoped_loaded or rail_loaded:
                if isinstance(n, str) and n.strip():
                    state.skills_loaded_by_agent.add(n.strip())
                    state.skills_offered_in_prompt.add(n.strip())
                    state.skills_delivered.add(n.strip())

        # Infer offered/delivered for DISK skills the SDK did not inject.
        # STRICTLY AFTER every rail merge above, so the precedence rule sees
        # this run's complete router-accounted set.
        self._infer_skill_rungs_from_prompts(state)

        routing_id = (
            state.routing_id or ctx_routing_id or scoped_routing_id or rail_routing_id
        )
        agent_name = self._agent_name_or_default(state)

        # Derive trace status from collected spans/LLM calls: if any errored,
        # the run errored. on_chain_error/on_llm_error mark these ERROR.
        trace_status = Status.SUCCESS
        if any(s.status == Status.ERROR for s in state.spans.values()) or any(
            lc.status == Status.ERROR for lc in state.llm_calls.values()
        ):
            trace_status = Status.ERROR

        return RunTrace(
            id=state.trace_id,
            project=config.project if config else None,
            agent_name=agent_name,
            session_id=self.session_id,
            parent_trace_id=self.parent_trace_id,
            status=trace_status,
            source_type="production",
            started_at=state.trace_started_at,
            ended_at=datetime.now(timezone.utc),
            user_input_preview=state.user_input_preview,
            final_output_preview=state.final_output_preview,
            spans=list(state.spans.values()),
            llm_calls=list(state.llm_calls.values()),
            active_skills=active_skills_list,
            # THIS agent's manifest. A single process-global slot stamped
            # whichever agent registered most recently onto everybody's
            # traces, so the manifest a trace pointed at could belong to a
            # different agent entirely.
            manifest_id=_manifest_ids.get(agent_name),
            # SkillRouter: the routing_id this run was actually given, so the
            # offered-vs-activated join can close on the right decision.
            routing_id=routing_id,
            # Skill Rater discovery telemetry. Sorted for
            # deterministic output (tests, diffs).
            skills_offered_in_prompt=sorted(state.skills_offered_in_prompt),
            skills_loaded_by_agent=sorted(state.skills_loaded_by_agent),
            skills_delivered=sorted(state.skills_delivered),
        )

    def _attach_tool_calls(self, state: _RunState) -> None:
        """Hang each ToolCallRecord off the model turn that requested it.

        A ToolCallRecord only reaches the wire through
        ``LlmCallRecord.tool_calls``, so a record with no LLM call to hang
        off is a record the platform never sees.

        The old rule was one exact match — ``tool_span.parent_span_id ==
        llm_call.span_id``. That holds for a flat LCEL agent and for nothing
        else. In a LangGraph / ``create_agent`` graph the model runs under
        the ``agent`` node and the tool under a sibling ``tools`` node, so no
        tool span's parent ever equals an LLM call's span id and EVERY tool
        record was dropped. Fall back to the turn that asked for this tool by
        name, then to the most recent turn that started before it.
        """
        if not state.tool_calls or not state.llm_calls:
            return

        def _started(record_id: UUID) -> datetime:
            span = state.spans.get(record_id)
            return (span.started_at if span and span.started_at else None) or _EPOCH

        by_span: Dict[UUID, LlmCallRecord] = {}
        for lc in state.llm_calls.values():
            if lc.span_id is not None:
                by_span.setdefault(lc.span_id, lc)

        # name -> the turns that requested it, oldest first. Popped as they
        # are claimed so a turn asking for the same tool twice (parallel tool
        # calls) matches two separate records.
        requested_by: Dict[str, List[LlmCallRecord]] = {}
        for call_id, names in state.tool_requests.items():
            lc = state.llm_calls.get(call_id)
            if lc is None:
                continue
            for name in names:
                requested_by.setdefault(name, []).append(lc)
        for queue in requested_by.values():
            queue.sort(key=lambda lc: lc.started_at or _EPOCH)

        ordered_calls = sorted(
            state.llm_calls.values(), key=lambda lc: lc.started_at or _EPOCH
        )

        for tc in sorted(state.tool_calls.values(), key=lambda t: _started(t.id)):
            tool_span = state.spans.get(tc.id)
            target = None
            if tool_span and tool_span.parent_span_id:
                target = by_span.get(tool_span.parent_span_id)
            if target is None:
                queue = requested_by.get(tc.tool_name)
                if queue:
                    target = queue.pop(0)
            if target is None:
                tool_started = _started(tc.id)
                for lc in ordered_calls:
                    if (lc.started_at or _EPOCH) <= tool_started:
                        target = lc
            if target is not None and tc not in target.tool_calls:
                target.tool_calls.append(tc)

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

    def _auto_send(self, state: Optional[_RunState] = None) -> None:
        """Send one run's trace via the background sender (on root end)."""
        from . import _config

        state = state or self._current

        if not _config._is_enabled():
            logger.debug("Tracing disabled, skipping auto-send")
            return

        # Auto-register manifest on first trace (or when agent changes)
        self._maybe_register_manifest(state)

        try:
            client = _config._get_client()
            trace = self.build_trace(state)
            self._retire(state)

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
            logger.exception("Failed to queue trace %s", state.trace_id)
            self._retire(state)

    def _retire(self, state: _RunState) -> None:
        """Drop a shipped run so the legacy `handler._*` view moves on."""
        with self._state_lock:
            self._forget_run(state)
            if self._current is state:
                self._current = _RunState()

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

    def _infer_skill_rungs_from_prompts(self, state: Optional[_RunState] = None) -> None:
        """Infer OFFERED / DELIVERED for disk skills the SDK did not inject.

        The registry comes from ``instrument(skills=...)`` else
        ``discover_skills()`` — disk either way — so it describes skills a
        harness may have put in the prompt itself. Matching that prompt text
        shows the skill was put in front of the model, never that the model
        reached for it, so ``state.active_skills`` is untouched: on this rail
        activation arrives only through an explicit ``log_skill_activation``
        or a ``load_skill`` serve landing in ``skills_loaded_by_agent``.

        Call from ``build_trace`` AFTER the rail merges — the precedence rule
        needs this run's complete router-accounted set.
        """
        state = state or self._current
        skills_registry = (_explicit_manifest_config or {}).get("skills")
        if not skills_registry or state is None:
            return

        try:
            from .skills import infer_prompt_rungs
            # Build system text from seen prompts
            system_text = "\n".join(state.seen_prompts.values())
            if not system_text:
                return

            # Passed split for readability; infer_prompt_rungs pools them —
            # see the trade-off note there on why suppression is blanket.
            router_offered = set(state.skills_offered_in_prompt)
            router_delivered = set(state.skills_delivered) | set(state.skills_loaded_by_agent)
            offered, delivered = infer_prompt_rungs(
                [[{"role": "system", "content": system_text}]],
                skills_registry,
                router_offered=router_offered,
                router_delivered=router_delivered,
            )
            # NO delivered->offered fold on the INFERRED path. That fold is
            # sound for the router's own rails — it offered the menu row and
            # then delivered the body, so it observed both — but here both
            # rungs are guesses read off prompt text, and Tier-2 matches a
            # BODY whose name may never appear in the prompt at all. Folding
            # would then assert `skills_offered_in_prompt`, which means "the
            # menu row was in the prompt the model was shown", for a name that
            # was not. That is the same fabrication this rewiring exists to
            # remove, moved one rung down.
            state.skills_offered_in_prompt.update(offered)
            state.skills_delivered.update(delivered)
        except Exception:
            _warn_once_then_debug(
                "skill_prompt_presence_inference",
                "Skill prompt-presence inference failed",
            )

    def _resolve_manifest_for_empty_run(self, agent_name: str) -> None:
        """Give a run with nothing to declare a manifest_id to ship under.

        See the long note in `_maybe_register_manifest`. Order: keep what
        this process already has FOR THIS AGENT → adopt that agent's active
        manifest from the platform → register a zero-component placeholder.

        Per-agent throughout: a process-global "do we have one yet?" answered
        yes on behalf of an agent that had never registered anything, and its
        traces then carried another agent's manifest id.
        """
        global _manifest_id

        adopted: Optional[str] = None
        with _manifest_lock:
            _forget_manifests_if_tracker_reset()
            if _manifest_ids.get(agent_name):
                return  # Already have one — never regress it to an empty manifest.
            probe = agent_name not in _manifest_adoption_probed
            if probe:
                _manifest_adoption_probed.add(agent_name)
        if probe:
            adopted = _adopt_active_manifest(agent_name)
        if adopted:
            with _manifest_lock:
                _manifest_ids.setdefault(agent_name, adopted)
                _manifest_id = adopted
            logger.debug(
                "Adopted the platform's active manifest %s for %s "
                "(this run had nothing to declare)",
                adopted, agent_name,
            )
            return

        # No manifest anywhere for this agent — register the placeholder.
        # Zero components on purpose: a fake `{"provider": "unknown"}`
        # model would both lie and still diff as major later. Zero
        # components hash to a per-agent constant, so repeat
        # registrations dedup onto the same v1 instead of minting
        # versions.
        _register_snapshot(agent_name, extract_from_config(agent_name=agent_name))

    def _maybe_register_manifest(self, state: Optional[_RunState] = None) -> None:
        """Extract and register this RUN's manifest if not already done.

        Thread-safe via _manifest_lock (held inside `_register_snapshot`).
        """
        from . import _config
        if not _config._is_enabled():
            return

        state = state or self._current

        # Use explicit config from instrument() if provided
        tools = None
        prompts = None
        models = None

        if _explicit_manifest_config:
            tools = _explicit_manifest_config.get("tools")
            prompts = _explicit_manifest_config.get("prompts")
            models = _explicit_manifest_config.get("models")

        # Fall back to auto-detected values
        if not tools and state.seen_tools:
            tools = list(state.seen_tools.values())
        if not prompts and state.seen_prompts:
            prompts = dict(state.seen_prompts)
        if not models and state.seen_model:
            models = {"default": state.seen_model}

        # Include output contract if detected
        output_contract = None
        if state.seen_output_contract:
            output_contract = state.seen_output_contract

        skills = (_explicit_manifest_config or {}).get("skills")
        subagents = self.subagents

        # ── "Nothing to declare" ────────────────────────────
        # This used to `return`, and that return was the single largest
        # source of trace loss on this adapter: ingest requires a
        # manifest_id (`require_manifest_on_ingest` defaults true and is
        # true on prod), so a legitimate run that exposes no model, tool or
        # prompt — a pure-Python LCEL chain, a `RunnableLambda` step, a
        # completion (non-chat) model — lost 100% of its traces to a 400.
        #
        # The naive fix ("always register") has a trap we measured against
        # the local backend: a zero-component manifest registered AFTER a
        # populated one supersedes it, and the shared agentversion diff
        # reads the absent surfaces as deletions — `model_runtime: provider
        # 'openai' → ''` breaking/major AND `tool_registry: search removed`,
        # recommended_decision "replay". That is a false claim that the
        # user deleted their tools, and it poisons the version history that
        # manifest-aware versioning exists to keep honest.
        #
        # So an empty snapshot is treated as "we have nothing to add",
        # never as a new declaration:
        #   1. if this process already has a manifest for the agent, keep it
        #   2. else adopt the agent's active manifest from the platform
        #      (survives process restarts and multi-process deployments,
        #      where a process-local guard alone would still regress)
        #   3. only if the agent has no manifest at all do we register the
        #      zero-component placeholder — and with no predecessor there
        #      is no diff to fabricate.
        # The forward transition (placeholder → first real declaration) is
        # left as a real version bump: it says a model was declared where
        # none had been, which is true, and it deletes nothing.
        # Same label the trace ships under, or the manifest lands on a
        # different agent than the traces that reference its manifest_id.
        agent_name = self._agent_name_or_default(state)

        if not tools and not prompts and not models and not skills and not subagents:
            self._resolve_manifest_for_empty_run(agent_name)
            return

        snapshot = extract_from_config(
            agent_name=agent_name,
            tools=tools,
            prompts=prompts,
            models=models,
            subagents=subagents,
            output_schema=output_contract,
            skills=skills,
        )

        # Thread-safe, and keyed by (agent, hash) — see `_register_snapshot`.
        _register_snapshot(agent_name, snapshot)

    # ── Backwards compatibility ────────────────────────────

    def reset(self) -> None:
        """Reset state (alias for backwards compat)."""
        self._reset_state()

    def get_completed_trace(self) -> RunTrace:
        """Alias for get_trace() (backwards compat)."""
        return self.get_trace()


# The one handler `instrument()` publishes process-wide, and — critically —
# the ContextVar's DEFAULT, so `var.get()` answers with it inside a worker
# thread's fresh empty Context too. Constructed here rather than inside
# `instrument()` because a ContextVar's default is fixed at construction.
# Inert until `instrument()` registers the configure hook: LangChain never
# reads the var before that.
_global_handler = CallbackHandler(auto_send=True)

_decimal_callback_var: ContextVar[Optional[CallbackHandler]] = ContextVar(
    "decimal_langchain_callback", default=_global_handler,
)


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
