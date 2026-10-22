"""OpenAI Agents SDK integration.

An agent's prompt and skills from DecimalAI, attached explicitly — nothing in the
Agents SDK is patched, and no other agent in the process is affected::

    from agents import Agent, Runner
    from decimalai.openai_agents import agent_config

    support = agent_config("support", fallback=BASE_INSTRUCTIONS)
    agent = Agent(
        name=support.name,
        instructions=support.instructions,      # saved prompt + skill list, every run
        tools=[lookup_order, *support.tools],   # + load_skill
    )

Tracing every run (what ``agent_config`` also registers, and all ``init(openai_agents=True)`` does)::

    import decimalai
    decimalai.init()

    from decimalai.openai_agents import instrument
    instrument()  # all Agent runs are now traced

Custom path (exclusive — replaces default OpenAI tracing)::

    from decimalai.openai_agents import instrument
    instrument(exclusive=True)

Native skills: the sandbox's own ``Skills`` capability loads the skills
``decimalai skills add`` installed (:func:`sandbox_skills`). Each ``load_skill``
becomes an activation naming the version of the copy the model can read, and the
SkillRouter adds no skill list or tool of its own to that agent::

    from decimalai.openai_agents import sandbox_skills
    SandboxAgent(..., capabilities=[Shell(), sandbox_skills()], default_manifest=Manifest())
"""

from __future__ import annotations

import functools
import json
import logging
import re
import threading
import time
import weakref
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, NoReturn, Optional, Set, Tuple
from uuid import NAMESPACE_OID, UUID, uuid4, uuid5

from ._agent_binding import AgentBinding, attach_prompt_records, is_claude
from ._agent_binding import MissingPromptVariable as MissingPromptVariable
from ._agent_binding import template_variables as template_variables
from ._native import Load, NativeLoads
from ._native import record as _record_native_load
from ._skill_witness import copy_delivered_versions, merge_delivered_versions


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
from .schema.trace import (
    FINAL_OUTPUT_MAX_CHARS,
    LlmCallRecord,
    RunTrace,
    ToolCallRecord,
    TraceSpan,
)

logger = logging.getLogger("decimalai.openai_agents")

# ── Global manifest state ──────────────────────────────────
_manifest_tracker = ManifestTracker()
_manifest_id: Optional[str] = None  # Most recent successful registration
_manifest_lock = threading.Lock()  # Thread safety for manifest registration

# Per-agent manifest id. `_manifest_id` alone is a single process-global
# slot, so in a process running two differently-named agents the second
# agent's traces were stamped with the FIRST agent's manifest — the
# manifest→trace join then attributes one agent's runs to another's
# contract. Keyed by agent_name; `_manifest_id` is kept as the
# last-registered value for back-compat with callers that read it.
_manifest_ids: Dict[str, str] = {}
_manifest_hashes: Dict[str, str] = {}

#: agent -> (snapshot, refusing exception) for a registration the platform REFUSED.
_pending_manifests: Dict[str, tuple] = {}


def _reregister_refused(agent_name: str) -> Optional[str]:
    """Re-attempt ONE agent's refused registration, on the sender's thread."""
    # `_config` is imported per-function in this module, never at module level.
    from . import _config

    pending = _pending_manifests.get(agent_name)
    if not pending:
        return _manifest_ids.get(agent_name)
    snapshot, exc = pending
    if exc is not None and ("401" in str(exc) or "403" in str(exc)):
        return None
    try:
        result = _config._get_client().register_manifest(snapshot)
    except Exception:  # noqa: BLE001 — the sender retries; raising would drop it
        return None
    manifest_id = result.get("manifest_id")
    if not manifest_id:
        return None
    _manifest_ids[agent_name] = manifest_id
    _manifest_hashes[agent_name] = snapshot.manifest_hash
    _pending_manifests.pop(agent_name, None)
    return manifest_id


# Everything this process has ever observed about an agent's structure,
# unioned across traces: ``agent_name -> {"tools": {...}, "models": {...},
# "subagents": {...}, "prompts": {...}}``. See `_declare`.
_declared: Dict[str, Dict[str, Any]] = {}

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


def _clean_names(names: Any) -> List[str]:
    """Keep only non-blank strings."""
    if not isinstance(names, (list, tuple, set)):
        return []
    return [n for n in names if isinstance(n, str) and n.strip()]


# ── Per-run rails, keyed by the Agents SDK's own trace id ───
# Telemetry produced INSIDE a run (the routing decision assembled by the
# instructions callable, the bodies the load_skill tool served, the agent
# object itself) has to reach `_send_trace`, which runs later on the
# processor's `on_trace_end`. Two earlier carriers both fail here:
#
#   ContextVar — the runner awaits `get_system_prompt` and every tool call
#     through `asyncio.gather`, which puts each coroutine in its own Task
#     with a COPIED context. Reads see the outer values; writes are
#     discarded when the task ends, so nothing set in there ever arrives.
#   Router instance state — survives the copy, but one process-global
#     singleton serves every concurrent run, so two parallel `Runner.run`
#     calls drain each other's rails: whichever trace ends first takes the
#     other's routing_id and offered set, and the second reports none.
#
# `get_current_trace()` is a plain contextvar READ, which the copied
# context does propagate, and it returns the same trace id the processor
# sees on `on_trace_end` — so it is the one key both sides agree on.
_run_rails: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_run_rails_lock = threading.Lock()
# A run whose trace never ends (crashed processor, `shutdown()` never
# called) would otherwise pin its rail forever; evict oldest-first.
_RUN_RAILS_MAX = 256


#: The id an untraced run's trace reports: `NoOpTrace.trace_id` is this literal
#: for every run, so it names no run at all.
_UNTRACED_TRACE_ID = "no-op"


class _UntracedRunKey(str):
    """The run key of a run with tracing disabled. A plain string to every
    store it keys; the type says no trace will ever end for it."""

    __slots__ = ()


#: The key each live untraced run is scoped by, held by its trace OBJECT and
#: gone with it. See `_current_run_key`.
_untraced_run_keys: "weakref.WeakKeyDictionary[Any, _UntracedRunKey]" = weakref.WeakKeyDictionary()
_untraced_run_keys_lock = threading.Lock()


def _current_run_key() -> Optional[str]:
    """The key naming the active Agents-SDK run, or None outside one.

    A traced run's key is its trace id. A run with tracing disabled
    (``RunConfig(tracing_disabled=True)``, ``set_tracing_disabled(True)``,
    ``OPENAI_AGENTS_DISABLE_TRACING``) runs under a ``NoOpTrace``, whose id is
    ``"no-op"`` for EVERY run. Keyed on that, all concurrent untraced runs
    shared one scope: one body budget, and one routing decision for their
    loads, so a run's ``load_skill`` named whichever run had routed last, and
    that decides what the platform serves from the public registry.

    The SDK still builds a fresh ``NoOpTrace`` for each run and makes it the
    current trace for exactly that run, so the object is per run even though
    its id is not: an untraced run gets an `_UntracedRunKey` minted once per
    trace object. No processor ever sees a ``NoOpTrace`` end, so nothing
    would ever pop a rail recorded under that key; `_record_run_rail`
    records none.
    """
    try:
        from agents.tracing import get_current_trace
        trace = get_current_trace()
    except Exception:
        return None
    trace_id = getattr(trace, "trace_id", None)
    if not isinstance(trace_id, str):
        return None
    if trace_id != _UNTRACED_TRACE_ID:
        return trace_id
    try:
        with _untraced_run_keys_lock:
            key = _untraced_run_keys.get(trace)
            if key is None:
                key = _UntracedRunKey(f"untraced_{uuid4().hex}")
                _untraced_run_keys[trace] = key
    except TypeError:
        # A trace object that cannot be weakly referenced or hashed names no
        # run, so it gets the unscoped path rather than one shared scope.
        return None
    return key


def _rails_for(run_key: str) -> Dict[str, Any]:
    """The rail dict for one run, created on first use."""
    with _run_rails_lock:
        rail = _run_rails.get(run_key)
        if rail is None:
            rail = {
                "routing_id": None,
                "offered": [],
                "delivered": [],
                "loaded": [],
                "delivered_versions": [],
                "agent": None,
                "user_input": None,
                # Every distinct string `Agent.get_system_prompt` resolved on
                # this run, in turn order. The FALLBACK evidence for what the
                # model was shown; the server's own echo is preferred when the
                # Responses API returns one. See `_attach_system_prompts`.
                "system_prompts": [],
                # Agent `name=` -> the prompt record `agent_config()` made for
                # that agent on this run (version, variables, source).
                "decimal_prompts": {},
            }
            _run_rails[run_key] = rail
            while len(_run_rails) > _RUN_RAILS_MAX:
                _run_rails.popitem(last=False)
        else:
            _run_rails.move_to_end(run_key)
        return rail


def _record_run_rail(
    *,
    routing_id: Optional[str] = None,
    offered: Optional[List[str]] = None,
    delivered: Optional[List[str]] = None,
    loaded: Optional[List[str]] = None,
    delivered_versions: Optional[List[Dict[str, str]]] = None,
    agent: Any = None,
    user_input: Optional[str] = None,
    system_prompt: Optional[str] = None,
    prompt: Optional[Dict[str, Any]] = None,
) -> bool:
    """Stamp routing/loading telemetry onto the CURRENT run's rail.

    Returns False when there is no active run to attribute to (the caller
    then leaves the value on the legacy contextvar/router rails), and for an
    untraced run: its trace never reaches `_send_trace`, the only reader, so
    a rail recorded for it would be held until evicted and read by nobody.
    """
    run_key = _current_run_key()
    if run_key is None or isinstance(run_key, _UntracedRunKey):
        return False
    rail = _rails_for(run_key)
    with _run_rails_lock:
        if routing_id:
            rail["routing_id"] = routing_id
        for key, names in (
            ("offered", offered), ("delivered", delivered), ("loaded", loaded),
        ):
            for name in _clean_names(names):
                if name not in rail[key]:
                    rail[key].append(name)
        rail["delivered_versions"] = merge_delivered_versions(
            rail.get("delivered_versions"),
            copy_delivered_versions(delivered_versions, names=_clean_names(delivered) + _clean_names(loaded)),
        )
        if agent is not None:
            rail["agent"] = agent
        # First turn wins: it carries the run's original ask.
        if user_input and not rail.get("user_input"):
            rail["user_input"] = user_input
        # Turn order, de-duplicated. A run whose prompt never changes leaves
        # ONE entry, which is what lets `_attach_system_prompts` give the same
        # string to every call without guessing at a mapping.
        if isinstance(system_prompt, str) and system_prompt.strip():
            prompts = rail.setdefault("system_prompts", [])
            if system_prompt not in prompts:
                prompts.append(system_prompt)
        # First record per agent wins: a run reads its prompt once.
        if isinstance(prompt, dict) and prompt.get("agent"):
            rail.setdefault("decimal_prompts", {}).setdefault(prompt["agent"], dict(prompt))
    return True


#: How long sending a trace waits for its run's background routing request.
#: A run normally outlasts that request by seconds; this covers a run faster
#: than one HTTP round trip, which would otherwise ship with no routing id.
_PENDING_ROUTING_WAIT_S = 3.0


def _note_pending_routing(run_key: Optional[str], pending: Any) -> None:
    """Keep a traced run's background routing request on its rail."""
    if pending is None or run_key is None or isinstance(run_key, _UntracedRunKey):
        return
    rail = _rails_for(run_key)
    with _run_rails_lock:
        rail.setdefault("pending_routing", []).append(pending)


def _wait_for_pending_routing(run_key: Optional[str]) -> None:
    """Let a run's background routing request finish before its trace is sent."""
    if not run_key:
        return
    with _run_rails_lock:
        rail = _run_rails.get(run_key)
        pending = list(rail.get("pending_routing") or []) if rail else []
    deadline = time.monotonic() + _PENDING_ROUTING_WAIT_S
    for future in pending:
        try:
            future.result(timeout=max(0.0, deadline - time.monotonic()))
        except Exception:
            logger.debug("background routing did not finish before the trace was sent", exc_info=True)


def _run_decimal_agent(run_key: Optional[str], agent_name: Optional[str]) -> Optional[str]:
    """The DecimalAI agent whose agent config served ``agent_name`` on this run."""
    if not run_key or not agent_name:
        return None
    with _run_rails_lock:
        rail = _run_rails.get(run_key)
        record = ((rail or {}).get("decimal_prompts") or {}).get(agent_name)
    return record.get("decimal_agent") if isinstance(record, dict) else None


def _pop_run_rail(run_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read + remove one run's rail. None when the run recorded nothing."""
    if not run_key:
        return None
    with _run_rails_lock:
        return _run_rails.pop(run_key, None)


def _drain_router_rails() -> tuple[Optional[str], List[str], List[str], List[str]]:
    """Drain the Router singleton's process-global instance rails.

    Kept as the fallback for runs that produced no per-run rail (a caller
    driving the Router outside `Runner.run`) and, on every send, as a RESET: an undrained
    global rail would otherwise leak into the next trace. When the run did
    record its own rail, `_send_trace` discards what this returns —
    those names are already attributed correctly and per-run, and under
    concurrency the global copy is a mix of every live run.
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
    # Sanitize here rather than at the merge: `set.update("abc")` on a
    # stray string would silently add three one-letter "skills".
    return (
        routing_id if isinstance(routing_id, str) else None,
        _clean_names(offered),
        _clean_names(delivered),
        _clean_names(loaded),
    )


def _drain_router_loaded_hashes(
    scope: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """``name -> content_hash`` for the bodies load_skill served.

    A SEPARATE drain from `_drain_router_rails` rather than a fifth element on
    its tuple: that tuple is unpacked in this module and mirrored in
    `langchain.py`, and widening it to carry advisory metadata would touch
    every one of those sites for no gain. A router that has never heard of this
    method (an older SDK, or a caller's stand-in) answers `{}` and the adapter
    reports what it always did — the name, with a null hash.

    Both rails are drained: the scoped one because it is this run's own, the
    unscoped one because leaving it would leak into the NEXT trace, exactly as
    `_drain_router_rails` explains for the names. The scoped values win on a
    key collision.

    Safe to read across runs even so, and that is a property of the map rather
    than of the caller: it is keyed by skill NAME, the caller only ever looks up
    names its own rail already claims, and the Router degrades a name loaded at
    two different versions to `None` instead of guessing between them.
    """
    router = _skill_router_singleton
    if router is None:
        return {}
    merged: Dict[str, Optional[str]] = {}
    try:
        merged.update(router.consume_loaded_hashes() or {})
        if scope:
            merged.update(router.consume_loaded_hashes(scope=scope) or {})
    except (AttributeError, TypeError):
        return {}       # a router predating the hash rail, or one taking no scope
    except Exception:
        logger.debug("router hash-rail drain failed (non-fatal)", exc_info=True)
        return {}
    return merged


def _drain_router_delivered_versions(scope: Optional[str]) -> List[Dict[str, str]]:
    """Take only this invocation's versions; discard the shared reset copy."""
    router = _skill_router_singleton
    consume = getattr(router, "consume_delivered_versions", None)
    if not callable(consume):
        return []
    versions = []
    if scope:
        try:
            versions = copy_delivered_versions(consume(scope=scope))
        except Exception:
            logger.debug("scoped body-version drain failed (non-fatal)", exc_info=True)
    try:
        consume()
    except Exception:
        logger.debug("shared body-version reset failed (non-fatal)", exc_info=True)
    return versions


# ── The SkillRouter behind agent_config() ──────────────────────

_skill_router_singleton: Any = None


def _get_skill_router() -> Any:
    """Lazily construct a SkillRouter using the SDK's global config."""
    global _skill_router_singleton
    # The kill switch: a later init(enabled=False) withholds the router (kept
    # for a re-enable) from a loader an earlier init installed.
    from ._config import _disabled_by_init

    if _disabled_by_init():
        return None
    if _skill_router_singleton is not None:
        return _skill_router_singleton
    try:
        from ._config import _get_config
        from .skill_router import SkillRouter
        config = _get_config()
        _skill_router_singleton = SkillRouter(
            api_key=config.api_key,
            base_url=config.base_url,
            # Every routing call names its agent; no router default (see
            # `skill_router._AGENT_FROM_INIT`).
            agent_name=None,
            # agent_config() gives every agent a load_skill tool; bodies are never
            # pasted into the prompt.
            inject_body=False,
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter singleton init failed", exc_info=True)
        return None


# Responses-API input items that carry no user-authored text — a turn whose
# last item is one of these is still routed off the last real user message.
_NON_TEXT_ITEM_TYPES = frozenset({
    "function_call", "function_call_output", "computer_call",
    "computer_call_output", "reasoning", "file_search_call",
    "web_search_call", "code_interpreter_call", "image_generation_call",
})


def _item_text(item: Any) -> str:
    """Flatten one Responses-API input item's `content` to plain text.

    ``content`` is either a bare string or a list of parts
    (``{"type": "input_text", "text": ...}`` / ``output_text`` / ``text`` /
    ``refusal``). Non-text parts (images, audio) contribute nothing.
    """
    content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            text = (
                part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            )
            if not isinstance(text, str):
                text = (
                    part.get("refusal") if isinstance(part, dict)
                    else getattr(part, "refusal", None)
                )
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return ""


def _item_role(item: Any) -> Optional[str]:
    role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
    return role if isinstance(role, str) else None


def _item_type(item: Any) -> Optional[str]:
    itype = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)
    return itype if isinstance(itype, str) else None


def _query_from_input_items(raw: Any) -> Optional[str]:
    """The routable query for a turn: the LAST user message's text.

    ``turn_input`` is the whole conversation the caller handed
    ``Runner.run`` — on turn 6 of a chat that is six items, not one — so
    routing on ``items[0]`` would pin every turn to the opening message and
    routing on the concatenation would drown the current ask in history.
    The last user message IS the current ask. Falls back to the last item
    with any text (e.g. a caller that passes only assistant context), then
    to None (full-menu mode).
    """
    if isinstance(raw, str):
        return raw.strip() or None
    if not isinstance(raw, (list, tuple)) or not raw:
        return None

    fallback: Optional[str] = None
    for item in reversed(raw):
        if _item_type(item) in _NON_TEXT_ITEM_TYPES:
            continue
        text = _item_text(item).strip()
        if not text:
            continue
        if _item_role(item) == "user":
            return text
        if fallback is None:
            fallback = text
    return fallback


def _note_user_input(acc: Any, raw: Any) -> None:
    """Record the run's ORIGINAL ask as ``user_input_preview``, once.

    Nothing used to populate this field, so every trace this adapter sent
    carried ``user_input_preview: null`` — a trace list in which no row shows
    what was asked, and a backend that cannot derive one either.

    The turn's input items ARE the conversation the runner handed the model, so
    the user message is right there; ``_query_from_input_items`` already knows
    how to pull it out (skipping ``function_call`` / ``function_call_output`` /
    ``reasoning`` items, which carry no role and no text). First writer wins:
    later turns of the same run replay the same conversation plus tool traffic,
    and the field is the run's ask, not the latest turn's.
    """
    if getattr(acc, "user_input_preview", None) is not None:
        return
    try:
        text = _query_from_input_items(raw)
    except Exception:  # pragma: no cover - a preview must never break a run
        return
    if text:
        acc.user_input_preview = _preview(text)


def _handle_load_skill(name: str, *, agent_name: Optional[str] = None) -> str:
    """Tool callback — always returns a string the model can act on."""
    router = _get_skill_router()
    if router is None:
        return "load_skill error: skill router unavailable."
    run_key = _current_run_key()
    # Only pass `scope` when there IS a run to scope to, so a router that
    # predates the parameter is untouched on the unscoped path.
    scope_options = {"scope": run_key} if run_key else {}
    kwargs = {**scope_options, **({"agent_name": agent_name} if agent_name else {})}
    try:
        try:
            result = router.load_skill(name, **kwargs)
        except TypeError:
            # Older routers may lack per-run scope. Retain agent context: an
            # unscoped fallback could silently serve a different saved version.
            fallback = {"agent_name": agent_name} if agent_name else {}
            if fallback == kwargs:
                raise
            result = router.load_skill(name, **fallback)
    except TypeError:
        return "load_skill error: this router cannot honor the agent's saved skill configuration."
    except Exception:
        logger.debug("load_skill handler failed (non-fatal)", exc_info=True)
        return f"load_skill error: could not load {name!r} (transient error)."
    if scope_options:
        # The router mirrors each load under the run's scope. This adapter
        # records successful callbacks on its own rail below, so release the
        # router's copy of the names here.
        try:
            router.consume_loaded_names(**scope_options)
        except Exception:
            logger.debug("router keeps no scoped loaded rail to release", exc_info=True)
    # Attribute the load to THIS run rather than to whichever trace next
    # drains the router's shared `_loaded_names`. A body actually served
    # is the only thing that counts as loaded; a budget refusal or a
    # not-found message is not.
    if isinstance(result, str) and result.startswith(f"## Skill: {name}"):
        versions = []
        consume_versions = getattr(router, "consume_delivered_versions", None)
        if run_key and callable(consume_versions):
            try:
                # A run-scoped drain can include another parallel callback's
                # body. Keep every witnessed delivery with its own name;
                # filtering to this callback would discard the other's pin.
                versions = copy_delivered_versions(consume_versions(scope=run_key))
            except Exception:
                logger.debug("load_skill body-version drain failed (non-fatal)", exc_info=True)
        _record_run_rail(
            delivered=[version["name"] for version in versions],
            loaded=[name],
            delivered_versions=versions,
        )
        if isinstance(run_key, _UntracedRunKey):
            # The router files each load's version under the run's scope for
            # `_send_trace` to drain, and an untraced run never sends one.
            try:
                router.consume_loaded_hashes(scope=run_key)
            except Exception:
                logger.debug("router keeps no scoped hash rail to release", exc_info=True)
    return result


_SKILL_TOOL_AGENT_ATTR = "__decimalai_skill_agent_name__"

#: The name of `agent_config()`'s tool.
_TOOL_NAME = "load_skill"
#: Every name this adapter's own tool can have. An SDK affordance, not part of
#: the agent's declared contract, so the manifest never lists it.
_SKILL_TOOL_NAMES = frozenset({_TOOL_NAME})


def _has_native_skills(agent: Any) -> bool:
    """Whether the sandbox's own ``Skills`` capability serves this agent's skills.

    Read off ``capabilities``, which the sandbox's execution clone and the agent the
    user built both carry: the eager form (``Skills(skills=...)``, ``from_=``)
    registers no tool, so the tool list cannot say it is there.
    """
    try:
        from agents.sandbox.capabilities import Skills
    except Exception:  # an openai-agents with no sandbox: no native loader
        return False
    try:
        return any(isinstance(c, Skills) for c in (getattr(agent, "capabilities", None) or ()))
    except TypeError:
        return False


#: (DecimalAI name, Agent name) pairs `_note_name_override` has already logged.
_name_overrides_noted: set = set()


def _note_name_override(decimal_name: Optional[str], agent_name: Optional[str]) -> None:
    """Say once per pair, at INFO, that a run of ``Agent(name=agent_name)`` was
    filed and routed as ``decimal_name``. Without it, someone who named an Agent
    "triage" and finds no "triage" on the dashboard has nothing to go on."""
    if not decimal_name or not agent_name or decimal_name == agent_name:
        return
    key = (decimal_name, agent_name)
    if key in _name_overrides_noted:
        return
    _name_overrides_noted.add(key)
    logger.info(
        "DecimalAI: runs of Agent(name=%r) are traced and routed as agent %r, the "
        "name instrument()/init() was given — the agent the skills are attached to.",
        agent_name, decimal_name,
    )


# ── Native skills: the sandbox's own `Skills` capability ─────────────
#
# `Skills(lazy_from=...)` gives the model a `load_skill` tool whose result is only
# `{"status", "skill_name", "path"}`: it copies the skill folder into the sandbox
# workspace, and the model then reads `SKILL.md` from that copy. So the bytes the
# model gets are the COPY's, and the hook reads the copy, through the sandbox
# session, right after `load_skill` made it, never the source folder (a session
# kept across runs keeps serving its copy after `decimalai skills update`).
# Observed by the `Skills` subclass `sandbox_skills()` returns (the sandbox clones
# the capability per run, and the clone keeps its class); the function span
# carries only the status dict. A plain `Skills(...)` is not observed: nothing in
# the Agents SDK is patched. The eager forms (`skills=`, `from_=`) mount every
# skill up front and have no load step, so only their listing is seen.

#: A SKILL.md larger than this is not read back to identify it.
_MAX_NATIVE_SKILL_MD = 8 * 1024 * 1024
_two_catalogs_warned = False


def _run_native_loads() -> Optional[NativeLoads]:
    """This run's native-load collector, held on its rail; None when no trace of this
    adapter's will carry it (an untraced run, or no DecimalAI processor registered).
    Then `record` falls back to a `decimalai.start_trace()` trace, or says it dropped it.
    """
    run_key = _current_run_key()
    if run_key is None or isinstance(run_key, _UntracedRunKey):
        return None
    processor = _tracing_processor
    if processor is None or not _processor_is_registered(processor):
        return None
    rail = _rails_for(run_key)
    with _run_rails_lock:
        loads = rail.get("native")
        if loads is None:
            loads = rail["native"] = NativeLoads()
    return loads


def _source_index(skills: Any) -> Any:
    """The lock of the project the capability's local source folder is in."""
    from pathlib import Path

    from .skillpkg.identity import current_index

    src = getattr(getattr(getattr(skills, "lazy_from", None), "source", None), "src", None)
    if src is None:
        src = getattr(getattr(skills, "from_", None), "src", None)
    if src is None:
        return current_index()
    # The SDK resolves a relative source against the working directory; so does this.
    return current_index(Path(src) if Path(src).is_absolute() else Path.cwd() / src)


async def _observe_sandbox_load(skills: Any, skill_name: Any, result: Any) -> None:
    """Record one `load_skill` from the copy the model will read. Never raises."""
    try:
        if not isinstance(result, dict) or result.get("status") not in ("loaded", "already_loaded"):
            return
        from pathlib import Path, PurePosixPath

        session = getattr(skills, "session", None)
        folder = PurePosixPath(str(result.get("path") or skill_name)).name
        rel = PurePosixPath(str(getattr(skills, "skills_path", ".agents"))) / folder
        text: Optional[str] = None
        if session is not None:
            path = Path(str(session.state.manifest.root)) / str(rel) / "SKILL.md"
            # The backend session, not its instrumented wrapper: the wrapper would
            # add a read the model never made to the user's own trace.
            reader = getattr(session, "_inner", None) or session
            handle = await reader.read(path, user=getattr(skills, "run_as", None))
            try:
                data = handle.read(_MAX_NATIVE_SKILL_MD + 1)
            finally:
                handle.close()
            if isinstance(data, bytes):
                data = data.decode("utf-8", errors="replace")
            if isinstance(data, str) and len(data) <= _MAX_NATIVE_SKILL_MD:
                text = data
        name = result.get("skill_name") if isinstance(result.get("skill_name"), str) else skill_name
        _record_native_load(
            Load(framework="openai-agents", name=name, text=text, path=str(rel)),
            loads=_run_native_loads(), index=_source_index(skills),
        )
    except Exception:  # noqa: BLE001 - observing must never break the user's run
        logger.debug("native load_skill observation failed (non-fatal)", exc_info=True)


def _observe_sandbox_listing(skills: Any) -> None:
    """Record the DecimalAI skills in the listing the sandbox put in the prompt.

    Only locked skills are named, by their wire name: a name the lock does not hold
    says nothing the platform can attribute. Never raises.
    """
    try:
        loads = _run_native_loads()
        metadata = getattr(skills, "_skills_metadata", None)  # what `instructions` listed
        if loads is None or not metadata:
            return
        index = _source_index(skills)
        listed = []
        for item in metadata:
            skill = index.by_install_name(getattr(item, "name", None) or "")
            if skill is not None:
                listed.append(skill.wire_name)
        loads.listed(listed)
    except Exception:  # noqa: BLE001
        logger.debug("native skill listing observation failed (non-fatal)", exc_info=True)


def _warn_two_catalogs(agent_name: Optional[str]) -> None:
    global _two_catalogs_warned
    if _two_catalogs_warned:
        return
    _two_catalogs_warned = True
    logger.warning(
        "DecimalAI: a run of %r loaded a skill with the sandbox Skills capability and also "
        "carried DecimalAI's routed skill list, so two skill catalogs reached the model. "
        "The capability was out of the adapter's sight when it injected (for instance on "
        "another agent of the run); give the skills to one of the two.",
        agent_name or "<unnamed>",
    )


def sandbox_skills(directory: Any = None) -> Any:
    """The sandbox ``Skills`` capability over the skills ``decimalai skills add`` installed.

    ::

        from decimalai.openai_agents import sandbox_skills
        agent = SandboxAgent(..., capabilities=[Shell(), sandbox_skills()],
                             default_manifest=Manifest())

    The source is :func:`decimalai.skills_dir` relative to the working directory: the
    SDK lists nothing from a source outside it, so a project that is not under the
    working directory raises here instead of listing zero skills.

    Two guards the plain capability lacks:

    * **An empty listing says why.** The sandbox lists only real folders holding a
      ``SKILL.md`` and skips a symlinked one without a word, so a project whose skills are
      symlinks gives the model no skills and no error. A WARNING here names the folder.
    * **An unknown name does not end the run.** The sandbox's ``load_skill`` raises on a
      name it cannot find (or that two folders share), which aborts the whole run; the
      capability returned here answers the model instead, with the names it can load.

    Args:
        directory: where to look for the project (default: the working directory).
    """
    from pathlib import Path

    try:
        from agents.sandbox.capabilities import LocalDirLazySkillSource
        from agents.sandbox.entries import LocalDir
    except ImportError:
        raise ImportError(
            "sandbox_skills() needs the OpenAI Agents SDK with its sandbox: "
            "pip install \"decimalai[openai-agents]\""
        ) from None
    from .skillpkg.project import skills_dir

    source = skills_dir(directory, relative_to=Path.cwd())
    _warn_if_nothing_to_list(Path.cwd() / source)
    return _guarded_skills_class()(lazy_from=LocalDirLazySkillSource(source=LocalDir(src=source)))


#: Built on first use: it subclasses the SDK's ``Skills``, which may not be installed.
_GUARDED_SKILLS: Any = None
#: What the sandbox raises for a name it cannot load: the guard answers these.
_UNLOADABLE = ("lazy skill not found", "lazy skill name is ambiguous")


def _guarded_skills_class() -> Any:
    global _GUARDED_SKILLS
    if _GUARDED_SKILLS is not None:
        return _GUARDED_SKILLS
    from agents.sandbox.capabilities import Skills
    from agents.sandbox.errors import SkillsConfigError

    class DecimalSkills(Skills):
        """``Skills`` that records what it lists and loads as this run's activations,
        and answers an unloadable name instead of raising."""

        async def instructions(self, *args: Any, **kwargs: Any) -> Any:
            text = await super().instructions(*args, **kwargs)
            if text:
                _observe_sandbox_listing(self)
            return text

        async def load_skill(self, skill_name: str) -> Dict[str, str]:
            result = await self._load_or_explain(skill_name)
            await _observe_sandbox_load(self, skill_name, result)
            return result

        async def _load_or_explain(self, skill_name: str) -> Dict[str, str]:
            try:
                return await super().load_skill(skill_name)
            except SkillsConfigError as exc:
                if getattr(exc, "message", None) not in _UNLOADABLE:
                    raise
                available = _loadable_names(self)
                return {
                    "status": "not_found",
                    "skill_name": skill_name,
                    "error": (f"No skill can be loaded as {skill_name!r} "
                              f"({getattr(exc, 'message', exc)}). "
                              + (f"Load one of: {', '.join(available)}." if available
                                 else "No skills are available.")),
                }

    _GUARDED_SKILLS = DecimalSkills
    return DecimalSkills


def _loadable_names(skills: Any) -> List[str]:
    try:
        session = getattr(skills, "session", None)
        grants = session.state.manifest.extra_path_grants if session is not None else ()
        return sorted({m.name for m in skills.lazy_from.list_skill_metadata(
            skills_path=skills.skills_path, source_grants=grants)})
    except Exception:  # noqa: BLE001 - the answer is still better than an aborted run
        logger.debug("could not list the sandbox's skills", exc_info=True)
        return []


def _warn_if_nothing_to_list(folder: Any) -> None:
    """Say so when the sandbox would list no skills from ``folder``: it lists real folders
    holding a ``SKILL.md`` and skips symlinks silently."""
    try:
        entries = [p for p in folder.iterdir() if p.is_dir()]
        real = [p for p in entries if not p.is_symlink() and (p / "SKILL.md").is_file()]
        if real:
            return
        linked = [p.name for p in entries if p.is_symlink()]
        logger.warning(
            "decimalai: sandbox_skills(): the sandbox will list no skills from %s%s. Run "
            "`decimalai skills install` to write real skill folders there.", folder,
            f" (it skips symlinked folders: {', '.join(sorted(linked))})" if linked else
            " (it holds no skill folder with a SKILL.md)")
    except OSError:
        logger.debug("could not check the sandbox skill folder", exc_info=True)

# ── Type aliases for duck-typing against the OpenAI Agents SDK ──
# We avoid hard imports so the module loads even without openai-agents installed.
# At runtime, the actual Trace and Span objects are passed to us by the SDK.


def _agent_model_name(agent: Any) -> Optional[str]:
    """Resolve the model NAME an Agent declares.

    ``agent.model`` is either a bare string ("gpt-5-mini") or a Model
    instance (e.g. an OpenAIChatCompletionsModel pointed at a non-OpenAI
    provider). ``str()`` on a Model instance yields a useless object repr,
    so pull ``.model`` / ``.name``.
    """
    model = getattr(agent, "model", None)
    if model is None:
        return None
    if isinstance(model, str):
        return model or None
    name = getattr(model, "model", None) or getattr(model, "name", None)
    return name if isinstance(name, str) and name else None


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

    # Extract tools with full schemas. load_skill is excluded: it is an SDK
    # affordance this adapter attaches, not part of the agent's declared
    # contract — including it would make the manifest (and therefore the
    # version history) differ purely on whether the skill loader was on.
    agent_tools = getattr(agent, "tools", None) or []
    if agent_tools:
        tools_list = []
        for t in agent_tools:
            name = getattr(t, "name", None) or str(t)
            if name in _SKILL_TOOL_NAMES:
                continue
            tool_entry: Dict[str, Any] = {"name": name}
            # FunctionTool carries params_json_schema; hosted tools carry none.
            schema = getattr(t, "params_json_schema", None)
            if schema:
                tool_entry["schema"] = schema
            tools_list.append(tool_entry)
        if tools_list:
            result["tools"] = tools_list

    # Extract instructions as prompt. An agent config's instructions is
    # its bound method; declare the saved prompt TEMPLATE (variables unfilled),
    # which is what the dashboard versions.
    instructions = getattr(agent, "instructions", None)
    if callable(instructions):
        owner = getattr(instructions, "__self__", None)
        instructions = owner._template() if isinstance(owner, OpenAIAgentConfig) else None
    if instructions and isinstance(instructions, str):
        result["prompts"] = {"system": instructions}

    model_name = _agent_model_name(agent)
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


#: The ONE `DecimalTracingProcessor` this process registers, shared by every
#: `instrument()` and `agent_config()` call. Two on the provider would send each
#: run twice, and only the copy that ended first would carry the routing id and
#: the loads (it pops the run's rail): double the trace volume, and per-skill
#: rates computed over a population half of which could never show an activation.
_tracing_processor: Optional["DecimalTracingProcessor"] = None
#: Whether `_tracing_processor` was ever handed to the provider. Consulted only
#: when the provider's own list cannot be read (a custom TraceProvider).
_tracing_processor_added = False
_tracing_processor_lock = threading.Lock()


def _processor_is_registered(processor: Any) -> bool:
    """Whether ``processor`` is on the Agents SDK's live processor list.

    The SDK has no public getter. Its `DefaultTraceProvider` keeps the list on
    `_multi_processor._processors`, a tuple it replaces under a lock, which is
    what this reads — so a processor the user removed with
    `set_trace_processors(...)` is registered again by the next `instrument()`.
    A custom provider may keep its list anywhere; then the answer is whether we
    ever registered it.
    """
    try:
        from agents.tracing import get_trace_provider

        live = tuple(get_trace_provider()._multi_processor._processors)
    except Exception:
        return _tracing_processor_added
    return any(p is processor for p in live)


def _register_processor(
    agent_name: Optional[str],
    skills_registry: Optional[List[Dict[str, Any]]],
    *,
    exclusive: bool,
    add_trace_processor: Any,
    set_trace_processors: Any,
    agent_binding: Optional[Tuple[str, str]] = None,
    fallback_agent_name: Optional[str] = None,
) -> "DecimalTracingProcessor":
    """Register this process's one processor, configured by THIS call.

    A repeat call reconfigures the existing processor rather than adding a
    second: its skills registry becomes this call's (the same registry a single
    call would have produced), and the process's DecimalAI agent name changes
    only when this call names one — `instrument()` or `agent_config()` after
    `init(agent_name="support", openai_agents=True)` keeps "support". With no
    name from this call or an earlier one, it is the agent `decimalai.init()`
    named (or DECIMAL_AGENT_NAME). ``agent_binding`` is ``(Agent name,
    DecimalAI name)`` for a name given with ``agent=``.
    """
    global _tracing_processor, _tracing_processor_added
    from ._config import default_agent_name

    with _tracing_processor_lock:
        processor = _tracing_processor
        if agent_name is None and (processor is None or processor.default_agent_name is None):
            agent_name = default_agent_name()
        if processor is None:
            processor = _tracing_processor = DecimalTracingProcessor(
                agent_name=agent_name, skills_registry=skills_registry,
            )
        else:
            if agent_name is not None:
                processor.default_agent_name = agent_name
            processor._skills_registry = skills_registry or []
        if agent_binding is not None:
            processor.agent_names[agent_binding[0]] = agent_binding[1]
        if fallback_agent_name is not None:
            processor.fallback_agent_name = fallback_agent_name
        if exclusive:
            set_trace_processors([processor])
        elif not _processor_is_registered(processor):
            add_trace_processor(processor)
        _tracing_processor_added = True
        return processor


def instrument(
    agent_name: Optional[str] = None,
    *,
    agent: Any = None,
    exclusive: bool = False,
    skills: Optional[List[Dict[str, Any]]] = None,
    skill_dirs: Optional[List[str]] = None,
    disk_sync: Optional[bool] = None,
) -> None:
    """Register DecimalAI as a trace processor for OpenAI Agents SDK.

    After calling it, every ``Runner.run()`` / ``Runner.run_sync()`` call is
    traced and sent to DecimalAI. Tracing only: it changes no Agents SDK class
    and gives no agent a prompt or skills. For those, use :func:`agent_config`.

    Args:
        agent_name: The DecimalAI agent traces are filed under, over the Agent's
            own ``name=``. Given with ``agent=``, it is that Agent's (and any
            Agent of the same ``name=``); every other Agent keeps its own. Given
            alone, it is every Agent's, and so is the agent ``decimalai.init()``
            named (or ``DECIMAL_AGENT_NAME``) when no call names one; a later
            call that names none keeps it. A run of an agent built from
            :func:`agent_config` is always filed under that config's name. With no
            DecimalAI name at all, a trace is named by its root Agent's ``name=``.
        agent: Optional ``agents.Agent`` instance. If provided, DecimalAI
            will introspect it for full tool schemas, instructions, model,
            and handoffs to register a manifest immediately. This gives
            the highest fidelity manifest (full JSON schemas for tools).
            If omitted, manifest is auto-detected from span data at trace
            time (tool names only, no schemas). Its ``name`` names a trace
            only when nothing else does.
        exclusive: If True, replaces ALL existing trace processors (including
            OpenAI's default). If False (default), adds alongside existing
            processors so traces still go to the OpenAI dashboard too.
        skills: Skill descriptors to declare for the agent's manifest (and to
            sync to the platform when ``disk_sync`` is on).
        skill_dirs: Folders to read ``SKILL.md`` files from. Naming them asks
            for them to be read: ``disk_sync`` then defaults to on.
        disk_sync: Read the project's ``SKILL.md`` files and sync the ones you
            wrote to the platform. Off by default, because an OpenAI agent's
            skills come from DecimalAI through :func:`agent_config`; on with
            ``skill_dirs``, or with ``init(skill_authority="harness")``.

    Safe to call more than once. There is only ever one DecimalAI processor,
    so each run is sent once; a later call replaces the skills registry and
    (when it names one) the DecimalAI agent name.

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

    # An OpenAI agent's skills come from DecimalAI's router, through agent_config(),
    # so "auto" is router authority: local SKILL.md files are read only when asked
    # (disk_sync=True, skill_dirs=, or the harness authority). An explicit
    # disk_sync=... always wins.
    if disk_sync is None:
        if skill_dirs:
            disk_sync = True
        else:
            try:
                from ._config import _get_config
                disk_sync = _get_config().resolve_disk_sync(loader_active=True)
            except Exception:
                disk_sync = False

    # A name given WITH an Agent is that Agent's DecimalAI name, not the
    # process's; the Agent's own name= is only ever a fallback. Either one
    # process-wide would outrank every other Agent's name= in the process.
    own_name = getattr(agent, "name", None) if agent is not None else None
    agent_binding = (own_name, agent_name) if own_name and agent_name is not None else None
    process_name = None if agent_binding is not None else agent_name
    fallback_name = (agent_name or own_name) if agent is not None else None

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

    # Sync the skills you wrote up to the platform for observability, gated on
    # disk_sync=True. When False, the SDK treats the platform as the sole source
    # of truth (useful for pure-Python stacks that don't use a disk-loading
    # runtime, or to suppress duplicate injection when the SkillRouter loader is
    # the active injector). Nothing here writes disk: the background pull that
    # mirrored every org skill into the working directory is gone, and registry
    # skills reach disk only when a `decimalai skills` command runs. Folders that
    # command installed are never uploaded (that would copy a registry skill into
    # your org).
    sync_job = None
    if disk_sync and resolved_skills:
        try:
            from ._config import _get_config

            config = _get_config()

            def _sync_skills_background(sync_name: Optional[str]):
                try:
                    from .skill_router import SkillRouter, _sync_up_items
                    from .skills import _with_local_timestamps

                    items, managed = _sync_up_items(_with_local_timestamps(resolved_skills))
                    if managed:
                        logger.debug(
                            "Not syncing %d skill(s) `decimalai skills` installed: %s",
                            len(managed), ", ".join(managed),
                        )
                    if not items:
                        return
                    router = SkillRouter(
                        api_key=config.api_key,
                        base_url=config.base_url,
                    )
                    try:
                        from ._install import get_install_identity
                        _ident = get_install_identity()
                    except Exception:
                        _ident = {}
                    result = router.sync_skills(
                        items,
                        # Don't blind-clobber a teammate's newer dashboard
                        # edit on agent boot; record this checkout's baseline.
                        conflict_policy="newer_wins",
                        install_id=_ident.get("install_id"),
                        install_label=_ident.get("install_label"),
                        # A skill written for this agent goes on this agent (new
                        # skills are opt-in), under the name its traces carry. One
                        # a person removed from the agent on the dashboard is not
                        # added back on the next boot.
                        agent_name=sync_name,
                        add_to_agent="unless_removed",
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

            sync_job = _sync_skills_background
        except Exception:
            logger.debug("Skill sync setup failed (non-fatal)", exc_info=True)

    processor = _register_processor(
        process_name,
        resolved_skills,
        exclusive=exclusive,
        add_trace_processor=add_trace_processor,
        set_trace_processors=set_trace_processors,
        agent_binding=agent_binding,
        fallback_agent_name=fallback_name,
    )
    agent_name = processor.decimal_name_for(own_name)

    if sync_job is not None:
        # Submitted only now, under the name this Agent's runs are filed under: its DecimalAI name
        # (instrument(agent_name=...) or init's), and the Agent's own name= only when there is no
        # other. Syncing under name= ("Assistant") made a phantom agent that got the skills while
        # the agent its traces named got none.
        sync_name = agent_name or own_name
        try:
            from ._config import _sender

            _sender.submit(lambda: sync_job(sync_name))
        except Exception:
            logger.debug("Skill sync setup failed (non-fatal)", exc_info=True)

    # If an Agent object was provided, introspect it for manifest — under the
    # name its traces will carry.
    if agent is not None:
        _register_manifest_from_agent(agent, agent_name or own_name, resolved_skills)

    logger.info(
        "DecimalAI OpenAI Agents tracing installed (agent_name=%s, exclusive=%s, agent_introspected=%s, disk_sync=%s)",
        agent_name,
        exclusive,
        agent is not None,
        disk_sync,
    )


# ── agent_config(): an agent's DecimalAI prompt and skills, attached by you ───
#
# The supported way to give an OpenAI Agents SDK agent its DecimalAI prompt and
# skills. Nothing in the OpenAI SDK is patched: the developer puts the two pieces
# into their own Agent(...), using features OpenAI documents —
#
#     support = agent_config("support", fallback=BASE_INSTRUCTIONS)
#     agent = Agent(name=support.name, instructions=support.instructions,
#                   tools=[lookup_order, *support.tools])
#
# `support.instructions` is a dynamic-instructions function (OpenAI calls it
# before each model request): the agent's saved prompt with any {{variables}}
# filled from `Runner.run(context=...)`, then the agent's skill list.
#
# Every run is routed on the customer's message. When the agent's whole menu fits
# the platform's caps, a run shows the local copy and records its own decision
# (ranked on its message) in the background, so it never waits; when the menu is
# CUT to fit, which rows make it depends on the message, so the run is routed
# before its model call. The first run always routes. The prompt is a local copy
# re-read at most every `_PROMPT_REFRESH_SECONDS`. `support.tools` is one
# `load_skill` FunctionTool. Each run's record (prompt version, variable names,
# skills offered / loaded / delivered) goes on the run rail, which
# `DecimalTracingProcessor` sends.

#: `getattr` default telling "the SDK has no `turn_input`" from "it is empty".
_NO_TURN_INPUT = object()


def _require_agents_sdk() -> None:
    """Fail before any setup or network call when the Agents SDK is missing."""
    try:
        import agents  # noqa: F401
    except ImportError:
        raise ImportError(
            "agent_config() needs the OpenAI Agents SDK: "
            "pip install \"decimalai[openai-agents]\""
        ) from None


def _ensure_tracing() -> None:
    """Register this process's trace processor; no Agents SDK class is touched."""
    try:
        from agents.tracing import add_trace_processor, set_trace_processors
    except ImportError:
        raise ImportError(
            "agent_config() needs the OpenAI Agents SDK: "
            "pip install \"decimalai[openai-agents]\""
        ) from None
    if _tracing_processor is not None and _processor_is_registered(_tracing_processor):
        return
    _register_processor(
        None, None, exclusive=False,
        add_trace_processor=add_trace_processor,
        set_trace_processors=set_trace_processors,
    )


class OpenAIAgentConfig(AgentBinding):
    """One agent's prompt and skills in DecimalAI, for an OpenAI Agents SDK ``Agent``.

    Made by :func:`agent_config`. Put its pieces into your own agent::

        support = agent_config("support", fallback=BASE_INSTRUCTIONS)
        agent = Agent(name=support.name, instructions=support.instructions,
                      tools=[lookup_order, *support.tools])

    Attributes:
        name: The agent's canonical DecimalAI name.
        instructions: Pass as ``Agent(instructions=...)``. OpenAI calls it before
            each model request; it returns the agent's prompt (variables
            filled from ``Runner.run(context=...)``), then its skill list.
        tools: The tools DecimalAI adds — today one, ``load_skill``.
        load_skill: That tool.
    """

    _log = logger
    _VALUES_HINT = "Pass them in Runner.run(..., context=...), as attributes or dict keys."

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str) -> None:
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        self.load_skill = self._build_tool()
        self.tools: List[Any] = [self.load_skill]

    def __repr__(self) -> str:
        return (
            f"OpenAIAgentConfig(name={self.name!r}, prompt_version={self.prompt_version!r}, "
            f"skills={self.skill_names!r})"
        )

    # ── the dynamic instructions ────────────────────────────
    async def instructions(self, context: Any, agent: Any) -> str:
        """The agent's instructions for this run. OpenAI calls it; you don't."""
        run_key = _current_run_key()
        state = await self._prepare_run(context, agent, run_key)
        if state["text"] is None:
            state["text"] = self._compose(context, agent, run_key, state)
            self._warn_if_claude_replays_thinking(context, agent)
        return state["text"]

    def _warn_if_claude_replays_thinking(self, context: Any, agent: Any) -> None:
        """Say once when a Claude chat sends thinking back: a saved prompt would break it.

        Claude rejects a turn whose system prompt or tools changed since a thinking
        block in its history was made (HTTP 400, "bound to a different conversation").
        These instructions follow the newest saved prompt and skill list on every run,
        and the Agents SDK gives them no conversation to keep them per chat by.
        """
        model = getattr(agent, "model", None)
        if not is_claude(model if isinstance(model, str) else getattr(model, "model", None)):
            return
        items = getattr(context, "turn_input", None)
        if not isinstance(items, list) or not any(
                isinstance(item, dict) and item.get("type") == "reasoning" for item in items):
            return
        self._warn_once(agent, (
            "this Claude chat sends earlier turns' thinking back, and Claude rejects a turn whose "
            "instructions or tools changed since (HTTP 400, \"bound to a different "
            "conversation\"). These instructions follow the newest saved prompt and skill list on "
            "every run, so a prompt saved in DecimalAI breaks open chats. Leave "
            "ModelSettings.reasoning unset (thinking is then not sent back), or set Anthropic's "
            "thinking.block_binding.prefix_mismatch_behavior=\"drop_block\" (beta "
            "thinking-binding-controls-2026-08-01) to drop the stale blocks instead."))

    # ── the seams AgentBinding asks for ─────────────────────
    def _router(self) -> Any:
        return _get_skill_router()

    def _run_key(self) -> Optional[str]:
        return _current_run_key()

    def _values(self, context: Any) -> Any:
        return getattr(context, "context", None)

    def _record(self, **fields: Any) -> bool:
        return _record_run_rail(**fields)

    def _note_pending(self, run_key: Optional[str], pending: Any) -> None:
        _note_pending_routing(run_key, pending)

    def _stamp_routing(self, run_key: Optional[str], routing_id: str) -> None:
        # Only a traced run's rail is ever sent, and only while it is still open.
        if run_key is None or isinstance(run_key, _UntracedRunKey):
            return
        with _run_rails_lock:
            still_open = run_key in _run_rails
        if still_open:
            _record_run_rail(routing_id=routing_id)

    def _on_compose(self, agent: Any) -> None:
        self._bind_agent_name(agent)

    def _on_menu(self, names: List[str]) -> None:
        self.load_skill.params_json_schema = self._tool_schema(names)

    def _serve_load(self, name: str) -> str:
        return _handle_load_skill(name, agent_name=self.name)

    def _message(self, context: Any, agent: Any) -> Optional[str]:
        """The customer's latest message, which routing ranks the skills on.

        Read off ``RunContextWrapper.turn_input``: the Agents SDK fills it with the
        turn's input before it asks for the instructions, but does not document
        it. Read only. Without it the agent still runs, with its skills in the
        platform's fixed order, and this says so once.
        """
        raw = getattr(context, "turn_input", _NO_TURN_INPUT)
        if raw is _NO_TURN_INPUT:
            self._warn_once(agent, (
                "this OpenAI Agents SDK version does not expose the run's input "
                "(RunContextWrapper.turn_input), so skills are listed in a fixed order "
                "instead of being ranked on each message."
            ))
            return None
        try:
            return _query_from_input_items(raw)
        except Exception:
            logger.debug("could not read the run's message (non-fatal)", exc_info=True)
            return None

    def _bind_agent_name(self, agent: Any) -> None:
        """File this Agent's runs under the DecimalAI name, whatever its ``name=``."""
        own = getattr(agent, "name", None)
        processor = _tracing_processor
        if own and processor is not None and processor.agent_names.get(own) != self.name:
            processor.agent_names[own] = self.name

    # ── the load_skill tool ─────────────────────────────────
    def _build_tool(self) -> Any:
        try:
            from agents import FunctionTool
        except ImportError:
            raise ImportError(
                "agent_config() needs the OpenAI Agents SDK: "
                "pip install \"decimalai[openai-agents]\""
            ) from None
        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION

        async def invoke(tool_context: Any, raw_arguments: str) -> str:
            try:
                name = str((json.loads(raw_arguments or "{}") or {}).get("name") or "").strip()
            except (ValueError, AttributeError):
                name = ""
            return await self._load(name)

        tool = FunctionTool(
            name=_TOOL_NAME,
            description=LOAD_SKILL_TOOL_DESCRIPTION,
            params_json_schema=self._tool_schema([]),
            on_invoke_tool=invoke,
            strict_json_schema=True,
            is_enabled=self._tool_enabled,
        )
        setattr(tool, _SKILL_TOOL_AGENT_ATTR, self.name)
        return tool

    def _skills_blocked(self, agent: Any, *, tool: bool = True, inject: bool = False) -> Optional[str]:
        """Why DecimalAI's skills cannot work on ``agent``, or None when they can."""
        base = super()._skills_blocked(agent, tool=tool, inject=inject)
        if base:
            return base
        if tool and getattr(agent, "tool_use_behavior", None) == "stop_on_first_tool":
            return ("with tool_use_behavior='stop_on_first_tool' a skill's text would become "
                    "the answer. Use StopAtTools([...your tools]) instead.")
        if _has_native_skills(agent):
            return ("it loads skills with the sandbox Skills capability. Use "
                    "decimalai.openai_agents.sandbox_skills() there instead.")
        if not tool:
            return None  # the text arrives in the instructions; the tool is not needed
        tools = list(getattr(agent, "tools", None) or [])
        if any(getattr(t, "name", None) == _TOOL_NAME and t is not self.load_skill for t in tools):
            return "it already has a tool named load_skill. Rename that tool to use both."
        if not any(t is self.load_skill for t in tools):
            # A list that tells the model to call a tool it does not have: the
            # Agents SDK ends the run with "tool not found" when it tries.
            return (f"its tools do not include agent_config({self.name!r}).tools, so it gets "
                    "its prompt without the skill list. Add *support.tools to "
                    "Agent(tools=[...]) to give it its skills.")
        return None

    async def _tool_enabled(self, context: Any, agent: Any) -> bool:
        behavior = getattr(agent, "tool_use_behavior", None)
        if callable(behavior) and not isinstance(behavior, str):
            self._warn_once(agent, (
                "this agent's tool_use_behavior function also receives load_skill results; "
                "ignore them there, or a skill's text could become the answer."
            ))
        # A plain string cannot be ours. (A callable may be: a SandboxAgent run
        # executes a clone whose instructions is the sandbox's builder around ours,
        # and a developer may wrap ours in a function of their own.)
        if isinstance(getattr(agent, "instructions", None), str):
            self._warn_once(agent, (
                "load_skill is on this agent, but its instructions are a fixed string, not "
                f"agent_config({self.name!r}).instructions, so the model gets no skill list."
            ))
        state = await self._prepare_run(context, agent, _current_run_key())
        return bool(state["names"]) and state["tool"]



def agent_config(
    agent_name: str,
    *,
    fallback: Optional[str] = None,
    on_unavailable: str = "continue",
) -> OpenAIAgentConfig:
    """Pull an agent's prompt and skills from DecimalAI, for an OpenAI Agents SDK agent.

    ::

        from agents import Agent, Runner
        from decimalai.openai_agents import agent_config

        support = agent_config("support", fallback=BASE_INSTRUCTIONS)
        agent = Agent(
            name=support.name,
            instructions=support.instructions,
            tools=[lookup_order, *support.tools],
        )
        result = await Runner.run(agent, question, context=Customer(...))

    Reads the agent's saved prompt here and
    keeps a local copy, re-read in the background at most once a minute. The
    skill list is routed on each run's message: an agent whose skills all fit
    the list shows a local copy and records each run's decision in the
    background, so no run waits; an agent with more skills than fit is routed
    before each model call (about one HTTP round trip), so the list is picked for
    that message. The first run always routes. Nothing in the OpenAI SDK is
    patched and no other agent in the process is affected. Initializes the SDK
    from ``DECIMAL_API_KEY`` when ``decimalai.init()`` has not been called.

    The model reads a listed skill's full text with the ``load_skill`` tool. With
    that tool switched off (``DECIMALAI_LOAD_SKILL_TOOL=0`` or
    ``init(load_skill_tool=False)``), every run is routed on its message and the
    text of the skills picked for it goes into the instructions instead
    (``inject_skill_body``, on by default once the tool is off).

    With no API key and no ``init()``, the agent runs keyless, as
    :func:`decimalai.agent_config` describes: its ``fallback`` is the prompt, each
    run lists the skills the project lists in ``.decimal/skills.toml``, and nothing
    is sent. With a key, an agent the workspace has no agent by is created.

    Args:
        agent_name: The agent's name in DecimalAI.
        fallback: Instructions to use when the agent has no saved prompt, when no
            API key is set, or when DecimalAI cannot be reached at startup. With a
            key and without one, an unreachable DecimalAI raises here rather than
            starting an agent with no instructions.
        on_unavailable: ``"continue"`` (default) runs without skills when
            DecimalAI never returned them, and the run's record says so;
            ``"fail"`` raises before the model call instead.

    Raises:
        DecimalConfigError: no API key is set in a project ``.decimal/link.toml``
            connects to a workspace.
        ValueError: ``on_unavailable`` is neither ``"continue"`` nor ``"fail"``.
    """
    if on_unavailable not in ("continue", "fail"):
        raise ValueError("on_unavailable must be 'continue' or 'fail'")
    _require_agents_sdk()
    from ._agent import _open_prompt

    prompt = _open_prompt(agent_name, fallback)
    config = OpenAIAgentConfig(prompt, fallback=fallback, on_unavailable=on_unavailable)
    if not prompt.keyless:  # keyless runs are traced nowhere
        _ensure_tracing()
    return config


def load_agent(*args: Any, **kwargs: Any) -> NoReturn:
    """Renamed in decimalai 0.21: :func:`agent_config`."""
    raise TypeError(
        "decimalai.openai_agents.load_agent() was renamed agent_config() in decimalai 0.21; "
        "it takes the same arguments and returns the same config: "
        "from decimalai.openai_agents import agent_config"
    )


def _declare(
    agent_name: str,
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    models: Optional[Dict[str, Dict[str, Any]]] = None,
    subagents: Optional[List[Dict[str, Any]]] = None,
    prompts: Optional[Dict[str, str]] = None,
    from_agent: bool = False,
) -> Dict[str, Any]:
    """Merge what one observation saw into the agent's running declaration.

    The manifest an agent registers must describe the AGENT, not whichever
    slice of it a particular run happened to expose. Registering per-run
    slices makes the declared surface flap — a run that never reached the
    tool loop looks like "all tools removed", which agentversion classifies
    breaking/major and the platform turns into a replay verdict and a
    fabricated version bump. So the declaration only ever grows: each
    observation is unioned in, and the union is what gets registered.

    ``from_agent`` marks an observation taken off the ``Agent`` object
    itself — the author's declaration, as opposed to what a response
    happened to report.
    """
    with _manifest_lock:
        state = _declared.setdefault(
            agent_name,
            {"tools": {}, "models": {}, "subagents": {}, "prompts": {},
             "models_declared": set()},
        )
        for tool in tools or ():
            name = tool.get("name")
            if not name:
                continue
            # A later observation with a full JSON schema beats an earlier
            # name-only one (span data carries names; the Agent object carries
            # schemas), but never the reverse.
            if len(tool) >= len(state["tools"].get(name, {})):
                state["tools"][name] = tool
        for key, cfg in (models or {}).items():
            if not cfg:
                continue
            # The model the agent DECLARES wins over the one a response
            # reports. They differ — `model="gpt-4.1-mini"` comes back as
            # `gpt-4.1-mini-2025-04-14` — and letting the resolved snapshot
            # into the contract means every silent snapshot rotation at the
            # provider mints a breaking version bump nobody asked for. What
            # actually ran is already on the trace: `llm_calls[].model_name`
            # carries the resolved id per call. The manifest states the
            # declaration; the trace states the execution.
            if key in state["models_declared"] and not from_agent:
                continue
            state["models"][key] = cfg
            if from_agent:
                state["models_declared"].add(key)
        for sub in subagents or ():
            name = sub.get("name")
            if name:
                state["subagents"][name] = sub
        for key, text in (prompts or {}).items():
            if text:
                state["prompts"][key] = text
        return {
            "tools": list(state["tools"].values()) or None,
            "models": dict(state["models"]) or None,
            "subagents": list(state["subagents"].values()) or None,
            "prompts": dict(state["prompts"]) or None,
        }


def _adopt_existing_manifest(agent_name: str) -> Optional[str]:
    """The id of the manifest this agent is already registered under.

    Used when a run has NOTHING structural to declare (no tool, no model —
    a guardrail tripwire, a run that died before its first model call).
    Registering an empty manifest there would be a lie in the loud
    direction: the diff engine reads a contract that goes from empty to
    populated as ``provider: '' → 'openai'``, which is breaking/major, so
    one unlucky first run fabricates a "replay everything" version bump on
    the next healthy one. Pointing the trace at the contract already on
    file says the honest thing instead — this run declared nothing new.
    """
    from . import _config

    try:
        client = _config._get_client()
        result = client.list_manifests(agent_name=agent_name, limit=10)
    except Exception:
        logger.debug("manifest lookup for %s failed (non-fatal)", agent_name, exc_info=True)
        return None
    rows = (result or {}).get("manifests") if isinstance(result, dict) else None
    if not rows:
        return None
    # Newest-first; prefer the ACTIVE row (a superseded one would attribute
    # the run to a contract the agent has already moved off).
    candidates = [r for r in rows if isinstance(r, dict)]
    for row in candidates:
        if row.get("status") == "active" and isinstance(row.get("id"), str):
            return row["id"]
    for row in candidates:
        if isinstance(row.get("id"), str):
            return row["id"]
    return None


def _register_snapshot(agent_name: str, snapshot: Any) -> Optional[str]:
    """Register one snapshot, returning the manifest id (None on failure).

    Dedup is keyed by (agent, hash). ``ManifestTracker`` is a single slot
    and the hash it stores does NOT include the agent name, so two agents
    with the same structure in one process deduped against each other and
    the second one's traces came back with no manifest at all.
    """
    global _manifest_id
    from . import _config

    with _manifest_lock:
        # A caller (notably a test fixture) that swapped in a fresh
        # ManifestTracker is asking for registration state to be forgotten;
        # honour that for the per-agent map too, or the reset is a no-op.
        if _manifest_tracker.last_hash is None and _manifest_hashes:
            _manifest_hashes.clear()
            _manifest_ids.clear()
        known = _manifest_ids.get(agent_name)
        if known and _manifest_hashes.get(agent_name) == snapshot.manifest_hash:
            return known  # Same agent, same structure — already registered
        _manifest_tracker.check_and_update(snapshot)
        try:
            client = _config._get_client()
            result = client.register_manifest(snapshot)
            manifest_id = result.get("manifest_id", snapshot.id)
        except Exception as exc:
            # Leave the per-agent hash unset so the next trace retries; a
            # transient blip must not permanently stop this agent from ever
            # declaring a manifest.
            # One line: DecimalAI being unreachable is handled, not a crash.
            # The traceback is at DEBUG for whoever needs it.
            logger.warning(
                "decimalai: could not register %s's manifest (%s: %s); its traces "
                "are held and retried in the background.",
                agent_name, type(exc).__name__, str(exc)[:200],
            )
            logger.debug("Manifest registration for %s failed", agent_name, exc_info=True)
            # Hand the send side something to retry WITH. Returning None here ships the
            # trace with no manifest_id, which a strict backend answers 400
            # "manifest_id is required" — the trace is lost. The attempt above runs on
            # the caller's thread in front of the user's agent and cannot outlast a
            # no-available-instance window, so the retry belongs on the sender.
            _pending_manifests[agent_name] = (snapshot, exc)
            return None
        _manifest_ids[agent_name] = manifest_id
        _manifest_hashes[agent_name] = snapshot.manifest_hash
        _pending_manifests.pop(agent_name, None)
        _manifest_id = manifest_id
        logger.info(
            "Registered manifest %s for %s (hash=%s, components=%d)",
            manifest_id,
            agent_name,
            snapshot.manifest_hash[:12],
            len(snapshot.components),
        )
        return manifest_id


def _register_manifest_from_agent(
    agent: Any,
    agent_name: Optional[str],
    skills: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Register a manifest by introspecting an Agent object at install time."""
    from . import _config

    if not _config._is_enabled():
        return

    try:
        data = _introspect_agent(agent)
        resolved_name = agent_name or getattr(agent, "name", "unknown")
        declared = _declare(
            resolved_name,
            tools=data.get("tools"),
            models=data.get("models"),
            subagents=data.get("subagents"),
            prompts=data.get("prompts"),
            from_agent=True,
        )

        snapshot = extract_from_config(
            agent_name=resolved_name,
            tools=declared["tools"],
            prompts=declared["prompts"],
            models=declared["models"],
            subagents=declared["subagents"],
            skills=skills,
        )
        _register_snapshot(resolved_name, snapshot)
    except Exception:
        logger.warning("Failed to register manifest from Agent introspection", exc_info=True)


#: Span types whose ``error`` does NOT mean the run failed. See
#: ``DecimalTracingProcessor._note_span_error`` for why each one is here.
_RECOVERABLE_SPAN_TYPES = frozenset({"function", "custom"})

#: Longest text a tool span preview, or a tool call's recorded result, carries.
#: 2000 is what the platform keeps: ingest cuts every span preview there, so a
#: longer one would be cut anyway, and the LangChain adapter's previews use the
#: same bound. The old 200 stopped inside the first sentence or two of a loaded
#: skill body, which is the text a reader opens a skill-loading trace to see.
_TOOL_TEXT_LIMIT = 2000


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
        self.final_output: Optional[str] = None
        self.status = Status.SUCCESS
        self.error_message: Optional[str] = None
        # Manifest auto-detection accumulators
        self.seen_tools: Dict[str, Dict[str, Any]] = {}  # name -> {name, ...}
        self.seen_model: Optional[Dict[str, Any]] = None
        self.seen_handoffs: List[str] = []
        self.seen_guardrails: List[str] = []
        self.active_skills: Dict[str, Optional[str]] = {}
        # SkillRouter: routing decision that surfaced the skills for this
        # trace. Stamped at trace-end from the run's rail, so
        # `routing_decision × trace_skill_activation` joins can close.
        self.routing_id: Optional[str] = None
        # Skill Rater discovery telemetry. ``skills_offered_in_prompt``
        # is auto-populated by the dynamic-instructions callable from the
        # Router's offered set; ``skills_loaded_by_agent`` is auto-populated
        # from the Router's loaded-names rail when the native load_skill
        # tool serves a body (drained at trace-send), plus any manual
        # annotations (use `decimalai.log_skill_loaded`).
        self.skills_offered_in_prompt: set[str] = set()
        self.skills_loaded_by_agent: set[str] = set()
        # Bodies that reached the model (Router body injection) —
        # between offered and activated; never implies activation.
        self.skills_delivered: set[str] = set()
        self.skills_delivered_versions: List[Dict[str, str]] = []
        # llm_call id -> the system prompt the SERVER echoed back for that
        # call (`Response.instructions`). `ResponseSpanData` has no
        # instructions slot, so this side table is where the system half of
        # the prompt waits until `_attach_system_prompts` splices it onto
        # `rendered_input` at trace-send — before the skill inference reads
        # it, so a skill carried in the instructions is visible to it.
        # (It was held until AFTER, back when the inference wrote ACTIVATION
        # and the menu would have been promoted wholesale. The inference
        # writes offered/delivered now; see the call site in `_send_trace`.)
        self.system_prompt_by_call: Dict[Any, str] = {}
        # Tool calls a model turn asked for whose run has not been seen yet:
        # (the record on that turn's llm_call, the names its function span can
        # carry, the arguments as the model wrote them). `_handle_function`
        # takes the matching entry and fills in the result, so each call is ONE
        # record from request to result. An entry still here at trace end was
        # never seen to run (a handoff, an approval that never came) and stays
        # `running`: requested, outcome unseen.
        self.open_tool_calls: List[Tuple[ToolCallRecord, Set[str], str]] = []

    @property
    def live_agent(self) -> Any:
        """The `agents.Agent` object this run is executing, if we saw it.

        Recorded by an agent config's instructions, which the runner
        calls on every turn — including turns that go on to fail. It is the
        only source of the agent's declared model and tool schemas for runs
        whose spans carry neither.
        """
        with _run_rails_lock:
            rail = _run_rails.get(self.trace_id)
            return rail.get("agent") if rail else None

    @property
    def declared_model_name(self) -> Optional[str]:
        agent = self.live_agent
        return _agent_model_name(agent) if agent is not None else None


class DecimalTracingProcessor:
    """OpenAI Agents SDK TracingProcessor that captures traces for DecimalAI.

    Implements the ``TracingProcessor`` protocol via duck typing (no
    inheritance required — the SDK checks methods, not class hierarchy).

    Thread-safe: each trace gets its own ``_TraceAccumulator`` keyed by
    ``trace_id``, protected by a lock.
    """

    def __init__(self, agent_name: Optional[str] = None, skills_registry: Optional[List[Dict[str, Any]]] = None):
        # The process's DecimalAI name: `instrument(agent_name=...)` without
        # `agent=`, else the one init() named. When set it names every trace
        # no agent config served, over the root Agent's own `name=`.
        self.default_agent_name = agent_name
        # Agent `name=` -> DecimalAI name, from `instrument(agent=a,
        # agent_name=...)`: a name given WITH an Agent is that Agent's, ahead of
        # the process's, and leaves every other Agent alone. A process that
        # installs once per agent ("each its own agent") would otherwise file
        # every run under whichever install came last.
        self.agent_names: Dict[str, str] = {}
        # The name for a trace no agent span named: the Agent handed to
        # `instrument(agent=...)`, by its DecimalAI name when it was given one.
        self.fallback_agent_name: Optional[str] = None
        self._skills_registry = skills_registry or []
        self._traces: Dict[str, _TraceAccumulator] = {}
        self._lock = threading.Lock()

    def decimal_name_for(self, agent_name: Optional[str]) -> Optional[str]:
        """The DecimalAI name of the Agent called ``agent_name``: its own from
        ``instrument(agent=..., agent_name=...)``, else the process's. None
        when there is neither."""
        own = self.agent_names.get(agent_name) if agent_name else None
        return own or self.default_agent_name

    def _agent_name(self, acc: _TraceAccumulator) -> Optional[str]:
        """The agent this trace is filed under: the agent config that
        served the root Agent on this run, else the root Agent's DecimalAI name,
        else its ``name=`` (the root agent span's), else the fallback.

        The run's own record comes first: two configs whose Agents share one
        ``name=`` would otherwise file each other's concurrent runs under
        whichever ran last. The DecimalAI name used to be the LAST resort, under
        the root agent's own name, against `instrument()`'s own docstring
        ("Default agent name for all traces"), so ``init(agent_name=...)``
        named nothing on this adapter.
        """
        return (
            _run_decimal_agent(acc.trace_id, acc.agent_name)
            or self.decimal_name_for(acc.agent_name)
            or acc.agent_name
            or self.fallback_agent_name
        )

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

        self._note_span_error(span, span_type, acc)

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

    def _note_span_error(self, span: Any, span_type: Optional[str], acc: _TraceAccumulator) -> None:
        """Fail the RUN when a span reports an error.

        ``acc.status`` used to be ``Status.SUCCESS`` from construction to send,
        with nothing anywhere able to change it — so a run that raised (a bad
        model id, an unreachable endpoint, a guardrail tripwire) was ingested as
        a SUCCESS. Every error-rate figure derived from these traces read zero,
        which is worse than having no trace at all: it is a confident wrong
        answer to "is this agent healthy?".

        The Agents SDK already carries the signal — ``Span.error`` is set by the
        runner via ``set_error()``. Two span types are deliberately EXCLUDED:

        * ``function`` — a tool that raises is handed back to the model as an
          error string by the SDK's default tool-error handler, and the agent
          routinely recovers and answers. The tool's own span and
          ``ToolCallRecord`` still carry ERROR (``_handle_function``); the run
          does not.
        * ``custom`` — ``custom_span(...).set_error()`` is the caller's own
          annotation, with the caller's own meaning. Reading it as "the run
          failed" would put this adapter's interpretation on their span.

        Everything else (agent, turn, response, generation, handoff, guardrail,
        and any span type a future SDK adds) does fail the run, because on this
        contract the expensive mistake is silence about a failure, not an
        over-loud error.
        """
        error = getattr(span, "error", None)
        if not error or span_type in _RECOVERABLE_SPAN_TYPES:
            return
        message = error.get("message") if isinstance(error, dict) else None
        if not isinstance(message, str) or not message:
            message = str(error)
        data = error.get("data") if isinstance(error, dict) else None
        if isinstance(data, dict) and data.get("error"):
            message = f"{message}: {_stringify(data['error'])}"
        with self._lock:
            acc.status = Status.ERROR
            # First error wins — it is the one that ended the run; the ones
            # after it are that failure unwinding through the enclosing spans.
            if acc.error_message is None:
                acc.error_message = message[:2000]

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

        _note_user_input(acc, raw_input)

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

        # Parse token usage. Every model the SDK ships (chat completions,
        # LiteLLM, any-llm) writes the Responses-style keys onto the span.
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        cache_read_tokens = _span_usage_cached_tokens(usage)

        # Latency
        latency_ms = None
        if started_at and ended_at:
            latency_ms = int((ended_at - started_at).total_seconds() * 1000)

        # The tool calls this turn asked for. On this path they are already in
        # `output` (the chat message carries them), but not on the record.
        requested = _requested_tool_calls(
            raw_output if isinstance(raw_output, (list, tuple)) else [raw_output]
        )

        # Error handling
        span_error = getattr(span, "error", None)
        status = Status.SUCCESS
        finish_reason = FinishReason.TOOL_CALLS if requested else FinishReason.STOP
        if span_error:
            status = Status.ERROR
            finish_reason = FinishReason.ERROR
            if output_dict is None:
                error_msg = span_error.get("message", "") if isinstance(span_error, dict) else str(span_error)
                output_dict = {"error": error_msg[:500]}

        call = LlmCallRecord(
            id=uuid4(),
            span_id=parent_id,
            agent_name=self._agent_name(acc),
            provider=provider,
            model_name=model,
            temperature=model_config.get("temperature") if isinstance(model_config, dict) else None,
            max_output_tokens=model_config.get("max_tokens") if isinstance(model_config, dict) else None,
            rendered_input=rendered_input,
            output=output_dict,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
        )
        self._record_requested_tool_calls(acc, call, requested)
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
        """Map a FunctionSpanData to a TraceSpan, and its result onto the
        ToolCallRecord of the model turn that asked for this call."""
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
            input_preview=str(func_input)[:_TOOL_TEXT_LIMIT] if func_input else None,
            output_preview=str(func_output)[:_TOOL_TEXT_LIMIT] if func_output else None,
        )
        acc.spans.append(trace_span)

        record = self._claim_requested_tool_call(acc, func_name, func_input)
        if record is not None:
            # The record IS this span: the backend reads tool calls from both
            # the records and the tool spans and joins them on id, so a record
            # with an id of its own was counted as a second call.
            record.id = span_id
            record.status = status
            record.result = (
                None if func_output is None else _stringify(func_output)[:_TOOL_TEXT_LIMIT]
            )
            record.latency_ms = latency_ms

    def _record_requested_tool_calls(
        self, acc: _TraceAccumulator, call: LlmCallRecord, requested: List[Dict[str, Any]],
    ) -> None:
        """Put the calls a model turn asked for on its llm_call, as requested.

        `running` until the tool's own span is seen: the turn only shows that
        the model ASKED. `_handle_function` records what the tool returned.
        """
        for req in requested:
            record = ToolCallRecord(
                tool_name=req["name"],
                status=Status.RUNNING,
                args=_tool_call_args(req["arguments"]),
            )
            call.tool_calls.append(record)
            # The SDK names a namespaced tool's span `namespace.name`.
            names = {req["name"]}
            if req["namespace"]:
                names.add(f"{req['namespace']}.{req['name']}")
            with self._lock:
                acc.open_tool_calls.append((record, names, req["arguments"]))

    def _claim_requested_tool_call(
        self, acc: _TraceAccumulator, name: Any, arguments: Any,
    ) -> Optional[ToolCallRecord]:
        """The open request a finished function span answers, removed from the
        open list, or None. A function span carries the tool name and the
        arguments string but no call id, so the match is on the name, and on
        the arguments where two open calls share one: parallel calls to the
        same tool can finish in any order."""
        with self._lock:
            matches = [i for i, (_, names, _) in enumerate(acc.open_tool_calls) if name in names]
            if not matches:
                return None
            exact = [i for i in matches if acc.open_tool_calls[i][2] == arguments]
            record, _, _ = acc.open_tool_calls.pop((exact or matches)[0])
            return record

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

        # Accumulate tools for manifest (names only from span data).
        # The skill tool (under either of its names) is skipped for the same
        # reason `_introspect_agent` skips it: it is an SDK affordance this
        # adapter attaches at run time, so counting it would make the declared
        # tool set — and therefore the manifest version — depend on whether
        # the skill loader happened to be on.
        span_tools = getattr(span_data, "tools", None) or []
        for tool_name in span_tools:
            name_str = str(tool_name)
            if name_str in _SKILL_TOOL_NAMES:
                continue
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
        """Map a ResponseSpanData (Responses API call) to LlmCallRecord + TraceSpan.

        The default OpenAI Agents path (OpenAIResponsesModel) emits `response`
        spans — not `generation` spans — so this handler owns LLM-call capture
        (model, tokens, latency) for most runs.
        """
        span_id = _coerce_span_id(getattr(span, "span_id", None)) or uuid4()
        parent_id = _coerce_span_id(getattr(span, "parent_id", None))
        started_at = _parse_iso(getattr(span, "started_at", None))
        ended_at = _parse_iso(getattr(span, "ended_at", None))

        response_obj = getattr(span_data, "response", None)
        response_id = None
        model = None
        input_tokens = None
        output_tokens = None
        cache_read_tokens = None
        temperature = None
        max_output_tokens = None
        if response_obj is not None:
            response_id = getattr(response_obj, "id", None)
            raw_model = getattr(response_obj, "model", None)
            model = raw_model if isinstance(raw_model, str) else None
            usage = getattr(response_obj, "usage", None)
            if usage is not None:
                _in = getattr(usage, "input_tokens", None)
                _out = getattr(usage, "output_tokens", None)
                input_tokens = _in if isinstance(_in, int) else None
                output_tokens = _out if isinstance(_out, int) else None
                cache_read_tokens = _reported_cached_tokens(usage)
            _temp = getattr(response_obj, "temperature", None)
            temperature = _temp if isinstance(_temp, (int, float)) else None
            _max = getattr(response_obj, "max_output_tokens", None)
            max_output_tokens = _max if isinstance(_max, int) else None
        # Fallback: the SDK also stamps a usage dict on the span data itself
        # (populated on streaming paths where response.usage may be absent).
        if input_tokens is None and output_tokens is None:
            usage_dict = getattr(span_data, "usage", None)
            if isinstance(usage_dict, dict):
                _in = usage_dict.get("input_tokens")
                _out = usage_dict.get("output_tokens")
                input_tokens = _in if isinstance(_in, int) else None
                output_tokens = _out if isinstance(_out, int) else None
                cache_read_tokens = _span_usage_cached_tokens(usage_dict)

        # Accumulate model info for manifest auto-detection
        if model and acc.seen_model is None:
            acc.seen_model = {
                "provider": _infer_provider(model),
                "model": model,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
            }

        raw_input = getattr(span_data, "input", None)
        _note_user_input(acc, raw_input)
        output_text = _response_output_text(response_obj) if response_obj else None
        # The tool calls the model asked for. A turn that calls a tool usually
        # has no text at all, so keeping only text recorded the turn that
        # decided to call `load_skill` as one that said nothing and stopped.
        requested = (
            _requested_tool_calls(getattr(response_obj, "output", None))
            if response_obj is not None
            else []
        )

        # The system half of the prompt, as the SERVER reports having received
        # it. `span_data.input` is the input-items list only, so the
        # instructions — which is where the skills menu lives — are simply not
        # on the span. `Response.instructions` is the API's own echo of the
        # value it was sent, which makes this a round-trip receipt rather than
        # our own hook marking its own homework. Only a plain string counts:
        # the field also admits a list of input items, and a shape we cannot
        # render verbatim is one we decline to claim.
        echoed_instructions = (
            getattr(response_obj, "instructions", None)
            if response_obj is not None
            else None
        )
        if not (isinstance(echoed_instructions, str) and echoed_instructions.strip()):
            echoed_instructions = None

        latency_ms = None
        if started_at and ended_at:
            latency_ms = int((ended_at - started_at).total_seconds() * 1000)

        # Error handling
        span_error = getattr(span, "error", None)
        status = Status.SUCCESS
        finish_reason = FinishReason.TOOL_CALLS if requested else FinishReason.STOP
        output_dict: Optional[Dict[str, Any]] = {}
        if output_text:
            output_dict["content"] = output_text
        if requested:
            # The chat-completions shape, which is the one that carries the
            # call id and the one the trace view renders as a tool request.
            output_dict["tool_calls"] = [
                {
                    "id": req["call_id"],
                    "type": "function",
                    "function": {"name": req["name"], "arguments": req["arguments"]},
                }
                for req in requested
            ]
        output_dict = output_dict or None
        if span_error:
            status = Status.ERROR
            finish_reason = FinishReason.ERROR
            if output_dict is None:
                error_msg = span_error.get("message", "") if isinstance(span_error, dict) else str(span_error)
                output_dict = {"error": error_msg[:500]}

        # A failed call has no Response object and so no model name — but
        # ingest REJECTS an llm_call without one ("llm_calls[0]:
        # 'model_name' is required"), which threw away the whole trace,
        # errors included. Fall back to the model the agent declared;
        # if even that is unknown, keep the error on the span and skip
        # the record rather than poison the trace with it.
        call_model = model or acc.declared_model_name
        if response_obj is not None or (span_error and call_model):
            call = LlmCallRecord(
                id=uuid4(),
                span_id=parent_id,
                agent_name=self._agent_name(acc),
                provider=_infer_provider(call_model),
                model_name=call_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                rendered_input=_normalize_messages(raw_input),
                output=output_dict,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                latency_ms=latency_ms,
                finish_reason=finish_reason,
                status=status,
                started_at=started_at,
                ended_at=ended_at,
            )
            self._record_requested_tool_calls(acc, call, requested)
            acc.llm_calls.append(call)
            if echoed_instructions:
                acc.system_prompt_by_call[call.id] = echoed_instructions

        trace_span = TraceSpan(
            id=span_id,
            parent_span_id=parent_id,
            span_type=SpanType.LLM if response_obj is not None or span_error else SpanType.OTHER,
            name=f"response:{model}" if model else "response",
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            input_preview=_preview(raw_input),
            output_preview=_preview(output_text),
            attributes={"response_id": response_id} if response_id else {},
        )
        acc.spans.append(trace_span)

        # Capture final output from the response — the extracted text, not the
        # ResponseOutputMessage repr. Text-less turns (pure tool calls) don't
        # overwrite a preview captured from an earlier turn. The whole answer
        # goes with it: the 200-character preview is all evals and the
        # platform's judges used to see.
        if output_text:
            acc.final_output_preview = _preview(output_text)
            acc.final_output = _preview(output_text, FINAL_OUTPUT_MAX_CHARS)

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

        # A handoff is a function call the model made (`transfer_to_<agent>`),
        # run by this span rather than a function span. Without this its record
        # stayed `running` for good, which reads as a tool call that never
        # returned. A handoff given a custom tool name is not matched here and
        # keeps `running`: the span names the target agent, not the tool.
        if isinstance(to_agent, str) and to_agent:
            record = self._claim_requested_tool_call(
                acc, _handoff_tool_name(to_agent), None,
            )
            if record is not None:
                record.id = span_id
                record.status = Status.SUCCESS
                record.result = f"Handed off to {to_agent}"
                if started_at and ended_at:
                    record.latency_ms = int((ended_at - started_at).total_seconds() * 1000)

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

        agent_name = self._agent_name(acc) or acc.trace_name
        _note_name_override(self.decimal_name_for(acc.agent_name), acc.agent_name)
        ended_at = datetime.now(timezone.utc)

        # Auto-register manifest from span data + the live Agent object
        manifest_id = self._maybe_register_manifest(acc, agent_name)

        # This run's own rail — routing decision, delivered bodies, loads —
        # keyed by the trace id both the instructions callable and this
        # method see. Popped, so it can't leak into a later trace.
        _wait_for_pending_routing(acc.trace_id)
        run_rail = _pop_run_rail(acc.trace_id) or {}

        # Drain the Router's process-global rails unconditionally: an
        # undrained rail leaks into the NEXT trace. When this run recorded
        # its own, the global copy is DISCARDED rather than merged — under
        # concurrency it holds a mix of every live run, and merging it is
        # exactly how one trace used to steal another's routing_id and
        # offered set.
        g_routing_id, g_offered, g_delivered, g_loaded = _drain_router_rails()
        if run_rail:
            rail_routing_id = run_rail.get("routing_id")
            rail_offered = _clean_names(run_rail.get("offered"))
            rail_delivered = _clean_names(run_rail.get("delivered"))
            rail_loaded = _clean_names(run_rail.get("loaded"))
        else:
            rail_routing_id, rail_offered, rail_delivered, rail_loaded = (
                g_routing_id, g_offered, g_delivered, g_loaded,
            )

        # The run's routing decision and what it offered, off its rail: written
        # by `agent_config()`'s instructions, which the runner calls AFTER
        # on_trace_start, in a Task of its own. Merged with any direct
        # `log_skill_offered` calls already on the accumulator.
        if acc.routing_id is None:
            acc.routing_id = rail_routing_id
        if rail_offered:
            acc.skills_offered_in_prompt.update(rail_offered)
        if rail_delivered:
            acc.skills_delivered.update(rail_delivered)
            acc.skills_offered_in_prompt.update(rail_delivered)

        # Bodies the load_skill tool served mid-run. Loaded implies
        # offered + delivered — same ladder semantics as
        # `log_skill_loaded` on the generic tracer.
        if rail_loaded:
            acc.skills_loaded_by_agent.update(rail_loaded)
            acc.skills_delivered.update(rail_loaded)
            acc.skills_offered_in_prompt.update(rail_loaded)

        acc.skills_delivered_versions = merge_delivered_versions(
            acc.skills_delivered_versions, run_rail.get("delivered_versions"),
        )
        acc.skills_delivered_versions = merge_delivered_versions(
            acc.skills_delivered_versions, _drain_router_delivered_versions(acc.trace_id),
        )

        # The VERSION of each body the model read. Drained unconditionally, for
        # the same leak reason the names are, and stamped only onto names this
        # run already claims as loaded.
        #
        # This CANNOT change which skills the trace reports as activated, and
        # that is the point: every name written here is already in
        # `skills_loaded_by_agent`, which the backend unions into the activation
        # set and dedupes by name with the `active_skills` entry winning
        # (trace_service._record_skill_activations). So the entry replaces a
        # string with a dict of the same name, and the only observable
        # difference is `TraceSkillActivation.skill_hash` — null before this,
        # which is what broke the join from a measured lift to the skill VERSION
        # that produced it.
        #
        # A name with NO hash is deliberately not written: it would be an entry
        # carrying nothing the plain string does not already carry.
        _loaded_hashes = _drain_router_loaded_hashes(acc.trace_id)
        for _name in acc.skills_loaded_by_agent:
            _digest = _loaded_hashes.get(_name)
            if _digest and _name not in acc.active_skills:
                acc.active_skills[_name] = _digest

        # Put the system half of the prompt back on `rendered_input`. STRICTLY
        # BEFORE `_infer_skill_rungs` below, and that order is a REVERSAL —
        # read this before moving it back.
        #
        # `ResponseSpanData` has no instructions slot (only response, input,
        # usage and a response id), so on
        # the Responses path (the SDK default) the span carries the input items
        # ALONE and this splice is the ONLY way the instructions ever reach
        # `rendered_input`. Splicing after the inference therefore made the
        # instructions invisible to it, and putting a skill in the agent
        # instructions is the ordinary way to use one on this SDK: a
        # disk-discovered skill whose text lived there read as an EMPTY
        # offered/delivered rung on every trace, while its body sat on the
        # shipped record for anyone to see.
        #
        # The old order was NOT arbitrary. The inference used to write
        # ACTIVATION, so splicing the skills menu — which names every OFFERED
        # skill — in first would have reported the whole menu as activated.
        # That is gone: `_infer_skill_rungs` writes `skills_offered_in_prompt`
        # and `skills_delivered` only (see its body), nothing on this adapter
        # writes `acc.active_skills` at all, and `acc.skills_loaded_by_agent`
        # is written from the `loaded` rail alone — a body `_handle_load_skill`
        # saw the router actually serve. No ordering can route prompt text to
        # either.
        #
        # What still protects the DELIVERED rung is the precedence rule, not
        # this ordering: the rail merges above run first, so `infer_prompt_rungs`
        # subtracts every name the router already accounted for and cannot
        # re-read the router's own menu as evidence. That is why the merges
        # stay above and the inference stays below.
        _attach_system_prompts(acc, run_rail.get("system_prompts"))

        # Infer offered/delivered for DISK skills the SDK did not inject.
        # STRICTLY AFTER the rail merges above: the precedence rule needs this
        # run's full router-accounted set, or a skill the router only put a
        # menu row for could be re-inferred here as delivered.
        self._infer_skill_rungs(acc)

        # Build active_skills list. The ONE writer of `acc.active_skills` on
        # this adapter is the hash stamp above, and it only ever re-states a
        # name `skills_loaded_by_agent` already carries — so this list still
        # names no skill the model did not demonstrably read. Nothing else
        # writes it: not a rail, and explicitly not the inference above.
        # `log_skill_activation` fills the equivalent field on the GENERIC
        # tracer, which is a different accumulator and never reaches this one,
        # so on the Agents SDK the activation SIGNAL remains
        # `skills_loaded_by_agent`: a body the model asked for by calling
        # load_skill. What this list adds is the VERSION of that body.
        active_skills_list: List[Dict[str, Any]] = []
        for name, h in acc.active_skills.items():
            entry: Dict[str, Any] = {"name": name}
            if h:
                entry["hash"] = h
            active_skills_list.append(entry)

        attach_prompt_records(acc.spans, run_rail.get("decimal_prompts"))

        trace = RunTrace(
            id=uuid4(),
            agent_name=agent_name,
            status=acc.status,
            source_type="production",
            started_at=acc.started_at or ended_at,
            ended_at=ended_at,
            user_input_preview=acc.user_input_preview or _preview(run_rail.get("user_input")),
            final_output_preview=acc.final_output_preview,
            final_output=acc.final_output,
            error_message=acc.error_message,
            spans=acc.spans,
            llm_calls=acc.llm_calls,
            active_skills=active_skills_list,
            manifest_id=manifest_id,
            parent_trace_id=get_parent_trace(),
            routing_id=acc.routing_id,
            # Sorted for deterministic output (tests, diffs).
            skills_offered_in_prompt=sorted(acc.skills_offered_in_prompt),
            skills_loaded_by_agent=sorted(acc.skills_loaded_by_agent),
            skills_delivered=sorted(acc.skills_delivered),
            skills_delivered_versions=copy_delivered_versions(
                acc.skills_delivered_versions, names=acc.skills_delivered,
            ),
        )

        # What the sandbox's own Skills capability listed and loaded on this run, with
        # the version of the copy the model read (see `_observe_sandbox_load`).
        native = run_rail.get("native")
        if native:
            try:
                native.apply(trace)
                if native.activations and (rail_offered or rail_delivered or rail_loaded):
                    _warn_two_catalogs(agent_name)
            except Exception:  # noqa: BLE001 - a trace without them beats no trace
                logger.debug("native skill loads not applied (non-fatal)", exc_info=True)

        try:
            client = _config._get_client()
            _agent = trace.agent_name
            if _agent in _pending_manifests:
                _config.submit_trace_pending_manifest(
                    client, trace, lambda: _reregister_refused(_agent)
                )
            else:
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

    def _infer_skill_rungs(self, acc: _TraceAccumulator) -> None:
        """Infer OFFERED / DELIVERED for disk skills the SDK did not inject.

        ``_skills_registry`` is disk-derived, so it describes skills a harness
        may have injected itself. Prompt text can show they were put in front
        of the model; it can never show the model reaching for one, so nothing
        here writes ``acc.active_skills``. On this rail the activation signal
        is ``_handle_load_skill`` — which records a load only when the router
        actually returned a body.

        Must run AFTER this run's rails are merged into ``acc``: the router's
        own names are excluded from the inference (see ``infer_prompt_rungs``).
        Must also run after ``_attach_system_prompts``, which is what puts the
        instructions on ``rendered_input`` — on the Responses path the span
        carries the input items alone, so before that splice the haystack read
        here is missing the half of the prompt a harness injects skills into.
        """
        if not self._skills_registry or not acc.llm_calls:
            return

        try:
            from .skills import infer_prompt_rungs

            # Passed split for readability; infer_prompt_rungs pools them —
            # see the trade-off note there on why suppression is blanket.
            router_offered = set(acc.skills_offered_in_prompt)
            router_delivered = set(acc.skills_delivered) | set(acc.skills_loaded_by_agent)
            offered, delivered = infer_prompt_rungs(
                (call.rendered_input for call in acc.llm_calls),
                self._skills_registry,
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
            acc.skills_offered_in_prompt.update(offered)
            acc.skills_delivered.update(delivered)
        except Exception:
            logger.debug("Skill prompt-presence inference failed", exc_info=True)

    def _maybe_register_manifest(
        self, acc: _TraceAccumulator, agent_name: str
    ) -> Optional[str]:
        """Resolve the manifest id this trace should be ingested under.

        Ingest REQUIRES a manifest_id (``require_manifest_on_ingest``
        defaults on, and is on in production), so anything that returns
        None here costs the entire trace: the POST comes back 400 and the
        run is never recorded. This used to short-circuit on
        ``if not tools and not models: return``, which is precisely the
        state of every run that trips an input guardrail or dies before its
        first model call — the failures you most want to see were the ones
        systematically dropped.

        Three sources, in descending order of authority:

        1. The live ``Agent`` object, recorded by an agent config's
           instructions. It carries the declared model and full tool schemas
           even when the run produced no successful model call.
        2. Span data — model from the response/generation span, tool names
           from the agent span.
        3. Whatever this process already declared for this agent
           (`_declare` unions 1 and 2 across every trace, so the declared
           surface only ever grows).

        When all three are structurally empty we do NOT invent a manifest;
        see `_adopt_existing_manifest` for why an empty one is the
        expensive answer.
        """
        from . import _config

        if not _config._is_enabled():
            return None

        agent = acc.live_agent
        if agent is not None:
            try:
                data = _introspect_agent(agent)
                _declare(
                    agent_name,
                    tools=data.get("tools"),
                    models=data.get("models"),
                    subagents=data.get("subagents"),
                    prompts=data.get("prompts"),
                    from_agent=True,
                )
            except Exception:
                logger.debug("live-agent introspection failed", exc_info=True)

        declared = _declare(
            agent_name,
            tools=list(acc.seen_tools.values()) or None,
            models={"default": acc.seen_model} if acc.seen_model else None,
            subagents=[{"name": h} for h in acc.seen_handoffs] or None,
        )

        # Structural identity is tools + models (MANIFEST_VERSIONING §2.5);
        # a declaration with neither says nothing about the agent's contract.
        if not declared["tools"] and not declared["models"]:
            existing = _manifest_ids.get(agent_name)
            if existing:
                return existing
            adopted = _adopt_existing_manifest(agent_name)
            if adopted:
                _manifest_ids[agent_name] = adopted
                logger.debug(
                    "Run declared nothing structural; attributing trace to %s's "
                    "existing manifest %s", agent_name, adopted,
                )
                return adopted
            # Genuinely nothing on file either — this agent's first
            # observation. A component-less manifest is a supported shape
            # and it has no predecessor, so it mints no diff and no version
            # bump; it exists so the trace is kept rather than 400'd.
            logger.info(
                "Registering a component-less manifest for %s — this run "
                "exposed no model or tools (it did not reach a model call). "
                "Pass instrument(agent=your_agent) to declare the contract "
                "up front.",
                agent_name,
            )

        snapshot = extract_from_config(
            agent_name=agent_name,
            tools=declared["tools"],
            prompts=declared["prompts"],
            models=declared["models"],
            subagents=declared["subagents"],
            skills=self._skills_registry or None,
        )
        registered = _register_snapshot(agent_name, snapshot)
        if registered:
            return registered
        # Registration failed (network blip, auth). Fall back to the last
        # id we held for this agent so a hiccup doesn't drop the trace too.
        return _manifest_ids.get(agent_name) or snapshot.id


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


def _response_output_text(response_obj: Any) -> Optional[str]:
    """Extract the assistant's text from a Responses API ``Response``.

    Prefers the SDK's ``output_text`` convenience property, falling back to
    walking the ``output`` message items so synthetic/partial objects still
    work. Returns None when the response produced no text (e.g. a pure
    tool-call turn).
    """
    try:
        text = getattr(response_obj, "output_text", None)
        if isinstance(text, str) and text:
            return text
        texts: List[str] = []
        for item in getattr(response_obj, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str):
                    texts.append(part_text)
        return "".join(texts) or None
    except Exception:
        logger.debug("Response output text extraction failed", exc_info=True)
        return None


def _count(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _reported_cached_tokens(usage: Any) -> Optional[int]:
    """The cached-input count a Responses API ``ResponseUsage`` reports, 0 included.

    OpenAI's ``cached_tokens`` is a SUBSET of ``input_tokens`` (see
    ``LlmCallRecord.cache_read_tokens``). None when the response carried no
    ``input_tokens_details``: not reported is not a cache miss.
    """
    return _count(getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", None))


def _span_usage_cached_tokens(usage: Any) -> Optional[int]:
    """The cached-input count from a span's usage dict, when it can be told from a default.

    ``openai-agents`` writes ``input_tokens_details`` with zeros when the
    provider sent none (``agents.usage.Usage.__post_init__``), so a 0 here may
    be a measured cache miss or nothing measured. A positive count can only have
    come from the provider; a 0 reads as None, because recording it would claim
    a cold cache nobody measured.
    """
    details = usage.get("input_tokens_details") if isinstance(usage, dict) else None
    cached = _count(details.get("cached_tokens")) if isinstance(details, dict) else None
    return cached if cached else None


def _requested_tool_calls(items: Any) -> List[Dict[str, Any]]:
    """The function calls a model turn asked for, in the order it asked.

    Reads the shapes the Agents SDK puts on a span: Responses API output items
    (``type == "function_call"``, as objects or as their dumps, including a
    dumped ``Response`` whose ``output`` holds them) and chat-completions
    messages carrying ``tool_calls``. Each entry has the call id, the tool
    name, the arguments string exactly as the model wrote it, and the
    namespace when the call has one. Never raises: a shape it cannot read
    yields nothing, and the record stays as it would have been.
    """
    found: List[Dict[str, Any]] = []
    try:
        for item in items or ():
            is_dict = isinstance(item, dict)
            kind = item.get("type") if is_dict else getattr(item, "type", None)
            if kind == "function_call":
                get = item.get if is_dict else functools.partial(getattr, item)
                name = get("name", None)
                if isinstance(name, str) and name:
                    arguments = get("arguments", None)
                    namespace = get("namespace", None)
                    call_id = get("call_id", None)
                    found.append({
                        "call_id": call_id if isinstance(call_id, str) else None,
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str) else "",
                        "namespace": namespace if isinstance(namespace, str) and namespace else None,
                    })
                continue
            if not is_dict:
                continue
            for tc in item.get("tool_calls") or ():
                fn = tc.get("function") if isinstance(tc, dict) else None
                name = fn.get("name") if isinstance(fn, dict) else None
                if isinstance(name, str) and name:
                    arguments = fn.get("arguments")
                    call_id = tc.get("id")
                    found.append({
                        "call_id": call_id if isinstance(call_id, str) else None,
                        "name": name,
                        "arguments": arguments if isinstance(arguments, str) else "",
                        "namespace": None,
                    })
            nested = item.get("output")
            if isinstance(nested, (list, tuple)):
                found.extend(_requested_tool_calls(nested))
    except Exception:
        logger.debug("Tool-call extraction failed", exc_info=True)
        return []
    return found


def _handoff_tool_name(agent_name: str) -> str:
    """The tool name the Agents SDK gives a handoff to ``agent_name`` by default:
    ``transfer_to_<name>`` with whitespace and anything outside
    ``[A-Za-z0-9_]`` turned into ``_``, lower-cased."""
    name = re.sub(r"\s", "_", f"transfer_to_{agent_name}")
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()


def _tool_call_args(arguments: str) -> Dict[str, Any]:
    """The argument dict the model wrote. Arguments that are not a JSON object
    are kept verbatim under ``raw`` rather than dropped."""
    if not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return {"raw": arguments}
    return parsed if isinstance(parsed, dict) else {"raw": arguments}


def _attach_system_prompts(
    acc: "_TraceAccumulator", rail_prompts: Any = None
) -> None:
    """Prepend each call's system prompt to its ``rendered_input``.

    ``ResponseSpanData`` carries the response, input, usage and a response id
    and nothing else,
    so ``span_data.input`` is the input-items list ALONE. The instructions —
    the half of the prompt the skills menu is injected into — never appear on
    the span, which is why a trace could claim to have offered a skill whose
    name was nowhere in ``rendered_input``: the claim was true and the record
    was incomplete.

    Two sources, best evidence first:

    * ``Response.instructions`` — the server's echo of what it received,
      captured per call in ``_handle_response``. Authoritative, and immune to
      anything that rewrites instructions after our hook has run (a
      ``RunConfig.call_model_input_filter`` may legally replace them).
    * the run rail — the exact return value of ``Agent.get_system_prompt``.
      Used only for calls the server echoed nothing for.

    Nothing is ever reconstructed or inferred. When the fallback cannot be
    mapped onto calls without guessing, NOTHING is attached: an incomplete
    record beats an invented one.

    MUST be called after this run's rails are merged onto the accumulator and
    BEFORE ``_infer_skill_rungs`` — see the call site for why that order is
    what keeps a skill carried in the instructions visible to the inference
    without letting the router's own menu be re-read as evidence.
    """
    calls = acc.llm_calls
    if not calls:
        return
    resolved: Dict[Any, str] = dict(acc.system_prompt_by_call)

    unmapped = [c for c in calls if c.id not in resolved]
    prompts = [
        p for p in (rail_prompts or []) if isinstance(p, str) and p.strip()
    ]
    if unmapped and prompts:
        if len(prompts) == 1:
            # The run never changed its system prompt: one string, every call.
            for call in unmapped:
                resolved[call.id] = prompts[0]
        elif len(unmapped) == len(calls) and len(prompts) == len(calls):
            # Distinct prompt per turn, and no call has a server echo to
            # interleave with. `get_system_prompt` runs once per turn, before
            # that turn's model call, so rail order is turn order.
            for call, prompt in zip(calls, prompts):
                resolved[call.id] = prompt
        # Otherwise the mapping is a guess. Attach nothing to the unmapped
        # calls and let the record be visibly incomplete.

    for call in calls:
        text = resolved.get(call.id)
        if not text:
            continue
        messages = call.rendered_input
        if messages is None:
            call.rendered_input = [{"role": "system", "content": text}]
            continue
        if not isinstance(messages, list):
            continue
        # The chat-completions path (`_handle_generation`) already renders the
        # system message itself; adding it again would double the prompt.
        already = "\n".join(
            str(m.get("content") or "")
            for m in messages
            if isinstance(m, dict) and m.get("role") in ("system", "developer")
        )
        if text in already:
            continue
        messages.insert(0, {"role": "system", "content": text})


def _normalize_messages(raw: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize Responses-API input items to the rendered_input format.

    The default Agents path speaks the Responses API, whose input list is
    NOT chat-completions messages: a tool-using turn is mostly
    ``function_call`` / ``function_call_output`` / ``reasoning`` items that
    carry no ``role`` and no ``content``. Applying the chat shape to them
    — ``{"role": item.get("role", "user"), "content": str(item.get(
    "content", ""))}`` — collapsed 6 of 7 recorded messages to
    ``{"role": "user", "content": ""}``, so the trace showed an agent that
    apparently said nothing and called nothing, and the backend derived
    ``user_input_preview`` from the last of those empty rows.

    Each item now keeps its real role, its text flattened out of the parts
    list, and its item ``type`` where that is the only thing identifying it.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    if not isinstance(raw, (list, tuple)):
        return [{"role": "user", "content": str(raw)}]

    result: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            # A pydantic Responses item — dump it, else fall back to str().
            dumped = getattr(item, "model_dump", None)
            if callable(dumped):
                try:
                    item = dumped(exclude_none=True)
                except Exception:
                    item = None
            if not isinstance(item, dict):
                result.append({"role": "user", "content": str(item)})
                continue

        itype = _item_type(item)
        role = _item_role(item)

        if itype == "function_call":
            result.append({
                "role": role or "assistant",
                "type": "function_call",
                "name": item.get("name"),
                "content": str(item.get("arguments") or ""),
            })
        elif itype == "function_call_output":
            result.append({
                "role": role or "tool",
                "type": "function_call_output",
                "content": _stringify(item.get("output")),
            })
        elif itype == "reasoning":
            summary = item.get("summary") or []
            text = "".join(
                s.get("text", "") if isinstance(s, dict) else str(s) for s in summary
            ) if isinstance(summary, (list, tuple)) else _stringify(summary)
            result.append({"role": role or "assistant", "type": "reasoning", "content": text})
        else:
            entry: Dict[str, Any] = {
                "role": role or "user",
                "content": _item_text(item) or _stringify(item.get("content")),
            }
            if itype and itype != "message":
                entry["type"] = itype
            result.append(entry)
    return result


def _stringify(value: Any) -> str:
    """A message body as text — '' for None, verbatim for str."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


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
