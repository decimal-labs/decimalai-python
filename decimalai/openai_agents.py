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

import logging
import threading
import warnings
from collections import OrderedDict
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


def _clean_names(names: Any) -> List[str]:
    """Keep only non-blank strings — same filter as `_add_skills_offered`."""
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


def _current_run_key() -> Optional[str]:
    """The active Agents-SDK trace id, or None outside a run."""
    try:
        from agents.tracing import get_current_trace
        trace = get_current_trace()
    except Exception:
        return None
    trace_id = getattr(trace, "trace_id", None)
    return trace_id if isinstance(trace_id, str) else None


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
                "agent": None,
                "user_input": None,
                # Every distinct string `Agent.get_system_prompt` resolved on
                # this run, in turn order. The FALLBACK evidence for what the
                # model was shown; the server's own echo is preferred when the
                # Responses API returns one. See `_attach_system_prompts`.
                "system_prompts": [],
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
    agent: Any = None,
    user_input: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> bool:
    """Stamp routing/loading telemetry onto the CURRENT run's rail.

    Returns False when there is no active run to attribute to (the caller
    then leaves the value on the legacy contextvar/router rails).
    """
    run_key = _current_run_key()
    if run_key is None:
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
    return True


def _pop_run_rail(run_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read + remove one run's rail. None when the run recorded nothing."""
    if not run_key:
        return None
    with _run_rails_lock:
        return _run_rails.pop(run_key, None)


def _drain_router_rails() -> tuple[Optional[str], List[str], List[str], List[str]]:
    """Drain the Router singleton's process-global instance rails.

    Kept as the fallback for runs that produced no per-run rail (a caller
    driving the Router outside `Runner.run`, or an Agents SDK old enough to
    lack `get_current_trace`) and, on every send, as a RESET: an undrained
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


# ── SkillRouter dynamic loader ──────────────────────────────
# When `install(enable_skill_loader=True)` runs, we monkey-patch
# `agents.Agent.__init__` so every Agent created afterwards has its
# string `instructions` wrapped into a callable that appends skill
# content fetched from the platform per-run — after the agent's own
# instructions, so those stay a stable cacheable prefix. User-supplied
# callables pass through untouched (their judgment wins).

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
            inject_body=config.resolve_inject_body(has_tool_loop=_has_tool_loop()),
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


def _extract_query(ctx: Any) -> Optional[str]:
    """Pull the turn's user input out of an ``agents.RunContextWrapper``.

    The runner stamps the turn's input items on ``ctx.turn_input``
    immediately before awaiting ``get_system_prompt`` (which is what calls
    us), so that attribute is the supported seam. It was previously probed
    for ``ctx.input`` / ``ctx.user_input`` / ``ctx.query`` — none of which
    the wrapper has ever defined (its public fields are ``context``,
    ``usage``, ``turn_input``, ``tool_input``) — so every call fell through
    to ``query=None`` and semantic routing never engaged: the platform
    logged ``strategy: full_menu`` and dumped the entire catalogue into the
    prompt instead of the one skill the turn called for.

    The legacy attribute probes are kept AFTER ``turn_input`` for callers
    who hand us their own duck-typed context object.
    """
    query = _query_from_input_items(getattr(ctx, "turn_input", None))
    if query:
        return query
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


#: Set True when `_make_load_skill_tool()` could not produce a tool. Read by
#: `_has_tool_loop()` so a failed registration falls back to prompt injection
#: rather than leaving the model with a menu it cannot read.
_load_skill_tool_registration_failed = False


def _has_tool_loop() -> bool:
    """Whether this adapter will really deliver bodies via a load_skill tool.

    NOT the same question as `_load_skill_tool_enabled()`, which reads a config
    flag. A flag saying "register the tool" plus a registration that failed adds
    up to zero body channels — the same conjunction that made langchain ship
    broken. Answer with the outcome where one is known.
    """
    return _load_skill_tool_enabled() and not _load_skill_tool_registration_failed


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
    run_key = _current_run_key()
    # Only pass `scope` when there IS a run to scope to, so a router that
    # predates the parameter is untouched on the unscoped path.
    kwargs = {"scope": run_key} if run_key else {}
    try:
        result = router.load_skill(name, **kwargs)
    except TypeError:
        # Router from an older SDK — no per-run scope parameter.
        result = router.load_skill(name)
    except Exception:
        logger.debug("load_skill handler failed (non-fatal)", exc_info=True)
        return f"load_skill error: could not load {name!r} (transient error)."
    # Attribute the load to THIS run rather than to whichever trace next
    # drains the router's shared `_loaded_names`. A body actually served
    # is the only thing that counts as loaded; a budget refusal or a
    # not-found message is not.
    if isinstance(result, str) and result.startswith(f"## Skill: {name}"):
        _record_run_rail(loaded=[name])
    return result


def _make_load_skill_tool() -> Any:
    """Build the native load_skill FunctionTool (the progressive-disclosure path).

    The OpenAI Agents SDK owns its tool loop, so a load_skill tool result is
    routed back to the model mid-turn. This adapter therefore registers
    load_skill as a real tool; the langchain / anthropic adapters wrap a layer
    with no tool loop, so they surface skills in the prompt instead."""
    global _load_skill_tool_registration_failed
    try:
        from agents import function_tool
    except Exception:
        # Loud, not DEBUG: if this returns None the model gets no load_skill tool,
        # and `resolve_inject_body(has_tool_loop=...)` would still be told this
        # adapter HAS a tool loop — leaving zero body channels, silently. Exactly
        # the conjunction that shipped broken on langchain.
        _load_skill_tool_registration_failed = True
        logger.warning(
            "load_skill tool unavailable: could not import `function_tool` from "
            "`agents`. Skill bodies will be prompt-injected instead."
        )
        return None
    from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION

    def load_skill(name: str) -> str:
        return _handle_load_skill(name)

    load_skill.__doc__ = LOAD_SKILL_TOOL_DESCRIPTION
    try:
        tool = function_tool(load_skill)
    except Exception:
        _load_skill_tool_registration_failed = True
        logger.warning(
            "load_skill tool could not be built (function_tool raised) — skill "
            "bodies will be prompt-injected instead. Set logging to DEBUG for the "
            "traceback.",
        )
        logger.debug("function_tool(load_skill) failed", exc_info=True)
        return None
    _load_skill_tool_registration_failed = False
    return tool


def _agent_has_load_skill_tool(agent: Any) -> bool:
    return any(
        getattr(t, "name", None) == "load_skill"
        for t in (getattr(agent, "tools", None) or [])
    )


def _load_skill_reachable(agent: Any) -> bool:
    """Whether load_skill will be on the tool list this turn.

    An agent built BEFORE `instrument(enable_skill_loader=True)` doesn't
    carry the tool on `agent.tools` — `get_all_tools` appends it at
    resolution time instead — so checking the declared list alone would
    suppress the prompt hint for exactly the agents the retrofit exists
    to serve.
    """
    if _agent_has_load_skill_tool(agent):
        return True
    return bool(_skill_loader_installed and _load_skill_tool_enabled())


# Marks our own wrapped-instructions callable so the class-level
# `get_system_prompt` retrofit can tell "already skill-aware" from "a
# callable the user wrote", and so `_introspect_agent` can still see the
# static base prompt underneath it.
_BASE_INSTRUCTIONS_ATTR = "__decimalai_base_instructions__"


def _make_skill_aware_instructions(base: str):
    """Return a sync callable usable as `Agent.instructions`.

    The callable emits `base` FIRST and the routed skill fragment after it.
    `base` is the agent's own instructions — identical bytes on every turn —
    while the fragment is rebuilt from this turn's query and so differs turn to
    turn. OpenAI auto-caches a request's stable leading prefix, so the varying
    half has to be the tail: in front, it pushed the agent's whole (often much
    larger) prompt off its cached prefix and made every call a full miss.
    """

    def instructions_fn(ctx: Any, agent: Any) -> str:
        try:
            router = _get_skill_router()
            if router is None:
                return base
            run_key = _current_run_key()
            # Only pass `scope` when there IS a run to scope to, so a router
            # that predates the parameter is untouched on the unscoped path.
            kwargs = {"scope": run_key} if run_key else {}
            try:
                fragment, routing_id = router.build_prompt_fragment(
                    query=_extract_query(ctx),
                    agent_name=getattr(agent, "name", None),
                    **kwargs,
                )
            except TypeError:
                # Router from an older SDK — no per-run scope parameter.
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
            # The contextvar writes above are made inside the Task the
            # runner gathers this callable in, so they die with it. Mirror
            # everything onto THIS run's rail, which `_send_trace` can read.
            _record_run_rail(
                routing_id=routing_id,
                offered=offered,
                delivered=delivered,
                agent=agent,
            )
            if not fragment:
                return base
            # Tell the model how bodies arrive — only when load_skill will
            # actually be on this turn's tool list. We append to the server
            # fragment and leave its activation-statement instruction unchanged.
            if _load_skill_reachable(agent):
                from .skill_router import LOAD_SKILL_PROMPT_HINT
                fragment = f"{fragment}\n{LOAD_SKILL_PROMPT_HINT}"
            # base first, fragment second — see the docstring above.
            return f"{base}\n\n{fragment}".strip() if base else fragment
        except Exception:
            # Never break a run because of a Router hiccup.
            logger.debug("Skill loader callable failed (non-fatal)", exc_info=True)
            return base

    setattr(instructions_fn, _BASE_INSTRUCTIONS_ATTR, base)
    return instructions_fn


# ── Class-level retrofit (works on Agents built before instrument()) ──
# `__init__` patching can only ever reach objects constructed AFTER the
# patch, so an Agent defined at import time — the overwhelmingly common
# shape, since agents are module-level constants — silently got no skills
# and no load_skill tool. `Agent.get_system_prompt` and
# `Agent.get_all_tools` are resolved off the CLASS on every turn, so
# patching them serves every agent regardless of when it was built.
_agent_hooks_installed = False
_HOOK_MARKER = "__decimalai_hooked__"
# Emitted once, not per turn, when we retrofit an agent the constructor
# patch missed.
_retrofit_notice_emitted = False


def _install_agent_hooks() -> bool:
    """Wrap `Agent.get_system_prompt` / `Agent.get_all_tools` once.

    Always installed by `instrument()`, whether or not the skill loader is
    on: the prompt hook is also how the tracer gets its hands on the live
    Agent object, which is the only place the agent's declared model and
    tool schemas exist for runs that never complete a model call.

    Returns True when the hooks are in place.
    """
    global _agent_hooks_installed
    if _agent_hooks_installed:
        return True
    try:
        from agents import Agent
    except ImportError:
        return False

    original_gsp = getattr(Agent, "get_system_prompt", None)
    original_gat = getattr(Agent, "get_all_tools", None)
    if original_gsp is None or getattr(original_gsp, _HOOK_MARKER, False):
        return original_gsp is not None

    async def patched_get_system_prompt(self, run_context):  # type: ignore[no-untyped-def]
        base = await original_gsp(self, run_context)
        try:
            # Hand the live agent to the trace processor. This is the ONLY
            # path on which a run that trips a guardrail or fails its first
            # model call can still declare a manifest — the spans such a run
            # emits carry no model and an empty tool list. The turn's input
            # rides along for the same reason: such a run emits no model span
            # either, so this is the only place its prompt is visible.
            _record_run_rail(
                agent=self,
                user_input=_query_from_input_items(
                    getattr(run_context, "turn_input", None)
                ),
            )
        except Exception:
            logger.debug("agent observation failed (non-fatal)", exc_info=True)
        # ONE exit, so the string recorded below is the string returned. The
        # runner hands this exact value to the model as `system_instructions`
        # (agents/run_internal/run_loop.py -> ModelSettings ->
        # openai_responses.py `"instructions": system_instructions`), and it is
        # the half of the prompt no span carries: `ResponseSpanData.__slots__`
        # is ("response", "input", "usage") — there is no instructions slot, so
        # without this capture the skills menu is invisible to the tracer.
        resolved = base
        if _skill_loader_installed:
            try:
                current = getattr(self, "instructions", None)
                if callable(current) and hasattr(current, _BASE_INSTRUCTIONS_ATTR):
                    # Built after instrument(): the constructor already wrapped
                    # `instructions`, and `original_gsp` just called it. Adding
                    # the fragment again here would inject the skills menu twice.
                    pass
                elif callable(current):
                    pass  # a callable the user wrote — their judgment wins
                else:
                    _note_retrofit(self)
                    resolved = _make_skill_aware_instructions(base or "")(
                        run_context, self
                    )
            except Exception:
                logger.debug(
                    "skill-aware prompt retrofit failed (non-fatal)", exc_info=True
                )
                resolved = base
        try:
            _record_run_rail(system_prompt=resolved)
        except Exception:
            logger.debug("system-prompt observation failed (non-fatal)", exc_info=True)
        return resolved

    setattr(patched_get_system_prompt, _HOOK_MARKER, True)
    Agent.get_system_prompt = patched_get_system_prompt  # type: ignore[method-assign]

    if original_gat is not None and not getattr(original_gat, _HOOK_MARKER, False):
        async def patched_get_all_tools(self, run_context):  # type: ignore[no-untyped-def]
            tools = await original_gat(self, run_context)
            if not _skill_loader_installed or not _load_skill_tool_enabled():
                return tools
            try:
                if any(getattr(t, "name", None) == "load_skill" for t in tools):
                    return tools  # already declared (constructor path, or the user's own)
                tool = _make_load_skill_tool()
                if tool is None:
                    return tools
                _note_retrofit(self)
                # Return a new list — never mutate the caller's resolved
                # tools or the agent's declared `tools` attribute.
                return [*tools, tool]
            except Exception:
                logger.debug(
                    "load_skill tool retrofit failed (non-fatal)", exc_info=True
                )
                return tools

        setattr(patched_get_all_tools, _HOOK_MARKER, True)
        Agent.get_all_tools = patched_get_all_tools  # type: ignore[method-assign]

    _agent_hooks_installed = True
    return True


def _note_retrofit(agent: Any) -> None:
    """Log once that we served an Agent the constructor patch never saw."""
    global _retrofit_notice_emitted
    if _retrofit_notice_emitted:
        return
    _retrofit_notice_emitted = True
    logger.info(
        "DecimalAI retrofitted skills onto agent %r, which was constructed "
        "before instrument(enable_skill_loader=True). Skills and the "
        "load_skill tool are attached per-run, so nothing is lost; "
        "call instrument() before building your agents to have them "
        "attached at construction instead.",
        getattr(agent, "name", "<unnamed>"),
    )


def _install_skill_loader() -> None:
    """Turn on per-run skill loading for `agents.Agent`.

    Two layers, both idempotent:

    1. `Agent.__init__` — agents built from here on get their string
       `instructions` wrapped into the skill-aware callable and the
       load_skill tool appended to their declared tool list. A
       user-supplied instructions callable is left alone (their
       judgment > ours).
    2. `Agent.get_system_prompt` / `Agent.get_all_tools` (see
       `_install_agent_hooks`) — the same two affordances, attached at
       RUN time, for agents that already existed when this was called.
       Layer 1 marks what it wraps so layer 2 never double-injects.
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

    if _install_agent_hooks():
        logger.info(
            "DecimalAI SkillRouter loader installed (OpenAI Agents; agents "
            "constructed before this call are retrofitted per-run)"
        )
    else:
        # No class hooks to patch → the constructor patch is all we have,
        # and an already-built agent will silently get no skills. Say so.
        warnings.warn(
            "decimalai: this openai-agents build has no "
            "Agent.get_system_prompt/get_all_tools to hook, so only Agents "
            "constructed AFTER instrument(enable_skill_loader=True) receive "
            "skills and the load_skill tool. Move instrument() above your "
            "Agent(...) definitions, or upgrade openai-agents.",
            RuntimeWarning,
            stacklevel=2,
        )
        logger.warning(
            "DecimalAI SkillRouter loader installed WITHOUT run-time retrofit "
            "(OpenAI Agents) — agents built before this call get no skills"
        )

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
            if name == "load_skill":
                continue
            tool_entry: Dict[str, Any] = {"name": name}
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
        if tools_list:
            result["tools"] = tools_list

    # Extract instructions as prompt. When the skill loader wrapped a
    # string `instructions` into its per-run callable, read the static base
    # back off the wrapper: without this, turning the loader on silently
    # DELETED the prompt component from the manifest, and prompt-drift
    # detection went blind for exactly the installs using skills.
    instructions = getattr(agent, "instructions", None)
    if callable(instructions):
        instructions = getattr(instructions, _BASE_INSTRUCTIONS_ATTR, None)
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
    # Captured before the derivation below overwrites it: `disk_sync=True`
    # passed by hand is a request, the derived default is only a guess.
    _disk_sync_explicit = disk_sync is True
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
    # `instructions` wrapped into a per-run callable that appends
    # skill content from the platform after them. See _install_skill_loader().
    if enable_skill_loader:
        from .skill_router import _warn_if_disk_runtime_detected
        _warn_if_disk_runtime_detected("openai_agents")
        _install_skill_loader()
    else:
        # Always hook the Agent class, loader or not: the get_system_prompt
        # wrapper is how the trace processor gets the live Agent object, and
        # without it a run that never completes a model call has nothing to
        # declare a manifest from — so ingest 400s and the trace is lost.
        _install_agent_hooks()

    logger.info(
        "DecimalAI OpenAI Agents tracing installed (agent_name=%s, exclusive=%s, agent_introspected=%s, skill_loader=%s, disk_sync=%s)",
        agent_name,
        exclusive,
        agent is not None,
        enable_skill_loader,
        disk_sync,
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
        except Exception:
            # Leave the per-agent hash unset so the next trace retries; a
            # transient blip must not permanently stop this agent from ever
            # declaring a manifest.
            logger.warning("Failed to register manifest for %s", agent_name, exc_info=True)
            return None
        _manifest_ids[agent_name] = manifest_id
        _manifest_hashes[agent_name] = snapshot.manifest_hash
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
        # Router's offered set; ``skills_loaded_by_agent`` is auto-populated
        # from the Router's loaded-names rail when the native load_skill
        # tool serves a body (drained at trace-send), plus any manual
        # annotations (use `decimalai.log_skill_loaded`).
        self.skills_offered_in_prompt: set[str] = set()
        self.skills_loaded_by_agent: set[str] = set()
        # Bodies that reached the model (Router body injection) —
        # between offered and activated; never implies activation.
        self.skills_delivered: set[str] = set()
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

    @property
    def live_agent(self) -> Any:
        """The `agents.Agent` object this run is executing, if we saw it.

        Recorded by the `get_system_prompt` hook, which the runner calls on
        every turn — including turns that go on to fail. It is the only
        source of the agent's declared model and tool schemas for runs whose
        spans carry neither.
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

        # Accumulate tools for manifest (names only from span data).
        # `load_skill` is skipped for the same reason `_introspect_agent`
        # skips it: it is an SDK affordance this adapter attaches at run
        # time, so counting it would make the declared tool set — and
        # therefore the manifest version — depend on whether the skill
        # loader happened to be on.
        span_tools = getattr(span_data, "tools", None) or []
        for tool_name in span_tools:
            name_str = str(tool_name)
            if name_str == "load_skill":
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
        finish_reason = FinishReason.STOP
        output_dict = {"content": output_text} if output_text else None
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
                agent_name=acc.agent_name or self.default_agent_name,
                provider=_infer_provider(call_model),
                model_name=call_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                rendered_input=_normalize_messages(raw_input),
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
        # overwrite a preview captured from an earlier turn.
        if output_text:
            acc.final_output_preview = _preview(output_text)

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

        # Auto-register manifest from span data + the live Agent object
        manifest_id = self._maybe_register_manifest(acc, agent_name)

        config = _config._config

        # This run's own rail — routing decision, delivered bodies, loads —
        # keyed by the trace id both the instructions callable and this
        # method see. Popped, so it can't leak into a later trace.
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

        # SkillRouter: consume the routing_id set by the dynamic
        # instructions callable. We read at trace-end (not start)
        # because the instructions callable fires AFTER on_trace_start.
        # An explicit annotation that DID reach this context wins; the rail
        # is what covers the normal path, where the runner gathers that
        # callable in its own Task and the contextvar write never arrives.
        if acc.routing_id is None:
            acc.routing_id = _consume_routing_id() or rail_routing_id

        # Drain the per-trace offered-names contextvar populated
        # by the skill loader callable. Merge with any direct
        # `log_skill_offered` calls already accumulated on the accumulator.
        drained_offered = _consume_skills_offered()
        if drained_offered:
            acc.skills_offered_in_prompt.update(drained_offered)
        if rail_offered:
            acc.skills_offered_in_prompt.update(rail_offered)

        # Drain the delivered-names contextvar (Router body injection).
        drained_delivered = _consume_skills_delivered()
        if drained_delivered:
            acc.skills_delivered.update(drained_delivered)
            acc.skills_offered_in_prompt.update(drained_delivered)  # delivered implies offered
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
        # `ResponseSpanData.__slots__` is ("response", "input", "usage"), so on
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

        trace = RunTrace(
            id=uuid4(),
            project=config.project if config else None,
            agent_name=agent_name,
            status=acc.status,
            source_type="production",
            started_at=acc.started_at or ended_at,
            ended_at=ended_at,
            user_input_preview=acc.user_input_preview or _preview(run_rail.get("user_input")),
            final_output_preview=acc.final_output_preview,
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

        1. The live ``Agent`` object, captured by the `get_system_prompt`
           hook. It carries the declared model and full tool schemas even
           when the run produced no successful model call.
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


def _attach_system_prompts(
    acc: "_TraceAccumulator", rail_prompts: Any = None
) -> None:
    """Prepend each call's system prompt to its ``rendered_input``.

    ``ResponseSpanData`` carries ``(response, input, usage)`` and nothing else,
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
