"""Pydantic AI integration.

One-line install path::

    import decimalai
    decimalai.init()

    from decimalai.pydantic_ai import instrument
    instrument(enable_skill_loader=True)

    from pydantic_ai import Agent
    agent = Agent("openai:gpt-4o", system_prompt="You are helpful")
    # Skills are now auto-loaded into every agent.run() call.

Supports pydantic-ai-slim 2.56 and later only.

The install monkey-patches `pydantic_ai.Agent.__init__` so each newly
constructed Agent gets an `@agent.instructions` function that calls
`SkillRouter.build_prompt_fragment()` once per RUN and returns the routed menu.

Instructions, not a system prompt, because of what Pydantic AI does with each:

- A plain `@agent.system_prompt` is built only when there is no
  `message_history`. Turn 2 of a conversation reused turn 1's menu, was never
  routed, and its trace carried no routing_id.
- `@agent.system_prompt(dynamic=True)` is re-evaluated only where its part is
  still in the history. A history processor that trims the first request takes
  routing with it. The menu also gets stored in the history, and the OpenAI
  model skips its `openai_cache_instructions` breakpoint whenever a dynamic
  system prompt is present.
- Instructions are rebuilt for every request, and the model is sent only the
  current request's instructions, never older ones from the history. Pydantic
  AI puts them after the system-prompt messages and marks a function's output
  as dynamic. The caller's `system_prompt=` and static `instructions=` stay a
  byte-identical leading prefix, and provider cache breakpoints go in front of
  the routed menu. The menu still sits ahead of the conversation history, so a
  turn that routes to a DIFFERENT menu re-reads the history after it uncached.
  That is the price of routing every turn; a turn whose menu comes out the
  same keeps the whole prefix.

"Every request" includes each tool-loop step of a run, so the fragment is
remembered per run (`ctx.run_id`): one routing decision per run, the same menu
on every step of it. One consequence of being an instruction: inside
`agent.override(instructions=...)`, Pydantic AI replaces every instruction,
this one included, so runs in that block get no menu.

Tracing for Pydantic AI is observed via the underlying provider SDK
(OpenAI/Anthropic) — install the matching tracing adapter alongside::

    decimalai.init(openai=True)               # carries the spans
    decimalai.pydantic_ai.instrument()        # carries the run boundary

Both halves are needed. The provider instrumentor sees provider calls and
nothing else, so on its own it emits one unparented root span — and therefore
one DecimalAI trace — per call: a single `agent.run_sync()` that asks for a
tool and then answers arrives as two unrelated one-span traces, both filed
under whichever agent happened to run first in the process. `instrument()`
patches `Agent.iter` (the one place `run`/`run_sync`/`run_stream` all funnel
through) to open an `agent.run` span around each run, which gives those calls a
real parent to nest under and puts the Agent's own name on the trace.

The provider pairing is not the only route, and `decimalai init --framework
pydantic-ai` takes the other one::

    decimalai.init(otel=True)                 # the DecimalAI OTel exporter
    Agent.instrument_all()                    # Pydantic AI's OWN OTel spans
    decimalai.pydantic_ai.instrument(...)     # skills + the run boundary

Pydantic AI has native OpenTelemetry instrumentation, off by default, and it
covers every provider — so this route needs no per-provider instrumentor
package and works unchanged for `google:…` models. Pick ONE: with both, an
OpenInference span and a Pydantic AI span describe the same model call. Neither
is installed from in here, deliberately — turning `instrument_all()` on inside
this adapter would double up for every caller who already pairs a provider
instrumentor.
"""

from __future__ import annotations

import logging
import threading
import warnings
import weakref
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any, Optional, Tuple

from decimalai._provider_recovery import retry_model_requests as retry_model_requests

logger = logging.getLogger("decimalai.pydantic_ai")


# ── SkillRouter routing-id context ──────────────────────────
# Pydantic AI doesn't have its own DecimalAI trace processor (yet);
# downstream provider adapters (openai_agents / anthropic) read from
# their own contextvars. We surface our routing_id via a getter so
# the provider adapter can pull it in if both are installed.
_routing_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "decimalai_skill_router_routing_id_pydantic_ai", default=None,
)


def _set_routing_id(routing_id: Optional[str]) -> None:
    _routing_id_ctx.set(routing_id)


def get_current_routing_id() -> Optional[str]:
    """Read (without clearing) the current routing_id.

    Exposed so downstream framework adapters can read the routing_id
    set by the Pydantic AI skill loader and stamp it onto a trace.
    """
    return _routing_id_ctx.get()


def _scope() -> Optional[str]:
    """This run's routing scope — the live OTel trace id, or None.

    ``instrument()`` opens an ``agent.run`` span around every run, so inside a
    run this is stable and unique per run: both the prompt build and the
    ``load_skill`` tool call land on the same key. Without it, eight concurrent
    runs of one agent share one fragment-cache slot (they all route with
    ``query=None`` and the same agent name) and would be handed one routing
    decision between them.
    """
    try:
        from .otel import current_run_key
        key = current_run_key()
    except Exception:
        return None
    return None if key is None else f"{key:032x}"


# ── SkillRouter dynamic loader ──────────────────────────────
_skill_loader_installed = False
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
            inject_body=config.resolve_inject_body(has_tool_loop=_load_skill_tool_enabled()),
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter singleton init failed", exc_info=True)
        return None


# True once at least one Agent got the load_skill tool — gates the prompt
# hint in _skills_system_prompt (per-agent introspection of registered tools
# is private API in pydantic_ai; a module flag is the stable alternative).
_load_skill_tool_active = False

#: The name DecimalAI's tool falls back to when something else in the run
#: already exposes ``load_skill``, and the key it is always registered under in
#: the Agent's function toolset.
#:
#: Registered under this key, never under ``load_skill``, so that a caller's own
#: ``@agent.tool_plain def load_skill`` still registers. Under one key, two tools
#: raise ``UserError`` at the caller's decorator line. The model is shown the
#: tool as ``load_skill`` unless that name is taken; see _exposed_tool_name.
_NAMESPACED_TOOL_NAME = "decimalai_load_skill"
_tool_clash_warned = False


def _exposed_tool_name(ctx: Any) -> str:
    """The name the model sees DecimalAI's tool under in this run.

    ``load_skill`` (the name in the docs, the menu hint and every other adapter)
    unless another tool in the run already uses it. Pydantic AI does not let two
    toolsets share a tool name: every run raised ``UserError``. Namespacing
    keeps both catalogs callable, so the caller's skill loader keeps working
    and DecimalAI's skills still reach the model. Skipping DecimalAI's tool
    would have dropped DecimalAI's bodies without a word.

    What is checked: every function toolset in the run. That covers the Agent's
    own (``tools=`` and ``@agent.tool_plain``), its ``toolsets=``, and the run's
    ``toolsets=``. What is not: a toolset whose tools are known only once it is
    asked for them, such as an MCP server. That clash still raises Pydantic AI's
    own ``UserError``, which tells the caller to rename it or wrap it in
    ``PrefixedToolset``.

    Pure in ``ctx``, so the tool's ``prepare`` and the menu hint, which call it
    separately, give the same answer for the same run. If it cannot tell, it
    answers the namespaced name, which can never clash.
    """
    from .skill_router import LOAD_SKILL_TOOL_NAME

    try:
        roots = []
        tool_manager = getattr(ctx, "tool_manager", None)
        if tool_manager is not None:
            # The run's whole toolset tree, run-time `toolsets=` included.
            roots.append(tool_manager.toolset)
        roots.extend(getattr(getattr(ctx, "agent", None), "toolsets", None) or ())
        if not roots:
            # Nothing to clash with: a context that carries no toolsets at all.
            return LOAD_SKILL_TOOL_NAME

        from pydantic_ai.toolsets import FunctionToolset

        taken = False

        def visit(toolset: Any) -> None:
            nonlocal taken
            if isinstance(toolset, FunctionToolset) and LOAD_SKILL_TOOL_NAME in toolset.tools:
                taken = True

        for root in roots:
            root.apply(visit)
    except Exception:  # noqa: BLE001 - a tool name must never break the run
        logger.debug("could not inspect the run's toolsets", exc_info=True)
        return _NAMESPACED_TOOL_NAME
    if not taken:
        return LOAD_SKILL_TOOL_NAME
    global _tool_clash_warned
    if not _tool_clash_warned:
        _tool_clash_warned = True
        logger.warning(
            "another toolset on this Pydantic AI agent already has a tool named "
            "%r, so DecimalAI's skill loader is exposed to the model as %r "
            "instead. Both stay callable; the DecimalAI skills menu names %r.",
            LOAD_SKILL_TOOL_NAME, _NAMESPACED_TOOL_NAME, _NAMESPACED_TOOL_NAME,
        )
    return _NAMESPACED_TOOL_NAME


def _prepare_load_skill(ctx: Any, tool_def: Any) -> Any:
    """Per-step ``prepare`` for DecimalAI's tool: pick its exposed name."""
    import dataclasses

    name = _exposed_tool_name(ctx)
    return tool_def if tool_def.name == name else dataclasses.replace(tool_def, name=name)


def _load_skill_hint(ctx: Any) -> str:
    """The menu's how-to line, naming the tool the model can actually call."""
    from .skill_router import LOAD_SKILL_PROMPT_HINT, LOAD_SKILL_TOOL_NAME

    name = _exposed_tool_name(ctx)
    if name == LOAD_SKILL_TOOL_NAME:
        return LOAD_SKILL_PROMPT_HINT
    return LOAD_SKILL_PROMPT_HINT.replace(LOAD_SKILL_TOOL_NAME, name)


def _load_skill_tool_enabled() -> bool:
    """Config gate for the load_skill tool (kill switch:
    DECIMALAI_LOAD_SKILL_TOOL=0 / init(load_skill_tool=False))."""
    try:
        from ._config import _get_config
        return bool(getattr(_get_config(), "load_skill_tool", True))
    except Exception:
        return True


def _handle_load_skill(name: str, agent_name: Optional[str] = None) -> str:
    """Tool callback — always returns a string the model can act on.

    ``agent_name`` is the name this run was ROUTED under (``_run_agent_name``).
    Without it the platform resolves the body for no agent: it misses the
    agent's own Use rows, so it serves the registry's latest version instead of
    the agent's pin, or nothing at all for a Use target the registry hides.
    """
    router = _get_skill_router()
    if router is None:
        return "load_skill error: skill router unavailable."
    try:
        scope = _scope()
        body = router.load_skill(name, scope=scope, agent_name=agent_name)
        if scope is not None:
            # Only names the router actually served a body for reach this rail
            # — a budget refusal or a not-found never records, so
            # `skills_loaded_by_agent` means "a body reached the model", not
            # "the model asked for one".
            served = router.consume_loaded_names(scope=scope)
            if served:
                try:
                    from .otel import record_skill_rail
                    # Drained in the same breath as the names and from the same
                    # scope, so the hash describes the body THIS run just read.
                    # `{}` on a router that predates the hash rail, which is the
                    # pre-existing behaviour: the name, with a null hash.
                    try:
                        digests = router.consume_loaded_hashes(scope=scope) or {}
                    except (AttributeError, TypeError):
                        digests = {}
                    try:
                        versions = router.consume_delivered_versions(scope=scope)
                    except (AttributeError, TypeError):
                        versions = []
                    record_skill_rail(loaded=served, loaded_hashes=digests, delivered_versions=versions)
                except Exception:
                    logger.debug(
                        "skill rail recording failed (non-fatal)", exc_info=True
                    )
        return body
    except Exception:
        logger.debug("load_skill handler failed (non-fatal)", exc_info=True)
        return f"load_skill error: could not load {name!r} (transient error)."


def _run_query(ctx: Any) -> Optional[str]:
    """The turn's user text from a RunContext, or None.

    Best-effort and never raises: this runs inside a system-prompt hook, and an
    adapter that crashes a user's agent to improve its own routing has made
    things worse. `prompt` is the current turn; `messages` is the fallback for
    shapes that put the text there instead. A None result is a real answer —
    the caller degrades to full-menu mode.
    """
    try:
        prompt = getattr(ctx, "prompt", None)
        text = _content_text(prompt)
        if text:
            return text
        messages = list(getattr(ctx, "messages", None) or [])
        if not messages:
            return None
        from pydantic_ai.messages import UserPromptPart

        for msg in reversed(messages):
            for part in (getattr(msg, "parts", None) or []):
                if isinstance(part, UserPromptPart):
                    text = _content_text(part.content)
                    if text:
                        return text
    except Exception:  # noqa: BLE001 — routing must never break the run
        logger.debug("could not read the turn's query from RunContext", exc_info=True)
    return None


def _content_text(content: Any) -> Optional[str]:
    """Text of a prompt that may be a string or a list of content parts."""
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
        joined = "\n".join(x for x in parts if x and x.strip()).strip()
        return joined or None
    return None


# ── one routing decision per run ────────────────────────────
#: The fragment each live run was given, by ``RunContext.run_id``.
#:
#: Pydantic AI resolves instructions for every model request, and a run that
#: calls a tool makes several. Routing each of them would ask the router again
#: on every tool-loop step, hand the model a different menu halfway through the
#: run, and file several routing decisions against one trace. So the first
#: request of a run routes and the rest of the run reuses that answer.
#:
#: An entry is reused only for the same run_id AND the same live RunUsage and
#: Agent objects. Pydantic AI requires a run_id to be distinct only within one
#: history. A caller who passes the same explicit ``run_id=`` to two separate
#: conversations is not served the first conversation's menu, because each run
#: gets a fresh RunUsage. Bounded, oldest evicted first: a run that outlives
#: 256 newer runs routes again, which costs one router call and nothing else.
_RUN_FRAGMENTS_MAX = 256
_run_fragments: "OrderedDict[str, Tuple[Any, Any, str]]" = OrderedDict()
_run_fragments_lock = threading.Lock()


def _run_memo_key(ctx: Any) -> Optional[str]:
    run_id = getattr(ctx, "run_id", None)
    return run_id if isinstance(run_id, str) and run_id else None


def _remembered_fragment(ctx: Any) -> Optional[str]:
    key = _run_memo_key(ctx)
    if key is None:
        return None
    with _run_fragments_lock:
        entry = _run_fragments.get(key)
    if entry is None:
        return None
    usage_ref, agent_ref, fragment = entry
    if usage_ref() is getattr(ctx, "usage", None) and agent_ref() is getattr(ctx, "agent", None):
        return fragment
    return None


def _remember_fragment(ctx: Any, fragment: str) -> None:
    import weakref

    key = _run_memo_key(ctx)
    if key is None:
        return
    try:
        entry = (
            weakref.ref(getattr(ctx, "usage", None)),
            weakref.ref(getattr(ctx, "agent", None)),
            fragment,
        )
    except TypeError:  # not weak-referenceable (None, or a stand-in): no memo
        return
    with _run_fragments_lock:
        _run_fragments[key] = entry
        _run_fragments.move_to_end(key)
        while len(_run_fragments) > _RUN_FRAGMENTS_MAX:
            _run_fragments.popitem(last=False)


_Routed = Tuple[str, Optional[str], list, list, list]


def _route_turn(ctx: Any) -> Optional[_Routed]:
    """The blocking half of the skills prompt: the router call and its rails.

    Runs in a worker thread (see _skills_system_prompt) and must not touch the
    event loop. Everything that reads what the router wrote into contextvars
    runs here, in the same thread and the same copied context as the call that
    wrote it. The worker's context is a copy, so nothing written here reaches
    the awaiting coroutine. Returns None when there is no router or it failed.
    """
    router = _get_skill_router()
    if router is None:
        return None
    try:
        agent_obj = getattr(ctx, "agent", None)
        # The name the TRACE uses for this run (see _run_agent_name), not
        # `agent.name`: for an unnamed Agent that is the variable name Pydantic
        # AI inferred, so routing asked for agent "agent" while the trace was
        # filed under instrument(agent_name=...), and skills assigned to the
        # real agent were silently never offered.
        agent_name = _run_agent_name(agent_obj) if agent_obj is not None else None
        from .skill_router import (
            _release_scoped_routing_rail,
            consume_last_delivered_names,
            consume_last_delivered_versions,
            consume_last_offered_names,
        )
        scope = _scope()
        consume_last_delivered_versions()
        try:
            fragment, routing_id = router.build_prompt_fragment(
                query=_run_query(ctx), agent_name=agent_name, scope=scope,
            )
        except Exception:
            consume_last_delivered_versions()
            raise
        finally:
            # The router keeps its own copy of this decision under the run's
            # scope, for adapters that read it back from there. This one reads
            # the per-call rails below, so the copy is released here or never.
            # Only the decision: the body budget filed under the same scope is
            # still needed by this run's load_skill calls.
            _release_scoped_routing_rail(router, scope)
        # Drain the per-call contextvar rails NOW, one statement after the
        # router wrote them and on the same thread, so these are this call's
        # names. The router's instance rails are not used here on purpose: they
        # are process-global and clear-on-read, so under concurrent runs the
        # first drainer takes every lane's names.
        return (
            fragment or "",
            routing_id,
            consume_last_offered_names(),
            consume_last_delivered_names(),
            consume_last_delivered_versions(),
        )
    except Exception:
        logger.debug("Pydantic AI skill loader failed (non-fatal)", exc_info=True)
        return None


async def _skills_system_prompt(ctx: Any) -> str:
    """The instructions function registered on every Agent: the routed menu.

    Routes on the turn's user message (``RunContext.prompt``, else the last
    user part in ``messages``), once per run; later requests in the same run
    get the same fragment back (see _run_fragments). Falls back to full-menu
    mode when no text can be read, which is the honest answer for a turn that
    genuinely has none. With no query, ``SkillRouter`` also skips body
    injection, so reading the query is what keeps the injected channel alive
    when ``DECIMALAI_LOAD_SKILL_TOOL=0``.

    The router call is blocking HTTP, so it runs in a worker thread. Called on
    the loop, it stalled every other coroutine there for up to the router's
    5 s hot-path budget, including the other runs of a concurrent agent.

    Never raises: any failure is an empty string, i.e. an un-routed turn.
    """
    remembered = _remembered_fragment(ctx)
    if remembered is not None:
        return remembered
    try:
        import anyio.to_thread

        from .skill_router import consume_last_delivered_versions

        # A version witness some earlier, failed build left in THIS context
        # must not outlive the build that replaces it. The worker starts from a
        # copy of this context and clears its copy too, but only this call can
        # clear the caller's.
        consume_last_delivered_versions()
        routed = await anyio.to_thread.run_sync(_route_turn, ctx)
    except Exception:
        logger.debug("Pydantic AI skill loader failed (non-fatal)", exc_info=True)
        routed = None
    fragment = ""
    if routed is not None:
        fragment, routing_id, offered, delivered, delivered_versions = routed
        if routing_id:
            # Here, not in the worker: the worker's context is a copy, and a
            # routing id set there would never reach this coroutine's caller.
            _set_routing_id(routing_id)
        # Tell the model how bodies arrive when the tool exists.
        # The server fragment keeps the activation-statement instruction
        # unchanged (Stage-M parity).
        if fragment and _load_skill_tool_active:
            fragment = f"{fragment}\n{_load_skill_hint(ctx)}"
        if fragment:
            # Only for a prompt we actually returned. An empty fragment is an
            # un-routed turn and must claim neither a routing_id nor a name.
            try:
                from .otel import record_skill_rail
                record_skill_rail(
                    routing_id=routing_id,
                    offered=offered,
                    delivered=delivered,
                    delivered_versions=delivered_versions,
                    prompt_text=fragment,
                )
            except Exception:
                logger.debug("skill rail recording failed (non-fatal)", exc_info=True)
    # Remembered even when empty: a router that failed on this run's first
    # request is not asked again on every tool step of the same run.
    _remember_fragment(ctx, fragment)
    return fragment


def _install_skill_loader() -> None:
    """Monkey-patch `pydantic_ai.Agent.__init__` to auto-register skills.

    Also registers the load_skill tool on every new Agent (the progressive-disclosure path):
    Pydantic AI owns its tool loop, so the tool result routes back mid-turn
    for free — this adapter ships the tool live."""
    global _skill_loader_installed
    if _skill_loader_installed:
        return
    try:
        from pydantic_ai import Agent
    except ImportError:
        logger.warning(
            "enable_skill_loader=True but pydantic-ai not installed; "
            "skipping skill loader install. "
            "Install it with: pip install \"decimalai[pydantic-ai]\""
        )
        return

    original_init = Agent.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            # An instructions function, not a system prompt: see the module
            # docstring for why. Do NOT move the fragment into the caller's
            # `system_prompt=` / `instructions=` either. Pydantic AI sends
            # those first and our dynamic part after them, which is what keeps
            # the caller's text a byte-identical, cacheable prefix.
            self.instructions(_skills_system_prompt)
        except Exception:
            logger.debug(
                "Failed to register the skills instructions on Agent (non-fatal)",
                exc_info=True,
            )
        # Register load_skill so surfaced descriptions are
        # executable — an agent that can see a skill can pull its body.
        if _load_skill_tool_enabled():
            try:
                from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION

                # Weak: the tool lives on the agent, and a strong reference
                # would make every Agent a reference cycle.
                agent_ref = weakref.ref(self)

                def load_skill(name: str) -> str:
                    # Resolved per call, the same way routing resolves it, so
                    # the body is loaded for the agent the menu was built for.
                    agent = agent_ref()
                    return _handle_load_skill(
                        name,
                        agent_name=(
                            _run_agent_name(agent) if agent is not None
                            else _default_agent_name
                        ),
                    )

                # Keyed `decimalai_load_skill`, shown to the model as
                # `load_skill` unless that name is taken (see _exposed_tool_name).
                self.tool_plain(
                    load_skill,
                    name=_NAMESPACED_TOOL_NAME,
                    description=LOAD_SKILL_TOOL_DESCRIPTION,
                    prepare=_prepare_load_skill,
                )
                global _load_skill_tool_active
                _load_skill_tool_active = True
            except Exception:
                logger.debug(
                    "load_skill tool registration failed (non-fatal)", exc_info=True
                )

    Agent.__init__ = patched_init  # type: ignore[method-assign]
    _skill_loader_installed = True
    logger.info("DecimalAI SkillRouter loader installed (Pydantic AI)")


# ── agent name ──────────────────────────────────────────────
#: The name `instrument(agent_name=...)` was called with, used for runs of an
#: Agent that carries no explicit one. None means "never set", which is the
#: pre-2026-08-29 behaviour exactly.
_default_agent_name: Optional[str] = None

#: Where the patched ``Agent.__init__`` stashes the name the CALLER passed.
#:
#: Read rather than ``Agent.name`` because those stop being the same thing the
#: moment a run starts: Pydantic AI's ``_infer_name`` fills a missing ``name``
#: from the local VARIABLE the Agent was assigned to (``agent = Agent(...)`` runs
#: as agent name ``"agent"``, measured on 2.56). By the time ``iter`` opens,
#: ``self.name`` is that invented name and nothing distinguishes it from one the
#: user chose — so the distinction has to be captured at construction.
_CONSTRUCTED_NAME_ATTR = "_decimalai_constructed_name"

_name_capture_installed = False


def _install_name_capture() -> None:
    """Patch ``Agent.__init__`` to remember whether the caller named the Agent."""
    global _name_capture_installed
    if _name_capture_installed:
        return
    try:
        from pydantic_ai import Agent
    except ImportError:
        return

    original_init = Agent.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        try:
            # `name` is keyword-only on Agent.__init__, so kwargs is the whole
            # story — there is no positional spelling to miss.
            setattr(self, _CONSTRUCTED_NAME_ATTR, kwargs.get("name"))
        except Exception:  # noqa: BLE001 — naming must never break construction
            logger.debug("could not record the constructed agent name",
                         exc_info=True)

    Agent.__init__ = patched_init  # type: ignore[method-assign]
    _name_capture_installed = True


def _run_agent_name(agent: Any) -> Optional[str]:
    """Whose run this is, most specific answer first.

    1. the name the caller passed to ``Agent(name=...)`` — per-agent, and the
       only one that stays right in a process running two agents;
    2. ``instrument(agent_name=...)`` — the process-wide binding, which is what
       `decimalai init` writes so that dropping ``name=`` from the generated
       Agent does not silently move the traces;
    3. ``agent.name`` as Pydantic AI left it, which after ``_infer_name`` is the
       variable name. Last, because it is a guess about a local variable, but
       still ahead of nothing: it is what this adapter used before there was
       anything else to prefer.
    """
    constructed = getattr(agent, _CONSTRUCTED_NAME_ATTR, None)
    return constructed or _default_agent_name or getattr(agent, "name", None)


# ── run scope ───────────────────────────────────────────────
_run_scope_installed = False


def _install_run_scope() -> None:
    """Monkey-patch ``Agent.iter`` so one agent run is one span, one trace.

    Pydantic AI emits no spans of its own — the docs say so, and it is the
    reason this integration's documented setup is *two* calls, the second being
    a provider pairing like ``decimalai.init(openai=True)``. But a provider
    instrumentor only sees provider calls. Each one becomes an unparented root
    span in its own OTel trace, so a single ``agent.run_sync()`` that asks for a
    tool and then answers arrived as two unrelated one-span DecimalAI traces —
    no waterfall, no way to tell which calls belonged to the same run, and
    (because the exporter's agent name is fixed when it is built) every one of
    them filed under whichever agent ran first.

    ``iter`` is the patch point because it is the one place every entry point
    funnels through: ``run_sync`` awaits ``run``, ``run``/``run_stream``/
    ``run_stream_events`` all open ``self.iter(...)``, and an ``AgentRun`` is
    exactly one run. Patching ``run_sync`` instead would have covered less and
    cost more — it calls ``self._infer_name(inspect.currentframe())``, so a
    wrapper frame between it and the caller breaks the name inference this fix
    depends on.

    The span this opens is the parent that genuinely wraps the run. It does not
    synthesize the steps Pydantic AI never reported: the tool call it executes
    in-process emits no span here, and none is invented for it.
    """
    global _run_scope_installed
    if _run_scope_installed:
        return
    try:
        from pydantic_ai import Agent
    except ImportError:
        logger.warning(
            "pydantic-ai is not installed; skipping the DecimalAI run scope. "
            "Install it with: pip install \"decimalai[pydantic-ai]\""
        )
        return

    original_iter = Agent.iter

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def patched_iter(self, *args, **kwargs):
        from .providers import agent_run

        # The Agent's own name, per run — resolved rather than read, because
        # by now `self.name` may be one Pydantic AI invented from a local
        # variable. See _run_agent_name.
        with agent_run(_run_agent_name(self)):
            async with original_iter(self, *args, **kwargs) as run:
                yield run

    Agent.iter = patched_iter  # type: ignore[method-assign]
    _run_scope_installed = True
    logger.info("DecimalAI run scope installed (Pydantic AI)")


def instrument(
    *,
    agent_name: Optional[str] = None,
    enable_skill_loader: bool = False,
    trace_runs: bool = True,
) -> None:
    """Install DecimalAI integration for Pydantic AI.

    Pydantic AI does no tracing of its own, so the *content* of a trace still
    comes from the provider pairing you install alongside
    (``decimalai.init(openai=True)`` / ``init(anthropic=True)``). What this adds
    is the run boundary that pairing cannot see: an ``agent.run`` span around
    each agent run, so the run's provider calls land in ONE trace, nested, under
    the name of the agent that made them.

    Note: this adapter never reads from or writes to disk, so there is
    no ``disk_sync`` parameter. If you're running inside a disk-loading
    runtime (Claude Code, Cursor), ``_warn_if_disk_runtime_detected``
    will log a one-shot warning on enable to flag the duplicate-injection
    risk.

    Args:
        agent_name: The DecimalAI agent runs of an UNNAMED Agent are filed
            under. An ``Agent(name=...)`` always wins — this is the fallback,
            and it exists because Pydantic AI's own fallback is worse:
            ``_infer_name`` fills a missing name from the local variable, so
            ``agent = Agent(...)`` traces as the agent ``"agent"`` and the
            dashboard page for the agent you configured stays empty. Leave it
            None to keep exactly that pre-existing behaviour.
        enable_skill_loader: When True, monkey-patch Agent so new
            instances get the routed skills menu as an instructions function
            (routed once per run, on every turn of a conversation) and the
            ``load_skill`` tool. The tool is exposed as
            ``decimalai_load_skill`` instead when another toolset in the run
            already has a ``load_skill``.
        trace_runs: When True (the default), wrap every agent run in a parent
            span. Turn it off only if something else already opens a span around
            the run — a second one would be a redundant layer in the waterfall,
            not a wrong one.
    """
    global _default_agent_name
    if enable_skill_loader:
        from .skill_router import _warn_if_disk_runtime_detected
        _warn_if_disk_runtime_detected("pydantic_ai")
        _install_skill_loader()
    if agent_name is not None:
        _default_agent_name = agent_name
    if trace_runs or enable_skill_loader:
        # Installed whenever the run scope or the skill loader is, not only when
        # a name was given: the capture is what tells an explicit
        # ``Agent(name=…)`` apart from an inferred one, and both name a run with
        # it. Routing without it, under ``trace_runs=False``, would let the
        # process-wide ``agent_name=`` fallback take a name the caller chose. A
        # LATER instrument(agent_name=…) call must not find that Agents built
        # in between were never recorded, either.
        _install_name_capture()
    if trace_runs:
        _install_run_scope()
    logger.info(
        "DecimalAI Pydantic AI integration installed "
        "(agent_name=%s, skill_loader=%s, trace_runs=%s)",
        agent_name,
        enable_skill_loader,
        trace_runs,
    )


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
        "decimalai.pydantic_ai.install() is deprecated; use "
        "decimalai.pydantic_ai.instrument() instead. It turns on tracing for pydantic_ai "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
