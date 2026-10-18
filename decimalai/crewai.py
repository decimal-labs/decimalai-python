"""CrewAI integration: tracing, and automatic skill delivery.

Setup::

    import decimalai
    decimalai.init()

    from decimalai.crewai import instrument
    instrument(agent_name="support", enable_skill_loader=True)

    config = decimalai.load_agent("support")
    agent = Agent(role="Support", goal="Resolve the ticket",
                  backstory=config.system_prompt or "", llm=llm)
    crew = Crew(agents=[agent], tasks=[Task(description=question,
                                            expected_output="the answer", agent=agent)])
    crew.kickoff()

``instrument()`` turns on the same tracing ``decimalai.init(crewai=True)`` does —
DecimalAI's OpenTelemetry exporter plus the OpenInference CrewAI instrumentor,
because CrewAI emits no spans to your tracer provider on its own — and does it
once per process. ``init(crewai=True)`` followed by
``instrument(enable_skill_loader=True)`` therefore adds the loader without
building a second exporter.

Skills (opt-in)
---------------
With the loader on, every model call an agent makes inside a CrewAI executor is
routed through ``SkillRouter``, and the routed fragment — the menu plus the
skill BODIES, since nothing on this rail can fetch a body on demand — goes into
that call's messages, right after CrewAI's own system prompt. Same semantics and
telemetry as the LangChain and ADK loaders: the run's trace carries the
``routing_id``, the names offered in the prompt, and the names whose body was
delivered. Delivered is never promoted to activated: there is no ``load_skill``
tool here, so the model has no way to ASK for a skill, and an activation
recorded anyway would be a fabrication (conformance item C13).

SUPPORTED: crewai 1.15.27 with openinference-instrumentation-crewai 1.1.21 —
the latest releases, and the only ones this adapter is written and run against.
Older CrewAI releases are not supported, and nothing here works around them.

THE SEAM, read out of the installed crewai 1.15.27 rather than remembered:

* ``crewai.hooks.register_before_llm_call_hook`` is public, documented API.
  Its registry is the generic hook dispatcher's ``PRE_MODEL_CALL`` list, so
  this hook and an ``@on(InterceptionPoint.PRE_MODEL_CALL)`` hook share one
  ordered queue. Every agent executor copies the global hook list when it is
  built (``experimental.AgentExecutor._setup_executor``, the default; the
  deprecated ``CrewAgentExecutor.__init__``), and
  ``utilities/agent_utils.py::_prepare_llm_call`` runs that copy immediately
  before ``llm.call``. It also marks the call as dispatched, so the LLM layer
  does not run the hooks a second time. It is CrewAI's own extension point for
  exactly this, the way ADK's ``before_model_callback`` is ADK's.
* The hook context's ``messages`` IS the executor's live message list
  (``LLMCallHookContext.__init__``: ``self.messages = executor.messages``), and
  ``_prepare_llm_call`` yields that same list to ``llm.call`` after the hooks
  ran. What the hook inserts is what the provider is sent — and what the
  provider instrumentor records, which is how it reaches the trace's
  ``rendered_input``.
* That list PERSISTS across the iterations of one task (tool call, observation,
  answer), so the block inserted for the previous call is still in it on the
  next one. It is replaced, never stacked: the inserted messages are
  :class:`_SkillMessage` instances, found and removed by type before the next
  insert.
* A call dispatched with NO executor is left alone. The LLM layer also
  dispatches these hooks for calls that are not an agent turn — CrewAI's own
  output converter, guardrails and planner, or ``llm.call()`` straight from
  user code (``BaseLLM._invoke_before_llm_call_hooks``, context built with
  ``executor=None``). A skill body has no business in a JSON conversion prompt,
  and claiming delivery for it would credit the skill with a call the agent
  never made.

What the hook cannot reach, stated rather than discovered later: an executor
built BEFORE ``instrument(enable_skill_loader=True)`` keeps the hook list it
copied. CrewAI builds an agent's executor on its first task and reuses it, so
enable the loader before the first ``kickoff()``. ``Crew.kickoff()``,
``Crew.kickoff_async()``, ``Crew.akickoff()``, ``Agent.kickoff()`` and
``Agent.kickoff_async()`` all dispatch through an executor and all get skills.

Threads and event loops
-----------------------
CrewAI hooks are synchronous by CrewAI's design (``crewai/hooks/dispatch.py``:
they "may be invoked from async seams, so they must not block on heavy I/O").
This hook cannot await, so the routing call is a blocking HTTP request on
whatever thread CrewAI runs the hook on. Measured on 1.15.27
(``tests/test_crewai_phase1.py``):

* The default executor, ``AgentExecutor``, is a Flow, and a Flow runs every
  sync step — the model call and its hooks included — through
  ``asyncio.to_thread(ctx.run, ...)``. ``Crew.kickoff_async()`` is itself
  ``asyncio.to_thread(self.kickoff)``. So on every kickoff entry point, sync or
  async, the hook runs on a worker thread with the caller's context copied in,
  and the caller's event loop keeps running while skills are routed.
* The exception is the deprecated ``CrewAgentExecutor``
  (``Agent(executor_class=CrewAgentExecutor)``) under ``Crew.akickoff()``. Its
  native-async loop awaits ``aget_llm_response``, which runs the hooks inline on
  the event-loop thread, so the routing call blocks that loop until it
  returns. The adapter says so once, as a WARNING, and keeps delivering:
  dropping skills there would trade a visible stall for silent non-delivery.

Native mode: CrewAI's own skills
--------------------------------
CrewAI ships Agent Skills of its own: ``Agent(skills=...)`` and
``Crew(skills=...)``, merged onto each agent at kickoff. They render into
CrewAI's system prompt — an ``<available_skills>`` catalog plus a native
``load_skill`` tool for metadata-level skills, an always-on ``<skills>`` block
for activated ones. Pointed at the skills ``decimalai skills add`` installed::

    Agent(..., skills=decimalai.crewai.skills(), tools=decimalai.crewai.skill_tools())

``skill_tools()`` adds what CrewAI does not have: ``load_skill_resource`` reads
a file of an installed skill and ``run_skill_script`` runs one of its scripts
(see :func:`skill_tools`). With tracing on, each load is recorded on the run's
trace as an activation carrying the registry identity of the body the MODEL
received (``decimalai._native``), and the catalog's names as offered:

* a ``load_skill`` call, from a ``post_tool_call`` hook that reads the text the
  tool returned — the body CrewAI read for this call, and the one the model
  gets. CrewAI parses a skill's frontmatter when the agent is built but reads
  its body from disk on each load, so after ``decimalai skills update`` the
  same agent delivers the new body and the activation says so;
* an always-on (activated) skill, from CrewAI's ``SkillUsedEvent``, whose body
  is the agent's parsed copy;
* the catalog, from the ``<available_skills>`` block of each model call.

The hook and the event handler run where the run's OpenTelemetry context is
current (the tool hook on the run's own thread; CrewAI's event bus runs a
handler in a copy of the emitting context), so each lands on that run's trace.
``Crew.kickoff()`` drains the event bus before the run ends.

The router's injection stays off for an agent with CrewAI skills of its own:
its catalog already offers them, and a second one in the same call would give
the model two sets of instructions for one job. That is said once per agent at
INFO. If a native load still lands in a run the router delivered into (another
agent of the same crew was routed), a WARNING says once that two catalogs
reached the model.

The run's trace, handed back (``on_trace``) and filed under its caller
----------------------------------------------------------------------
The CrewAI trace is assembled from spans by the OpenTelemetry exporter, on the
exporter's thread, under an id minted there — so nothing the caller holds names
it. Two things close that, with the contract the LangChain and ADK adapters
already keep:

* ``instrument(on_trace=callable)`` calls the observer once per CrewAI run —
  a failed run included — with a detached copy of that run's finished trace (a
  ``RunTrace``), before export, on the thread that exports it. An observer that
  raises is logged and costs neither the run nor its trace.
* A run started inside ``decimalai.start_trace()`` is filed as that trace's
  child (``parent_trace_id``). The router stamps a delivery onto the enclosing
  trace too, and the platform credits the use once only for a linked child.

Both are captured when the run STARTS — by a span processor
(:class:`_RunLinker`) whose ``on_start`` runs on the caller's thread, inside the
caller's context, at the run's first CrewAI span — and handed to the exporter
through ``decimalai.otel.record_run_link``. By export time that context is gone,
and an ``instrument()`` call made in between must not move a run that is
already in flight.

``decimalai.flush()`` pushes this pipeline out too, bounded by a timeout, so a
short-lived process can report on a trace that exists. Without it the spans sit
in the batch processor for up to five seconds, or until the process exits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import subprocess
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union

if TYPE_CHECKING:
    from .schema.trace import RunTrace

logger = logging.getLogger("decimalai.crewai")

# ── module state ────────────────────────────────────────────

_install_lock = threading.Lock()

#: True once CrewAI tracing has been set up in this process — by
#: ``instrument()`` here, or by ``decimalai.init(crewai=True)`` (which sets it
#: through ``decimalai._activate_crewai_instrumentation``). Guards against a
#: second exporter, which would sit on a provider no instrumentor feeds.
_tracing_installed = False

_skill_loader_installed = False

#: Set by ``instrument(agent_name=...)`` and by ``init(crewai=True,
#: agent_name=...)``. The routing agent name when the run itself names none —
#: see ``_routing_agent_name``.
_install_agent_name: Optional[str] = None

_skill_router_singleton: Any = None

#: One-shot warnings, once per process like
#: ``skill_router._warn_if_disk_runtime_detected``: a crew makes a model call
#: per step, and the same sentence on every one of them would bury it.
_native_skills_warned = False
_loop_block_warned = False

#: Agents (by role) the router's injection was turned off for, named once each.
_router_off_logged: set = set()

#: Runs (OTel trace ids) the router delivered into, newest last, so a native
#: load in one of them can say that two catalogs reached the model. Bounded:
#: only a run still in flight can matter.
_routed_runs: "OrderedDict[int, None]" = OrderedDict()
_ROUTED_RUNS_MAX = 1024
_native_lock = threading.Lock()

#: True once the native-load observer is registered with CrewAI (see
#: ``_install_native_observer``); the event-bus subscription cannot be read back.
_native_bus_subscribed = False

#: At most this many native skill names in the two-catalogs warning.
_MAX_NAMED_SKILLS = 10


@dataclass(frozen=True)
class _InstrumentationConfig:
    """What ``instrument()`` last said about runs that have yet to start."""

    on_trace: Optional[Callable[["RunTrace"], None]] = None


#: Published whole by ``instrument()`` and read whole by :class:`_RunLinker` at
#: a run's start. Its own lock, not ``_install_lock``: span starts read it, and
#: they must never wait behind a SkillRouter being built.
_config_lock = threading.Lock()
_instrument_config = _InstrumentationConfig()

#: The tracer providers this SDK wired CrewAI's spans to — what
#: ``decimalai.flush()`` pushes out. Filled by ``_wire_tracer_provider``, which
#: ``decimalai._activate_crewai_instrumentation`` calls on both setup paths.
_tracer_providers: List[Any] = []

#: How long ``decimalai.flush()`` waits for this pipeline. The SDK's own
#: ``BatchSpanProcessor.force_flush`` ignores its timeout (opentelemetry-sdk
#: 1.42.1 and 1.45.1 both export everything, inline, under a lock), so the
#: bound is enforced here. Five seconds is the sender's own per-trace wait, and
#: leaves room inside the ten seconds a container runtime allows between
#: SIGTERM and SIGKILL.
_FLUSH_TIMEOUT_S = 5.0

#: The instrumentation scope the OpenInference CrewAI instrumentor's spans carry
#: (its tracer is ``trace_api.get_tracer(__name__, …)``). A span from this scope
#: is a CrewAI run's; the provider-SDK spans beneath it are not, and neither is
#: anything else sharing the pipeline.
_CREWAI_SCOPE = "openinference.instrumentation.crewai"

#: Roles CrewAI's leading instructions arrive under. "developer" is OpenAI's
#: current name for the system role; a caller using it has a system prefix like
#: anyone else, and the skill block belongs after it.
_INSTRUCTION_ROLES = ("system", "developer")


class _SkillMessage(dict):
    """A message this adapter put into CrewAI's message list.

    A ``dict`` subclass, so to every consumer it IS a message — CrewAI copies
    messages into plain dicts before formatting them for a provider
    (``BaseLLM._format_messages``), and a provider receives JSON either way, so
    the type never leaves the process. The type is what lets the next model
    call of the same task find the block it is replacing, where matching on
    text would miss the moment the routed fragment changed.

    A marker KEY was the obvious alternative and is wrong here: CrewAI strips
    only its own ``cache_breakpoint`` key before a request goes out
    (``BaseLLM._format_messages``), so any other key reaches the provider, and
    OpenAI rejects a message carrying a property it does not know.

    ``query`` is the routing query this block was built for, carried to the next
    iteration so one task routes on one question even after CrewAI has appended
    reflection prompts of its own.
    """

    query: Optional[str] = None


# ── SkillRouter ─────────────────────────────────────────────


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
    with _install_lock:
        if _skill_router_singleton is not None:
            return _skill_router_singleton
        try:
            from ._config import _get_config
            from .skill_router import SkillRouter

            config = _get_config()
            _skill_router_singleton = SkillRouter(
                api_key=config.api_key,
                base_url=config.base_url,
                # No `load_skill` tool on this rail, so injection is the only
                # body channel — the `has_tool_loop=False` case
                # `resolve_inject_body` answers True for. An explicit
                # init(inject_skill_body=...) / DECIMALAI_INJECT_SKILL_BODY
                # still wins.
                inject_body=config.resolve_inject_body(has_tool_loop=False),
            )
        except Exception:
            logger.debug("SkillRouter singleton init failed", exc_info=True)
            return None
    return _skill_router_singleton


def _scope() -> Optional[str]:
    """This run's routing scope — the live OTel trace id, or None.

    The same key the trace is assembled under (the OpenInference crew span is
    the trace root), so two concurrent kickoffs of one agent cannot share a
    fragment-cache slot, and therefore a routing decision. None outside a traced
    run, which keeps the router's unscoped behaviour.
    """
    try:
        from .otel import current_run_key

        key = current_run_key()
    except Exception:
        return None
    return None if key is None else f"{key:032x}"


def _routing_agent_name() -> Optional[str]:
    """The DecimalAI agent this call's skills are routed for.

    The name the TRACE will be filed under, so routing and attribution agree:
    the run-scoped name the OTel rail stamps onto every span this run starts
    (``decimalai.otel._active_agent_name``, set by ``instrument(agent_name=...)``
    and by ``agent_run(...)``) first, then this adapter's own ``agent_name``.
    Agent-scope skills — every registry skill pulled onto one agent — only
    resolve when the name is sent.
    """
    try:
        from .otel import _active_agent_name

        name = _active_agent_name.get()
    except Exception:
        name = None
    return name or _install_agent_name


# ── the routing query ───────────────────────────────────────


def _content_text(content: Any) -> str:
    """The text of a message's content — a string or a list of blocks."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(p for p in parts if p.strip()).strip()
    return ""


def _query_for(context: Any, messages: Sequence[Any]) -> Optional[str]:
    """What this agent turn should be routed on.

    The TASK DESCRIPTION when there is a task — a crew run. CrewAI wraps it in a
    user message of its own ("Current Task: … This is the expected criteria for
    your final answer … Begin! This is VERY important to you …"), and routing on
    that wrapper would rank skills against CrewAI's boilerplate as much as
    against the question. ``Task.description`` is the question, already
    interpolated with the kickoff inputs.

    With no task (``Agent.kickoff()``), the last user message that carries text.
    """
    task = getattr(context, "task", None)
    description = getattr(task, "description", None) if task is not None else None
    if isinstance(description, str) and description.strip():
        return description.strip()
    for msg in reversed(list(messages)):
        if isinstance(msg, _SkillMessage) or not isinstance(msg, dict):
            continue
        if str(msg.get("role") or "").lower() != "user":
            continue
        text = _content_text(msg.get("content"))
        if text:
            return text
    return None


# ── injection ───────────────────────────────────────────────


def _remove_previous_block(messages: List[Any]) -> Optional[_SkillMessage]:
    """Take out the block an earlier call of this task inserted; return one of it."""
    previous: Optional[_SkillMessage] = None
    for index in range(len(messages) - 1, -1, -1):
        msg = messages[index]
        if isinstance(msg, _SkillMessage):
            previous = msg
            del messages[index]
    return previous


def _insertion_point(messages: Sequence[Any]) -> tuple:
    """``(index, role)`` for the skill block.

    Directly after the caller's leading instruction messages — CrewAI's own
    system prompt (role, goal, backstory, tool instructions). That is the
    STABLE part of the request, so it stays first and stays cacheable; the
    routed block follows it. Every native CrewAI provider that has a separate
    system field (Anthropic, Gemini, Bedrock) joins consecutive system messages
    in order, so the block lands after the agent's instructions there too.

    No leading system message means the agent was built with
    ``use_system_prompt=False`` — CrewAI's switch for models that refuse a
    system role. Inserting one would break exactly those calls, so the block
    goes first as a user message instead. The model is shown it either way.
    """
    cut = 0
    for msg in messages:
        if isinstance(msg, dict) and str(msg.get("role") or "").lower() in _INSTRUCTION_ROLES:
            cut += 1
            continue
        break
    return (cut, "system") if cut else (0, "user")


# ── what the routing call shares the model call with ────────


def _event_loop_running_here() -> bool:
    """True when this thread is running an asyncio event loop right now."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _warn_if_blocking_the_loop() -> None:
    """Say once that the routing call is about to block a running event loop.

    CrewAI hooks are synchronous, so there is nothing to await: when CrewAI runs
    one on the loop's own thread, the routing request stalls every coroutine on
    that loop until it returns. On crewai 1.15.27 only the deprecated
    ``CrewAgentExecutor``'s native-async path does that (see the module
    docstring). The call still goes ahead — see ``_inject_skills``.
    """
    global _loop_block_warned
    if _loop_block_warned or not _event_loop_running_here():
        return
    _loop_block_warned = True
    logger.warning(
        "DecimalAI CrewAI: CrewAI ran its before_llm_call hook on a thread with a "
        "running asyncio event loop. CrewAI hooks are synchronous, so DecimalAI's "
        "skill routing request blocks that event loop until it returns, stalling "
        "every other coroutine on it. On crewai 1.15.27 only the deprecated "
        "CrewAgentExecutor (Agent(executor_class=CrewAgentExecutor)) does this, "
        "under Crew.akickoff(); the default AgentExecutor runs the hook on a "
        "worker thread. Drop executor_class, or use Crew.kickoff_async(). Skills "
        "are still delivered. Logged once per process."
    )


def _native_skills(context: Any) -> tuple:
    """``(names, cataloged, always_on)`` for CrewAI's own skills on this agent.

    Read exactly as CrewAI's prompt builder reads them
    (``utilities/prompts.py::_build_skill_block``): the ``Skill`` objects in
    ``agent.skills``, which by now include the crew's (``Crew(skills=...)`` is
    merged onto each agent at kickoff). ``cataloged`` is CrewAI's own answer to
    "is a ``load_skill`` tool registered": a metadata-level skill makes
    ``build_skill_catalog`` non-empty, and only then does the agent get the
    tool (``Agent._add_skill_loader_tool``). ``always_on`` is an activated
    skill, rendered in full in a ``<skills>`` block on every call.
    """
    agent = getattr(context, "agent", None)
    configured = getattr(agent, "skills", None) if agent is not None else None
    if not configured or not isinstance(configured, (list, tuple)):
        return [], False, False
    from crewai.skills.loader import build_skill_catalog
    from crewai.skills.models import INSTRUCTIONS, Skill

    loaded = [skill for skill in configured if isinstance(skill, Skill)]
    return (
        [skill.name for skill in loaded],
        bool(build_skill_catalog(loaded)),
        any(skill.disclosure_level >= INSTRUCTIONS for skill in loaded),
    )


def _say_router_off(context: Any, names: Sequence[str], cataloged: bool,
                    always_on: bool) -> None:
    """Say once per agent that the router stays out of its calls, and why.

    INFO, not WARNING: nothing is wrong. The agent's own CrewAI skills are its
    catalog (D7 of the Phase 3 design), and their loads are recorded as
    activations. Never raises.
    """
    try:
        role = getattr(getattr(context, "agent", None), "role", None)
        with _native_lock:
            if role in _router_off_logged:
                return
            _router_off_logged.add(role)
        shown = ", ".join(names[:_MAX_NAMED_SKILLS])
        if len(names) > _MAX_NAMED_SKILLS:
            shown += f" (+{len(names) - _MAX_NAMED_SKILLS} more)"
        native = " and ".join(
            part for part, present in (
                ("CrewAI's <available_skills> catalog with its own load_skill tool",
                 cataloged),
                ("CrewAI's always-on <skills> block", always_on),
            ) if present
        )
        logger.info(
            "DecimalAI CrewAI: agent %r has CrewAI skills of its own (%s: %s), so "
            "DecimalAI's skill router does not inject its menu or bodies into that "
            "agent's model calls; CrewAI's loads are recorded as activations. "
            "Agents without skills= are routed as before. Logged once per agent.",
            role, native or "CrewAI skills", shown,
        )
    except Exception:  # noqa: BLE001
        logger.debug("router-off notice failed (non-fatal)", exc_info=True)


def _note_routed_run() -> None:
    """Remember that the router delivered into the run that is live here."""
    from .otel import current_run_key

    key = current_run_key()
    if key is None:
        return
    with _native_lock:
        _routed_runs[key] = None
        _routed_runs.move_to_end(key)
        while len(_routed_runs) > _ROUTED_RUNS_MAX:
            _routed_runs.popitem(last=False)


def _warn_if_two_catalogs(run_key: Optional[int], skill_name: Optional[str]) -> None:
    """Say once that a native load landed in a run the router also delivered into.

    The router stays out of an agent with CrewAI skills of its own, so this is
    the case it could not see: another agent of the same run was routed (or the
    loader tool was given to an agent through ``tools=``). The model was then
    shown two skill catalogs in one run. Never raises.
    """
    global _native_skills_warned
    try:
        with _native_lock:
            if _native_skills_warned or run_key is None or run_key not in _routed_runs:
                return
            _native_skills_warned = True
        logger.warning(
            "DecimalAI CrewAI: CrewAI's own skill loader loaded %r in a run whose model "
            "calls also carried DecimalAI's routed skills, so two skill catalogs "
            "reached the model in one run. The router stays out only of an agent with "
            "CrewAI skills (skills=) of its own; give the skills to every agent that "
            "loads them, or leave enable_skill_loader off. Logged once per process.",
            skill_name,
        )
    except Exception:  # noqa: BLE001
        logger.debug("two-catalogs check failed (non-fatal)", exc_info=True)


def _inject_skills(context: Any) -> None:
    """Route this agent turn's skills and insert them into its messages.

    Nothing is claimed that was not put in front of the model: the routing id
    and the offered / delivered names are recorded only after the inserted
    messages have been read back out of the very list CrewAI is about to send.
    """
    if getattr(context, "executor", None) is None:
        return  # not an agent turn — see the module docstring
    messages = getattr(context, "messages", None)
    if not isinstance(messages, list):
        return

    # Replace, never stack: the previous call of this task left its block here.
    previous = _remove_previous_block(messages)

    # An agent with CrewAI skills of its own is served by its own catalog. Read
    # before the routing call, so that call is not made for nothing.
    try:
        names, cataloged, always_on = _native_skills(context)
    except Exception:  # noqa: BLE001 - undetectable is not native: route as before
        logger.debug("native CrewAI skill detection failed (non-fatal)", exc_info=True)
        names, cataloged, always_on = [], False, False
    if names:
        _say_router_off(context, names, cataloged, always_on)
        return

    router = _get_skill_router()
    if router is None:
        return

    query = previous.query if previous is not None and previous.query else _query_for(
        context, messages
    )
    # The routing request below is blocking I/O and there is nothing to await
    # it with. On a running loop it stalls the loop; that is said out loud, and
    # the call still goes ahead, because no skills at all would be the silent
    # failure this adapter exists to prevent.
    _warn_if_blocking_the_loop()
    scope = _scope()
    agent_name = _routing_agent_name()
    from .skill_router import (
        _release_scoped_routing_rail,
        consume_last_delivered_names,
        consume_last_delivered_versions,
        consume_last_offered_names,
    )

    consume_last_delivered_versions()
    try:
        parts_fn = getattr(router, "build_prompt_parts", None)
        if callable(parts_fn):
            # The prefix/tail split: `prefix` is byte-identical turn to turn and
            # carries the bodies; `tail` is the one sentence that depends on
            # this query.
            prefix, tail, routing_id = parts_fn(
                query=query, agent_name=agent_name, scope=scope,
            )
        else:
            # A router object that predates the split still has to deliver.
            # Without this branch the AttributeError is swallowed below and the
            # adapter traces perfectly while injecting nothing.
            prefix, routing_id = router.build_prompt_fragment(
                query=query, agent_name=agent_name, scope=scope,
            )
            tail = ""
    except Exception:
        consume_last_delivered_versions()
        logger.debug("build_prompt_parts failed (non-fatal)", exc_info=True)
        return
    finally:
        # This adapter reads the per-call rails below and records them on the
        # OTel rail itself, so nothing will ever read the router's per-scope
        # copy of the decision: it is released here or never.
        _release_scoped_routing_rail(router, scope)

    # Drain the router's per-call rails FIRST, whatever happens next: they are
    # contextvars scoped to the call that just ran, and leaving them full would
    # attribute this call's skills to the next one.
    offered = consume_last_offered_names()
    delivered = consume_last_delivered_names()
    delivered_versions = consume_last_delivered_versions()

    texts = [t for t in (prefix, tail) if isinstance(t, str) and t]
    if not texts:
        return  # nothing routed: claim nothing, inject nothing

    index, role = _insertion_point(messages)
    block: List[_SkillMessage] = []
    for text in texts:
        msg = _SkillMessage(role=role, content=text)
        msg.query = query
        block.append(msg)
    messages[index:index] = block

    if not all(any(m is inserted for m in messages) for inserted in block):
        # Belt and braces, as on ADK: reaching here means the insert did not
        # land, and the one thing that must not happen is reporting the skills
        # as delivered.
        logger.warning(
            "DecimalAI CrewAI: the skill block did not survive the insert; "
            "reporting nothing offered or delivered for this call"
        )
        return

    try:
        from .otel import record_skill_rail

        record_skill_rail(
            routing_id=routing_id,
            offered=offered,
            delivered=delivered,
            delivered_versions=delivered_versions,
            prompt_text="\n\n".join(texts),
        )
    except Exception:
        logger.debug("skill rail recording failed (non-fatal)", exc_info=True)
    try:
        _note_routed_run()
    except Exception:
        logger.debug("routed run not noted (non-fatal)", exc_info=True)


def _skill_hook(context: Any) -> None:
    """The ``before_llm_call`` hook. Never blocks the call and never raises.

    Returning ``False`` from a CrewAI before-hook BLOCKS the model call, so this
    returns None on every path: a routing failure degrades to an unskilled call,
    it does not take the caller's agent down.
    """
    try:
        _inject_skills(context)
    except Exception:  # noqa: BLE001
        logger.debug("CrewAI skill injection failed (non-fatal)", exc_info=True)
    return None


def _install_skill_loader() -> bool:
    """Register the hook with CrewAI, once. True when it is registered.

    Checked against CrewAI's live registry rather than a flag alone, so a caller
    who cleared the global hooks (``crewai.hooks.clear_all_global_hooks()``) and
    calls ``instrument(enable_skill_loader=True)`` again gets the loader back.
    """
    global _skill_loader_installed
    try:
        from crewai.hooks import (
            get_before_llm_call_hooks,
            register_before_llm_call_hook,
        )
    except ImportError:
        logger.warning(
            "DecimalAI CrewAI skill loader: crewai cannot be imported, so there "
            "is no CrewAI hook registry to install it into (pip install "
            "'crewai>=1.15.27'). Skills will NOT be delivered; tracing is "
            "unaffected."
        )
        return False
    with _install_lock:
        if _skill_hook not in get_before_llm_call_hooks():
            register_before_llm_call_hook(_skill_hook)
        _skill_loader_installed = True
    logger.info("DecimalAI SkillRouter loader installed (CrewAI before_llm_call hook)")
    return True


# ── native mode: CrewAI's own skill loader ──────────────────
#
# Read out of crewai 1.15.27:
#
# * ``Agent(skills=[path])`` parses each skill's FRONTMATTER when the agent is
#   built (``discover_skills``: metadata level) and reads its BODY from disk
#   each time ``load_skill`` runs (``activate_skill`` -> ``load_skill_
#   instructions``), returning ``format_skill_context(activated)``. Nothing
#   keeps the activated copy: the text the tool returned is the only place the
#   delivered body exists, so that text is what is recorded.
# * ``SkillUsedEvent`` carries the skill's name and folder, not its body, and
#   the agent's parsed copy of a metadata-level skill has no body. It does carry
#   one for an always-on (activated) skill, which no tool call delivers.
# * Global ``post_tool_call`` hooks are resolved at call time and run on the
#   run's thread, after the tool and before the result goes to the model; a
#   result CrewAI's tool cache served goes through them too.


def _wire_names(names: Sequence[str]) -> List[str]:
    """Each listed name as the trace reports that skill (decimalai._native, D6)."""
    from .skillpkg.identity import current_index

    index = current_index()
    out: List[str] = []
    for name in names:
        managed = index.get(name)
        wire = managed.wire_name if managed is not None else name
        if wire not in out:
            out.append(wire)
    return out


def _metadata_of(skill: Any) -> Optional[dict]:
    metadata = getattr(getattr(skill, "frontmatter", None), "metadata", None)
    return dict(metadata) if isinstance(metadata, dict) else None


def _body_in_skill_context(text: str, label: str, skill: Any) -> Optional[str]:
    """The body inside ``format_skill_context(activated, label=label)``'s text:
    ``<skill name="{label}">``, the description, a blank line, the body,
    ``</skill>``. None when the text is not that shape."""
    head = f'<skill name="{label}">\n'
    tail = "\n</skill>"
    if not (text.startswith(head) and text.endswith(tail)):
        return None
    inner = text[len(head):-len(tail)]
    description = getattr(skill, "description", None)
    if isinstance(description, str) and inner.startswith(description + "\n\n"):
        return inner[len(description) + 2:]
    return None


def _record_crewai_load(load: Any) -> None:
    """Record ``load`` on the CrewAI run that is live here.

    On that run's own rail (the exporter applies it to the run's trace), not
    through ``record``'s default order, which would put it on an enclosing
    ``decimalai.start_trace()``: the CrewAI run is that trace's child, and it is
    the run that loaded the skill.
    """
    from ._native import record
    from .otel import current_run_key, native_loads_for_current_run

    loads = native_loads_for_current_run()
    activation = record(load, loads=loads) if loads is not None else record(load)
    if activation is not None:
        _warn_if_two_catalogs(current_run_key(), activation.name)


def _observe_load_skill(context: Any) -> None:
    from crewai.skills.tool import LoadSkillTool

    from ._native import Load

    original = getattr(getattr(context, "tool", None), "_original_tool", None)
    if not isinstance(original, LoadSkillTool):
        return
    tool_input = getattr(context, "tool_input", None)
    label = tool_input.get("skill_name") if isinstance(tool_input, dict) else None
    skill = original.catalog.get(label) if isinstance(label, str) else None
    text = getattr(context, "tool_result", None)
    if skill is None or not isinstance(text, str) or not text.startswith(f'<skill name="{label}">'):
        return  # an unknown name: the tool answered with a refusal, nothing was delivered
    body = _body_in_skill_context(text, label, skill)
    # When the shape is not the one known here, the whole text goes in: the
    # body hash then names no version, and the activation is the skill alone.
    _record_crewai_load(Load(
        framework="crewai", name=skill.name, body=body, text=None if body is not None else text,
        metadata=_metadata_of(skill), path=str(getattr(skill, "path", "") or "") or None,
    ))


def _native_tool_hook(context: Any) -> None:
    """The ``post_tool_call`` hook. Returns None (the result is left as it is)
    on every path, and never raises."""
    try:
        _observe_load_skill(context)
    except Exception:  # noqa: BLE001 - observing must never break the run
        logger.debug("CrewAI native skill load not recorded (non-fatal)", exc_info=True)
    return None


def _on_skill_used(source: Any, event: Any) -> None:
    """``SkillUsedEvent``: an always-on skill went into this task's prompt.

    ``source`` is the agent (the event's own ``from_agent`` is cleared by
    CrewAI). A metadata-level skill is left to the tool hook, which saw its
    body. Runs on CrewAI's handler pool, in a copy of the emitting context, so
    the run's OpenTelemetry context is current here too. Never raises.
    """
    try:
        from crewai.skills.models import INSTRUCTIONS

        from ._native import Load

        name = getattr(event, "skill_name", None)
        for skill in getattr(source, "skills", None) or ():
            if getattr(skill, "name", None) != name:
                continue
            body = getattr(skill, "instructions", None)
            if getattr(skill, "disclosure_level", 0) < INSTRUCTIONS or not isinstance(body, str):
                return
            _record_crewai_load(Load(
                framework="crewai", name=skill.name, body=body, metadata=_metadata_of(skill),
                path=str(getattr(skill, "path", "") or "") or None,
            ))
            return
    except Exception:  # noqa: BLE001
        logger.debug("CrewAI SkillUsedEvent not recorded (non-fatal)", exc_info=True)


_AVAILABLE_SKILLS = re.compile(r"<available_skills>(.*?)</available_skills>", re.S)
_CATALOG_ENTRY = re.compile(r'<skill name="([^"]+)">')


def _native_listing_hook(context: Any) -> None:
    """A ``before_llm_call`` hook: the names in CrewAI's ``<available_skills>``
    catalog of this call, recorded as offered on the run. Never blocks the call
    (returns None) and never raises."""
    try:
        if getattr(context, "executor", None) is None:
            return None
        names: List[str] = []
        for message in getattr(context, "messages", None) or ():
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            if role not in _INSTRUCTION_ROLES:
                continue
            content = message.get("content") if isinstance(message, dict) else getattr(
                message, "content", None)
            for block in _AVAILABLE_SKILLS.findall(_content_text(content)):
                names.extend(n for n in _CATALOG_ENTRY.findall(block) if n not in names)
        if not names:
            return None
        from .otel import native_loads_for_current_run

        loads = native_loads_for_current_run()
        if loads is not None:
            loads.listed(_wire_names(names))
    except Exception:  # noqa: BLE001
        logger.debug("CrewAI native skill listing not recorded (non-fatal)", exc_info=True)
    return None


def _install_native_observer() -> bool:
    """Register the native-load observer with CrewAI, once. True when it is.

    Called wherever CrewAI tracing is set up (``_wire_tracer_provider``): a load
    is recorded on a run's trace, so without tracing there is nothing to record
    it on. The hooks are checked against CrewAI's live registries, as
    ``_install_skill_loader`` does; the event-bus subscription cannot be read
    back, so a flag guards it.
    """
    global _native_bus_subscribed
    try:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.skill_events import SkillUsedEvent
        from crewai.hooks import (
            get_after_tool_call_hooks,
            get_before_llm_call_hooks,
            register_after_tool_call_hook,
            register_before_llm_call_hook,
        )
    except ImportError:
        return False
    with _install_lock:
        if _native_tool_hook not in get_after_tool_call_hooks():
            register_after_tool_call_hook(_native_tool_hook)
        if _native_listing_hook not in get_before_llm_call_hooks():
            register_before_llm_call_hook(_native_listing_hook)
        if not _native_bus_subscribed:
            crewai_event_bus.on(SkillUsedEvent)(_on_skill_used)
            _native_bus_subscribed = True
    return True


# ── the pointer helpers ─────────────────────────────────────

_PathLike = Union[str, "os.PathLike[str]"]

#: The folders of a skill ``load_skill_resource`` reads from (the Agent Skills
#: layout), and the one ``run_skill_script`` runs from.
_RESOURCE_FOLDERS = ("references", "assets", "scripts")
_SCRIPT_FOLDERS = ("scripts",)
#: Bounds on what one tool call hands the model.
_MAX_RESOURCE_CHARS = 200_000
_MAX_SCRIPT_OUTPUT_CHARS = 50_000


def skills(directory: Optional[_PathLike] = None) -> List[Path]:
    """The project's installed skills, for CrewAI's own loader::

        Agent(..., skills=decimalai.crewai.skills(), tools=decimalai.crewai.skill_tools())

    ``[decimalai.skills_dir()]``: CrewAI discovers every skill folder under it,
    offers them in its ``<available_skills>`` catalog and gives the agent its
    ``load_skill`` tool. CrewAI has no tool to read a skill's other files or run
    its scripts; :func:`skill_tools` adds them.

    Args:
        directory: where to start looking for the project (default: the working
            directory), as for :func:`decimalai.skills_dir`.

    Raises:
        Whatever :func:`decimalai.skills_dir` raises for a project with no
        installed skills.
    """
    from .skillpkg.project import skills_dir

    return [skills_dir(directory)]


def skill_tools(directory: Optional[_PathLike] = None, *, scripts: bool = True,
                timeout: float = 60.0) -> List[Any]:
    """CrewAI tools for the parts of a skill CrewAI's loader does not reach.

    For ``Agent(tools=...)``, beside ``skills=`` (:func:`skills`). A separate
    helper because CrewAI keeps the two apart: ``skills=`` takes skill folders,
    which cannot carry tools, and its loader has no hook to add one.

    * ``load_skill_resource(skill_name, file_path)`` returns a file of an
      installed skill, under its ``references/``, ``assets/`` or ``scripts/``.
    * ``run_skill_script(skill_name, file_path, args)`` runs a script under its
      ``scripts/``: ``.py`` with this interpreter, ``.sh`` with ``bash``, any
      other executable file directly; no shell, the skill's folder as the
      working directory, ``timeout`` seconds at most. It returns the script's
      output and exit status. Leave it out with ``scripts=False``.

    Both accept only the skills installed under :func:`decimalai.skills_dir`
    (by the name ``load_skill`` knows them by) and only paths inside that
    skill's folder, after resolving symlinks. Neither result is cached by
    CrewAI's tool cache: after ``decimalai skills update`` a call reads the
    installed files.

    Args:
        directory: where to start looking for the project, as for :func:`skills`.
        scripts: include ``run_skill_script``.
        timeout: the longest a script may run, in seconds.

    Raises:
        Whatever :func:`decimalai.skills_dir` raises; ``ImportError`` without crewai.
    """
    from .skillpkg.project import skills_dir

    resource_tool, script_tool = _skill_tool_classes()
    root = str(skills_dir(directory))
    tools: List[Any] = [resource_tool(root=root)]
    if scripts:
        tools.append(script_tool(root=root, timeout=float(timeout)))
    return tools


def _skill_file(root: Path, skill_name: Any, file_path: Any,
                folders: Sequence[str]) -> Union[Path, str]:
    """The file ``file_path`` names in the installed skill ``skill_name``, or the
    error message the model gets instead."""
    if not isinstance(skill_name, str) or not skill_name or skill_name in (".", "..") \
            or "/" in skill_name or "\\" in skill_name:
        return f"Error: there is no installed skill named {skill_name!r}."
    root = root.resolve()
    folder = root / skill_name
    if not (folder / "SKILL.md").is_file() or folder.resolve().parent != root:
        return f"Error: there is no installed skill named {skill_name!r}."
    folder = folder.resolve()
    if not isinstance(file_path, str) or not file_path:
        return "Error: file_path is required, e.g. references/<file>."
    target = (folder / file_path).resolve()
    try:
        parts = target.relative_to(folder).parts
    except ValueError:
        return f"Error: {file_path!r} is outside the skill {skill_name!r}."
    if not parts or parts[0] not in folders:
        where = ", ".join(f"{f}/" for f in folders)
        return f"Error: {file_path!r} is not under {where} of the skill {skill_name!r}."
    if not target.is_file():
        return f"Error: the skill {skill_name!r} has no file {file_path!r}."
    return target


def _read_skill_file(root: Path, skill_name: Any, file_path: Any) -> str:
    target = _skill_file(root, skill_name, file_path, _RESOURCE_FOLDERS)
    if isinstance(target, str):
        return target
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: {file_path!r} is not a text file."
    except OSError as exc:
        return f"Error: could not read {file_path!r}: {exc}"
    if len(text) > _MAX_RESOURCE_CHARS:
        text = text[:_MAX_RESOURCE_CHARS] + f"\n[... cut at {_MAX_RESOURCE_CHARS} characters]"
    return text


def _run_skill_script(root: Path, skill_name: Any, file_path: Any, args: Any,
                      timeout: float) -> str:
    target = _skill_file(root, skill_name, file_path, _SCRIPT_FOLDERS)
    if isinstance(target, str):
        return target
    argv = [str(a) for a in args] if isinstance(args, (list, tuple)) else (
        [] if args in (None, "") else [str(args)])
    suffix = target.suffix.lower()
    if suffix == ".py":
        command = [sys.executable, str(target), *argv]
    elif suffix == ".sh":
        command = ["bash", str(target), *argv]
    elif os.access(target, os.X_OK):
        command = [str(target), *argv]
    else:
        return f"Error: {file_path!r} is not a .py or .sh script, nor an executable file."
    skill_folder = (root / skill_name).resolve()  # checked by _skill_file
    try:
        done = subprocess.run(command, cwd=str(skill_folder), capture_output=True, text=True,
                              timeout=timeout, stdin=subprocess.DEVNULL, check=False)
    except subprocess.TimeoutExpired:
        return f"Error: {file_path!r} did not finish within {timeout:g}s and was stopped."
    except OSError as exc:
        return f"Error: could not run {file_path!r}: {exc}"
    # stdout first: it is the answer, and a trace's span preview keeps the head.
    out = [done.stdout[:_MAX_SCRIPT_OUTPUT_CHARS]]
    if done.stderr:
        out.append(f"[stderr]\n{done.stderr[:_MAX_SCRIPT_OUTPUT_CHARS]}")
    out.append(f"[exit status {done.returncode}]")
    return "\n".join(out)


_SkillToolClasses: Optional[tuple] = None


def _skill_tool_classes() -> tuple:
    """Build (once) the two CrewAI tool classes. Deferred so that importing this
    module never imports crewai."""
    global _SkillToolClasses
    if _SkillToolClasses is not None:
        return _SkillToolClasses
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    def _never_cache(_args: Any, _result: Any) -> bool:
        return False

    class _ResourceArgs(BaseModel):
        skill_name: str = Field(..., description="The skill's name, as load_skill takes it.")
        file_path: str = Field(..., description=(
            "The file's path inside the skill, under references/, assets/ or scripts/, "
            "e.g. references/fees.md."))

    class _ScriptArgs(BaseModel):
        skill_name: str = Field(..., description="The skill's name, as load_skill takes it.")
        file_path: str = Field(..., description=(
            "The script's path inside the skill, under scripts/, e.g. scripts/run.py."))
        args: List[str] = Field(default_factory=list,
                                description="Command-line arguments for the script.")

    class LoadSkillResourceTool(BaseTool):
        name: str = "load_skill_resource"
        description: str = (
            "Read a file that belongs to an installed skill: one under its references/, "
            "assets/ or scripts/ folder. Load the skill first; its instructions name the "
            "files it uses.")
        args_schema: type = _ResourceArgs
        cache_function: Callable[..., bool] = _never_cache
        root: str = ""

        def _run(self, skill_name: str, file_path: str, **_: Any) -> str:
            return _read_skill_file(Path(self.root), skill_name, file_path)

    class RunSkillScriptTool(BaseTool):
        name: str = "run_skill_script"
        description: str = (
            "Run a script that belongs to an installed skill (under its scripts/ folder) "
            "with the given arguments, and return its output. Load the skill first; its "
            "instructions say when to run which script.")
        args_schema: type = _ScriptArgs
        cache_function: Callable[..., bool] = _never_cache
        root: str = ""
        timeout: float = 60.0

        def _run(self, skill_name: str, file_path: str, args: Optional[List[str]] = None,
                 **_: Any) -> str:
            return _run_skill_script(Path(self.root), skill_name, file_path, args or [],
                                     self.timeout)

    _SkillToolClasses = (LoadSkillResourceTool, RunSkillScriptTool)
    return _SkillToolClasses


# ── the run's trace: its caller, its observer, its flush ────


def _enclosing_trace_id() -> Optional[str]:
    """The live ``decimalai.start_trace()`` context's id, or None.

    The same source and the same fail-open rule as the LangChain adapter's
    ``_enclosing_generic_trace_id`` and ADK's ``before_run_callback``: a tracer
    that raises while deciding how to label a trace is worse than an unlinked
    trace, and None means "no enclosing trace" — never a wrong parent.
    """
    try:
        from .generic import _get_current_trace

        ctx = _get_current_trace()
        return ctx.get_trace_id() if ctx is not None else None
    except Exception:
        return None


def _span_scope_name(span: Any) -> str:
    """The instrumentation scope a span was started under, or ""."""
    scope = getattr(span, "instrumentation_scope", None)
    name = getattr(scope, "name", None)
    return name if isinstance(name, str) else ""


def _span_trace_id(span: Any) -> int:
    """The span's OTel trace id — the key the exporter assembles its trace by."""
    ctx = getattr(span, "context", None)
    if ctx is None:
        get = getattr(span, "get_span_context", None)
        ctx = get() if callable(get) else None
    tid = getattr(ctx, "trace_id", 0)
    return tid if isinstance(tid, int) else 0


class _RunLinker:
    """Span processor: at a CrewAI run's first span, record who asked for it.

    ``on_start`` runs on the thread that starts the span, in that thread's
    context — inside ``Crew.kickoff()`` (or ``Agent.kickoff()``, or the
    ``asyncio.to_thread`` that ``kickoff_async`` copies the context into), so it
    can still read the enclosing ``decimalai.start_trace()`` and the observer
    ``instrument()`` configured. The exporter cannot: it assembles the trace
    later, on its own thread. What is read here travels to it through
    ``decimalai.otel.record_run_link``, keyed by the trace id, and the first span
    of a run decides; every later CrewAI span of that run is a no-op there.

    Implements the OTel ``SpanProcessor`` protocol by duck typing, in full —
    the SDK calls ``_on_ending`` from inside ``span.end()`` — like
    ``decimalai.otel._AgentNameStamper``. Never raises: a span start is the
    caller's own code path.
    """

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        try:
            if not _span_scope_name(span).startswith(_CREWAI_SCOPE):
                return
            tid = _span_trace_id(span)
            if not tid:
                return
            with _config_lock:
                settings = _instrument_config
            from .otel import record_run_link

            record_run_link(
                tid,
                parent_trace_id=_enclosing_trace_id(),
                on_trace=settings.on_trace,
            )
        except Exception:  # pragma: no cover - linking must never break a span
            logger.debug("Could not record the CrewAI run's link", exc_info=True)

    def _on_ending(self, span: Any) -> None:
        return None

    def on_end(self, span: Any) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _wire_tracer_provider(provider: Any) -> None:
    """Remember a provider CrewAI's spans now go to, and link its runs.

    Called by ``decimalai._activate_crewai_instrumentation`` once the CrewAI
    instrumentor is on — the one path both ``init(crewai=True)`` and
    :func:`instrument` take — so every pipeline this SDK builds for CrewAI gets
    the :class:`_RunLinker` and is flushed by ``decimalai.flush()``. A pipeline
    this SDK did not build is never touched. Idempotent per provider.
    """
    # The native-load observer rides the same setup: both ``init(crewai=True)``
    # and ``instrument()`` come through here, and a load needs a traced run.
    try:
        _install_native_observer()
    except Exception:  # noqa: BLE001 - observing must never cost tracing
        logger.debug("CrewAI native-load observer not installed", exc_info=True)
    with _install_lock:
        if any(p is provider for p in _tracer_providers):
            return
        _tracer_providers.append(provider)
    add = getattr(provider, "add_span_processor", None)
    if not callable(add):
        return
    try:
        add(_RunLinker())
    except Exception:
        logger.warning(
            "decimalai: could not attach the CrewAI run linker to the tracer "
            "provider; CrewAI traces will carry no parent_trace_id and on_trace "
            "will not be called.",
            exc_info=True,
        )


def _flush_tracing(timeout_s: Optional[float] = None) -> bool:
    """Export every CrewAI run whose spans are still queued. True if it finished.

    What ``decimalai.flush()`` calls first, so the trace exists — assembled,
    observed, handed to the sender — before the sender is drained. Bounded: the
    export runs on a helper thread and is waited for at most ``timeout_s``
    (default ``_FLUSH_TIMEOUT_S``). The SDK's ``force_flush`` cannot bound
    itself, and an unbounded wait would also deadlock a flush called while the
    export lock is held on this thread — from inside an observer, or from a
    SIGTERM handler that interrupted an export. Past the bound the export
    carries on in the background; the trace still ships when it finishes.

    A no-op when this SDK never wired CrewAI tracing, so a process that does not
    use CrewAI is unaffected.
    """
    if timeout_s is None:
        timeout_s = _FLUSH_TIMEOUT_S
    with _install_lock:
        providers = list(_tracer_providers)
    if not providers:
        return True
    done = threading.Event()
    budget_ms = max(1, int(timeout_s * 1000))

    def _export() -> None:
        try:
            for provider in providers:
                flush = getattr(provider, "force_flush", None)
                if callable(flush):
                    try:
                        flush(budget_ms)
                    except Exception:
                        logger.debug("CrewAI tracer provider flush failed", exc_info=True)
        finally:
            done.set()

    try:
        threading.Thread(
            target=_export, name="decimalai-crewai-flush", daemon=True,
        ).start()
    except RuntimeError:
        # The interpreter is shutting down and will not start a thread. The
        # provider's own exit hook (``otel._register_flush_atexit``) flushes it.
        logger.debug("CrewAI trace flush skipped: interpreter shutting down")
        return False
    if done.wait(timeout_s):
        return True
    logger.warning(
        "decimalai.flush(): CrewAI's trace pipeline did not finish exporting "
        "within %.0fs. The export continues in the background and the trace "
        "is still sent when it completes, but export_status() does not cover "
        "it yet.",
        timeout_s,
    )
    return False


# ── tracing ─────────────────────────────────────────────────


def _install_tracing(agent_name: Optional[str]) -> None:
    """What ``init(crewai=True)`` does, once per process."""
    global _tracing_installed
    with _install_lock:
        already = _tracing_installed
        _tracing_installed = True
    if already:
        if agent_name:
            # The exporter's default name was fixed when it was built; a later
            # name can only travel with the spans. Same move otel.instrument()
            # makes on a second call.
            from .otel import _active_agent_name

            _active_agent_name.set(agent_name)
        return

    from . import _activate_crewai_instrumentation
    from .otel import instrument as _otel_instrument

    try:
        provider = _otel_instrument(agent_name=agent_name)
    except ImportError:
        logger.warning(
            "decimalai.crewai.instrument(): the OpenTelemetry SDK is missing, so "
            "CrewAI runs will not be traced. It ships as a core dependency of "
            "decimalai — reinstall with: pip install decimalai"
        )
        return
    _activate_crewai_instrumentation(provider, agent_name=agent_name)


def instrument(
    agent_name: Optional[str] = None,
    *,
    enable_skill_loader: bool = False,
    enable_load_skill_tool: bool = False,
    on_trace: Optional[Callable[["RunTrace"], None]] = None,
) -> None:
    """Install DecimalAI for CrewAI: tracing, and optionally skill delivery.

    Idempotent. Call it before the first ``kickoff()``: CrewAI copies the global
    hook list into each agent executor when it builds one.

    A run started inside ``decimalai.start_trace()`` is filed as that trace's
    child (``parent_trace_id``), whether or not an observer is set — the same
    link the LangChain and ADK adapters make.

    Args:
        agent_name: The DecimalAI agent these runs belong to. Names the traces,
            and scopes the routed skill menu to this agent — agent-scope skills
            (a registry skill pulled onto one agent) are only offered when it is
            sent.
        enable_skill_loader: Route every agent turn through ``SkillRouter`` and
            insert the result — skill bodies included — into the model call's
            messages, after CrewAI's own system prompt. Off by default, like
            every other adapter's loader.
        enable_load_skill_tool: Accepted and DORMANT. The loader is a
            before-call hook: it edits the messages of one model call and owns
            no tool registry, so there is no ``load_skill`` tool for the model to
            ask for a body with. True logs a warning naming the adapters that do
            have the tool loop, and stays on prompt injection. Accepted rather
            than rejected so the refusal happens out loud — the conformance
            suite asks for the tool loop precisely so this warning has to be
            emitted on the run.
        on_trace: Optional observer, called once per CrewAI run — a failed run
            included — with a detached copy of that run's finished trace (a
            ``RunTrace``: its ``id``, ``parent_trace_id``, delivery rails and
            model calls), before export, on the thread that exports it (the
            OpenTelemetry batch worker, or the caller of ``decimalai.flush()``).
            Exceptions it raises are logged and reach neither the run nor the
            export. Captured when a run starts: each ``instrument()`` call sets
            the observer for runs that start afterwards (``None`` removes it),
            and a run already in flight keeps the one it started with.
    """
    global _install_agent_name, _instrument_config
    if agent_name is not None:
        _install_agent_name = agent_name
    # Published before tracing is installed, so the first run after this call
    # already starts with it.
    with _config_lock:
        _instrument_config = _InstrumentationConfig(on_trace=on_trace)
    if enable_load_skill_tool:
        logger.warning(
            "enable_load_skill_tool is not supported on the crewai adapter "
            "(a before_llm_call hook edits one model call's messages and registers "
            "no tool, so the model cannot ask for a body); staying on prompt "
            "injection. Use openai_agents or pydantic_ai for the native load_skill "
            "tool."
        )
    _install_tracing(agent_name)
    if enable_skill_loader:
        from .skill_router import _warn_if_disk_runtime_detected

        _warn_if_disk_runtime_detected("crewai")
        _install_skill_loader()
    logger.info(
        "DecimalAI CrewAI integration installed (agent_name=%s, skill_loader=%s, "
        "trace_observer=%s)",
        agent_name, enable_skill_loader, on_trace is not None,
    )
