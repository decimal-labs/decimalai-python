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

SUPPORTED: crewai >= 1.15.3, the lowest version on which the conformance column
— tracing, the skills rail and the delivery cells — passes end to end (run on
1.15.3, 1.15.20 and 1.15.23). The hook API itself exists from 1.5.0, but every
delivery claim rides the run's trace, and below 1.15.3 the trace is the problem:
the OpenInference CrewAI instrumentor only instruments crewai >= 1.10.1, crewai
1.7–1.14 pin ``click<8.2`` and cannot be installed beside decimalai, and 1.15.0
records no completion text. On an older crewai the hook still inserts the block;
with no crew span to attribute it to, the trace never says so.

THE SEAM, read out of the installed package (crewai 1.15.20 and 1.6.1) rather
than remembered:

* ``crewai.hooks.register_before_llm_call_hook`` is public, documented API,
  present from crewai 1.5.0. Every agent executor copies the global hook list
  when it is built (``CrewAgentExecutor.__init__``;
  ``experimental.AgentExecutor._setup_executor`` on 1.15) and runs it in
  ``utilities/agent_utils.py::_setup_before_llm_call_hooks`` immediately before
  ``llm.call``. It is CrewAI's own extension point for exactly this, the way
  ADK's ``before_model_callback`` is ADK's.
* The hook context's ``messages`` IS the executor's live message list
  (``LLMCallHookContext.__init__``: ``self.messages = executor.messages``), and
  ``get_llm_response`` hands that same list to ``llm.call`` after the hooks ran
  (1.15: ``_prepare_llm_call`` yields ``executor_context.messages``; 1.6.1:
  ``messages = executor_context.messages``). What the hook inserts is what the
  provider is sent — and what the provider instrumentor records, which is how it
  reaches the trace's ``rendered_input``.
* That list PERSISTS across the iterations of one task (tool call, observation,
  answer), so the block inserted for the previous call is still in it on the
  next one. It is replaced, never stacked: the inserted messages are
  :class:`_SkillMessage` instances, found and removed by type before the next
  insert.
* A call dispatched with NO executor is left alone. On crewai >= 1.7 the LLM
  layer also dispatches these hooks for calls that are not an agent turn —
  CrewAI's own output converter, guardrails and planner, or ``llm.call()``
  straight from user code (``BaseLLM._invoke_before_llm_call_hooks``, context
  built with ``executor=None``). A skill body has no business in a JSON
  conversion prompt, and claiming delivery for it would credit the skill with a
  call the agent never made.

What the hook cannot reach, stated rather than discovered later: an executor
built BEFORE ``instrument(enable_skill_loader=True)`` keeps the hook list it
copied. crewai 1.15 builds an agent's executor on its first task and reuses it,
so enable the loader before the first ``kickoff()``. ``Crew.kickoff()``,
``Crew.kickoff_async()`` and ``Agent.kickoff()`` all dispatch through an
executor and all get skills.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence

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


def _release_scoped_rail(router: Any, scope: Optional[str]) -> None:
    """Drain what the router filed under this run's scope.

    This adapter reads the per-call contextvar rails and records them on the OTel
    rail itself, so nothing else will ever read the router's per-scope copy — and
    left in place, every run would hold one of the router's 4096 scoped slots
    until eviction, which then warns on every new run.
    """
    if scope is None:
        return
    try:
        router.consume_routing_id(scope=scope)
        router.consume_offered_names(scope=scope)
        router.consume_delivered_names(scope=scope)
    except Exception:
        logger.debug("router keeps no scoped rail to release", exc_info=True)


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

    router = _get_skill_router()
    if router is None:
        return

    query = previous.query if previous is not None and previous.query else _query_for(
        context, messages
    )
    scope = _scope()
    agent_name = _routing_agent_name()
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
        logger.debug("build_prompt_parts failed (non-fatal)", exc_info=True)
        return

    # Drain the router's per-call rails FIRST, whatever happens next: they are
    # contextvars scoped to the call that just ran, and leaving them full would
    # attribute this call's skills to the next one.
    from .skill_router import consume_last_delivered_names, consume_last_offered_names

    offered = consume_last_offered_names()
    delivered = consume_last_delivered_names()
    _release_scoped_rail(router, scope)

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
            prompt_text="\n\n".join(texts),
        )
    except Exception:
        logger.debug("skill rail recording failed (non-fatal)", exc_info=True)


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
            "enable_skill_loader=True but CrewAI's before_llm_call hook API is not "
            "available (it arrived in crewai 1.5.0; the loader is supported on "
            "crewai>=1.15.3 — pip install -U 'crewai>=1.15.3'). Skills will NOT be "
            "delivered to this crew; tracing is unaffected."
        )
        return False
    with _install_lock:
        if _skill_hook not in get_before_llm_call_hooks():
            register_before_llm_call_hook(_skill_hook)
        _skill_loader_installed = True
    logger.info("DecimalAI SkillRouter loader installed (CrewAI before_llm_call hook)")
    return True


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
) -> None:
    """Install DecimalAI for CrewAI: tracing, and optionally skill delivery.

    Idempotent. Call it before the first ``kickoff()``: CrewAI copies the global
    hook list into each agent executor when it builds one.

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
    """
    global _install_agent_name
    if agent_name is not None:
        _install_agent_name = agent_name
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
        "DecimalAI CrewAI integration installed (agent_name=%s, skill_loader=%s)",
        agent_name, enable_skill_loader,
    )
