"""``decimalai.langchain.agent_config()``: one agent's prompt and skills, as a LangChain middleware.

::

    from langchain.agents import create_agent
    from decimalai.langchain import agent_config

    support = agent_config("support", fallback=BASE_PROMPT)
    agent = create_agent("openai:gpt-5.4-mini", tools=[lookup_order], middleware=[support.middleware])
    agent.invoke({"messages": [{"role": "user", "content": question}]}, context=customer)

Per invocation of the agent (one user message):

- Before the agent runs, the message is routed and DecimalAI's system text is composed: the
  saved prompt, filled from the run's ``context=``, then the skill list.
- A thread keeps its first invocation's system text and ``load_skill`` definition for every
  later message. Claude rejects a conversation whose system prompt or tool definitions changed
  once a thinking block is in its history (HTTP 400), and an unchanged prefix is what a
  provider's prompt cache reuses. Both live in the thread's private state, so a checkpointer
  carries them; a newer saved prompt or skill list reaches the next thread.
- A later message whose skill list is cut to fit is routed on its own text. The skills it picks
  that the thread has not shown go into the conversation as one tagged human message after the
  user's (``additional_kwargs["lc_source"] == "decimalai_skills"``), the way Deep Agents adds a
  pinned skill, so earlier messages never change.
- The agent's always-include skills (set on the agent in DecimalAI) go into the conversation in
  full as one tagged human message at the thread's first message
  (``lc_source == "decimalai_always_include"``), and one more when the agent gains another
  mid-thread. ``load_skill`` answers for them that they are already loaded.
- Every model call gets that system text and ``load_skill``, whose definition is fixed for the
  thread. The model reads a listed skill's full text with ``load_skill``; it arrives as the tool
  result.
- The run's record (prompt version, variable names, skills offered and loaded, routing id)
  reaches :class:`decimalai.langchain.CallbackHandler` through its per-call rails, and the trace
  is filed under the DecimalAI agent. ``agent_config()`` turns LangChain tracing on for the
  process (``instrument()``), as ``decimalai.init(langchain=True)`` does.

A hand-built LangGraph graph uses :meth:`LangChainAgentConfig.system_message` and
:attr:`LangChainAgentConfig.tools` instead of the middleware.

This module imports LangChain at import time; ``decimalai.langchain`` imports it only when
``agent_config()`` is called, so tracing alone still needs only ``langchain-core``.
"""

from __future__ import annotations

import re
import sys
import threading
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.agents.middleware.types import PrivateStateAttr
from langchain.tools import InjectedState
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import Annotated, NotRequired

from ._agent_binding import (
    _MAX_CONFIG_RUNS,
    AgentBinding,
    ConversationRecord,
    _in_thread,
    new_skills_block,
)
from .skill_router import _load_routing_id_ctx

#: The run the code on this context belongs to (a model call, a load, a routing call).
_RUN: ContextVar[Optional[str]] = ContextVar("decimalai_langchain_config_run", default=None)

#: The thread's private state keys: the invocation's run, and what the thread keeps.
RUN_KEY = "decimalai_run"
CONVERSATION_KEY = "decimalai_conversation"
#: ``additional_kwargs["lc_source"]`` of the human message carrying a later message's new skills.
SKILLS_SOURCE = "decimalai_skills"
#: ``additional_kwargs["lc_source"]`` of the human message carrying the always-include skills' text.
ALWAYS_SOURCE = "decimalai_always_include"

#: Background routing decisions the CallbackHandler waits for when it builds a run's trace.
_MAX_LATE_DECISIONS = 1024
_late_lock = threading.Lock()
_late: "OrderedDict[str, Tuple[Any, LangChainAgentConfig]]" = OrderedDict()


def late_decision(run_key: str, wait_s: float) -> Optional[str]:
    """The routing id a run's background decision named, waiting up to ``wait_s`` for it.

    For :class:`decimalai.langchain.CallbackHandler`: a later run of an agent whose skill list
    fits shows the local copy at once and records its own decision in the background, which
    may land after the run's last model call.
    """
    with _late_lock:
        entry = _late.pop(run_key, None)
    if entry is None:
        return None
    pending, config = entry
    if pending is not None and not pending.done():
        try:
            pending.result(timeout=wait_s)
        except Exception:
            config._log.debug("a run's routing decision did not land in time", exc_info=True)
    with config._lock:
        return (config._records.get(run_key) or {}).get("routing_id")


#: Below 3.11, ``asyncio.create_task`` takes no ``context=``, so LangGraph cannot hand an async
#: run's config to the model node: its model calls carry no callbacks of the run's.
_ASYNC_RUN_LOSES_ITS_CALLS = sys.version_info < (3, 11)
_async_warned = False


def _warn_async_before_311(config: "LangChainAgentConfig") -> None:
    """Warn once that an async run on Python 3.10 is traced without its model calls.

    Skills are unaffected: the run's key travels in the graph state, not the context.
    """
    global _async_warned
    if _async_warned or not _ASYNC_RUN_LOSES_ITS_CALLS:
        return
    _async_warned = True
    config._log.warning(
        "On Python 3.10 an async agent run (ainvoke, astream) cannot hand its LangChain run to "
        "the model calls inside it, so the run's trace leaves them out (a handler installed by "
        "decimalai.init(langchain=True) traces each one apart from the run). Skills still reach "
        "the model. Use Python 3.11+, or run the agent with .invoke(). Logged once.")


def _always_message(texts: List[Tuple[str, str]]) -> HumanMessage:
    """The always-include skills' full text, as the tagged human message a thread carries."""
    names = [name for name, _section in texts]
    return HumanMessage(
        content=("The full text of skills this agent always uses (already loaded; follow them "
                 "where they apply):\n\n" + "\n\n".join(section for _name, section in texts)),
        additional_kwargs={"lc_source": ALWAYS_SOURCE, "decimalai_injected": True,
                           "decimalai_skills": names},
    )


def _is_ours(message: Any) -> bool:
    kwargs = getattr(message, "additional_kwargs", None)
    return isinstance(kwargs, dict) and bool(kwargs.get("decimalai_injected"))


def _last_human(messages: Any) -> Optional[Any]:
    """The run's latest user message: the latest human message DecimalAI did not add."""
    for message in reversed(list(messages or [])):
        kind = getattr(message, "type", None)
        if (isinstance(message, HumanMessage) or kind == "human") and not _is_ours(message):
            return message
    return None


def _run_key_of(state: Any) -> Optional[str]:
    """The run a graph state belongs to: the key ``before_agent`` minted, else its latest user message."""
    state = state if isinstance(state, Mapping) else {}
    key = state.get(RUN_KEY)
    if isinstance(key, str) and key:
        return key
    human = _last_human(state.get("messages"))
    message_id = getattr(human, "id", None) if human is not None else None
    return f"msg:{message_id}" if message_id else None


class LangChainAgentConfig(AgentBinding):
    """One DecimalAI agent's prompt and skills, for LangChain and LangGraph.

    Made by :func:`decimalai.langchain.agent_config`. Pass :attr:`middleware` to
    ``create_agent(middleware=[...])``; a hand-built graph uses :meth:`system_message` and
    :attr:`tools`.

    Attributes:
        name: The agent's canonical DecimalAI name.
        middleware: A LangChain ``AgentMiddleware``: the system text and ``load_skill`` for
            every model call, the thread's record in its private state.
        tools: ``[load_skill]``, for a hand-built graph's ``ToolNode`` and ``bind_tools``.
    """

    _VALUES_HINT = ("Pass them as the run's context=... (create_agent(context_schema=...)), "
                    "as attributes or dict keys.")

    #: The always-include skills' text goes into the conversation as a tagged message, so the
    #: system text a thread keeps stays the prompt and the list.
    _ALWAYS_IN_SYSTEM = False

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str) -> None:
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        #: Each run's record for its trace, by run key: prompt, offered, routing id, delivered.
        self._records: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self.tools: List[Any] = [self._build_tool()]
        self.middleware = _build_middleware(self)

    def __repr__(self) -> str:
        return (f"LangChainAgentConfig(name={self.name!r}, prompt_version={self.prompt_version!r}, "
                f"skills={self.skill_names!r})")

    @property
    def prompt(self) -> Optional[str]:
        """The saved prompt as written (variables unfilled), else the fallback."""
        with self._lock:
            text = getattr(self._prompt, "system_prompt", None)
        return text if text is not None else self._fallback

    # ── the seams AgentBinding asks for ─────────────────────
    def _router(self) -> Any:
        from .langchain import _get_skill_router

        return _get_skill_router()

    def _run_key(self) -> Optional[str]:
        return _RUN.get()

    def _message(self, context: Any, agent: Any) -> Optional[str]:
        from .langchain import _content_text

        human = _last_human(context.get("messages") if isinstance(context, Mapping) else None)
        return _content_text(getattr(human, "content", None)) if human is not None else None

    def _values(self, context: Any) -> Any:
        return context.get("values") if isinstance(context, Mapping) else None

    def _agent_label(self, agent: Any) -> Optional[str]:
        return None

    def _record(self, **fields: Any) -> bool:
        run_key = _RUN.get()
        if run_key is None:
            return False
        with self._lock:
            record = self._records.setdefault(run_key, {})
            record.update({key: value for key, value in fields.items() if key != "agent"})
            self._records.move_to_end(run_key)
            while len(self._records) > _MAX_CONFIG_RUNS:
                self._records.popitem(last=False)
        return True

    def _note_pending(self, run_key: Optional[str], pending: Any) -> None:
        if run_key is None or pending is None:
            return
        with _late_lock:
            _late[run_key] = (pending, self)
            _late.move_to_end(run_key)
            while len(_late) > _MAX_LATE_DECISIONS:
                _late.popitem(last=False)

    def _stamp_routing(self, run_key: Optional[str], routing_id: str) -> None:
        if run_key is None:
            return
        with self._lock:
            record = self._records.get(run_key)
            if record is not None and not record.get("routing_id"):
                record["routing_id"] = routing_id

    def _serve_load(self, name: str) -> str:
        router = self._router()
        if router is None:
            return "load_skill: DecimalAI is not configured in this process, so no skill can be loaded."
        # Filed under the run's key, where the run's CallbackHandler drains it.
        return router.load_skill(name, scope=self._run_key(), agent_name=self.name)

    def _conversation(self, context: Any, agent: Any) -> Optional[ConversationRecord]:
        state = context.get("conversation") if isinstance(context, Mapping) else None
        return ConversationRecord.from_state(state, self.name)

    def _keep_conversation(self, context: Any, agent: Any, run_key: Optional[str],
                           record: ConversationRecord) -> None:
        if isinstance(context, dict):
            context["kept"] = record

    def _routed_continuation(self, context: Any, agent: Any) -> Any:
        routed = context.get("routed") if isinstance(context, Mapping) else None
        return routed if routed is not None and routed[2] else None

    # ── one run ─────────────────────────────────────────────
    def _routes_later(self, record: ConversationRecord) -> bool:
        """Whether a later message of this thread is routed before its model call."""
        from ._config import _disabled_by_init

        tool, _inject = self._channels()
        # A thread in an experiment keeps its first decision (its arm) for every message.
        return record.cut and record.tool and tool and not _disabled_by_init() and not record.experiment

    def _route_later_run(self, run_key: str, context: Dict[str, Any]) -> Any:
        """This message's own decision, for a thread whose list is cut (or None)."""
        try:
            routed = self._route(run_key, self._message(context, None))
        except Exception:
            self._log.debug("skill routing failed (non-fatal)", exc_info=True)
            return None
        prefix, offered, routing_id, truncated, _delivered, _versions, _run_prompt = routed
        self._store_menu(prefix, offered, routing_id, truncated)
        return routed

    def _start_sync(self, context: Dict[str, Any], run_key: str) -> Dict[str, Any]:
        """Prepare one run and return the update for the thread's state."""
        token = _RUN.set(run_key)
        try:
            record = self._conversation(context, None)
            if record is not None and self._routes_later(record):
                context["routed"] = self._route_later_run(run_key, context)
            state = self._prepare_run_sync(context, None, run_key)
            if state.get("conversation") is None:
                self._compose(context, None, run_key, state)
            return self._started(context, run_key, state)
        finally:
            _RUN.reset(token)

    async def _start(self, context: Dict[str, Any], run_key: str) -> Dict[str, Any]:
        """`_start_sync` on an event loop: routing calls run in a worker thread."""
        token = _RUN.set(run_key)
        try:
            record = self._conversation(context, None)
            if record is not None and self._routes_later(record):
                context["routed"] = await _in_thread(self._route_later_run, run_key, context)
            state = await self._prepare_run(context, None, run_key)
            if state.get("conversation") is None:
                self._compose(context, None, run_key, state)
            return self._started(context, run_key, state)
        finally:
            _RUN.reset(token)

    def _started(self, context: Dict[str, Any], run_key: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """The thread's state update once a run is prepared: its key, record and new skills."""
        record: ConversationRecord = state["conversation"]
        update: Dict[str, Any] = {RUN_KEY: run_key}
        changed = context.get("kept") is record
        routed = state.get("routed")
        if routed is not None and routed[2]:
            # A later message of a cut-list thread, routed on its own text: the skills it picked
            # that the thread has not shown go in after the user's message; the system text
            # and the tool stay as the thread's first message got them.
            offered, routing_id = list(routed[1]), routed[2]
            new = [name for name in offered if name not in record.shown]
            state["offered_now"], state["routing_now"] = offered, routing_id
            self._record(routing_id=routing_id, offered=offered)
            if new:
                for name in new:
                    record.shown.setdefault(name, routing_id)
                update["messages"] = [HumanMessage(
                    content=new_skills_block(routed[0], new),
                    additional_kwargs={"lc_source": SKILLS_SOURCE, "decimalai_injected": True,
                                       "decimalai_skills": list(new)},
                )]
                changed = True
        from ._config import _disabled_by_init

        always = state.get("always_now")
        if always is None and not _disabled_by_init():
            # A later message: always-include skills the agent gained since the thread began.
            always, versions = self._always_now(skip=record.delivered)
            if always:
                names = [name for name, _section in always]
                record.delivered = [*record.delivered, *names]
                state["loaded"] = set(state.get("loaded") or ()) | set(names)
                self._record(delivered=names, delivered_versions=versions)
                changed = True
        if always:
            update["messages"] = [*update.get("messages", []), _always_message(always)]
        if changed:
            update[CONVERSATION_KEY] = record.to_state(self.name)
        return update

    def _end_run(self, run_key: Optional[str]) -> None:
        """After the agent: give back what a run with no DecimalAI handler leaves on the router."""
        if not run_key:
            return
        from .langchain import _call_handler

        handler, _parent = _call_handler()
        if handler is not None:
            return  # its trace build drains them
        with _late_lock:
            _late.pop(run_key, None)
        router = self._router()
        if router is None:
            return
        from .skill_router import _release_scoped_routing_rail

        _release_scoped_routing_rail(router, run_key)
        for drain in ("consume_loaded_names", "consume_loaded_hashes"):
            try:
                getattr(router, drain)(scope=run_key)
            except Exception:
                self._log.debug("could not release a run's load rail", exc_info=True)

    # ── a model call ────────────────────────────────────────
    def _call_record(self, request: Any) -> Tuple[Optional[str], Optional[ConversationRecord]]:
        """The run and thread record a model call belongs to; composed here if no hook ran."""
        state = getattr(request, "state", None) or {}
        run_key = _run_key_of(state)
        record = ConversationRecord.from_state(
            state.get(CONVERSATION_KEY) if isinstance(state, Mapping) else None, self.name)
        if record is not None and run_key is not None:
            return run_key, record
        # A graph without this middleware's before_agent (or a state it never saw): this call
        # is a run of its own, with nothing kept for the thread.
        run_key = run_key or f"run_{uuid4().hex}"
        runtime = getattr(request, "runtime", None)
        context: Dict[str, Any] = {"messages": getattr(request, "messages", None),
                                   "values": getattr(runtime, "context", None), "conversation": None}
        self._start_sync(context, run_key)
        prepared = self._run_state(run_key) or {}
        return run_key, prepared.get("conversation")

    def _model_request(self, request: Any) -> Tuple[Any, list]:
        """The request with DecimalAI's system text and tool, and the rails the call reads."""
        run_key, record = self._call_record(request)
        if record is None:
            return request, []
        _warn_native_skills(self, getattr(request, "state", None))
        overrides: Dict[str, Any] = {"tools": self._tools_for(list(getattr(request, "tools", None) or []), record)}
        if record.text:
            mine = getattr(request, "system_message", None)
            if mine is not None and not _is_ours(mine):
                self._warn_once(None, (
                    "create_agent(system_prompt=...) is replaced by the agent's prompt from "
                    f"agent_config({self.name!r}); pass your code's prompt as agent_config(..., "
                    "fallback=...) instead."))
            overrides["system_message"] = SystemMessage(
                content=record.text, additional_kwargs={"decimalai_injected": True})
        return request.override(**overrides), self._open_rails(run_key, record)

    def _tools_for(self, tools: List[Any], record: ConversationRecord) -> List[Any]:
        """The call's tools with ``load_skill`` as the thread defined it (or without it)."""
        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION, LOAD_SKILL_TOOL_NAME

        ours = self.tools[0]
        others = [t for t in tools if t is not ours]
        if any(getattr(t, "name", None) == LOAD_SKILL_TOOL_NAME
               or (isinstance(t, Mapping) and (t.get("function") or {}).get("name") == LOAD_SKILL_TOOL_NAME)
               for t in others):
            self._warn_once(None, (
                "this agent already has a tool named load_skill, so DecimalAI's is left out and "
                "its skill list cannot be loaded from. Rename that tool to use both."))
            return others
        if not record.tool:
            return others
        # A dict, built from the thread's record: the same definition on every call.
        return [*others, {"type": "function", "function": {
            "name": LOAD_SKILL_TOOL_NAME, "description": LOAD_SKILL_TOOL_DESCRIPTION,
            "parameters": self._tool_schema(list(record.tool_names))}}]

    def _open_rails(self, run_key: Optional[str], record: Optional[ConversationRecord]) -> list:
        """Hand the run's record to the CallbackHandler for the model call starting now."""
        from . import langchain as lc

        with self._lock:
            fields = dict(self._records.get(run_key) or {}) if run_key is not None else {}
        offered = fields.get("offered")
        if offered is None:
            offered = list(record.names) if record is not None else []
        routing_id = fields.get("routing_id")
        prompt = fields.get("prompt") or (record.prompt if record is not None else None)
        rails = [
            (lc._call_scope_ctx, lc._call_scope_ctx.set(run_key)),
            (lc._routing_id_ctx, lc._routing_id_ctx.set(routing_id)),
            (lc._skills_offered_ctx, lc._skills_offered_ctx.set(set(offered))),
            (lc._config_agent_ctx, lc._config_agent_ctx.set(
                {"agent": self.name, "prompt": prompt, "template": self._template()})),
        ]
        delivered = list(fields.get("delivered") or [])
        if delivered:
            # Skill text pasted into the system text (the load_skill tool switched off).
            rails.append((lc._skills_delivered_ctx, lc._skills_delivered_ctx.set(set(delivered))))
            rails.append((lc._skills_delivered_versions_ctx, lc._skills_delivered_versions_ctx.set(
                list(fields.get("delivered_versions") or []))))
        return rails

    # ── load_skill ──────────────────────────────────────────
    def _load_target(self, name: str, state: Any) -> Tuple[Optional[str], Optional[str]]:
        """(the run a load belongs to, the routing decision it names) for a cut-list thread."""
        run_key = _run_key_of(state)
        record = ConversationRecord.from_state(
            state.get(CONVERSATION_KEY) if isinstance(state, Mapping) else None, self.name)
        if record is None or not record.cut:
            return run_key, None
        key = (name or "").strip().lower()
        prepared = self._run_state(run_key) or {}
        if any(key == offered.lower() for offered in prepared.get("offered_now") or ()):
            return run_key, prepared.get("routing_now") or None
        origin = next((rid for shown, rid in record.shown.items() if shown.lower() == key), None)
        return run_key, origin or None

    def _build_tool(self) -> Any:
        from langchain_core.tools import StructuredTool

        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION, LOAD_SKILL_TOOL_NAME

        config = self

        def load_skill(name: str, state: Annotated[dict, InjectedState]) -> str:
            run_key, decision = config._load_target(name, state)
            token, named = _RUN.set(run_key), _load_routing_id_ctx.set(decision)
            try:
                return config._load_sync(str(name or "").strip(), config._run_state(run_key))
            finally:
                _load_routing_id_ctx.reset(named)
                _RUN.reset(token)

        async def aload_skill(name: str, state: Annotated[dict, InjectedState]) -> str:
            run_key, decision = config._load_target(name, state)
            token, named = _RUN.set(run_key), _load_routing_id_ctx.set(decision)
            try:
                return await config._load(str(name or "").strip(), config._run_state(run_key))
            finally:
                _load_routing_id_ctx.reset(named)
                _RUN.reset(token)

        return StructuredTool.from_function(
            func=load_skill, coroutine=aload_skill, name=LOAD_SKILL_TOOL_NAME,
            description=LOAD_SKILL_TOOL_DESCRIPTION,
        )

    # ── a hand-built graph ──────────────────────────────────
    def system_message(self, messages: Any, context: Any = None) -> SystemMessage:
        """The run's system message, for a hand-built LangGraph graph's model node.

        Routes the latest user message (the first call of a run; later calls of the same run
        reuse it) and composes the saved prompt, filled from ``context``, then the skill list.
        Call it in the node that calls the model, so the call's trace gets the run's record.
        A hand-built graph keeps no thread record: each run composes afresh, so on Claude with
        thinking keep a thread's first system message in your graph's state and reuse it.
        """
        run_key = _run_key_of({"messages": messages}) or f"run_{uuid4().hex}"
        state = self._run_state(run_key)
        if state is None or state.get("conversation") is None:
            self._start_sync({"messages": messages, "values": context, "conversation": None}, run_key)
            state = self._run_state(run_key) or {}
        return self._handed_over(run_key, state.get("conversation"))

    async def asystem_message(self, messages: Any, context: Any = None) -> SystemMessage:
        """:meth:`system_message` on an event loop."""
        run_key = _run_key_of({"messages": messages}) or f"run_{uuid4().hex}"
        state = self._run_state(run_key)
        if state is None or state.get("conversation") is None:
            await self._start({"messages": messages, "values": context, "conversation": None}, run_key)
            state = self._run_state(run_key) or {}
        return self._handed_over(run_key, state.get("conversation"))

    def _handed_over(self, run_key: str, record: Optional[ConversationRecord]) -> SystemMessage:
        # The rails stay set: a node's context ends with the node, after its model call.
        self._open_rails(run_key, record)
        text = record.text if record is not None else ""
        # No thread state to carry a message in: the always-include skills' text ends the system text.
        always = (self._run_state(run_key) or {}).get("always_now") or []
        if always:
            block = "\n\n".join(section for _name, section in always)
            text = f"{text}\n\n{block}" if text else block
        return SystemMessage(content=text, additional_kwargs={"decimalai_injected": True})


def _context(state: Any, runtime: Any) -> Dict[str, Any]:
    state = state if isinstance(state, Mapping) else {}
    return {"messages": state.get("messages"), "values": getattr(runtime, "context", None),
            "conversation": state.get(CONVERSATION_KEY)}


def _close(rails: list) -> None:
    for var, token in reversed(rails):
        var.reset(token)


class DecimalAIState(AgentState):
    """The agent state, plus the thread's DecimalAI record (private: not in the input or output)."""

    decimalai_run: NotRequired[Annotated[str, PrivateStateAttr]]
    decimalai_conversation: NotRequired[Annotated[Dict[str, Any], PrivateStateAttr]]


def _build_middleware(config: LangChainAgentConfig) -> Any:
    class DecimalAIMiddleware(AgentMiddleware):
        """DecimalAI's system text and ``load_skill`` for every model call of one agent."""

        state_schema = DecimalAIState

        def __init__(self) -> None:
            super().__init__()
            self.tools = list(config.tools)

        @property
        def name(self) -> str:
            # LangChain refuses ':' and '|' in a middleware name.
            return "DecimalAI_" + re.sub(r"[^A-Za-z0-9_.-]", "_", config.name)

        def before_agent(self, state: Any, runtime: Any) -> Dict[str, Any]:
            return config._start_sync(_context(state, runtime), f"run_{uuid4().hex}")

        async def abefore_agent(self, state: Any, runtime: Any) -> Dict[str, Any]:
            _warn_async_before_311(config)
            return await config._start(_context(state, runtime), f"run_{uuid4().hex}")

        def wrap_model_call(self, request: Any, handler: Any) -> Any:
            prepared, rails = config._model_request(request)
            try:
                return handler(prepared)
            finally:
                _close(rails)

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            prepared, rails = config._model_request(request)
            try:
                return await handler(prepared)
            finally:
                _close(rails)

        def after_agent(self, state: Any, runtime: Any) -> None:
            config._end_run(_run_key_of(state))
            return None

        async def aafter_agent(self, state: Any, runtime: Any) -> None:
            config._end_run(_run_key_of(state))
            return None

    return DecimalAIMiddleware()


def _warn_native_skills(config: LangChainAgentConfig, state: Any) -> None:
    """Deep Agents' own skills beside DecimalAI's: the model is shown two lists.

    Checked at each model call, when every ``before_agent`` hook has run (Deep Agents' writes
    ``skills_metadata``), so the order of the middleware list does not matter."""
    if isinstance(state, Mapping) and state.get("skills_metadata") is not None:
        config._warn_once(None, (
            "this agent also loads skills with Deep Agents' SkillsMiddleware, so the model is "
            "shown two skill lists. Use one: agent_config()'s middleware, or Deep Agents' skills "
            "with the files `decimalai skills add` installs."))
