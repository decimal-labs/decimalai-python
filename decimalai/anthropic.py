"""Anthropic Messages API: an agent's DecimalAI prompt and skills, one run at a time.

::

    import anthropic
    from decimalai.anthropic import agent_config

    client = anthropic.Anthropic()
    support = agent_config("support", fallback=BASE_SYSTEM)

    with support.run(question, variables=customer) as run:
        runner = client.beta.messages.tool_runner(
            model="claude-haiku-5-5", max_tokens=4096,
            system=run.system,                          # the saved prompt, then the skill list
            tools=[lookup_order, *run.tools],           # your tools, then load_skill
            messages=[{"role": "user", "content": question}],
        )
        answer = runner.until_done()

A hand-written loop around ``client.messages.create`` takes ``run.tool_params``
(the same tool as a plain definition) and ``run.tool_result(block)``, which
answers a ``load_skill`` call and returns ``None`` for any other tool.

One run is one user message, from the question to the final answer. Opening it
routes the message, fills the prompt's ``{{variables}}`` from ``variables`` and
starts the run's trace, filed under the DecimalAI agent; closing it sends the
trace with the run's record (prompt version, skills offered and loaded, routing
id). ``support.run()`` itself calls no model. Make the model calls inside the
``with`` block: a Tool Runner is lazy, so consume it there. ``async with`` routes
off the event loop and gives async tools for the async Tool Runner.

A chat keeps one conversation, whose ``system`` and tools stay the same for
every message::

    chat = support.conversation(variables=customer)        # or conversation(state=saved)
    with chat.run(question) as run:
        ...
    saved = chat.state                                   # store it beside the chat's history

Anthropic rejects a conversation whose system prompt or tool definitions change
once a thinking block is in its history, so a newer saved prompt or skill list
reaches the next conversation, not an open one. ``chat.state`` lets a server that
keeps no chat in memory resume the conversation exactly.

Nothing DecimalAI does changes the SDK's behaviour, and other clients get no
skills. ``agent_config()`` turns on Anthropic tracing (OpenInference's
instrumentor, which wraps the SDK's classes for the process; installed by
``decimalai[anthropic]``) on the same pipeline as ``decimalai.init(anthropic=True)``,
the global tracer provider (an app with its own sets it up first), and sends only its
runs: no Claude call made outside one and none of the app's own spans, unless
``init(anthropic=True)`` asked for every Claude call (``init(otel=True)``: the app's spans).

Anthropic's own Agent Skills (``container={"skills": [...]}``) are the other way
to give an agent skills; an agent that uses both shows the model two lists.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from contextvars import ContextVar
from typing import Any, Dict, List, NoReturn, Optional, Tuple
from uuid import uuid4

from ._agent_binding import AgentBinding, ConversationRecord, _in_thread
from ._agent_binding import new_skills_block as _new_skills_block

# A closed run's span waits for its routing decision there, shared with decimalai.openai's
# runs, so one decimalai.flush() drains both. The names stay importable from here.
from ._provider_run import (  # noqa: F401 - re-exported under their old names
    _drain_finishing,
    _enclosing_trace_id,
    _event_loop_running,
    _finish_later,
    _is_error,
    open_run_span,
)
from .skill_router import _load_routing_id_ctx

logger = logging.getLogger("decimalai.anthropic")

#: The run whose ``with`` block is open in this context. The background routing
#: and a load's scope read it; a tool sets it around its own call.
_RUN: "ContextVar[Optional[str]]" = ContextVar("decimalai_anthropic_run", default=None)

#: ``agent_config(cache_control=...)``'s default: one breakpoint on the system text.
_DEFAULT_CACHE_CONTROL: Dict[str, Any] = {"type": "ephemeral"}

_UNSET: Any = object()

#: The SkillRouter behind every config this module builds. `decimalai.init()`
#: drops it when a re-init changes the key or workspace.
_skill_router_singleton: Any = None

_tracing_lock = threading.Lock()
_tracing_turned_on = False


def _get_skill_router() -> Any:
    """The SkillRouter for this module's configs, built on first use; None when switched off."""
    global _skill_router_singleton
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
            api_key=config.api_key, base_url=config.base_url,
            # Every routing call names its agent; no router default.
            agent_name=None,
            # Skill text reaches the model through load_skill, not the prompt.
            inject_body=False,
        )
        return _skill_router_singleton
    except Exception:
        logger.debug("SkillRouter init failed", exc_info=True)
        return None


def _require_anthropic() -> None:
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "decimalai.anthropic.agent_config() needs the Anthropic SDK: "
            'pip install "decimalai[anthropic]"'
        ) from exc


def _turn_on_tracing() -> None:
    """Anthropic tracing for DecimalAI runs, once per process."""
    global _tracing_turned_on
    from ._config import _disabled_by_init

    if _disabled_by_init():
        return
    # Held while tracing is set up: a second first agent_config() on another thread
    # waits for it, rather than opening a run before there is anything to trace it.
    with _tracing_lock:
        if _tracing_turned_on:
            return
        try:
            from .providers import instrument_for_runs

            instrument_for_runs("anthropic")
        except Exception:
            logger.warning("DecimalAI could not turn on Anthropic tracing (continuing)", exc_info=True)
        _tracing_turned_on = True


def _tool_use_parts(block: Any) -> Tuple[Any, Any, Any]:
    """(name, input, id) of a ``tool_use`` block, as an SDK object or a plain dict."""
    if isinstance(block, dict):
        return block.get("name"), block.get("input"), block.get("id")
    return getattr(block, "name", None), getattr(block, "input", None), getattr(block, "id", None)


# ── a conversation ───────────────────────────────────────────


class Conversation:
    """Messages that share one ``system`` and one ``load_skill`` definition.

    Made by :meth:`AnthropicAgentConfig.conversation`. Its first run composes the
    system text (the prompt filled from ``variables``, then the skill list) and the
    tool; every later run reuses both, byte for byte, and is still routed on its
    own message. Keep one per chat, beside the chat's message history; a server
    that keeps no chat in memory stores :attr:`state` with the history and
    resumes with ``support.conversation(state=...)``.

    What it keeps is the ``ConversationRecord`` every DecimalAI binding shares for
    a chat (``decimalai._agent_binding``); this module renders it as Anthropic
    system blocks and a ``load_skill`` definition.

    Everything in ``system`` and ``tools`` has to stay the same for the whole
    conversation, your own parts included: per-message facts belong in the user
    message.

    Attributes:
        id: This conversation's id; its runs' traces carry it as ``session_id``.
    """

    def __init__(self, config: "AnthropicAgentConfig", variables: Any,
                 record: Optional[ConversationRecord] = None,
                 conversation_id: Optional[str] = None) -> None:
        self._config = config
        self._values = variables
        self._lock = threading.Lock()
        self._record: Optional[ConversationRecord] = record
        self.id: str = conversation_id or uuid4().hex
        self._runs = 0

    def __repr__(self) -> str:
        return f"Conversation(agent={self._config.name!r}, id={self.id!r}, runs={self._runs})"

    def run(self, message: Optional[str]) -> "Run":
        """One user message of this conversation, from the question to the final answer.

        Use it as a context manager (``with`` or ``async with``); see :class:`Run`.
        """
        return Run(self._config, self, message)

    @property
    def state(self) -> Optional[Dict[str, Any]]:
        """What the conversation keeps, as JSON-ready data; None before its first run.

        The system text (variables already filled) with its skill list, the
        ``load_skill`` definition, the prompt version, the routing decision the
        list came from and the one that first showed each skill. Pass it to
        ``support.conversation(state=...)`` to continue the conversation in
        another process.
        """
        record = self._current()
        if record is None:
            return None
        with self._lock:
            return {**record.to_state(self._config.name), "id": self.id}

    def _fix(self, record: ConversationRecord) -> ConversationRecord:
        """Keep ``record`` unless another run fixed the conversation first."""
        with self._lock:
            if self._record is None:
                self._record = record
            return self._record

    def _current(self) -> Optional[ConversationRecord]:
        with self._lock:
            return self._record


class Run:
    """One user message, from the question to the final answer.

    Open it with ``with`` (a sync client and the sync Tool Runner) or ``async
    with`` (``AsyncAnthropic`` and the async Tool Runner). Opening routes the
    message and starts the run's trace; closing sends it. The model calls for
    this message belong inside the block.

    Attributes:
        system: The value for ``system=``: text blocks holding the agent's prompt,
            then its skill list and one sentence naming ``load_skill``. Add your
            own blocks after it (``[*run.system, {...}]``), never format it into a
            string.
        tools: ``load_skill`` for ``client.beta.messages.tool_runner(tools=[...])``,
            async after ``async with``; empty when the run has no skill list.
        tool_params: ``load_skill`` as a plain definition, for ``client.messages.create``.
    """

    def __init__(self, config: "AnthropicAgentConfig", conversation: Conversation,
                 message: Optional[str]) -> None:
        self._config = config
        self._conversation = conversation
        self._message = message
        self.key = f"run_{uuid4().hex}"
        self._lock = threading.Lock()
        self._mode: Optional[str] = None
        self._exited = False
        self._ended = False
        self._exit_ns: Optional[int] = None
        self._record: Optional[ConversationRecord] = None
        self._recorded_by_me = False
        self._blocks: List[Dict[str, Any]] = []
        self._tool_def: Optional[Dict[str, Any]] = None
        self._state: Optional[Dict[str, Any]] = None
        self._span: Any = None
        self._otel_ctx: Any = None
        self._tokens: List[Tuple[Any, Any]] = []
        self._read: set = set()
        self._tools: Optional[List[Any]] = None

    def __repr__(self) -> str:
        return f"Run(agent={self._config.name!r}, open={self._mode is not None and not self._exited})"

    # ── opening and closing ─────────────────────────────────
    def __enter__(self) -> "Run":
        if _event_loop_running():
            self._config._warn_once(None, (
                "a run opened with `with` inside a running event loop routes on that loop "
                "and blocks it; use `async with support.run(...)` with AsyncAnthropic."))
        self._open("sync")
        try:
            self._prepare_sync()
        except BaseException as exc:
            self._exit(type(exc), exc)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._exit(exc_type, exc)

    async def __aenter__(self) -> "Run":
        self._open("async")
        try:
            await self._prepare_async()
        except BaseException as exc:
            self._exit(type(exc), exc)
            raise
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._exit(exc_type, exc)

    def _open(self, mode: str) -> None:
        if self._mode is not None:
            raise RuntimeError("a DecimalAI run is opened once; start another with support.run(...)")
        self._mode = mode
        self._start_span()
        self._tokens.append((_RUN, _RUN.set(self.key)))
        self._config._open(self)
        try:
            trace_id = self._span.get_span_context().trace_id if self._span is not None else 0
            parent = _enclosing_trace_id()
            if trace_id and (parent or self._config._on_trace is not None):
                from .otel import record_run_link

                record_run_link(trace_id, parent_trace_id=parent, on_trace=self._config._on_trace)
        except Exception:
            self._config._log.debug("could not link the run's trace", exc_info=True)

    def _start_span(self) -> None:
        """The run's span: a new trace of its own, on the provider Anthropic calls are traced on."""
        self._span, self._otel_ctx, tokens = open_run_span(
            "anthropic", self._config.name, self._conversation.id)
        self._tokens.extend(tokens)

    def _exit(self, exc_type: Any, exc: Any) -> None:
        with self._lock:
            if self._exited:
                return
            self._exited = True
            self._exit_ns = time.time_ns()
        if exc_type is None:
            self._warn_at_exit()
        self._record_this_messages_decision()
        for holder, token in reversed(self._tokens):
            try:
                if hasattr(holder, "detach"):
                    holder.detach(token)
                else:
                    holder.reset(token)
            except Exception:
                # Closed in another context than it was opened in (a generator
                # driven step by step); the values die with that context.
                pass
        self._tokens.clear()
        if self._span is not None and exc is not None:
            try:
                from opentelemetry.trace import Status, StatusCode

                self._span.record_exception(exc)
                self._span.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
        pending = (self._state or {}).get("pending")
        if self._span is not None and pending is not None and not pending.done():
            _finish_later(self, pending, self._exit_ns)
        else:
            self._end(self._exit_ns)

    def _record_this_messages_decision(self) -> None:
        """A later message of a cut-list conversation: its own decision goes on its trace
        when every skill it offered has been shown to the model."""
        state, record = self._state or {}, self._record
        routing_id = state.get("routing_now")
        if record is None or routing_id is None:
            return
        new = state.get("new_names") or []
        if new and "message" not in self._read:
            self._config._warn_once(None, (
                "this message's routed skills were not shown to the model: send run.message as the "
                "user turn (it carries the skills new to the conversation)."))
            return
        try:
            from .otel import record_skill_rail

            record_skill_rail(routing_id=routing_id, offered=list(state.get("offered_now") or []))
        except Exception:
            self._config._log.debug("could not record the message's decision", exc_info=True)
        with self._conversation._lock:
            for name in new:
                record.shown.setdefault(name, routing_id)

    def _end(self, end_ns: int) -> None:
        with self._lock:
            if self._ended:
                return
            self._ended = True
        if self._span is not None:
            try:
                self._span.end(end_time=end_ns)
            except Exception:
                self._config._log.debug("could not end the run's span", exc_info=True)
        self._config._close(self)

    def _warn_at_exit(self) -> None:
        config = self._config
        if self._tool_def is not None and "system" in self._read \
                and not self._read & {"tools", "tool_params", "tool_result"}:
            config._warn_once(None, (
                "run.system names the load_skill tool, but this run's tools never included it. "
                "Add *run.tools (Tool Runner) or *run.tool_params (messages.create) to tools=."))
        if self._span is None or not self._read & {"system", "tools", "tool_params"}:
            return
        try:
            from . import otel, providers

            if not providers._traced_for_runs("anthropic"):
                return
            started = otel.spans_started(self._span.get_span_context().trace_id)
            if started is not None and started <= 1:  # None: evicted by newer traces, unknown
                advice = (
                    "no Claude request was traced inside this run. Make the model calls inside "
                    "the `with` block (a Tool Runner is lazy: consume it there). On AnthropicVertex "
                    "and the legacy AnthropicBedrock client, the Tool Runner's requests are not "
                    "traced; a hand-written messages.create loop is.")
                if "anthropic" in providers._runs_via_another_tool:
                    advice += (" Anthropic tracing was turned on outside DecimalAI: DecimalAI "
                               "receives those calls only if that tracing sends its spans to the "
                               "global tracer provider.")
                config._warn_once(None, advice)
        except Exception:
            pass

    # ── preparing ───────────────────────────────────────────
    def _context(self) -> Dict[str, Any]:
        return {"message": self._message, "values": self._conversation._values,
                "conversation": self._conversation}

    def _prepare_sync(self) -> None:
        config, ctx = self._config, self._context()
        record = self._conversation._current()
        if record is not None and config._routes_later(record):
            ctx["routed"] = config._route_later_run(self.key, ctx)
        state = config._prepare_run_sync(ctx, None, self.key)
        if state.get("conversation") is None:
            config._compose(ctx, None, self.key, state)
        self._done_preparing(state)

    async def _prepare_async(self) -> None:
        config, ctx = self._config, self._context()
        record = self._conversation._current()
        if record is not None and config._routes_later(record):
            ctx["routed"] = await _in_thread(config._route_later_run, self.key, ctx)
        state = await config._prepare_run(ctx, None, self.key)
        if state.get("conversation") is None:
            config._compose(ctx, None, self.key, state)
        self._done_preparing(state)

    def _done_preparing(self, state: Dict[str, Any]) -> None:
        mine = state["conversation"]
        record = self._conversation._current() or mine
        with self._conversation._lock:
            first = self._conversation._runs == 0
            self._conversation._runs += 1
        self._recorded_by_me = first and record is mine
        self._record, self._state = record, state
        self._blocks, self._tool_def = self._config._shape(record)
        routed = state.get("routed")
        if routed is not None and routed[2]:
            # A later message of a cut-list conversation, routed on its own text: the
            # skills it picked that the conversation has not shown go in its user turn.
            offered, routing_id = list(routed[1]), routed[2]
            with self._conversation._lock:
                new = [name for name in offered if name not in record.shown]
            state["offered_now"], state["routing_now"], state["new_names"] = offered, routing_id, new
            state["new_block"] = _new_skills_block(routed[0], new) if new else None

    # ── what the developer reads ────────────────────────────
    def _require_open(self, what: str) -> ConversationRecord:
        if self._record is None:
            raise RuntimeError(
                f"run.{what} is ready once the run is open: use it inside `with support.run(...) as run:`")
        return self._record

    @property
    def system(self) -> List[Dict[str, Any]]:
        self._require_open("system")
        self._read.add("system")
        return copy.deepcopy(self._blocks)

    @property
    def message(self) -> Dict[str, Any]:
        """This run's user turn, for ``messages=``: the question, plus any skills new to the conversation.

        On a later message of a conversation whose skill list is cut to fit, the
        skills routed for this message that the conversation has not shown yet
        follow the question as a second text block. They go in the user turn, not
        in ``system``: a conversation's history only grows. Append it to the history
        instead of your own ``{"role": "user", ...}``.
        """
        self._require_open("message")
        self._read.add("message")
        extra = (self._state or {}).get("new_block")
        text = self._message if isinstance(self._message, str) else ""
        if not extra:
            return {"role": "user", "content": text}
        return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "text", "text": extra}]}

    @property
    def tool_params(self) -> List[Dict[str, Any]]:
        self._require_open("tool_params")
        self._read.add("tool_params")
        return [copy.deepcopy(self._tool_def)] if self._tool_def is not None else []

    @property
    def tools(self) -> List[Any]:
        self._require_open("tools")
        self._read.add("tools")
        if self._tool_def is None:
            return []
        if self._tools is None:
            sync_cls, async_cls = _tool_classes()
            self._tools = [(async_cls if self._mode == "async" else sync_cls)(self)]
        return list(self._tools)

    def tool_result(self, block: Any) -> Optional[Dict[str, Any]]:
        """The ``tool_result`` answering a ``load_skill`` ``tool_use`` block; ``None`` for any other tool.

        In an ``async with`` run, ``await run.atool_result(block)`` keeps the load
        off the event loop.
        """
        self._require_open("tool_result")
        self._read.add("tool_result")
        name, tool_input, tool_use_id = _tool_use_parts(block)
        if self._tool_def is None or name != self._tool_def["name"]:
            return None
        if self._mode == "async" and _event_loop_running():
            self._config._warn_once(None, (
                "run.tool_result() loads on the event loop; in an `async with` run use "
                "`await run.atool_result(block)`."))
        text, error = self._serve(tool_input)
        return _result_block(tool_use_id, text, error)

    async def atool_result(self, block: Any) -> Optional[Dict[str, Any]]:
        """`tool_result` for an async loop: the load runs off the event loop."""
        self._require_open("atool_result")
        self._read.add("tool_result")
        name, tool_input, tool_use_id = _tool_use_parts(block)
        if self._tool_def is None or name != self._tool_def["name"]:
            return None
        text, error = await self._aserve(tool_input)
        return _result_block(tool_use_id, text, error)

    # ── serving load_skill ──────────────────────────────────
    def _requested(self, tool_input: Any) -> Tuple[Optional[str], Optional[str]]:
        """(the skill to load, or None and a sentence for the model)."""
        from ._config import _disabled_by_init

        if self._exited:
            self._config._warn_once(None, (
                "load_skill was called after its run closed: the skill is served, but the run's "
                "record will not show it. Consume the Tool Runner inside the `with` block."))
        if _disabled_by_init() or not (self._state or {}).get("tool"):
            return None, "DecimalAI skills are switched off for this agent. Answer without them."
        raw = tool_input.get("name") if isinstance(tool_input, dict) else None
        name = str(raw or "").strip()
        if not name:
            return None, "load_skill needs the name of one skill from the list."
        record = self._record
        if record is not None and not record.cut:
            # The SDK does not hold a tool's input to its schema, so the name is
            # checked here. A cut list names only some skills; the platform checks those.
            match = next((n for n in record.names if n.lower() == name.lower()), None)
            if match is None:
                return None, (f"{name!r} is not one of this agent's skills. Load one of: "
                              f"{', '.join(record.names)}; or answer without one.")
            name = match
        return name, None

    def _load_context(self) -> Any:
        """The context a load records in: this run's while it is open, none after."""
        try:
            from opentelemetry import context as otel_context
        except ImportError:
            return None, None
        ctx = self._otel_ctx if (self._otel_ctx is not None and not self._exited) else otel_context.Context()
        return otel_context, ctx

    def _decision_for(self, name: str) -> Optional[str]:
        """The routing decision a load of ``name`` names, in a cut-list conversation.

        The platform serves a public-registry skill only to a load whose decision
        offered it. A skill this message's decision offered names that one; a skill
        an earlier message showed names the decision that showed it. None leaves
        the load to the run's latest decision (a whole list has one per run).
        """
        state, record = self._state or {}, self._record
        if record is None or not record.cut:
            return None
        key = name.lower()
        if any(key == offered.lower() for offered in state.get("offered_now") or ()):
            return state.get("routing_now") or None
        with self._conversation._lock:
            origin = next((rid for shown, rid in record.shown.items() if shown.lower() == key), None)
        return origin or None

    def _serve(self, tool_input: Any) -> Tuple[str, bool]:
        name, answer = self._requested(tool_input)
        if name is None:
            return answer or "", True
        otel_context, ctx = self._load_context()
        otel_token = otel_context.attach(ctx) if otel_context is not None else None
        token = _RUN.set(self.key)
        decision = _load_routing_id_ctx.set(self._decision_for(name))
        try:
            text = self._config._load_sync(name, self._state)
        finally:
            _load_routing_id_ctx.reset(decision)
            _RUN.reset(token)
            if otel_token is not None:
                otel_context.detach(otel_token)
        return text, _is_error(text)

    async def _aserve(self, tool_input: Any) -> Tuple[str, bool]:
        name, answer = self._requested(tool_input)
        if name is None:
            return answer or "", True
        otel_context, ctx = self._load_context()
        otel_token = otel_context.attach(ctx) if otel_context is not None else None
        token = _RUN.set(self.key)
        decision = _load_routing_id_ctx.set(self._decision_for(name))
        try:
            text = await self._config._load(name, self._state)
        finally:
            _load_routing_id_ctx.reset(decision)
            _RUN.reset(token)
            if otel_token is not None:
                otel_context.detach(otel_token)
        return text, _is_error(text)


def _result_block(tool_use_id: Any, text: str, error: bool) -> Dict[str, Any]:
    block: Dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
    if error:
        block["is_error"] = True
    return block


_tool_classes_cache: Optional[tuple] = None
_tool_classes_lock = threading.Lock()


def _tool_classes() -> tuple:
    """The sync and async ``load_skill`` tool classes, built on first use.

    Built on the Anthropic SDK's public ``BetaBuiltinFunctionTool`` bases, whose
    ``call`` gets the model's input as given, so this module imports without the SDK.
    """
    global _tool_classes_cache
    with _tool_classes_lock:
        if _tool_classes_cache is None:
            from anthropic.lib.tools import (
                BetaAsyncBuiltinFunctionTool,
                BetaBuiltinFunctionTool,
                ToolError,
            )

            class LoadSkillTool(BetaBuiltinFunctionTool):
                """DecimalAI's ``load_skill`` for one run, for the sync Tool Runner."""

                def __init__(self, run: Run) -> None:
                    self._run = run

                def to_dict(self) -> Any:
                    return copy.deepcopy(self._run._tool_def)

                def call(self, input: object) -> str:
                    text, error = self._run._serve(input)
                    if error:
                        raise ToolError(text)
                    return text

                def __repr__(self) -> str:
                    return f"LoadSkillTool(agent={self._run._config.name!r})"

            class AsyncLoadSkillTool(BetaAsyncBuiltinFunctionTool):
                """DecimalAI's ``load_skill`` for one run, for the async Tool Runner."""

                def __init__(self, run: Run) -> None:
                    self._run = run

                def to_dict(self) -> Any:
                    return copy.deepcopy(self._run._tool_def)

                async def call(self, input: object) -> str:
                    text, error = await self._run._aserve(input)
                    if error:
                        raise ToolError(text)
                    return text

                def __repr__(self) -> str:
                    return f"AsyncLoadSkillTool(agent={self._run._config.name!r})"

            LoadSkillTool.__module__ = AsyncLoadSkillTool.__module__ = __name__
            _tool_classes_cache = (LoadSkillTool, AsyncLoadSkillTool)
        return _tool_classes_cache


# ── the config ───────────────────────────────────────────────


class AnthropicAgentConfig(AgentBinding):
    """One agent's prompt and skills in DecimalAI, for the Anthropic Messages API.

    Made by :func:`agent_config`. Open one run per user message::

        with support.run(question, variables=customer) as run:
            ...  # system=run.system, tools=[*your_tools, *run.tools]

    or keep one :meth:`conversation` per chat.

    Attributes:
        name: The agent's canonical DecimalAI name.
    """

    _log = logger
    _VALUES_HINT = "Pass them as support.run(..., variables=...), as attributes or dict keys."

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str,
                 cache_control: Optional[Dict[str, Any]], on_trace: Any) -> None:
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        self._cache_control = copy.deepcopy(cache_control) if cache_control else None
        self._on_trace = on_trace
        self._open_runs: Dict[str, Run] = {}

    def __repr__(self) -> str:
        return (
            f"AnthropicAgentConfig(name={self.name!r}, prompt_version={self.prompt_version!r}, "
            f"skills={self.skill_names!r})"
        )

    # ── the pieces ───────────────────────────────────────────
    @property
    def prompt(self) -> Optional[str]:
        """The saved prompt as written (variables unfilled), else the fallback."""
        with self._lock:
            text = getattr(self._prompt, "system_prompt", None)
        return text if text is not None else self._fallback

    def run(self, message: Optional[str], *, variables: Any = None) -> Run:
        """One user message with no earlier turns, from the question to the final answer.

        ``support.run()`` calls no model: use it as a context manager, and inside it
        ``run.system``, ``run.tools``, ``run.tool_params`` and ``run.tool_result()``
        are this run's. For a message in a chat, use :meth:`conversation`, which
        keeps ``system`` the same for every message.

        Args:
            message: The user's message, which the skills are routed on.
            variables: Values for the prompt's ``{{variables}}``, as attributes or
                dict keys (``{{customer.name}}`` reads ``variables.customer.name`` or
                ``variables["customer"]["name"]``).

        Raises (when the run opens):
            MissingPromptVariable: the prompt needs a value ``variables`` lacks.
            RuntimeError: ``on_unavailable="fail"`` and DecimalAI never returned
                this agent's skills.
        """
        return Conversation(self, variables).run(message)

    def conversation(self, *, variables: Any = None, state: Optional[Dict[str, Any]] = None) -> Conversation:
        """A chat of several user messages that share one ``system`` and one tool.

        ``variables`` fills the prompt's ``{{variables}}`` once, for the whole
        conversation. A prompt saved, a skill assigned or a value changed later
        reaches the next conversation. ``state`` resumes a conversation from its
        :attr:`Conversation.state`, in this process or another.

        Raises:
            ValueError: ``state`` belongs to another agent or another version of
                this format, or both ``state`` and ``variables`` were given.
        """
        if state is not None:
            if variables is not None:
                raise ValueError("a resumed conversation keeps the values it started with; "
                                 "pass state= or variables=, not both")
            if isinstance(state, dict) and state.get("agent") != self.name:
                raise ValueError(f"this conversation state belongs to agent {state.get('agent')!r}, "
                                 f"not to {self.name!r}")
            record = ConversationRecord.from_state(state, self.name)
            if record is None:
                raise ValueError("state= takes a Conversation.state from this version of decimalai")
            return Conversation(self, None, record, str(state.get("id") or "") or None)
        return Conversation(self, variables)

    # ── open runs ───────────────────────────────────────────
    def _open(self, run: Run) -> None:
        with self._lock:
            self._open_runs[run.key] = run

    def _close(self, run: Run) -> None:
        with self._lock:
            self._open_runs.pop(run.key, None)

    def _shape(self, record: ConversationRecord) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """A conversation's record as Anthropic sees it: system blocks and the tool.

        The same record always gives the same blocks and the same definition, so a
        conversation resumed from its state in another process sends what its first
        run sent.
        """
        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION, LOAD_SKILL_TOOL_NAME

        text, menu = record.text, record.menu
        head = text[: len(text) - len(menu)].rstrip("\n") if menu else text
        blocks = self._blocks(head, menu)
        if blocks and self._cache_control:
            # At the end of the system text: a run's model calls, and a conversation's
            # every turn, send the same tools and system, so each request after the
            # first reads them from the cache.
            blocks[-1]["cache_control"] = copy.deepcopy(self._cache_control)
            # And on the prompt when later runs share it but not the list after it (a cut
            # list, or pasted skill text): the next run reads the prompt from the cache. A
            # prompt with {{variables}} differs per customer, so it gets none.
            if len(blocks) == 2 and (record.cut or record.pasted) and not (record.prompt or {}).get("variables"):
                blocks[0]["cache_control"] = copy.deepcopy(self._cache_control)
        tool_def = None
        if record.tool:
            tool_def = {"name": LOAD_SKILL_TOOL_NAME, "description": LOAD_SKILL_TOOL_DESCRIPTION,
                        "input_schema": self._tool_schema(list(record.tool_names))}
        return blocks, tool_def

    @staticmethod
    def _blocks(head: str, menu: str) -> List[Dict[str, Any]]:
        """The system text as blocks: the prompt exactly as filled, then the skill list."""
        blocks: List[Dict[str, Any]] = []
        if head.strip():
            blocks.append({"type": "text", "text": head})
        if menu.strip():
            blocks.append({"type": "text", "text": f"\n\n{menu}" if blocks else menu})
        return blocks

    def _routes_later(self, record: ConversationRecord) -> bool:
        """Whether a later message of this conversation is routed before its model call."""
        from ._config import _disabled_by_init

        tool, _inject = self._channels()
        # A conversation in an experiment keeps its first decision (its arm) for every message.
        return record.cut and record.tool and tool and not _disabled_by_init() and not record.experiment

    def _route_later_run(self, run_key: str, ctx: Dict[str, Any]) -> Any:
        """This message's own decision, for a conversation whose list is cut (or None)."""
        try:
            routed = self._route(run_key, self._message(ctx, None))
        except Exception:
            self._log.debug("skill routing failed (non-fatal)", exc_info=True)
            return None
        prefix, offered, routing_id, truncated, _delivered, _versions, _run_prompt = routed
        self._store_menu(prefix, offered, routing_id, truncated)
        return routed

    # ── the seams AgentBinding asks for ─────────────────────
    def _router(self) -> Any:
        return _get_skill_router()

    def _run_key(self) -> Optional[str]:
        return _RUN.get()

    def _message(self, context: Any, agent: Any) -> Optional[str]:
        message = context.get("message")
        return message.strip() or None if isinstance(message, str) else None

    def _values(self, context: Any) -> Any:
        return context.get("values")

    def _agent_label(self, agent: Any) -> Optional[str]:
        return None

    def _record(self, **fields: Any) -> bool:
        from ._agent import _record_turn

        return _record_turn(fields)

    def _conversation(self, context: Any, agent: Any) -> Optional[ConversationRecord]:
        conversation = context.get("conversation") if isinstance(context, dict) else None
        return conversation._current() if conversation is not None else None

    def _keep_conversation(self, context: Any, agent: Any, run_key: Optional[str],
                           record: ConversationRecord) -> None:
        conversation = context.get("conversation") if isinstance(context, dict) else None
        if conversation is not None:
            conversation._fix(record)

    def _routed_continuation(self, context: Any, agent: Any) -> Any:
        routed = context.get("routed") if isinstance(context, dict) else None
        return routed if routed is not None and routed[2] else None

    def _stamp_routing(self, run_key: Optional[str], routing_id: str) -> None:
        """A background decision lands on its run's trace only while that trace is open."""
        from .otel import record_skill_rail

        with self._lock:
            run = self._open_runs.get(run_key) if run_key is not None else None
        if run is None:
            return
        with run._lock:
            if run._ended:
                return
            record_skill_rail(routing_id=routing_id)
            record = run._record
            if run._recorded_by_me and record is not None and not record.routing_id:
                with run._conversation._lock:
                    record.routing_id = routing_id
                    for name, origin in list(record.shown.items()):
                        if origin is None:
                            record.shown[name] = routing_id

    def _serve_load(self, name: str) -> str:
        from ._agent import _serve_and_record

        return _serve_and_record(self._router(), name, scope=self._run_key(), agent_name=self.name)


def agent_config(
    agent_name: str,
    *,
    fallback: Optional[str] = None,
    on_unavailable: str = "continue",
    cache_control: Optional[Dict[str, Any]] = _UNSET,
    on_trace: Any = None,
) -> AnthropicAgentConfig:
    """Pull an agent's prompt and skills from DecimalAI, for the Anthropic Messages API.

    ::

        from decimalai.anthropic import agent_config

        support = agent_config("support", fallback=BASE_SYSTEM)
        with support.run(question, variables=customer) as run:
            runner = client.beta.messages.tool_runner(
                model=MODEL, max_tokens=4096, system=run.system,
                tools=[lookup_order, *run.tools],
                messages=[{"role": "user", "content": question}])
            answer = runner.until_done()

    Reads the agent's saved prompt here and keeps
    a local copy, re-read in the background at most once a minute. Each run is
    routed on its own message: the agent's first run before it opens, later runs
    from the local copy while their own decision is recorded in the background,
    and every one-message run of an agent with more skills than fit the list.
    Turns on Anthropic tracing for DecimalAI runs, and initializes the SDK from
    ``DECIMAL_API_KEY`` when ``decimalai.init()`` has not been called.

    With the ``load_skill`` tool switched off (``DECIMALAI_LOAD_SKILL_TOOL=0`` or
    ``init(load_skill_tool=False)``), each run is routed and the text of the
    skills picked for it goes into ``run.system`` instead.

    With no API key and no ``init()``, the agent runs keyless, as
    :func:`decimalai.agent_config` describes: its ``fallback`` is the prompt, each
    run lists the skills the project lists in ``.decimal/skills.toml``, and nothing
    is sent. With a key, an agent the workspace has no agent by is created.

    Args:
        agent_name: The agent's name in DecimalAI.
        fallback: The system prompt to use when the agent has no saved prompt, when
            no API key is set, or when DecimalAI cannot be reached now. With a key and
            without one, an unreachable DecimalAI raises here rather than starting an
            agent with no prompt.
        on_unavailable: ``"continue"`` (default) runs without skills when DecimalAI
            never returned them, and the run's record says so; ``"fail"`` raises
            when the run opens instead.
        cache_control: The prompt-caching breakpoint DecimalAI puts on
            ``run.system``: at its end, and also after the prompt when later runs
            share the prompt but not the list (two of a request's four).
            ``{"type": "ephemeral"}`` by default; ``{"type": "ephemeral", "ttl":
            "1h"}`` when a later breakpoint of yours lives an hour (an earlier
            breakpoint may not have the shorter life); ``None`` for none.
        on_trace: Called with each run's finished trace before it is sent.

    Raises:
        ImportError: the Anthropic SDK is not installed.
        DecimalConfigError: no API key is set in a project ``.decimal/link.toml``
            connects to a workspace.
        ValueError: ``on_unavailable`` is neither ``"continue"`` nor ``"fail"``,
            or ``cache_control`` is not a dict or None.
    """
    if on_unavailable not in ("continue", "fail"):
        raise ValueError("on_unavailable must be 'continue' or 'fail'")
    if cache_control is _UNSET:
        cache_control = _DEFAULT_CACHE_CONTROL
    if cache_control is not None and not isinstance(cache_control, dict):
        raise ValueError('cache_control must be a dict such as {"type": "ephemeral"}, or None')
    _require_anthropic()
    from ._agent import _open_prompt

    prompt = _open_prompt(agent_name, fallback)
    if not prompt.keyless:  # keyless runs are traced nowhere
        _turn_on_tracing()
    return AnthropicAgentConfig(prompt, fallback=fallback, on_unavailable=on_unavailable,
                                cache_control=cache_control, on_trace=on_trace)


def load_agent(*args: Any, **kwargs: Any) -> NoReturn:
    """The 0.21 name of :func:`agent_config`, kept to say so."""
    raise TypeError(
        "decimalai.anthropic.load_agent() is agent_config() (every module was renamed in decimalai 0.21); "
        "it takes the same arguments and returns the same config: "
        "from decimalai.anthropic import agent_config"
    )


def instrument(*args: Any, **kwargs: Any) -> None:
    """Removed in decimalai 0.23: use :func:`agent_config`.

    ``instrument(enable_skill_loader=True)`` patched every Anthropic client in the
    process. Skills now come per agent from ``agent_config()``; tracing is
    ``decimalai.init(anthropic=True)``, which ``agent_config()`` also turns on for
    its own runs.
    """
    raise TypeError(
        "decimalai.anthropic.instrument() was removed in decimalai 0.23. For an agent's "
        "prompt and skills use decimalai.anthropic.agent_config(name) and open a run per "
        "message (with support.run(question) as run: system=run.system, "
        "tools=[*your_tools, *run.tools]). For tracing alone: decimalai.init(anthropic=True)."
    )


__all__ = ["AnthropicAgentConfig", "Conversation", "Run", "agent_config"]
