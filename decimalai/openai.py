"""OpenAI SDK: an agent's DecimalAI prompt and skills, one run at a time.

For code that calls OpenAI's Chat Completions or Responses API itself, with a tool
loop of its own::

    from openai import OpenAI
    from decimalai.openai import agent_config

    client = OpenAI()
    support = agent_config("support", fallback=BASE_PROMPT)

    with support.run(question, variables=customer) as run:
        messages = [{"role": "system", "content": run.system}, run.message]
        while True:
            reply = client.chat.completions.create(
                model="gpt-5.4-mini", messages=messages,
                tools=[*my_tools, *run.tools],          # your tools, then load_skill
            ).choices[0].message
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                messages.append(run.tool_result(call) or my_tool_result(call))
    answer = reply.content

On the Responses API the same run gives ``instructions=run.system``,
``input=[run.message]`` and ``tools=[*my_tools, *run.response_tools]``, and
``run.tool_result(call)`` answers a ``function_call`` item with its
``function_call_output``. ``run.tool_result()`` returns ``None`` for any tool that
is not ``load_skill``, so ``or`` hands the call to your own tools.

One run is one user message, from the question to the final answer. Opening it
routes the message, fills the prompt's ``{{variables}}`` from ``variables`` and
starts the run's trace, filed under the DecimalAI agent; closing it sends the
trace with the run's record (prompt version, skills offered and loaded, routing
id). ``support.run()`` itself calls no model: make the model calls inside the
``with`` block. ``async with`` routes off the event loop, for ``AsyncOpenAI``.

The tools are plain dicts in OpenAI's shapes, and ``run.tool_result()`` reads a
call as the SDK returns it or as a dict, so the same run serves any client that
speaks those shapes: Azure OpenAI, the OpenAI-compatible endpoints of other model
providers, and LiteLLM's ``completion()``. Calls made through ``openai`` clients
are traced; calls made through another library carry the run's record only.

A chat keeps one conversation, whose ``system`` and ``load_skill`` stay the same
for every message::

    chat = support.conversation(variables=customer)        # or conversation(state=saved)
    with chat.run(question) as run:
        ...
    saved = chat.state                                   # store it beside the chat's history

A newer saved prompt or skill list reaches the next conversation, not an open
one, so a conversation's requests keep one prefix, which OpenAI's prompt cache
reuses, and ``chat.state`` lets a server that keeps no chat in memory resume it
with the same text and tool.

Nothing DecimalAI does changes the SDK's behaviour, and other clients get no
skills. ``agent_config()`` turns on OpenAI tracing (OpenInference's instrumentor,
which wraps the SDK's classes for the process; installed by ``decimalai[openai]``)
on the same pipeline as ``decimalai.init(openai=True)``, the global tracer
provider (an app with its own sets it up first), and sends only its runs: no
OpenAI call made outside one and none of the app's own spans, unless
``init(openai=True)`` asked for every OpenAI call.
"""

from __future__ import annotations

import copy
import json
import logging
import threading
import time
from contextvars import ContextVar
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ._agent_binding import AgentBinding, ConversationRecord, _in_thread
from ._agent_binding import new_skills_block as _new_skills_block
from ._provider_run import (
    _drain_finishing,  # noqa: F401 - decimalai.flush() drains it; tests reach it here
    _enclosing_trace_id,
    _event_loop_running,
    _finish_later,
    _is_error,
    open_run_span,
)
from .skill_router import _load_routing_id_ctx

logger = logging.getLogger("decimalai.openai")

#: The run whose ``with`` block is open in this context. The background routing
#: and a load's scope read it; a load sets it around its own call.
_RUN: "ContextVar[Optional[str]]" = ContextVar("decimalai_openai_run", default=None)

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


def _require_openai() -> None:
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "decimalai.openai.agent_config() needs the OpenAI SDK: "
            'pip install "decimalai[openai]"'
        ) from exc


def _turn_on_tracing() -> None:
    """OpenAI tracing for DecimalAI runs, once per process."""
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

            instrument_for_runs("openai")
        except Exception:
            logger.warning("DecimalAI could not turn on OpenAI tracing (continuing)", exc_info=True)
        _tracing_turned_on = True


def _field(value: Any, key: str) -> Any:
    """``value[key]`` for a dict, ``value.key`` for an SDK object; None when absent."""
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _call_parts(call: Any) -> Tuple[Optional[str], Any, Any, Any]:
    """(api, name, arguments, id) of a function call, as an SDK object or a plain dict.

    ``api`` is ``"chat"`` for a Chat Completions tool call (``call.function.name``,
    answered by ``tool_call_id``) and ``"responses"`` for a Responses API
    ``function_call`` item (``call.name``, answered by ``call_id``); None for
    anything else.
    """
    function = _field(call, "function")
    if function is not None:
        return "chat", _field(function, "name"), _field(function, "arguments"), _field(call, "id")
    if _field(call, "type") == "function_call":
        return "responses", _field(call, "name"), _field(call, "arguments"), _field(call, "call_id")
    return None, None, None, None


def _arguments(raw: Any) -> Any:
    """A call's arguments as a dict: OpenAI sends them as a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            parsed = json.loads(raw or "{}")
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _result_message(api: str, call_id: Any, text: str) -> Dict[str, Any]:
    """The answer to one call in its API's shape."""
    if api == "responses":
        return {"type": "function_call_output", "call_id": call_id, "output": text}
    return {"role": "tool", "tool_call_id": call_id, "content": text}


# ── a conversation ───────────────────────────────────────────


class Conversation:
    """Messages that share one ``system`` and one ``load_skill`` definition.

    Made by :meth:`OpenAIAgentConfig.conversation`. Its first run composes the
    system text (the prompt filled from ``variables``, then the skill list) and the
    tool; every later run reuses both, byte for byte, and is still routed on its
    own message. Keep one per chat, beside the chat's message history; a server
    that keeps no chat in memory stores :attr:`state` with the history and
    resumes with ``support.conversation(state=...)``.

    What it keeps is the ``ConversationRecord`` every DecimalAI binding shares for
    a chat (``decimalai._agent_binding``); this module renders it as a system
    string and OpenAI's tool definitions.

    Attributes:
        id: This conversation's id; its runs' traces carry it as ``session_id``.
    """

    def __init__(self, config: "OpenAIAgentConfig", variables: Any,
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

    Open it with ``with`` (``OpenAI``) or ``async with`` (``AsyncOpenAI``). Opening
    routes the message and starts the run's trace; closing sends it. The model
    calls for this message belong inside the block.

    Attributes:
        system: The system text: the agent's prompt, then its skill list and one
            sentence naming ``load_skill``. A Chat Completions system message's
            ``content``, or the Responses API's ``instructions``.
        message: This run's user turn, ``{"role": "user", "content": ...}``.
        tools: ``load_skill`` for Chat Completions ``tools=``; empty when the run
            has no skill list.
        response_tools: ``load_skill`` for the Responses API's ``tools=``.
    """

    def __init__(self, config: "OpenAIAgentConfig", conversation: Conversation,
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
        self._system = ""
        self._tool_def: Optional[Dict[str, Any]] = None
        self._state: Optional[Dict[str, Any]] = None
        self._span: Any = None
        self._otel_ctx: Any = None
        self._tokens: List[Tuple[Any, Any]] = []
        self._read: set = set()

    def __repr__(self) -> str:
        return f"Run(agent={self._config.name!r}, open={self._mode is not None and not self._exited})"

    # ── opening and closing ─────────────────────────────────
    def __enter__(self) -> "Run":
        if _event_loop_running():
            self._config._warn_once(None, (
                "a run opened with `with` inside a running event loop routes on that loop "
                "and blocks it; use `async with support.run(...)` with AsyncOpenAI."))
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
        self._span, self._otel_ctx, tokens = open_run_span("openai", self._config.name, self._conversation.id)
        self._tokens.extend(tokens)
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
                and not self._read & {"tools", "response_tools", "tool_result", "load_skill"}:
            config._warn_once(None, (
                "run.system names the load_skill tool, but this run's tools never included it. "
                "Add *run.tools (Chat Completions) or *run.response_tools (Responses API) to tools=, "
                "and answer its calls with run.tool_result(call)."))
        if self._span is None or not self._read & {"system", "tools", "response_tools"}:
            return
        try:
            from . import otel, providers

            if not providers._traced_for_runs("openai"):
                return
            started = otel.spans_started(self._span.get_span_context().trace_id)
            if started is not None and started <= 1:  # None: evicted by newer traces, unknown
                advice = (
                    "no OpenAI request was traced inside this run. Make the model calls inside the "
                    "`with` block. Calls made through another library than `openai` (LiteLLM, a raw "
                    "HTTP client) are not traced; the run's record is.")
                if "openai" in providers._runs_via_another_tool:
                    advice += (" OpenAI tracing was turned on outside DecimalAI: DecimalAI receives "
                               "those calls only if that tracing sends its spans to the global "
                               "tracer provider.")
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
        self._system, self._tool_def = self._config._shape(record)
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
    def system(self) -> str:
        """The system text: the prompt with its ``{{variables}}`` filled, then the skill list.

        The skill list ends with one sentence naming the ``load_skill`` tool. Give it
        as the system message's ``content`` (Chat Completions) or as ``instructions``
        (Responses API, on every request: a ``previous_response_id`` does not carry
        instructions over). To add text of your own, put it after this, keeping it
        the same for every message of a conversation.
        """
        self._require_open("system")
        self._read.add("system")
        return self._system

    @property
    def message(self) -> Dict[str, Any]:
        """This run's user turn: ``{"role": "user", "content": question}``.

        On a later message of a conversation whose skill list is cut to fit, the
        skills routed for this message that the conversation has not shown yet
        follow the question in the same text. They go in the user turn, not in
        ``system``: a conversation's history only grows. Append it to the history
        (Chat Completions ``messages``, Responses API ``input``) instead of your own
        user message.
        """
        self._require_open("message")
        self._read.add("message")
        extra = (self._state or {}).get("new_block")
        text = self._message if isinstance(self._message, str) else ""
        return {"role": "user", "content": f"{text}\n\n{extra}" if extra else text}

    @property
    def tools(self) -> List[Dict[str, Any]]:
        """``load_skill`` as a Chat Completions tool: ``{"type": "function", "function": {...}}``."""
        self._require_open("tools")
        self._read.add("tools")
        if self._tool_def is None:
            return []
        return [{"type": "function", "function": copy.deepcopy(self._tool_def)}]

    @property
    def response_tools(self) -> List[Dict[str, Any]]:
        """``load_skill`` as a Responses API function tool: ``{"type": "function", "name": ...}``.

        Strict, as the Responses API makes a function tool by default: the model can
        name only a listed skill. ``run.tools`` leaves ``strict`` out, because some
        OpenAI-compatible endpoints refuse it in Chat Completions requests.
        """
        self._require_open("response_tools")
        self._read.add("response_tools")
        if self._tool_def is None:
            return []
        return [{"type": "function", **copy.deepcopy(self._tool_def), "strict": True}]

    def tool_result(self, call: Any) -> Optional[Dict[str, Any]]:
        """The answer to a ``load_skill`` call; ``None`` for any other tool's call.

        Takes a Chat Completions tool call (``reply.tool_calls[i]``) and returns its
        ``{"role": "tool", "tool_call_id": ..., "content": ...}`` message, or a
        Responses API ``function_call`` item and returns its ``function_call_output``.
        The SDK's objects and plain dicts both work. In an ``async with`` run,
        ``await run.atool_result(call)`` keeps the load off the event loop.
        """
        self._require_open("tool_result")
        self._read.add("tool_result")
        api, name, raw, call_id = _call_parts(call)
        if api is None or self._tool_def is None or name != self._tool_def["name"]:
            return None
        if self._mode == "async" and _event_loop_running():
            self._config._warn_once(None, (
                "run.tool_result() loads on the event loop; in an `async with` run use "
                "`await run.atool_result(call)`."))
        text, _error = self._serve(_arguments(raw))
        return _result_message(api, call_id, text)

    async def atool_result(self, call: Any) -> Optional[Dict[str, Any]]:
        """:meth:`tool_result` for an async loop: the load runs off the event loop."""
        self._require_open("atool_result")
        self._read.add("tool_result")
        api, name, raw, call_id = _call_parts(call)
        if api is None or self._tool_def is None or name != self._tool_def["name"]:
            return None
        text, _error = await self._aserve(_arguments(raw))
        return _result_message(api, call_id, text)

    def load_skill(self, name: str) -> str:
        """One listed skill's full text, for a ``load_skill`` call your own code dispatches.

        What :meth:`tool_result` puts in its message, for a loop or a tool runner
        that wants the text alone. Always returns text the model can act on: an
        unlisted name, a skill already loaded in this run or an outage comes back
        as a sentence.
        """
        self._require_open("load_skill")
        self._read.add("load_skill")
        text, _error = self._serve({"name": name})
        return text

    async def aload_skill(self, name: str) -> str:
        """:meth:`load_skill` off the event loop."""
        self._require_open("aload_skill")
        self._read.add("load_skill")
        text, _error = await self._aserve({"name": name})
        return text

    # ── serving load_skill ──────────────────────────────────
    def _requested(self, tool_input: Any) -> Tuple[Optional[str], Optional[str]]:
        """(the skill to load, or None and a sentence for the model)."""
        from ._config import _disabled_by_init

        if self._exited:
            self._config._warn_once(None, (
                "load_skill was called after its run closed: the skill is served, but the run's "
                "record will not show it. Make the model calls inside the `with` block."))
        if _disabled_by_init() or not (self._state or {}).get("tool"):
            return None, "DecimalAI skills are switched off for this agent. Answer without them."
        raw = tool_input.get("name") if isinstance(tool_input, dict) else None
        name = str(raw or "").strip()
        if not name:
            return None, "load_skill needs the name of one skill from the list."
        record = self._record
        if record is not None and not record.cut:
            # Chat Completions does not hold a call to the tool's schema (its tool is
            # not strict), so the name is checked here. A cut list names only some
            # skills; the platform checks those.
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


# ── the config ───────────────────────────────────────────────


class OpenAIAgentConfig(AgentBinding):
    """One agent's prompt and skills in DecimalAI, for the OpenAI SDK.

    Made by :func:`agent_config`. Open one run per user message::

        with support.run(question, variables=customer) as run:
            ...  # system: run.system, tools=[*your_tools, *run.tools]

    or keep one :meth:`conversation` per chat.

    Attributes:
        name: The agent's canonical DecimalAI name.
    """

    _log = logger
    _VALUES_HINT = "Pass them as support.run(..., variables=...), as attributes or dict keys."

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str,
                 on_trace: Any) -> None:
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        self._on_trace = on_trace
        self._open_runs: Dict[str, Run] = {}

    def __repr__(self) -> str:
        return (
            f"OpenAIAgentConfig(name={self.name!r}, prompt_version={self.prompt_version!r}, "
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
        ``run.system``, ``run.message``, ``run.tools`` (``run.response_tools``) and
        ``run.tool_result()`` are this run's. For a message in a chat, use
        :meth:`conversation`, which keeps ``system`` the same for every message.

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

    def _shape(self, record: ConversationRecord) -> Tuple[str, Optional[Dict[str, Any]]]:
        """A conversation's record as OpenAI sees it: the system text and the tool.

        The tool is ``{name, description, parameters}``, which ``run.tools`` and
        ``run.response_tools`` wrap in each API's shape. Its schema already meets
        strict mode's rules (every property required, no others allowed), so the
        Responses API's strict tool can only name a listed skill. The same record
        always gives the same text and definition, so a conversation resumed from
        its state in another process sends what its first run sent.
        """
        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION, LOAD_SKILL_TOOL_NAME

        tool_def = None
        if record.tool:
            tool_def = {"name": LOAD_SKILL_TOOL_NAME, "description": LOAD_SKILL_TOOL_DESCRIPTION,
                        "parameters": self._tool_schema(list(record.tool_names))}
        return record.text, tool_def

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
    on_trace: Any = None,
) -> OpenAIAgentConfig:
    """Pull an agent's prompt and skills from DecimalAI, for the OpenAI SDK.

    ::

        from decimalai.openai import agent_config

        support = agent_config("support", fallback=BASE_PROMPT)
        with support.run(question, variables=customer) as run:
            messages = [{"role": "system", "content": run.system}, run.message]
            reply = client.chat.completions.create(
                model=MODEL, messages=messages, tools=[*my_tools, *run.tools],
            ).choices[0].message
            # for each call in reply.tool_calls: run.tool_result(call) or your tool's answer

    Reads the agent's saved prompt here and keeps a local copy, re-read in the
    background at most once a minute. Each run is routed on its own message: the
    agent's first run before it opens, later runs from the local copy while their
    own decision is recorded in the background, and every one-message run of an
    agent with more skills than fit the list. Turns on OpenAI tracing for
    DecimalAI runs, and initializes the SDK from ``DECIMAL_API_KEY`` when
    ``decimalai.init()`` has not been called.

    With the ``load_skill`` tool switched off (``DECIMALAI_LOAD_SKILL_TOOL=0`` or
    ``init(load_skill_tool=False)``), each run is routed and the text of the
    skills picked for it goes into ``run.system`` instead.

    With no API key and no ``init()``, the agent runs keyless, as
    :func:`decimalai.agent_config` describes: its ``fallback`` is the prompt, its
    skills come from the project and the public registry, and nothing is traced.
    With a key, an agent the workspace has no agent by is created.

    Args:
        agent_name: The agent's name in DecimalAI.
        fallback: The system prompt to use when the agent has no saved prompt, when
            no API key is set, or when DecimalAI cannot be reached now. With a key and
            without one, an unreachable DecimalAI raises here rather than starting an
            agent with no prompt.
        on_unavailable: ``"continue"`` (default) runs without skills when DecimalAI
            never returned them, and the run's record says so; ``"fail"`` raises
            when the run opens instead.
        on_trace: Called with each run's finished trace before it is sent.

    Raises:
        ImportError: the OpenAI SDK is not installed.
        DecimalConfigError: no API key is set in a project ``.decimal/link.toml``
            connects to a workspace.
        ValueError: ``on_unavailable`` is neither ``"continue"`` nor ``"fail"``.
    """
    if on_unavailable not in ("continue", "fail"):
        raise ValueError("on_unavailable must be 'continue' or 'fail'")
    _require_openai()
    from ._agent import _open_prompt

    prompt = _open_prompt(agent_name, fallback)
    if not prompt.keyless:  # keyless runs are traced nowhere
        _turn_on_tracing()
    return OpenAIAgentConfig(prompt, fallback=fallback, on_unavailable=on_unavailable,
                             on_trace=on_trace)


__all__ = ["Conversation", "OpenAIAgentConfig", "Run", "agent_config"]
