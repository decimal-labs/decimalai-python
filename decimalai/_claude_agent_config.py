"""Per-agent DecimalAI configuration for Claude Agent SDK queries and chats.

This is the config path. The separate native ``skills_plugin()`` path remains
available for applications that deliberately use Claude Code's disk loader.
"""

from __future__ import annotations

import contextvars
import copy
import threading
from dataclasses import replace
from typing import Any, AsyncIterator, Dict, Optional
from uuid import uuid4

from ._agent_binding import (
    AgentBinding,
    ConversationRecord,
    _in_thread,
    attach_prompt_records,
    new_skills_block,
)
from ._skill_witness import merge_delivered_versions

_RUN: contextvars.ContextVar[Any] = contextvars.ContextVar("decimalai_claude_config_run", default=None)
_SERVER = "decimalai"
_TOOL = "mcp__decimalai__load_skill"


class Conversation:
    """An in-process chat that keeps its first prompt and resumes its Claude session.

    Keep one beside your application's chat. Its runs must be sequential; raw
    ``resume`` flags are rejected because they could restore another config.
    """

    def __init__(self, config: "ClaudeAgentConfig", variables: Any) -> None:
        self.config = config
        self.variables = variables
        self._record: Optional[ConversationRecord] = None
        self._session_id: Optional[str] = None
        self._manifest_prompt: Optional[str] = None
        self._running = False
        self._lock = threading.Lock()

    def run(self, message: str) -> "Run":
        if not isinstance(message, str):
            raise TypeError("message must be the user's text")
        return Run(self.config, message, self.variables, conversation=self)


class Run:
    """One user message, used as ``async with support.run(question) as run``.

    Consume :meth:`query` inside the block. It uses the saved prompt, supplies
    ``load_skill`` through an in-process MCP server, and records one trace.
    """

    def __init__(self, config: "ClaudeAgentConfig", message: str, variables: Any,
                 conversation: Optional[Conversation] = None) -> None:
        self.config = config
        self.message = message
        self._query_message = message
        self.variables = variables
        self._conversation = conversation
        self.key = f"claude_config_{uuid4().hex}"
        self.system = ""
        self._manifest_prompt: Optional[str] = None
        self._state: Dict[str, Any] = {}
        self._fields: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._token: Any = None
        self._active = False
        self._entered = False
        self._queried = False
        self._sealed = False
        self._stream: Any = None
        self._provider_stream: Any = None

    async def __aenter__(self) -> "Run":
        if self._entered:
            raise RuntimeError("a Claude config run can only be entered once")
        if self._conversation is not None:
            with self._conversation._lock:
                if self._conversation._running:
                    raise RuntimeError("run a Claude conversation's messages sequentially")
                self._conversation._running = True
        self._entered = self._active = True
        self._token = _RUN.set(self)
        with self.config._lock:
            self.config._active_runs[self.key] = self
        try:
            context = {"message": self.message, "values": self.variables,
                       "conversation": self._conversation}
            previous = self._conversation._record if self._conversation is not None else None
            if previous is not None and self.config._routes_later(previous):
                context["routed"] = await _in_thread(self.config._route_later, self.key, self.message)
            self._state = await self.config._prepare_run(context, None, self.key)
            # A manifest declares the unfilled template, never this customer's
            # values or the per-run routed menu. A resumed chat keeps its first.
            self._manifest_prompt = (
                self._conversation._manifest_prompt if self._conversation is not None
                and self._conversation._record is not None else self.config._template())
            if self._conversation is not None and self._conversation._manifest_prompt is None:
                self._conversation._manifest_prompt = self._manifest_prompt
            self.system = self._state.get("text")
            if self.system is None:
                self.system = await _in_thread(self.config._compose, context, None, self.key, self._state)
            self._state["text"] = self.system
            routed = self._state.get("routed")
            if routed is not None and routed[2] and previous is not None:
                offered, decision = list(routed[1]), routed[2]
                new = [name for name in offered if name not in previous.shown]
                if new:
                    self._query_message = f"{self.message}\n\n{new_skills_block(routed[0], new)}"
                self._state.update(offered_now=offered, routing_now=decision, new_names=new)
                self._record(routing_id=decision, offered=offered)
            return self
        except BaseException:
            self._close()
            raise

    async def __aexit__(self, *exc: Any) -> None:
        try:
            try:
                if self._stream is not None:
                    await self._stream.aclose()
            finally:
                if self._provider_stream is not None:
                    await self._provider_stream.aclose()
        finally:
            self._close()

    def _close(self) -> None:
        self._active = False
        if self._conversation is not None:
            with self._conversation._lock:
                self._conversation._running = False
        with self._lock:
            self._sealed = True
        with self.config._lock:
            self.config._active_runs.pop(self.key, None)
            self.config._runs.pop((self.key, None), None)
        if self._token is not None:
            _RUN.reset(self._token)
            self._token = None

    def _record(self, **fields: Any) -> bool:
        with self._lock:
            if self._sealed:
                return False
            for key in ("offered", "loaded", "delivered"):
                self._fields[key] = list(dict.fromkeys([
                    *self._fields.get(key, []), *(fields.get(key) or []),
                ]))
            # A successful model-initiated load delivered the body too.
            self._fields["delivered"] = list(dict.fromkeys([
                *self._fields["delivered"], *self._fields["loaded"],
            ]))
            self._fields.setdefault("loaded_hashes", {}).update(fields.get("loaded_hashes") or {})
            self._fields["delivered_versions"] = merge_delivered_versions(
                self._fields.get("delivered_versions"), fields.get("delivered_versions"))
            for key in ("routing_id", "prompt"):
                if fields.get(key) is not None:
                    self._fields[key] = copy.deepcopy(fields[key])
        return True

    def _apply(self, trace: Any) -> None:
        """Called on the existing Claude trace's send thread, never an OTel run."""
        pending = self._state.get("pending")
        if pending is not None and not pending.done():
            try:
                pending.result(timeout=2.0)
            except Exception:
                pass  # routing is best effort, as on the other config bindings
        with self._lock:
            fields = copy.deepcopy(self._fields)
            self._sealed = True
        trace.routing_id = fields.get("routing_id")
        for field, key in (("skills_offered_in_prompt", "offered"),
                           ("skills_loaded_by_agent", "loaded"), ("skills_delivered", "delivered")):
            setattr(trace, field, list(dict.fromkeys([*getattr(trace, field), *fields.get(key, [])])))
        trace.skills_delivered_versions = merge_delivered_versions(
            trace.skills_delivered_versions, fields.get("delivered_versions"))
        for name in fields.get("loaded", []):
            entry = {"name": name}
            digest = fields.get("loaded_hashes", {}).get(name)
            if digest:
                entry["hash"] = digest
            if not any(item.get("name") == name for item in trace.active_skills):
                trace.active_skills.append(entry)
        if fields.get("prompt"):
            attach_prompt_records(trace.spans, {self.config.name: fields["prompt"]})

    async def _load_skill(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        from ._provider_run import _is_error
        from .skill_router import _load_routing_id_ctx

        if not self._active or self._sealed:
            return {"content": [{"type": "text", "text": "load_skill: this run has ended."}],
                    "is_error": True}
        token = _RUN.set(self)  # MCP may call from a task other than the query's task.
        name = str(arguments.get("name") or "").strip()
        record = self._state.get("conversation")
        decision = None
        if record is not None and record.cut:
            if name in self._state.get("offered_now", []):
                decision = self._state.get("routing_now")
            else:
                decision = record.shown.get(name)
        decision_token = _load_routing_id_ctx.set(decision)
        try:
            text = await self.config._load(name, self._state)
            return {"content": [{"type": "text", "text": text}], "is_error": _is_error(text)}
        finally:
            _load_routing_id_ctx.reset(decision_token)
            _RUN.reset(token)

    def _options(self, options: Any) -> Any:
        from claude_agent_sdk import ClaudeAgentOptions, create_sdk_mcp_server, tool

        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION

        if options is None:
            options = ClaudeAgentOptions()
        if not isinstance(options, ClaudeAgentOptions):
            raise TypeError("options must be ClaudeAgentOptions")
        # Replaying a session under a different prompt/tool schema breaks Claude's
        # signed thinking history. Only a config-owned conversation can resume.
        if options.resume or options.continue_conversation or getattr(options, "session_id", None):
            raise ValueError("raw Claude resume, session_id and continue_conversation options bypass config; "
                             "use support.conversation().run(message) to keep its prompt and session together")
        if not isinstance(options.mcp_servers, dict):
            raise ValueError("Claude config needs mcp_servers as a dict so it can add load_skill; "
                             "pass your MCP server configurations as dictionary entries")
        if _SERVER in options.mcp_servers:
            raise ValueError("mcp_servers['decimalai'] is reserved for this config's load_skill tool")
        if getattr(options, "skills", None):
            raise ValueError("Claude config supplies routed skills; do not also enable native skills. "
                             "Use the separate tracing/native-loader setup for disk skills")
        servers = dict(options.mcp_servers)
        allowed = list(options.allowed_tools)
        enabled = bool(self._state.get("tool") and self._state.get("names"))
        if enabled:
            if _TOOL in options.disallowed_tools:
                raise ValueError("load_skill is disallowed by these Claude options")
            definition = tool("load_skill", LOAD_SKILL_TOOL_DESCRIPTION,
                              self.config._tool_schema(self._state.get("tool_names") or []))(self._load_skill)
            servers[_SERVER] = create_sdk_mcp_server(_SERVER, tools=[definition])
            allowed = list(dict.fromkeys([*allowed, _TOOL]))
        resume = self._conversation._session_id if self._conversation is not None else None
        return replace(options, system_prompt=self.system, mcp_servers=servers, allowed_tools=allowed,
                       resume=resume)

    async def query(self, *, options: Any = None, transport: Any = None) -> AsyncIterator[Any]:
        """Run the configured query once, preserving the caller's other options.

        ``system_prompt`` comes from config; other MCP servers, tools, model and
        limits stay as supplied. Existing ``ClaudeAgentOptions`` is not mutated.
        A ``support.conversation()`` resumes its own session automatically.
        """
        if not self._active:
            raise RuntimeError("consume run.query() inside 'async with support.run(question) as run:'")
        if self._queried:
            raise RuntimeError("each Claude config run can query only once; start another support.run()")
        from claude_agent_sdk import query

        from .claude_agent_sdk import trace_stream

        configured = self._options(options)
        self._queried = True
        if self._conversation is not None and self._conversation._record is not None:
            for name in self._state.get("new_names", []):
                self._conversation._record.shown.setdefault(name, self._state.get("routing_now"))
        # Global tracing may already be installed by an existing application.
        # Bypass only DecimalAI's wrapper; the explicit stream is this run's trace.
        original = getattr(query, "__decimalai_original__", query)
        arguments = {"prompt": self._query_message, "options": configured}
        if transport is not None:
            arguments["transport"] = transport
        self._provider_stream = original(**arguments)
        self._stream = trace_stream(
            self._provider_stream,
            agent_name=self.config.name, user_input=self._query_message, options=configured,
            _config_run=self,
        )
        async for message in self._stream:
            if self._conversation is not None and type(message).__name__ == "ResultMessage":
                session_id = getattr(message, "session_id", None)
                if session_id:
                    with self._conversation._lock:
                        self._conversation._session_id = session_id
            yield message


class ClaudeAgentConfig(AgentBinding):
    """A saved prompt and routed skills for the Claude Agent SDK."""

    _VALUES_HINT = "Pass them as support.run(..., variables=...), as attributes or dict keys."

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str) -> None:
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        self._active_runs: Dict[str, Run] = {}

    def run(self, message: str, *, variables: Any = None) -> Run:
        if not isinstance(message, str):
            raise TypeError("message must be the user's text")
        return Run(self, message, variables)

    def conversation(self, *, variables: Any = None) -> Conversation:
        """Keep one chat's prompt/tool schema and resume its native Claude session.

        Kept in this process only; storing/resuming config state across process
        restarts is not part of this initial API.
        """
        return Conversation(self, variables)

    def _router(self) -> Any:
        from ._agent import _get_skill_router
        return _get_skill_router()

    def _run_key(self) -> Optional[str]:
        run = _RUN.get()
        return run.key if run is not None and run.config is self else None

    def _message(self, context: Any, agent: Any) -> Optional[str]:
        return context["message"]

    def _values(self, context: Any) -> Any:
        return context["values"]

    def _conversation(self, context: Any, agent: Any) -> Optional[ConversationRecord]:
        chat = context.get("conversation")
        return chat._record if chat is not None else None

    def _keep_conversation(self, context: Any, agent: Any, run_key: Optional[str],
                           record: ConversationRecord) -> None:
        chat = context.get("conversation")
        if chat is not None:
            chat._record = record

    def _routes_later(self, record: ConversationRecord) -> bool:
        from ._config import _disabled_by_init
        tool, _inject = self._channels()
        return record.cut and record.tool and tool and not _disabled_by_init() and not record.experiment

    def _route_later(self, run_key: str, message: str) -> Any:
        try:
            routed = self._route(run_key, message)
        except Exception:
            return None
        self._store_menu(routed[0], routed[1], routed[2], routed[3])
        return routed

    def _routed_continuation(self, context: Any, agent: Any) -> Any:
        routed = context.get("routed")
        return routed if routed is not None and routed[2] else None

    def _record(self, **fields: Any) -> bool:
        run = _RUN.get()
        return run._record(**fields) if run is not None and run.config is self else False

    def _stamp_routing(self, run_key: Optional[str], routing_id: str) -> None:
        with self._lock:
            run = self._active_runs.get(run_key)
        if run is not None:
            run._record(routing_id=routing_id)

    def _serve_load(self, name: str) -> str:
        from ._agent import _serve_and_record
        return _serve_and_record(self._router(), name, scope=self._run_key(), agent_name=self.name,
                                 record=self._record)


def agent_config(agent_name: str, *, fallback: Optional[str] = None,
                 on_unavailable: str = "continue") -> ClaudeAgentConfig:
    """Read this agent's saved prompt and skills; no global query patch needed.

    ::

        from decimalai.claude_agent_sdk import agent_config
        support = agent_config("support", fallback="You are a helpful agent.")
        async with support.run(question) as run:
            async for message in run.query(options=ClaudeAgentOptions(max_turns=6)):
                print(message)

    With no Decimal key and no ``init()`` it runs keyless, as the other bindings do: the
    ``fallback`` is the prompt, the skills are the ones ``.decimal/skills.toml`` lists, read
    from disk, and nothing is sent. Prompt refresh, variables and unavailable behavior are
    shared with the other config bindings. ``support.run`` starts a fresh query;
    ``support.conversation().run`` keeps the first prompt/tool schema and resumes
    its native Claude session. Raw session flags bypassing that state are rejected.
    """
    from ._agent import _open_prompt

    if on_unavailable not in ("continue", "fail"):
        raise ValueError("on_unavailable must be 'continue' or 'fail'")
    return ClaudeAgentConfig(_open_prompt(agent_name, fallback), fallback=fallback,
                             on_unavailable=on_unavailable)
