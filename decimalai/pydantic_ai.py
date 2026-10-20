"""Pydantic AI: an agent's DecimalAI prompt and skills, as one capability.

::

    from pydantic_ai import Agent
    from decimalai.pydantic_ai import agent_config

    support = agent_config("support", fallback=BASE_INSTRUCTIONS)
    agent = Agent("openai:gpt-5.4-mini", name=support.name,
                  tools=[lookup_order], capabilities=[support])
    result = await agent.run(question, deps=customer)

``support`` is a Pydantic AI capability, built only from Pydantic AI's documented
extension points:

- its instructions are the agent's saved prompt (``{{variables}}`` filled from
  the run's ``deps``), then the skill list routed on the run's prompt, then one
  sentence naming the tool;
- its toolset is ``load_skill``, which the model calls for a listed skill's text;
- ``wrap_run`` opens a DecimalAI run span around each run, so the run's model calls
  land in one trace, filed under the DecimalAI agent (Pydantic AI's own run span, the
  trace's root under ``Agent.instrument_all()``, is named for it too).

Nothing in Pydantic AI is patched, and agents without the capability are untouched.
Every run is routed on its own prompt: the first run before its first model request,
later runs from a local copy while their own decision is recorded in the background,
and every run of an agent with more skills than fit the list.

On a Claude model, a conversation keeps its first run's instructions and ``load_skill``:
Claude rejects a conversation whose system prompt or tools changed once a thinking
block is in its history (HTTP 400, "bound to a different conversation"; Pydantic AI
then drops the reasoning and retries). The capability keeps what it showed in the
run's request ``metadata``, which travels with ``result.all_messages()``, so a run
given that ``message_history`` (in any process) shows the same text and tool. A newer
saved prompt or skill list reaches the next conversation.

Tracing is DecimalAI's OpenTelemetry exporter, ``decimalai.init(otel=True)``, carrying
Pydantic AI's own spans (``Agent.instrument_all()``; any provider) or a provider
instrumentor's (``decimalai.init(openai=True)``). Pick one, or a model call is reported
twice.

The harness's ``Skills`` capability is the other way to give an agent skills: files the
harness loads (``decimalai.pydantic_ai_harness.skills``). An agent that has it gets no
DecimalAI skills from ``agent_config()``, and a warning says so once.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any, List, NoReturn, Optional

from decimalai._provider_recovery import retry_model_requests as retry_model_requests

from ._agent_binding import AgentBinding, ConversationRecord, is_claude

logger = logging.getLogger("decimalai.pydantic_ai")

#: Where a run's conversation record goes: ``ModelRequest.metadata[...][agent name]``.
#: Pydantic AI keeps request metadata in the history and never sends it to the model.
_HISTORY_KEY = "decimalai.conversation"

#: The run whose ``wrap_run`` is active in this context. Pydantic AI hands the
#: instructions function and the tool their ``RunContext``; everything else that
#: needs the run (the background routing, the load's scope) reads it here.
_RUN: "ContextVar[Optional[str]]" = ContextVar("decimalai_pydantic_ai_run", default=None)

#: The runs a DecimalAI config showed a skill list to, newest last. Read by the
#: harness's two-catalog warning (`decimalai.pydantic_ai_harness`).
_RUNS_WITH_MENU: "OrderedDict[str, None]" = OrderedDict()
_RUNS_WITH_MENU_MAX = 256
_runs_lock = threading.Lock()

#: The SkillRouter behind every config. `init()` drops it when a re-init changes
#: the key or workspace (`decimalai._drop_adapter_skill_routers`).
_skill_router_singleton: Any = None


def _menu_shown(ctx: Any) -> bool:
    """Whether a DecimalAI config put its skill list into this run."""
    run_id = getattr(ctx, "run_id", None)
    with _runs_lock:
        return run_id in _RUNS_WITH_MENU


def _note_menu_shown(run_id: Optional[str]) -> None:
    if not run_id:
        return
    with _runs_lock:
        _RUNS_WITH_MENU[run_id] = None
        _RUNS_WITH_MENU.move_to_end(run_id)
        while len(_RUNS_WITH_MENU) > _RUNS_WITH_MENU_MAX:
            _RUNS_WITH_MENU.popitem(last=False)


def _get_skill_router() -> Any:
    """The SkillRouter for these configs, built on first use; None when switched off."""
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


def _require_pydantic_ai() -> None:
    """Fail before any setup or network call when Pydantic AI is missing."""
    try:
        import pydantic_ai  # noqa: F401
    except ImportError:
        raise ImportError(
            "agent_config() needs Pydantic AI: pip install \"decimalai[pydantic-ai]\""
        ) from None


# ── the run's message ────────────────────────────────────────


def _run_query(ctx: Any) -> Optional[str]:
    """The run's user text from a RunContext, or None.

    ``prompt`` is the run's own prompt; the last user part in ``messages`` is the
    fallback for shapes that put the text there instead. Never raises: a run
    whose text cannot be read gets the skills in the platform's fixed order.
    """
    try:
        text = _content_text(getattr(ctx, "prompt", None))
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
        logger.debug("could not read the run's prompt from RunContext", exc_info=True)
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


# ── the harness's own skills ─────────────────────────────────


def _native_skills(agent: Any) -> Optional[bool]:
    """Whether the Agent was built with the harness's ``Skills`` capability.

    None: it was not. True: it was, and every one reports its loads (built with
    ``decimalai.pydantic_ai_harness.skills``). False: a plain ``Skills``, whose loads
    nobody reports.

    That capability is a skill loader of its own: a catalog in the instructions and
    ``load_capability`` for the bodies. DecimalAI's list and ``load_skill`` beside it
    would be a second catalog of the same skills. Read off the Agent's capability
    tree; a ``Skills`` a capability FUNCTION returns per run is not visible here
    (`decimalai.pydantic_ai_harness` warns once if that run also got DecimalAI's
    list). Never raises: when the answer is unknown, DecimalAI's skills stay on.
    """
    try:
        from . import pydantic_ai_harness as harness

        found: list = []
        agent.root_capability.apply(
            lambda capability: found.append(capability) if harness.is_skills(capability) else None)
        if not found:
            return None
        return all(harness.is_observed(capability) for capability in found)
    except Exception:  # noqa: BLE001
        logger.debug("native-skills check failed (non-fatal)", exc_info=True)
        return None


def _model_name(ctx: Any) -> Optional[str]:
    """The name of the model a run calls, or None."""
    return getattr(getattr(ctx, "model", None), "model_name", None)


def _record_in_history(ctx: Any, agent_name: str) -> Optional[ConversationRecord]:
    """The record an earlier run of this conversation kept in its request, or None."""
    from pydantic_ai.messages import ModelRequest

    run_id = getattr(ctx, "run_id", None)
    for message in reversed(list(getattr(ctx, "messages", None) or [])):
        if not isinstance(message, ModelRequest) or getattr(message, "run_id", None) == run_id:
            continue
        kept = (getattr(message, "metadata", None) or {}).get(_HISTORY_KEY)
        if isinstance(kept, dict) and agent_name in kept:
            return ConversationRecord.from_state(kept[agent_name], agent_name)
    return None


def _keep_in_history(ctx: Any, agent_name: str, record: ConversationRecord) -> None:
    """Put the record in this run's first request, where the next run's history has it."""
    from pydantic_ai.messages import ModelRequest

    run_id = getattr(ctx, "run_id", None)
    if run_id is None:
        return
    for message in getattr(ctx, "messages", None) or []:
        if isinstance(message, ModelRequest) and getattr(message, "run_id", None) == run_id:
            metadata = dict(message.metadata or {})
            kept = dict(metadata.get(_HISTORY_KEY) or {})
            if agent_name not in kept:
                kept[agent_name] = record.to_state(agent_name)
                metadata[_HISTORY_KEY] = kept
                message.metadata = metadata
            return


def _name_framework_run_span(run_id: Optional[str], agent_name: str) -> None:
    """Name Pydantic AI's own run span for the DecimalAI agent, when it is open here.

    With Pydantic AI's instrumentation on (``Agent.instrument_all()``), its ``invoke_agent``
    span opens before any capability's ``wrap_run``, so it is the trace's root and the
    DecimalAI run span is its child. The exporter files a trace under its root's
    ``decimal.agent_name``, which that span took from the context when it started:
    ``init(agent_name=...)``'s name, or none. Matched on the run id Pydantic AI puts on the
    span (``gen_ai.agent.call.id``), so any other span keeps its name. Never raises.
    """
    if not run_id:
        return
    try:
        from opentelemetry import trace as trace_api

        from .otel import _DECIMAL_AGENT_NAME

        span = trace_api.get_current_span()
        attributes = getattr(span, "attributes", None) or {}
        if span.is_recording() and attributes.get("gen_ai.agent.call.id") == run_id:
            span.set_attribute(_DECIMAL_AGENT_NAME, agent_name)
    except Exception:  # noqa: BLE001 — naming must never cost the run
        logger.debug("could not name Pydantic AI's run span", exc_info=True)


# ── the capability ───────────────────────────────────────────


class _PydanticAIBinding(AgentBinding):
    """Everything of `PydanticAIAgentConfig` except Pydantic AI's base class.

    Kept apart so this module imports without Pydantic AI installed: the class
    `agent_config()` returns mixes in ``AbstractCapability`` when it is first needed.
    """

    _log = logger
    _VALUES_HINT = "Pass them as the run's deps=..., as attributes or dict keys."

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str) -> None:
        super().__init__(prompt, fallback=fallback, on_unavailable=on_unavailable)
        # Unique per agent in a run; Pydantic AI reserves ":" in capability ids.
        self.id = "decimalai-" + self.name.replace(":", "-")
        self._toolset = self._build_toolset()

    def __repr__(self) -> str:
        return (
            f"PydanticAIAgentConfig(name={self.name!r}, prompt_version={self.prompt_version!r}, "
            f"skills={self.skill_names!r})"
        )

    # ── what Pydantic AI asks a capability for ───────────────
    def get_instructions(self) -> Any:
        async def decimalai_instructions(ctx: Any) -> str:
            """The agent's prompt, filled, then its skill list. Pydantic AI calls it."""
            agent = getattr(ctx, "agent", None)
            run_key = getattr(ctx, "run_id", None)
            state = await self._prepare_run(ctx, agent, run_key)
            if state["text"] is None:
                state["text"] = self._compose(ctx, agent, run_key, state)
            return state["text"]

        return decimalai_instructions

    def get_toolset(self) -> Any:
        return self._toolset

    async def prepare_tools(self, ctx: Any, tool_defs: List[Any]) -> List[Any]:
        """Show ``load_skill`` only beside a list, naming the listed skills."""
        import dataclasses

        from .skill_router import LOAD_SKILL_TOOL_NAME

        state = await self._prepare_run(ctx, getattr(ctx, "agent", None), getattr(ctx, "run_id", None))
        show = bool(state["names"]) and state["tool"]
        names = list(state.get("tool_names") or [])
        prepared = []
        for tool_def in tool_defs:
            if getattr(tool_def, "name", None) == LOAD_SKILL_TOOL_NAME:
                if not show:
                    continue
                tool_def = dataclasses.replace(tool_def, parameters_json_schema=self._tool_schema(names))
            prepared.append(tool_def)
        return prepared

    async def before_model_request(self, ctx: Any, request_context: Any) -> Any:
        """On Claude, keep what this run showed in its request, for the conversation's next run."""
        if is_claude(_model_name(ctx)):
            state = self._run_state(getattr(ctx, "run_id", None))
            record = state.get("conversation") if state is not None else None
            if record is not None:
                _keep_in_history(ctx, self.name, record)
        return request_context

    async def wrap_run(self, ctx: Any, *, handler: Any) -> Any:
        """One DecimalAI run span around the run, named for the DecimalAI agent."""
        from .otel import agent_run

        run_id = getattr(ctx, "run_id", None)
        _name_framework_run_span(run_id, self.name)
        token = _RUN.set(run_id)
        try:
            with agent_run(self.name):
                return await handler()
        finally:
            _RUN.reset(token)

    # ── the seams AgentBinding asks for ─────────────────────
    def _router(self) -> Any:
        return _get_skill_router()

    def _run_key(self) -> Optional[str]:
        return _RUN.get()

    def _message(self, context: Any, agent: Any) -> Optional[str]:
        return _run_query(context)

    def _values(self, context: Any) -> Any:
        return getattr(context, "deps", None)

    def _skills_blocked(self, agent: Any, *, tool: bool = True, inject: bool = False) -> Optional[str]:
        base = super()._skills_blocked(agent, tool=tool, inject=inject)
        if base:
            return base
        native = _native_skills(agent) if agent is not None else None
        if native is None:
            return None
        reported = ("Each skill the model loads through it is reported as an activation."
                    if native else
                    "Its loads are not reported: build it with "
                    "decimalai.pydantic_ai_harness.skills(...) to report them.")
        return ("it loads skills with the Pydantic AI harness's Skills capability, a catalog "
                f"of its own, so DecimalAI adds neither its skill list nor load_skill. {reported}")

    def _conversation(self, context: Any, agent: Any) -> Optional[ConversationRecord]:
        if not is_claude(_model_name(context)):
            return None
        return _record_in_history(context, self.name)

    def _record(self, **fields: Any) -> bool:
        from ._agent import _record_turn

        if fields.get("offered"):
            _note_menu_shown(_RUN.get())
        return _record_turn(fields)

    def _serve_load(self, name: str) -> str:
        from ._agent import _serve_and_record

        return _serve_and_record(self._router(), name, scope=self._run_key(), agent_name=self.name)

    def _build_toolset(self) -> Any:
        from pydantic_ai.toolsets import FunctionToolset

        from .skill_router import LOAD_SKILL_TOOL_DESCRIPTION, LOAD_SKILL_TOOL_NAME

        # No RunContext parameter: the run is `_RUN`, which `wrap_run` sets for the
        # whole run, tool calls included.
        async def load_skill(name: str) -> str:
            return await self._load(str(name or "").strip())

        toolset = FunctionToolset()
        toolset.add_function(
            load_skill, takes_ctx=False, name=LOAD_SKILL_TOOL_NAME,
            description=LOAD_SKILL_TOOL_DESCRIPTION,
        )
        return toolset


_config_class_cache: Any = None
_config_class_lock = threading.Lock()


def _config_class() -> Any:
    """`PydanticAIAgentConfig`, built once: the binding with Pydantic AI's capability base."""
    global _config_class_cache
    with _config_class_lock:
        if _config_class_cache is None:
            from pydantic_ai.capabilities import AbstractCapability

            class PydanticAIAgentConfig(_PydanticAIBinding, AbstractCapability):
                """One agent's prompt and skills in DecimalAI, as a Pydantic AI capability.

                Made by :func:`agent_config`. Pass it in ``Agent(capabilities=[...])``::

                    support = agent_config("support", fallback=BASE_INSTRUCTIONS)
                    agent = Agent(model, name=support.name, tools=[lookup_order],
                                  capabilities=[support])

                Attributes:
                    name: The agent's canonical DecimalAI name.
                """

                # One config is one object: Pydantic AI's base is a dataclass, whose
                # generated equality would make two agents' configs compare equal.
                __eq__ = object.__eq__
                __hash__ = object.__hash__

            PydanticAIAgentConfig.__module__ = __name__
            _config_class_cache = PydanticAIAgentConfig
        return _config_class_cache


def __getattr__(name: str) -> Any:
    if name == "PydanticAIAgentConfig":
        return _config_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def agent_config(
    agent_name: str,
    *,
    fallback: Optional[str] = None,
    on_unavailable: str = "continue",
) -> Any:
    """Pull an agent's prompt and skills from DecimalAI, as a Pydantic AI capability.

    ::

        from pydantic_ai import Agent
        from decimalai.pydantic_ai import agent_config

        support = agent_config("support", fallback=BASE_INSTRUCTIONS)
        agent = Agent("openai:gpt-5.4-mini", name=support.name,
                      tools=[lookup_order], capabilities=[support])
        result = await agent.run(question, deps=customer)

    Reads the agent's saved prompt here (so a misspelled name fails now) and keeps
    a local copy, re-read in the background at most once a minute. Each run is
    routed on its own prompt: the first run before its first model request, later
    runs from a local copy while their decision is recorded in the background, and
    every run of an agent with more skills than fit the list. ``{{variables}}``
    are filled from the run's ``deps`` (attributes or dict keys). Initializes the SDK
    from ``DECIMAL_API_KEY`` when ``decimalai.init()`` has not been called.

    With the ``load_skill`` tool switched off (``DECIMALAI_LOAD_SKILL_TOOL=0`` or
    ``init(load_skill_tool=False)``), every run is routed and the text of the skills
    picked for it goes into the instructions instead (``inject_skill_body``).

    Args:
        agent_name: The agent's name in DecimalAI.
        fallback: Instructions to use when the agent has no saved prompt, or when
            DecimalAI cannot be reached at startup. Without one, an unreachable
            DecimalAI raises here rather than starting an agent with no instructions.
        on_unavailable: ``"continue"`` (default) runs without skills when DecimalAI
            never returned them, and the run's record says so; ``"fail"`` raises
            before the model request instead.

    Raises:
        ImportError: Pydantic AI is not installed.
        AgentNotFoundError: DecimalAI has no agent by this name.
        ValueError: ``on_unavailable`` is neither ``"continue"`` nor ``"fail"``.
    """
    if on_unavailable not in ("continue", "fail"):
        raise ValueError("on_unavailable must be 'continue' or 'fail'")
    _require_pydantic_ai()
    from ._agent import _ensure_initialized, read_prompt

    _ensure_initialized()
    prompt = read_prompt(agent_name, fallback=fallback)
    return _config_class()(prompt, fallback=fallback, on_unavailable=on_unavailable)


def load_agent(*args: Any, **kwargs: Any) -> NoReturn:
    """Renamed in decimalai 0.21: :func:`agent_config`."""
    raise TypeError(
        "decimalai.pydantic_ai.load_agent() was renamed agent_config() in decimalai 0.21; "
        "it takes the same arguments and returns the same capability: "
        "from decimalai.pydantic_ai import agent_config"
    )
