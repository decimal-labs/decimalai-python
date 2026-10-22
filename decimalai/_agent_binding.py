"""What every framework's ``agent_config()`` shares: one agent's prompt and skills.

An :class:`AgentBinding` holds one DecimalAI agent's saved prompt and skill list
for one process, and serves them run by run:

- the prompt is a local copy, re-read in the background at most once every
  ``_PROMPT_REFRESH_SECONDS``; its ``{{variables}}`` are filled per run;
- every run is routed on its own message. The first run is routed before its
  model call. Later runs show the local copy of the list at once and record
  their own decision, ranked on their message, in the background. An agent whose
  list the platform cut to fit its caps is routed before every model call,
  because which skills fit depends on the message;
- the model reads a listed skill's full text with ``load_skill``. A load waits
  briefly for its run's own decision and never names another agent's;
- each run's record (prompt version, variable names, skills offered and
  delivered, routing id) goes wherever the framework's trace is built;
- a conversation keeps what its first run was shown. Claude rejects a
  conversation whose system prompt or tool definitions changed once a thinking
  block is in its history (HTTP 400, "bound to a different conversation"), so a
  framework that can name the conversation a run belongs to hands its record
  back (`ConversationRecord`) and the run reuses that system text and
  ``load_skill`` definition byte for byte. A newer saved prompt, a changed skill
  list or a re-picked cut list reaches the next conversation instead.

Nothing here knows a framework. A framework module subclasses
:class:`AgentBinding` and supplies the seams: how to name the current run, read
its message and its variables' values, where the record goes, which router
serves it, and what would make DecimalAI's skills misleading for one of its
agents. ``decimalai.openai_agents.agent_config`` is the first such subclass.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import functools
import logging
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("decimalai")

_TEMPLATE_PLACEHOLDER = re.compile(
    r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*\}\}"
)

#: How old the local copy of a prompt may get before a run refreshes it.
_PROMPT_REFRESH_SECONDS = 60.0

#: Runs whose per-run state a binding keeps (instructions text, loaded skills).
_MAX_CONFIG_RUNS = 256

#: How long a load waits for its run's background routing decision.
_LOAD_WAITS_FOR_ROUTING_S = 2.0

#: The version of `ConversationRecord.to_state`'s shape.
_CONVERSATION_STATE_VERSION = 1

_config_pool: Optional[ThreadPoolExecutor] = None
_config_pool_lock = threading.Lock()


def _running_asyncio_loop() -> Optional[asyncio.AbstractEventLoop]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


async def _in_thread(func: Any, *args: Any) -> Any:
    """``func(*args)`` in a worker thread, with this context's variables.

    asyncio's thread when an asyncio loop is running; otherwise anyio's, which
    is how an async client runs under trio (the Anthropic SDK is built on anyio).
    """
    if _running_asyncio_loop() is not None:
        return await asyncio.to_thread(func, *args)
    import anyio.to_thread

    call = functools.partial(contextvars.copy_context().run, func, *args)
    # A cancelled run returns at once, as under asyncio; the thread finishes alone.
    return await anyio.to_thread.run_sync(call, **{_anyio_abandon_keyword(): True})


@functools.lru_cache(maxsize=None)
def _anyio_abandon_keyword() -> str:
    """``abandon_on_cancel`` (anyio >= 4.1), or its older name ``cancellable``."""
    import inspect

    import anyio.to_thread

    parameters = inspect.signature(anyio.to_thread.run_sync).parameters
    return "abandon_on_cancel" if "abandon_on_cancel" in parameters else "cancellable"


async def _wait_without_cancelling(pending: Any, timeout: float) -> None:
    """Wait up to ``timeout`` for a background future: never cancelling it, holding no thread."""
    import anyio

    deadline = time.monotonic() + timeout
    while not pending.done() and time.monotonic() < deadline:
        await anyio.sleep(0.02)


def is_claude(model_name: Any) -> bool:
    """Whether a model name is Claude's, in any provider's spelling.

    ``claude-haiku-5-5``, ``anthropic/claude-…`` (LiteLLM), ``anthropic.claude-…``
    (Bedrock), ``claude-…@…`` (Vertex). Claude binds a thinking block to the system
    prompt and tools it was made under, so a Claude conversation must keep both.
    """
    return isinstance(model_name, str) and "claude" in model_name.lower()


class ConversationRecord:
    """What one conversation keeps from its first run: DecimalAI's text and ``load_skill``.

    Attributes:
        text: The system text DecimalAI composed: the prompt, filled, then the list.
        names: The skills that list offers.
        tool: Whether ``load_skill`` was given to the model.
        tool_names: The names its ``name`` argument is an enum of (empty: any name).
        cut: The list was cut to fit; later loads name ``routing_id``'s decision.
        routing_id: The decision a cut list came from.
        prompt: The prompt record (version, hash, variable names).
        delivered: Skills whose text was pasted into ``text``.
        unavailable: Why the list is missing, when it is.
        pasted: Skill text was pasted in (``load_skill`` switched off).
        menu: The skill-list part at the end of ``text`` ("" without one), so a
            framework that sends the prompt and the list as separate pieces (the
            Anthropic module's system blocks) rebuilds them exactly.
        shown: Every skill the conversation has shown the model, and the routing
            decision that first showed it. A load names that decision: the
            platform serves a public-registry skill only to a load whose decision
            offered it.
        experiment: The first run was routed while the agent ran an experiment, so the
            conversation is one unit of it: later runs and their loads name ``routing_id``'s
            decision (its arm) and are not routed again.
    """

    __slots__ = ("text", "names", "tool", "tool_names", "cut", "routing_id", "prompt",
                 "delivered", "unavailable", "pasted", "menu", "shown", "experiment")

    def __init__(self, *, text: str, names: List[str], tool: bool, tool_names: List[str],
                 cut: bool, routing_id: Optional[str], prompt: Optional[Dict[str, Any]],
                 delivered: List[str], unavailable: Optional[str], pasted: bool,
                 menu: str = "", shown: Optional[Dict[str, Optional[str]]] = None,
                 experiment: bool = False) -> None:
        self.text = text
        self.names = names
        self.tool = tool
        self.tool_names = tool_names
        self.cut = cut
        self.routing_id = routing_id
        self.prompt = prompt
        self.delivered = delivered
        self.unavailable = unavailable
        self.pasted = pasted
        self.menu = menu if text.endswith(menu) else ""
        self.shown: Dict[str, Optional[str]] = (
            dict(shown) if shown is not None else {name: routing_id for name in names})
        self.experiment = bool(experiment)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ConversationRecord) and all(
            getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)

    def to_state(self, agent: str) -> Dict[str, Any]:
        """JSON-ready, for whatever the framework keeps the conversation in."""
        return copy.deepcopy({
            "version": _CONVERSATION_STATE_VERSION, "agent": agent, "system": self.text,
            "skills": self.names, "load_skill": self.tool, "load_skill_names": self.tool_names,
            "skills_cut": self.cut, "routing_id": self.routing_id, "prompt": self.prompt,
            "delivered": self.delivered, "skills_unavailable": self.unavailable,
            "pasted": self.pasted, "skills_menu": self.menu, "skills_shown": self.shown,
            "experiment": self.experiment,
        })

    @classmethod
    def from_state(cls, state: Any, agent: str) -> Optional["ConversationRecord"]:
        """The record in ``state``, or None when it is another agent's or another shape's."""
        if (not isinstance(state, Mapping) or state.get("version") != _CONVERSATION_STATE_VERSION
                or state.get("agent") != agent or not isinstance(state.get("system"), str)):
            return None
        return cls(
            text=state["system"], names=_clean_names(state.get("skills")),
            tool=bool(state.get("load_skill")), tool_names=_clean_names(state.get("load_skill_names")),
            cut=bool(state.get("skills_cut")), routing_id=state.get("routing_id"),
            prompt=copy.deepcopy(state.get("prompt")), delivered=_clean_names(state.get("delivered")),
            unavailable=state.get("skills_unavailable"), pasted=bool(state.get("pasted")),
            menu=state.get("skills_menu") if isinstance(state.get("skills_menu"), str) else "",
            shown=({str(k): (v if isinstance(v, str) else None) for k, v in state["skills_shown"].items()}
                   if isinstance(state.get("skills_shown"), Mapping) else None),
            experiment=bool(state.get("experiment")),
        )


def new_skills_block(prefix: str, names: List[str]) -> str:
    """The rows of ``prefix``'s skill table naming ``names``, as a block for the user turn.

    For a later message of a conversation whose list is cut: the skills its own decision picked
    that the conversation has not shown. They go in a user turn, never the system text, which a
    conversation keeps.
    """
    rows = []
    for line in prefix.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and cells and cells[0] in names:
            rows.append(line.strip())
    if not rows:
        rows = [f"| {name} | |" for name in names]
    return ("More skills you can load with load_skill for this message:\n\n"
            "| Skill | Description |\n|-------|-------------|\n" + "\n".join(rows))


class MissingPromptVariable(KeyError):
    """The saved prompt has a ``{{variable}}`` the run does not supply a value for."""

    def __str__(self) -> str:  # KeyError would quote the whole message
        return str(self.args[0]) if self.args else ""


def template_variables(text: Optional[str]) -> List[str]:
    """The ``{{name}}`` / ``{{a.b}}`` variables in a prompt, sorted and unique."""
    return sorted(set(_TEMPLATE_PLACEHOLDER.findall(text or "")))


def _fill_template(text: str, values: Any, *, where: str, how: str) -> str:
    """Fill variables from ``values``' attributes or keys, dotted paths included.

    A missing value raises `MissingPromptVariable` naming every gap, before any
    model call: a half-filled prompt is a prompt nobody wrote. ``how`` tells the
    reader where their framework takes the values from.
    """
    missing: List[str] = []

    def lookup(match: "re.Match[str]") -> str:
        current = values
        for part in match.group(1).split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            elif current is not None and not isinstance(current, Mapping) and hasattr(current, part):
                current = getattr(current, part)
            else:
                missing.append(match.group(1))
                return match.group(0)
        return str(current)

    filled = _TEMPLATE_PLACEHOLDER.sub(lookup, text)
    if missing:
        raise MissingPromptVariable(f"{where} needs {', '.join(sorted(set(missing)))}. {how}")
    return filled


def _in_background(fn: Any, *args: Any) -> Any:
    """Run ``fn(*args)`` on a small worker pool, in a copy of this context.

    The copy carries the current run (an OpenAI trace, an OpenTelemetry span), so
    the run key and the run record still name the run that scheduled the work.
    """
    global _config_pool
    for _attempt in range(2):
        with _config_pool_lock:
            if _config_pool is None:
                _config_pool = ThreadPoolExecutor(
                    max_workers=4, thread_name_prefix="decimalai-agent-config",
                )
            pool = _config_pool
        try:
            return pool.submit(contextvars.copy_context().run, fn, *args)
        except RuntimeError:
            # Shut down (interpreter exit, or someone shut it): replace it once,
            # and never fail a run over background bookkeeping.
            with _config_pool_lock:
                if _config_pool is pool:
                    _config_pool = None
    logger.debug("background work skipped: no worker pool available")
    return None


def _clean_names(names: Any) -> List[str]:
    """Keep only non-blank strings."""
    if not isinstance(names, (list, tuple, set)):
        return []
    return [n for n in names if isinstance(n, str) and n.strip()]


def attach_prompt_records(spans: List[Any], records: Any) -> None:
    """Put each `agent_config()` prompt record on its agent's span, as attributes.

    The backend stores span attributes as they come, so a trace says which
    prompt version a run used, with which variable NAMES (never values:
    those are often personal data), without a column of its own. An agent with
    no span of its own gets a span named ``decimalai.prompt``, under the run's root
    span and spanning the run: the backend refuses a span with no start or end, and
    that refusal used to lose the whole trace whenever the framework named its run
    otherwise (a LangChain graph is "LangGraph" unless named).
    """
    if not isinstance(records, dict) or not records:
        return
    for agent_name, record in records.items():
        attributes = {
            "decimal.agent": record.get("decimal_agent"),
            "decimal.prompt.source": record.get("source"),
            "decimal.prompt.version": record.get("version"),
            "decimal.prompt.content_hash": record.get("content_hash"),
            "decimal.prompt.variables": list(record.get("variables") or []),
        }
        if record.get("skills_unavailable"):
            attributes["decimal.skills.unavailable"] = record["skills_unavailable"]
        from .schema.common import SpanType
        from .schema.trace import TraceSpan

        target = next(
            (s for s in spans if s.span_type == SpanType.AGENT and s.name == agent_name),
            None,
        )
        if target is None:
            timed = [s for s in spans if getattr(s, "started_at", None) is not None]
            root = next((s for s in timed if getattr(s, "parent_span_id", None) is None), None)
            now = datetime.now(timezone.utc)
            spans.append(TraceSpan(
                span_type=SpanType.OTHER, name="decimalai.prompt",
                parent_span_id=root.id if root is not None else None,
                started_at=min((s.started_at for s in timed), default=now),
                ended_at=max((s.ended_at or s.started_at for s in timed), default=now),
                attributes={**attributes, "decimal.prompt.agent": agent_name},
            ))
        else:
            target.attributes.update(attributes)


class AgentBinding:
    """One DecimalAI agent's prompt and skills, served per run.

    A framework module subclasses this and overrides the seams below (the
    methods that raise ``NotImplementedError`` or say "override"). The rest is
    the shared behaviour described in the module docstring.
    """

    #: Where this binding's warnings go: the framework module's own logger.
    _log = logger

    #: How a caller passes variables' values, said in `MissingPromptVariable`.
    _VALUES_HINT = "Pass them as variables=..., as attributes or dict keys."

    #: Where the agent's always-include skills' text goes: after the skill list in the system
    #: text (True), or somewhere the framework module puts it itself (LangChain: a tagged message).
    _ALWAYS_IN_SYSTEM = True

    def __init__(self, prompt: Any, *, fallback: Optional[str], on_unavailable: str) -> None:
        self.name: str = prompt.agent_name
        self._fallback = fallback
        self._on_unavailable = on_unavailable
        self._lock = threading.Lock()
        self._prompt = prompt
        self._prompt_read_at = time.monotonic()
        self._prompt_refreshing = False
        self._menu_prefix = ""
        self._menu_names: List[str] = []
        self._menu_unavailable: Optional[str] = None
        self._menu_loaded = False
        #: The latest decision this agent was routed with (any run's).
        self._menu_routing_id: Optional[str] = None
        #: The platform cut this agent's menu to fit its caps: which rows make
        #: it depends on the message, so every run is routed before its call.
        self._menu_truncated = False
        self._runs: "OrderedDict[Tuple[Any, Any], Dict[str, Any]]" = OrderedDict()
        self._warned: Set[Tuple[str, str]] = set()
        #: No API key and no ``init()``: the project's skills and the registry's,
        #: served by `decimalai._keyless`, and nothing recorded (no run is sent).
        self._keyless = bool(getattr(prompt, "keyless", False))
        #: The agent's running experiment, as the last routing answer named it (skills-first
        #: Phase 3), or None. While one runs, every run routes first, so its arm is its own.
        self._experiment: Optional[Dict[str, Any]] = None
        #: The agent's always-include skills (D-C), as the last routing answer listed them, with
        #: their text: fetched again only when the list or a version changes.
        self._always: Dict[str, Any] = {"key": None, "texts": [], "versions": []}

    # ── the seams a framework supplies ───────────────────────
    def _router(self) -> Any:
        """The SkillRouter that routes and serves this binding, or None. Override."""
        raise NotImplementedError

    def _run_key(self) -> Optional[str]:
        """The key naming the active run, or None outside one. Override."""
        raise NotImplementedError

    def _message(self, context: Any, agent: Any) -> Optional[str]:
        """The run's latest user message, which routing ranks the skills on. Override."""
        raise NotImplementedError

    def _values(self, context: Any) -> Any:
        """Where this run's variables' values come from. Override."""
        return context

    def _agent_label(self, agent: Any) -> Optional[str]:
        """The framework's own name for ``agent``, used in warnings and records."""
        return getattr(agent, "name", None)

    def _skills_blocked(self, agent: Any, *, tool: bool = True, inject: bool = False) -> Optional[str]:
        """Why DecimalAI's skills cannot work on ``agent``, or None when they can."""
        if not tool and not inject:
            return ("the load_skill tool is switched off (DECIMALAI_LOAD_SKILL_TOOL=0 or "
                    "init(load_skill_tool=False)) and so is inject_skill_body, so nothing "
                    "could give the model a skill's text.")
        return None

    def _record(self, **fields: Any) -> bool:
        """Put one piece of the run's record wherever its trace is built. Override."""
        return False

    def _note_pending(self, run_key: Optional[str], pending: Any) -> None:
        """Keep the run's background routing request where its trace can wait for it."""

    def _stamp_routing(self, run_key: Optional[str], routing_id: str) -> None:
        """Name a background decision on its run's record, when the run is still open."""
        self._record(routing_id=routing_id)

    def _on_compose(self, agent: Any) -> None:
        """Called once per composed run, before its record is made."""

    def _on_menu(self, names: List[str]) -> None:
        """The names ``load_skill`` should accept changed (empty: any name)."""

    def _serve_load(self, name: str) -> str:
        """Fetch one skill's text for the model, recording a delivery. Override."""
        raise NotImplementedError

    def _conversation(self, context: Any, agent: Any) -> Optional[ConversationRecord]:
        """The record of the conversation this run continues, or None: compose afresh.

        A framework that can name a run's conversation, and knows the model needs
        it kept (`is_claude`), returns what that conversation's first run kept.
        """
        return None

    def _keep_conversation(self, context: Any, agent: Any, run_key: Optional[str],
                           record: ConversationRecord) -> None:
        """A run composed afresh: keep its record where its conversation's later runs look."""

    def _routed_continuation(self, context: Any, agent: Any) -> Any:
        """A later run's own routing result, when the framework routed its message already.

        Only for a conversation whose list is cut. None (the default): the run is
        not routed again and names the conversation's decision. A framework that
        can show the model this message's new skills without touching the system
        text (the Anthropic module's ``run.message``) routes before the run and
        returns `_route`'s result; the run's state then carries it as ``routed``,
        and recording the decision is that framework's.
        """
        return None

    # ── keyed or keyless ────────────────────────────────────
    def _skill_router(self) -> Any:
        """The router this binding routes and serves through: the framework's, or keyless."""
        if self._keyless:
            from ._keyless import keyless_router

            return keyless_router()
        return self._router()

    def _serve(self, name: str) -> str:
        """One load: the framework's `_serve_load`, or the keyless router's (nothing recorded)."""
        if self._keyless:
            return self._skill_router().load_skill(name, scope=self._run_key(), agent_name=self.name)
        return self._serve_load(name)

    def _record_run(self, **fields: Any) -> bool:
        """`_record`, except keyless: a keyless run is sent nowhere, so nothing is kept for it."""
        return False if self._keyless else self._record(**fields)

    # ── what the agent currently gets ────────────────────────
    @property
    def prompt_version(self) -> Optional[int]:
        """The saved prompt version in use, or None for the fallback text."""
        return getattr(self._prompt, "version_number", None)

    @property
    def prompt_hash(self) -> Optional[str]:
        """The content hash of the saved prompt in use, or None for fallback text."""
        return getattr(self._prompt, "content_hash", None)

    @property
    def prompt_source(self) -> str:
        """Where the prompt in use came from.

        ``"decimalai"`` (read from DecimalAI), ``"decimalai (cached copy)"`` (DecimalAI
        was unreachable at the last read, so the copy read before it),
        ``"fallback (unreachable)"`` (DecimalAI unreachable and nothing cached: the
        ``fallback=`` text), ``"fallback (none saved)"`` (the agent has no saved
        prompt: the ``fallback=`` text), or ``"fallback (no key)"`` (no API key is
        set, so DecimalAI was not asked: the ``fallback=`` text). Each fallback value
        applies with or without a ``fallback=``; without one there is no prompt
        text, and the instructions are the skill list alone.
        """
        with self._lock:
            prompt = self._prompt
        return self._source_of(prompt)

    @staticmethod
    def _source_of(prompt: Any) -> str:
        if getattr(prompt, "keyless", False):
            return "fallback (no key)"
        if getattr(prompt, "system_prompt", None) is None:
            return "fallback (none saved)"
        if getattr(prompt, "is_fallback", False) and getattr(prompt, "version_number", None) is None:
            return "fallback (unreachable)"
        if getattr(prompt, "is_fallback", False):
            return "decimalai (cached copy)"
        return "decimalai"

    @property
    def variables(self) -> List[str]:
        """The variables (``{{name}}``) the prompt in use needs a value for on each run."""
        return template_variables(self._template())

    @property
    def skill_names(self) -> List[str]:
        """The skills on the agent's skill list, in its order."""
        with self._lock:
            return list(self._menu_names)

    def _template(self) -> str:
        text = getattr(self._prompt, "system_prompt", None)
        return text if text is not None else (self._fallback or "")

    # ── one run ─────────────────────────────────────────────
    def _begin_run(self, context: Any, agent: Any, run_key: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """The cheap half of preparing a run: (finished state, or None and a plan)."""
        key = (run_key, self._agent_label(agent))
        if run_key is not None:
            with self._lock:
                state = self._runs.get(key)
            if state is not None:
                return state, {}
        try:
            record = self._conversation(context, agent)
        except Exception:  # noqa: BLE001 — a record we cannot read means compose afresh
            self._log.debug("could not read the conversation's record (non-fatal)", exc_info=True)
            record = None
        if record is not None:
            return self._continue_conversation(context, agent, run_key, key, record), {}
        self._refresh_prompt_if_stale()
        from ._config import _disabled_by_init

        tool, inject = self._channels()
        # The kill switch: a later init(enabled=False) stops routing and skills
        # for agents built before it. The prompt read earlier still runs.
        switched_off = _disabled_by_init()
        blocked = None if switched_off else self._skills_blocked(agent, tool=tool, inject=inject)
        if blocked:
            self._warn_once(agent, f"DecimalAI skills are off for this agent: {blocked}")
        if switched_off or blocked:
            # Neither a list nor a loader: a list the model cannot load from, or
            # a loader beside one that already serves the agent, only misleads.
            state = {"query": None, "route_now": True, "routing_id": None, "prefix": "",
                     "names": [], "unavailable": None, "text": None, "loaded": set(),
                     "tool": False, "tool_names": [], "cut": False, "pasted": False,
                     "delivered": [], "delivered_versions": []}
            if run_key is not None:
                with self._lock:
                    self._runs[key] = state
            return state, {}
        query = self._message(context, agent)
        with self._lock:
            # Pasted skill text depends on the message, so such a run is always
            # routed for its own.
            route_now = (inject or not self._menu_loaded or (self._menu_truncated and query is not None)
                         or self._experiment is not None)
            unavailable = self._menu_unavailable
        return None, {"key": key, "query": query, "tool": tool, "inject": inject,
                      "route_now": route_now, "unavailable": unavailable}

    def _finish_run(self, run_key: Optional[str], plan: Dict[str, Any], routed_result: Any) -> Dict[str, Any]:
        """The rest of preparing a run, from the plan and its routing result (or None)."""
        inject, tool, query = plan["inject"], plan["tool"], plan["query"]
        unavailable = plan["unavailable"]
        routing_id: Optional[str] = None
        routed: Optional[Tuple[str, List[str]]] = None
        delivered: List[str] = []
        versions: List[Dict[str, str]] = []
        run_prompt = None
        if routed_result is not None:
            prefix, offered, routing_id, truncated, delivered, versions, run_prompt = routed_result
            if inject:
                # This message's skill text is no other run's local copy.
                unavailable = None if (prefix or routing_id) else (
                    "DecimalAI did not return this agent's skills")
            else:
                self._store_menu(prefix, offered, routing_id, truncated)
                with self._lock:
                    unavailable = self._menu_unavailable
            if prefix or routing_id:
                routed = (prefix, list(offered))  # this run's own menu
        if routed is None:
            delivered, versions, run_prompt = [], [], None
        with self._lock:
            experiment = self._experiment is not None and routed is not None
            # Otherwise the local copy: the last good menu, whatever was asked.
            menu = routed or (("", []) if inject else (self._menu_prefix, list(self._menu_names)))
            state = {
                "query": query, "route_now": plan["route_now"], "routing_id": routing_id,
                "prefix": menu[0], "names": menu[1], "unavailable": unavailable,
                "text": None, "tool": tool,
                # The load_skill names for the whole run: a decision landing mid-run
                # must not change a tool the model already saw (one Claude run
                # replays its thinking between tool calls). A cut list names none,
                # and so does a run in an experiment: the platform checks the name
                # against the run's own decision (a held-out skill is refused).
                "tool_names": [] if (self._menu_truncated or experiment) else list(menu[1]),
                # Skills-first Phase 3: a prompt split's prompt for this run, and whether the
                # run was routed for its own arm.
                "run_prompt": run_prompt, "experiment": experiment,
                "cut": bool(self._menu_truncated), "pasted": inject,
                # A skill whose text is already in the instructions is not sent again.
                "loaded": set(delivered) if tool else set(),
                "delivered": delivered, "delivered_versions": versions,
                # Gets the agent's always-include skills' text (D-C); the kill switch and a
                # blocked agent's early answer in `_begin_run` do not.
                "always_on": True,
            }
            if run_key is not None:
                self._runs[plan["key"]] = state
                while len(self._runs) > _MAX_CONFIG_RUNS:
                    self._runs.popitem(last=False)
        return state

    async def _prepare_run(self, context: Any, agent: Any, run_key: Optional[str]) -> Dict[str, Any]:
        """Once per run and agent: the menu the run shows, routed now when it must be.

        Whichever comes first does it: a framework may ask for the tools before
        the instructions, and both need the same answer.
        """
        state, plan = self._begin_run(context, agent, run_key)
        if state is not None:
            return state
        routed_result = None
        if plan["route_now"]:
            # The first run, or an agent whose menu depends on the message: this
            # run's own decision, ranked on its message, before the model call.
            try:
                routed_result = await _in_thread(
                    self._route, run_key, plan["query"], plan["inject"], "this_run",
                )
            except Exception:
                self._log.debug("skill routing failed (non-fatal)", exc_info=True)
        return self._finish_run(run_key, plan, routed_result)

    def _prepare_run_sync(self, context: Any, agent: Any, run_key: Optional[str]) -> Dict[str, Any]:
        """`_prepare_run` for a caller that is not on an event loop."""
        state, plan = self._begin_run(context, agent, run_key)
        if state is not None:
            return state
        routed_result = None
        if plan["route_now"]:
            try:
                routed_result = self._route(run_key, plan["query"], plan["inject"], "this_run")
            except Exception:
                self._log.debug("skill routing failed (non-fatal)", exc_info=True)
        return self._finish_run(run_key, plan, routed_result)

    def _compose(self, context: Any, agent: Any, run_key: Optional[str], state: Dict[str, Any]) -> str:
        """The run's instructions: the prompt, filled, then its menu. Records the run."""
        if state["unavailable"] and self._on_unavailable == "fail":
            raise RuntimeError(
                f"DecimalAI skills for agent {self.name!r} are unavailable ({state['unavailable']}) "
                "and agent_config(on_unavailable='fail') was set."
            )
        with self._lock:
            prompt = self._prompt
        template = self._template()
        run_prompt = state.get("run_prompt")
        if run_prompt and isinstance(run_prompt.get("content"), str):
            # A prompt split's B arm: this run uses the version its decision named, and its trace
            # says so. The agent's own prompt (what manifests declare) is untouched.
            template = run_prompt["content"]
            prompt = SimpleNamespace(
                system_prompt=template, version_number=run_prompt.get("version"),
                content_hash=run_prompt.get("content_hash"), is_fallback=False, keyless=False,
            )
        text = template
        menu = ""
        if template_variables(template):
            version = getattr(prompt, "version_number", None)
            where = f"Prompt version {version} of {self.name!r}" if version else f"The fallback prompt of {self.name!r}"
            text = _fill_template(template, self._values(context), where=where, how=self._VALUES_HINT)
        if state["prefix"]:
            from .skill_router import LOAD_SKILL_PROMPT_HINT

            # The usage sentence names the load_skill tool, so only beside it.
            menu = f"{state['prefix']}\n{LOAD_SKILL_PROMPT_HINT}" if state["tool"] else state["prefix"]
        # The always-include skills (D-C): their full text in every run, but one this run's own
        # text already pasted. Part of the skill section, so a conversation keeps it with the list.
        always, always_versions = (
            self._always_now(skip=state["delivered"]) if state.get("always_on") else ([], []))
        state["always_now"] = always
        if always:
            if self._ALWAYS_IN_SYSTEM:
                block = "\n\n".join(section for _name, section in always)
                menu = f"{menu}\n\n{block}" if menu else block
            names = [name for name, _section in always]
            state["delivered"] = [*state["delivered"], *names]
            state["delivered_versions"] = [*state["delivered_versions"], *always_versions]
            if state["tool"]:
                # Its text is already in the run: load_skill answers so instead of sending it again.
                state["loaded"] = set(state["loaded"]) | set(names)
        if menu:
            text = f"{text}\n\n{menu}" if text else menu
        self._on_compose(agent)
        prompt_record = self._prompt_record(prompt, template, state["unavailable"], agent)
        self._record_run(
            agent=agent, system_prompt=text, offered=state["names"],
            routing_id=state["routing_id"], user_input=state["query"],
            prompt=prompt_record,
            delivered=state["delivered"], delivered_versions=state["delivered_versions"],
        )
        record = ConversationRecord(
            text=text, names=list(state["names"]), tool=bool(state["names"]) and bool(state["tool"]),
            tool_names=list(state.get("tool_names") or []), cut=bool(state.get("cut")),
            routing_id=state["routing_id"], prompt=copy.deepcopy(prompt_record),
            delivered=list(state["delivered"]), unavailable=state["unavailable"],
            pasted=bool(state.get("pasted")), menu=menu, experiment=bool(state.get("experiment")),
        )
        state["conversation"] = record
        try:
            self._keep_conversation(context, agent, run_key, record)
        except Exception:  # noqa: BLE001 — keeping a record must never cost the run
            self._log.debug("could not keep the conversation's record (non-fatal)", exc_info=True)
        if not state["route_now"]:
            # The local copy is this agent's whole menu, whatever was asked. The
            # run's own decision, ranked on its message, is recorded in the
            # background so the run never waits on it.
            pending = _in_background(self._route_run, run_key, state["names"], state["query"])
            if not self._keyless:
                self._note_pending(run_key, pending)
            state["pending"] = pending
        return text

    def _continue_conversation(self, context: Any, agent: Any, run_key: Optional[str],
                               key: Tuple[Any, Any], record: ConversationRecord) -> Dict[str, Any]:
        """A later run of a conversation: its system text and tool are the conversation's.

        A whole list: the message is routed in the background and the decision is
        recorded on this run when it offers the list shown. A cut list: the message
        is not routed again, and the run and its loads name the decision that
        produced the conversation's list, so a skill the model was shown is served.
        The kill switch leaves both in place (changing them would break the chat);
        ``load_skill`` then answers with an error.
        """
        from ._config import _disabled_by_init

        self._refresh_prompt_if_stale()  # for the next conversation
        query = self._message(context, agent)
        # A conversation in an experiment is one unit of it: every message keeps the first
        # decision (its arm), whatever the list's shape, and is not routed again.
        keeps_decision = record.cut or record.experiment
        routed = self._routed_continuation(context, agent) if (record.cut and not record.experiment) else None
        routing_id = record.routing_id if (keeps_decision and routed is None) else None
        state: Dict[str, Any] = {
            "query": query, "route_now": False, "routing_id": routing_id, "prefix": "",
            "names": list(record.names), "unavailable": record.unavailable, "text": record.text,
            "tool": record.tool, "tool_names": list(record.tool_names), "cut": record.cut,
            "pasted": record.pasted,
            # Skill text already in the system text is not sent again.
            "loaded": set(record.delivered), "delivered": [], "delivered_versions": [],
            "conversation": record,
        }
        if run_key is not None:
            with self._lock:
                self._runs[key] = state
                while len(self._runs) > _MAX_CONFIG_RUNS:
                    self._runs.popitem(last=False)
        if record.pasted:
            self._warn_once(agent, (
                "the load_skill tool is off, so skill text is pasted into the instructions; a "
                "conversation keeps its first message's text (Claude rejects a changed system "
                "prompt), so later messages get no new skill text."))
        self._on_compose(agent)
        self._record_run(
            agent=agent, system_prompt=record.text, offered=list(record.names), routing_id=routing_id,
            user_input=query, prompt=copy.deepcopy(record.prompt), delivered=[], delivered_versions=[],
        )
        if _disabled_by_init() or not record.names:
            return state
        if record.cut and not record.experiment and routed is not None:
            # Routed on its own message already; the router named that decision for
            # this run's loads, and the framework records it.
            state["routed"] = routed
        elif keeps_decision:
            router = self._skill_router()
            remember = getattr(router, "_remember_run_routing", None)
            if run_key is not None and routing_id and callable(remember):
                remember(run_key, routing_id)
        else:
            pending = _in_background(self._route_run, run_key, list(record.names), query)
            if not self._keyless:
                self._note_pending(run_key, pending)
            state["pending"] = pending
        return state

    def _prompt_record(self, prompt: Any, template: str, unavailable: Optional[str], agent: Any) -> Dict[str, Any]:
        return {
            "agent": self._agent_label(agent) or self.name,
            "decimal_agent": self.name,
            "source": self._source_of(prompt),
            "version": getattr(prompt, "version_number", None),
            "content_hash": getattr(prompt, "content_hash", None),
            "variables": template_variables(template),
            "skills_unavailable": unavailable,
        }

    # ── the local copy ──────────────────────────────────────
    def _route(
        self, scope: Optional[str], query: Optional[str] = None, inject: bool = False,
        mark: Optional[str] = None,
    ) -> Tuple[str, List[str], Optional[str], bool, List[str], List[Dict[str, str]], Optional[Dict[str, Any]]]:
        """One routing decision: (menu text, offered names, routing id, menu was
        cut, skills whose text it pasted in, their versions, and a prompt split's
        prompt for this run). Text is pasted in only with ``inject``.

        ``mark`` (skills-first Phase 3): ``"this_run"`` when the answer shapes the run
        that asked (a route-first call), ``"capable"`` for a background call. Only a
        ``this_run`` call can get an experiment's arm; either one learns whether the
        agent runs an experiment, which switches every later run to routing first.
        """
        router = self._skill_router()
        if router is None:
            return "", [], None, False, [], [], None
        from ._skill_witness import copy_delivered_versions
        from .skill_router import (
            _release_scoped_routing_rail,
            consume_last_delivered_names,
            consume_last_delivered_versions,
            consume_last_menu_truncated,
            consume_last_offered_names,
        )
        # A version witness an earlier, failed build left in this context must not
        # be read as this build's (on the synchronous path the context is the caller's).
        consume_last_delivered_versions()
        from .skill_router import (
            consume_last_always_include,
            consume_last_experiment,
            consume_last_run_prompt,
        )

        consume_last_experiment()
        consume_last_run_prompt()
        consume_last_always_include()
        try:
            if self._keyless or mark is None:
                prefix, _tail, routing_id = router.build_prompt_parts(
                    query, agent_name=self.name, inject_body=inject, scope=scope,
                )
            else:
                prefix, _tail, routing_id = router.build_prompt_parts(
                    query, agent_name=self.name, inject_body=inject, scope=scope, experiments=mark,
                )
        except Exception:
            consume_last_delivered_versions()
            raise
        finally:
            _release_scoped_routing_rail(router, scope)
        offered = _clean_names(consume_last_offered_names())
        truncated = bool(consume_last_menu_truncated())
        # Drained either way, so nothing leaks into a later call.
        delivered = _clean_names(consume_last_delivered_names())
        versions = copy_delivered_versions(consume_last_delivered_versions())
        experiment = consume_last_experiment()
        run_prompt = consume_last_run_prompt()
        always = consume_last_always_include()
        if prefix or routing_id:
            # An answer: the agent's always-include list is as it says, none included.
            self._store_always(always)
        if mark is not None and not self._keyless and (prefix or routing_id):
            # A marked answer says whether the agent runs an experiment: named, or not.
            with self._lock:
                self._experiment = experiment
        if not inject:
            delivered, versions = [], []
        return prefix or "", offered, routing_id, truncated, delivered, versions, run_prompt

    def _store_always(self, rows: List[Dict[str, Any]]) -> None:
        """Keep the always-include skills' text, fetched again only when the list changes."""
        key = tuple((row.get("name"), row.get("version")) for row in rows)
        with self._lock:
            if key == self._always["key"]:
                return
        texts: List[Tuple[str, str]] = []
        versions: List[Dict[str, str]] = []
        router = self._skill_router()
        if rows and router is not None and callable(getattr(router, "always_include_texts", None)):
            try:
                texts, versions = router.always_include_texts(rows, agent_name=self.name)
            except Exception:  # noqa: BLE001 - the next answer tries again
                self._log.debug("could not fetch the always-include skills' text", exc_info=True)
                return
        with self._lock:
            self._always = {"key": key, "texts": list(texts), "versions": [dict(v) for v in versions]}

    def _always_now(self, skip: Any = ()) -> Tuple[List[Tuple[str, str]], List[Dict[str, str]]]:
        """The always-include texts and their versions now, but those named in ``skip``."""
        with self._lock:
            texts = [(name, text) for name, text in self._always["texts"] if name not in skip]
            names = {name for name, _text in texts}
            versions = [dict(v) for v in self._always["versions"] if v.get("name") in names]
        return texts, versions

    def _store_menu(
        self, prefix: str, offered: List[str], routing_id: Optional[str], truncated: bool,
    ) -> None:
        with self._lock:
            if not prefix and not routing_id:
                # No answer. Keep serving the last good list rather than none.
                if not self._menu_loaded:
                    self._menu_unavailable = "DecimalAI did not return this agent's skills"
                return
            self._menu_prefix, self._menu_names = prefix, list(offered)
            self._menu_unavailable, self._menu_loaded = None, True
            self._menu_truncated = truncated
            if routing_id:
                self._menu_routing_id = routing_id
            names = list(offered)
        # One tool serves every run. Its name list is the agent's whole menu when
        # that fits; a menu picked per message would be another run's, so then it
        # names none and the platform checks the name instead.
        self._on_menu([] if truncated else names)

    def _route_run(self, run_key: Optional[str], shown: List[str], query: Optional[str]) -> None:
        """Background: this run's own decision on its message, and a fresh copy."""
        try:
            prefix, offered, routing_id, truncated, _delivered, _versions, _run_prompt = self._route(
                run_key, query, mark="capable",
            )
        except Exception:
            self._log.debug("background skill routing failed (non-fatal)", exc_info=True)
            return
        self._store_menu(prefix, offered, routing_id, truncated)
        # Stamp the run only when the decision offered exactly what it was shown;
        # otherwise the offer/activation join would describe another list.
        if routing_id and not self._keyless and sorted(offered) == sorted(shown):
            self._stamp_routing(run_key, routing_id)

    def _refresh_prompt_if_stale(self) -> None:
        from ._config import _disabled_by_init

        if _disabled_by_init() or self._keyless:
            return
        with self._lock:
            due = (
                not self._prompt_refreshing
                and time.monotonic() - self._prompt_read_at > _PROMPT_REFRESH_SECONDS
            )
            if due:
                self._prompt_refreshing = True
        if due:
            _in_background(self._refresh_prompt)

    def _refresh_prompt(self) -> None:
        from ._agent import read_prompt as _load_prompt
        from ._client import AgentNotFoundError

        try:
            fresh = _load_prompt(self.name, fallback=self._fallback)
            with self._lock:
                self._prompt = fresh
        except AgentNotFoundError:
            self._log.warning(
                "DecimalAI: agent %r no longer exists; still serving the prompt read earlier.",
                self.name,
            )
        except Exception:
            self._log.debug("prompt refresh failed (non-fatal); keeping the last copy", exc_info=True)
        finally:
            with self._lock:
                self._prompt_read_at = time.monotonic()
                self._prompt_refreshing = False

    # ── load_skill ──────────────────────────────────────────
    @staticmethod
    def _tool_schema(names: List[str]) -> Dict[str, Any]:
        name_schema: Dict[str, Any] = {
            "type": "string",
            "description": "The exact name of one skill listed under Available Skills.",
        }
        if names:
            name_schema["enum"] = list(names)
        return {
            "type": "object",
            "properties": {"name": name_schema},
            "required": ["name"],
            "additionalProperties": False,
        }

    def _run_state(self, run_key: Optional[str]) -> Optional[Dict[str, Any]]:
        if run_key is None:
            return None
        with self._lock:
            return next((st for (key_run, _a), st in self._runs.items() if key_run == run_key), None)

    def _already_loaded(self, name: str, state: Optional[Dict[str, Any]]) -> Optional[str]:
        loaded = state["loaded"] if state is not None else None
        if loaded is not None and name in loaded:
            return (
                f"{name!r} is already loaded earlier in this run. Follow those "
                "instructions; do not load it again."
            )
        return None

    def _mark_loaded(self, name: str, state: Optional[Dict[str, Any]], result: Any) -> None:
        loaded = state["loaded"] if state is not None else None
        if loaded is not None and isinstance(result, str) and result.startswith(f"## Skill: {name}"):
            with self._lock:
                loaded.add(name)

    async def _load(self, name: str, state: Optional[Dict[str, Any]] = None) -> str:
        """Serve one load_skill call of the current run.

        ``state`` is the run's own, when the caller holds it: the binding keeps
        only its latest runs' states, and a long run would otherwise lose its own.
        """
        run_key = self._run_key()
        if state is None:
            state = self._run_state(run_key)
        repeat = self._already_loaded(name, state)
        if repeat is not None:
            return repeat
        await self._name_this_runs_decision(run_key, state)
        result = await _in_thread(self._serve, name)
        result = self._word_for_run(name, state, result)
        self._mark_loaded(name, state, result)
        return result

    @staticmethod
    def _word_for_run(name: str, state: Optional[Dict[str, Any]], result: Any) -> Any:
        """In an experiment, a refused load says the skill isn't this run's, not that it doesn't exist.

        The platform refuses a held-out skill to its run's routing id (a 404, like a missing
        name). Telling the model the skill doesn't exist invites a guessed name.
        """
        if (state is not None and state.get("experiment") and isinstance(result, str)
                and result.startswith("load_skill: no skill named")):
            return (f"load_skill: {name!r} isn't available for this run. Proceed without it, "
                    "and don't try another name for it.")
        return result

    def _load_sync(self, name: str, state: Optional[Dict[str, Any]] = None) -> str:
        """`_load` for a caller that is not on an event loop."""
        run_key = self._run_key()
        if state is None:
            state = self._run_state(run_key)
        repeat = self._already_loaded(name, state)
        if repeat is not None:
            return repeat
        pending = state.get("pending") if state is not None else None
        if pending is not None and not pending.done():
            try:
                pending.result(timeout=_LOAD_WAITS_FOR_ROUTING_S)
            except Exception:
                self._log.debug("this run's routing decision is not in yet", exc_info=True)
        self._name_latest_decision(run_key)
        result = self._serve(name)
        result = self._word_for_run(name, state, result)
        self._mark_loaded(name, state, result)
        return result

    async def _name_this_runs_decision(self, run_key: Optional[str], state: Optional[Dict[str, Any]]) -> None:
        """Make the load name a decision of THIS run, or at least of this agent.

        The platform serves a public-registry skill only to a load whose routing
        decision offered it. A run's own decision may still be in flight (it is
        recorded in the background); its load waits for it briefly, and failing
        that names this agent's latest decision — never the router's latest,
        which another agent in the process may have made.
        """
        if run_key is None:
            return
        pending = state.get("pending") if state is not None else None
        if pending is not None and not pending.done():
            # Not `asyncio.wait_for(wrap_future(pending))`: that cancels a decision
            # still queued when the wait times out.
            try:
                await _wait_without_cancelling(pending, _LOAD_WAITS_FOR_ROUTING_S)
            except Exception:
                self._log.debug("this run's routing decision is not in yet", exc_info=True)
        self._name_latest_decision(run_key)

    def _name_latest_decision(self, run_key: Optional[str]) -> None:
        """Without a decision of its own, a run's loads name this agent's latest."""
        if run_key is None:
            return
        router = self._skill_router()
        known = getattr(router, "_run_routing_ids", None)
        remember = getattr(router, "_remember_run_routing", None)
        if not isinstance(known, Mapping) or not callable(remember):
            return
        with self._lock:
            fallback = self._menu_routing_id
        with router._rail_lock:
            has_own = run_key in known
        if not has_own and fallback:
            remember(run_key, fallback)

    # ── settings ────────────────────────────────────────────
    def _warn_once(self, agent: Any, message: str) -> None:
        key = (self._agent_label(agent) or "", message)
        if key not in self._warned:
            self._warned.add(key)
            self._log.warning("DecimalAI (%s): %s", key[0] or self.name, message)

    @staticmethod
    def _channels() -> Tuple[bool, bool]:
        """(the load_skill tool is on, skill text is pasted into the instructions).

        The two documented settings, read per run so a re-init applies: the
        tool's kill switch (``DECIMALAI_LOAD_SKILL_TOOL=0`` /
        ``init(load_skill_tool=False)``) and ``inject_skill_body``
        (``DECIMALAI_INJECT_SKILL_BODY``), which is off while the tool is on
        unless set, and on when the tool is off.
        """
        try:
            from ._config import _get_config

            config = _get_config()
        except Exception:
            return True, False
        tool = bool(getattr(config, "load_skill_tool", True))
        return tool, bool(config.resolve_inject_body(has_tool_loop=tool))
