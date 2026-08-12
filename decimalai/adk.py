"""Google ADK (Agent Development Kit) integration.

First-class tracing for ``google-adk`` via a native ADK ``BasePlugin``. One
DecimalAI :class:`RunTrace` is captured per ADK *invocation* (one
``Runner.run`` / ``run_async``), with LLM generations, tool calls, and
multi-agent (sub-agent) activity recorded against it. The root agent's
model / instruction / tools / sub-agents are auto-registered as a manifest.

Simple path (global — every ``Runner`` is traced)::

    import decimalai
    decimalai.init(adk=True)          # or: from decimalai.adk import instrument; instrument()

    from google.adk.runners import Runner
    runner = Runner(agent=my_agent, app_name="support", session_service=svc)
    # all runner.run_async(...) invocations are now traced

Explicit path (add the plugin to your own ``Runner``)::

    from decimalai.adk import DecimalaiPlugin
    runner = Runner(agent=my_agent, app_name="support", session_service=svc,
                    plugins=[DecimalaiPlugin(agent_name="support")])

ADK is built around Gemini; the release gate pairs this adapter with the
``google`` provider only.
"""

from __future__ import annotations

import warnings

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from .schema.common import CallRole, FinishReason, SpanType, Status
from .schema.manifest import ManifestTracker, extract_from_config
from .schema.trace import LlmCallRecord, RunTrace, ToolCallRecord, TraceSpan

logger = logging.getLogger("decimalai.adk")

# ── Module-level manifest state (mirrors langchain.py) ─────────
# A manifest is registered once per distinct (agent, tools, prompt, model,
# subagents) shape and reused across traces until that shape changes.
_manifest_tracker = ManifestTracker()
_manifest_id: Optional[str] = None
_manifest_lock = threading.Lock()

# Set by instrument(); used as the fallback agent_name when a trace can't
# resolve one from the ADK agent itself.
_install_agent_name: Optional[str] = None
_runner_patched = False

# Cached plugin class (built lazily so importing this module never requires
# google-adk to be installed).
_PluginClass: Any = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _content_to_text(content: Any) -> str:
    """Flatten a genai ``Content`` (or plain str) into text for previews."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = getattr(content, "parts", None) or []
    chunks: List[str] = []
    for p in parts:
        text = getattr(p, "text", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def _provider_for_model(model: Optional[str]) -> Optional[str]:
    """Best-effort provider id from a model string (ADK is Gemini-first)."""
    if not model:
        return None
    m = str(model).lower()
    if "gemini" in m or "google" in m:
        return "google"
    if "claude" in m:
        return "anthropic"
    if "gpt" in m or m.startswith(("o1", "o3", "o4")):
        return "openai"
    return None


def _map_finish_reason(raw: Any, has_tool_calls: bool) -> FinishReason:
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    s = str(raw or "").upper()
    if "MAX_TOKEN" in s or "LENGTH" in s:
        return FinishReason.LENGTH
    return FinishReason.STOP


def _function_calls_from_content(content: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Extract (name, args) for every function_call part in a genai Content."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    parts = getattr(content, "parts", None) or []
    for p in parts:
        fc = getattr(p, "function_call", None)
        if fc is not None and getattr(fc, "name", None):
            args = getattr(fc, "args", None)
            out.append((fc.name, dict(args) if isinstance(args, dict) else {}))
    return out


def _introspect_agent(agent: Any) -> Dict[str, Any]:
    """Pull manifest-relevant config off an ADK agent (best-effort, never raises)."""
    info: Dict[str, Any] = {}
    try:
        model = getattr(agent, "model", None)
        # ADK models can be a string id or a model object with a `.model` attr.
        if model is not None and not isinstance(model, str):
            model = getattr(model, "model", None) or str(model)
        if model:
            info["model"] = {"name": str(model), "provider": _provider_for_model(str(model))}

        instruction = getattr(agent, "instruction", None)
        if isinstance(instruction, str) and instruction.strip():
            info["prompts"] = {"system": instruction.strip()}

        tools = []
        for t in getattr(agent, "tools", None) or []:
            name = getattr(t, "name", None) or getattr(t, "__name__", None)
            if not name:
                continue
            entry: Dict[str, Any] = {"name": str(name)}
            desc = getattr(t, "description", None) or getattr(t, "__doc__", None)
            if isinstance(desc, str) and desc.strip():
                entry["description"] = desc.strip()
            tools.append(entry)
        if tools:
            info["tools"] = tools

        subagents = []
        for sub in getattr(agent, "sub_agents", None) or []:
            sub_name = getattr(sub, "name", None)
            if sub_name:
                subagents.append({"name": str(sub_name)})
        if subagents:
            info["subagents"] = subagents
    except Exception:
        logger.debug("ADK agent introspection failed (non-fatal)", exc_info=True)
    return info


class _RunState:
    """Per-invocation trace accumulator, keyed by ADK ``invocation_id``."""

    __slots__ = (
        "trace_id", "agent_name", "root_agent_name", "started_at", "ended_at",
        "user_input_preview", "final_output_preview",
        "llm_calls", "spans", "pending_llm", "pending_tools",
        "status", "error_code", "error_message", "manifest",
    )

    def __init__(self, *, agent_name: Optional[str], started_at: datetime):
        self.trace_id: UUID = uuid4()
        self.agent_name = agent_name
        # The ADK root agent's own ``.name`` (a Python identifier). Distinct
        # from ``agent_name`` (the DecimalAI label, which an explicit plugin
        # name can override). Used only to detect when the *root* agent
        # finishes — that ``after_agent`` is our end-of-invocation signal.
        self.root_agent_name: Optional[str] = None
        self.started_at = started_at
        self.ended_at: Optional[datetime] = None
        self.user_input_preview: Optional[str] = None
        self.final_output_preview: Optional[str] = None
        self.llm_calls: List[LlmCallRecord] = []
        self.spans: List[TraceSpan] = []
        # before_model pushes; after_model / on_model_error pops (LIFO; calls
        # are sequential within an invocation, so depth is ~1).
        self.pending_llm: List[LlmCallRecord] = []
        # key -> (ToolCallRecord, TraceSpan, started_at)
        self.pending_tools: Dict[str, Tuple[ToolCallRecord, TraceSpan, datetime]] = {}
        self.status: Status = Status.SUCCESS
        self.error_code: Optional[str] = None
        self.error_message: Optional[str] = None
        # Manifest config introspected from the root agent in before_run.
        self.manifest: Dict[str, Any] = {}


def _plugin_class() -> Any:
    """Build (once) and return the concrete BasePlugin subclass.

    Deferred so that ``import decimalai.adk`` works without google-adk; the
    google-adk import only happens when a plugin is actually constructed.
    """
    global _PluginClass
    if _PluginClass is not None:
        return _PluginClass

    from google.adk.plugins.base_plugin import BasePlugin

    class _DecimalaiPlugin(BasePlugin):
        """ADK BasePlugin that emits one DecimalAI RunTrace per invocation."""

        def __init__(
            self,
            agent_name: Optional[str] = None,
            *,
            name: str = "decimalai",
            project: Optional[str] = None,
            parent_trace_id: Optional[str] = None,
        ):
            super().__init__(name=name)
            self.agent_name = agent_name
            self.project = project
            self.parent_trace_id = parent_trace_id
            self._runs: Dict[str, _RunState] = {}
            self._lock = threading.Lock()

        # ── helpers ────────────────────────────────────────
        def _get_run(self, invocation_id: Optional[str]) -> Optional[_RunState]:
            if not invocation_id:
                return None
            with self._lock:
                return self._runs.get(invocation_id)

        def _pop_run(self, invocation_id: Optional[str]) -> Optional[_RunState]:
            if not invocation_id:
                return None
            with self._lock:
                return self._runs.pop(invocation_id, None)

        # ── run boundary ───────────────────────────────────
        async def before_run_callback(self, *, invocation_context: Any):  # noqa: ANN001
            from . import _config
            if not _config._is_enabled():
                return None
            agent = getattr(invocation_context, "agent", None)
            # Explicit plugin agent_name wins over ADK's internal node name (a
            # constrained Python identifier); global instrument() supplies no
            # explicit name, so it falls back to the ADK agent's own name.
            agent_name = (
                self.agent_name or getattr(agent, "name", None) or _install_agent_name
            )
            state = _RunState(agent_name=agent_name, started_at=_now())
            state.root_agent_name = getattr(agent, "name", None)
            state.user_input_preview = _content_to_text(
                getattr(invocation_context, "user_content", None)
            ) or None
            if agent is not None:
                state.manifest = _introspect_agent(agent)
            inv_id = getattr(invocation_context, "invocation_id", None)
            if inv_id:
                with self._lock:
                    self._runs[inv_id] = state
            return None

        async def _complete(self, inv_id: Optional[str]) -> None:
            """Pop the run and build+send its trace. Idempotent: the pop means
            a second caller for the same invocation gets ``None`` and no-ops."""
            state = self._pop_run(inv_id)
            if state is None:
                return
            state.ended_at = _now()
            # Finalize off the event loop: manifest registration is a sync
            # network round-trip, and the trace ingest is queued there too.
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._finalize, state)
            except RuntimeError:
                self._finalize(state)

        async def after_agent_callback(self, *, agent: Any, callback_context: Any):  # noqa: ANN001
            # Primary end-of-invocation signal. ADK 2.x runs an LlmAgent root
            # through the node runtime, which fires before_run but *not*
            # after_run (see runners._run_node_async — after_run is only wired
            # in the legacy _run_with_trace path). The root agent's after_agent
            # is the last callback of the invocation, so finalize here — but
            # only for the *root*: in a multi-agent run each sub-agent also
            # fires after_agent, and we want exactly one trace per invocation.
            inv_id = getattr(callback_context, "invocation_id", None)
            state = self._get_run(inv_id)
            if state is None:
                return None
            if state.root_agent_name is not None and (
                getattr(agent, "name", None) != state.root_agent_name
            ):
                return None  # a sub-agent finished, not the root
            await self._complete(inv_id)
            return None

        async def after_run_callback(self, *, invocation_context: Any):  # noqa: ANN001
            # Fallback for the legacy (non-node) runtime path, which does fire
            # after_run. No-ops when after_agent already finalized this run.
            await self._complete(getattr(invocation_context, "invocation_id", None))
            return None

        # ── model boundary ─────────────────────────────────
        async def before_model_callback(self, *, callback_context: Any, llm_request: Any):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            model = getattr(llm_request, "model", None)
            rendered_input: List[Dict[str, Any]] = []
            for c in getattr(llm_request, "contents", None) or []:
                txt = _content_to_text(c)
                if txt:
                    rendered_input.append({"role": getattr(c, "role", "user"), "content": txt})
            rec = LlmCallRecord(
                span_id=uuid4(),
                agent_name=getattr(callback_context, "agent_name", None) or state.agent_name,
                call_role=CallRole.OTHER,
                provider=_provider_for_model(str(model) if model else None),
                model_name=str(model) if model else None,
                rendered_input=rendered_input or None,
                started_at=_now(),
            )
            state.pending_llm.append(rec)
            return None

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None or not state.pending_llm:
                return None
            rec = state.pending_llm.pop()
            rec.ended_at = _now()
            content = getattr(llm_response, "content", None)
            text = _content_to_text(content)
            if text:
                rec.output = {"role": "model", "content": text}
                state.final_output_preview = text
            # Tool calls requested by the model in this turn.
            fcs = _function_calls_from_content(content)
            for fname, fargs in fcs:
                rec.tool_calls.append(ToolCallRecord(tool_name=fname, args=fargs))
            rec.finish_reason = _map_finish_reason(
                getattr(llm_response, "finish_reason", None), bool(fcs)
            )
            usage = getattr(llm_response, "usage_metadata", None)
            if usage is not None:
                rec.input_tokens = getattr(usage, "prompt_token_count", None)
                rec.output_tokens = getattr(usage, "candidates_token_count", None)
            err_msg = getattr(llm_response, "error_message", None)
            if err_msg:
                rec.status = Status.ERROR
            state.llm_calls.append(rec)
            return None

        async def on_model_error_callback(self, *, callback_context: Any, llm_request: Any, error: Exception):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            if state.pending_llm:
                rec = state.pending_llm.pop()
                rec.ended_at = _now()
                rec.status = Status.ERROR
                rec.finish_reason = FinishReason.ERROR
                state.llm_calls.append(rec)
            state.status = Status.ERROR
            state.error_message = str(error)
            return None

        # ── tool boundary ──────────────────────────────────
        def _tool_key(self, tool: Any, tool_context: Any) -> str:
            fcid = getattr(tool_context, "function_call_id", None)
            return str(fcid) if fcid else getattr(tool, "name", None) or "tool"

        async def before_tool_callback(self, *, tool: Any, tool_args: Any, tool_context: Any):  # noqa: ANN001
            state = self._get_run(getattr(tool_context, "invocation_id", None))
            if state is None:
                return None
            started = _now()
            tool_name = getattr(tool, "name", None) or "tool"
            rec = ToolCallRecord(
                tool_name=str(tool_name),
                args=dict(tool_args) if isinstance(tool_args, dict) else {},
            )
            span = TraceSpan(
                span_type=SpanType.TOOL,
                name=str(tool_name),
                started_at=started,
                input_preview=str(tool_args)[:500] if tool_args else None,
            )
            state.pending_tools[self._tool_key(tool, tool_context)] = (rec, span, started)
            return None

        async def after_tool_callback(self, *, tool: Any, tool_args: Any, tool_context: Any, result: Any):  # noqa: ANN001
            state = self._get_run(getattr(tool_context, "invocation_id", None))
            if state is None:
                return None
            key = self._tool_key(tool, tool_context)
            pending = state.pending_tools.pop(key, None)
            if pending is None:
                return None
            rec, span, started = pending
            ended = _now()
            rec.result = result
            rec.latency_ms = int((ended - started).total_seconds() * 1000)
            span.ended_at = ended
            span.output_preview = str(result)[:500] if result is not None else None
            state.spans.append(span)
            # Attach the tool record to the most recent LLM call if it didn't
            # already capture this function_call, so tool counts stay accurate.
            if state.llm_calls and not any(
                tc.tool_name == rec.tool_name for tc in state.llm_calls[-1].tool_calls
            ):
                state.llm_calls[-1].tool_calls.append(rec)
            return None

        async def on_tool_error_callback(self, *, tool: Any, tool_args: Any, tool_context: Any, error: Exception):  # noqa: ANN001
            state = self._get_run(getattr(tool_context, "invocation_id", None))
            if state is None:
                return None
            key = self._tool_key(tool, tool_context)
            pending = state.pending_tools.pop(key, None)
            if pending is not None:
                rec, span, started = pending
                ended = _now()
                rec.status = Status.ERROR
                rec.latency_ms = int((ended - started).total_seconds() * 1000)
                span.status = Status.ERROR
                span.ended_at = ended
                span.output_preview = str(error)[:500]
                state.spans.append(span)
            state.status = Status.ERROR
            state.error_message = str(error)
            return None

        # ── finalize (runs in executor thread) ─────────────
        def _finalize(self, state: _RunState) -> None:
            from . import _config
            if not _config._is_enabled():
                return
            manifest_id = self._maybe_register_manifest(state)
            try:
                client = _config._get_client()
                config = _config._get_config()
                trace = RunTrace(
                    id=state.trace_id,
                    project=self.project or (config.project if config else None),
                    agent_name=state.agent_name,
                    parent_trace_id=self.parent_trace_id,
                    status=state.status,
                    source_type="production",
                    started_at=state.started_at,
                    ended_at=state.ended_at or _now(),
                    user_input_preview=state.user_input_preview,
                    final_output_preview=state.final_output_preview,
                    error_code=state.error_code,
                    error_message=state.error_message,
                    spans=list(state.spans),
                    llm_calls=list(state.llm_calls),
                    manifest_id=manifest_id,
                )
                _config._sender.submit(client.ingest_trace, trace)
                logger.debug(
                    "Queued ADK trace %s (%d spans, %d llm_calls, manifest=%s)",
                    trace.id, len(trace.spans), len(trace.llm_calls),
                    trace.manifest_id or "none",
                )
            except Exception:
                logger.exception("Failed to queue ADK trace %s", state.trace_id)

        def _maybe_register_manifest(self, state: _RunState) -> Optional[str]:
            """Register the root agent's manifest if its shape is new."""
            global _manifest_id
            from . import _config
            if not _config._is_enabled():
                return _manifest_id

            man = state.manifest or {}
            tools = man.get("tools")
            prompts = man.get("prompts")
            model = man.get("model")
            subagents = man.get("subagents")
            if not (tools or prompts or model or subagents):
                return _manifest_id

            snapshot = extract_from_config(
                agent_name=state.agent_name or "unknown",
                tools=tools,
                prompts=prompts,
                models={"default": model} if model else None,
                subagents=subagents,
            )
            with _manifest_lock:
                if not _manifest_tracker.check_and_update(snapshot):
                    return _manifest_id  # same hash, already registered
                try:
                    client = _config._get_client()
                    result = client.register_manifest(snapshot)
                    _manifest_id = result.get("manifest_id", snapshot.id)
                    logger.info(
                        "Registered ADK manifest %s (hash=%s, components=%d)",
                        _manifest_id, snapshot.manifest_hash[:12], len(snapshot.components),
                    )
                except Exception:
                    logger.warning("Failed to register ADK manifest, continuing", exc_info=True)
                    _manifest_id = snapshot.id
            return _manifest_id

    _PluginClass = _DecimalaiPlugin
    return _PluginClass


def DecimalaiPlugin(  # noqa: N802 — factory presents as a class for ergonomics
    agent_name: Optional[str] = None,
    *,
    name: str = "decimalai",
    project: Optional[str] = None,
    parent_trace_id: Optional[str] = None,
) -> Any:
    """Construct a DecimalAI ADK plugin to add to a ``Runner``.

    Args:
        agent_name: Fallback agent name when the ADK agent doesn't supply one.
        name: ADK plugin name (must be unique among a Runner's plugins).
        project: Optional project grouping for the traces.
        parent_trace_id: When this Runner runs as a sub-agent of another,
            the parent's trace id — links the child traces in the backend.
    """
    return _plugin_class()(
        agent_name=agent_name, name=name, project=project, parent_trace_id=parent_trace_id,
    )


def instrument(agent_name: Optional[str] = None) -> None:
    """Install DecimalAI tracing globally for google-adk.

    Monkeypatches ``Runner.__init__`` so a single shared DecimalAI plugin is
    auto-injected into every ``Runner`` created afterwards. Idempotent.

    Args:
        agent_name: Default agent name for traces whose ADK agent doesn't
            supply one of its own.
    """
    global _install_agent_name, _runner_patched
    _install_agent_name = agent_name

    if _runner_patched:
        return

    try:
        from google.adk.runners import Runner
    except ImportError:
        logger.warning(
            "decimalai.adk.instrument() called but google-adk is not installed; "
            "ADK tracing not active. "
            "Install it with: pip install \"decimalai[adk]\""
        )
        return

    shared_plugin = DecimalaiPlugin(agent_name=agent_name)
    original_init = Runner.__init__

    def patched_init(self, **kwargs):  # Runner.__init__ is keyword-only
        plugins = list(kwargs.pop("plugins", None) or [])
        if not any(getattr(p, "name", None) == shared_plugin.name for p in plugins):
            plugins.insert(0, shared_plugin)
        kwargs["plugins"] = plugins
        original_init(self, **kwargs)

    Runner.__init__ = patched_init  # type: ignore[method-assign]
    _runner_patched = True
    logger.info("DecimalAI ADK tracing installed globally (agent_name=%s)", agent_name)


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
        "decimalai.adk.install() is deprecated; use "
        "decimalai.adk.instrument() instead. It turns on tracing for adk "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
