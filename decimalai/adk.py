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

import asyncio
import logging
import threading
import warnings
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
#
# Keyed BY AGENT NAME, not process-wide. Two ADK agents in one process very
# often share a manifest shape — same model, same instruction, same tool
# names, differing only in the label the traces are filed under — so a single
# tracker answered "same hash, already registered" for the second agent and
# handed back the FIRST agent's manifest_id. The trace then pointed at a
# manifest registered for somebody else, which is a cross-agent identity leak,
# not a cache hit. Per-agent state keeps the dedup (an agent that runs twice
# still mints one version) while making the reuse agent-scoped, which is also
# how the backend itself scopes manifest hashes.
_manifest_trackers: Dict[str, ManifestTracker] = {}
_manifest_ids: Dict[str, str] = {}
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
        "llm_calls", "spans", "pending_llm", "pending_tools", "agent_stack",
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
        # are sequential within an invocation, so depth is ~1). Each entry is
        # the record AND the LLM span it is attached to, so the two can never
        # drift apart.
        self.pending_llm: List[Tuple[LlmCallRecord, TraceSpan]] = []
        # key -> (ToolCallRecord, TraceSpan, started_at)
        self.pending_tools: Dict[str, Tuple[ToolCallRecord, TraceSpan, datetime]] = {}
        # Open agent spans, outermost first: before_agent pushes, after_agent
        # pops. ADK nests sub-agents inside their parent's agent callback, so
        # the top of this stack is the span everything else hangs off.
        self.agent_stack: List[TraceSpan] = []
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

        @staticmethod
        def _parent_id(state: _RunState) -> Optional[UUID]:
            """The span everything opened right now hangs off: the innermost
            agent still running, or None when there is no agent span (which is
            what makes that span the trace root)."""
            return state.agent_stack[-1].id if state.agent_stack else None

        @staticmethod
        def _close_span(
            span: TraceSpan, *, output: Optional[str] = None, error: bool = False
        ) -> None:
            """Stamp a span's end. Previews are only set when they carry text —
            an empty preview reads as "the model said nothing" rather than as
            "there was nothing to preview"."""
            if span.ended_at is None:
                span.ended_at = _now()
            if error:
                span.status = Status.ERROR
            if output and span.output_preview is None:
                span.output_preview = output[:500]

        def _close_open_spans(self, state: _RunState) -> None:
            """Ship the spans of an invocation that ended abnormally.

            A run that raises out of the model never reaches ``after_agent``, so
            its agent span (and any in-flight generation) is still open. Ingest
            rejects the WHOLE trace over one span with no ``ended_at``, so these
            are closed here — they did end, abnormally, now.
            """
            error = state.status == Status.ERROR
            while state.pending_llm:
                _, span = state.pending_llm.pop()
                self._close_span(span, output=state.error_message, error=error)
                state.spans.append(span)
            for _, span, _started in list(state.pending_tools.values()):
                self._close_span(span, output=state.error_message, error=error)
                state.spans.append(span)
            state.pending_tools.clear()
            while state.agent_stack:
                span = state.agent_stack.pop()
                self._close_span(
                    span,
                    output=state.error_message if error else state.final_output_preview,
                    error=error,
                )
                state.spans.append(span)

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
            self._close_open_spans(state)
            # Finalize off the event loop: manifest registration is a sync
            # network round-trip, and the trace ingest is queued there too.
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._finalize, state)
            except RuntimeError:
                self._finalize(state)

        # ── agent boundary ─────────────────────────────────
        async def before_agent_callback(self, *, agent: Any, callback_context: Any):  # noqa: ANN001
            # The agent span is the spine of the trace: without it every model
            # turn and every tool call is a root, so an invocation arrives as a
            # flat list the UI cannot lay out as a waterfall — and an agent that
            # calls no tools arrives as an EMPTY timeline, because tool spans
            # used to be the only spans this plugin ever built.
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            span = TraceSpan(
                parent_span_id=self._parent_id(state),
                span_type=SpanType.AGENT,
                name=str(getattr(agent, "name", None) or state.agent_name or "agent"),
                started_at=_now(),
                input_preview=(state.user_input_preview or None),
            )
            state.agent_stack.append(span)
            return None

        def _close_agent_span(
            self, state: _RunState, agent: Any, *, error: Optional[Exception] = None
        ) -> None:
            """Pop this agent's span off the stack and record it.

            Matched by name from the top of the stack rather than blindly
            popping: a callback that never fired (an agent ADK abandoned) must
            not close somebody else's span.
            """
            name = getattr(agent, "name", None)
            span = None
            for candidate in reversed(state.agent_stack):
                if name is None or candidate.name == str(name):
                    span = candidate
                    break
            if span is None:
                return
            state.agent_stack.remove(span)
            self._close_span(
                span,
                output=str(error)[:500] if error else state.final_output_preview,
                error=error is not None,
            )
            state.spans.append(span)

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
            # Close the span BEFORE finalizing: _complete builds the trace from
            # state.spans, and a root span appended afterwards would be lost.
            self._close_agent_span(state, agent)
            if state.root_agent_name is not None and (
                getattr(agent, "name", None) != state.root_agent_name
            ):
                return None  # a sub-agent finished, not the root
            await self._complete(inv_id)
            return None

        async def on_agent_error_callback(self, *, agent: Any, callback_context: Any, error: Exception):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            self._close_agent_span(state, agent, error=error)
            state.status = Status.ERROR
            if not state.error_message:
                state.error_message = str(error)
            return None

        async def after_run_callback(self, *, invocation_context: Any):  # noqa: ANN001
            # Fallback for the legacy (non-node) runtime path, which does fire
            # after_run. No-ops when after_agent already finalized this run.
            await self._complete(getattr(invocation_context, "invocation_id", None))
            return None

        async def on_run_error_callback(self, *, invocation_context: Any, error: Exception):  # noqa: ANN001
            # End-of-invocation signal for the *error* path. When an exception
            # escapes the runner (e.g. a model 429), the root agent never
            # completes, so neither after_agent nor after_run fires — without
            # this the run state would be orphaned and no trace sent. ADK only
            # notifies on_run_error when the exception actually escapes (a
            # recovered model/tool error never reaches here), and it re-raises
            # afterwards, so finalizing here can't fire on success — and
            # _complete's pop makes a second finalize a no-op regardless.
            inv_id = getattr(invocation_context, "invocation_id", None)
            state = self._get_run(inv_id)
            if state is None:
                return None
            state.status = Status.ERROR
            if not state.error_message:
                state.error_message = str(error)
            await self._complete(inv_id)
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
            started = _now()
            # One span per generation, parented to the agent that asked for it.
            # `LlmCallRecord.span_id` used to be a fresh uuid4 that named no
            # span at all, so the call record could not be joined back to the
            # timeline it belonged to.
            span = TraceSpan(
                parent_span_id=self._parent_id(state),
                span_type=SpanType.LLM,
                name=str(model) if model else "llm",
                started_at=started,
                input_preview=(
                    "\n".join(e["content"] for e in rendered_input)[:500] or None
                    if rendered_input else None
                ),
            )
            rec = LlmCallRecord(
                span_id=span.id,
                agent_name=getattr(callback_context, "agent_name", None) or state.agent_name,
                call_role=CallRole.OTHER,
                provider=_provider_for_model(str(model) if model else None),
                model_name=str(model) if model else None,
                rendered_input=rendered_input or None,
                started_at=started,
            )
            state.pending_llm.append((rec, span))
            return None

        async def after_model_callback(self, *, callback_context: Any, llm_response: Any):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None or not state.pending_llm:
                return None
            rec, span = state.pending_llm.pop()
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
            # A pure tool-call turn answers with no text, so `text` is empty;
            # name what the model DID ask for instead of previewing "".
            self._close_span(
                span,
                output=text or (
                    "→ " + ", ".join(name for name, _ in fcs) if fcs else None
                ),
                error=bool(err_msg),
            )
            state.spans.append(span)
            return None

        async def on_model_error_callback(self, *, callback_context: Any, llm_request: Any, error: Exception):  # noqa: ANN001
            state = self._get_run(getattr(callback_context, "invocation_id", None))
            if state is None:
                return None
            if state.pending_llm:
                rec, span = state.pending_llm.pop()
                rec.ended_at = _now()
                rec.status = Status.ERROR
                rec.finish_reason = FinishReason.ERROR
                state.llm_calls.append(rec)
                self._close_span(span, output=str(error), error=True)
                state.spans.append(span)
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
                parent_span_id=self._parent_id(state),
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
            """Register the root agent's manifest if its shape is new.

            Every lookup and every write is scoped to ``state.agent_name``: the
            manifest this trace carries must belong to the agent this trace
            names, and a second agent's identical shape is a different agent's
            manifest, not a cache hit.
            """
            from . import _config
            agent = state.agent_name or "unknown"
            if not _config._is_enabled():
                return _manifest_ids.get(agent)

            man = state.manifest or {}
            tools = man.get("tools")
            prompts = man.get("prompts")
            model = man.get("model")
            subagents = man.get("subagents")
            if not (tools or prompts or model or subagents):
                # Nothing to declare — ride whatever THIS agent already has
                # rather than re-declaring a shape we did not observe.
                return _manifest_ids.get(agent)

            snapshot = extract_from_config(
                agent_name=agent,
                tools=tools,
                prompts=prompts,
                models={"default": model} if model else None,
                subagents=subagents,
            )
            with _manifest_lock:
                tracker = _manifest_trackers.get(agent)
                if tracker is None:
                    tracker = _manifest_trackers[agent] = ManifestTracker()
                if not tracker.check_and_update(snapshot):
                    return _manifest_ids.get(agent)  # same hash, already registered
                try:
                    client = _config._get_client()
                    result = client.register_manifest(snapshot)
                    _manifest_ids[agent] = result.get("manifest_id", snapshot.id)
                    logger.info(
                        "Registered ADK manifest %s for %s (hash=%s, components=%d)",
                        _manifest_ids[agent], agent, snapshot.manifest_hash[:12],
                        len(snapshot.components),
                    )
                except Exception:
                    logger.warning("Failed to register ADK manifest, continuing", exc_info=True)
                    _manifest_ids[agent] = snapshot.id
                    # The hash is already committed to this agent's tracker;
                    # without the reset every later run short-circuits on it and
                    # registration is never retried.
                    tracker.reset()
            return _manifest_ids.get(agent)

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
