"""Claude Agent SDK integration for DecimalAI.

First-class tracing for Anthropic's **Claude Agent SDK** (``claude-agent-sdk``,
import ``claude_agent_sdk``) — the agentic framework whose ``query()`` runs a
tool-use loop against the Claude Code engine. This is distinct from the raw
Anthropic Messages API (traced via ``decimalai.trace`` / the OTEL exporter) and
from ``decimalai.anthropic`` (which is the SkillRouter prompt-injection adapter).

The Claude Agent SDK exposes no global callback/plugin system: ``query()`` is an
async generator that yields ``SystemMessage`` (init) → ``AssistantMessage`` (model
turns, each with text + ``ToolUseBlock``s) → ``UserMessage`` (the matching
``ToolResultBlock``s) → ``ResultMessage`` (cumulative usage, cost, final text).
So we trace by *wrapping the stream*: pass each message through to the caller
while accumulating one DecimalAI :class:`RunTrace`, finalized when the stream
ends. The wrapper is observability-only — it never alters or swallows the
caller's messages, and an ingest failure never breaks the user's run.

Simple path (global — every ``query()`` is traced)::

    import decimalai
    decimalai.init(claude_agent_sdk=True)   # or: from decimalai.claude_agent_sdk import instrument; instrument()

    from claude_agent_sdk import query, ClaudeAgentOptions
    async for message in query(prompt="...", options=ClaudeAgentOptions(...)):
        ...   # traced automatically

Explicit path (no monkeypatch — wrap a stream you already have)::

    from decimalai.claude_agent_sdk import trace_stream
    from claude_agent_sdk import query
    async for message in trace_stream(query(prompt="..."), agent_name="support"):
        ...

The Claude Agent SDK is Anthropic-native; the release gate pairs this adapter
with the ``anthropic`` provider only.

Requires: ``pip install "decimalai[claude-agent-sdk]"``
"""

from __future__ import annotations

import asyncio
import logging
import threading
import warnings
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from .schema.common import CallRole, FinishReason, SpanType, Status
from .schema.manifest import ManifestTracker, extract_from_config
from .schema.trace import LlmCallRecord, RunTrace, ToolCallRecord, TraceSpan

logger = logging.getLogger("decimalai.claude_agent_sdk")

# ── Module-level manifest state (mirrors adk.py) ───────────────────
# A manifest is registered once per distinct (agent, tools, prompt, model,
# subagents) shape and reused across traces until that shape changes.
_manifest_tracker = ManifestTracker()
_manifest_id: Optional[str] = None
_manifest_lock = threading.Lock()

# Set by instrument(); the fallback agent_name when a wrapped stream supplies none.
_install_agent_name: Optional[str] = None
_query_patched = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _preview(obj: Any, max_len: int = 500) -> Optional[str]:
    """Best-effort string preview of a tool input/result (dict, list, or str)."""
    if obj is None:
        return None
    try:
        text = obj if isinstance(obj, str) else str(obj)
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    return (text[:max_len] + "…") if len(text) > max_len else text


def _opt(options: Any, name: str) -> Any:
    """Read a ``ClaudeAgentOptions`` field (dataclass or dict), never raising."""
    if options is None:
        return None
    if isinstance(options, dict):
        return options.get(name)
    return getattr(options, name, None)


def _extract_usage(usage: Any) -> Tuple[Optional[int], Optional[int]]:
    """Pull (input_tokens, output_tokens) from a ResultMessage usage payload.

    Claude reports Anthropic-shaped token names (``input_tokens`` /
    ``output_tokens``), as a dict from the CLI JSON or as an object.

    Anthropic's ``input_tokens`` is the UNCACHED remainder only — the
    effective context adds ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens``. Sum them so traces report what the
    model actually consumed (parity with the OpenAI handler, whose
    ``prompt_tokens`` includes cached tokens); a Claude Code run with a
    warm cache would otherwise read as a few-K-token call."""
    if usage is None:
        return None, None

    def _field(name: str) -> Optional[int]:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return value if isinstance(value, int) else None

    inp = _field("input_tokens")
    if inp is not None:
        inp += (_field("cache_read_input_tokens") or 0) + (
            _field("cache_creation_input_tokens") or 0
        )
    return inp, _field("output_tokens")


class _RunState:
    """Per-stream trace accumulator. One ``query()`` invocation → one RunTrace."""

    __slots__ = (
        "trace_id", "agent_name", "project", "parent_trace_id",
        "started_at", "ended_at", "user_input_preview", "final_output_preview",
        "llm_calls", "spans", "pending_tools",
        "status", "error_message", "session_id", "init_data", "options", "finalized",
    )

    def __init__(
        self,
        *,
        agent_name: Optional[str],
        project: Optional[str],
        parent_trace_id: Optional[str],
        user_input: Optional[str],
        options: Any,
    ):
        self.trace_id: UUID = uuid4()
        self.agent_name = agent_name or _install_agent_name or "claude-agent"
        self.project = project
        self.parent_trace_id = parent_trace_id
        self.started_at = _now()
        self.ended_at: Optional[datetime] = None
        self.user_input_preview = _preview(user_input) if isinstance(user_input, str) else None
        self.final_output_preview: Optional[str] = None
        self.llm_calls: List[LlmCallRecord] = []
        self.spans: List[TraceSpan] = []
        # tool_use_id -> (ToolCallRecord, TraceSpan, started_at); closed by the
        # matching ToolResultBlock in the following UserMessage.
        self.pending_tools: Dict[str, Tuple[ToolCallRecord, TraceSpan, datetime]] = {}
        self.status: Status = Status.SUCCESS
        self.error_message: Optional[str] = None
        self.session_id: Optional[str] = None
        # Raw `init` SystemMessage data — manifest fallback when options omit a field.
        self.init_data: Dict[str, Any] = {}
        self.options = options
        self.finalized = False


# ── Message ingestion ──────────────────────────────────────────────
#
# We dispatch on ``type(message).__name__`` (a string) rather than importing
# claude_agent_sdk's classes, so ``import decimalai.claude_agent_sdk`` never
# requires the SDK to be installed — and the unit suite can drive synthetic
# messages whose classes simply share these names.

def _ingest_message(state: _RunState, message: Any) -> None:
    """Fold one streamed message into the run state. Best-effort: a malformed
    message is logged and skipped, never propagated into the caller's stream."""
    try:
        kind = type(message).__name__
        if kind == "SystemMessage":
            _ingest_system(state, message)
        elif kind == "AssistantMessage":
            _ingest_assistant(state, message)
        elif kind == "UserMessage":
            _ingest_user(state, message)
        elif kind == "ResultMessage":
            _ingest_result(state, message)
    except Exception:
        logger.debug(
            "claude_agent_sdk: failed to ingest %s (non-fatal)",
            type(message).__name__, exc_info=True,
        )


def _ingest_system(state: _RunState, message: Any) -> None:
    if getattr(message, "subtype", None) != "init":
        return
    data = getattr(message, "data", None)
    if isinstance(data, dict):
        state.init_data = data
        if state.session_id is None:
            state.session_id = data.get("session_id")


def _ingest_assistant(state: _RunState, message: Any) -> None:
    """One AssistantMessage = one model turn = one LlmCallRecord. Its
    ToolUseBlocks become tool_calls on the record and open TOOL spans that the
    following UserMessage's ToolResultBlocks close."""
    model = getattr(message, "model", None)
    content = getattr(message, "content", None) or []
    text_parts: List[str] = []
    tool_uses: List[Tuple[Optional[str], Optional[str], Any]] = []
    for block in content:
        bkind = type(block).__name__
        if bkind == "TextBlock":
            txt = getattr(block, "text", None)
            if txt:
                text_parts.append(txt)
        elif bkind == "ToolUseBlock":
            tool_uses.append(
                (getattr(block, "id", None), getattr(block, "name", None), getattr(block, "input", None))
            )
        # ThinkingBlock and other block types carry no trace-relevant fields.

    text_joined = "\n".join(text_parts).strip()
    rec = LlmCallRecord(
        span_id=uuid4(),
        agent_name=state.agent_name,
        # A turn that asks for tools is planning; a pure-text turn is responding.
        call_role=CallRole.PLANNER if tool_uses else CallRole.RESPONDER,
        provider="anthropic",
        model_name=str(model) if model else None,
        output={"role": "assistant", "content": text_joined} if text_joined else None,
        finish_reason=FinishReason.TOOL_CALLS if tool_uses else FinishReason.STOP,
        started_at=_now(),
        ended_at=_now(),
    )
    if text_joined:
        # Last assistant text seen so far; ResultMessage.result overrides at finalize.
        state.final_output_preview = text_joined[:500]

    for tid, tname, tinput in tool_uses:
        name = str(tname) if tname else "tool"
        tc = ToolCallRecord(tool_name=name, args=tinput if isinstance(tinput, dict) else {})
        rec.tool_calls.append(tc)
        span = TraceSpan(
            span_type=SpanType.TOOL,
            name=name,
            started_at=_now(),
            input_preview=_preview(tinput),
        )
        if tid:
            state.pending_tools[tid] = (tc, span, _now())
        else:
            # No id to match a result against — close the span now.
            span.ended_at = _now()
            state.spans.append(span)

    state.llm_calls.append(rec)


def _ingest_user(state: _RunState, message: Any) -> None:
    """A streamed UserMessage carries the ToolResultBlocks for the prior turn's
    tool calls. (The initial user prompt is the input, not a streamed message.)"""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return
    for block in content:
        if type(block).__name__ != "ToolResultBlock":
            continue
        tuid = getattr(block, "tool_use_id", None)
        pending = state.pending_tools.pop(tuid, None) if tuid else None
        if pending is None:
            continue
        tc, span, started = pending
        ended = _now()
        result_content = getattr(block, "content", None)
        tc.result = result_content
        tc.latency_ms = int((ended - started).total_seconds() * 1000)
        span.ended_at = ended
        span.output_preview = _preview(result_content)
        # A tool's is_error marks that tool/span ERROR, but does NOT fail the run:
        # the agent may recover and still succeed. ResultMessage.is_error is the
        # authoritative run status.
        if getattr(block, "is_error", False):
            tc.status = Status.ERROR
            span.status = Status.ERROR
        state.spans.append(span)


def _ingest_result(state: _RunState, message: Any) -> None:
    state.ended_at = _now()
    if state.session_id is None:
        state.session_id = getattr(message, "session_id", None)
    result_text = getattr(message, "result", None)
    if isinstance(result_text, str) and result_text:
        state.final_output_preview = result_text[:500]
    if getattr(message, "is_error", False):
        state.status = Status.ERROR
    # Usage + cost are cumulative for the whole run (per-turn usage isn't exposed
    # in the stream), so attach the totals to the final LLM call.
    if state.llm_calls:
        last = state.llm_calls[-1]
        inp, out = _extract_usage(getattr(message, "usage", None))
        if inp is not None:
            last.input_tokens = inp
        if out is not None:
            last.output_tokens = out
        cost = getattr(message, "total_cost_usd", None)
        if cost is not None:
            last.cost_usd = cost


# ── Manifest + finalize ─────────────────────────────────────────────

def _build_manifest(state: _RunState) -> Dict[str, Any]:
    """Manifest-relevant config from options (preferred) + the init message."""
    info: Dict[str, Any] = {}
    opts = state.options
    init = state.init_data or {}

    model = _opt(opts, "model") or init.get("model")
    if model:
        info["model"] = {"name": str(model), "provider": "anthropic"}

    system_prompt = _opt(opts, "system_prompt")
    if isinstance(system_prompt, str) and system_prompt.strip():
        info["prompts"] = {"system": system_prompt.strip()}

    tools = _opt(opts, "allowed_tools") or init.get("tools")
    if isinstance(tools, (list, tuple)):
        entries = [{"name": str(t)} for t in tools if t]
        if entries:
            info["tools"] = entries

    agents = _opt(opts, "agents")
    if isinstance(agents, dict) and agents:
        info["subagents"] = [{"name": str(k)} for k in agents]

    return info


def _maybe_register_manifest(state: _RunState) -> Optional[str]:
    """Register the agent's manifest if its shape is new."""
    global _manifest_id
    from . import _config
    if not _config._is_enabled():
        return _manifest_id

    man = _build_manifest(state)
    if not (man.get("tools") or man.get("prompts") or man.get("model") or man.get("subagents")):
        return _manifest_id

    snapshot = extract_from_config(
        agent_name=state.agent_name or "unknown",
        tools=man.get("tools"),
        prompts=man.get("prompts"),
        models={"default": man["model"]} if man.get("model") else None,
        subagents=man.get("subagents"),
    )
    with _manifest_lock:
        if not _manifest_tracker.check_and_update(snapshot):
            return _manifest_id  # same hash, already registered
        try:
            client = _config._get_client()
            result = client.register_manifest(snapshot)
            _manifest_id = result.get("manifest_id", snapshot.id)
            logger.info(
                "Registered Claude Agent SDK manifest %s (hash=%s, components=%d)",
                _manifest_id, snapshot.manifest_hash[:12], len(snapshot.components),
            )
        except Exception:
            logger.warning(
                "Failed to register Claude Agent SDK manifest, continuing", exc_info=True
            )
            _manifest_id = snapshot.id
    return _manifest_id


def _finalize(state: _RunState) -> None:
    """Build + queue the RunTrace. Idempotent via ``state.finalized`` so the
    ResultMessage path and the stream-teardown fallback never double-send."""
    if state.finalized:
        return
    state.finalized = True

    from . import _config
    if not _config._is_enabled():
        return

    manifest_id = _maybe_register_manifest(state)
    try:
        client = _config._get_client()
        config = _config._get_config()
        trace = RunTrace(
            id=state.trace_id,
            project=state.project or (config.project if config else None),
            agent_name=state.agent_name,
            parent_trace_id=state.parent_trace_id,
            session_id=state.session_id,
            status=state.status,
            source_type="production",
            started_at=state.started_at,
            ended_at=state.ended_at or _now(),
            user_input_preview=state.user_input_preview,
            final_output_preview=state.final_output_preview,
            error_message=state.error_message,
            spans=list(state.spans),
            llm_calls=list(state.llm_calls),
            manifest_id=manifest_id,
        )
        _config._sender.submit(client.ingest_trace, trace)
        logger.debug(
            "Queued Claude Agent SDK trace %s (%d spans, %d llm_calls, manifest=%s)",
            trace.id, len(trace.spans), len(trace.llm_calls), trace.manifest_id or "none",
        )
    except Exception:
        logger.exception("Failed to queue Claude Agent SDK trace %s", state.trace_id)


async def _finalize_off_loop(state: _RunState) -> None:
    """Finalize without blocking the event loop on manifest registration (a
    sync network round-trip), mirroring adk.py's executor hop."""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _finalize, state)
    except RuntimeError:
        _finalize(state)


# ── Public API ──────────────────────────────────────────────────────

async def trace_stream(
    stream: AsyncIterator[Any],
    *,
    agent_name: Optional[str] = None,
    user_input: Optional[str] = None,
    project: Optional[str] = None,
    parent_trace_id: Optional[str] = None,
    options: Any = None,
) -> AsyncIterator[Any]:
    """Wrap a Claude Agent SDK message stream, emitting one DecimalAI RunTrace.

    Yields every message from ``stream`` unchanged while accumulating the trace;
    finalizes (and queues the trace) when the stream ends — on normal
    completion, on an exception from the run, or on early consumer break.

    Args:
        stream: The async iterator returned by ``claude_agent_sdk.query(...)``
            (or ``ClaudeSDKClient.receive_response()``).
        agent_name: Agent label for the trace. Falls back to the global
            ``instrument()`` name, then ``"claude-agent"``.
        user_input: The prompt text, for ``user_input_preview``.
        project: Optional project grouping.
        parent_trace_id: Link this run to a parent orchestrator trace.
        options: The ``ClaudeAgentOptions`` used for the run — read for manifest
            extraction (model / system_prompt / allowed_tools / agents).
    """
    state = _RunState(
        agent_name=agent_name, project=project, parent_trace_id=parent_trace_id,
        user_input=user_input, options=options,
    )
    try:
        async for message in stream:
            _ingest_message(state, message)
            if type(message).__name__ == "ResultMessage":
                await _finalize_off_loop(state)
            yield message
    except BaseException as exc:
        # A real error from the run (not a clean early break) → mark the trace.
        if not isinstance(exc, GeneratorExit):
            state.status = Status.ERROR
            if state.error_message is None:
                state.error_message = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Fallback for any path that didn't see a ResultMessage (exception,
        # early break, or a stream that simply ended). No-ops if already done.
        if not state.finalized:
            _finalize(state)


def traced_query(
    *,
    prompt: Any,
    options: Any = None,
    agent_name: Optional[str] = None,
    project: Optional[str] = None,
    parent_trace_id: Optional[str] = None,
) -> AsyncIterator[Any]:
    """Convenience wrapper around ``claude_agent_sdk.query`` with tracing.

    Equivalent to ``trace_stream(query(prompt=prompt, options=options), ...)``.
    Returns an async iterator of the same messages ``query`` yields.

    Example::

        from decimalai.claude_agent_sdk import traced_query
        async for message in traced_query(prompt="Fix the bug", agent_name="dev"):
            ...
    """
    import claude_agent_sdk

    stream = claude_agent_sdk.query(prompt=prompt, options=options)
    return trace_stream(
        stream,
        agent_name=agent_name,
        user_input=prompt if isinstance(prompt, str) else None,
        project=project,
        parent_trace_id=parent_trace_id,
        options=options,
    )


def instrument(agent_name: Optional[str] = None) -> None:
    """Install DecimalAI tracing globally for the Claude Agent SDK.

    Monkeypatches ``claude_agent_sdk.query`` so every call is wrapped with
    :func:`trace_stream`. Idempotent. Call before importing ``query`` by name
    (``from claude_agent_sdk import query``) so the bound symbol is the patched
    one — or just call ``query`` off the module (``claude_agent_sdk.query``).

    Args:
        agent_name: Default agent name for traces whose stream supplies none.
    """
    global _install_agent_name, _query_patched
    _install_agent_name = agent_name

    if _query_patched:
        return

    try:
        import claude_agent_sdk
    except ImportError:
        logger.warning(
            "decimalai.claude_agent_sdk.instrument() called but claude-agent-sdk is "
            "not installed; tracing not active. "
            "Install it with: pip install \"decimalai[claude-agent-sdk]\""
        )
        return

    original_query = claude_agent_sdk.query

    def patched_query(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        stream = original_query(*args, **kwargs)
        prompt = kwargs.get("prompt")
        if prompt is None and args:
            prompt = args[0]
        return trace_stream(
            stream,
            agent_name=_install_agent_name,
            user_input=prompt if isinstance(prompt, str) else None,
            options=kwargs.get("options"),
        )

    claude_agent_sdk.query = patched_query  # type: ignore[assignment]
    _query_patched = True
    logger.info(
        "DecimalAI Claude Agent SDK tracing installed globally (agent_name=%s)",
        agent_name,
    )


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
        "decimalai.claude_agent_sdk.install() is deprecated; use "
        "decimalai.claude_agent_sdk.instrument() instead. It turns on tracing for claude_agent_sdk "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
