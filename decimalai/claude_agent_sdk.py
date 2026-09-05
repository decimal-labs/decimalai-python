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
import json
import logging
import threading
import warnings
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, NamedTuple, Optional, Tuple
from uuid import UUID, uuid4

from .schema.common import CallRole, FinishReason, SpanType, Status
from .schema.manifest import ManifestTracker, extract_from_config
from .schema.trace import LlmCallRecord, RunTrace, ToolCallRecord, TraceSpan

logger = logging.getLogger("decimalai.claude_agent_sdk")

# ── Module-level manifest state (mirrors openai_agents.py) ─────────
# A manifest is registered once per distinct (agent, tools, prompt, model,
# subagents) shape and reused across traces until that shape changes.
_manifest_tracker = ManifestTracker()
_manifest_id: Optional[str] = None
_manifest_lock = threading.Lock()

# Per-agent manifest id + hash. `_manifest_id` alone is a single process-global
# slot, and `ManifestTracker` is a single slot whose stored hash does NOT
# include the agent name — so in a process running two differently-named agents
# the second agent deduped against the first and its traces were stamped with
# the FIRST agent's manifest id. The manifest→trace join then attributes one
# agent's runs to another agent's contract. Key both by agent_name;
# `_manifest_id` is kept as the last-registered value for back-compat with
# callers (and tests) that read it.
_manifest_ids: Dict[str, str] = {}
_manifest_hashes: Dict[str, str] = {}

#: agent -> (snapshot, refusing exception) for a registration the platform REFUSED.
_pending_manifests: Dict[str, tuple] = {}


def _reregister_refused(agent: str) -> Optional[str]:
    """Re-attempt ONE agent's refused registration, on the sender's thread."""
    # `_config` is imported per-function in this module, never at module level.
    from . import _config

    with _manifest_lock:
        pending = _pending_manifests.get(agent)
    if not pending:
        return _manifest_ids.get(agent)
    snapshot, exc = pending
    if exc is not None and ("401" in str(exc) or "403" in str(exc)):
        return None
    try:
        result = _config._get_client().register_manifest(snapshot)
    except Exception:  # noqa: BLE001 — the sender retries; raising would drop it
        return None
    manifest_id = result.get("manifest_id")
    if not manifest_id:
        return None
    with _manifest_lock:
        _manifest_ids[agent] = manifest_id
        _manifest_hashes[agent] = snapshot.manifest_hash
        _pending_manifests.pop(agent, None)
    return manifest_id


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


def _text_content(value: Any) -> str:
    """Flatten a message/block payload down to the TEXT the model saw.

    ``str(value)`` is the wrong last resort here: a list of SDK block objects
    stringifies to ``[TextBlock(text='…')]``, and a consumer reading
    ``rendered_input[i].content`` to show the conversation would get a Python
    repr instead of the prompt. Walk the shapes the Claude wire format actually
    uses (str, block dicts, lists of either, objects carrying ``.text``) and
    only fall back to ``str`` for a genuine scalar.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "thinking"):
            if key in value:
                nested = _text_content(value[key])
                if nested:
                    return nested
        return ""
    if isinstance(value, (list, tuple)):
        parts = [_text_content(v) for v in value]
        return "\n".join(p for p in parts if p)
    for attr in ("text", "content", "thinking"):
        if hasattr(value, attr):
            nested = _text_content(getattr(value, attr))
            if nested:
                return nested
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _display(value: Any) -> Any:
    """The most readable form of ``value`` that is still not a Python repr.

    Text if there is text; otherwise a JSON dump (structured content still
    carries the information, and a consumer can read it); otherwise the value
    itself for ``_preview`` to stringify.
    """
    text = _text_content(value)
    if text:
        return text
    if isinstance(value, (dict, list, tuple)):
        try:
            # `default` never emits `<pkg.Class object at 0x…>`: a repr in a
            # preview is exactly what the contract calls junk.
            return json.dumps(value, default=lambda o: type(o).__name__)
        except Exception:
            return None
    return value


def _transcript_text(messages: List[Dict[str, Any]], max_len: int = 500) -> Optional[str]:
    """A ``role: text`` preview of the conversation as rendered to the model."""
    lines: List[str] = []
    for entry in messages:
        text = _text_content(entry.get("content"))
        if text:
            lines.append(f"{entry.get('role', 'user')}: {text}")
    return _preview("\n".join(lines), max_len)


def _opt(options: Any, name: str) -> Any:
    """Read a ``ClaudeAgentOptions`` field (dataclass or dict), never raising."""
    if options is None:
        return None
    if isinstance(options, dict):
        return options.get(name)
    return getattr(options, name, None)


class _Usage(NamedTuple):
    """One usage frame, kept SPLIT — see :func:`_extract_usage`."""

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cache_read_tokens: Optional[int]
    cache_creation_tokens: Optional[int]


def _extract_usage(usage: Any) -> _Usage:
    """Split a ResultMessage/AssistantMessage usage payload into four counts.

    Claude reports Anthropic-shaped token names (``input_tokens`` /
    ``output_tokens`` / ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``), as a dict from the CLI JSON or as an
    object.

    BEHAVIOUR CHANGE (2026-08-22) — ``input_tokens`` MEANS SOMETHING NEW HERE.
    ---------------------------------------------------------------------
    This function used to return ONE input number:

        inp += cache_read_input_tokens + cache_creation_input_tokens

    i.e. the effective context size, chosen for parity with the OpenAI handler
    (whose ``prompt_tokens`` already includes cached tokens). It now returns
    ``input_tokens`` EXACTLY as Anthropic reported it — the UNCACHED REMAINDER —
    and carries the two cache counts as their own fields.

    Why the fold had to go: DecimalAI injects a query-routed skill menu at
    position ZERO of the system prompt, and it is rebuilt per query. Varying
    bytes at position zero defeat a provider's prefix cache for EVERYTHING
    behind them, so a customer's otherwise-cacheable system prompt becomes a
    full miss on every call. That is a cost regression DecimalAI causes, and
    the folded number is precisely the number that cannot show it: a call that
    went from 180k cached + 4k fresh to 184k fresh sums to 184k either way.
    Summing three counts is not lossy-but-cheap, it is lossy in the one
    dimension the product exists to measure.

    What downstream changes, checked before landing:
      * ``LlmCallRecord.input_tokens`` on the Claude path drops by the cached
        amount — a warm Claude Code run now reads as a few-K-token call, which
        is what the provider actually charged fresh input for.
      * The platform's ``estimate_cost`` (input+output rates only, no cache
        rate) therefore ESTIMATES LOWER on this path than it did. That is a
        known, deliberate consequence: the old number over-billed cache reads
        at ~10x their real rate, and there is no honest cache rate to apply
        until ``MODEL_PRICING`` grows one. Cost is not silently patched here.
      * ``_ingest_result``'s remainder allocation sums ``input_tokens`` across
        turns; both sides of that comparison move together, so it is unaffected.
      * The conformance driver (tests/conformance/drivers/claude_agent_sdk.py)
        emits per-turn usage with no cache fields, so the contract's
        "input_tokens is a positive int" row is unchanged.

    Do NOT add these back together to get "total input". On Anthropic they are
    additive to ``input_tokens``; on OpenAI ``cached_tokens`` is already inside
    it. A provider-blind sum is wrong for one of them.
    """
    if usage is None:
        return _Usage(None, None, None, None)

    def _field(name: str) -> Optional[int]:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        # `bool` is a subclass of `int`; True must not become a token count of 1.
        if isinstance(value, bool):
            return None
        return value if isinstance(value, int) else None

    return _Usage(
        input_tokens=_field("input_tokens"),
        output_tokens=_field("output_tokens"),
        # Reported 0 is KEPT as 0 (a measured cache miss), not normalised to
        # None (never measured). The distinction survives all the way into the
        # platform's `llm_call.cache_read_tokens` column.
        cache_read_tokens=_field("cache_read_input_tokens"),
        cache_creation_tokens=_field("cache_creation_input_tokens"),
    )


class _RunState:
    """Per-stream trace accumulator. One ``query()`` invocation → one RunTrace."""

    __slots__ = (
        "trace_id", "agent_name", "project", "parent_trace_id",
        "started_at", "ended_at", "user_input_preview", "final_output_preview",
        "llm_calls", "spans", "pending_tools", "transcript", "root_span",
        "saw_turn_usage",
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
        # The conversation AS RENDERED to the model, grown message by message.
        # Snapshotted onto each LlmCallRecord.rendered_input, so a trace carries
        # the prompt the turn actually saw rather than nothing at all.
        self.transcript: List[Dict[str, Any]] = []
        system_prompt = _opt(options, "system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            self.transcript.append({"role": "system", "content": system_prompt})
        if isinstance(user_input, str) and user_input.strip():
            self.transcript.append({"role": "user", "content": user_input})
        # The run's root span. Every LLM turn hangs off it, and every tool span
        # hangs off the turn that asked for the tool, so the waterfall shows the
        # shape of the run instead of a flat list of tool calls.
        self.root_span = TraceSpan(
            span_type=SpanType.AGENT,
            name="claude_agent_sdk.query",
            started_at=self.started_at,
            input_preview=self.user_input_preview,
        )
        # True once any AssistantMessage reported its own per-turn usage, which
        # makes the ResultMessage's CUMULATIVE totals a double count.
        self.saw_turn_usage = False
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
    started = _now()
    # What this turn was shown: everything on the transcript BEFORE it.
    rendered_input = [dict(entry) for entry in state.transcript]
    # Per-turn usage. The CLI reports usage on every assistant frame, so each
    # model turn can carry its own token counts; ResultMessage.usage is the
    # run TOTAL and is only a fallback (see _ingest_result).
    turn_usage = _extract_usage(getattr(message, "usage", None))
    # Keyed on input/output only, exactly as before the cache split landed:
    # `saw_turn_usage` gates the run-total REDISTRIBUTION below, and only
    # input/output are ever redistributed. Widening it to "any of the four"
    # would let a frame carrying cache counts but no input/output suppress the
    # run-total fallback and leave the trace with no token counts at all.
    if turn_usage.input_tokens is not None or turn_usage.output_tokens is not None:
        state.saw_turn_usage = True

    # One span per model turn, hung off the run root, sharing the LLM call's
    # span_id so the call and the span are the same node in the waterfall.
    llm_span = TraceSpan(
        parent_span_id=state.root_span.id,
        span_type=SpanType.LLM,
        name=str(model) if model else "llm",
        started_at=started,
        ended_at=started,
        input_preview=_transcript_text(rendered_input),
        output_preview=_preview(text_joined) if text_joined else None,
    )
    rec = LlmCallRecord(
        span_id=llm_span.id,
        agent_name=state.agent_name,
        # A turn that asks for tools is planning; a pure-text turn is responding.
        call_role=CallRole.PLANNER if tool_uses else CallRole.RESPONDER,
        provider="anthropic",
        model_name=str(model) if model else None,
        rendered_input=rendered_input or None,
        output={"role": "assistant", "content": text_joined} if text_joined else None,
        finish_reason=FinishReason.TOOL_CALLS if tool_uses else FinishReason.STOP,
        # Verbatim from the provider — `input_tokens` is Anthropic's UNCACHED
        # remainder and the two cache counts sit alongside it, never folded in.
        input_tokens=turn_usage.input_tokens,
        output_tokens=turn_usage.output_tokens,
        cache_read_tokens=turn_usage.cache_read_tokens,
        cache_creation_tokens=turn_usage.cache_creation_tokens,
        started_at=started,
        ended_at=started,
    )
    if text_joined:
        # Last assistant text seen so far; ResultMessage.result overrides at finalize.
        state.final_output_preview = text_joined[:500]

    turn_entry: Dict[str, Any] = {"role": "assistant", "content": text_joined}
    for tid, tname, tinput in tool_uses:
        name = str(tname) if tname else "tool"
        tc = ToolCallRecord(tool_name=name, args=tinput if isinstance(tinput, dict) else {})
        rec.tool_calls.append(tc)
        turn_entry.setdefault("tool_calls", []).append(
            {"id": tid, "name": name, "arguments": tinput if isinstance(tinput, dict) else {}}
        )
        span = TraceSpan(
            # The tool was requested BY this model turn — parent it there, not
            # at the root, so a multi-turn run reads as a tree.
            parent_span_id=llm_span.id,
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

    state.transcript.append(turn_entry)
    state.spans.append(llm_span)
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
        span.output_preview = _preview(_display(result_content))
        # A tool's is_error marks that tool/span ERROR, but does NOT fail the run:
        # the agent may recover and still succeed. ResultMessage.is_error is the
        # authoritative run status.
        is_error = bool(getattr(block, "is_error", False))
        if is_error:
            tc.status = Status.ERROR
            span.status = Status.ERROR
        # The requesting model turn owns this tool, so its span has to cover it —
        # otherwise the child outlives the parent in the waterfall.
        for candidate in state.spans:
            if candidate.id == span.parent_span_id:
                if candidate.ended_at is None or candidate.ended_at < ended:
                    candidate.ended_at = ended
                break
        # The result goes back to the model as a user turn (Anthropic wire
        # shape), so the NEXT turn's rendered_input must contain it.
        state.transcript.append({
            "role": "user",
            "content": _text_content(result_content),
            "tool_use_id": tuid,
            **({"is_error": True} if is_error else {}),
        })
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
    if not state.llm_calls:
        return

    run_usage = _extract_usage(getattr(message, "usage", None))
    inp, out = run_usage.input_tokens, run_usage.output_tokens
    cost = getattr(message, "total_cost_usd", None)

    if not state.saw_turn_usage:
        # Nothing reported per-turn usage, so the run totals are all there is:
        # attach them to the final LLM call, as this adapter always has.
        last = state.llm_calls[-1]
        if inp is not None:
            last.input_tokens = inp
        if out is not None:
            last.output_tokens = out
        # The cache counts come along on the same frame and are RUN TOTALS.
        # `saw_turn_usage` is keyed on input/output only, so this branch can be
        # reached while per-turn frames DID report cache counts and those are
        # already recorded on individual calls. Stamping the run total on the
        # last call as well makes any sum across calls count the cache twice.
        # So: only fall back to the totals when nothing recorded a cache count.
        # A reported 0 is written as 0 (measured cache miss); an absent field
        # leaves the record's None (never measured) alone.
        already_have_cache = any(
            c.cache_read_tokens is not None or c.cache_creation_tokens is not None
            for c in state.llm_calls
        )
        if not already_have_cache:
            if run_usage.cache_read_tokens is not None:
                last.cache_read_tokens = run_usage.cache_read_tokens
            if run_usage.cache_creation_tokens is not None:
                last.cache_creation_tokens = run_usage.cache_creation_tokens
        if cost is not None:
            last.cost_usd = cost
        return

    # Per-turn usage IS present (the CLI reports usage on every assistant
    # frame), and ResultMessage.usage is the run TOTAL — adding it to the last
    # call on top of that call's own tokens would count the run twice. Keep the
    # per-turn numbers, and only fill a turn the CLI left blank with whatever
    # the totals have not already accounted for.
    known_in = sum(c.input_tokens or 0 for c in state.llm_calls)
    known_out = sum(c.output_tokens or 0 for c in state.llm_calls)
    for call in state.llm_calls:
        if call.input_tokens is None and inp is not None and inp > known_in:
            call.input_tokens = inp - known_in
            known_in = inp
        if call.output_tokens is None and out is not None and out > known_out:
            call.output_tokens = out - known_out
            known_out = out

    # The cache counts are deliberately NOT redistributed the same way. They
    # arrive on the SAME usage object as input_tokens on every assistant frame,
    # so a turn that has input_tokens has its cache counts too; a turn that has
    # neither is a turn the CLI reported nothing for, and inventing a cache
    # split for it would be fabrication rather than inference. A per-turn
    # None here means "this turn's cache behaviour is unknown", which is the
    # honest answer and the one the platform's NULL column is built to hold.

    # ``total_cost_usd`` is also a run total. Splitting it across the turns by
    # their token share keeps per-call cost consistent with per-call tokens;
    # parking the whole run's cost on the last call (what this adapter used to
    # do) reads as one enormous final turn in any cost-by-call view.
    if cost is None:
        return
    weights = [(c.input_tokens or 0) + (c.output_tokens or 0) for c in state.llm_calls]
    total_weight = sum(weights)
    if total_weight <= 0:
        state.llm_calls[-1].cost_usd = cost
        return
    assigned = 0.0
    for call, weight in zip(state.llm_calls[:-1], weights[:-1]):
        share = cost * weight / total_weight
        call.cost_usd = share
        assigned += share
    # The remainder, so the parts always add back up to the reported total.
    state.llm_calls[-1].cost_usd = cost - assigned


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
    """Register the agent's manifest if its shape is new.

    Dedup is keyed by (agent_name, manifest_hash). ``ManifestTracker`` is a
    single slot and the hash it stores does not include the agent name, so two
    agents with the same shape in one process used to dedup against each other:
    the second agent registered nothing and its traces went out stamped with the
    FIRST agent's manifest id.
    """
    global _manifest_id
    from . import _config

    agent = state.agent_name or "unknown"
    with _manifest_lock:
        # A caller (notably a test fixture) that swapped in a fresh
        # ManifestTracker is asking for registration state to be forgotten;
        # honour that for the per-agent maps too, or the reset is a no-op.
        # In a live process this is dead after the first registration, because
        # the tracker only has a null hash while nothing has been registered.
        if _manifest_tracker.last_hash is None and _manifest_hashes:
            _manifest_hashes.clear()
            _manifest_ids.clear()

    if not _config._is_enabled():
        return _manifest_ids.get(agent)

    man = _build_manifest(state)
    if not (man.get("tools") or man.get("prompts") or man.get("model") or man.get("subagents")):
        # Nothing structural to declare — point at whatever this agent is
        # already registered under rather than at another agent's manifest.
        return _manifest_ids.get(agent)

    snapshot = extract_from_config(
        agent_name=agent,
        tools=man.get("tools"),
        prompts=man.get("prompts"),
        models={"default": man["model"]} if man.get("model") else None,
        subagents=man.get("subagents"),
    )
    with _manifest_lock:
        known = _manifest_ids.get(agent)
        if known and _manifest_hashes.get(agent) == snapshot.manifest_hash:
            return known  # same agent, same shape — already registered
        _manifest_tracker.check_and_update(snapshot)
        try:
            client = _config._get_client()
            result = client.register_manifest(snapshot)
            manifest_id = result.get("manifest_id", snapshot.id)
        except Exception as exc:
            # Leave this agent's hash unset so the next trace retries; a
            # transient blip must not permanently stop it declaring a manifest.
            logger.warning(
                "Failed to register Claude Agent SDK manifest for %s, continuing",
                agent, exc_info=True,
            )
            # Hand the send side something to retry WITH. Without it the trace goes
            # out with no manifest_id at all and a strict backend answers 400
            # "manifest_id is required" — the same loss the other adapters took under
            # the other 400. The caller-thread attempt runs in front of the user's
            # agent and cannot outlast a no-available-instance window, so the retry
            # belongs on the background sender.
            _pending_manifests[agent] = (snapshot, exc)
            return _manifest_ids.get(agent)
        _manifest_ids[agent] = manifest_id
        _manifest_hashes[agent] = snapshot.manifest_hash
        _pending_manifests.pop(agent, None)
        _manifest_id = manifest_id
        logger.info(
            "Registered Claude Agent SDK manifest %s for %s (hash=%s, components=%d)",
            manifest_id, agent, snapshot.manifest_hash[:12], len(snapshot.components),
        )
        return manifest_id


def _closed_spans(state: _RunState) -> List[TraceSpan]:
    """The run's spans, root first, with anything still open closed at run end.

    A tool whose result never arrived (the CLI died mid-stream, the consumer
    broke out early) leaves its span open; a span with no ``ended_at`` renders
    as a zero-width bar, which reads as "this step never happened" rather than
    "this step never finished".
    """
    ended = state.ended_at or _now()
    root = state.root_span
    root.ended_at = ended
    root.status = state.status
    if root.output_preview is None:
        root.output_preview = _preview(state.final_output_preview)
    spans = [root, *state.spans]
    for span in spans:
        if span.ended_at is None:
            span.ended_at = ended
        if span.started_at is None:
            span.started_at = state.started_at
    return spans


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
        spans = _closed_spans(state)
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
            spans=spans,
            llm_calls=list(state.llm_calls),
            manifest_id=manifest_id,
        )
        _agent = trace.agent_name
        if _agent in _pending_manifests:
            _config.submit_trace_pending_manifest(
                client, trace, lambda: _reregister_refused(_agent)
            )
        else:
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
