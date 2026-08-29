"""Generic tracing for any framework — decorator + context manager.

Usage:

    @decimalai.trace(agent_name="my-agent")
    def run_agent(query):
        resp = openai.chat.completions.create(...)
        decimalai.log_llm_call(model="gpt-4o", input=msgs, output=resp)
        return resp.choices[0].message.content

    # Or with a context manager:
    with decimalai.start_trace(agent_name="my-agent") as trace:
        trace.log_llm_call(model="gpt-4o", input=msgs, output=resp)
        trace.log_tool_call(name="search", input=q, output=results)
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from .schema.common import CallRole, FinishReason, SpanType, Status
from .schema.manifest import ManifestTracker, extract_from_config
from .schema.trace import LlmCallRecord, RunTrace, TraceSpan

logger = logging.getLogger("decimalai")

# ContextVar for async-safe trace context (replaces threading.local)
_current_trace: ContextVar[Optional["TraceContext"]] = ContextVar(
    "decimal_current_trace", default=None
)

# Global manifest state for the generic tracer
_manifest_tracker = ManifestTracker()
_manifest_id: Optional[str] = None
_manifest_lock = __import__("threading").Lock()


class TraceContext:
    """Active trace context returned by ``start_trace()``.

    Provides ``log_llm_call()`` and ``log_tool_call()`` for manual instrumentation.
    """

    def __init__(
        self,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        auto_send: bool = True,
        session_metadata: Optional[Dict[str, Any]] = None,
        turn_index: Optional[int] = None,
        parent_trace_id: Optional[str] = None,
        subagents: Optional[List[Dict[str, Any]]] = None,
    ):
        # Validate agent_name client-side. Without this check, a 500-char
        # agent_name is accepted by the SDK, sent to the backend, rejected
        # by the backend's VARCHAR(255) constraint, and dropped. Raising
        # ValueError here means the user sees the problem before any
        # backend round-trip, instead of losing traces to a rejection they
        # never see.
        if agent_name is not None and len(agent_name) > 255:
            raise ValueError(
                f"agent_name must be ≤255 characters (got {len(agent_name)}). "
                f"Backend stores agent_name as VARCHAR(255); oversized names "
                f"would be silently rejected."
            )
        if agent_name is not None and not agent_name.strip():
            raise ValueError(
                "agent_name must contain non-whitespace characters; "
                "got empty/whitespace-only value. "
                "Use a stable identifier like 'support-bot' or 'rag-pipeline'."
            )
        self.agent_name = agent_name
        self.session_id = session_id
        self.auto_send = auto_send
        self.session_metadata = session_metadata or {}
        self.turn_index = turn_index
        self.parent_trace_id = parent_trace_id
        self.subagents = list(subagents) if subagents else None

        self._trace_id = uuid4()
        self._started_at = datetime.now(timezone.utc)
        self._spans: List[TraceSpan] = []
        self._llm_calls: List[LlmCallRecord] = []
        self._user_input: Optional[str] = None
        self._final_output: Optional[str] = None
        self._status = Status.SUCCESS
        self._error_message: Optional[str] = None
        # Manifest auto-detection accumulators
        self._seen_models: Dict[str, Dict[str, Any]] = {}  # node_name -> config
        self._seen_tools: Dict[str, Dict[str, Any]] = {}  # tool_name -> {name, ...}
        self._active_skills: Dict[str, Optional[str]] = {}  # skill_name -> optional hash
        self._skills_registry: Optional[List[Dict[str, Any]]] = None
        # SkillRouter: the `rt_<24-hex>` routing-decision id for this trace.
        # Stamped onto the RunTrace so the platform's
        # `routing_decision × trace_skill_activation` join (offered-vs-
        # activated) can close on the native @decimalai.trace path — the
        # framework adapters (langchain/anthropic/pydantic_ai) already
        # carry it; this is the parity fix for the generic tracer.
        self._routing_id: Optional[str] = None
        # Skill discovery telemetry. `log_skill_offered` records
        # names the agent COULD have used (system-prompt registry); the
        # Skill Rater uses this to compute discoverability gap. `log_skill_loaded`
        # records names the agent actually READ. Using sets so repeated
        # calls within a turn are idempotent.
        self._skills_offered_in_prompt: set[str] = set()
        self._skills_loaded_by_agent: set[str] = set()
        # 'delivered' = the full skill BODY reached the model (Router body
        # injection or a load_skill serve) — a rung between
        # offered (menu row only) and activated. Deliberately NOT written
        # into _active_skills: delivery is not activation.
        self._skills_delivered: set[str] = set()

    def log_llm_call(
        self,
        *,
        model: str,
        input: Optional[List[Dict[str, Any]]] = None,
        output: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cache_read_tokens: Optional[int] = None,
        cache_creation_tokens: Optional[int] = None,
        latency_ms: Optional[int] = None,
        temperature: Optional[float] = None,
        finish_reason: Optional[str] = None,
        call_role: str = "other",
        cost_usd: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        content_type: str = "text",
    ) -> None:
        """Log an LLM call within this trace.

        ``cache_read_tokens`` / ``cache_creation_tokens`` carry the provider's
        prompt-cache split. Pass them EXACTLY as the provider reported them and
        do not fold them into ``input_tokens`` — whether a prompt prefix stayed
        cacheable is the thing they exist to measure, and a summed number
        cannot answer it.

            Anthropic  ``usage.cache_read_input_tokens`` /
                       ``usage.cache_creation_input_tokens``. These are
                       ADDITIONAL to ``usage.input_tokens``, which is the
                       uncached remainder.
            OpenAI     ``usage.prompt_tokens_details.cached_tokens`` →
                       ``cache_read_tokens``. This is a SUBSET of
                       ``prompt_tokens``; leave ``cache_creation_tokens``
                       unset (the auto-cache reports no creation step).

        Leave a field UNSET when the provider did not report it. ``None`` and
        ``0`` are stored as different facts all the way into the platform:
        ``None`` means "never measured", ``0`` means "measured, and the cache
        was cold". Defaulting an unknown to ``0`` manufactures a cache miss.
        """
        now = datetime.now(timezone.utc)
        started = now
        if latency_ms:
            from datetime import timedelta

            started = now - timedelta(milliseconds=latency_ms)

        fr = None
        if finish_reason:
            try:
                fr = FinishReason(finish_reason)
            except ValueError:
                fr = None

        try:
            cr = CallRole(call_role)
        except ValueError:
            cr = CallRole.OTHER

        call = LlmCallRecord(
            model_name=model,
            provider=provider,
            rendered_input=input,
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            latency_ms=latency_ms,
            temperature=temperature,
            finish_reason=fr,
            call_role=cr,
            agent_name=self.agent_name,
            status=Status.SUCCESS,
            started_at=started,
            ended_at=now,
            cost_usd=cost_usd,
            response_format=response_format,
            content_type=content_type,
        )
        self._llm_calls.append(call)

        # Accumulate model for manifest auto-detection
        model_key = model or "default"
        if model_key not in self._seen_models:
            self._seen_models[model_key] = {
                "provider": provider or _infer_provider(model),
                "model": model,
                "temperature": temperature,
            }

    def log_tool_call(
        self,
        *,
        name: str,
        input: Any = None,
        output: Any = None,
        latency_ms: Optional[int] = None,
        status: str = "success",
    ) -> None:
        """Log a tool/function call within this trace."""
        now = datetime.now(timezone.utc)
        started = now
        if latency_ms:
            from datetime import timedelta

            started = now - timedelta(milliseconds=latency_ms)

        try:
            st = Status(status)
        except ValueError:
            st = Status.SUCCESS

        span = TraceSpan(
            span_type=SpanType.TOOL,
            name=name,
            status=st,
            started_at=started,
            ended_at=now,
            input_preview=str(input)[:200] if input else None,
            output_preview=str(output)[:200] if output else None,
        )
        self._spans.append(span)

        # Accumulate tool for manifest auto-detection (name only)
        if name not in self._seen_tools:
            self._seen_tools[name] = {"name": name}

    def set_input(self, text: str) -> None:
        """Set the user input preview for this trace."""
        self._user_input = str(text)[:200]

    def set_output(self, text: str) -> None:
        """Set the final output preview for this trace."""
        self._final_output = str(text)[:200]

    def set_session_metadata(self, metadata: Dict[str, Any]) -> None:
        """Set session-level metadata for multi-turn context."""
        self.session_metadata.update(metadata)

    def get_trace_id(self) -> str:
        """Return the trace ID as a string, for use as a child's parent_trace_id."""
        return str(self._trace_id)

    def log_skill_activation(self, *, name: str, hash: Optional[str] = None) -> None:
        """Record that a skill was activated during this trace.

        Args:
            name: Skill identifier (e.g. "code-review").
            hash: Optional content hash of the skill body.
        """
        self._active_skills[name] = hash

    def set_routing_id(self, routing_id: Optional[str]) -> None:
        """Record the SkillRouter routing-decision id for this trace.

        Args:
            routing_id: The `rt_<24-hex>` id returned by
                ``SkillRouter.build_prompt_fragment()``. Stamped onto the
                RunTrace so the platform can join the routing decision
                against the skills this trace actually activated.
        """
        self._routing_id = routing_id

    def log_skill_offered(self, *, names: List[str]) -> None:
        """Record skills whose descriptions were OFFERED to the agent.

        Use this for skills that appeared in the system prompt's skill
        registry but the agent may or may not have actually loaded /
        used. The Skill Rater computes the "discoverability gap" — how
        often a relevant skill was right there but went unread — from
        this signal.

        Args:
            names: List of skill identifiers offered (e.g. system prompt
                contained ``[code-review, pii-classifier, refund-policy]``).
        """
        for name in names:
            if isinstance(name, str) and name.strip():
                self._skills_offered_in_prompt.add(name.strip())

    def log_skill_delivered(self, *, names: List[str]) -> None:
        """Record skills whose full BODY reached the model.

        A rung above "offered" (menu row only) and below "activated":
        the SkillRouter's body injection auto-populates this. A
        delivered skill was necessarily also offered, so this implies
        offered — but it does NOT imply activation.

        Args:
            names: Skill identifiers whose bodies were injected.
        """
        for name in names:
            if isinstance(name, str) and name.strip():
                n = name.strip()
                self._skills_delivered.add(n)
                self._skills_offered_in_prompt.add(n)

    def log_skill_loaded(self, *, name: str, hash: Optional[str] = None) -> None:
        """Record that the agent actually READ a skill's body.

        Use this when the agent fetched the SKILL.md content (not just
        saw the one-line description in the registry). Semantically a
        superset of "offered" — a loaded skill was also offered, and
        the body serve means it was delivered.

        Args:
            name: Skill identifier whose body the agent read.
            hash: The body's ``content_hash``, as the platform returned it
                alongside the body. Optional and advisory: with it, the
                activation resolves to the skill VERSION the agent read;
                without it, the activation is recorded exactly as before, with
                a null hash. Never compute this yourself — an SDK-computed
                digest is not a version the platform ever minted.
        """
        if isinstance(name, str) and name.strip():
            n = name.strip()
            self._skills_loaded_by_agent.add(n)
            # A loaded skill is implicitly also offered + delivered.
            self._skills_offered_in_prompt.add(n)
            self._skills_delivered.add(n)
            # `setdefault`, not assignment: an explicit `log_skill_activation`
            # is the caller SAYING which version influenced the output, and a
            # rail observation must not overwrite a caller's own declaration.
            if isinstance(hash, str) and hash:
                self._active_skills.setdefault(n, hash)

    def build_trace(self) -> RunTrace:
        """Assemble the collected data into a RunTrace."""
        from . import _config

        config = _config._get_config()

        # Build active_skills list from collected activations
        active_skills_list = []
        for name, h in self._active_skills.items():
            entry: Dict[str, Any] = {"name": name}
            if h:
                entry["hash"] = h
            active_skills_list.append(entry)

        return RunTrace(
            id=self._trace_id,
            project=config.project,
            agent_name=self.agent_name,
            session_id=self.session_id,
            status=self._status,
            source_type="production",
            started_at=self._started_at,
            ended_at=datetime.now(timezone.utc),
            user_input_preview=self._user_input,
            final_output_preview=self._final_output,
            error_message=self._error_message,
            spans=list(self._spans),
            llm_calls=list(self._llm_calls),
            active_skills=active_skills_list,
            # SkillRouter: stamp the routing-decision id so the
            # offered-vs-activated join can close on the native path.
            routing_id=self._routing_id,
            # Sort for deterministic output (tests, diffs).
            skills_offered_in_prompt=sorted(self._skills_offered_in_prompt),
            skills_loaded_by_agent=sorted(self._skills_loaded_by_agent),
            skills_delivered=sorted(self._skills_delivered),
            session_metadata=self.session_metadata,
            turn_index=self.turn_index,
            manifest_id=_manifest_id,
            parent_trace_id=self.parent_trace_id,
        )

    def _send(self) -> None:
        """Send the trace to the DecimalAI backend (via background sender)."""
        from . import _config

        if not _config._is_enabled():
            return

        # Auto-register manifest before sending trace
        self._maybe_register_manifest()

        # Infer the offered/delivered rungs for skills the SDK did NOT inject
        self._infer_skill_rungs_from_prompt()

        try:
            client = _config._get_client()
            trace = self.build_trace()
            # Use background sender for non-blocking send
            _config._sender.submit(client.ingest_trace, trace)
            logger.debug(
                "Queued trace %s (%d spans, %d llm_calls, %d active_skills, manifest=%s)",
                trace.id,
                len(trace.spans),
                len(trace.llm_calls),
                len(trace.active_skills),
                trace.manifest_id or "none",
            )
        except Exception:
            # Elevated so a dropped trace is never silent.
            # logger.exception already emits at ERROR level with traceback,
            # but the message wording is opaque to first-time users. Make
            # it explicit that the trace did NOT arrive at the backend.
            logger.error(
                "decimalai: Failed to queue trace %s for background send. "
                "The trace was NOT ingested. Check that init() was called "
                "with a valid api_key + base_url, or set logging to DEBUG "
                "for the full traceback.",
                self._trace_id,
            )
            logger.debug("Failed to queue trace %s", self._trace_id, exc_info=True)

    def _infer_skill_rungs_from_prompt(self) -> None:
        """Infer OFFERED / DELIVERED for disk skills the SDK did not inject.

        ``_skills_registry`` comes from disk (explicit ``skills=`` else
        ``discover_skills()``), so it describes skills a harness like Claude
        Code may have injected itself, without ever telling the SDK. The
        rendered prompt is then the only observable, and what it can establish
        is that the skill was put in front of the model — the offered rung for
        a bare name, the delivered rung for real body content.

        It deliberately does NOT touch ``_active_skills``. Prompt text can
        never show the model REACHING for a skill: ``_extract_system_text``
        keeps system/developer messages only, so an assistant message or a
        tool result — the only two shapes a model-initiated choice arrives in
        — are discarded before any matching runs. Activation on this path
        comes from ``log_skill_activation`` (a caller's declaration) or
        ``log_skill_loaded`` (the model called ``load_skill``), and when
        neither fired the activated rung stays honestly empty.

        Skills the router already accounted for on THIS trace are excluded —
        see ``infer_prompt_rungs``.
        """
        if not self._skills_registry or not self._llm_calls:
            return

        try:
            from .skills import infer_prompt_rungs

            # Snapshot BEFORE writing: these are the names the router observed
            # directly on this run, and an observation outranks an inference.
            # Passed split for readability; infer_prompt_rungs pools them —
            # see the trade-off note there on why suppression is blanket.
            router_offered = set(self._skills_offered_in_prompt)
            router_delivered = set(self._skills_delivered) | set(self._skills_loaded_by_agent)
            offered, delivered = infer_prompt_rungs(
                (call.rendered_input for call in self._llm_calls),
                self._skills_registry,
                router_offered=router_offered,
                router_delivered=router_delivered,
            )
            if offered:
                self.log_skill_offered(names=offered)
            for name in delivered:
                # Written directly rather than via log_skill_delivered, which
                # folds delivered into offered. That fold is right for the
                # ROUTER — it offered the menu row and then delivered the body,
                # so it observed both — but both rungs here are guesses read off
                # prompt text, and Tier-2 matches a BODY whose name may never
                # appear in the prompt. Folding would assert
                # `skills_offered_in_prompt` ("the menu row was in the prompt
                # the model was shown") for a name that was not, which is the
                # fabrication this rewiring exists to remove, one rung down.
                self._skills_delivered.add(name)
        except Exception:
            logger.debug("Skill prompt-presence inference failed", exc_info=True)

    def _maybe_register_manifest(self) -> None:
        """Extract and register manifest from accumulated trace data.

        Merges auto-detected tools/models with @decimalai.tool registry.
        Thread-safe via _manifest_lock.
        """
        global _manifest_id
        from . import _config
        from .decorators import get_registered_tools

        if not _config._is_enabled():
            return

        # Merge auto-detected tools with @decimalai.tool registry
        tools_dict = dict(self._seen_tools)
        for registered_tool in get_registered_tools():
            name = registered_tool["name"]
            if name not in tools_dict:
                tools_dict[name] = registered_tool
            else:
                # @decimalai.tool has richer schema — prefer it
                tools_dict[name] = registered_tool

        tools = list(tools_dict.values()) if tools_dict else None
        models = self._seen_models if self._seen_models else None
        subagents = self.subagents

        # Always register a manifest, even a minimal one (agent name only).
        # The backend requires a manifest_id on ingest, so a trace with no
        # tools/models/skills/subagents still needs one — do not early-return
        # here when the accumulators are empty.
        agent_name = self.agent_name or "unknown"
        snapshot = extract_from_config(
            agent_name=agent_name,
            tools=tools,
            models=models,
            subagents=subagents,
            skills=self._skills_registry,
        )

        with _manifest_lock:
            if not _manifest_tracker.check_and_update(snapshot):
                return  # Same hash — already registered

            client = _config._get_client()
            # Retry transient failures (network blip, brief backend
            # restart) so a single hiccup doesn't poison every subsequent
            # trace with a "manifest_id required" 400. We DO NOT retry
            # 4xx auth errors since they'll never resolve on their own —
            # falling through to the synthetic-id branch is the right
            # behavior there, and the actual cause (401/403) is captured
            # so export_status() can surface it.
            _backoffs_s = (0.0, 0.1, 0.5)
            last_exc: Optional[BaseException] = None
            registered = False
            for attempt, delay in enumerate(_backoffs_s):
                if delay:
                    import time
                    time.sleep(delay)
                try:
                    result = client.register_manifest(snapshot)
                    mid = result.get("manifest_id", snapshot.id) if isinstance(result, dict) else snapshot.id
                    _manifest_id = str(mid)
                    logger.info(
                        "Registered manifest %s from generic tracer (hash=%s, components=%d, attempt=%d)",
                        _manifest_id,
                        snapshot.manifest_hash[:12],
                        len(snapshot.components),
                        attempt + 1,
                    )
                    registered = True
                    break
                except Exception as exc:
                    last_exc = exc
                    # Don't retry on permanent failures — 401/403 won't
                    # come back to life and burning the retry budget
                    # delays the trace POST that's about to surface the
                    # real error to export_status().
                    msg = str(exc)
                    if "401" in msg or "403" in msg:
                        break
            if not registered and last_exc is not None:
                # Final fallback: synthetic id. The trace POST that
                # follows will likely 400 with "manifest_id required"
                # against a strict backend, but the underlying cause
                # is now captured on the sender so export_status()
                # surfaces "your manifest registration failed" instead
                # of leaving the user with a confusing trace-side error.
                logger.warning(
                    "Failed to register manifest after %d attempt(s); "
                    "trace will use synthetic id. Last error: %s: %s",
                    len(_backoffs_s),
                    type(last_exc).__name__,
                    str(last_exc)[:200],
                )
                try:
                    _config._sender.record_manifest_error(last_exc)
                except Exception:
                    pass
                _manifest_id = snapshot.id
                # Roll back the tracker so a transient first-trace failure
                # doesn't permanently poison ingestion: check_and_update
                # already committed this hash, so without a reset every
                # subsequent trace with the same manifest would early-return
                # and registration would never be re-attempted even after
                # the backend recovers.
                _manifest_tracker.reset()


def _get_current_trace() -> Optional[TraceContext]:
    """Get the current trace context (async-safe via ContextVar)."""
    return _current_trace.get()


def _set_current_trace(ctx: Optional[TraceContext]) -> None:
    """Set the current trace context (async-safe via ContextVar)."""
    _current_trace.set(ctx)


@contextmanager
def start_trace(
    agent_name: Optional[str] = None,
    session_id: Optional[str] = None,
    auto_send: bool = True,
    session_metadata: Optional[Dict[str, Any]] = None,
    turn_index: Optional[int] = None,
    parent_trace_id: Optional[str] = None,
    subagents: Optional[List[Dict[str, Any]]] = None,
    skills: Optional[List[Dict[str, Any]]] = None,
    skill_dirs: Optional[List[str]] = None,
):
    """Context manager for manual trace instrumentation.

    Usage::

        with decimalai.start_trace(agent_name="my-agent") as trace:
            trace.log_llm_call(model="gpt-4o", input=msgs, output=resp)

    Multi-agent linking::

        # Orchestrator trace
        with decimalai.start_trace(agent_name="orchestrator") as parent:
            parent.log_llm_call(model="gpt-4o", input=msgs, output=resp)
            parent_id = str(parent._trace_id)

        # Child trace linked to orchestrator
        with decimalai.start_trace(
            agent_name="billing-agent",
            parent_trace_id=parent_id,
        ) as child:
            child.log_llm_call(model="gpt-4o", input=msgs, output=resp)

    The trace is auto-sent on context exit unless ``auto_send=False``.
    """
    ctx = TraceContext(
        agent_name=agent_name,
        session_id=session_id,
        auto_send=auto_send,
        session_metadata=session_metadata,
        turn_index=turn_index,
        parent_trace_id=parent_trace_id,
        subagents=subagents,
    )

    # Resolve skills registry (auto-discover or explicit)
    resolved_skills = skills
    if not resolved_skills:
        try:
            from .skills import discover_skills
            resolved_skills = discover_skills(skill_dirs) or None
        except Exception:
            logger.debug("Skill auto-discovery failed", exc_info=True)
    ctx._skills_registry = resolved_skills

    # Set as current trace (async-safe via ContextVar)
    prev = _get_current_trace()
    _set_current_trace(ctx)

    try:
        yield ctx
    except Exception as exc:
        ctx._status = Status.ERROR
        ctx._error_message = str(exc)[:500]
        raise
    finally:
        _set_current_trace(prev)
        if ctx.auto_send:
            ctx._send()


def trace(
    agent_name: Optional[str] = None,
    session_id: Optional[str] = None,
    auto_send: bool = True,
    session_metadata: Optional[Dict[str, Any]] = None,
    turn_index: Optional[int] = None,
    parent_trace_id: Optional[str] = None,
    subagents: Optional[List[Dict[str, Any]]] = None,
):
    """Decorator for tracing a function.

    Usage::

        @decimalai.trace(agent_name="my-agent")
        def run_agent(query):
            resp = openai.chat.completions.create(...)
            decimalai.log_llm_call(model="gpt-4o", input=msgs, output=resp)
            return resp.choices[0].message.content

    Multi-agent::

        @decimalai.trace(agent_name="child-agent", parent_trace_id=parent_id)
        def run_child(task):
            ...

    The trace is auto-sent when the function returns.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_trace(
                agent_name=agent_name or fn.__name__,
                session_id=session_id,
                auto_send=auto_send,
                parent_trace_id=parent_trace_id,
                subagents=subagents,
            ) as ctx:
                # Capture the first positional arg as input if it's a string
                if args and isinstance(args[0], str):
                    ctx.set_input(args[0])

                result = fn(*args, **kwargs)

                if isinstance(result, str):
                    ctx.set_output(result)
                return result

        return wrapper

    return decorator


def log_llm_call(**kwargs: Any) -> None:
    """Log an LLM call on the current active trace.

    Convenience function — calls ``ctx.log_llm_call()`` on the active trace.
    Raises ``DecimalConfigError`` if no trace is active.
    """
    ctx = _get_current_trace()
    if ctx is None:
        from ._config import DecimalConfigError

        raise DecimalConfigError(
            "No active trace. Use @decimalai.trace() or decimalai.start_trace() first."
        )
    ctx.log_llm_call(**kwargs)


def log_tool_call(**kwargs: Any) -> None:
    """Log a tool call on the current active trace.

    Convenience function — calls ``ctx.log_tool_call()`` on the active trace.
    """
    ctx = _get_current_trace()
    if ctx is None:
        from ._config import DecimalConfigError

        raise DecimalConfigError(
            "No active trace. Use @decimalai.trace() or decimalai.start_trace() first."
        )
    ctx.log_tool_call(**kwargs)


def log_skill_activation(*, name: str, hash: Optional[str] = None) -> None:
    """Record that a skill was activated on the current active trace.

    Convenience function — calls ``ctx.log_skill_activation()`` on the
    active trace. Raises ``DecimalConfigError`` if no trace is active.
    """
    ctx = _get_current_trace()
    if ctx is None:
        from ._config import DecimalConfigError

        raise DecimalConfigError(
            "No active trace. Use @decimalai.trace() or decimalai.start_trace() first."
        )
    ctx.log_skill_activation(name=name, hash=hash)


def set_routing_id(routing_id: Optional[str]) -> None:
    """Stamp the SkillRouter routing-decision id on the active trace.

    Convenience function — calls ``ctx.set_routing_id()`` on the current
    active trace so the platform's offered-vs-activated join can close on
    the native ``@decimalai.trace`` path. Unlike ``log_skill_activation``,
    this is a safe no-op when there is no active trace: routing is an
    optional enrichment, so a missing trace shouldn't raise. (The sibling
    ``log_skill_*`` helpers DO raise without an active trace — routing is the
    deliberate exception.) The no-op is logged at debug so a silently-dropped
    routing id is still discoverable.
    """
    ctx = _get_current_trace()
    if ctx is None:
        logger.debug(
            "set_routing_id(%r) called with no active trace; routing id dropped. "
            "Call inside @decimalai.trace() / decimalai.start_trace() to record it.",
            routing_id,
        )
        return
    ctx.set_routing_id(routing_id)


def log_skill_offered(*, names: List[str]) -> None:
    """Record skills offered in the active trace's system prompt.

    Convenience function — calls ``ctx.log_skill_offered()`` on the
    active trace. The SDK auto-populates this when the SkillRouter loader
    is enabled; call this manually if you build your own registry menu.
    """
    ctx = _get_current_trace()
    if ctx is None:
        from ._config import DecimalConfigError

        raise DecimalConfigError(
            "No active trace. Use @decimalai.trace() or decimalai.start_trace() first."
        )
    ctx.log_skill_offered(names=names)


def log_skill_delivered(*, names: List[str]) -> None:
    """Record skills whose full body reached the active trace's model.

    Convenience function — calls ``ctx.log_skill_delivered()`` on the
    active trace. The SDK auto-populates this when the SkillRouter's
    ``inject_skill_body`` path injects bodies (they count as 'delivered');
    call this manually if you inject skill bodies yourself.
    """
    ctx = _get_current_trace()
    if ctx is None:
        from ._config import DecimalConfigError

        raise DecimalConfigError(
            "No active trace. Use @decimalai.trace() or decimalai.start_trace() first."
        )
    ctx.log_skill_delivered(names=names)


def log_skill_loaded(*, name: str, hash: Optional[str] = None) -> None:
    """Record that the active trace's agent read a skill's body.

    Convenience function — calls ``ctx.log_skill_loaded()`` on the
    active trace. Use this when your agent fetches the full SKILL.md
    body (e.g., via T2 progressive disclosure) — distinct from
    ``log_skill_offered`` (description-only menu) and
    ``log_skill_activation`` (the skill influenced the output).

    ``hash`` is the body's ``content_hash`` as the platform returned it, and is
    what lets the activation resolve to a skill VERSION rather than just a name.
    Omitting it is the pre-2026-08-29 behaviour and stays fully supported.
    """
    ctx = _get_current_trace()
    if ctx is None:
        from ._config import DecimalConfigError

        raise DecimalConfigError(
            "No active trace. Use @decimalai.trace() or decimalai.start_trace() first."
        )
    ctx.log_skill_loaded(name=name, hash=hash)


def _infer_provider(model: Optional[str]) -> Optional[str]:
    """Infer provider from model name."""
    if not model:
        return None
    m = model.lower()
    if "gpt" in m or "o1" in m or "o3" in m:
        return "openai"
    if "claude" in m:
        return "anthropic"
    if "gemini" in m:
        return "google"
    if "mistral" in m or "mixtral" in m:
        return "mistral"
    if "llama" in m:
        return "meta"
    return None
