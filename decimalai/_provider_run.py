"""What a provider SDK's ``agent_config()`` runs share: the run's span, and its finishing.

``decimalai.anthropic`` and ``decimalai.openai`` give code that calls a provider SDK
itself one run per user message, opened with ``with support.run(question) as run:``;
``decimalai.pydantic_ai`` opens one around each Pydantic AI run.
Opening a run starts a span of its own, the root of the run's trace, and the
provider's calls made inside the block nest under it. Closing it ends the span,
unless the run's routing decision is still in flight: the span then waits for it
here, off the caller's path, so the trace carries the decision.

Nothing here knows a provider. The span is opened on the tracer provider that
provider's calls are traced on (``providers._run_provider``), so the calls and the
run land in one trace.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
import time
from typing import Any, List, Optional, Tuple

logger = logging.getLogger("decimalai")

#: How long a closed run's trace waits for a routing decision still in flight.
#: The wait is off the caller's path: the block exits at once.
_EXIT_WAITS_FOR_ROUTING_S = 3.0


def _enclosing_trace_id() -> Optional[str]:
    """The live ``decimalai.start_trace()`` context's id, or None (never a wrong parent)."""
    try:
        from .generic import _get_current_trace

        ctx = _get_current_trace()
        return ctx.get_trace_id() if ctx is not None else None
    except Exception:
        return None


def _is_error(text: str) -> bool:
    """Whether a load's answer is a refusal rather than a skill (or a repeat notice)."""
    return not (text.startswith("## Skill:") or "already loaded earlier in this run" in text)


def _event_loop_running() -> bool:
    """Whether this thread is running async code: asyncio's loop, or trio's (via sniffio)."""
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        pass
    try:
        import sniffio

        sniffio.current_async_library()
        return True
    except Exception:
        return False


def open_run_span(provider: Optional[str], agent_name: str, session_id: str, *,
                  tracer_provider: Any = None) -> Tuple[Any, Any, List[Tuple[Any, Any]]]:
    """A run's span: a new trace of its own, linked to any span already open.

    Opened on ``tracer_provider`` when given (a framework that emits its own spans,
    Pydantic AI), else on the one ``provider``'s calls are traced on.

    Returns ``(span, context, tokens)``: the span (None without OpenTelemetry or
    on failure), the OpenTelemetry context holding it, and the (holder, token)
    pairs to detach or reset when the run closes, newest last.
    """
    tokens: List[Tuple[Any, Any]] = []
    span: Any = None
    ctx: Any = None
    try:
        from opentelemetry import context as otel_context
        from opentelemetry import trace as trace_api

        from . import otel, providers
    except ImportError:
        return span, ctx, tokens
    try:
        # The provider this process's calls of `provider` are traced on, so the
        # calls nest under the run (not whichever provider was attached last).
        tracer_provider = (tracer_provider or (providers._run_provider(provider) if provider else None)
                           or trace_api.get_tracer_provider())
        current = trace_api.get_current_span().get_span_context()
        links = [trace_api.Link(current)] if current.is_valid else None
        attributes = {
            otel._GENAI_OPERATION: otel._INVOKE_AGENT, otel._GENAI_AGENT_NAME: agent_name,
            otel._DECIMAL_AGENT_NAME: agent_name, otel.SESSION_ID_ATTRIBUTE: session_id,
        }
        # A new root, not a child of whatever is open (a web server's request
        # span, another run): one run, one trace.
        span = tracer_provider.get_tracer(otel.RUN_TRACER).start_span(
            otel.RUN_SPAN_NAME, context=otel_context.Context(), attributes=attributes, links=links)
        ctx = trace_api.set_span_in_context(span, otel_context.get_current())
        tokens.append((otel_context, otel_context.attach(ctx)))
        tokens.append((otel._active_agent_name, otel._active_agent_name.set(agent_name)))
    except Exception:
        # A span already started is still returned, so the run ends it.
        logger.debug("could not start the run's span", exc_info=True)
    return span, ctx, tokens


# ── runs whose trace is still finishing ──────────────────────

_finishing: set = set()
_finishing_lock = threading.Lock()
_finishing_done = threading.Condition(_finishing_lock)
_drain_registered = False


def _finish_later(run: Any, pending: Any, end_ns: int) -> None:
    """End ``run``'s span once its routing decision is in, or after the wait: off the caller's path.

    ``run`` is any provider module's run: it has ``_end(end_ns)`` and ``_exit_ns``.
    """
    global _drain_registered
    with _finishing_lock:
        _finishing.add(run)
        if not _drain_registered:
            _drain_registered = True
            _register_drain_at_exit()

    def finish(*_args: Any) -> None:
        run._end(end_ns)
        with _finishing_lock:
            _finishing.discard(run)
            _finishing_done.notify_all()

    timer = threading.Timer(_EXIT_WAITS_FOR_ROUTING_S, finish)
    timer.daemon = True
    timer.start()
    pending.add_done_callback(finish)


def _drain_finishing(timeout: float = _EXIT_WAITS_FOR_ROUTING_S + 0.5, *, at_exit: bool = False) -> None:
    """Wait for closed runs whose trace is still waiting on a decision; end the rest now.

    ``decimalai.flush()`` calls it, so a short script reports on traces that exist.
    """
    deadline = time.monotonic() + timeout
    with _finishing_lock:
        # A flush waits for the runs closed before it: a run another thread closes
        # meanwhile keeps its own timer, so a flush neither ends it early nor waits
        # on a stream of new ones. At exit no timer will fire, so every run counts.
        waiting = _finishing if at_exit else set(_finishing)
        while waiting & _finishing and time.monotonic() < deadline:
            _finishing_done.wait(timeout=max(0.0, deadline - time.monotonic()))
        left = list(waiting & _finishing)
    for run in left:
        run._end(run._exit_ns or time.time_ns())
        with _finishing_lock:
            _finishing.discard(run)


def _register_drain_at_exit() -> None:
    """Drain during ``threading._shutdown()``, before the OTel pipelines shut down (see otel)."""
    drain = functools.partial(_drain_finishing, at_exit=True)
    try:
        threading._register_atexit(drain)  # type: ignore[attr-defined]
    except Exception:
        import atexit

        atexit.register(drain)
