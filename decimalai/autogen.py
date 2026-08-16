"""AutoGen / AG2 integration — via OpenTelemetry.

AG2 (the ``autogen``/``ag2`` distribution) emits NO spans by default: its
tracing is opt-in via ``autogen.opentelemetry`` and is applied *per agent
instance* (``instrument_agent``), so installing an exporter alone captures
nothing. ``instrument()`` therefore wires DecimalAI's OTEL exporter, then
instruments the agents: the ones that already exist (a sweep — see
:func:`_activate_ag2_instrumentation`) and the ones constructed later (a
``ConversableAgent.__init__`` hook), plus AG2's global LLM-call wrapper, which
carries the model/token detail and the request/response content.

Microsoft AutoGen v0.4+ (``autogen-agentchat``/``autogen-core``, imported as
``autogen_core``) is a DIFFERENT framework that happens to share the name, and
it is **not a supported integration**. Its runtime spans do reach the exporter
via the global tracer provider, but conformance shows what arrives is not worth
calling an integration: no model, no token counts, and one run split across many
traces named after internal message-bus plumbing. Upstream has also been silent
since 2025-09-30. Users on it get the generic OpenTelemetry rail; say so rather
than implying an adapter exists.

Usage::

    import decimalai

    # Option 1: the init flag
    decimalai.init(api_key="...", autogen=True)

    # Agents created AFTER init are traced automatically
    from autogen import AssistantAgent
    agent = AssistantAgent("assistant", llm_config={...})

    # Option 2: the instrument() helper (equivalent to Option 1)
    decimalai.init(api_key="...")
    from decimalai.autogen import instrument
    instrument(agent_name="my-autogen-agent")

Agents constructed *before* init()/instrument() are picked up too: AG2's
``instrument_agent`` mutates an agent IN PLACE, so activation sweeps the live
ones and instruments them as well. Nothing about construction order is
load-bearing.
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import Any, Optional

logger = logging.getLogger("decimalai.autogen")

# Ensures the ConversableAgent.__init__ hook is applied at most once per
# process, even if instrument() / init(autogen=True) runs again.
_ag2_hook_installed = False


def _activate_ag2_instrumentation(
    tracer_provider: Any, *, capture_messages: bool = True
) -> None:
    """Wire AG2's native OpenTelemetry instrumentation onto ``tracer_provider``.

    AG2 emits no spans until ``autogen.opentelemetry.instrument_agent`` is
    called on each agent — so an exporter without this activation means
    ``init(autogen=True)`` silently produces zero traces. This activates:

    * ``instrument_llm_wrapper`` — AG2's global LLM-call hook (model name,
      token usage, and — see ``capture_messages`` — the request/response
      content on every LLM span),
    * a ``ConversableAgent.__init__`` hook that passes every agent
      constructed from now on through ``instrument_agent`` automatically, and
    * a one-shot sweep of the agents that ALREADY exist.

    The sweep is what makes construction order stop mattering. Patching
    ``__init__`` can only ever reach objects built after the patch, so an agent
    constructed before ``init()`` used to go untraced with no warning at all.
    ``instrument_agent`` returns "the instrumented agent instance (same object,
    modified in place)" and each of AG2's per-method instrumentators is
    idempotent (they check their own ``__otel_wrapped__`` marker), so walking
    the live objects and instrumenting them is safe and repeatable.

    Failure to activate is a loud warning, never a crash: the actionable
    fallback (call ``instrument_agent`` yourself) is spelled out in the
    message, because the alternative is invisible zero-trace behaviour.

    Args:
        tracer_provider: the provider AG2's spans should be emitted on.
        capture_messages: record the LLM request/response content on the
            ``chat`` spans. AG2 defaults this OFF, which left DecimalAI as the
            only rail shipping AG2 traces with no rendered input/output — no
            SFT artifact, no system prompt for the manifest, empty previews —
            while every other adapter captures content. It is not a privacy
            setting in practice either: AG2 already writes tool-call arguments,
            tool results, and the agent-span messages unconditionally. Pass
            ``False`` to keep the model conversation off the spans.
    """
    global _ag2_hook_installed
    try:
        from autogen import ConversableAgent
        from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper
    except ImportError:
        _warn_ag2_not_instrumentable()
        return

    if not _ag2_hook_installed:
        try:
            instrument_llm_wrapper(
                tracer_provider=tracer_provider, capture_messages=capture_messages
            )
        except TypeError:
            # AG2 predating the capture_messages flag — still worth wiring.
            try:
                instrument_llm_wrapper(tracer_provider=tracer_provider)
            except Exception:
                _warn_llm_wrapper_failed()
        except Exception:
            _warn_llm_wrapper_failed()

        original_init = ConversableAgent.__init__

        @functools.wraps(original_init)
        def _traced_init(self, *args: Any, **kwargs: Any) -> None:
            original_init(self, *args, **kwargs)
            # A subclass chain reaches the base __init__ once, but AG2's own
            # executor instrumentation can hand agents back through here —
            # instrument each instance at most once, and never let tracing
            # setup break agent construction.
            _instrument_one_agent(self, instrument_agent, tracer_provider)

        ConversableAgent.__init__ = _traced_init
        _ag2_hook_installed = True

    swept = _sweep_existing_agents(ConversableAgent, instrument_agent, tracer_provider)
    logger.info(
        "DecimalAI tracing enabled for AG2 — %d already-constructed agent(s) "
        "instrumented, and agents constructed from now on are instrumented "
        "automatically",
        swept,
    )


def _instrument_one_agent(
    agent: Any, instrument_agent: Any, tracer_provider: Any
) -> bool:
    """Instrument one AG2 agent in place. Returns True if this call did it."""
    if getattr(agent, "_decimalai_ag2_instrumented", False):
        return False
    try:
        instrument_agent(agent, tracer_provider=tracer_provider)
        agent._decimalai_ag2_instrumented = True
        return True
    except Exception:
        logger.warning(
            "decimalai: failed to auto-instrument AG2 agent %r — its "
            "spans will not be captured. Instrument it manually with "
            "autogen.opentelemetry.instrument_agent(agent, "
            "tracer_provider=...).",
            getattr(agent, "name", "<unnamed>"), exc_info=True,
        )
        return False


def _sweep_existing_agents(
    agent_cls: Any, instrument_agent: Any, tracer_provider: Any
) -> int:
    """Instrument every ``agent_cls`` instance that already exists.

    Walks the live heap once. ``instrument_agent`` mutates the instance, so
    this reaches agents built before ``init()`` — the case the ``__init__``
    hook structurally cannot cover. Idempotent: instances already instrumented
    are skipped by the marker, and AG2's own instrumentators re-check theirs.
    """
    import gc

    swept = 0
    try:
        objects = gc.get_objects()
    except Exception:  # pragma: no cover - defensive; gc is always available
        logger.warning(
            "decimalai: could not enumerate live objects, so AG2 agents "
            "constructed before init() are NOT traced. Instrument them with "
            "autogen.opentelemetry.instrument_agent(agent, tracer_provider=...).",
            exc_info=True,
        )
        return 0

    for obj in objects:
        try:
            is_agent = isinstance(obj, agent_cls)
        except Exception:
            # A proxy/mock whose __class__ lookup raises — not our agent.
            continue
        if is_agent and _instrument_one_agent(obj, instrument_agent, tracer_provider):
            swept += 1
    return swept


def _warn_llm_wrapper_failed() -> None:
    logger.warning(
        "decimalai: failed to instrument AG2's LLM wrapper — LLM spans "
        "will lack model/token detail (continuing)", exc_info=True,
    )


def _warn_ag2_not_instrumentable() -> None:
    """Explain, loudly, why AutoGen tracing could not be auto-wired."""
    import importlib.util

    if importlib.util.find_spec("autogen") is not None:
        # `autogen` importable but without autogen.opentelemetry — an AG2 /
        # pyautogen version predating AG2's native OTel support.
        logger.warning(
            "decimalai.init(autogen=True): this AutoGen/AG2 version has no "
            "autogen.opentelemetry module, so NO traces will be captured — "
            "AG2 emits no spans without it. Upgrade with "
            "`pip install -U 'autogen[openai]'` to get automatic per-agent "
            "instrumentation. Do NOT `pip install ag2`: ag2 1.x is a different "
            "package that no longer provides the classic `autogen` API this "
            "integration traces, so it would leave you with no tracing at all."
        )
    elif importlib.util.find_spec("autogen_core") is not None:
        # Microsoft AutoGen v0.4+ — its runtime traces through the tracer
        # provider it's given, defaulting to the global one the exporter
        # install just set.
        logger.warning(
            "decimalai.init(autogen=True): Microsoft AutoGen detected "
            "(autogen_core). This is a DIFFERENT framework from the AutoGen/AG2 "
            "lineage DecimalAI integrates with, and it is not a supported "
            "integration: its runtime spans do reach the exporter, but they "
            "carry no LLM detail (no model, no tokens) and one run arrives as "
            "many small traces named after internal message-bus plumbing. "
            "Treat it as generic OpenTelemetry, not as an AutoGen integration."
        )
    else:
        logger.warning(
            "decimalai.init(autogen=True): no AutoGen distribution is "
            "importable, so NO traces will be captured. Install the classic "
            "AutoGen/AG2 distribution — `pip install 'autogen[openai]'` — and "
            "initialize again. (`ag2` 1.x and Microsoft's `autogen-agentchat` "
            "are different packages this integration does not trace.)"
        )


def instrument(
    agent_name: Optional[str] = None,
    provider: Optional[Any] = None,
    *,
    capture_messages: bool = True,
) -> Any:
    """Install DecimalAI tracing for AutoGen / AG2.

    Wires the manifest-capable ``decimalai.otel.DecimalSpanExporter`` (which
    buffers spans by root span and registers a manifest from the captured
    model/tools/prompt), then activates AG2's native instrumentation against
    it — AG2 emits no spans on its own, so without the activation step the
    exporter would receive nothing (see :func:`_activate_ag2_instrumentation`).

    Args:
        agent_name: Default agent name for all captured traces.
        provider: Existing OTel ``TracerProvider`` to add the exporter to.
            If None, creates a new one and sets it as global.
        capture_messages: Record LLM request/response content on AG2's ``chat``
            spans. On by default — see :func:`_activate_ag2_instrumentation`
            for why, and pass ``False`` to keep the conversation off the spans.

    Returns:
        The TracerProvider being used.

    Example::

        import decimalai
        decimalai.init(api_key="...")

        from decimalai.autogen import instrument
        instrument(agent_name="my-autogen-agent")

        # AG2 agents are traced — the ones already built, and the ones built next
    """
    from opentelemetry import trace as _trace_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from .otel import DecimalSpanExporter, _register_flush_atexit

    logger.info(
        "AutoGen uses OpenTelemetry — wiring the manifest-capable DecimalAI OTEL "
        "exporter (agent_name=%s)",
        agent_name,
    )
    exporter = DecimalSpanExporter(agent_name=agent_name)
    if provider is None:
        # shutdown_on_exit=False: the SDK's own exit flush runs after CPython
        # has stopped the thread pool the trace is sent on, so a plain script
        # exported nothing. _register_flush_atexit runs early enough.
        provider = TracerProvider(shutdown_on_exit=False)
        _trace_api.set_tracer_provider(provider)
        _register_flush_atexit(provider)
    # BatchSpanProcessor (not SimpleSpanProcessor): AutoGen runs are multi-span,
    # so buffering avoids fragmenting one run into many traces.
    provider.add_span_processor(BatchSpanProcessor(exporter))
    _activate_ag2_instrumentation(provider, capture_messages=capture_messages)
    return provider


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
        "decimalai.autogen.install() is deprecated; use "
        "decimalai.autogen.instrument() instead. It turns on tracing for autogen "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
