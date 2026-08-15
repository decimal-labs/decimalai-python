"""AutoGen / AG2 integration — via OpenTelemetry.

AG2 (the ``autogen``/``ag2`` distribution) emits NO spans by default: its
tracing is opt-in via ``autogen.opentelemetry`` and is applied *per agent
instance* (``instrument_agent``), so installing an exporter alone captures
nothing. ``instrument()`` therefore does two things: it wires DecimalAI's
OTEL exporter, then hooks agent construction so every ``ConversableAgent``
created afterwards is instrumented automatically (plus AG2's global LLM-call
wrapper, which carries the model/token detail).

Microsoft AutoGen v0.4+ (``autogen-agentchat``/``autogen-core``, imported as
``autogen_core``) routes its runtime tracing through the tracer provider it
is given, defaulting to the global one — for it the exporter install alone
is enough, provided the runtime is created after ``init()``.

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

Agents constructed *before* init()/instrument() are not hooked — instrument
those yourself with AG2's one-liner::

    from autogen.opentelemetry import instrument_agent
    instrument_agent(agent, tracer_provider=provider)  # provider = instrument()'s return
"""

from __future__ import annotations

import functools
import warnings

import logging
from typing import Any, Optional

logger = logging.getLogger("decimalai.autogen")

# Ensures the ConversableAgent.__init__ hook is applied at most once per
# process, even if instrument() / init(autogen=True) runs again.
_ag2_hook_installed = False


def _activate_ag2_instrumentation(tracer_provider: Any) -> None:
    """Wire AG2's native OpenTelemetry instrumentation onto ``tracer_provider``.

    AG2 emits no spans until ``autogen.opentelemetry.instrument_agent`` is
    called on each agent — so an exporter without this activation means
    ``init(autogen=True)`` silently produces zero traces. This activates:

    * ``instrument_llm_wrapper`` — AG2's global LLM-call hook (model name,
      token usage on every LLM span), and
    * a ``ConversableAgent.__init__`` hook that passes every agent
      constructed from now on through ``instrument_agent`` automatically.

    Failure to activate is a loud warning, never a crash: the actionable
    fallback (call ``instrument_agent`` yourself) is spelled out in the
    message, because the alternative is invisible zero-trace behaviour.
    """
    global _ag2_hook_installed
    try:
        from autogen import ConversableAgent
        from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper
    except ImportError:
        _warn_ag2_not_instrumentable()
        return

    if _ag2_hook_installed:
        return

    try:
        instrument_llm_wrapper(tracer_provider=tracer_provider)
    except Exception:
        logger.warning(
            "decimalai: failed to instrument AG2's LLM wrapper — LLM spans "
            "will lack model/token detail (continuing)", exc_info=True,
        )

    original_init = ConversableAgent.__init__

    @functools.wraps(original_init)
    def _traced_init(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        # A subclass chain reaches the base __init__ once, but AG2's own
        # executor instrumentation can hand agents back through here —
        # instrument each instance at most once, and never let tracing
        # setup break agent construction.
        if getattr(self, "_decimalai_ag2_instrumented", False):
            return
        try:
            instrument_agent(self, tracer_provider=tracer_provider)
            self._decimalai_ag2_instrumented = True
        except Exception:
            logger.warning(
                "decimalai: failed to auto-instrument AG2 agent %r — its "
                "spans will not be captured. Instrument it manually with "
                "autogen.opentelemetry.instrument_agent(agent, "
                "tracer_provider=...).",
                getattr(self, "name", "<unnamed>"), exc_info=True,
            )

    ConversableAgent.__init__ = _traced_init
    _ag2_hook_installed = True
    logger.info(
        "DecimalAI tracing enabled for AG2 — agents constructed from now on "
        "are instrumented automatically"
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
            "AG2 emits no spans without it. Upgrade AG2 (pip install -U ag2) "
            "to get automatic per-agent instrumentation."
        )
    elif importlib.util.find_spec("autogen_core") is not None:
        # Microsoft AutoGen v0.4+ — its runtime traces through the tracer
        # provider it's given, defaulting to the global one the exporter
        # install just set.
        logger.info(
            "decimalai.init(autogen=True): Microsoft AutoGen detected "
            "(autogen_core) — runtimes created after init() trace through "
            "the global tracer provider automatically; pass tracer_provider= "
            "explicitly if you construct your own."
        )
    else:
        logger.warning(
            "decimalai.init(autogen=True): no AutoGen distribution is "
            "importable, so NO traces will be captured. Install AG2 "
            "(pip install ag2) or Microsoft AutoGen (pip install "
            "autogen-agentchat) and initialize again."
        )


def instrument(agent_name: Optional[str] = None, provider: Optional[Any] = None) -> Any:
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

    Returns:
        The TracerProvider being used.

    Example::

        import decimalai
        decimalai.init(api_key="...")

        from decimalai.autogen import instrument
        instrument(agent_name="my-autogen-agent")

        # AG2 agents constructed from here on are traced automatically
    """
    from opentelemetry import trace as _trace_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from .otel import DecimalSpanExporter

    logger.info(
        "AutoGen uses OpenTelemetry — wiring the manifest-capable DecimalAI OTEL "
        "exporter (agent_name=%s)",
        agent_name,
    )
    exporter = DecimalSpanExporter(agent_name=agent_name)
    if provider is None:
        provider = TracerProvider()
        _trace_api.set_tracer_provider(provider)
    # BatchSpanProcessor (not SimpleSpanProcessor): AutoGen runs are multi-span,
    # so buffering avoids fragmenting one run into many traces.
    provider.add_span_processor(BatchSpanProcessor(exporter))
    _activate_ag2_instrumentation(provider)
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
