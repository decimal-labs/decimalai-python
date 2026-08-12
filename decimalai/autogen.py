"""AutoGen / AG2 integration — via OpenTelemetry.

AutoGen v0.4+ and the AG2 community distribution both emit standard
OpenTelemetry spans using the GenAI semantic conventions. This means
they work natively with DecimalAI's OTEL exporter — no dedicated
AutoGen adapter needed.

Usage::

    import decimalai

    # Option 1: Use the OTEL exporter directly
    decimalai.init(api_key="...", otel=True)

    # Then run AutoGen as normal — traces are captured automatically
    from autogen import AssistantAgent
    agent = AssistantAgent("assistant", llm_config={...})
    result = await agent.run(task="Analyze this data...")

    # Option 2: Use the instrument() helper (equivalent to Option 1)
    decimalai.init(api_key="...")
    from decimalai.autogen import instrument
    instrument(agent_name="my-autogen-agent")

If you need the ``openinference`` instrumentation for AutoGen, install::

    pip install openinference-instrumentation-autogen
"""

from __future__ import annotations

import warnings

import logging
from typing import Any, Optional

logger = logging.getLogger("decimalai.autogen")


def instrument(agent_name: Optional[str] = None, provider: Optional[Any] = None) -> Any:
    """Install DecimalAI tracing for AutoGen via the OTEL exporter.

    AutoGen emits standard OpenTelemetry spans, so this wires the
    manifest-capable ``decimalai.otel.DecimalSpanExporter`` (which buffers spans
    by root span and registers a manifest from the captured model/tools/prompt).

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

        # AutoGen traces are now captured via OTEL
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
