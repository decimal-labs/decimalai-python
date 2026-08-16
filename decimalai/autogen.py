"""AutoGen / AG2 — RETIRED as an integration; the generic OpenTelemetry rail.

Neither AutoGen lineage is a supported DecimalAI integration any more, and this
module exists only so that ``init(autogen=True)`` and
``decimalai.autogen.instrument()`` keep working for the people who already wrote
them. What they do now is install DecimalAI's generic OTel exporter — the same
thing ``init(otel=True)`` does — and say so out loud. Nobody's tracing goes
dark; nobody is told an adapter is doing more than that.

Why the AG2 side was retired
----------------------------
AG2 — the community fork of the original AutoGen — reaches us through the
classic ``autogen`` PyPI distribution, and that distribution is frozen at 0.14.1
forever: AG2 moved to the ``ag2`` package for 1.x and never republished the old
name, and ``ag2`` 1.x no longer provides the classic ``autogen`` API this
adapter traced. Usage matches: ``autogen`` is at ~156K downloads/month and
falling ~20%/month (pypistats, mirrors excluded, July 2026), against 15.1M for
pydantic-ai — the SMALLEST other framework on the supported list — and 169M for
langchain-core. Maintaining per-agent auto-instrumentation for a frozen package
with ~100x less usage than anything else we support is not a trade worth making.

Microsoft AutoGen v0.4+ (``autogen-agentchat``/``autogen-core``, imported as
``autogen_core``) is a DIFFERENT framework that happens to share the name, and
it was dropped earlier for its own reasons: its runtime spans reach the exporter
but carry no model and no token counts, and one run arrives as many traces named
after internal message-bus plumbing.

What you get, and what you have to do yourself
----------------------------------------------
The exporter is installed on the global tracer provider, so anything emitting
OpenTelemetry ``gen_ai.*`` spans into that provider is captured. AG2 emits NO
spans on its own, so on AG2 that means calling AG2's own instrumentation
yourself — one line per agent::

    import decimalai
    decimalai.init(api_key="...", autogen=True)   # or init(otel=True)

    from autogen import AssistantAgent
    from autogen.opentelemetry import instrument_agent, instrument_llm_wrapper

    instrument_llm_wrapper(capture_messages=True)   # model + tokens + content
    agent = AssistantAgent("assistant", llm_config={...})
    instrument_agent(agent)

See :mod:`decimalai.otel` and the generic OpenTelemetry docs page for the rail
this lands on.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

logger = logging.getLogger("decimalai.autogen")


def _warn_autogen_not_supported() -> None:
    """Say, loudly and once per call, what this flag does and does not do.

    The failure mode being prevented is the quiet one: a user passes
    ``autogen=True``, sees no error, and assumes agents are being instrumented
    for them the way they were before. They are not — so the message names the
    rail they are actually on and spells out the AG2 one-liners that make it
    produce anything, because AG2 emits zero spans until they are called.
    """
    logger.warning(
        "decimalai.init(autogen=True): AutoGen/AG2 is NO LONGER a supported "
        "DecimalAI integration. The generic OpenTelemetry exporter is installed "
        "(identical to init(otel=True)), so tracing does not go dark — but "
        "nothing instruments your agents for you any more. AG2 emits NO spans "
        "on its own: call autogen.opentelemetry.instrument_llm_wrapper("
        "capture_messages=True) once and autogen.opentelemetry.instrument_agent("
        "agent) per agent, and their spans will reach DecimalAI through this "
        "exporter. (The classic `autogen` distribution is frozen at 0.14.1 — AG2 "
        "moved to `ag2` 1.x, which no longer provides the `autogen` API — and "
        "Microsoft's `autogen-core`/`autogen-agentchat` is a different framework "
        "that is also not integrated. Both are generic OpenTelemetry now.)"
    )


def instrument(
    agent_name: Optional[str] = None,
    provider: Optional[Any] = None,
) -> Any:
    """Install DecimalAI's generic OpenTelemetry exporter, and warn.

    Kept as public API so existing ``decimalai.autogen.instrument()`` calls keep
    running, but there is no AutoGen/AG2-specific behaviour left: this is
    :func:`decimalai.otel.instrument` plus :func:`_warn_autogen_not_supported`.
    Prefer ``decimalai.init(otel=True)`` in new code.

    Args:
        agent_name: Default agent name for all captured traces.
        provider: Existing OTel ``TracerProvider`` to add the exporter to.
            If None, creates a new one and sets it as global.

    Returns:
        The TracerProvider being used.
    """
    from opentelemetry import trace as _trace_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from .otel import DecimalSpanExporter, _register_flush_atexit

    _warn_autogen_not_supported()
    exporter = DecimalSpanExporter(agent_name=agent_name)
    if provider is None:
        # shutdown_on_exit=False: the SDK's own exit flush runs after CPython
        # has stopped the thread pool the trace is sent on, so a plain script
        # exported nothing. _register_flush_atexit runs early enough.
        provider = TracerProvider(shutdown_on_exit=False)
        _trace_api.set_tracer_provider(provider)
        _register_flush_atexit(provider)
    # BatchSpanProcessor (not SimpleSpanProcessor): an agent run is multi-span,
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
