"""Direct provider-SDK tracing for DecimalAI — the no-framework one-liner.

The single largest slice of real-world LLM usage is code that calls a provider
SDK directly — ``openai.OpenAI().chat.completions.create(...)``,
``anthropic.Anthropic().messages.create(...)``,
``google.genai.Client().models.generate_content(...)`` — with no agent
framework in between. This module gives those calls one-line auto-tracing::

    import decimalai
    decimalai.init(openai=True)              # trace every OpenAI SDK call
    decimalai.init(anthropic=True)           # ...or Anthropic
    decimalai.init(google=True)              # ...or Google GenAI

    # Framework-agnostic — trace whatever provider SDKs are importable:
    import decimalai.providers
    decimalai.providers.instrument()

It works by enabling the matching **OpenInference** instrumentor for each
provider and routing the OpenTelemetry spans they emit through DecimalAI's OTEL
exporter (:class:`decimalai.otel.DecimalSpanExporter`), which already maps
OpenInference's ``llm.*`` attributes — model, token counts, provider, inline
tool calls — into a :class:`RunTrace` (with manifest auto-detection). So this
module stays thin: install the exporter pipeline once, then call ``.instrument()``
on each requested provider.

Each provider needs its OpenInference instrumentor installed::

    pip install openinference-instrumentation-openai          # openai
    pip install openinference-instrumentation-anthropic       # anthropic
    pip install openinference-instrumentation-google-genai    # google

A missing instrumentor is a *soft* failure — logged with the ``pip install``
hint and skipped, never raised, so one absent package can't break init.

Double-capture caveat: don't enable a provider here *and* a framework that
already traces the same provider (e.g. ``init(langchain=True, openai=True)``) —
the underlying SDK call would be recorded twice. Use the framework adapter for
framework code, and this for raw SDK calls.

Requires the OpenTelemetry SDK (a core dependency of decimalai) plus the
per-provider OpenInference instrumentor(s) above.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
from typing import Any, List, NamedTuple, Optional, Set, Tuple

logger = logging.getLogger("decimalai.providers")


class _ProviderSpec(NamedTuple):
    """How to detect and instrument one provider's SDK."""

    sdk_module: str            # import name used to detect the SDK is present
    instrumentor_module: str   # OpenInference instrumentor module
    instrumentor_class: str    # the Instrumentor class within it
    pip: str                   # pip package that ships the instrumentor


# The three providers DecimalAI's exporter already understands. ``google`` keys
# off the modern ``google.genai`` SDK (the ``google-genai`` package), matching
# the live suite and the ``crewai`` extra's google instrumentor.
_PROVIDERS: dict[str, _ProviderSpec] = {
    "openai": _ProviderSpec(
        "openai",
        "openinference.instrumentation.openai",
        "OpenAIInstrumentor",
        "openinference-instrumentation-openai",
    ),
    "anthropic": _ProviderSpec(
        "anthropic",
        "openinference.instrumentation.anthropic",
        "AnthropicInstrumentor",
        "openinference-instrumentation-anthropic",
    ),
    "google": _ProviderSpec(
        "google.genai",
        "openinference.instrumentation.google_genai",
        "GoogleGenAIInstrumentor",
        "openinference-instrumentation-google-genai",
    ),
}

# Process-global state for the default (global-provider) path: which providers
# we've already instrumented, and the TracerProvider our exporter sits on. Both
# are bypassed when the caller passes an explicit ``tracer_provider``.
_instrumented: Set[str] = set()
_pipeline_provider: Any = None


def _sdk_present(sdk_module: str) -> bool:
    """True if the provider's SDK is importable (without importing it)."""
    try:
        return importlib.util.find_spec(sdk_module) is not None
    except (ImportError, ValueError, ModuleNotFoundError):
        return False


def _load_instrumentor(spec: _ProviderSpec) -> Optional[type]:
    """Import a provider's OpenInference instrumentor class, or None if absent."""
    try:
        module = importlib.import_module(spec.instrumentor_module)
    except ImportError:
        return None
    return getattr(module, spec.instrumentor_class, None)


def _ensure_pipeline(agent_name: Optional[str], tracer_provider: Any = None) -> Any:
    """Attach a DecimalAI OTel exporter to a TracerProvider and return it.

    Reuses the OpenInference-aware exporter from :mod:`decimalai.otel`.

    * ``tracer_provider`` given → attach our exporter to that caller-managed
      provider and return it (no global state touched). This is the escape
      hatch for users who run their own OTel setup, mirroring
      ``install_otel(provider=...)``.
    * ``tracer_provider`` is None → use the process-global provider, creating
      one if none is set yet. If a real provider is already global (the user's
      own OTel), attach to it rather than fighting OTel's once-only
      ``set_tracer_provider`` guard. Cached so the exporter is added at most
      once per process.

    Uses ``SimpleSpanProcessor``: the exporter hands traces to DecimalAI's
    non-blocking background sender, so synchronous span export stays cheap while
    avoiding the batch-flush timing that drops traces in short scripts.
    """
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from .otel import DecimalSpanExporter

    if tracer_provider is not None:
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(DecimalSpanExporter(agent_name=agent_name))
        )
        return tracer_provider

    global _pipeline_provider
    if _pipeline_provider is not None:
        return _pipeline_provider

    processor = SimpleSpanProcessor(DecimalSpanExporter(agent_name=agent_name))
    current = trace_api.get_tracer_provider()
    if hasattr(current, "add_span_processor"):
        current.add_span_processor(processor)
        provider = current
    else:
        provider = TracerProvider()
        provider.add_span_processor(processor)
        trace_api.set_tracer_provider(provider)

    _pipeline_provider = provider
    return provider


def instrument(
    *,
    openai: Optional[bool] = None,
    anthropic: Optional[bool] = None,
    google: Optional[bool] = None,
    agent_name: Optional[str] = None,
    tracer_provider: Any = None,
) -> Any:
    """Auto-trace direct provider-SDK calls via their OpenInference instrumentors.

    Each provider flag is tri-state:

    * ``None`` — *auto*: instrument the provider iff its SDK is importable.
      Auto only applies when **all three** flags are ``None`` (the bare
      ``instrument()`` call), so an explicit ``instrument(openai=True)`` enables
      OpenAI alone, not whatever else happens to be installed.
    * ``True`` — *force on*: instrument it; warn with a ``pip install`` hint if
      the OpenInference instrumentor isn't installed.
    * ``False`` — *skip*.

    Args:
        openai/anthropic/google: Per-provider tri-state flag (see above).
        agent_name: Default agent name stamped on captured traces. A trace can
            still override it via a ``decimal.agent_name`` / ``gen_ai.agent.name``
            span attribute (see :class:`decimalai.otel.DecimalSpanExporter`).
        tracer_provider: Optional OTel ``TracerProvider`` to attach the exporter
            to instead of the process-global one. When given, the global cache
            and the once-per-provider idempotency guard are bypassed (the caller
            owns the provider's lifecycle).

    Returns:
        The ``TracerProvider`` the DecimalAI exporter is attached to, or the
        cached/passed provider if nothing new was instrumented (``None`` if no
        provider was requested and none was set up yet).
    """
    explicit_provider = tracer_provider is not None
    requested = {"openai": openai, "anthropic": anthropic, "google": google}
    auto_all = all(v is None for v in requested.values())

    targets: List[Tuple[str, bool]] = []  # (provider_name, forced)
    for name, flag in requested.items():
        if flag is True:
            targets.append((name, True))
        elif flag is None and auto_all and _sdk_present(_PROVIDERS[name].sdk_module):
            targets.append((name, False))
        # flag is False, or None while a sibling was set explicitly → skip

    if not explicit_provider:
        targets = [(n, f) for (n, f) in targets if n not in _instrumented]

    if not targets:
        return tracer_provider or _pipeline_provider

    try:
        provider = _ensure_pipeline(agent_name, tracer_provider)
    except ImportError:
        logger.warning(
            "decimalai.providers.instrument(): OpenTelemetry SDK not available, "
            "cannot trace provider calls. It ships as a core dependency of "
            "decimalai — reinstall with: pip install decimalai"
        )
        return None

    for name, forced in targets:
        spec = _PROVIDERS[name]
        instrumentor_cls = _load_instrumentor(spec)
        if instrumentor_cls is None:
            msg = (
                "decimalai: %s tracing requested but its instrumentor isn't "
                "installed — skipping. Enable it with: pip install %s"
            )
            logger.warning(msg, name, spec.pip) if forced else logger.info(msg, name, spec.pip)
            continue
        try:
            instrumentor_cls().instrument(tracer_provider=provider)
        except Exception:
            logger.warning(
                "decimalai: failed to instrument the %s SDK (continuing)",
                name, exc_info=True,
            )
            continue
        if not explicit_provider:
            _instrumented.add(name)
        logger.info("DecimalAI tracing enabled for direct %s SDK calls", name)

    return provider
