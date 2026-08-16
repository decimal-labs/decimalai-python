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

One run, one trace
------------------
``instrument()`` is a process-wide switch, and a process-wide switch cannot know
where one *run* of your agent starts and stops. A provider instrumentor emits
one span per SDK call with nothing above it, so each call is an unparented root
in its own OTel trace — and the ordinary tool-use loop (call, run the tool, call
again) lands as N unrelated one-span traces with no key to group them by.
:func:`agent_run` is the other half: a context manager that opens a real parent
span for the duration of a run, so those calls nest under it and arrive as one
trace, filed under the agent that made them::

    import decimalai
    decimalai.init(anthropic=True)

    with decimalai.providers.agent_run("support-bot"):
        client.messages.create(...)   # asks for a tool
        client.messages.create(...)   # answers

Without it, tracing still works — you just get one trace per provider call, and
the agent name is whichever one ``init()`` was given first.

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
from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

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

# The TracerProvider a DecimalAI exporter was attached to MOST RECENTLY —
# including via the explicit ``tracer_provider=`` escape hatch, which
# deliberately leaves ``_pipeline_provider`` alone. ``agent_run()`` defaults to
# it, because a run span opened on the process-global provider when the exporter
# lives on somebody else's would never reach the exporter, and the provider
# spans nested under it would be orphaned instead of merely unparented — worse
# than the defect being fixed. Callers running several providers pass
# ``tracer_provider=`` explicitly.
_last_provider: Any = None


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

    An :class:`~decimalai.otel._AgentNameStamper` goes on alongside the
    exporter. ``agent_name`` here is baked into the exporter when it is BUILT,
    which on this rail is once per process — so before the stamper, a second
    agent in the same process had its traces filed under the first agent's name
    with no way to say otherwise. The stamper reads the run-scoped ContextVar at
    span START and writes the answer onto the span, where the exporter finds it.
    """
    global _pipeline_provider, _last_provider
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from .otel import DecimalSpanExporter, _AgentNameStamper

    if tracer_provider is not None:
        tracer_provider.add_span_processor(_AgentNameStamper())
        tracer_provider.add_span_processor(
            SimpleSpanProcessor(DecimalSpanExporter(agent_name=agent_name))
        )
        _last_provider = tracer_provider
        return tracer_provider

    if _pipeline_provider is not None:
        return _pipeline_provider

    processor = SimpleSpanProcessor(DecimalSpanExporter(agent_name=agent_name))
    current = trace_api.get_tracer_provider()
    if hasattr(current, "add_span_processor"):
        provider = current
    else:
        provider = TracerProvider()
        trace_api.set_tracer_provider(provider)
    provider.add_span_processor(_AgentNameStamper())
    provider.add_span_processor(processor)

    _pipeline_provider = provider
    _last_provider = provider
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

    # Publish the name BEFORE the early return below. The instrumentors are
    # process-wide singletons, so a second call has nothing left to instrument
    # and used to be a total no-op — including the ``agent_name`` it was handed,
    # which is the one thing about a second call that is genuinely new. The
    # ContextVar is where a name can still land after the exporter is built.
    if agent_name:
        from .otel import _active_agent_name

        _active_agent_name.set(agent_name)

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


def agent_run(
    agent_name: Optional[str] = None,
    *,
    span_name: Optional[str] = None,
    tracer_provider: Any = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Any:
    """Group one logical run's provider calls into ONE trace, under one parent.

    ``instrument()`` turns the provider's OpenInference instrumentor on and
    stops there, which is all a *process-wide* switch can do. What it cannot
    know is where one run of your agent begins and ends — so every provider call
    it captures is an unparented root span in its own OTel trace, and a two-step
    tool-use loop lands as two unrelated one-span traces. This is the missing
    half: the entry point that says "this is one run, and it is this agent's"::

        import decimalai
        decimalai.init(anthropic=True)

        def answer(question: str) -> str:
            with decimalai.providers.agent_run("support-bot"):
                first = client.messages.create(...)     # tool call
                second = client.messages.create(...)    # final answer
            return second                               # ONE trace, two calls

    Both calls now nest under a real parent span, so the trace has the shape the
    run actually had. Nothing else is invented: DecimalAI adds the span that
    genuinely wraps the calls and no others — the tool you ran between them
    emitted no span, so the waterfall does not claim one.

    ``agent_name`` is scoped to the ``with`` block rather than fixed at
    ``init()`` time, so a process running two agents (or eight concurrent runs
    of eight agents) files each trace under the right one.

    Args:
        agent_name: Whose run this is. ``None`` keeps the ambient name.
        span_name: Span name; defaults to ``decimalai.otel.RUN_SPAN_NAME``.
        tracer_provider: Which ``TracerProvider`` to open the span on. Defaults
            to the one this module attached the DecimalAI exporter to — which is
            the process-global provider unless ``instrument(tracer_provider=…)``
            was used.
        attributes: Extra span attributes.

    Returns:
        A context manager yielding the OTel span (``None`` without the OTel SDK).
    """
    from .otel import RUN_SPAN_NAME
    from .otel import agent_run as _agent_run

    return _agent_run(
        agent_name,
        span_name=span_name or RUN_SPAN_NAME,
        tracer_provider=tracer_provider or _last_provider,
        attributes=attributes,
    )
