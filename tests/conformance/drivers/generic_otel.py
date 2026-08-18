"""Generic OpenTelemetry driver — the rail with no framework at all.

The documented snippet at ``https://docs.decimal.ai/sdk/python/frameworks/otel`` is
``decimalai.init(api_key=..., otel=True)`` plus the sentence "Any framework
emitting OTEL spans will be captured", backed by a table of the ``gen_ai.*``
attributes the exporter reads. So for THIS rail the snippet is the attributes
themselves: this driver hand-emits spans carrying exactly the keys that table
promises — ``gen_ai.request.model``, ``gen_ai.prompt.{i}.role|content``,
``gen_ai.completion.0.content``, ``gen_ai.usage.{input,output}_tokens`` — and
nothing else. Anything the contract then finds missing is a gap between the
documented mapping and the exporter, not a gap in some third-party
instrumentor.

Two operational notes, both true of every OTel-backed adapter and neither of
them an assertion:

* **The provider is used explicitly, not through the OTel global.**
  ``decimalai.otel.instrument()`` calls ``trace.set_tracer_provider()``, which
  OpenTelemetry honours ONCE per process — every later call logs "Overriding of
  current TracerProvider is not allowed" and leaves the global pointing at the
  first exporter. In a suite that runs several adapters in one process the
  global would route every framework's spans into whichever driver ran first,
  so the tracer is taken from the provider ``instrument()`` returns. The SDK
  documents this exact escape hatch ("Callers that need to activate an
  instrumentor against this exact provider should pass it explicitly rather
  than rely on the global").

* **Both the driver's providers and the OTel global are force-flushed before
  ``run`` returns.** Spans reach the DecimalAI exporter through a
  ``BatchSpanProcessor`` whose default schedule delay is five seconds; without
  the flush the harness would be measuring that timer instead of the adapter,
  and spans that some layer emitted through the *global* tracer would surface
  minutes later inside a different driver's phase, against a different probe.
  See ``_flush`` below. For a real user the same flush happens at process exit
  via ``_register_flush_atexit``.

``instrument()`` is called once per run, with that run's agent name. On this
rail the exporter's ``agent_name`` is fixed at construction and there is no
per-call override, so a process serving two differently-named agents has to
build two exporters — which is exactly what a second ``instrument()`` call
does.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import json
import threading
from typing import Any, List

from . import (
    STUB_MODEL_NAME,
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    DriverError,
    fanout_threads,
    stub_script,
    tool_result,
    user_message,
)

#: Every TracerProvider this driver has built, so a run can flush the batch
#: processors that hold its spans. Not state the contract reads — bookkeeping
#: that stands in for process exit.
_PROVIDERS: List[Any] = []
_PROVIDERS_LOCK = threading.Lock()


def _instrument(ctx: Ctx) -> Any:
    from decimalai.otel import instrument

    provider = instrument(agent_name=ctx.agent_name)
    with _PROVIDERS_LOCK:
        _PROVIDERS.append(provider)
    return provider


def _flush() -> None:
    """Force every provider that could be holding this run's spans.

    Two of them, and the second is not optional. This driver's own providers
    hold the spans it emitted deliberately; the OTel **global** provider holds
    the ones some layer emitted through ``trace.get_tracer()`` instead of
    through the provider it was handed. The global is a process-wide singleton
    that the FIRST ``set_tracer_provider`` in the process wins, so those spans
    belong to whichever adapter ran first — and if they are left to the
    ``BatchSpanProcessor``'s five-second timer they are exported minutes later,
    into whatever probe is current by then, smearing one driver's traffic across
    another's phases. Flushing both here keeps a phase's traffic inside that
    phase; where the spans went WRONG is then a fact the contract can grade
    rather than a timing artifact.
    """
    with _PROVIDERS_LOCK:
        providers = list(_PROVIDERS)
    try:
        from opentelemetry import trace as _trace_api

        providers.append(_trace_api.get_tracer_provider())
    except Exception:  # pragma: no cover - OTel is a hard dependency here
        pass
    for provider in providers:
        flush = getattr(provider, "force_flush", None)
        if flush is None:  # a no-op ProxyTracerProvider
            continue
        try:
            flush()
        except Exception:  # pragma: no cover - a flush must not mask the run
            pass


# ── the documented attribute mapping, emitted by hand ────────────────────────


def _llm_span(tracer: Any, ctx: Ctx, turn: Any, *, fail: bool = False) -> None:
    """One model turn, carrying the gen_ai.* keys the docs table lists."""
    with tracer.start_as_current_span(f"chat {STUB_MODEL_NAME}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", "conformance-stub")
        span.set_attribute("gen_ai.request.model", STUB_MODEL_NAME)
        span.set_attribute("gen_ai.request.temperature", 0.0)
        span.set_attribute("gen_ai.prompt.0.role", "system")
        span.set_attribute("gen_ai.prompt.0.content", SYSTEM_PROMPT)
        span.set_attribute("gen_ai.prompt.1.role", "user")
        span.set_attribute("gen_ai.prompt.1.content", user_message(ctx))
        if fail:
            raise DriverError("conformance: the model failed on purpose")
        span.set_attribute("gen_ai.completion.0.role", "assistant")
        span.set_attribute("gen_ai.completion.0.content", turn.content)
        span.set_attribute("gen_ai.usage.input_tokens", turn.input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", turn.output_tokens)
        span.set_attribute("gen_ai.response.finish_reasons", ["stop"])


def _tool_span(tracer: Any, ctx: Ctx, name: str, args: dict) -> None:
    with tracer.start_as_current_span(f"execute_tool {name}") as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)
        span.set_attribute("gen_ai.tool.call.arguments", json.dumps(args))
        span.set_attribute(
            "gen_ai.tool.call.result", tool_result(ctx, args.get("query", ""))
        )


def _emit(ctx: Ctx, provider: Any, *, fail: bool = False) -> None:
    """The whole run: an agent span parenting the model turns and the tool."""
    tracer = provider.get_tracer("conformance.generic_otel")
    with tracer.start_as_current_span("invoke_agent conformance") as root:
        root.set_attribute("gen_ai.operation.name", "invoke_agent")
        root.set_attribute("gen_ai.agent.name", ctx.agent_name)
        root.set_attribute("input.value", user_message(ctx))
        for turn in stub_script(ctx):
            _llm_span(tracer, ctx, turn, fail=fail)
            if turn.tool_call:
                _tool_span(tracer, ctx, *turn.tool_call)
        root.set_attribute("output.value", ctx.reply_sentinel)


def run(ctx: Ctx) -> Any:
    provider = _instrument(ctx)
    try:
        _emit(ctx, provider)
    finally:
        _flush()


def run_error(ctx: Ctx) -> Any:
    """The same run, with the first model turn raising."""
    provider = _instrument(ctx)
    try:
        _emit(ctx, provider, fail=True)
    finally:
        _flush()


def run_degenerate(ctx: Ctx) -> Any:
    """A pipeline step that is not a model call: nothing to declare."""
    provider = _instrument(ctx)
    tracer = provider.get_tracer("conformance.generic_otel")
    try:
        with tracer.start_as_current_span("pipeline step") as span:
            span.set_attribute("step.kind", "conformance-degenerate")
    finally:
        _flush()


DRIVER = Driver(
    name="generic-otel",
    covers=frozenset({"generic-otel"}),
    requires=("opentelemetry.sdk",),
    entrypoint="decimalai.otel.instrument()",
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_degenerate=run_degenerate,
    capabilities=Capabilities(
        has_skills_rail=False,
        reasons={
            "has_skills_rail": (
                "the generic OTel rail has no skills surface at all — the docs "
                "capability table records '—' for it, decimalai.otel.instrument() "
                "takes skills only as a list to MATCH against spans somebody else "
                "already rendered, and there is no framework here to inject a "
                "prompt fragment into. Nothing offers, so nothing can be routed. "
                "C13/C13b are silenced for a sharper reason: the only activation "
                "channel here is the decimal.active_skills span attribute the USER "
                "writes, which decimalai.otel reads back — a driver that set it would "
                "be feeding the suite its own answer and grading itself. N/A is the "
                "only honest verdict, and it stays N/A even though making this cell "
                "'green' would be one line."
            ),
        },
    ),
)
