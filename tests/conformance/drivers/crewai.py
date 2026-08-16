"""CrewAI driver — the two-halves OpenInference rail.

Runs the snippet documented at ``decimalai-docs/sdk/python/frameworks/crewai.mdx``:
DecimalAI's OTel exporter for the RECEIVING half, and the OpenInference CrewAI
+ LiteLLM instrumentors for the EMITTING half (CrewAI itself emits nothing to
your tracer provider — its own telemetry runs on a private internal one). Then
``crew.kickoff()``.

The install set is the documented one, exactly::

    pip install decimalai openinference-instrumentation-crewai \\
                openinference-instrumentation-litellm \\
                openinference-instrumentation-openai

**The stub model is an HTTP endpoint, not a Python class** — the shared
``_openai_wire.OpenAIWire``. CrewAI's LLM detail (model name, token counts, the
messages themselves) comes from whichever provider instrumentor sits under the
model it was given, so a stub that replaced ``crewai.LLM`` with a ``BaseLLM``
subclass would bypass that layer entirely and the resulting absence of
``llm_calls`` would be an artifact of this driver rather than a fact about the
adapter. Pointing ``crewai.LLM`` at the local wire keeps the whole stack real —
CrewAI → its provider client → socket — and fakes only the inference at the far
end.

Which provider instrumentor that is has MOVED, and the third package above is
why. Up to CrewAI 1.14 an ``openai/…`` model went through LiteLLM, so
``LiteLLMInstrumentor`` carried the LLM detail. From 1.15 ``crewai.LLM.__new__``
routes it to ``crewai.llms.providers.openai.completion.OpenAICompletion``, which
calls the ``openai`` SDK directly and never imports litellm — so
``LiteLLMInstrumentor`` patches a function nothing calls and ``OpenAIInstrumentor``
is the one emitting the ``ChatCompletion`` spans. Both are activated here
because both are in the documented install set and either can be the live one
depending on the model string the user passes.

``decimalai.otel.instrument()`` + ``decimalai._activate_crewai_instrumentation()``
is used rather than ``decimalai.init(crewai=True)``. That pair is literally what
``init`` runs (``init`` calls the same two functions, in that order), but the
returned provider can be handed to the instrumentors explicitly. OpenTelemetry
honours ``set_tracer_provider`` only once per process, so the global form would
route CrewAI's spans into whichever adapter happened to run first in a
multi-driver suite; the SDK documents this escape hatch itself ("Callers that
need to activate an instrumentor against this exact provider should pass it
explicitly rather than rely on the global"). Calling ``init``'s own activation
helper rather than re-listing instrumentors by hand is what keeps this driver
from grading a rail the documented path does not give a user.

The provider is force-flushed before ``run`` returns: spans reach the exporter
through a ``BatchSpanProcessor`` whose default schedule delay is five seconds,
and without the flush the harness would be timing that instead of the adapter.
For a real user the same flush happens at process exit.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import os
import threading
from typing import Any, List, Optional

from . import (
    STUB_MODEL_NAME,
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    fanout_threads,
    tool_result,
    user_message,
)
from ._openai_wire import STUB_API_KEY, OpenAIWire

# CrewAI phones home by default and, on a first run, prints an interactive
# tracing-preference panel. Both are turned off here — at import, before any
# crewai module loads — so the hermetic tier stays hermetic and nothing blocks
# waiting on a prompt.
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


# ── the hermetic model ───────────────────────────────────────────────────────

_WIRE: Optional[OpenAIWire] = None
_WIRE_LOCK = threading.Lock()


def _wire() -> OpenAIWire:
    global _WIRE
    with _WIRE_LOCK:
        if _WIRE is None:
            _WIRE = OpenAIWire().start()
        return _WIRE


# ── the two halves ───────────────────────────────────────────────────────────

_PROVIDERS: List[Any] = []
_PROVIDERS_LOCK = threading.Lock()


def _instrument(ctx: Ctx) -> Any:
    """DecimalAI's exporter, then the OpenInference emitters onto it."""
    from openinference.instrumentation.litellm import LiteLLMInstrumentor

    from decimalai import _activate_crewai_instrumentation
    from decimalai.otel import instrument

    provider = instrument(agent_name=ctx.agent_name)
    with _PROVIDERS_LOCK:
        _PROVIDERS.append(provider)
    # What `init(crewai=True)` does, called the way `init` calls it. NOT a
    # hand-written list of instrumentors: `init` activates the CrewAI
    # instrumentor AND every importable provider SDK's instrumentor, and
    # re-listing them here would let the driver and the documented path drift
    # apart in either direction — a driver that activates less grades a rail
    # thinner than the user's, one that activates more grades a rail no user
    # gets. Calling the same function is the only version that cannot drift.
    _activate_crewai_instrumentation(provider)
    # The one emitter `init(crewai=True)` does NOT activate, and the docs'
    # snippet does.
    LiteLLMInstrumentor().instrument(tracer_provider=provider)
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


# ── the documented snippet ───────────────────────────────────────────────────


def _crew(ctx: Ctx) -> Any:
    from crewai import LLM, Agent, Crew, Task
    from crewai.tools import tool

    @tool(ctx.tool_name)
    def lookup(query: str) -> str:
        """Look a value up for the conformance run."""
        return tool_result(ctx, query)

    llm = LLM(
        model=f"openai/{STUB_MODEL_NAME}",
        base_url=_wire().base_url,
        api_key=STUB_API_KEY,
        temperature=0.0,
    )
    agent = Agent(
        role="Conformance Fixture",
        goal=SYSTEM_PROMPT,
        backstory=SYSTEM_PROMPT,
        llm=llm,
        tools=[lookup],
        verbose=False,
    )
    task = Task(
        description=user_message(ctx),
        expected_output="The looked-up value, reported back verbatim.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task], verbose=False, memory=False)


def run(ctx: Ctx) -> Any:
    _wire().register(ctx)
    _instrument(ctx)
    try:
        return _crew(ctx).kickoff()
    finally:
        _flush()


def run_error(ctx: Ctx) -> Any:
    """The same crew, with the model endpoint refusing the request."""
    _wire().register(ctx, fail=True)
    _instrument(ctx)
    try:
        return _crew(ctx).kickoff()
    finally:
        _flush()


DRIVER = Driver(
    name="crewai",
    covers=frozenset({"crewai"}),
    requires=(
        "crewai",
        "litellm",
        "openinference.instrumentation.crewai",
        "openinference.instrumentation.litellm",
        "openinference.instrumentation.openai",
        "opentelemetry.sdk",
    ),
    entrypoint=(
        "decimalai.otel.instrument() + _activate_crewai_instrumentation() "
        "(what init(crewai=True) runs) + LiteLLMInstrumentor"
    ),
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    capabilities=Capabilities(
        has_skills_rail=False,
        supports_degenerate=False,
        reasons={
            "has_skills_rail": (
                "CrewAI has no skills rail on this adapter — the docs capability table "
                "records '—' for it, there is no loader tool and no prompt-fragment "
                "injection point, and the OTel exporter underneath can only MATCH skill "
                "text somebody else already rendered into the prompt. Nothing offers, so "
                "no routing_id can exist."
            ),
            "supports_degenerate": (
                "CrewAI has no model-less run to make. Agent requires an llm, executing a "
                "Task IS a model call, and a Crew with no agents raises before it starts — "
                "so there is no crew shape in which the adapter could observe nothing and "
                "fabricate an 'undeclared' manifest. C7's main+repeat clause still grades "
                "manifest stability here."
            ),
        },
    ),
)
