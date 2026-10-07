"""CrewAI driver — the two-halves OpenInference rail.

Runs the snippet documented at ``https://docs.decimal.ai/sdk/python/frameworks/crewai``:
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

The skills rail IS graded here, since 2026-10-08. Until then this driver
declared the rail absent, and that was true: nothing in the SDK could put a
skill into a CrewAI prompt. ``decimalai/crewai.py`` now registers a
``before_llm_call`` hook — CrewAI's public extension point, whose context hands
over the executor's live message list, the same list ``llm.call`` then sends —
and inserts the routed menu and bodies after CrewAI's own system prompt. The
skills phase therefore adds the README's loader call,
``decimalai.crewai.instrument(agent_name=..., enable_skill_loader=True)``, on top
of the tracing pair every other phase runs. What remains absent is the LOADER:
no ``load_skill`` tool is registered, so the model cannot ask for a body and the
strongest rung reachable here is DELIVERED. C13b is N/A for that narrower
reason; C8, C13 and C14 are graded, and so is the injected delivery cell.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import os
import threading
from typing import Any, List, Optional, Sequence

from ..delivery import TOOL_LOADED
from . import (
    STUB_MODEL_NAME,
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    FrameworkLimit,
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


def _kickoff(ctx: Ctx) -> Any:
    return _crew(ctx).kickoff()


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail: instrument once, then N concurrent crews, one thread each.

    Once, on the calling thread, because that is the documented shape — a
    process calls ``instrument(agent_name=..., enable_skill_loader=True)`` at
    startup and then serves its crews — and because the other order is a race in
    the INSTRUMENTORS, not in the adapter: OpenTelemetry's ``BaseInstrumentor``
    is a singleton with an unlocked "already instrumented?" check, so eight
    threads instrumenting LiteLLM at once can each save another's wrapper as the
    "original" and leave ``litellm.completion`` calling itself. In the delivery
    cells the skills phase is the FIRST phase in the process, so nothing has
    instrumented anything yet; observed on crewai 1.15.0, which still routes
    ``openai/…`` models through LiteLLM, as unbounded recursion in
    ``_completion_wrapper``.

    Threads for the lanes, for the same reason ``run_concurrent`` uses them —
    ``Crew.kickoff`` is synchronous, and a server running several crews at once
    runs them on worker threads. Concurrency is the point of the phase: every
    lane shares the router singleton and the hook, so a routing decision or a
    skill block that leaked between runs would show up as one lane's trace
    carrying another lane's routing_id.

    In the ``tool_loaded`` delivery cell the driver ASKS for the tool loop rather
    than quietly not asking, so the adapter has to refuse out loud on this run.
    Asking is not an assertion — ``contract.grade_delivery`` grades what comes
    back.
    """
    from decimalai.crewai import instrument

    for ctx in ctxs:
        _wire().register(ctx)
    # One agent across the lanes (the harness derives them with rename=False).
    _instrument(ctxs[0])
    instrument(
        agent_name=ctxs[0].agent_name,
        enable_skill_loader=True,
        enable_load_skill_tool=ctxs[0].delivery_mode == TOOL_LOADED,
    )
    try:
        return fanout_threads(_kickoff)(ctxs)
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
        "(what init(crewai=True) runs) + LiteLLMInstrumentor; the skills phase adds "
        "decimalai.crewai.instrument(enable_skill_loader=True)"
    ),
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_skills=run_skills,
    capabilities=Capabilities(
        has_skills_rail=True,
        model_can_load_skill_bodies=False,
        supports_degenerate=False,
        reasons={
            "model_can_load_skill_bodies": (
                "this rail is prompt-injection only. decimalai/crewai.py inserts the "
                "routed menu and bodies from a before_llm_call hook and registers no "
                "load_skill tool — it says so when asked ('enable_load_skill_tool is "
                "not supported on the crewai adapter') — so the model has no way to "
                "ASK for a body and the strongest rung observable here is DELIVERED. "
                "Delivery is not activation. C13 still applies and is graded: with no "
                "loader, a delivered body is exactly what is most likely to be "
                "promoted to a fabricated activation."
            ),
            "supports_degenerate": (
                "CrewAI has no model-less run to make. Agent requires an llm, executing a "
                "Task IS a model call, and a Crew with no agents raises before it starts — "
                "so there is no crew shape in which the adapter could observe nothing and "
                "fabricate an 'undeclared' manifest. C7's main+repeat clause still grades "
                "manifest stability here."
            ),
        },
        delivery_limits={
            TOOL_LOADED: FrameworkLimit(
                reason=(
                    "The seam is a before_llm_call hook: it edits the messages of one "
                    "model call and registers no tool, so there is no load_skill tool "
                    "for a body to come back from and no RESULT for the hook to route "
                    "back into the turn. CrewAI itself does run tools in a loop — a "
                    "tool channel would need a load_skill tool attached to the Agent, "
                    "which is a different seam this adapter does not use. Prompt "
                    "injection is the whole rail, which is why the injected cell is "
                    "graded strictly and is not allowed to be N/A."
                ),
                adapter_module="decimalai/crewai.py",
                refusal_marker=(
                    "enable_load_skill_tool is not supported on the crewai adapter"
                ),
            ),
        },
    ),
)
