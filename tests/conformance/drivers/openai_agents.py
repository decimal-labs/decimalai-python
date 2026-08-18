"""OpenAI Agents SDK driver.

Runs the snippet documented at ``https://docs.decimal.ai/sdk/python/frameworks/openai-agents``
— build an ``agents.Agent``, hand it to ``instrument(agent=...)`` for full
introspection, then ``Runner.run_sync`` — plus the module docstring's
``enable_skill_loader=True`` form for the skills rail, which is where semantic
routing engages: the loader's dynamic-instructions callable reads the turn's
user message off ``RunContextWrapper.turn_input`` and routes on it.

The model is a stub, but the model *path* is real: ``OpenAIResponsesModel``
against :mod:`._openai_wire`. That is the API a plain ``Agent(model="gpt-…")``
uses, so the run produces the ``response`` spans ``_handle_response`` owns
capture from — the code most users' traces actually go through. A hand-written
``Model`` subclass would emit no span at all and grade nothing.

Two deliberate choices, both documented forms of the same entry point:

* ``exclusive=True`` (the adapter's "Custom path"). It replaces the SDK's
  default processors instead of adding alongside them, which keeps the tier
  hermetic — the stock exporter would otherwise try to ship every phase to
  OpenAI's own trace backend — and keeps re-installing per phase from stacking
  processors, which would double-count every trace and fail C9/C10 on a driver
  artifact rather than an adapter defect.
* ``instrument()`` before ``Agent(...)`` in the skills phase, which is what the
  adapter's own retrofit notice tells users to do.

``disk_sync`` is left at its default on purpose. It is what a user who follows
the docs gets, and whatever it writes into the working directory is exactly
what C11 exists to grade.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from . import (
    STUB_MODEL_NAME,
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    StubTurn,
    fanout_threads,
    stub_script,
    tool_result,
    user_message,
)
from ._openai_wire import OpenAIWire

# One stub-model server for the whole module: it is stateless (the answer is
# derived from the request), so every phase and every lane can share it.
_WIRE: Optional[OpenAIWire] = None


def _wire() -> OpenAIWire:
    global _WIRE
    if _WIRE is None:
        _WIRE = OpenAIWire().start()
    return _WIRE


# ── the documented snippet's pieces ──────────────────────────────────────────


def _model() -> Any:
    """The default model class, pointed at the stub server."""
    from agents import OpenAIResponsesModel

    return OpenAIResponsesModel(
        model=STUB_MODEL_NAME, openai_client=_wire().async_client()
    )


def _tool(ctx: Ctx) -> Any:
    from agents import function_tool

    def lookup(query: str) -> str:
        """Look a value up for the conformance run."""
        return tool_result(ctx, query)

    return function_tool(lookup, name_override=ctx.tool_name)


def _agent(ctx: Ctx, *, tools: bool = True, guardrails: Sequence[Any] = ()) -> Any:
    from agents import Agent

    return Agent(
        name=ctx.agent_name,
        instructions=SYSTEM_PROMPT,
        tools=[_tool(ctx)] if tools else [],
        model=_model(),
        input_guardrails=list(guardrails),
    )


def _skill_script(ctx: Ctx) -> List[StubTurn]:
    """The rail's script: call ``load_skill``, then answer.

    ``load_skill`` is the rail this framework advertises, and it is a real tool
    in a real tool loop here — the body comes back mid-run. Same shape and same
    reply text as the shared script; only the tool called differs, because the
    tool IS the mechanism under test.
    """
    turns = stub_script(ctx)
    if not ctx.skills:
        return turns
    ask, answer = turns[0], turns[-1]
    return [
        StubTurn(
            tool_call=("load_skill", {"name": ctx.skills[0]["name"]}),
            content=ask.content,
            input_tokens=ask.input_tokens,
            output_tokens=ask.output_tokens,
        ),
        answer,
    ]


# ── phases ───────────────────────────────────────────────────────────────────


def run(ctx: Ctx) -> Any:
    """The documented snippet: introspected install, then a run."""
    from agents import Runner

    from decimalai.openai_agents import instrument

    _wire().register(ctx)
    agent = _agent(ctx)
    instrument(agent=agent, agent_name=ctx.agent_name, exclusive=True)
    return Runner.run_sync(agent, user_message(ctx))


def run_error(ctx: Ctx) -> Any:
    """The same snippet, with the model failing the request."""
    from agents import Runner

    from decimalai.openai_agents import instrument

    _wire().register(ctx, fail=True)
    agent = _agent(ctx)
    instrument(agent=agent, agent_name=ctx.agent_name, exclusive=True)
    return Runner.run_sync(agent, user_message(ctx))


def run_degenerate(ctx: Ctx) -> Any:
    """A run that trips an input guardrail before the first model call.

    This is the manifest-gate case for this framework, and the exact one the
    adapter's ``_adopt_existing_manifest`` was written for: no model call, no
    tool call, nothing structural observed. The agent declares no tools either,
    so a run graded on its own slice would look like "all tools removed".
    """
    from agents import GuardrailFunctionOutput, Runner, input_guardrail

    from decimalai.openai_agents import instrument

    @input_guardrail
    async def stop_before_the_model(ctx_wrapper: Any, agent: Any, data: Any) -> Any:
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    _wire().register(ctx)
    agent = _agent(ctx, tools=False, guardrails=[stop_before_the_model])
    instrument(agent=agent, agent_name=ctx.agent_name, exclusive=True)
    return Runner.run_sync(agent, user_message(ctx))


def run_concurrent(ctxs: Sequence[Ctx]) -> Any:
    """N lanes at once, each its own agent.

    Every install happens BEFORE any lane starts. ``exclusive=True`` swaps the
    processor list, and swapping it out from under a trace already in flight
    would lose that trace — a driver artifact that would read as an adapter
    isolation bug.
    """
    from agents import Runner

    from decimalai.openai_agents import instrument

    agents = []
    for ctx in ctxs:
        _wire().register(ctx)
        agent = _agent(ctx)
        instrument(agent=agent, agent_name=ctx.agent_name, exclusive=True)
        agents.append(agent)

    def _one(pair: Any) -> Any:
        ctx, agent = pair
        return Runner.run_sync(agent, user_message(ctx))

    return fanout_threads(_one)(list(zip(ctxs, agents)))


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail: the loader on, agents built after it, N lanes at once.

    ``instrument(enable_skill_loader=True)`` monkey-patches ``Agent.__init__``
    and the ``Agent`` class itself, so it runs LAST and the agents are built
    after it — the order the adapter's own retrofit notice asks for. Each lane
    asks a different question, so the router mints a routing decision per lane
    and a leaked one is visible.
    """
    from agents import Runner

    from decimalai.openai_agents import instrument

    instrument(agent_name=ctxs[0].agent_name, exclusive=True, enable_skill_loader=True)

    pairs = []
    for ctx in ctxs:
        _wire().register(ctx, script=_skill_script(ctx))
        pairs.append((ctx, _agent(ctx)))

    def _one(pair: Any) -> Any:
        ctx, agent = pair
        return Runner.run_sync(agent, user_message(ctx))

    return fanout_threads(_one)(pairs)


DRIVER = Driver(
    name="openai-agents",
    covers=frozenset({"openai-agents"}),
    requires=("agents", "openai"),
    entrypoint="decimalai.openai_agents.instrument(agent=...) / enable_skill_loader",
    run=run,
    run_concurrent=run_concurrent,
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
    ),
)
