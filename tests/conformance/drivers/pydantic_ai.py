"""Pydantic AI driver.

Runs the snippet documented at ``https://docs.decimal.ai/sdk/python/frameworks/pydantic-ai``:
the agent's prompt and skills pulled from DecimalAI as one capability and handed to
the developer's own ``Agent``::

    support = agent_config(name)
    agent = Agent(model, name=support.name, tools=[...], capabilities=[support])

The capability is the whole integration: its instructions (the saved prompt, then
the skill list routed on the run's prompt), its ``load_skill`` toolset, and the
tracing: DecimalAI's run span around each run, the root of the run's trace, with
Pydantic AI's own spans for that run's model and tool calls inside it (its
``Instrumentation``, for the capability's runs alone), filed under the DecimalAI
agent. Nothing is patched, so every phase builds its agent that way
(``Driver.loads_agent_from_platform``: the harness puts each phase's agent in the
workspace first, as the dashboard would). ``init()`` only points the SDK at the
probe: no ``otel=True``, no ``openai=True``, no ``Agent.instrument_all()``.

The model is a stub and the model path is real: Pydantic AI's own
``OpenAIChatModel`` talking to :mod:`._openai_wire` over a socket, so the model-call
spans are the ones Pydantic AI records of real requests and responses.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..delivery import INJECTED
from . import (
    STUB_MODEL_NAME,
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

_WIRE: Optional[OpenAIWire] = None


def _wire() -> OpenAIWire:
    global _WIRE
    if _WIRE is None:
        _WIRE = OpenAIWire().start()
    return _WIRE


# ── the documented snippet's pieces ──────────────────────────────────────────


def _model() -> Any:
    """Pydantic AI's OpenAI chat model, pointed at the stub server."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        STUB_MODEL_NAME, provider=OpenAIProvider(openai_client=_wire().async_client())
    )


def _tool(ctx: Ctx) -> Any:
    """The conformance tool. Pydantic AI names a tool after the function."""

    def lookup(query: str) -> str:
        """Look a value up for the conformance run."""
        return tool_result(ctx, query)

    lookup.__name__ = ctx.tool_name
    return lookup


def _agent(ctx: Ctx, *, configs: Optional[Dict[str, Any]] = None, named: bool = True) -> Any:
    """The documented snippet: this agent's ``agent_config()`` capability on the
    developer's own ``Agent``. ``configs`` shares one config per agent name, as a
    module-level one would be."""
    from pydantic_ai import Agent

    from decimalai.pydantic_ai import agent_config

    configs = {} if configs is None else configs
    if ctx.agent_name not in configs:
        configs[ctx.agent_name] = agent_config(ctx.agent_name)
    support = configs[ctx.agent_name]
    return Agent(
        _model(),
        **({"name": support.name} if named else {}),
        tools=[_tool(ctx)],
        capabilities=[support],
    )


def _skill_script(ctx: Ctx) -> List[StubTurn]:
    """The rail's script: call ``load_skill``, then answer.

    ``load_skill`` is the rail this framework advertises in the capability
    table, and Pydantic AI owns its tool loop, so the body comes back mid-turn.
    Same shape and same reply text as the shared script; only the tool called
    differs, because the tool IS the mechanism under test.

    In the ``injected`` delivery cell the ``load_skill`` tool is switched OFF in
    this process's environment (``DECIMALAI_LOAD_SKILL_TOOL=0``), so the model is
    scripted onto the ordinary conformance tool instead: asking for a tool that
    does not exist would be a driver artifact dressed up as an adapter defect.
    The body has to arrive by injection in that cell, which is the point of it.
    """
    turns = stub_script(ctx)
    if not ctx.skills or ctx.delivery_mode == INJECTED:
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


def _setup(ctx: Ctx, *, fail: bool = False, skills_rail: bool = False) -> None:
    """The documented setup, plus this lane's script.

    Kept out of the threaded body: ``init()`` rebuilds process-global config,
    and eight threads racing to rebuild it would be a driver artifact dressed
    up as an isolation defect.
    """
    import decimalai

    _wire().register(
        ctx, script=_skill_script(ctx) if skills_rail else None, fail=fail
    )
    # Where the probe is, no more: agent_config() turns the tracing on itself.
    decimalai.init(
        api_key=ctx.api_key,
        base_url=ctx.base_url,
        verify=False,
        agent_name=ctx.agent_name,
    )


def _go(pair: Any) -> Any:
    ctx, agent = pair
    return agent.run_sync(user_message(ctx))


# ── phases ───────────────────────────────────────────────────────────────────


def run(ctx: Ctx) -> Any:
    _setup(ctx)
    return _go((ctx, _agent(ctx)))


def run_error(ctx: Ctx) -> Any:
    """The same snippet, with the provider failing the request."""
    _setup(ctx, fail=True)
    return _go((ctx, _agent(ctx)))


def run_concurrent(ctxs: Sequence[Ctx]) -> Any:
    configs: Dict[str, Any] = {}
    pairs = []
    for ctx in ctxs:
        _setup(ctx)
        pairs.append((ctx, _agent(ctx, configs=configs)))
    return fanout_threads(_go)(pairs)


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail, N lanes at once.

    The lanes share one agent name and so one config, the way every run of a
    module-level agent does. Each lane asks a different question, so the router
    mints a routing decision per lane and a leaked one is visible.
    """
    configs: Dict[str, Any] = {}
    pairs = []
    for ctx in ctxs:
        _setup(ctx, skills_rail=True)
        pairs.append((ctx, _agent(ctx, configs=configs)))
    return fanout_threads(_go)(pairs)


def run_unnamed(ctx: Ctx) -> Any:
    """The rail on an UNNAMED Agent.

    ``Agent(model, ...)`` with no ``name=``: Pydantic AI then infers one from the
    variable the Agent is assigned to, which is nothing anybody chose. The run
    is still the DecimalAI agent's, named once in ``agent_config()``: the trace and
    the routing call must both carry that name.
    """
    _setup(ctx, skills_rail=True)
    return _go((ctx, _agent(ctx, named=False)))


def run_conversation(ctxs: Sequence[Ctx]) -> Any:
    """Two turns of one conversation: ``run_sync(..., message_history=...)``.

    The second run is handed the first run's ``all_messages()``, Pydantic AI's
    documented way to continue a conversation. The capability's instructions are
    rebuilt for every request, so the second run is routed on its own prompt.
    """
    first, second = ctxs
    _setup(first, skills_rail=True)
    _wire().register(second, script=_skill_script(second))
    agent = _agent(first)
    turn_one = agent.run_sync(user_message(first))
    return agent.run_sync(user_message(second), message_history=turn_one.all_messages())


DRIVER = Driver(
    name="pydantic-ai",
    covers=frozenset({"pydantic-ai"}),
    requires=("pydantic_ai", "openai"),
    frameworks=("pydantic-ai-slim",),
    entrypoint="decimalai.pydantic_ai.agent_config(name)",
    run=run,
    run_concurrent=run_concurrent,
    run_error=run_error,
    run_degenerate=None,
    run_skills=run_skills,
    run_unnamed=run_unnamed,
    run_conversation=run_conversation,
    loads_agent_from_platform=True,
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=False,
        has_invocation_observer=False,
        reasons={
            "has_invocation_observer": (
                "decimalai/pydantic_ai.py hands its caller no run: agent_config() takes "
                "no on_trace, so C15 has no observer to grade. Its traces are built on "
                "the OTel rail, which links a run to an enclosing decimalai.start_trace() "
                "only where an adapter recorded that link when the run started "
                "(decimalai/crewai.py does; this adapter does not), so C16 is N/A with "
                "C15. A known gap: a run wrapped in start_trace() ships unlinked, and a "
                "routed delivery is credited on both traces."
            ),
            "supports_degenerate": (
                "a Pydantic AI run always makes a model request, so there is no "
                "degenerate run (one with no model call) for a manifest to be "
                "fabricated from. C7b has nothing to grade here; C7 (the same agent "
                "twice) still applies and is graded."
            ),
        },
    ),
)
