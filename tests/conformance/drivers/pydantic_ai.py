"""Pydantic AI driver.

Runs the snippet documented at ``decimalai-docs/sdk/python/frameworks/pydantic-ai.mdx``.
That page is explicit about the shape of this integration, and the shape decides
what this driver has to do:

    "It does not have its own tracing — Pydantic AI calls the provider SDK
    underneath, so tracing flows through a provider integration you install
    alongside."

So the documented setup is **two calls**, and the driver makes both:
``decimalai.init(openai=True)`` for the traces (OpenInference's instrumentor for
the ``openai`` SDK, routed through DecimalAI's OTel exporter) and
``decimalai.pydantic_ai.instrument(enable_skill_loader=True)`` for the skills
rail. Grading only the second half would grade a framework that, by design,
puts nothing on the wire — and the contract is a statement about the wire.

The model is a stub and the model path is real: Pydantic AI's own
``OpenAIChatModel`` talking to :mod:`._openai_wire` over a socket. That is
load-bearing here beyond the usual reason — the *only* spans this integration
produces are the ones OpenInference derives from real ``openai`` client calls,
so a monkeypatched client would leave nothing to grade at all.

Two consequences worth stating up front, because they are properties of the
integration rather than of this driver:

* The agent name is fixed process-wide, at the moment the exporter is built.
  This driver passes ``agent_name=ctx.agent_name`` on every run anyway — asking
  correctly and recording what comes back is the point.
* One trace per *provider call*, not per agent run, because each instrumented
  ``openai`` call is its own OTel root span.

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

_WIRE: Optional[OpenAIWire] = None


def _wire() -> OpenAIWire:
    global _WIRE
    if _WIRE is None:
        _WIRE = OpenAIWire().start()
    return _WIRE


# ── the documented snippet's pieces ──────────────────────────────────────────


def _model() -> Any:
    """Pydantic AI's OpenAI chat model, pointed at the stub server.

    ``OpenAIChatModel`` is the current name; older releases called the same
    class ``OpenAIModel``. Both are accepted so the floating lane (which
    resolves newest) and the pinned lane can run the same driver.
    """
    from pydantic_ai.providers.openai import OpenAIProvider

    try:
        from pydantic_ai.models.openai import OpenAIChatModel as _Model
    except ImportError:  # pragma: no cover - older pydantic-ai
        from pydantic_ai.models.openai import OpenAIModel as _Model

    return _Model(
        STUB_MODEL_NAME, provider=OpenAIProvider(openai_client=_wire().async_client())
    )


def _tool(ctx: Ctx) -> Any:
    """The conformance tool. Pydantic AI names a tool after the function."""

    def lookup(query: str) -> str:
        """Look a value up for the conformance run."""
        return tool_result(ctx, query)

    lookup.__name__ = ctx.tool_name
    return lookup


def _agent(ctx: Ctx) -> Any:
    from pydantic_ai import Agent

    return Agent(
        _model(),
        name=ctx.agent_name,
        system_prompt=SYSTEM_PROMPT,
        tools=[_tool(ctx)],
    )


def _skill_script(ctx: Ctx) -> List[StubTurn]:
    """The rail's script: call ``load_skill``, then answer.

    ``load_skill`` is the rail this framework advertises in the capability
    table, and Pydantic AI owns its tool loop, so the body comes back mid-turn.
    Same shape and same reply text as the shared script; only the tool called
    differs, because the tool IS the mechanism under test.
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


def _setup(ctx: Ctx, *, fail: bool = False, skills_rail: bool = False) -> None:
    """The documented two-call setup, plus this lane's script.

    Kept out of the threaded body: ``init()`` rebuilds process-global config,
    and eight threads racing to rebuild it would be a driver artifact dressed
    up as an isolation defect.
    """
    import decimalai
    from decimalai.pydantic_ai import instrument

    _wire().register(
        ctx, script=_skill_script(ctx) if skills_rail else None, fail=fail
    )
    decimalai.init(
        api_key=ctx.api_key,
        base_url=ctx.base_url,
        verify=False,
        openai=True,               # the provider pairing that carries the traces
        agent_name=ctx.agent_name,
    )
    instrument(enable_skill_loader=skills_rail)


def _go(ctx: Ctx) -> Any:
    """One run. The Agent is built here so the skills phase's patched
    ``Agent.__init__`` is the one that constructs it."""
    return _agent(ctx).run_sync(user_message(ctx))


# ── phases ───────────────────────────────────────────────────────────────────


def run(ctx: Ctx) -> Any:
    _setup(ctx)
    return _go(ctx)


def run_error(ctx: Ctx) -> Any:
    """The same snippet, with the provider failing the request."""
    _setup(ctx, fail=True)
    return _go(ctx)


def run_concurrent(ctxs: Sequence[Ctx]) -> Any:
    for ctx in ctxs:
        _setup(ctx)
    return fanout_threads(_go)(list(ctxs))


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail, N lanes at once.

    ``instrument(enable_skill_loader=True)`` monkey-patches
    ``pydantic_ai.Agent.__init__``, so it runs LAST and the agents are built
    after it — the adapter only ever reaches Agents constructed afterwards,
    which its own caveats say out loud.
    """
    for ctx in ctxs:
        _setup(ctx, skills_rail=True)
    return fanout_threads(_go)(list(ctxs))


DRIVER = Driver(
    name="pydantic-ai",
    covers=frozenset({"pydantic-ai"}),
    # `openinference` (not the dotted instrumentor path) because that is the
    # honest requirement: without the openai instrumentor installed,
    # `init(openai=True)` warns and skips, and the driver would be graded on a
    # tracing path the environment never turned on.
    requires=("pydantic_ai", "openai", "openinference"),
    entrypoint="decimalai.pydantic_ai.instrument() + decimalai.init(openai=True)",
    run=run,
    run_concurrent=run_concurrent,
    run_error=run_error,
    run_degenerate=None,
    run_skills=run_skills,
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=False,
        reasons={
            "supports_degenerate": (
                "this adapter does no tracing of its own — every trace it "
                "produces is derived from a real provider-SDK call underneath. "
                "A Pydantic AI run with no model call therefore emits no span "
                "and no trace at all, so there is no degenerate run for a "
                "manifest to be fabricated from. C7b has nothing to grade here; "
                "C7 (the same agent twice) still applies and is graded."
            ),
        },
    ),
)
