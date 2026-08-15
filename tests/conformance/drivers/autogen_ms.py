"""Microsoft AutoGen v0.4+ driver (``autogen-agentchat`` / ``autogen-core``).

The second distribution ``decimalai.autogen`` claims. Its module docstring is
the spec being tested here:

    Microsoft AutoGen v0.4+ (``autogen-agentchat``/``autogen-core``, imported as
    ``autogen_core``) routes its runtime tracing through the tracer provider it
    is given, defaulting to the global one — for it the exporter install alone
    is enough, provided the runtime is created after ``init()``.

So the snippet is: ``instrument()`` for the exporter, a
``SingleThreadedAgentRuntime`` handed that provider explicitly (the form the
adapter's own log message recommends — "pass tracer_provider= explicitly if you
construct your own"), an ``AssistantAgent`` with a tool, and a
``RoundRobinGroupChat`` run to completion.

This driver does NOT appear in the docs capability table, so ``covers`` is
empty: the table's "AutoGen / AG2" row is the classic ``initiate_chat`` lineage
covered by ``drivers/ag2.py``. The coverage guard is about frameworks the
product advertises; this one is advertised in the adapter's docstring and its
``autogen_core`` detection branch rather than in the table, and it earns a
driver on the strength of that claim.

The model is a stub ``ChatCompletionClient`` — the protocol AgentChat is built
against — so the whole run is hermetic and the runtime's spans are exactly the
ones a real provider would produce.

As with every OTel-backed adapter here, the provider is used explicitly and
force-flushed before ``run`` returns; see ``drivers/generic_otel.py`` for why.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, List, Sequence

from . import (
    STUB_MODEL_NAME,
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    DriverError,
    stub_script,
    tool_result,
    user_message,
)

_PROVIDERS: List[Any] = []
_PROVIDERS_LOCK = threading.Lock()

#: Long enough that a real hang is a failed phase rather than a hung suite.
_RUN_TIMEOUT_S = 60


def _instrument(ctx: Ctx) -> Any:
    from decimalai.autogen import instrument

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


# ── the stub model ───────────────────────────────────────────────────────────


def _stub_client(ctx: Ctx, *, fail: bool = False) -> Any:
    """Map the shared stub script onto autogen_core's ChatCompletionClient."""
    from autogen_core import FunctionCall
    from autogen_core.models import (
        ChatCompletionClient,
        CreateResult,
        ModelFamily,
        ModelInfo,
        RequestUsage,
    )

    script = stub_script(ctx)

    class StubModelClient(ChatCompletionClient):
        def __init__(self) -> None:
            self._turn = 0
            self._usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

        async def create(
            self,
            messages: Any,
            *,
            tools: Any = (),
            tool_choice: Any = "auto",
            json_output: Any = None,
            extra_create_args: Any = None,
            cancellation_token: Any = None,
        ) -> Any:
            if fail:
                raise DriverError("conformance: the model failed on purpose")
            turn = script[min(self._turn, len(script) - 1)]
            self._turn += 1
            usage = RequestUsage(
                prompt_tokens=turn.input_tokens, completion_tokens=turn.output_tokens
            )
            if turn.tool_call:
                name, args = turn.tool_call
                return CreateResult(
                    finish_reason="function_calls",
                    content=[
                        FunctionCall(
                            id="call_conformance_1",
                            name=name,
                            arguments=json.dumps(args),
                        )
                    ],
                    usage=usage,
                    cached=False,
                )
            return CreateResult(
                finish_reason="stop", content=turn.content, usage=usage, cached=False
            )

        async def create_stream(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError("the conformance stub does not stream")

        async def close(self) -> None:
            return None

        def actual_usage(self) -> Any:
            return self._usage

        def total_usage(self) -> Any:
            return self._usage

        def count_tokens(self, messages: Any, **kwargs: Any) -> int:
            return 0

        def remaining_tokens(self, messages: Any, **kwargs: Any) -> int:
            return 100_000

        @property
        def capabilities(self) -> Any:
            return self.model_info

        @property
        def model_info(self) -> Any:
            return ModelInfo(
                vision=False,
                function_calling=True,
                json_output=False,
                family=ModelFamily.UNKNOWN,
                structured_output=False,
                model_name=STUB_MODEL_NAME,
            )

    return StubModelClient()


# ── the documented snippet ───────────────────────────────────────────────────


def _team(ctx: Ctx, provider: Any, *, fail: bool = False) -> Any:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import SingleThreadedAgentRuntime
    from autogen_core.tools import FunctionTool

    def lookup(query: str) -> str:
        """Look a value up for the conformance run."""
        return tool_result(ctx, query)

    agent = AssistantAgent(
        "assistant",
        model_client=_stub_client(ctx, fail=fail),
        tools=[
            FunctionTool(
                lookup,
                name=ctx.tool_name,
                description="Look a value up for the conformance run.",
            )
        ],
        system_message=SYSTEM_PROMPT,
        reflect_on_tool_use=True,
    )
    runtime = SingleThreadedAgentRuntime(tracer_provider=provider)
    return runtime, RoundRobinGroupChat([agent], runtime=runtime, max_turns=2)


async def _run_team(ctx: Ctx, *, fail: bool = False) -> Any:
    provider = _instrument(ctx)
    runtime, team = _team(ctx, provider, fail=fail)
    runtime.start()
    try:
        return await asyncio.wait_for(
            team.run(task=user_message(ctx)), timeout=_RUN_TIMEOUT_S
        )
    finally:
        try:
            await runtime.stop_when_idle()
        finally:
            _flush()


def run(ctx: Ctx) -> Any:
    return asyncio.run(_run_team(ctx))


def run_error(ctx: Ctx) -> Any:
    """Same team, with the model client raising on its first turn."""
    return asyncio.run(_run_team(ctx, fail=True))


def run_concurrent(ctxs: Sequence[Ctx]) -> Any:
    """N lanes in ONE event loop — the native shape for an asyncio framework.

    Threads would give each lane its own loop and hide exactly the sharing this
    item exists to find; ``gather`` puts every lane on the same loop, which is
    what a real async host does.
    """

    async def _all() -> Any:
        return await asyncio.gather(
            *(_run_team(c) for c in ctxs), return_exceptions=True
        )

    return asyncio.run(_all())


def run_degenerate(ctx: Ctx) -> Any:
    """A team of one model-less agent: no model, no tools, nothing to declare."""
    from autogen_agentchat.agents import BaseChatAgent
    from autogen_agentchat.base import Response
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core import SingleThreadedAgentRuntime

    class EchoAgent(BaseChatAgent):
        @property
        def produced_message_types(self) -> Any:
            return (TextMessage,)

        async def on_messages(self, messages: Any, cancellation_token: Any) -> Any:
            return Response(
                chat_message=TextMessage(content=ctx.reply_sentinel, source=self.name)
            )

        async def on_reset(self, cancellation_token: Any) -> None:
            return None

    async def _go() -> Any:
        provider = _instrument(ctx)
        runtime = SingleThreadedAgentRuntime(tracer_provider=provider)
        team = RoundRobinGroupChat(
            [EchoAgent("echo", "A model-less conformance agent.")],
            runtime=runtime,
            max_turns=1,
        )
        runtime.start()
        try:
            return await asyncio.wait_for(
                team.run(task=ctx.prompt_sentinel), timeout=_RUN_TIMEOUT_S
            )
        finally:
            try:
                await runtime.stop_when_idle()
            finally:
                _flush()

    return asyncio.run(_go())


DRIVER = Driver(
    name="autogen-ms",
    # Deliberately empty — see the module docstring. The docs capability table's
    # "AutoGen / AG2" row is the classic lineage, covered by drivers/ag2.py.
    covers=frozenset(),
    requires=("autogen_core", "autogen_agentchat", "opentelemetry.sdk"),
    entrypoint="decimalai.autogen.instrument() → SingleThreadedAgentRuntime(tracer_provider=…)",
    run=run,
    run_concurrent=run_concurrent,
    run_error=run_error,
    run_degenerate=run_degenerate,
    capabilities=Capabilities(
        has_skills_rail=False,
        reasons={
            "has_skills_rail": (
                "there is no skills rail on this adapter: decimalai.autogen.instrument() "
                "takes no skills argument, installs no loader tool, and the OTel exporter "
                "underneath can only MATCH skill text somebody else already rendered into "
                "the prompt. Nothing offers, so no routing_id can exist."
            ),
        },
    ),
)
