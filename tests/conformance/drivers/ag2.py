"""AutoGen / AG2 driver — the classic ``initiate_chat`` lineage.

Runs the snippet documented at ``decimalai-docs/sdk/python/frameworks/autogen.mdx``:
``provider = install(agent_name=...)`` (now spelled ``instrument()``), an
``AssistantAgent`` + ``UserProxyAgent`` pair, ``instrument_agent(agent,
tracer_provider=provider)`` on each — AG2 emits nothing without it — and
``user_proxy.initiate_chat(assistant, message=...)``.

The model is a stub registered through AG2's own custom-model-client protocol
(``model_client_cls`` in the config entry + ``register_model_client``), which is
the supported way to run AG2 with no provider. It matters that the stub sits
THERE and not lower down: ``instrument_llm_wrapper`` patches
``OpenAIWrapper.create``, one layer above the client, so the ``chat`` span with
its model name, token counts and captured messages is emitted exactly as it
would be against a real provider.

Ordering trap worth writing down: ``register_function()`` rebuilds the
assistant's ``OpenAIWrapper``, which drops the registered client. The stub is
therefore registered AFTER the tool, or AG2 raises "Model client(s)
['StubClient'] are not activated" at the first turn.

As with every OTel-backed adapter here, the returned provider is used
explicitly and force-flushed before ``run`` returns; see
``drivers/generic_otel.py`` for why (``set_tracer_provider`` is honoured once
per process, and the ``BatchSpanProcessor`` holds spans for five seconds).

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import json
import threading
from typing import Any, List, Tuple

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

_PROVIDERS: List[Any] = []
_PROVIDERS_LOCK = threading.Lock()


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


# ── the stub model, as an AG2 custom model client ────────────────────────────


def _stub_client_cls(ctx: Ctx, *, fail: bool = False) -> Any:
    """Map the shared stub script onto AG2's custom-model-client protocol."""
    from openai.types.chat import ChatCompletion, ChatCompletionMessage
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageFunctionToolCall,
        Function,
    )
    from openai.types.completion_usage import CompletionUsage

    script = stub_script(ctx)

    class StubClient:
        def __init__(self, config: Any, **kwargs: Any) -> None:
            self._turn = 0

        def create(self, params: Any) -> Any:
            if fail:
                raise DriverError("conformance: the model failed on purpose")
            turn = script[min(self._turn, len(script) - 1)]
            self._turn += 1
            if turn.tool_call:
                name, args = turn.tool_call
                message = ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_conformance_1",
                            type="function",
                            function=Function(name=name, arguments=json.dumps(args)),
                        )
                    ],
                )
                finish_reason = "tool_calls"
            else:
                message = ChatCompletionMessage(role="assistant", content=turn.content)
                finish_reason = "stop"
            response = ChatCompletion(
                id="conformance-stub",
                model=STUB_MODEL_NAME,
                object="chat.completion",
                created=0,
                choices=[Choice(finish_reason=finish_reason, index=0, message=message)],
                usage=CompletionUsage(
                    prompt_tokens=turn.input_tokens,
                    completion_tokens=turn.output_tokens,
                    total_tokens=turn.input_tokens + turn.output_tokens,
                ),
            )
            response.cost = 0.0
            return response

        def message_retrieval(self, response: Any) -> Any:
            return [choice.message for choice in response.choices]

        def cost(self, response: Any) -> float:
            return 0.0

        @staticmethod
        def get_usage(response: Any) -> dict:
            return {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "cost": 0.0,
                "model": response.model,
            }

    return StubClient


# ── the documented snippet ───────────────────────────────────────────────────


def _agents(ctx: Ctx, provider: Any, *, fail: bool = False) -> Tuple[Any, Any]:
    from autogen import AssistantAgent, UserProxyAgent, register_function
    from autogen.opentelemetry import instrument_agent

    stub_cls = _stub_client_cls(ctx, fail=fail)
    llm_config = {
        "config_list": [
            {
                "model": STUB_MODEL_NAME,
                "model_client_cls": stub_cls.__name__,
                "api_type": "openai",
            }
        ],
        "cache_seed": None,
    }
    assistant = AssistantAgent(
        "assistant",
        llm_config=llm_config,
        system_message=SYSTEM_PROMPT,
        silent=True,
    )
    user_proxy = UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        silent=True,
    )

    def lookup(query: str) -> str:
        """Look a value up for the conformance run."""
        return tool_result(ctx, query)

    register_function(
        lookup,
        caller=assistant,
        executor=user_proxy,
        name=ctx.tool_name,
        description="Look a value up for the conformance run.",
    )
    # AFTER register_function — it rebuilds the wrapper and drops the client.
    assistant.register_model_client(model_client_cls=stub_cls)

    # AG2 emits nothing until each agent is instrumented (docs, step 1 of 2).
    instrument_agent(assistant, tracer_provider=provider)
    instrument_agent(user_proxy, tracer_provider=provider)
    return assistant, user_proxy


def _chat(ctx: Ctx, *, fail: bool = False) -> Any:
    provider = _instrument(ctx)
    assistant, user_proxy = _agents(ctx, provider, fail=fail)
    try:
        return user_proxy.initiate_chat(
            assistant, message=user_message(ctx), max_turns=2, silent=True
        )
    finally:
        _flush()


def run(ctx: Ctx) -> Any:
    return _chat(ctx)


def run_error(ctx: Ctx) -> Any:
    """Same conversation, with the model client raising on the first turn."""
    return _chat(ctx, fail=True)


def run_degenerate(ctx: Ctx) -> Any:
    """Two proxies exchanging a fixed reply: no model, no tools, nothing to declare."""
    from autogen import UserProxyAgent
    from autogen.opentelemetry import instrument_agent

    provider = _instrument(ctx)
    sender = UserProxyAgent(
        "degenerate_sender",
        human_input_mode="NEVER",
        code_execution_config=False,
        silent=True,
    )
    receiver = UserProxyAgent(
        "degenerate_receiver",
        human_input_mode="NEVER",
        code_execution_config=False,
        default_auto_reply=ctx.reply_sentinel,
        silent=True,
    )
    instrument_agent(sender, tracer_provider=provider)
    instrument_agent(receiver, tracer_provider=provider)
    try:
        return sender.initiate_chat(
            receiver, message=ctx.prompt_sentinel, max_turns=1, silent=True
        )
    finally:
        _flush()


DRIVER = Driver(
    name="ag2",
    covers=frozenset({"autogen", "ag2"}),
    requires=("autogen", "autogen.opentelemetry", "openai", "opentelemetry.sdk"),
    entrypoint="decimalai.autogen.instrument()",
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_degenerate=run_degenerate,
    capabilities=Capabilities(
        has_skills_rail=False,
        reasons={
            "has_skills_rail": (
                "AG2 has no skills rail on this adapter — the docs capability "
                "table records '—' for AutoGen / AG2, decimalai.autogen.instrument() "
                "takes no skills argument and installs no loader tool, and the OTel "
                "exporter underneath can only MATCH skill text somebody else already "
                "put in the prompt. There is no offered set to record and no "
                "routing_id to carry."
            ),
        },
    ),
)
