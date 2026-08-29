"""LangChain / LangGraph driver — the reference implementation.

Runs the snippets documented at ``https://docs.decimal.ai/sdk/python/frameworks/langchain``:
a compiled LangGraph invoked with ``config={"callbacks": [CallbackHandler(...)]}``
for the traced phases, and the process-wide ``instrument(enable_skill_loader=True)``
form for the skills rail (the only form the rail has).

The model is a stub — no key, no network, deterministic tokens — so every phase
runs on every commit. Everything the contract asserts (prompt text, completion
text, tool name, agent name) comes off ``ctx`` and the shared ``stub_script``;
nothing is invented here.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..delivery import TOOL_LOADED
from . import (
    STUB_MODEL_NAME,
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    DriverError,
    FrameworkLimit,
    fanout_threads,
    stub_script,
    tool_result,
    user_message,
)

# ── the stub model ───────────────────────────────────────────────────────────


def _stub_model(ctx: Ctx, *, use_tool: bool = True, fail: bool = False) -> Any:
    """Map the shared stub script onto langchain-core's chat-model interface."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    script = stub_script(ctx, use_tool=use_tool)

    class StubChatModel(BaseChatModel):
        fail: bool = False
        _turn: int = PrivateAttr(default=0)

        @property
        def _llm_type(self) -> str:
            return "conformance-stub"

        @property
        def _identifying_params(self) -> dict:
            return {"model_name": STUB_MODEL_NAME, "temperature": 0.0}

        def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
            return self.bind(tools=[getattr(t, "name", str(t)) for t in tools])

        def _generate(
            self, messages: List[Any], stop: Optional[List[str]] = None,
            run_manager: Any = None, **kwargs: Any,
        ) -> Any:
            if self.fail:
                raise DriverError("conformance: the model failed on purpose")
            turn = script[min(self._turn, len(script) - 1)]
            self._turn += 1
            tool_calls = []
            if turn.tool_call:
                name, args = turn.tool_call
                tool_calls = [{
                    "name": name, "args": args,
                    "id": "call_conformance_1", "type": "tool_call",
                }]
            message = AIMessage(
                content=turn.content,
                tool_calls=tool_calls,
                usage_metadata={
                    "input_tokens": turn.input_tokens,
                    "output_tokens": turn.output_tokens,
                    "total_tokens": turn.input_tokens + turn.output_tokens,
                },
            )
            return ChatResult(generations=[ChatGeneration(message=message)])

    return StubChatModel(fail=fail)


# ── the documented snippet ───────────────────────────────────────────────────


def _graph(ctx: Ctx, model: Any) -> Any:
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    @tool(ctx.tool_name)
    def lookup(query: str) -> str:
        """Look a value up for the conformance run."""
        return tool_result(ctx, query)

    bound = model.bind_tools([lookup])

    def call_model(state):
        return {"messages": [bound.invoke(state["messages"])]}

    def should_continue(state):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    builder = StateGraph(MessagesState)
    builder.add_node("model_turn", call_model)
    builder.add_node("tools", ToolNode([lookup]))
    builder.add_edge(START, "model_turn")
    builder.add_conditional_edges("model_turn", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "model_turn")
    return builder.compile()


def _messages(ctx: Ctx) -> list:
    from langchain_core.messages import HumanMessage, SystemMessage

    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_message(ctx))]


def run(ctx: Ctx) -> Any:
    """The documented per-call-handler snippet, over a compiled LangGraph."""
    from decimalai.langchain import CallbackHandler

    handler = CallbackHandler(agent_name=ctx.agent_name)
    graph = _graph(ctx, _stub_model(ctx))
    return graph.invoke({"messages": _messages(ctx)}, config={"callbacks": [handler]})


def run_error(ctx: Ctx) -> Any:
    """Same snippet, with the model raising partway through."""
    from decimalai.langchain import CallbackHandler

    handler = CallbackHandler(agent_name=ctx.agent_name)
    graph = _graph(ctx, _stub_model(ctx, fail=True))
    return graph.invoke({"messages": _messages(ctx)}, config={"callbacks": [handler]})


def run_degenerate(ctx: Ctx) -> Any:
    """A pure-Python LCEL step: no model, no tools — nothing to declare."""
    from langchain_core.runnables import RunnableLambda

    from decimalai.langchain import CallbackHandler

    handler = CallbackHandler(agent_name=ctx.agent_name)
    chain = RunnableLambda(lambda payload: {"echo": payload["input"]}).with_config(
        run_name="conformance-degenerate-step"
    )
    return chain.invoke({"input": ctx.prompt_sentinel}, config={"callbacks": [handler]})


def run_skills(ctxs: Sequence[Ctx]) -> Any:
    """The skills rail, which exists only in the process-wide form.

    ``instrument(enable_skill_loader=True)`` monkey-patches ``BaseChatModel``
    and publishes a process-global handler, so this phase runs LAST — it cannot
    be undone, and a global handler would double-trace every other phase. The
    lanes then run through that one global handler, which is what a real
    multi-request process does.

    In the ``tool_loaded`` delivery cell the driver ASKS for the tool loop
    (``enable_load_skill_tool=True``) instead of quietly not asking. That is what
    turns this framework's N/A from a sentence into an observation: the adapter
    has to refuse, out loud, on this run, or the cell fails instead of being
    excused. Asking is not an assertion — the grading of what comes back lives
    in ``contract.grade_delivery``.
    """
    from decimalai.langchain import instrument

    instrument(
        agent_name=ctxs[0].agent_name,
        enable_skill_loader=True,
        enable_load_skill_tool=ctxs[0].delivery_mode == TOOL_LOADED,
    )

    def _one(ctx: Ctx) -> Any:
        return _graph(ctx, _stub_model(ctx)).invoke({"messages": _messages(ctx)})

    return fanout_threads(_one)(ctxs)


DRIVER = Driver(
    name="langchain",
    covers=frozenset({"langchain", "langgraph"}),
    requires=("langchain_core", "langgraph"),
    entrypoint="decimalai.langchain.CallbackHandler / instrument()",
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        model_can_load_skill_bodies=False,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
        reasons={
            "model_can_load_skill_bodies": (
                "this rail is prompt-injection only. enable_load_skill_tool is accepted "
                "but DORMANT on this adapter — decimalai/langchain.py logs "
                "'enable_load_skill_tool is not supported on the langchain adapter "
                "(invoke-layer patch, no tool loop); staying on prompt injection. Use "
                "openai_agents or pydantic_ai for the native load_skill tool' — because "
                "an invoke-layer patch cannot route a tool result back mid-turn. The "
                "model therefore has no way to ASK for a body, so the strongest rung "
                "observable here is DELIVERED, and delivery is not activation. "
                "snippets/silent-noops.mdx already says it: 'activation isn't measurable "
                "for bare prompt-injection usage.' C13 still applies and is graded: with "
                "no loader, a delivered body is exactly what is most likely to be "
                "promoted to a fabricated activation."
            ),
        },
        delivery_limits={
            TOOL_LOADED: FrameworkLimit(
                reason=(
                    "LangChain's callback/invoke layer is an OBSERVER of a turn, not "
                    "the owner of a tool loop: this adapter patches "
                    "BaseChatModel.invoke/ainvoke, which sees one model call and "
                    "cannot route a tool RESULT back into the same turn. There is "
                    "nowhere for a load_skill result to go, so the tool-loaded "
                    "channel does not exist here at any setting of any flag. "
                    "Prompt injection is the whole rail — which is why the injected "
                    "cell is graded strictly and is not allowed to be N/A."
                ),
                adapter_module="decimalai/langchain.py",
                refusal_marker=(
                    "enable_load_skill_tool is not supported on the langchain adapter"
                ),
            ),
        },
    ),
)
