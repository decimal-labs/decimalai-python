"""LangChain / LangGraph driver — the reference implementation.

Runs the snippets documented at ``https://docs.decimal.ai/sdk/python/frameworks/langchain``:
a compiled LangGraph invoked with ``config={"callbacks": [CallbackHandler(...)]}`` for the
traced phases, and for the skills rail the ``agent_config(name)`` form: the agent's prompt
and skills pulled from DecimalAI as one ``create_agent`` middleware
(``Driver.loads_agent_from_platform``: the harness puts each phase's agent in the workspace
first, as the dashboard would). The middleware gives every model call the system message (the
saved prompt, then the skill list) and a real ``load_skill`` tool in create_agent's tool loop,
so the body comes back mid-run as the tool's result. Until the loader was removed this rail
was ``instrument(enable_skill_loader=True)``, prompt injection only.

``agent_config()`` turns LangChain tracing on for the process, as the docs say, so the skills
phases pass no handler of their own; the traced phases still pass a per-call one, which wins
over the process's (``handle_class`` dedupes by type), so nothing is traced twice.

The model is a stub — no key, no network, deterministic tokens — so every phase
runs on every commit. Everything the contract asserts (prompt text, completion
text, tool name, agent name) comes off ``ctx`` and the shared ``stub_script``;
nothing is invented here.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..delivery import INJECTED
from . import (
    STUB_MODEL_NAME,
    SYSTEM_PROMPT,
    Capabilities,
    Ctx,
    Driver,
    DriverError,
    StubTurn,
    fanout_threads,
    stub_script,
    tool_result,
    trace_observer,
    user_message,
)

# ── the stub model ───────────────────────────────────────────────────────────


def _stub_model(ctx: Ctx, *, use_tool: bool = True, fail: bool = False,
                script: Optional[List[StubTurn]] = None) -> Any:
    """Map the shared stub script onto langchain-core's chat-model interface."""
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    script = script if script is not None else stub_script(ctx, use_tool=use_tool)

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
            # A BaseTool by its name; the middleware's load_skill is an OpenAI-format dict.
            return self.bind(tools=[
                getattr(t, "name", None) or (t.get("function") or {}).get("name")
                if isinstance(t, dict) else getattr(t, "name", str(t))
                for t in tools
            ])

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
                    "id": f"call_conformance_{self._turn}", "type": "tool_call",
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


def _skill_script(ctx: Ctx) -> List[StubTurn]:
    """The rail's script: call ``load_skill``, then answer.

    ``load_skill`` is the rail the middleware gives the model, and it is a real tool in
    create_agent's real tool loop — the body comes back mid-run. Same shape and same reply
    text as the shared script; only the tool called differs, because the tool IS the
    mechanism under test.

    In the ``injected`` delivery cell the ``load_skill`` tool is switched OFF in this
    process's environment (``DECIMALAI_LOAD_SKILL_TOOL=0``), so the model is scripted onto
    the ordinary conformance tool instead: the body has to arrive pasted into the system
    message in that cell, which is the point of it.
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


# ── the documented snippets ──────────────────────────────────────────────────


def _tool(ctx: Ctx) -> Any:
    from langchain_core.tools import tool

    @tool(ctx.tool_name)
    def lookup(query: str) -> str:
        """Look a value up for the conformance run."""
        return tool_result(ctx, query)

    return lookup


def _graph(ctx: Ctx, model: Any) -> Any:
    """A hand-built LangGraph: model node, tool node, loop until no tool call."""
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    lookup = _tool(ctx)
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


def _config(ctx: Ctx, configs: Dict[str, Any]) -> Any:
    """The documented module-level ``agent_config()``, one per agent name."""
    from decimalai.langchain import agent_config

    if ctx.agent_name not in configs:
        configs[ctx.agent_name] = agent_config(ctx.agent_name)
    return configs[ctx.agent_name]


def _agent(ctx: Ctx, support: Any, *, checkpointer: Any = None) -> Any:
    """The documented ``agent_config()`` snippet: ``create_agent`` with the middleware.

    The graph is left unnamed (LangGraph calls it ``LangGraph``): the run is filed under the
    DecimalAI agent the config was built for, whatever the graph is called.
    """
    from langchain.agents import create_agent

    return create_agent(_stub_model(ctx, script=_skill_script(ctx)), tools=[_tool(ctx)],
                        middleware=[support.middleware], checkpointer=checkpointer)


def _ask(ctx: Ctx) -> dict:
    return {"messages": [{"role": "user", "content": user_message(ctx)}]}


# ── phases ───────────────────────────────────────────────────────────────────


def run(ctx: Ctx) -> Any:
    """The documented per-call-handler snippet, over a compiled LangGraph."""
    from decimalai.langchain import CallbackHandler

    handler = CallbackHandler(agent_name=ctx.agent_name)
    graph = _graph(ctx, _stub_model(ctx))
    return graph.invoke({"messages": _messages(ctx)}, config={"callbacks": [handler]})


def run_nested(ctx: Ctx) -> Any:
    """The per-call-handler snippet, with the suite's observer on the handler.

    ``handler.on_trace`` is the per-call form of the observer; the per-call handler adds no
    process-wide state. The harness runs this inside a ``decimalai.start_trace()`` of its own.
    """
    from decimalai.langchain import CallbackHandler

    handler = CallbackHandler(agent_name=ctx.agent_name)
    handler.on_trace = trace_observer(ctx)
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
    """The skills rail: ``agent_config()`` middleware agents, N lanes at once.

    The lanes share one agent name and so one config, the way every run of a module-level
    agent does; the config is built before any lane starts. Each lane asks a different
    question, so the router mints a routing decision per lane and a leaked one is visible.
    """
    configs: Dict[str, Any] = {}
    agents = [(ctx, _agent(ctx, _config(ctx, configs))) for ctx in ctxs]

    def _one(pair: Any) -> Any:
        ctx, agent = pair
        return agent.invoke(_ask(ctx))

    return fanout_threads(_one)(agents)


def run_unnamed(ctx: Ctx) -> Any:
    """The rail on a graph LangChain names one way and DecimalAI another.

    ``create_agent`` names its graph ``LangGraph`` unless told otherwise, while the
    DecimalAI agent, named once in ``agent_config()``, is this run's. The trace and the
    routing call must both name the DecimalAI agent.
    """
    return _agent(ctx, _config(ctx, {})).invoke(_ask(ctx))


def run_conversation(ctxs: Sequence[Ctx]) -> Any:
    """Two turns of one thread: a checkpointer and a ``thread_id``, the documented
    LangGraph conversation. The second invoke carries only the new message; the thread
    keeps the first's messages, system message and load_skill definition."""
    from langgraph.checkpoint.memory import InMemorySaver

    first, second = ctxs
    support = _config(first, {})
    agent = _agent(first, support, checkpointer=InMemorySaver())
    thread = {"configurable": {"thread_id": f"conformance-{first.agent_name}"}}
    agent.invoke(_ask(first), thread)
    return agent.invoke(_ask(second), thread)


DRIVER = Driver(
    name="langchain",
    covers=frozenset({"langchain", "langgraph"}),
    requires=("langchain_core", "langgraph", "langchain.agents.middleware"),
    frameworks=("langchain-core", "langgraph"),
    entrypoint="decimalai.langchain.CallbackHandler / agent_config(name).middleware",
    run=run,
    run_concurrent=fanout_threads(run),
    run_error=run_error,
    run_degenerate=run_degenerate,
    run_skills=run_skills,
    run_nested=run_nested,
    run_unnamed=run_unnamed,
    run_conversation=run_conversation,
    loads_agent_from_platform=True,
    capabilities=Capabilities(
        has_tools=True,
        has_skills_rail=True,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
    ),
)
