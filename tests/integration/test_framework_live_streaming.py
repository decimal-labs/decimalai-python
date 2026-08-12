"""Live-LLM Layer 5 — streaming responses.

Same simple shopping-cart agent, but driven through each framework's
streaming entrypoint. Verifies that the DecimalAI adapter merges N stream
events into ONE cohesive trace (not N tiny ones) and that token totals
still aggregate.

Cells:
  * LangChain: `agent.stream(...)` (sync) — produces step-event chunks
  * OpenAI Agents: `Runner.run_streamed(...)` (async) — yields RunItem events

Marker: live_llm + streaming.
"""

from __future__ import annotations

import pytest

from . import _live_helpers as h


# ═══════════════════════════════════════════════════════════════════
# LangChain — sync stream
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.streaming
@pytest.mark.parametrize("provider, model", h.matrix("langchain"))
def test_langchain_streaming(provider, model):
    """Run a langgraph agent via .stream() and verify the adapter still
    produces a single well-formed trace covering every streamed step."""
    h.require_key_for(provider)
    pytest.importorskip("langgraph")
    from langchain_core.tools import tool
    from decimalai.langchain import CallbackHandler

    if provider == "google":
        pytest.importorskip("langchain_google_genai")
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = h.chat_google_genai(model)
    elif provider == "anthropic":
        pytest.importorskip("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(model=model, temperature=0)
    else:
        pytest.importorskip("langchain_openai")
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model)

    @tool
    def get_price(item: str) -> int:
        """Return the unit price in dollars for a given item."""
        return h.lookup_price(item)

    @tool
    def calculate(expression: str) -> float:
        """Evaluate a basic math expression."""
        return h.safe_calculate(expression)

    agent_name = h.unique_agent(f"langchain-{provider}-streaming")
    agent = h.make_react_agent(llm, tools=[get_price, calculate])

    handler = CallbackHandler(agent_name=agent_name)

    events = list(agent.stream(
        {"messages": [{"role": "user", "content": h.SHOPPING_QUERY}]},
        config={"callbacks": [handler], "run_name": "live-streaming-agent"},
    ))
    assert events, "agent.stream returned no events"

    # The final state's last message is the agent's final response.
    final_state = events[-1]
    final_message = None
    # event shape from langgraph: {"<node>": {"messages": [...]}}
    for node, payload in final_state.items():
        if isinstance(payload, dict) and payload.get("messages"):
            final_message = payload["messages"][-1]
    assert final_message is not None, f"No message in final event: {final_state!r}"

    final_text = str(getattr(final_message, "content", final_message))
    assert str(h.SHOPPING_EXPECTED_TOTAL) in final_text, (
        f"Expected '{h.SHOPPING_EXPECTED_TOTAL}' in final answer, got: {final_text!r}"
    )

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    # Critical assertion: streaming must produce ONE trace, not N
    assert len(traces) == 1, (
        f"Streaming produced {len(traces)} traces — expected exactly 1 "
        f"(adapter should merge stream events into a single trace)."
    )
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=2)


# ═══════════════════════════════════════════════════════════════════
# OpenAI Agents — async stream
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.streaming
@pytest.mark.asyncio
@pytest.mark.parametrize("provider, model", h.matrix("openai_agents"))
async def test_openai_agents_streaming(provider, model):
    """Run an OpenAI Agents agent via Runner.run_streamed and consume the
    async event stream. Verify the adapter still produces a single trace."""
    h.require_key_for(provider)
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool
    from agents.tracing import flush_traces, set_trace_processors
    from decimalai.openai_agents import DecimalTracingProcessor

    agent_name = h.unique_agent(f"openai-agents-{provider}-streaming")
    processor = DecimalTracingProcessor(agent_name=agent_name)
    set_trace_processors([processor])

    @function_tool
    def get_price(item: str) -> int:
        """Return the unit price in dollars for a given item."""
        return h.lookup_price(item)

    @function_tool
    def calculate(expression: str) -> float:
        """Evaluate a basic math expression."""
        return h.safe_calculate(expression)

    agent = Agent(
        name=agent_name,
        instructions=(
            "You help with order pricing. Use get_price to look up unit prices "
            "and calculate to compute totals. Reply with the total in dollars."
        ),
        model=h.openai_agents_model(provider, model),
        tools=[get_price, calculate],
    )

    result = Runner.run_streamed(agent, h.SHOPPING_QUERY)
    event_count = 0
    async for _event in result.stream_events():
        event_count += 1
    assert event_count > 0, "stream_events yielded nothing"

    answer = str(result.final_output)
    assert str(h.SHOPPING_EXPECTED_TOTAL) in answer, (
        f"Expected '{h.SHOPPING_EXPECTED_TOTAL}' in answer, got: {answer!r}"
    )

    flush_traces()
    processor.shutdown()
    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)
    assert len(traces) == 1, (
        f"Streaming produced {len(traces)} traces — expected exactly 1."
    )
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=2)
