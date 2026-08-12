"""Live-LLM Layer 7 — parallel tool calls.

Modern LLMs frequently emit multiple tool calls in a single response —
e.g., `[get_price("widget"), get_price("gadget"), get_price("gizmo")]`.
The DecimalAI adapter must record all of them as siblings of the same
LLM step, not as a serial chain.

Two cells:
  * LangChain — `bind_tools(parallel_tool_calls=True)` is implicit for
    modern providers; we just need a prompt that demands fan-out.
  * OpenAI Agents — defaults to parallel tool calls.

Marker: live_llm + parallel_tools.
"""

from __future__ import annotations

import pytest

from . import _live_helpers as h


# Three items, deliberately worded so the model must look up all three
# before computing the total — pressure toward a single fan-out step.
PRICES_3 = {"widget": 10, "gadget": 25, "gizmo": 7}
PARALLEL_QUERY = (
    "Look up the unit price of EACH of the following three items: "
    "widget, gadget, gizmo. Issue all three lookups in a SINGLE step "
    "before doing anything else. Then sum them and reply with the total."
)
PARALLEL_EXPECTED_TOTAL = sum(PRICES_3.values())  # 42


def _lookup_price_3(item: str) -> int:
    return PRICES_3.get(item.lower().strip().rstrip("s"), 0)


# ═══════════════════════════════════════════════════════════════════
# LangChain — parallel via langgraph
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.parallel_tools
@pytest.mark.parametrize("provider, model", h.matrix("langchain"))
def test_langchain_parallel_tool_calls(provider, model):
    """LangChain agent fans out 3 tool calls in a single LLM step.

    Native Gemini (ChatGoogleGenerativeAI) supports parallel function calling,
    so the adapter must record the fan-out as siblings here too — not just on
    the OpenAI-compatible path the cell below exercises.
    """
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
        return _lookup_price_3(item)

    agent_name = h.unique_agent(f"langchain-{provider}-parallel")
    agent = h.make_react_agent(llm, tools=[get_price])
    handler = CallbackHandler(agent_name=agent_name)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": PARALLEL_QUERY}]},
        config={"callbacks": [handler], "run_name": "parallel-tools"},
    )

    final_text = str(result["messages"][-1].content)
    assert str(PARALLEL_EXPECTED_TOTAL) in final_text, (
        f"Expected total {PARALLEL_EXPECTED_TOTAL} in answer, got: {final_text!r}"
    )

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_parallel_tool_calls(detail, min_parallel=2)


# ═══════════════════════════════════════════════════════════════════
# OpenAI Agents — parallel via default behavior
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.parallel_tools
@pytest.mark.parametrize("provider, model", h.matrix("openai_agents"))
def test_openai_agents_parallel_tool_calls(provider, model):
    """OpenAI Agents agent fans out tool calls; adapter must record siblings."""
    h.require_key_for(provider)
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool
    from agents.tracing import flush_traces, set_trace_processors
    from decimalai.openai_agents import DecimalTracingProcessor

    agent_name = h.unique_agent(f"openai-agents-{provider}-parallel")
    processor = DecimalTracingProcessor(agent_name=agent_name)
    set_trace_processors([processor])

    @function_tool
    def get_price(item: str) -> int:
        """Return the unit price in dollars for a given item."""
        return _lookup_price_3(item)

    agent = Agent(
        name=agent_name,
        instructions=(
            "You help with order pricing. When the user asks for multiple "
            "items, look up ALL of them in a single step (call get_price "
            "in parallel for each), then sum and reply with the total."
        ),
        model=h.openai_agents_model(provider, model),
        tools=[get_price],
    )

    result = Runner.run_sync(agent, PARALLEL_QUERY)
    answer = str(result.final_output)
    assert str(PARALLEL_EXPECTED_TOTAL) in answer, (
        f"Expected total {PARALLEL_EXPECTED_TOTAL} in answer, got: {answer!r}"
    )

    flush_traces()
    processor.shutdown()
    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_parallel_tool_calls(detail, min_parallel=2)
