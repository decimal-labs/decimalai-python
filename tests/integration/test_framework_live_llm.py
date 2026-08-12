"""Live-LLM Layer 1 — simple-agent matrix (shopping cart, 2 tools).

Smoke layer that proves the adapter→backend wire works for each
(adapter, provider) combination. For richer agent behavior see:
  - test_framework_live_complex.py        (5 tools, multi-step branching)
  - test_framework_live_tool_errors.py    (failing tools, agent recovery)
  - test_framework_live_multi_agent.py    (orchestrator → specialist)
  - test_framework_live_streaming.py      (streamed events)

Marker: live_llm + simple_agent
"""

from __future__ import annotations

import json
import os
from uuid import uuid4

import pytest

from . import _live_helpers as h


# ─── Provider-specific generic loops ─────────────────────────────────

def _gemini_generic_loop(model: str, query: str) -> str:
    from google import genai
    from google.genai import types
    import decimalai

    client = h.gemini_client()
    tool_decls = types.Tool(function_declarations=[
        {
            "name": "get_price",
            "description": "Return the unit price in dollars for a given item.",
            "parameters": {"type": "OBJECT",
                           "properties": {"item": {"type": "STRING"}},
                           "required": ["item"]},
        },
        {
            "name": "calculate",
            "description": "Evaluate a basic math expression and return the result.",
            "parameters": {"type": "OBJECT",
                           "properties": {"expression": {"type": "STRING"}},
                           "required": ["expression"]},
        },
    ])
    handlers = {"get_price": h.lookup_price, "calculate": h.safe_calculate}

    contents: list = [{"role": "user", "parts": [{"text": query}]}]
    for _ in range(6):
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(tools=[tool_decls]),
        )
        decimalai.log_llm_call(
            model=model,
            input=h._loggable_turns(contents),
            output={"content": resp.text or ""},
            input_tokens=getattr(resp.usage_metadata, "prompt_token_count", None),
            output_tokens=getattr(resp.usage_metadata, "candidates_token_count", None),
        )
        cand = resp.candidates[0]
        func_calls = [p for p in (cand.content.parts or []) if getattr(p, "function_call", None)]
        if not func_calls:
            return resp.text or ""
        # Append the model turn verbatim — reconstructing it from function_call alone
        # drops the thought_signature that Gemini 3.x requires on the next turn.
        contents.append(cand.content)
        for fc in func_calls:
            name = fc.function_call.name
            args = dict(fc.function_call.args or {})
            result = handlers[name](**args)
            decimalai.log_tool_call(name=name, input=args, output={"result": result})
            contents.append({
                "role": "user",
                "parts": [{"function_response": {"name": name, "response": {"result": result}}}],
            })
    raise RuntimeError("Agent loop exceeded safety bound")


def _openai_generic_loop(model: str, query: str) -> str:
    from openai import OpenAI
    import decimalai

    client = OpenAI()
    tools = [
        {"type": "function",
         "function": {"name": "get_price",
                      "description": "Return the unit price in dollars for a given item.",
                      "parameters": {"type": "object",
                                     "properties": {"item": {"type": "string"}},
                                     "required": ["item"]}}},
        {"type": "function",
         "function": {"name": "calculate",
                      "description": "Evaluate a basic math expression and return the result.",
                      "parameters": {"type": "object",
                                     "properties": {"expression": {"type": "string"}},
                                     "required": ["expression"]}}},
    ]
    handlers = {"get_price": h.lookup_price, "calculate": h.safe_calculate}

    messages: list = [{"role": "user", "content": query}]
    for _ in range(6):
        resp = client.chat.completions.create(model=model, messages=messages, tools=tools)
        msg = resp.choices[0].message
        decimalai.log_llm_call(
            model=model,
            input=[{"role": m["role"], "content": str(m.get("content", ""))} for m in messages],
            output={"content": msg.content or ""},
            input_tokens=getattr(resp.usage, "prompt_tokens", None),
            output_tokens=getattr(resp.usage, "completion_tokens", None),
        )
        if not getattr(msg, "tool_calls", None):
            return msg.content or ""
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            result = handlers[tc.function.name](**args)
            decimalai.log_tool_call(name=tc.function.name, input=args, output={"result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps({"result": result})})
    raise RuntimeError("Agent loop exceeded safety bound")


def _anthropic_generic_loop(model: str, query: str) -> str:
    import decimalai

    tools = [
        {"name": "get_price",
         "description": "Return the unit price in dollars for a given item.",
         "input_schema": {"type": "object",
                          "properties": {"item": {"type": "string"}},
                          "required": ["item"]}},
        {"name": "calculate",
         "description": "Evaluate a basic math expression and return the result.",
         "input_schema": {"type": "object",
                          "properties": {"expression": {"type": "string"}},
                          "required": ["expression"]}},
    ]
    handlers = {"get_price": h.lookup_price, "calculate": h.safe_calculate}
    # Delegate to the shared native-Anthropic driver so this generic cell
    # exercises Anthropic's usage.input_tokens/output_tokens fields — the exact
    # token shape this provider lane exists to verify.
    return h.anthropic_tool_loop(
        model, query,
        tools=tools, handlers=handlers,
        log_llm=decimalai.log_llm_call, log_tool=decimalai.log_tool_call,
    )


_GENERIC_LOOPS = {
    "google": _gemini_generic_loop,
    "openai": _openai_generic_loop,
    "anthropic": _anthropic_generic_loop,
}

# Provider → the import that must be present for that provider's generic loop.
_GENERIC_IMPORT = {"google": "google.genai", "openai": "openai", "anthropic": "anthropic"}


# ═══════════════════════════════════════════════════════════════════
# Adapter 1: generic decorator
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.simple_agent
@pytest.mark.parametrize("provider, model", h.matrix("generic"))
def test_generic_simple_agent(provider, model):
    h.require_key_for(provider)
    pytest.importorskip(_GENERIC_IMPORT[provider])

    import decimalai

    agent_name = h.unique_agent(f"generic-{provider}-simple")
    loop_fn = _GENERIC_LOOPS[provider]

    @decimalai.trace(agent_name=agent_name)
    def run() -> str:
        return loop_fn(model, h.SHOPPING_QUERY)

    answer = run()
    assert str(h.SHOPPING_EXPECTED_TOTAL) in answer

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=2)


# ═══════════════════════════════════════════════════════════════════
# Adapter 2: LangChain via langgraph.create_react_agent
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.simple_agent
@pytest.mark.parametrize("provider, model", h.matrix("langchain"))
def test_langchain_simple_agent(provider, model):
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
        """Evaluate a basic math expression and return the result."""
        return h.safe_calculate(expression)

    agent_name = h.unique_agent(f"langchain-{provider}-simple")
    agent = h.make_react_agent(llm, tools=[get_price, calculate])

    handler = CallbackHandler(agent_name=agent_name)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": h.SHOPPING_QUERY}]},
        config={"callbacks": [handler], "run_name": "live-shopping-agent"},
    )
    final_text = str(result["messages"][-1].content)
    assert str(h.SHOPPING_EXPECTED_TOTAL) in final_text

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=2)


# ═══════════════════════════════════════════════════════════════════
# Adapter 3: OpenAI Agents SDK
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.simple_agent
@pytest.mark.parametrize("provider, model", h.matrix("openai_agents"))
def test_openai_agents_simple_agent(provider, model):
    h.require_key_for(provider)
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool
    from agents.tracing import flush_traces, set_trace_processors
    from decimalai.openai_agents import DecimalTracingProcessor

    agent_name = h.unique_agent(f"openai-agents-{provider}-simple")
    processor = DecimalTracingProcessor(agent_name=agent_name)
    set_trace_processors([processor])

    @function_tool
    def get_price(item: str) -> int:
        """Return the unit price in dollars for a given item."""
        return h.lookup_price(item)

    @function_tool
    def calculate(expression: str) -> float:
        """Evaluate a basic math expression and return the result."""
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

    result = Runner.run_sync(agent, h.SHOPPING_QUERY)
    answer = str(result.final_output)
    assert str(h.SHOPPING_EXPECTED_TOTAL) in answer

    flush_traces()
    processor.shutdown()
    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=2)
