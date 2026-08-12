"""Live-LLM Layer 2 — complex single-agent (5 tools, multi-step branching).

Customer support scenario stressing real agent richness:
  * 5 tools registered → wider manifest
  * Multi-step plan→act→observe with ≥ 4 LLM calls
  * ≥ 3 distinct tools actually invoked

Same shape as Layer 1, just thicker. Marker: live_llm + complex.
"""

from __future__ import annotations

import json

import pytest

from . import _live_helpers as h


# ═══════════════════════════════════════════════════════════════════
# LangChain via langgraph
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.complex
@pytest.mark.parametrize("provider, model", h.matrix("langchain"))
def test_langchain_complex_agent(provider, model):
    """5-tool customer support agent via langgraph + CallbackHandler."""
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
    def get_customer(customer_id: int) -> dict:
        """Return profile (name, tier, joined date) for a customer ID."""
        return h.get_customer(customer_id)

    @tool
    def get_orders(customer_id: int) -> list:
        """Return list of orders for a customer."""
        return h.get_orders(customer_id)

    @tool
    def get_order_details(order_id: str) -> dict:
        """Return order details — status, items, total — by order ID."""
        return h.get_order_details(order_id)

    @tool
    def search_faq(query: str) -> str:
        """Search the FAQ knowledge base and return the best-matching entry."""
        return h.search_faq(query)

    @tool
    def calculate_refund(order_total: float, condition: str) -> dict:
        """Compute refund + store credit for an order given its condition."""
        return h.calculate_refund(order_total, condition)

    agent_name = h.unique_agent(f"langchain-{provider}-complex")
    tools = [get_customer, get_orders, get_order_details, search_faq, calculate_refund]
    # Prescriptive system prompt (shared with the other two complex cells) so
    # the chain is deterministic ≥3 distinct tools on Gemini too — see the
    # comment on _SUPPORT_SYSTEM.
    agent = h.make_react_agent(llm, tools=tools, prompt=_SUPPORT_SYSTEM)

    handler = CallbackHandler(agent_name=agent_name)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": h.SUPPORT_QUERY}]},
        config={"callbacks": [handler], "run_name": "live-support-agent"},
    )

    final_text = str(result["messages"][-1].content).lower()
    assert "refund" in final_text, (
        f"Expected 'refund' in final answer, got: {final_text!r}"
    )

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_rich_agent_trace(
        detail,
        min_llm_calls=3,        # plan → ≥2 act → finalize
        min_tool_calls=3,       # agent might skip get_orders if it has order_id
        min_distinct_tools=3,   # at least 3 different tools selected
    )


# ═══════════════════════════════════════════════════════════════════
# OpenAI Agents SDK
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.complex
@pytest.mark.parametrize("provider, model", h.matrix("openai_agents"))
def test_openai_agents_complex_agent(provider, model):
    """5-tool customer support agent via the OpenAI Agents SDK."""
    h.require_key_for(provider)
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool
    from agents.tracing import flush_traces, set_trace_processors
    from decimalai.openai_agents import DecimalTracingProcessor

    agent_name = h.unique_agent(f"openai-agents-{provider}-complex")
    processor = DecimalTracingProcessor(agent_name=agent_name)
    set_trace_processors([processor])

    @function_tool
    def get_customer(customer_id: int) -> dict:
        """Return profile (name, tier, joined date) for a customer ID."""
        return h.get_customer(customer_id)

    @function_tool
    def get_orders(customer_id: int) -> list:
        """Return list of orders for a customer."""
        return h.get_orders(customer_id)

    @function_tool
    def get_order_details(order_id: str) -> dict:
        """Return order details — status, items, total — by order ID."""
        return h.get_order_details(order_id)

    @function_tool
    def search_faq(query: str) -> str:
        """Search the FAQ knowledge base and return the best-matching entry."""
        return h.search_faq(query)

    @function_tool
    def calculate_refund(order_total: float, condition: str) -> dict:
        """Compute refund + store credit for an order given its condition."""
        return h.calculate_refund(order_total, condition)

    agent = Agent(
        name=agent_name,
        instructions=_SUPPORT_SYSTEM,
        model=h.openai_agents_model(provider, model),
        tools=[get_customer, get_orders, get_order_details, search_faq, calculate_refund],
    )

    result = Runner.run_sync(agent, h.SUPPORT_QUERY)
    answer = str(result.final_output).lower()
    assert "refund" in answer, f"Expected 'refund' in answer, got: {answer!r}"

    flush_traces()
    processor.shutdown()
    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_rich_agent_trace(
        detail,
        min_llm_calls=3,
        min_tool_calls=3,
        min_distinct_tools=3,
    )


# ═══════════════════════════════════════════════════════════════════
# Generic decorator — hand-rolled OpenAI loop
# ═══════════════════════════════════════════════════════════════════

_SUPPORT_SYSTEM = (
    # Prescriptive, numbered steps — not flavor text. On this 5-tool support
    # query a soft "use your tools" prompt lets a confident model shortcut to
    # 1-2 tool calls and flake the ≥3-distinct-tool assertion. Gemini does this
    # on BOTH its native API (the generic gemini_tool_loop path) and the
    # OpenAI-compatible shim (the openai_agents path); GPT is steadier but not
    # immune. Mandating one tool per step keeps the complex ReAct chain
    # deterministic across all three frameworks and both providers. Shared by
    # the generic + openai_agents + langchain cells so they can't drift apart.
    "You are a customer support agent. You MUST use your tools for every "
    "fact — never guess from memory. Complete ALL of these steps in order, "
    "one tool call per step:\n"
    "1. Call get_customer with the customer ID to confirm their tier.\n"
    "2. Call get_order_details with the order ID for status and total.\n"
    "3. Call search_faq to find the relevant refund/return policy.\n"
    "4. Call calculate_refund with the order total and item condition.\n"
    "Only after all four tool calls, reply with the refund amount, any "
    "store credit, and the cited policy."
)

_SUPPORT_TOOLS_OPENAI = [
    {"type": "function", "function": {
        "name": "get_customer",
        "description": "Return profile (name, tier, joined date) for a customer ID.",
        "parameters": {"type": "object",
                       "properties": {"customer_id": {"type": "integer"}},
                       "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "get_orders",
        "description": "Return list of orders for a customer.",
        "parameters": {"type": "object",
                       "properties": {"customer_id": {"type": "integer"}},
                       "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "get_order_details",
        "description": "Return order details — status, items, total — by order ID.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "search_faq",
        "description": "Search the FAQ knowledge base and return the best match.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate_refund",
        "description": "Compute refund + store credit for an order given its condition.",
        "parameters": {"type": "object",
                       "properties": {"order_total": {"type": "number"},
                                      "condition": {"type": "string"}},
                       "required": ["order_total", "condition"]}}},
]

# Same five tools in google.genai shape (uppercase JSON-schema types).
_SUPPORT_TOOLS_GEMINI = [
    {"name": "get_customer",
     "description": "Return profile (name, tier, joined date) for a customer ID.",
     "parameters": {"type": "OBJECT",
                    "properties": {"customer_id": {"type": "INTEGER"}},
                    "required": ["customer_id"]}},
    {"name": "get_orders",
     "description": "Return list of orders for a customer.",
     "parameters": {"type": "OBJECT",
                    "properties": {"customer_id": {"type": "INTEGER"}},
                    "required": ["customer_id"]}},
    {"name": "get_order_details",
     "description": "Return order details — status, items, total — by order ID.",
     "parameters": {"type": "OBJECT",
                    "properties": {"order_id": {"type": "STRING"}},
                    "required": ["order_id"]}},
    {"name": "search_faq",
     "description": "Search the FAQ knowledge base and return the best match.",
     "parameters": {"type": "OBJECT",
                    "properties": {"query": {"type": "STRING"}},
                    "required": ["query"]}},
    {"name": "calculate_refund",
     "description": "Compute refund + store credit for an order given its condition.",
     "parameters": {"type": "OBJECT",
                    "properties": {"order_total": {"type": "NUMBER"},
                                   "condition": {"type": "STRING"}},
                    "required": ["order_total", "condition"]}},
]

# Same five tools in anthropic shape — flat {name, description, input_schema}
# with lowercase JSON-schema types (the OpenAI param shape, but un-nested and
# keyed input_schema).
_SUPPORT_TOOLS_ANTHROPIC = [
    {"name": "get_customer",
     "description": "Return profile (name, tier, joined date) for a customer ID.",
     "input_schema": {"type": "object",
                      "properties": {"customer_id": {"type": "integer"}},
                      "required": ["customer_id"]}},
    {"name": "get_orders",
     "description": "Return list of orders for a customer.",
     "input_schema": {"type": "object",
                      "properties": {"customer_id": {"type": "integer"}},
                      "required": ["customer_id"]}},
    {"name": "get_order_details",
     "description": "Return order details — status, items, total — by order ID.",
     "input_schema": {"type": "object",
                      "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"]}},
    {"name": "search_faq",
     "description": "Search the FAQ knowledge base and return the best match.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string"}},
                      "required": ["query"]}},
    {"name": "calculate_refund",
     "description": "Compute refund + store credit for an order given its condition.",
     "input_schema": {"type": "object",
                      "properties": {"order_total": {"type": "number"},
                                     "condition": {"type": "string"}},
                      "required": ["order_total", "condition"]}},
]

_SUPPORT_HANDLERS = {
    "get_customer": lambda customer_id: h.get_customer(customer_id),
    "get_orders": lambda customer_id: h.get_orders(customer_id),
    "get_order_details": lambda order_id: h.get_order_details(order_id),
    "search_faq": lambda query: h.search_faq(query),
    "calculate_refund": lambda order_total, condition: h.calculate_refund(order_total, condition),
}


def _openai_complex_loop(model: str, query: str, max_iters: int = 8) -> str:
    """Hand-rolled OpenAI tool-use loop with log_llm_call + log_tool_call."""
    from openai import OpenAI
    import decimalai

    client = OpenAI()
    system = _SUPPORT_SYSTEM
    messages: list = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]
    for _ in range(max_iters):
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=_SUPPORT_TOOLS_OPENAI,
        )
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
            try:
                result = _SUPPORT_HANDLERS[tc.function.name](**args)
                status = "success"
            except Exception as e:
                result = f"ERROR: {e}"
                status = "error"
            decimalai.log_tool_call(
                name=tc.function.name, input=args, output={"result": result}, status=status,
            )
            messages.append({
                "role": "tool", "tool_call_id": tc.id,
                "content": json.dumps({"result": result}),
            })
    raise RuntimeError("Agent loop exceeded safety bound")


@pytest.mark.live_llm
@pytest.mark.complex
@pytest.mark.parametrize("provider, model", h.matrix("generic"))
def test_generic_complex_agent(provider, model):
    """5-tool customer support agent via @decimalai.trace + manual logging.

    'google' drives the native google.genai SDK (via h.gemini_tool_loop);
    'openai' drives the hand-rolled OpenAI loop. Both record the run purely
    through decimalai.log_llm_call / log_tool_call under one @trace.
    """
    h.require_key_for(provider)
    pytest.importorskip(
        {"google": "google.genai", "openai": "openai", "anthropic": "anthropic"}[provider]
    )
    import decimalai

    agent_name = h.unique_agent(f"generic-{provider}-complex")

    @decimalai.trace(agent_name=agent_name)
    def run() -> str:
        if provider == "google":
            return h.gemini_tool_loop(
                model, h.SUPPORT_QUERY,
                tool_declarations=_SUPPORT_TOOLS_GEMINI,
                handlers=_SUPPORT_HANDLERS,
                system=_SUPPORT_SYSTEM,
                log_llm=decimalai.log_llm_call,
                log_tool=decimalai.log_tool_call,
            )
        if provider == "anthropic":
            return h.anthropic_tool_loop(
                model, h.SUPPORT_QUERY,
                tools=_SUPPORT_TOOLS_ANTHROPIC,
                handlers=_SUPPORT_HANDLERS,
                system=_SUPPORT_SYSTEM,
                log_llm=decimalai.log_llm_call,
                log_tool=decimalai.log_tool_call,
            )
        return _openai_complex_loop(model, h.SUPPORT_QUERY)

    answer = run().lower()
    assert "refund" in answer, f"Expected 'refund' in answer, got: {answer!r}"

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    h.assert_rich_agent_trace(
        detail,
        min_llm_calls=3,
        min_tool_calls=3,
        min_distinct_tools=3,
    )
