"""Live-LLM Layer 3 — tool-error recovery.

Same 5-tool customer support agent, but the user references an invalid
order ID. `get_order_details` raises ValueError. The agent should:
  * Observe the tool error.
  * Respond gracefully (apologize / ask for clarification / no crash).
  * Produce a trace that includes the failed tool span (status=error).

Asserts:
  * Final answer is non-empty and not just an exception trace.
  * Trace landed in the backend with status='success' (the agent recovered).
  * At least one tool span has an error indicator (status / output / name).

Marker: live_llm + error_recovery.
"""

from __future__ import annotations

import pytest

from . import _live_helpers as h


def _has_error_evidence(detail: dict) -> bool:
    """Look across spans for evidence that a tool failed and was reported."""
    for s in detail.get("spans", []):
        if s.get("span_type") != "tool":
            continue
        # The DecimalAI SDK records tool errors in a few possible ways
        # depending on adapter version — be permissive.
        if str(s.get("status", "")).lower() in ("error", "failed", "fail"):
            return True
        output = s.get("output_preview") or s.get("output") or ""
        if isinstance(output, str) and ("error" in output.lower() or "not found" in output.lower()):
            return True
        if s.get("error_message") or s.get("error"):
            return True
    return False


_ERROR_SYSTEM = (
    "You are a customer support agent. If a tool reports the order can't be "
    "found, do NOT retry the same lookup — instead, apologize and ask the "
    "customer to verify the ID."
)

# get_customer + get_order_details in google.genai shape. get_order_details
# raises for unknown IDs, so the loop records it as a tool error span.
_ERROR_TOOLS_GEMINI = [
    {"name": "get_customer",
     "description": "Return profile (name, tier, joined date) for a customer ID.",
     "parameters": {"type": "OBJECT",
                    "properties": {"customer_id": {"type": "INTEGER"}},
                    "required": ["customer_id"]}},
    {"name": "get_order_details",
     "description": "Return order details — status, items, total — by order ID.",
     "parameters": {"type": "OBJECT",
                    "properties": {"order_id": {"type": "STRING"}},
                    "required": ["order_id"]}},
]

# Same two tools in anthropic shape — flat {name, description, input_schema}.
_ERROR_TOOLS_ANTHROPIC = [
    {"name": "get_customer",
     "description": "Return profile (name, tier, joined date) for a customer ID.",
     "input_schema": {"type": "object",
                      "properties": {"customer_id": {"type": "integer"}},
                      "required": ["customer_id"]}},
    {"name": "get_order_details",
     "description": "Return order details — status, items, total — by order ID.",
     "input_schema": {"type": "object",
                      "properties": {"order_id": {"type": "string"}},
                      "required": ["order_id"]}},
]


# ═══════════════════════════════════════════════════════════════════
# LangChain via langgraph
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.error_recovery
@pytest.mark.parametrize("provider, model", h.matrix("langchain"))
def test_langchain_tool_error_recovery(provider, model):
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
    def get_order_details(order_id: str) -> dict | str:
        """Return order details — status, items, total — by order ID.

        Returns an error string instead of raising so the LLM can observe
        the failure and adapt — this is the production wrapping pattern.
        """
        try:
            return h.get_order_details(order_id)
        except ValueError as e:
            return f"ERROR: {e}"

    @tool
    def search_faq(query: str) -> str:
        """Search the FAQ knowledge base."""
        return h.search_faq(query)

    agent_name = h.unique_agent(f"langchain-{provider}-error")
    agent = h.make_react_agent(llm, tools=[get_customer, get_order_details, search_faq])

    handler = CallbackHandler(agent_name=agent_name)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": h.SUPPORT_QUERY_BAD_ORDER}]},
        config={"callbacks": [handler], "run_name": "live-error-recovery"},
    )

    final_text = str(result["messages"][-1].content)
    assert final_text.strip(), f"Agent produced empty final answer: {final_text!r}"

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    # Trace-level should still be success — the agent recovered, didn't crash.
    assert detail.get("status") == "success", (
        f"Trace status not 'success' after recovery: {detail.get('status')!r}"
    )
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=1)
    assert _has_error_evidence(detail), (
        f"No error-tool evidence found in trace {detail['id']} — "
        f"the failing tool call was not captured as an error."
    )


# ═══════════════════════════════════════════════════════════════════
# OpenAI Agents SDK
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.error_recovery
@pytest.mark.parametrize("provider, model", h.matrix("openai_agents"))
def test_openai_agents_tool_error_recovery(provider, model):
    h.require_key_for(provider)
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool
    from agents.tracing import flush_traces, set_trace_processors
    from decimalai.openai_agents import DecimalTracingProcessor

    agent_name = h.unique_agent(f"openai-agents-{provider}-error")
    processor = DecimalTracingProcessor(agent_name=agent_name)
    set_trace_processors([processor])

    @function_tool
    def get_customer(customer_id: int) -> dict:
        """Return profile (name, tier, joined date) for a customer ID."""
        return h.get_customer(customer_id)

    @function_tool
    def get_order_details(order_id: str) -> dict:
        """Return order details — status, items, total — by order ID."""
        return h.get_order_details(order_id)  # raises for unknown IDs

    @function_tool
    def search_faq(query: str) -> str:
        """Search the FAQ knowledge base."""
        return h.search_faq(query)

    agent = Agent(
        name=agent_name,
        instructions=(
            "You are a customer support agent. Use your tools to help the "
            "customer. If a tool reports the order can't be found, do not "
            "retry the same lookup — instead, apologize and ask the customer "
            "to verify the order ID."
        ),
        model=h.openai_agents_model(provider, model),
        tools=[get_customer, get_order_details, search_faq],
    )

    result = Runner.run_sync(agent, h.SUPPORT_QUERY_BAD_ORDER)
    answer = str(result.final_output)
    assert answer.strip(), f"Agent produced empty answer: {answer!r}"

    flush_traces()
    processor.shutdown()
    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    assert detail.get("status") == "success", (
        f"Trace status not 'success' after recovery: {detail.get('status')!r}"
    )
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=1)
    assert _has_error_evidence(detail), (
        f"No error-tool evidence found in trace {detail['id']}."
    )


# ═══════════════════════════════════════════════════════════════════
# Generic decorator — hand-rolled OpenAI loop with failing tool
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.error_recovery
@pytest.mark.parametrize("provider, model", h.matrix("generic"))
def test_generic_tool_error_recovery(provider, model):
    """Hand-rolled loop where get_order_details raises for an unknown order
    ID. The agent observes the error string, apologizes, and the failure is
    recorded as a tool span with status='error'. 'google' drives the native
    google.genai SDK; 'openai' the hand-rolled OpenAI loop."""
    h.require_key_for(provider)
    pytest.importorskip(
        {"google": "google.genai", "openai": "openai", "anthropic": "anthropic"}[provider]
    )
    import json as _json

    import decimalai

    agent_name = h.unique_agent(f"generic-{provider}-error")

    if provider == "google":
        @decimalai.trace(agent_name=agent_name)
        def run() -> str:
            return h.gemini_tool_loop(
                model, h.SUPPORT_QUERY_BAD_ORDER,
                tool_declarations=_ERROR_TOOLS_GEMINI,
                handlers={"get_customer": h.get_customer,
                          "get_order_details": h.get_order_details},
                system=_ERROR_SYSTEM,
                log_llm=decimalai.log_llm_call,
                log_tool=decimalai.log_tool_call,
            )

        answer = run()
        assert answer.strip(), f"Agent produced empty answer: {answer!r}"
        h.flush_sdk_sender()
        traces = h.poll_for_trace(agent_name)
        detail = h.get_trace_detail(traces[0]["id"])
        assert detail.get("status") == "success", (
            f"Trace status not 'success' after recovery: {detail.get('status')!r}"
        )
        h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=1)
        assert _has_error_evidence(detail), (
            f"No error-tool evidence found in trace {detail['id']}."
        )
        return

    if provider == "anthropic":
        @decimalai.trace(agent_name=agent_name)
        def run() -> str:
            return h.anthropic_tool_loop(
                model, h.SUPPORT_QUERY_BAD_ORDER,
                tools=_ERROR_TOOLS_ANTHROPIC,
                handlers={"get_customer": h.get_customer,
                          "get_order_details": h.get_order_details},
                system=_ERROR_SYSTEM,
                log_llm=decimalai.log_llm_call,
                log_tool=decimalai.log_tool_call,
            )

        answer = run()
        assert answer.strip(), f"Agent produced empty answer: {answer!r}"
        h.flush_sdk_sender()
        traces = h.poll_for_trace(agent_name)
        detail = h.get_trace_detail(traces[0]["id"])
        assert detail.get("status") == "success", (
            f"Trace status not 'success' after recovery: {detail.get('status')!r}"
        )
        h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=1)
        assert _has_error_evidence(detail), (
            f"No error-tool evidence found in trace {detail['id']}."
        )
        return

    from openai import OpenAI

    tools = [
        {"type": "function", "function": {
            "name": "get_customer",
            "description": "Return profile (name, tier, joined date) for a customer ID.",
            "parameters": {"type": "object",
                           "properties": {"customer_id": {"type": "integer"}},
                           "required": ["customer_id"]}}},
        {"type": "function", "function": {
            "name": "get_order_details",
            "description": "Return order details — status, items, total — by order ID.",
            "parameters": {"type": "object",
                           "properties": {"order_id": {"type": "string"}},
                           "required": ["order_id"]}}},
    ]

    @decimalai.trace(agent_name=agent_name)
    def run() -> str:
        client = OpenAI()
        messages: list = [
            {"role": "system", "content": (
                "You are a customer support agent. If a tool reports the "
                "order can't be found, do NOT retry the same lookup — "
                "instead, apologize and ask the customer to verify the ID."
            )},
            {"role": "user", "content": h.SUPPORT_QUERY_BAD_ORDER},
        ]
        for _ in range(6):
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=tools,
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
                "role": "assistant", "content": msg.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = _json.loads(tc.function.arguments) if tc.function.arguments else {}
                name = tc.function.name
                if name == "get_customer":
                    try:
                        result, status = h.get_customer(**args), "success"
                    except Exception as e:
                        result, status = f"ERROR: {e}", "error"
                elif name == "get_order_details":
                    try:
                        result, status = h.get_order_details(**args), "success"
                    except Exception as e:
                        # The deliberate failure path — record as tool error.
                        result, status = f"ERROR: {e}", "error"
                else:
                    result, status = f"ERROR: unknown tool {name}", "error"
                decimalai.log_tool_call(
                    name=name, input=args, output={"result": result}, status=status,
                )
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": _json.dumps({"result": result}),
                })
        raise RuntimeError("Agent loop exceeded safety bound")

    answer = run()
    assert answer.strip(), f"Agent produced empty answer: {answer!r}"

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    assert detail.get("status") == "success", (
        f"Trace status not 'success' after recovery: {detail.get('status')!r}"
    )
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=1)
    assert _has_error_evidence(detail), (
        f"No error-tool evidence found in trace {detail['id']}."
    )
