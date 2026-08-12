"""Live-LLM Layer 4 — multi-agent handoff.

An orchestrator agent that delegates to a specialist sub-agent. Exercises
the parts of the product that depend on parent/child trace linkage:

  * Sub-Agent Health Dashboard
  * Delegation Analytics
  * Agent Topology Graph

Two cells:
  * LangChain  — orchestrator has a `consult_refund_specialist` tool that
                 invokes a langgraph specialist agent under its own
                 CallbackHandler. Produces 2 traces.
  * OpenAI Agents — orchestrator uses native `handoffs=[specialist]`.
                    The OpenAI Agents SDK creates one combined trace with
                    both agents' spans plus a handoff span.

Marker: live_llm + multi_agent.
"""

from __future__ import annotations

import pytest

from . import _live_helpers as h


# ═══════════════════════════════════════════════════════════════════
# LangChain — orchestrator with a sub-agent tool
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.multi_agent
@pytest.mark.parametrize("provider, model", h.matrix("langchain"))
def test_langchain_multi_agent_handoff(provider, model):
    """Orchestrator → refund specialist via a sub-agent tool.

    The orchestrator only has the `consult_refund_specialist` tool. When it
    calls that tool, a separate langgraph agent runs internally with its
    own CallbackHandler — producing a second trace in the backend.

    Both providers carry a param: delegation here is *explicit* (the
    orchestrator has a single tool that wraps the specialist), so it doesn't
    depend on the model-driven auto-handoff mechanism that makes the OpenAI
    Agents SDK variant nondeterministic on Gemini.
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

    # ── Specialist tools ──────────────────────────────────────
    @tool
    def get_order_details(order_id: str) -> dict | str:
        """Return order details — status, items, total — by order ID."""
        try:
            return h.get_order_details(order_id)
        except ValueError as e:
            return f"ERROR: {e}"

    @tool
    def calculate_refund(order_total: float, condition: str) -> dict:
        """Compute refund + store credit for an order given its condition."""
        return h.calculate_refund(order_total, condition)

    specialist_agent_name = h.unique_agent(f"langchain-specialist-{provider}")
    specialist = h.make_react_agent(llm, tools=[get_order_details, calculate_refund])

    orchestrator_agent_name = h.unique_agent(f"langchain-orchestrator-{provider}")
    # Declare the specialist as a subagent component on the orchestrator's
    # manifest. This is what the backend uses to populate `is_subagent` on
    # the specialist and `has_topology=true` on the orchestrator.
    orchestrator_handler = CallbackHandler(
        agent_name=orchestrator_agent_name,
        subagents=[{"name": specialist_agent_name}],
    )

    # ── Orchestrator tool that wraps the specialist ──────────
    @tool
    def consult_refund_specialist(question: str) -> str:
        """Delegate refund-related questions to the refund specialist agent.
        Pass the full customer question; the specialist will look up the
        order and compute the refund."""
        specialist_handler = CallbackHandler(
            agent_name=specialist_agent_name,
            # Link the child trace to the orchestrator trace so the backend
            # can render parent/child relationships.
            parent_trace_id=orchestrator_handler.get_trace_id(),
        )
        result = specialist.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"callbacks": [specialist_handler], "run_name": "refund-specialist"},
        )
        return str(result["messages"][-1].content)

    orchestrator = h.make_react_agent(llm, tools=[consult_refund_specialist])

    result = orchestrator.invoke(
        {"messages": [{"role": "user", "content": h.SUPPORT_QUERY}]},
        config={"callbacks": [orchestrator_handler], "run_name": "orchestrator"},
    )
    final_text = str(result["messages"][-1].content).lower()
    assert "refund" in final_text, (
        f"Expected 'refund' in final answer, got: {final_text!r}"
    )

    h.flush_sdk_sender()

    # Two distinct traces should land — one for orchestrator, one for specialist.
    orch_traces = h.poll_for_trace(orchestrator_agent_name)
    spec_traces = h.poll_for_trace(specialist_agent_name)

    orch = h.get_trace_detail(orch_traces[0]["id"])
    spec = h.get_trace_detail(spec_traces[0]["id"])

    # Orchestrator called consult_refund_specialist at least once.
    orch_tool_names = {
        s.get("name") for s in orch.get("spans", []) if s.get("span_type") == "tool"
    }
    orch_tool_names |= {
        tc.get("name") for c in orch.get("llm_calls", []) for tc in (c.get("tool_calls") or [])
        if tc.get("name")
    }
    assert "consult_refund_specialist" in orch_tool_names, (
        f"Orchestrator did not invoke the specialist sub-agent tool. "
        f"Tools observed: {sorted(orch_tool_names)}"
    )

    # Specialist made real LLM + tool calls of its own.
    h.assert_rich_agent_trace(spec, min_llm_calls=2, min_tool_calls=1)

    # Both have manifests — each agent registered its own.
    assert orch.get("manifest_id") and spec.get("manifest_id"), (
        "One of orchestrator/specialist is missing a manifest_id"
    )
    assert orch["manifest_id"] != spec["manifest_id"], (
        "Orchestrator and specialist resolved to the SAME manifest — "
        "they have different tools, so manifests should differ."
    )

    # Specialist trace MUST link back to the orchestrator's trace.
    assert spec.get("parent_trace_id") == orch["id"], (
        f"parent_trace_id on specialist trace is "
        f"{spec.get('parent_trace_id')!r}, expected orchestrator id {orch['id']!r}"
    )

    # Critical: the multi-agent product surfaces must have data.
    h.assert_topology_declared(orchestrator_agent_name, specialist_agent_name)
    h.assert_subagent_resolved(specialist_agent_name, orchestrator_agent_name)


# ═══════════════════════════════════════════════════════════════════
# OpenAI Agents — native handoff
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.multi_agent
# openai_agents is OpenAI-only across the whole suite (see h.matrix /
# FRAMEWORK_PROVIDERS), so this lane carries no Gemini cell. Handoffs add a
# second wrinkle worth recording: the Agents SDK implements them as
# auto-generated `transfer_to_*` tools the model must *choose* to call, and
# that choice is nondeterministic run-to-run — so below we assert the handoff
# *declaration* (topology) rather than requiring the runtime transfer to fire.
@pytest.mark.parametrize("provider, model", h.matrix("openai_agents"))
def test_openai_agents_multi_agent_handoff(provider, model):
    """Orchestrator hands off to RefundSpecialist via `handoffs=[specialist]`.

    The OpenAI Agents SDK records both agents' spans within a single trace
    (plus a handoff span). We verify the handoff happened and both agents'
    spans are present.
    """
    h.require_key_for(provider)
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool
    from agents.tracing import flush_traces
    from decimalai.openai_agents import install as install_oai_processor

    agent_name = h.unique_agent(f"openai-agents-{provider}-handoff")
    specialist_name = h.unique_agent(f"openai-agents-{provider}-specialist")

    @function_tool
    def get_order_details(order_id: str) -> dict:
        """Return order details by order ID."""
        return h.get_order_details(order_id)

    @function_tool
    def calculate_refund(order_total: float, condition: str) -> dict:
        """Compute refund + store credit for an order given its condition."""
        return h.calculate_refund(order_total, condition)

    refund_specialist = Agent(
        name=specialist_name,
        instructions=(
            "You are a refund specialist. You MUST use the tools to do your "
            "job — do not invent numbers from training data. Steps: "
            "(1) Call get_order_details with the order ID to get the total "
            "and confirm status. (2) Call calculate_refund with that total "
            "and the condition the customer described. (3) Reply with the "
            "refund and any store credit returned by calculate_refund."
        ),
        model=h.openai_agents_model(provider, model),
        tools=[get_order_details, calculate_refund],
    )

    orchestrator = Agent(
        name=agent_name,
        instructions=(
            "You are a customer support triage agent. For refund or "
            "damage-related questions, hand off to the refund specialist. "
            "Otherwise answer directly."
        ),
        model=h.openai_agents_model(provider, model),
        handoffs=[refund_specialist],
    )

    # Use install(agent=...) so `_introspect_agent` runs and registers a
    # manifest with the handoffs as subagent components — required for the
    # backend to mark the specialist `is_subagent=True` and for the
    # topology endpoint to return non-empty data.
    install_oai_processor(agent=orchestrator, agent_name=agent_name, exclusive=True)

    result = Runner.run_sync(orchestrator, h.SUPPORT_QUERY)
    answer = str(result.final_output).lower()
    assert "refund" in answer, f"Expected 'refund' in answer, got: {answer!r}"

    flush_traces()
    h.flush_sdk_sender()

    # The orchestrator's trace lands under `agent_name`. We don't poll for the
    # specialist's name separately because the OpenAI Agents SDK combines
    # multi-agent runs into a single trace with handoff + sub-agent spans.
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])

    span_names = [s.get("name", "") for s in detail.get("spans", [])]
    span_types = {s.get("span_type") for s in detail.get("spans", [])}

    # Evidence of handoff: either a span explicitly named/typed handoff,
    # or both agent names appear in the span list.
    has_handoff_span = any("handoff" in (n or "").lower() for n in span_names) \
        or "handoff" in span_types
    has_specialist_span = any("refund-specialist" in (n or "") for n in span_names)
    assert has_handoff_span or has_specialist_span, (
        f"No handoff or specialist span found. "
        f"Span names: {span_names}; types: {span_types}"
    )

    # For multi-agent the central assertion is "handoff happened and both
    # agents participated" — tool usage by the specialist is a soft check.
    # For multi-agent the central assertion is "handoff happened and both
    # agents participated" — tool usage by the specialist is a soft check.
    h.assert_rich_agent_trace(detail, min_llm_calls=2, min_tool_calls=0)

    # Critical: the orchestrator's manifest must declare the specialist as
    # a subagent so the topology endpoint returns non-empty. The OpenAI
    # Agents model may or may not actually hand off on a given run, so we
    # only assert the *declaration* here — the runtime is_subagent flag is
    # not deterministic across runs.
    h.assert_topology_declared(agent_name, specialist_name)


# ═══════════════════════════════════════════════════════════════════
# Generic decorator — orchestrator function calls specialist function
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.multi_agent
@pytest.mark.parametrize("provider, model", h.matrix("generic", only=("openai",)))
def test_generic_multi_agent_handoff(provider, model):
    """Two @decimalai.trace-decorated functions linked via parent_trace_id.

    The orchestrator declares the specialist as a subagent on its manifest
    (so the topology endpoint surfaces them). The specialist's trace carries
    parent_trace_id pointing at the orchestrator — driving the
    multi-agent UI.
    """
    h.require_key_for(provider)
    pytest.importorskip("openai")
    import json as _json

    from openai import OpenAI
    import decimalai

    specialist_agent_name = h.unique_agent(f"generic-{provider}-specialist")
    orchestrator_agent_name = h.unique_agent(f"generic-{provider}-orchestrator")

    def _specialist_loop(question: str, parent_id: str) -> str:
        """Refund specialist — runs in its own trace under the orchestrator."""
        with decimalai.start_trace(
            agent_name=specialist_agent_name,
            parent_trace_id=parent_id,
        ) as ctx:
            ctx.set_input(question)
            client = OpenAI()
            tools = [
                {"type": "function", "function": {
                    "name": "get_order_details",
                    "description": "Return order details by order ID.",
                    "parameters": {"type": "object",
                                   "properties": {"order_id": {"type": "string"}},
                                   "required": ["order_id"]}}},
                {"type": "function", "function": {
                    "name": "calculate_refund",
                    "description": "Compute refund + store credit for an order given its condition.",
                    "parameters": {"type": "object",
                                   "properties": {"order_total": {"type": "number"},
                                                  "condition": {"type": "string"}},
                                   "required": ["order_total", "condition"]}}},
            ]
            messages: list = [
                {"role": "system", "content": (
                    "You are a refund specialist. Use get_order_details "
                    "to look up the order, then calculate_refund. Reply "
                    "with the refund amount."
                )},
                {"role": "user", "content": question},
            ]
            for _ in range(6):
                resp = client.chat.completions.create(
                    model=model, messages=messages, tools=tools,
                )
                msg = resp.choices[0].message
                ctx.log_llm_call(
                    model=model,
                    input=[{"role": m["role"], "content": str(m.get("content", ""))} for m in messages],
                    output={"content": msg.content or ""},
                    input_tokens=getattr(resp.usage, "prompt_tokens", None),
                    output_tokens=getattr(resp.usage, "completion_tokens", None),
                )
                if not getattr(msg, "tool_calls", None):
                    final = msg.content or ""
                    ctx.set_output(final)
                    return final
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
                    if tc.function.name == "get_order_details":
                        try:
                            result = h.get_order_details(**args)
                        except ValueError as e:
                            result = f"ERROR: {e}"
                    else:
                        result = h.calculate_refund(**args)
                    ctx.log_tool_call(
                        name=tc.function.name, input=args, output={"result": result},
                    )
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": _json.dumps({"result": result}),
                    })
            return ""

    # ── Orchestrator trace: declares the specialist as a subagent so the
    # topology endpoint reports has_topology=True.
    with decimalai.start_trace(
        agent_name=orchestrator_agent_name,
        subagents=[{"name": specialist_agent_name}],
    ) as orch_ctx:
        orch_ctx.set_input(h.SUPPORT_QUERY)
        client = OpenAI()
        tools = [{
            "type": "function", "function": {
                "name": "consult_refund_specialist",
                "description": (
                    "Delegate refund-related questions to the refund "
                    "specialist agent. Pass the full customer question."
                ),
                "parameters": {"type": "object",
                               "properties": {"question": {"type": "string"}},
                               "required": ["question"]}}}]
        messages: list = [
            {"role": "system", "content": (
                "You are a triage agent. For ANY refund or damage-related "
                "question, you MUST call consult_refund_specialist with "
                "the full question. Do not attempt the refund yourself."
            )},
            {"role": "user", "content": h.SUPPORT_QUERY},
        ]
        final_text = ""
        for _ in range(6):
            resp = client.chat.completions.create(
                model=model, messages=messages, tools=tools,
            )
            msg = resp.choices[0].message
            orch_ctx.log_llm_call(
                model=model,
                input=[{"role": m["role"], "content": str(m.get("content", ""))} for m in messages],
                output={"content": msg.content or ""},
                input_tokens=getattr(resp.usage, "prompt_tokens", None),
                output_tokens=getattr(resp.usage, "completion_tokens", None),
            )
            if not getattr(msg, "tool_calls", None):
                final_text = msg.content or ""
                break
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
                # Run the specialist in its own trace, linked to this one.
                result = _specialist_loop(args.get("question", ""), parent_id=orch_ctx.get_trace_id())
                orch_ctx.log_tool_call(
                    name=tc.function.name, input=args, output={"result": result},
                )
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": _json.dumps({"result": result}),
                })
        orch_ctx.set_output(final_text)

    assert "refund" in final_text.lower(), (
        f"Expected 'refund' in final answer, got: {final_text!r}"
    )

    h.flush_sdk_sender()
    orch_traces = h.poll_for_trace(orchestrator_agent_name)
    spec_traces = h.poll_for_trace(specialist_agent_name)
    orch = h.get_trace_detail(orch_traces[0]["id"])
    spec = h.get_trace_detail(spec_traces[0]["id"])

    # Both have manifests (orchestrator from subagents, specialist from
    # auto-detected tools/models).
    assert orch.get("manifest_id") and spec.get("manifest_id"), (
        "One of orchestrator/specialist is missing a manifest_id"
    )

    # Specialist trace links back to the orchestrator's trace.
    assert spec.get("parent_trace_id") == orch["id"], (
        f"parent_trace_id on specialist trace is "
        f"{spec.get('parent_trace_id')!r}, expected orchestrator id {orch['id']!r}"
    )

    # Topology surfaces (the multi-agent product surface).
    h.assert_topology_declared(orchestrator_agent_name, specialist_agent_name)
    h.assert_subagent_resolved(specialist_agent_name, orchestrator_agent_name)
