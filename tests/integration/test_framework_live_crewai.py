"""Live-LLM Layer 6 — CrewAI through the OpenTelemetry SpanExporter.

CrewAI is OTEL-native: its OpenInference instrumentor emits spans using Arize
Phoenix conventions (``llm.model_name``, inline ``llm.output_messages.*.message
.tool_calls.*``, ``llm.tools.*.tool.json_schema``, ``openinference.span.kind``)
rather than the GenAI semconv the Layer-5 test uses. This layer proves the
exporter handles *that* dialect end-to-end with a **real Gemini-backed Crew**:

  * Build a 2-tool (`get_price`, `calculate`) shopping Crew on the current
    Gemini budget-tier model (resolved by ``h.matrix``) and actually run it.
  * Bridge it through ``CrewAIInstrumentor`` + ``GoogleGenAIInstrumentor`` →
    ``DecimalSpanExporter`` → backend.
  * Assert the assembled trace carries every LLM step, both tools (harvested
    from the inline OpenInference tool-call attributes), and an auto-detected
    manifest — all as a *single* trace, not fragmented per export batch.

The single-trace assertion is load-bearing: a multi-second Crew run spans
several BatchSpanProcessor flush intervals, so the exporter must buffer spans
by trace_id and finalize on the root span. If that regresses, the run
fragments and no single trace holds both distinct tools.

Marker: live_llm + otel + crewai.
"""

from __future__ import annotations

import json

import pytest

from . import _live_helpers as h


@pytest.mark.live_llm
@pytest.mark.otel
@pytest.mark.crewai
@pytest.mark.parametrize("provider, model", h.matrix("crewai", only=("google",)))
def test_crewai_crew_ingests_through_otel(provider, model):
    """A real CrewAI Crew on Gemini → CrewAI/GoogleGenAI OpenInference spans →
    DecimalSpanExporter → one backend trace with both tools + a manifest."""
    h.require_key_for(provider)
    pytest.importorskip("crewai")
    pytest.importorskip("openinference.instrumentation.crewai")
    pytest.importorskip("openinference.instrumentation.google_genai")
    pytest.importorskip("opentelemetry.sdk")

    from crewai import Agent, Crew, LLM, Task
    from crewai.tools import tool
    from openinference.instrumentation.crewai import CrewAIInstrumentor
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from decimalai.otel import DecimalSpanExporter

    agent_name = h.unique_agent("crewai-gemini-shopping")

    @tool("get_price")
    def get_price(item: str) -> str:
        """Return the unit price in dollars for a given item name."""
        return str(h.lookup_price(item))

    @tool("calculate")
    def calculate(expression: str) -> str:
        """Evaluate a simple arithmetic expression like '3*10+2*25'."""
        return str(h.safe_calculate(expression))

    # Local provider with the DecimalAI exporter — same exporter install() wires
    # globally, just without touching the process-wide tracer provider (OTEL
    # honors set_tracer_provider only once per process).
    provider_otel = TracerProvider(
        resource=Resource.create({SERVICE_NAME: "decimal-agent"})
    )
    exporter = DecimalSpanExporter(agent_name=agent_name)
    provider_otel.add_span_processor(BatchSpanProcessor(exporter))

    crew_instr = CrewAIInstrumentor()
    genai_instr = GoogleGenAIInstrumentor()
    crew_instr.instrument(tracer_provider=provider_otel)
    genai_instr.instrument(tracer_provider=provider_otel)
    try:
        llm = LLM(model=f"gemini/{model}", temperature=0)
        agent = Agent(
            role="Shopping Assistant",
            goal="Compute the total cost of a cart using the tools.",
            backstory="You look up item prices and total them with the calculator.",
            tools=[get_price, calculate],
            llm=llm,
            verbose=False,
        )
        # 3 widgets + 2 gadgets → h.SHOPPING_EXPECTED_TOTAL (80); the quantities
        # are baked into that helper constant, so keep them in sync here.
        task = Task(
            description=(
                "A cart has 3 widgets and 2 gadgets. Use get_price to look up the "
                "unit price of 'widget' and of 'gadget', then use calculate with "
                "the expression '3*<widget_price>+2*<gadget_price>'. "
                "Reply with only the final total number."
            ),
            expected_output="The total cost as a single number.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = str(crew.kickoff())

        # Flush the whole span tree → buffered + finalized into one RunTrace.
        provider_otel.force_flush()
    finally:
        crew_instr.uninstrument()
        genai_instr.uninstrument()
        provider_otel.shutdown()

    assert str(h.SHOPPING_EXPECTED_TOTAL) in result, (
        f"Crew didn't compute the expected total {h.SHOPPING_EXPECTED_TOTAL}: {result!r}"
    )

    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)

    # Fragmentation guard: a multi-second Crew run straddles several batch-flush
    # intervals. The exporter must buffer by trace_id and emit ONE trace, not
    # one-per-batch. (The min_distinct_tools=2 assertion below would also fail
    # on a fragment, but assert the count directly for a clear signal.)
    assert len(traces) == 1, (
        f"Expected exactly 1 trace for the Crew run, got {len(traces)} — "
        f"OTEL trace fragmentation regressed. Trace ids: {[t['id'] for t in traces]}"
    )

    detail = h.get_trace_detail(traces[0]["id"])

    # ≥1 LLM step + both tools (get_price, calculate), harvested from the inline
    # OpenInference tool-call attributes, plus an auto-detected manifest.
    h.assert_rich_agent_trace(
        detail, min_llm_calls=1, min_tool_calls=2, min_distinct_tools=2,
    )

    # The model fingerprint must survive the llm.model_name → LlmCallRecord map.
    llm_models = " ".join(
        str(c.get("model_name") or c.get("model") or c.get("model_fingerprint") or "")
        for c in detail.get("llm_calls", [])
    ).lower()
    assert "gemini" in llm_models, (
        f"CrewAI trace didn't record the Gemini model. llm_calls={detail.get('llm_calls')}"
    )


@pytest.mark.live_llm
@pytest.mark.otel
@pytest.mark.crewai
@pytest.mark.parametrize("provider, model", h.matrix("crewai", only=("google",)))
def test_crewai_complex_crew_ingests_through_otel(provider, model):
    """A real 5-tool CrewAI customer-support Crew on Gemini → OpenInference
    spans → DecimalSpanExporter → ONE trace with ≥3 distinct tools + a manifest.

    This is the harder sibling of the 2-tool shopping crew above. A 4-step
    support resolution (get_customer → get_order_details → search_faq →
    calculate_refund) runs longer and emits more LLM + tool spans, so it
    straddles more BatchSpanProcessor flush intervals — exactly the condition
    that fragments a trace if the exporter stops buffering by trace_id and
    finalizing on the root span. The `len(traces) == 1` assertion is the load
    test for that path; `min_distinct_tools=3` proves the inline OpenInference
    tool-call attributes are harvested across the whole multi-tool chain.

    Marker: live_llm + otel + crewai.
    """
    h.require_key_for(provider)
    pytest.importorskip("crewai")
    pytest.importorskip("openinference.instrumentation.crewai")
    pytest.importorskip("openinference.instrumentation.google_genai")
    pytest.importorskip("opentelemetry.sdk")

    from crewai import Agent, Crew, LLM, Task
    from crewai.tools import tool
    from openinference.instrumentation.crewai import CrewAIInstrumentor
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from decimalai.otel import DecimalSpanExporter

    agent_name = h.unique_agent("crewai-gemini-support")

    # Five tools, same fixtures as the cross-framework complex layer. CrewAI
    # tools must return strings, so dict/list returns are JSON-encoded. Args are
    # coerced defensively — Gemini sometimes hands an int field a string.
    @tool("get_customer")
    def get_customer(customer_id: int) -> str:
        """Return profile (name, tier, joined date) for a customer ID."""
        try:
            return json.dumps(h.get_customer(int(customer_id)))
        except ValueError as e:
            return f"ERROR: {e}"

    @tool("get_orders")
    def get_orders(customer_id: int) -> str:
        """Return the list of orders for a customer."""
        return json.dumps(h.get_orders(int(customer_id)))

    @tool("get_order_details")
    def get_order_details(order_id: str) -> str:
        """Return order details — status, items, total — by order ID."""
        try:
            return json.dumps(h.get_order_details(str(order_id)))
        except ValueError as e:
            return f"ERROR: {e}"

    @tool("search_faq")
    def search_faq(query: str) -> str:
        """Search the FAQ knowledge base and return the best-matching entry."""
        return str(h.search_faq(str(query)))

    @tool("calculate_refund")
    def calculate_refund(order_total: float, condition: str) -> str:
        """Compute refund + store credit for an order given its condition."""
        return json.dumps(h.calculate_refund(float(order_total), str(condition)))

    provider_otel = TracerProvider(
        resource=Resource.create({SERVICE_NAME: "decimal-agent"})
    )
    exporter = DecimalSpanExporter(agent_name=agent_name)
    provider_otel.add_span_processor(BatchSpanProcessor(exporter))

    crew_instr = CrewAIInstrumentor()
    genai_instr = GoogleGenAIInstrumentor()
    crew_instr.instrument(tracer_provider=provider_otel)
    genai_instr.instrument(tracer_provider=provider_otel)
    try:
        llm = LLM(model=f"gemini/{model}", temperature=0)
        agent = Agent(
            role="Customer Support Specialist",
            goal="Resolve the customer's refund request using the tools.",
            backstory=(
                "You look up customers and orders, check the refund policy, and "
                "compute refunds. You never guess values — you call a tool."
            ),
            tools=[get_customer, get_orders, get_order_details, search_faq, calculate_refund],
            llm=llm,
            verbose=False,
        )
        # Prescriptive, numbered steps with concrete args — same lesson as the
        # cross-framework complex layer: Gemini shortcuts a soft prompt down to
        # 1-2 tool calls, which would fail the ≥3-distinct-tool assertion. This
        # mandates a 4-tool chain so the test stays deterministic.
        task = Task(
            description=(
                "A customer wrote in: \"I'm customer 1234. My order ORD-9001 "
                "arrived broken. What can you do for me?\" Resolve it by "
                "completing ALL of these steps in order, one tool call per "
                "step — never guess a value:\n"
                "1. Call get_customer with customer_id 1234 to confirm tier.\n"
                "2. Call get_order_details with order_id 'ORD-9001' for the "
                "status and total.\n"
                "3. Call search_faq with 'damaged delivery' for the policy.\n"
                "4. Call calculate_refund with the order total from step 2 and "
                "condition 'broken'.\n"
                "Then reply with the refund amount, the store credit, and the "
                "cited policy."
            ),
            expected_output="The refund amount, store credit, and policy citation.",
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        result = str(crew.kickoff())

        provider_otel.force_flush()
    finally:
        crew_instr.uninstrument()
        genai_instr.uninstrument()
        provider_otel.shutdown()

    # ORD-9001 total is 1200; condition 'broken' → full refund of 1200.
    result_norm = result.replace(",", "").replace("$", "")
    assert "1200" in result_norm, (
        f"Crew didn't compute the expected refund (1200): {result!r}"
    )

    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)

    # Fragmentation guard — the load-bearing assertion for this layer.
    assert len(traces) == 1, (
        f"Expected exactly 1 trace for the support Crew run, got {len(traces)} — "
        f"OTEL trace fragmentation regressed. Trace ids: {[t['id'] for t in traces]}"
    )

    detail = h.get_trace_detail(traces[0]["id"])

    # ≥3 LLM steps + ≥3 distinct tools harvested from the inline OpenInference
    # tool-call attributes, plus an auto-detected manifest.
    h.assert_rich_agent_trace(
        detail, min_llm_calls=3, min_tool_calls=3, min_distinct_tools=3,
    )

    llm_models = " ".join(
        str(c.get("model_name") or c.get("model") or c.get("model_fingerprint") or "")
        for c in detail.get("llm_calls", [])
    ).lower()
    assert "gemini" in llm_models, (
        f"CrewAI trace didn't record the Gemini model. llm_calls={detail.get('llm_calls')}"
    )
