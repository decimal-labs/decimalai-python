"""Live-LLM Layer 5 — OpenTelemetry SpanExporter ingest path.

DecimalAI ships an OTEL ``SpanExporter`` (``decimalai.otel``) whose job is to
route spans from *any* OTEL-native framework — CrewAI, AutoGen, Haystack,
Semantic Kernel, Google ADK, … — into the backend. This layer proves that path
end-to-end with a **real Gemini call**:

  * Emit a span tree the way an OTEL-instrumented agent would — a root span, two
    `execute_tool …` spans (GenAI semconv), and a `chat` span carrying GenAI
    attributes (`gen_ai.system`, `gen_ai.request.model`, token usage) around a
    real `google.genai` request.
  * Run it through ``DecimalSpanExporter`` and assert the assembled trace +
    auto-detected manifest (model + tools) land in the backend.

We wire a *local* ``TracerProvider`` rather than the global ``install()`` helper:
OTEL only honors ``set_tracer_provider`` once per process, which is unfriendly to
a shared pytest run. The exporter under test is identical either way — ``install()``
just attaches this same ``DecimalSpanExporter`` to the global provider.

Marker: live_llm + otel.
"""

from __future__ import annotations

import os

import pytest

from . import _live_helpers as h


def _emit_tool_span(tracer, tool_name: str, fn, *call_args):
    """Emit a tool-execution span (GenAI semconv: ``execute_tool <name>``)
    wrapping a real handler call, the way an OTEL instrumentation would."""
    with tracer.start_as_current_span(f"execute_tool {tool_name}") as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", tool_name)
        result = fn(*call_args)
        span.set_attribute("gen_ai.tool.message", str(result))
        return result


@pytest.mark.live_llm
@pytest.mark.otel
@pytest.mark.parametrize("provider, model", h.matrix("otel", only=("google",)))
def test_otel_exporter_ingests_real_gemini_run(provider, model):
    """A real Gemini call wrapped in OTEL GenAI spans → DecimalSpanExporter →
    backend trace with the right model fingerprint, tool spans, and a manifest."""
    h.require_key_for(provider)
    pytest.importorskip("google.genai")
    pytest.importorskip("opentelemetry.sdk")

    from google import genai
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from decimalai.otel import DecimalSpanExporter

    agent_name = h.unique_agent("otel-gemini-shopping")

    # Local provider with the DecimalAI exporter — same exporter install() wires
    # globally, just without touching the process-wide tracer provider.
    provider_otel = TracerProvider(
        resource=Resource.create({SERVICE_NAME: "decimal-agent"})
    )
    exporter = DecimalSpanExporter(agent_name=agent_name)
    provider_otel.add_span_processor(BatchSpanProcessor(exporter))
    tracer = provider_otel.get_tracer("decimal-live-otel-test")

    try:
        with tracer.start_as_current_span("otel-shopping-agent"):
            # Real tool executions, each emitted as its own OTEL span.
            widget = _emit_tool_span(tracer, "get_price", h.lookup_price, "widget")
            gadget = _emit_tool_span(tracer, "get_price", h.lookup_price, "gadget")
            expr = f"3*{widget}+2*{gadget}"
            total = _emit_tool_span(tracer, "calculate", h.safe_calculate, expr)

            # Real Gemini call inside a GenAI-semconv LLM span.
            client = h.gemini_client()
            prompt = (
                f"A shopping cart totals ${int(total)}. Reply with a single short "
                f"sentence confirming the order, and include the number {int(total)}."
            )
            with tracer.start_as_current_span("chat gemini") as llm_span:
                llm_span.set_attribute("gen_ai.system", "google")
                llm_span.set_attribute("gen_ai.request.model", model)
                llm_span.set_attribute("gen_ai.request.temperature", 0.0)
                resp = client.models.generate_content(
                    model=model,
                    contents=[{"role": "user", "parts": [{"text": prompt}]}],
                )
                um = getattr(resp, "usage_metadata", None)
                llm_span.set_attribute(
                    "gen_ai.usage.input_tokens",
                    getattr(um, "prompt_token_count", 0) or 0,
                )
                llm_span.set_attribute(
                    "gen_ai.usage.output_tokens",
                    getattr(um, "candidates_token_count", 0) or 0,
                )
                llm_span.set_attribute("gen_ai.response.finish_reasons", ["stop"])
                answer = resp.text or ""

        assert str(int(total)) in answer, (
            f"Gemini answer didn't echo the computed total {int(total)}: {answer!r}"
        )

        # Export the whole span tree as one batch → one assembled RunTrace.
        provider_otel.force_flush()
    finally:
        provider_otel.shutdown()

    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])

    # 1 real LLM call + 2 distinct tools (get_price, calculate) over 3 tool spans,
    # plus an auto-detected manifest — all assembled purely from OTEL spans.
    h.assert_rich_agent_trace(
        detail, min_llm_calls=1, min_tool_calls=2, min_distinct_tools=2,
    )

    # The model fingerprint must survive the gen_ai.* → LlmCallRecord mapping.
    llm_models = " ".join(
        str(c.get("model_name") or c.get("model") or c.get("model_fingerprint") or "")
        for c in detail.get("llm_calls", [])
    ).lower()
    assert "gemini" in llm_models, (
        f"OTel trace didn't record the Gemini model. llm_calls={detail.get('llm_calls')}"
    )
