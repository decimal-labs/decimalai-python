"""Lock in: decimalai.otel populates trace-level input/output previews.

Deep-audit finding (sdk-integrations): _assemble_trace initialized
``user_input``/``final_output`` to None and NEVER assigned them, so every
trace from ``decimalai.otel.install()`` had empty previews even though the
span attributes carried input/output. The fix derives the trace-level
previews from the root span (preferred) or the LLM spans (fallback).

No backend / no real OTEL — spans and the client are mocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest


class _MockContext:
    def __init__(self, trace_id, span_id):
        self.trace_id = trace_id
        self.span_id = span_id
        self.trace_flags = 1


class _MockStatus:
    def __init__(self, code="OK"):
        self.status_code = code


class _MockResource:
    def __init__(self, service_name="test-service"):
        self.attributes = {"service.name": service_name}


class _MockSpan:
    def __init__(self, name, trace_id, span_id, parent_span_id=None, attributes=None):
        self.name = name
        self.context = _MockContext(trace_id, span_id)
        self.parent = _MockContext(trace_id, parent_span_id) if parent_span_id else None
        self.attributes = attributes or {}
        self.start_time = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.end_time = self.start_time + 100_000_000
        self.status = _MockStatus("OK")
        self.resource = _MockResource()


@pytest.fixture(autouse=True)
def _reset_sdk():
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "m", "status": "active"}
    yield


def _assemble(spans):
    from decimalai.otel import DecimalSpanExporter

    with patch.dict("sys.modules", {
        "opentelemetry": MagicMock(),
        "opentelemetry.sdk": MagicMock(),
        "opentelemetry.sdk.trace": MagicMock(),
        "opentelemetry.sdk.trace.export": MagicMock(SpanExportResult=MagicMock()),
    }):
        exporter = DecimalSpanExporter(agent_name="test-agent")
        result = exporter._assemble_trace(spans)
    assert result is not None
    run_trace, _seen_model, _seen_tools, _seen_prompts = result
    return run_trace


def test_llm_span_input_output_become_trace_previews():
    """When only the LLM span carries input/output, the trace-level
    previews fall back to it (first input, last output)."""
    tid = 0xABC
    spans = [
        _MockSpan("agent-run", tid, 0x01),
        _MockSpan(
            "llm-call", tid, 0x02, parent_span_id=0x01,
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.input": "what is 2+2?",
                "gen_ai.output": "4",
            },
        ),
    ]
    rt = _assemble(spans)
    assert rt.user_input_preview == "what is 2+2?", (
        "trace user_input_preview must be populated from the LLM span input"
    )
    assert rt.final_output_preview == "4", (
        "trace final_output_preview must be populated from the LLM span output"
    )


def test_root_span_previews_win_over_llm():
    """When the root span carries its own input/output, those take
    precedence over the per-LLM-span fallback."""
    tid = 0xDEF
    spans = [
        _MockSpan(
            "agent-run", tid, 0x01,
            attributes={"input": "ROOT IN", "output": "ROOT OUT"},
        ),
        _MockSpan(
            "llm-call", tid, 0x02, parent_span_id=0x01,
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.input": "llm in",
                "gen_ai.output": "llm out",
            },
        ),
    ]
    rt = _assemble(spans)
    assert rt.user_input_preview == "ROOT IN"
    assert rt.final_output_preview == "ROOT OUT"


def test_no_previews_stays_none():
    """No input/output attrs anywhere → previews remain None (not crash)."""
    tid = 0x111
    spans = [
        _MockSpan("agent-run", tid, 0x01),
        _MockSpan(
            "llm-call", tid, 0x02, parent_span_id=0x01,
            attributes={"gen_ai.request.model": "gpt-4o"},
        ),
    ]
    rt = _assemble(spans)
    assert rt.user_input_preview is None
    assert rt.final_output_preview is None
