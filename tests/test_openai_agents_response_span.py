"""Tests for the Responses API span handler (decimalai.openai_agents._handle_response).

The default OpenAI Agents path (OpenAIResponsesModel) emits `response` spans,
not `generation` spans. Pre-fix, _handle_response extracted only response.id —
every trace on the default path shipped with llm_calls=[] and a token summary
of all zeros, and final_output_preview stored a raw ResponseOutputMessage repr
instead of the assistant's text.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest


class _MockSpanData:
    """Lightweight mock for OpenAI Agents SpanData subclasses."""

    def __init__(self, span_type: str, **kwargs):
        self._type = span_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def type(self) -> str:
        return self._type


class _MockSpan:
    """Lightweight mock for OpenAI Agents Span."""

    def __init__(
        self,
        trace_id: str,
        span_id: str | None = None,
        parent_id: str | None = None,
        span_data: _MockSpanData | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        error: dict | None = None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id or str(uuid4())
        self.parent_id = parent_id
        self.span_data = span_data
        self.started_at = started_at or datetime.now(timezone.utc).isoformat()
        self.ended_at = ended_at or datetime.now(timezone.utc).isoformat()
        self.error = error


class _MockTrace:
    """Lightweight mock for OpenAI Agents Trace."""

    def __init__(self, trace_id: str, name: str = "test-workflow"):
        self.trace_id = trace_id
        self.name = name


# ── Synthetic Responses API objects ─────────────────────────
# Plain attribute holders (not MagicMock) so the handler's isinstance
# guards see real str/int values, mirroring the openai SDK's typed models.


class _OutputText:
    def __init__(self, text: str):
        self.type = "output_text"
        self.text = text


class _OutputMessage:
    def __init__(self, *parts: _OutputText):
        self.type = "message"
        self.role = "assistant"
        self.content = list(parts)

    def __repr__(self) -> str:  # mirrors the openai SDK repr the bug leaked
        return f"ResponseOutputMessage(content={self.content!r}, role='assistant')"


class _FunctionToolCall:
    """A tool-call output item — has no `content`, produces no text."""

    def __init__(self, name: str):
        self.type = "function_call"
        self.name = name


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens


class _SyntheticResponse:
    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini-2024-07-18",
        usage: _Usage | None = None,
        output=(),
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ):
        self.id = f"resp_{uuid4().hex[:16]}"
        self.model = model
        self.usage = usage
        self.output = list(output)
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens


# ── Setup/Teardown ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Reset global SDK state before each test."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "test-manifest-id", "status": "active"}

    import decimalai.openai_agents as oai
    from decimalai.schema.manifest import ManifestTracker
    oai._manifest_id = None
    # Fresh tracker so a same-hash manifest registered by an earlier test
    # in this process doesn't dedupe away this test's registration.
    oai._manifest_tracker = ManifestTracker()
    yield


def _run_response_spans(processor, trace_id, spans):
    """Feed spans through the processor and return the ingested RunTrace."""
    import decimalai._config as cfg

    trace = _MockTrace(trace_id=trace_id)
    processor.on_trace_start(trace)
    for span in spans:
        processor.on_span_end(span)
    processor.on_trace_end(trace)

    from decimalai._config import _sender
    _sender.flush()

    cfg._client.ingest_trace.assert_called_once()
    return cfg._client.ingest_trace.call_args[0][0]


# ── Tests ───────────────────────────────────────────────────


class TestResponseSpanLlmCall:
    """A response span must produce an LlmCallRecord (the default path)."""

    def test_response_span_creates_llm_call_with_tokens(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        response = _SyntheticResponse(
            usage=_Usage(input_tokens=28, output_tokens=2),
            output=[_OutputMessage(_OutputText("4"))],
            temperature=0.7,
            max_output_tokens=256,
        )
        started = datetime.now(timezone.utc)
        span = _MockSpan(
            trace_id=trace_id,
            span_data=_MockSpanData(
                "response",
                response=response,
                input="What is 2+2?",
                usage=None,
            ),
            started_at=started.isoformat(),
            ended_at=(started + timedelta(milliseconds=1500)).isoformat(),
        )

        run_trace = _run_response_spans(processor, trace_id, [span])

        assert len(run_trace.llm_calls) == 1
        call = run_trace.llm_calls[0]
        assert call.model_name == "gpt-4o-mini-2024-07-18"
        assert call.provider == "openai"
        assert call.input_tokens == 28
        assert call.output_tokens == 2
        assert call.temperature == 0.7
        assert call.max_output_tokens == 256
        assert call.latency_ms == 1500
        assert call.rendered_input == [{"role": "user", "content": "What is 2+2?"}]
        assert call.output == {"content": "4"}

    def test_response_span_is_llm_span(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        response = _SyntheticResponse(output=[_OutputMessage(_OutputText("hi"))])
        span = _MockSpan(
            trace_id=trace_id,
            span_data=_MockSpanData("response", response=response, input="hi", usage=None),
        )

        run_trace = _run_response_spans(processor, trace_id, [span])

        llm_spans = [s for s in run_trace.spans if s.span_type.value == "llm"]
        assert len(llm_spans) == 1
        assert llm_spans[0].name == "response:gpt-4o-mini-2024-07-18"

    def test_usage_falls_back_to_span_data_dict(self):
        """Streaming paths stamp usage on span_data, not response.usage."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        response = _SyntheticResponse(
            usage=None,
            output=[_OutputMessage(_OutputText("streamed"))],
        )
        span = _MockSpan(
            trace_id=trace_id,
            span_data=_MockSpanData(
                "response",
                response=response,
                input="hi",
                usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            ),
        )

        run_trace = _run_response_spans(processor, trace_id, [span])

        call = run_trace.llm_calls[0]
        assert call.input_tokens == 10
        assert call.output_tokens == 3

    def test_response_model_feeds_manifest_autodetection(self):
        """The response model must register a manifest (no Agent passed)."""
        import decimalai._config as cfg
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        response = _SyntheticResponse(output=[_OutputMessage(_OutputText("ok"))])
        span = _MockSpan(
            trace_id=trace_id,
            span_data=_MockSpanData("response", response=response, input="hi", usage=None),
        )

        _run_response_spans(processor, trace_id, [span])

        cfg._client.register_manifest.assert_called_once()


class TestResponseSpanFinalOutputPreview:
    """final_output_preview must hold the text, not an object repr."""

    def test_preview_extracts_text(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        response = _SyntheticResponse(output=[_OutputMessage(_OutputText("4"))])
        span = _MockSpan(
            trace_id=trace_id,
            span_data=_MockSpanData("response", response=response, input="2+2?", usage=None),
        )

        run_trace = _run_response_spans(processor, trace_id, [span])

        assert run_trace.final_output_preview == "4"
        assert "ResponseOutputMessage" not in run_trace.final_output_preview

    def test_textless_turn_keeps_earlier_preview(self):
        """A pure tool-call turn must not clobber the last text preview."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        text_response = _SyntheticResponse(output=[_OutputMessage(_OutputText("done"))])
        tool_response = _SyntheticResponse(output=[_FunctionToolCall("get_weather")])
        spans = [
            _MockSpan(
                trace_id=trace_id,
                span_data=_MockSpanData("response", response=text_response, input="q", usage=None),
            ),
            _MockSpan(
                trace_id=trace_id,
                span_data=_MockSpanData("response", response=tool_response, input="q", usage=None),
            ),
        ]

        run_trace = _run_response_spans(processor, trace_id, spans)

        assert run_trace.final_output_preview == "done"
        # Both turns still count as LLM calls
        assert len(run_trace.llm_calls) == 2


class TestResponseOutputTextHelper:
    def test_prefers_output_text_property(self):
        from decimalai.openai_agents import _response_output_text

        response = _SyntheticResponse(output=[_OutputMessage(_OutputText("walked"))])
        response.output_text = "from property"
        assert _response_output_text(response) == "from property"

    def test_walks_output_items_without_property(self):
        from decimalai.openai_agents import _response_output_text

        response = _SyntheticResponse(
            output=[
                _FunctionToolCall("lookup"),
                _OutputMessage(_OutputText("a"), _OutputText("b")),
            ]
        )
        assert _response_output_text(response) == "ab"

    def test_returns_none_for_textless_response(self):
        from decimalai.openai_agents import _response_output_text

        response = _SyntheticResponse(output=[_FunctionToolCall("lookup")])
        assert _response_output_text(response) is None
