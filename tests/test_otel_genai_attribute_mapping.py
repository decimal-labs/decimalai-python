"""Lock in: gen_ai.* attribute mapping keeps content and counts apart.

Deep-audit finding (community tier): ``_preview_from_attrs`` matched the
bare direction substring ("input"/"output") against every attribute key,
so a span carrying only GenAI-semconv usage counters produced previews of
"42"/"7" — the TOKEN COUNTS — while otel.mdx's own attribute table says
``gen_ai.prompt``/``gen_ai.completion`` are the content keys and
``gen_ai.usage.*_tokens`` are counts. A sibling mapping bug: AG2 stamps
``gen_ai.request.model`` onto its agent/conversation spans as metadata,
which turned every such span into a phantom LlmCallRecord.

No backend / no real OTEL — spans and the client are mocked, mirroring
tests/test_otel_trace_previews.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from decimalai.otel import _preview_from_attrs


class _MockContext:
    def __init__(self, trace_id, span_id):
        self.trace_id = trace_id
        self.span_id = span_id
        self.trace_flags = 1


class _MockStatus:
    def __init__(self, code="OK"):
        self.status_code = code


class _MockSpan:
    def __init__(self, name, trace_id, span_id, parent_span_id=None, attributes=None):
        self.name = name
        self.context = _MockContext(trace_id, span_id)
        self.parent = _MockContext(trace_id, parent_span_id) if parent_span_id else None
        self.attributes = attributes or {}
        self.start_time = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.end_time = self.start_time + 100_000_000
        self.status = _MockStatus("OK")
        self.resource = None


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

    exporter = DecimalSpanExporter(agent_name="test-agent")
    result = exporter._assemble_trace(spans)
    assert result is not None
    run_trace, _seen_model, _seen_tools, _seen_prompts = result
    return run_trace


# ── Previews are content, token fields are counts ──────────────


def test_genai_semconv_previews_are_content_not_token_counts():
    """A span with both content keys and usage counters must map content
    to the previews and counters to the token fields — never crossed."""
    tid = 0xA1
    spans = [
        _MockSpan(
            "chat gpt-4o", tid, 0x01,
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 42,
                "gen_ai.usage.output_tokens": 7,
                "gen_ai.prompt": "What is 2+2?",
                "gen_ai.completion": "The answer is 4.",
            },
        ),
    ]
    rt = _assemble(spans)
    assert rt.user_input_preview == "What is 2+2?"
    assert rt.final_output_preview == "The answer is 4."
    llm = rt.llm_calls[0]
    assert llm.input_tokens == 42
    assert llm.output_tokens == 7


def test_token_counts_alone_never_become_previews():
    """Usage counters with no content keys → previews stay None instead of
    surfacing "42"/"7"."""
    attrs = {
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 42,
        "gen_ai.usage.output_tokens": 7,
        "llm.usage.prompt_tokens": 42,
        "llm.usage.completion_tokens": 7,
    }
    assert _preview_from_attrs(attrs, "input") is None
    assert _preview_from_attrs(attrs, "output") is None

    rt = _assemble([_MockSpan("chat", 0xB2, 0x01, attributes=attrs)])
    assert rt.user_input_preview is None
    assert rt.final_output_preview is None


def test_output_preview_never_falls_back_to_the_prompt():
    """gen_ai.prompt is an input-side key: with no output content present,
    the output preview must be None, not the prompt text."""
    attrs = {"gen_ai.request.model": "gpt-4o", "gen_ai.prompt": "the prompt"}
    assert _preview_from_attrs(attrs, "input") == "the prompt"
    assert _preview_from_attrs(attrs, "output") is None


def test_indexed_prompt_completion_keys_from_docs_table():
    """otel.mdx's table maps gen_ai.prompt.0.content → input and
    gen_ai.completion.0.content → output; the indexed spellings must work."""
    attrs = {
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.usage.input_tokens": 9,
        "gen_ai.usage.output_tokens": 3,
        "gen_ai.prompt.0.content": "hello",
        "gen_ai.completion.0.content": "world",
    }
    assert _preview_from_attrs(attrs, "input") == "hello"
    assert _preview_from_attrs(attrs, "output") == "world"


# ── Non-LLM operations carrying a model attribute ──────────────


def test_agent_spans_with_model_metadata_are_not_llm_calls():
    """AG2 stamps gen_ai.request.model onto conversation/invoke_agent spans;
    only the operation.name="chat" span is a real LLM request. The trace
    must carry exactly one LlmCallRecord, not one per stamped span."""
    tid = 0xC3
    spans = [
        _MockSpan(
            "conversation user", tid, 0x01,
            attributes={
                "gen_ai.operation.name": "conversation",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": 25,
                "gen_ai.usage.output_tokens": 1,
            },
        ),
        _MockSpan(
            "invoke_agent assistant", tid, 0x02, parent_span_id=0x01,
            attributes={
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.agent.name": "assistant",
            },
        ),
        _MockSpan(
            "chat gpt-4o-mini", tid, 0x03, parent_span_id=0x02,
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-4o-mini",
                "gen_ai.usage.input_tokens": 25,
                "gen_ai.usage.output_tokens": 1,
            },
        ),
    ]
    rt = _assemble(spans)
    assert len(rt.llm_calls) == 1, (
        f"expected 1 LlmCallRecord (the chat span), got {len(rt.llm_calls)} — "
        "agent/conversation spans with model metadata leaked into llm_calls"
    )
    assert rt.llm_calls[0].input_tokens == 25
    assert rt.llm_calls[0].output_tokens == 1
    span_types = {s.name: s.span_type.value for s in rt.spans}
    assert span_types["invoke_agent assistant"] == "agent"
    assert span_types["llm:gpt-4o-mini"] == "llm"
