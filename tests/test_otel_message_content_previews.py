"""Lock in: the OTel rail reads message CONTENT, not the role, and fills
``LlmCallRecord.rendered_input`` / ``.output``.

Post-deploy re-verification (2026-08-15) found three silent quality losses on
traces that ingest fine:

1. LLM span previews rendered the message ROLE. OpenInference nests role and
   content under indexed keys (``llm.input_messages.0.message.role`` /
   ``.content``) and the role key is inserted first, so the substring scan in
   ``_preview_from_attrs`` returned ``"system"`` / ``"assistant"`` instead of
   the prompt and the completion (CrewAI traces 15a90cb3, ef8f74d7).
2. ``rendered_input`` and ``output`` were ALWAYS null across the whole rail —
   ``_make_llm_call`` never set either field (22b8fc43, ef8f74d7, 36dc593c).
3. AG2 tool spans dropped their arguments and results: AG2 emits
   ``gen_ai.tool.call.arguments`` / ``gen_ai.tool.call.result``, neither of
   which contains the substring the preview scan looked for.

Attribute sets below are copied verbatim from live runs (CrewAI 1.15.16 +
openinference-instrumentation-crewai 1.1.12; AG2 1.0.1 / autogen 0.14.1).

No backend and no real OTEL — spans and the client are mocked.
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


# Verbatim from a live CrewAI run: the first LLM step, whose completion is a
# tool call (no output content), and the second, which answers in text.
_CREWAI_TOOLCALL_LLM_ATTRS = {
    "llm.provider": "openai",
    "llm.system": "openai",
    "input.value": '{"messages": [{"role": "system", "content": "You are City Analyst."}]}',
    "input.mime_type": "application/json",
    "output.value": '{"id":"chatcmpl-ECy","choices":[{"finish_reason":"tool_calls"}]}',
    "output.mime_type": "application/json",
    "llm.input_messages.0.message.role": "system",
    "llm.input_messages.0.message.content": "You are City Analyst. You are terse.",
    "llm.input_messages.1.message.role": "user",
    "llm.input_messages.1.message.content": "Use the population_lookup tool for Paris.",
    "llm.model_name": "gpt-4o-mini-2024-07-18",
    "llm.output_messages.0.message.role": "assistant",
    "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_8OuWK9",
    "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "population_lookup",
    "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"city":"Paris"}',
    "llm.token_count.prompt": 116,
    "llm.token_count.completion": 14,
}

_CREWAI_ANSWER_LLM_ATTRS = {
    "llm.provider": "openai",
    "llm.system": "openai",
    "llm.input_messages.0.message.role": "system",
    "llm.input_messages.0.message.content": "You are City Analyst. You are terse.",
    "llm.input_messages.1.message.role": "user",
    "llm.input_messages.1.message.content": "Use the population_lookup tool for Paris.",
    "llm.input_messages.2.message.role": "assistant",
    "llm.input_messages.2.message.tool_calls.0.tool_call.function.name": "population_lookup",
    "llm.input_messages.3.message.role": "tool",
    "llm.input_messages.3.message.content": "Paris has a population of 2,148,000.",
    "llm.model_name": "gpt-4o-mini-2024-07-18",
    "llm.output_messages.0.message.role": "assistant",
    "llm.output_messages.0.message.content": "Paris has a population of 2,148,000.",
    "llm.token_count.prompt": 150,
    "llm.token_count.completion": 13,
}

# Verbatim from a live AG2 run (`execute_tool population_lookup` span).
_AG2_TOOL_ATTRS = {
    "ag2.span.type": "tool",
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.call.arguments": '{"city":"Paris"}',
    "gen_ai.tool.call.id": "call_aWTy2i0LZKdyVE1TAxCFrDks",
    "gen_ai.tool.call.result": "Paris has a population of 2,148,000.",
    "gen_ai.tool.name": "population_lookup",
    "gen_ai.tool.type": "function",
}


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


def _span_named(run_trace, name):
    return next(s for s in run_trace.spans if s.name == name)


# ── 1. previews carry content, never the role ──────────────────


def test_llm_previews_use_message_content_not_role():
    """The load-bearing regression: OpenInference indexed messages must
    render the prompt/completion text, not "system"/"assistant"."""
    rt = _assemble([
        _MockSpan("root", 0xC1, 0x01),
        _MockSpan("ChatCompletion", 0xC1, 0x02, parent_span_id=0x01,
                  attributes=_CREWAI_ANSWER_LLM_ATTRS),
    ])
    span = _span_named(rt, "llm:gpt-4o-mini-2024-07-18")
    assert span.input_preview != "system"
    assert span.output_preview != "assistant"
    assert "You are City Analyst" in span.input_preview
    assert span.output_preview == "Paris has a population of 2,148,000."


def test_input_preview_keeps_message_order():
    """Multi-turn input: contents are concatenated in index order, so the
    system prompt precedes the user turn."""
    rt = _assemble([
        _MockSpan("root", 0xC2, 0x01),
        _MockSpan("ChatCompletion", 0xC2, 0x02, parent_span_id=0x01,
                  attributes=_CREWAI_ANSWER_LLM_ATTRS),
    ])
    preview = _span_named(rt, "llm:gpt-4o-mini-2024-07-18").input_preview
    assert preview.index("You are City Analyst") < preview.index(
        "Use the population_lookup tool"
    )


def test_toolcall_only_output_never_falls_back_to_the_role():
    """When the completion is a tool call there is no output content — the
    preview must fall through to the raw response, never to "assistant"."""
    rt = _assemble([
        _MockSpan("root", 0xC3, 0x01),
        _MockSpan("ChatCompletion", 0xC3, 0x02, parent_span_id=0x01,
                  attributes=_CREWAI_TOOLCALL_LLM_ATTRS),
    ])
    span = _span_named(rt, "llm:gpt-4o-mini-2024-07-18")
    assert span.output_preview != "assistant"
    assert "tool_calls" in span.output_preview


def test_trace_previews_inherit_the_content_fix():
    """Trace-level previews fall back to the LLM spans, so they carry the
    same content rather than "system"/"assistant"."""
    rt = _assemble([
        _MockSpan("root", 0xC4, 0x01),
        _MockSpan("ChatCompletion", 0xC4, 0x02, parent_span_id=0x01,
                  attributes=_CREWAI_ANSWER_LLM_ATTRS),
    ])
    assert "You are City Analyst" in rt.user_input_preview
    assert rt.final_output_preview == "Paris has a population of 2,148,000."


# ── 2. rendered_input / output populated ───────────────────────


def test_rendered_input_and_output_populated_from_openinference_messages():
    """LlmCallRecord carries the rendered request and response — the fields
    SFT derivation reads — normalized to role/content dicts."""
    rt = _assemble([
        _MockSpan("root", 0xC5, 0x01),
        _MockSpan("ChatCompletion", 0xC5, 0x02, parent_span_id=0x01,
                  attributes=_CREWAI_ANSWER_LLM_ATTRS),
    ])
    call = rt.llm_calls[0]
    assert call.rendered_input == [
        {"role": "system", "content": "You are City Analyst. You are terse."},
        {"role": "user", "content": "Use the population_lookup tool for Paris."},
        # The assistant turn made a tool call and carries no content; it stays
        # in the transcript so turn order survives.
        {"role": "assistant", "content": ""},
        {"role": "tool", "content": "Paris has a population of 2,148,000."},
    ]
    assert call.output == {
        "role": "assistant",
        "content": "Paris has a population of 2,148,000.",
    }


def test_rendered_input_and_output_populated_from_plain_genai_keys():
    """The generic-OTel dialect (a bare prompt/completion string) also fills
    the fields, wrapped as a single message."""
    rt = _assemble([
        _MockSpan("root", 0xC6, 0x01),
        _MockSpan("chat", 0xC6, 0x02, parent_span_id=0x01, attributes={
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.input": "What is the capital of France?",
            "gen_ai.output": "Paris.",
        }),
    ])
    call = rt.llm_calls[0]
    assert call.rendered_input == [
        {"role": "user", "content": "What is the capital of France?"}
    ]
    assert call.output == {"role": "assistant", "content": "Paris."}


def test_rendered_input_is_not_truncated_to_the_preview_length():
    """The preview is capped at 200 chars; the rendered request must keep
    the full prompt — it is the SFT artifact, not a display string."""
    long_prompt = "x" * 900
    rt = _assemble([
        _MockSpan("root", 0xC7, 0x01),
        _MockSpan("ChatCompletion", 0xC7, 0x02, parent_span_id=0x01, attributes={
            "llm.model_name": "gpt-4o-mini",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": long_prompt,
        }),
    ])
    assert rt.llm_calls[0].rendered_input[0]["content"] == long_prompt
    assert len(_span_named(rt, "llm:gpt-4o-mini").input_preview) == 200


def test_multi_part_message_content_is_joined_in_part_order():
    """OpenInference splits a multi-modal message's text across
    ``…contents.{j}.message_content.text``; the parts join in order."""
    rt = _assemble([
        _MockSpan("root", 0xCC, 0x01),
        _MockSpan("ChatCompletion", 0xCC, 0x02, parent_span_id=0x01, attributes={
            "llm.model_name": "gpt-4o-mini",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.contents.0.message_content.type": "text",
            "llm.input_messages.0.message.contents.0.message_content.text": "describe ",
            "llm.input_messages.0.message.contents.1.message_content.type": "text",
            "llm.input_messages.0.message.contents.1.message_content.text": "this image",
        }),
    ])
    assert rt.llm_calls[0].rendered_input == [
        {"role": "user", "content": "describe this image"}
    ]
    assert _span_named(rt, "llm:gpt-4o-mini").input_preview == "describe this image"


def test_no_message_attributes_leaves_the_fields_none():
    """A span with no content anywhere → both fields stay None (never an
    empty shell that looks like captured data)."""
    rt = _assemble([
        _MockSpan("root", 0xC8, 0x01),
        _MockSpan("chat", 0xC8, 0x02, parent_span_id=0x01, attributes={
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 42,
            "gen_ai.usage.output_tokens": 7,
        }),
    ])
    call = rt.llm_calls[0]
    assert call.rendered_input is None
    assert call.output is None


# ── 3. AG2 tool spans keep arguments and result ────────────────


def test_ag2_tool_span_carries_arguments_and_result():
    """AG2 stamps gen_ai.tool.call.arguments / .result; neither contains the
    substring the preview scan looked for, so both were dropped."""
    rt = _assemble([
        _MockSpan("conversation user", 0xC9, 0x01),
        _MockSpan("execute_tool population_lookup", 0xC9, 0x02,
                  parent_span_id=0x01, attributes=_AG2_TOOL_ATTRS),
    ])
    span = _span_named(rt, "execute_tool population_lookup")
    assert span.span_type.value == "tool"
    assert span.input_preview == '{"city":"Paris"}'
    assert span.output_preview == "Paris has a population of 2,148,000."


# ── the older integrations.otel exporter, same three defects ───


def _assemble_legacy(spans):
    from decimalai.integrations.otel import DecimalSpanExporter

    client = MagicMock()
    with patch.dict("sys.modules", {
        "opentelemetry": MagicMock(),
        "opentelemetry.sdk": MagicMock(),
        "opentelemetry.sdk.trace": MagicMock(),
        "opentelemetry.sdk.trace.export": MagicMock(SpanExportResult=MagicMock()),
    }):
        import decimalai._config as cfg

        exporter = DecimalSpanExporter(agent_name="test-agent")
        with patch.object(cfg._sender, "submit") as submit:
            exporter._export_trace("t", spans, client)
    return submit.call_args[0][1]


def test_legacy_exporter_also_reads_content_and_fills_rendered_input():
    # This exporter recognises an LLM span only by the GenAI-semconv model
    # attribute, so the fixture carries it alongside the OpenInference messages.
    attrs = dict(_CREWAI_ANSWER_LLM_ATTRS)
    attrs["gen_ai.request.model"] = "gpt-4o-mini-2024-07-18"
    trace = _assemble_legacy([
        _MockSpan("ChatCompletion", 0xCA, 0x02, attributes=attrs),
    ])
    assert "You are City Analyst" in trace.user_input_preview
    assert trace.final_output_preview == "Paris has a population of 2,148,000."
    call = trace.llm_calls[0]
    assert call.rendered_input[0] == {
        "role": "system", "content": "You are City Analyst. You are terse."
    }
    assert call.output == {
        "role": "assistant",
        "content": "Paris has a population of 2,148,000.",
    }


def test_legacy_exporter_tool_span_carries_arguments_and_result():
    trace = _assemble_legacy([
        _MockSpan("execute_tool population_lookup", 0xCB, 0x02,
                  attributes=_AG2_TOOL_ATTRS),
    ])
    span = trace.spans[0]
    assert span.input_preview == '{"city":"Paris"}'
    assert span.output_preview == "Paris has a population of 2,148,000."
