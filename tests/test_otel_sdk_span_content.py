"""A provider SDK's own span inside an instrumentor's: the call keeps the instrumentor's content.

anthropic >= 1.15 traces its own API calls. Under ``decimalai.init(anthropic=True)`` each
``messages.create()`` then arrives as two spans: the SDK's ``anthropic.messages.create``
(model, tokens, HTTP; no messages, because its ``content_mode`` defaults to
``metadata_only``) inside OpenInference's ``messages.create`` (the prompt and the
completion). The exporter keeps the innermost LLM span as the call, so that a stacked
instrumentor pair counts once, and so it kept the one with no content: the Claude prod
prober's raw-Anthropic traces (2026-10-20) had no previews and an llm_call with no
rendered input or output.

Attribute sets below are copied verbatim from a run of the prober's snippet with
anthropic 1.15.0, openinference-instrumentation-anthropic 3.0.3 and decimalai 0.21.0 (the
prober's pins), against a local Messages stub. ``tests/conformance/drivers/anthropic.py``
runs the real SDK the same way when the installed anthropic traces itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

TRACE = 0x5EED
ROOT, OPENINFERENCE, SDK = 0x914A763E52AE09DB, 0xFD506BFAB1498A2D, 0x2138FF82D54B2149

#: The run boundary, ``providers.agent_run()``.
RUN_ATTRS = {
    "decimal.agent_name": "claude-probe-messages",
    "gen_ai.agent.name": "claude-probe-messages",
    "gen_ai.operation.name": "invoke_agent",
}
#: OpenInference's span: the request and the response, in full.
OPENINFERENCE_ATTRS = {
    "decimal.agent_name": "claude-probe-messages",
    "input.mime_type": "application/json",
    "input.value": ('{"model": "claude-haiku-4-5", "max_tokens": 16, "system": "You are a '
                    'health check.", "messages": [{"role": "user", "content": "[probe '
                    'PROMPT-SENTINEL] Reply: pong"}]}'),
    "llm.finish_reason": "end_turn",
    "llm.input_messages.0.message.content": "You are a health check.",
    "llm.input_messages.0.message.role": "system",
    "llm.input_messages.1.message.content": "[probe PROMPT-SENTINEL] Reply: pong",
    "llm.input_messages.1.message.role": "user",
    "llm.invocation_parameters": '{"max_tokens": 16}',
    "llm.model_name": "claude-haiku-4-5",
    "llm.output_messages.0.message.contents.0.message_content.text": "pong-COMPLETION-SENTINEL",
    "llm.output_messages.0.message.contents.0.message_content.type": "text",
    "llm.output_messages.0.message.role": "assistant",
    "llm.provider": "anthropic",
    "llm.request.model_name": "claude-haiku-4-5",
    "llm.response.model_name": "claude-haiku-4-5",
    "llm.system": "anthropic",
    "llm.token_count.completion": 3,
    "llm.token_count.prompt": 12,
    "llm.token_count.total": 15,
    "openinference.span.kind": "LLM",
    "output.mime_type": "application/json",
    "output.value": ('{"id":"msg_1","content":[{"citations":null,"text":'
                     '"pong-COMPLETION-SENTINEL","type":"text"}],"model":"claude-haiku-4-5",'
                     '"role":"assistant","stop_reason":"end_turn","type":"message","usage":'
                     '{"input_tokens":12,"output_tokens":3}}'),
}
#: The SDK's own span: the call as it went over the wire, without its content.
SDK_ATTRS = {
    "anthropic.message.stop_reason": "end_turn",
    "decimal.agent_name": "claude-probe-messages",
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "anthropic",
    "gen_ai.request.max_tokens": 16,
    "gen_ai.request.model": "claude-haiku-4-5",
    "gen_ai.response.id": "msg_1",
    "gen_ai.response.model": "claude-haiku-4-5",
    "gen_ai.usage.input_tokens": 12,
    "gen_ai.usage.output_tokens": 3,
    "http.request.method": "POST",
    "http.response.status_code": 200,
    "http.route": "/v1/messages",
    "server.address": "127.0.0.1",
    "server.port": 50863,
    "url.full": "http://127.0.0.1:50863/v1/messages",
    "url.path": "/v1/messages",
    "url.scheme": "http",
}


class _Context:
    def __init__(self, span_id):
        self.trace_id, self.span_id, self.trace_flags = TRACE, span_id, 1


class _Status:
    status_code = "OK"


class _Resource:
    attributes = {"service.name": "test-service"}


class _Span:
    def __init__(self, name, span_id, parent_id, attributes):
        self.name = name
        self.context = _Context(span_id)
        self.parent = _Context(parent_id) if parent_id else None
        self.attributes = dict(attributes)
        self.start_time = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.end_time = self.start_time + 100_000_000
        self.status = _Status()
        self.resource = _Resource()


@pytest.fixture(autouse=True)
def _config():
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(api_key="dai_sk_test", base_url="http://localhost:8000",
                                enabled=True)
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "m", "status": "active"}
    yield


def _assemble(spans):
    from decimalai.otel import DecimalSpanExporter

    result = DecimalSpanExporter(agent_name="test-agent")._assemble_trace(spans)
    assert result is not None
    return result


def _probe_spans(sdk_attrs=SDK_ATTRS, openinference_attrs=OPENINFERENCE_ATTRS):
    return [_Span("anthropic.messages.create", SDK, OPENINFERENCE, sdk_attrs),
            _Span("messages.create", OPENINFERENCE, ROOT, openinference_attrs),
            _Span("agent.run", ROOT, None, RUN_ATTRS)]


def test_the_call_takes_its_content_from_the_instrumentor_span_around_it():
    trace, _model, _tools, prompts = _assemble(_probe_spans())
    (call,) = trace.llm_calls  # one request, not two
    assert call.rendered_input == [
        {"role": "system", "content": "You are a health check."},
        {"role": "user", "content": "[probe PROMPT-SENTINEL] Reply: pong"},
    ]
    assert call.output == {"role": "assistant", "content": "pong-COMPLETION-SENTINEL"}
    assert (call.input_tokens, call.output_tokens) == (12, 3)
    assert trace.user_input_preview == "[probe PROMPT-SENTINEL] Reply: pong"
    assert trace.final_output_preview == "pong-COMPLETION-SENTINEL"
    (llm_span,) = [s for s in trace.spans if s.name == "llm:claude-haiku-4-5"]
    assert "PROMPT-SENTINEL" in llm_span.input_preview
    assert llm_span.output_preview == "pong-COMPLETION-SENTINEL"
    # The manifest's system prompt is read off the call, so it was lost the same way.
    assert prompts.get("system") == "You are a health check."


def test_a_call_with_content_of_its_own_keeps_it():
    # The LiteLLM -> openai pair: both spans carry content, and the call's is the request
    # that went over the wire.
    own = dict(SDK_ATTRS, **{
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "the request as sent",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "the reply as received",
    })
    trace, *_ = _assemble(_probe_spans(sdk_attrs=own))
    (call,) = trace.llm_calls
    assert call.rendered_input == [{"role": "user", "content": "the request as sent"}]
    assert call.output == {"role": "assistant", "content": "the reply as received"}


def test_a_wrapper_around_several_calls_lends_its_content_to_none():
    second = 0x2138FF82D54B2150
    spans = _probe_spans() + [
        _Span("anthropic.messages.create", second, OPENINFERENCE, SDK_ATTRS)]
    trace, *_ = _assemble(spans)
    assert len(trace.llm_calls) == 2
    assert all(c.rendered_input is None and c.output is None for c in trace.llm_calls)
