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


# ── A turn that calls a tool records the call ───────────────
#
# A model turn whose output is a function call carries no text, and the handler
# kept only text: the llm_call that DECIDED to call `load_skill` was recorded as
# output=None, tool_calls=[] and finish_reason="stop" -- a trace in which the
# model apparently answered with nothing. And the tool span's preview stopped
# at 200 characters, inside the first sentence of a skill body, so the line a
# reader opens the trace to find was cut off.

#: A skill body whose policy line sits well past the old 200-character cut.
POLICY_LINE = "POLICY: refunds over $500 need a manager's approval."
SKILL_BODY = (
    "## Skill: refund-policy\n\n"
    + "Refunds are handled per the policy below. " * 10
    + "\n\n" + POLICY_LINE + "\n"
)


def _function_call(name, arguments, call_id="call_1"):
    """A real Responses API function-call output item."""
    from openai.types.responses import ResponseFunctionToolCall

    return ResponseFunctionToolCall(
        id=f"fc_{call_id}", call_id=call_id, type="function_call",
        name=name, arguments=arguments,
    )


def _function_span(trace_id, name, *, input, output, error=None, ms=40):
    started = datetime.now(timezone.utc)
    return _MockSpan(
        trace_id=trace_id,
        span_data=_MockSpanData("function", name=name, input=input, output=output),
        started_at=started.isoformat(),
        ended_at=(started + timedelta(milliseconds=ms)).isoformat(),
        error=error,
    )


def _tool_turn(trace_id, *items):
    return _MockSpan(
        trace_id=trace_id,
        span_data=_MockSpanData(
            "response",
            response=_SyntheticResponse(output=list(items)),
            input="how do refunds work?",
            usage=None,
        ),
    )


class TestToolCallTurn:
    def test_the_call_the_model_made_is_on_its_llm_call(self):
        from decimalai.openai_agents import DecimalTracingProcessor
        from decimalai.schema.common import FinishReason, Status

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"
        args = '{"name": "refund-policy"}'

        trace = _run_response_spans(
            processor, tid, [_tool_turn(tid, _function_call("load_skill", args, "call_abc"))],
        )

        (call,) = trace.llm_calls
        assert call.finish_reason == FinishReason.TOOL_CALLS
        (tc,) = call.tool_calls
        assert tc.tool_name == "load_skill"
        assert tc.args == {"name": "refund-policy"}
        # Requested and never seen to run: no result is claimed for it.
        assert tc.status == Status.RUNNING
        assert tc.result is None
        # The call id rides on the output, in the chat-completions shape the
        # trace view renders as a tool-call request.
        assert call.output == {
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "load_skill", "arguments": args},
            }],
        }

    def test_text_and_a_call_in_one_turn_keep_both(self):
        from decimalai.openai_agents import DecimalTracingProcessor
        from decimalai.schema.common import FinishReason

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"

        trace = _run_response_spans(processor, tid, [_tool_turn(
            tid,
            _OutputMessage(_OutputText("Let me check the policy.")),
            _function_call("load_skill", '{"name": "refund-policy"}'),
        )])

        (call,) = trace.llm_calls
        assert call.finish_reason == FinishReason.TOOL_CALLS
        assert call.output["content"] == "Let me check the policy."
        assert [t["function"]["name"] for t in call.output["tool_calls"]] == ["load_skill"]
        assert [t.tool_name for t in call.tool_calls] == ["load_skill"]

    def test_a_text_turn_is_unchanged(self):
        from decimalai.openai_agents import DecimalTracingProcessor
        from decimalai.schema.common import FinishReason

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"

        trace = _run_response_spans(
            processor, tid, [_tool_turn(tid, _OutputMessage(_OutputText("4")))],
        )

        (call,) = trace.llm_calls
        assert call.finish_reason == FinishReason.STOP
        assert call.tool_calls == []
        assert call.output == {"content": "4"}

    def test_the_tool_result_lands_on_the_call_that_asked_for_it(self):
        from decimalai.openai_agents import DecimalTracingProcessor
        from decimalai.schema.common import Status

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"
        args = '{"name": "refund-policy"}'

        trace = _run_response_spans(processor, tid, [
            _tool_turn(tid, _function_call("load_skill", args)),
            _function_span(tid, "load_skill", input=args, output=SKILL_BODY, ms=40),
        ])

        (tc,) = trace.llm_calls[0].tool_calls
        assert tc.status == Status.SUCCESS
        assert POLICY_LINE in tc.result
        assert tc.latency_ms == 40

    def test_parallel_calls_to_one_tool_are_told_apart_by_their_arguments(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"
        a, b = '{"order": "A-1"}', '{"order": "B-2"}'

        trace = _run_response_spans(processor, tid, [
            _tool_turn(tid, _function_call("lookup", a, "call_a"),
                       _function_call("lookup", b, "call_b")),
            # Finished out of order: B first.
            _function_span(tid, "lookup", input=b, output="order B-2: shipped"),
            _function_span(tid, "lookup", input=a, output="order A-1: refunded"),
        ])

        first, second = trace.llm_calls[0].tool_calls
        assert first.args == {"order": "A-1"} and first.result == "order A-1: refunded"
        assert second.args == {"order": "B-2"} and second.result == "order B-2: shipped"

    def test_a_tool_that_raised_is_recorded_as_an_error(self):
        from decimalai.openai_agents import DecimalTracingProcessor
        from decimalai.schema.common import Status

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"
        args = '{"order": "A-1"}'

        trace = _run_response_spans(processor, tid, [
            _tool_turn(tid, _function_call("lookup", args)),
            _function_span(tid, "lookup", input=args, output=None,
                           error={"message": "Error running tool", "data": {}}),
        ])

        (tc,) = trace.llm_calls[0].tool_calls
        assert tc.status == Status.ERROR

    def test_the_chat_completions_path_records_its_calls_too(self):
        """`OpenAIChatCompletionsModel` (and LiteLLM) emit `generation` spans,
        whose output is the chat message — tool calls included — but the
        record carried none of them and said "stop"."""
        from decimalai.openai_agents import DecimalTracingProcessor
        from decimalai.schema.common import FinishReason, Status

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"
        args = '{"name": "refund-policy"}'
        message = {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_chat_1", "type": "function",
                            "function": {"name": "load_skill", "arguments": args}}],
        }

        trace = _run_response_spans(processor, tid, [
            _MockSpan(trace_id=tid, span_data=_MockSpanData(
                "generation", model="gpt-4o", model_config={}, usage={},
                input=[{"role": "user", "content": "how do refunds work?"}],
                output=[message],
            )),
            _function_span(tid, "load_skill", input=args, output=SKILL_BODY),
        ])

        (call,) = trace.llm_calls
        assert call.finish_reason == FinishReason.TOOL_CALLS
        (tc,) = call.tool_calls
        assert (tc.tool_name, tc.args, tc.status) == ("load_skill", {"name": "refund-policy"},
                                                      Status.SUCCESS)
        assert POLICY_LINE in tc.result

    def test_arguments_that_are_not_json_are_kept_verbatim(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"

        trace = _run_response_spans(
            processor, tid, [_tool_turn(tid, _function_call("lookup", "not json"))],
        )

        (tc,) = trace.llm_calls[0].tool_calls
        assert tc.args == {"raw": "not json"}


class TestToolSpanPreview:
    def test_a_skill_body_keeps_its_first_paragraphs(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"
        args = '{"name": "refund-policy"}'
        assert SKILL_BODY.index(POLICY_LINE) > 200  # the line the old cut hid

        trace = _run_response_spans(processor, tid, [
            _function_span(tid, "load_skill", input=args, output=SKILL_BODY),
        ])

        (span,) = [s for s in trace.spans if s.span_type.value == "tool"]
        assert POLICY_LINE in span.output_preview
        assert span.input_preview == args

    def test_the_preview_is_still_bounded(self):
        """2000 characters: the length the platform keeps of a span preview."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test-agent")
        tid = f"trace_{uuid4().hex[:16]}"

        trace = _run_response_spans(processor, tid, [
            _function_span(tid, "load_skill", input="x" * 5000, output="y" * 5000),
        ])

        (span,) = [s for s in trace.spans if s.span_type.value == "tool"]
        assert len(span.output_preview) == 2000
        assert len(span.input_preview) == 2000


# ── The same, through the real SDK ──────────────────────────
#
# The real Runner and the SDK's default model class, OpenAIResponsesModel, so
# the spans are the SDK's own ResponseSpanData / FunctionSpanData rather than
# stand-ins. The model is a stub answering over an in-process httpx transport:
# nothing leaves the process.


def _responses_json(body):
    """One Responses API reply: call the tool first, answer once it has run."""
    import json

    answered = any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in body.get("input") or []
    )
    if answered:
        return _responses_payload([_wire_text("Over $500 needs a manager.")])
    return _responses_payload([
        _wire_call("call_policy_1", "lookup_policy", json.dumps({"topic": "refunds"})),
    ])


def _wire_text(text):
    return {
        "id": f"msg_{uuid4().hex[:8]}", "type": "message", "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _wire_call(call_id, name, arguments):
    return {
        "id": f"fc_{call_id}", "type": "function_call", "call_id": call_id,
        "name": name, "arguments": arguments, "status": "completed",
    }


def _responses_payload(output):
    import time

    return {
        "id": f"resp_{uuid4().hex}", "object": "response", "created_at": time.time(),
        "model": "gpt-4o-mini", "status": "completed", "output": output,
        "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
        "usage": {
            "input_tokens": 12, "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 17,
        },
    }


@pytest.fixture
def real_runner():
    """Our processor as the SDK's only trace processor, restored afterwards."""
    pytest.importorskip("agents")
    from agents.tracing import get_trace_provider, set_trace_processors

    from decimalai.openai_agents import DecimalTracingProcessor

    provider = get_trace_provider()
    saved = list(provider._multi_processor._processors)
    set_trace_processors([DecimalTracingProcessor(agent_name="policy-agent")])
    try:
        yield
    finally:
        set_trace_processors(saved)


def test_a_real_run_records_the_tool_decision_and_its_result(real_runner):
    import json

    import httpx
    from agents import Agent, OpenAIResponsesModel, Runner, function_tool
    from openai import AsyncOpenAI

    import decimalai._config as cfg
    from decimalai._config import _sender
    from decimalai.schema.common import FinishReason, Status

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_responses_json(json.loads(request.content)))

    client = AsyncOpenAI(
        api_key="sk-test", base_url="http://model.invalid/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    @function_tool
    def lookup_policy(topic: str) -> str:
        """Return the policy for a topic."""
        return SKILL_BODY

    agent = Agent(
        name="policy-agent", instructions="Answer from the policy.",
        model=OpenAIResponsesModel(model="gpt-4o-mini", openai_client=client),
        tools=[lookup_policy],
    )
    Runner.run_sync(agent, "how do refunds work?")
    _sender.flush()

    (trace,) = [c.args[0] for c in cfg._client.ingest_trace.call_args_list]
    decided, answered = trace.llm_calls
    assert decided.finish_reason == FinishReason.TOOL_CALLS
    (tc,) = decided.tool_calls
    assert (tc.tool_name, tc.args, tc.status) == ("lookup_policy", {"topic": "refunds"},
                                                  Status.SUCCESS)
    assert POLICY_LINE in tc.result
    assert decided.output["tool_calls"][0]["id"] == "call_policy_1"
    assert answered.finish_reason == FinishReason.STOP
    assert answered.tool_calls == []
    (span,) = [s for s in trace.spans if s.span_type.value == "tool"]
    assert POLICY_LINE in span.output_preview


# ── One call, one record: tool calls join their spans ───────
#
# The backend reads a trace's tool calls from BOTH `llm_calls[].tool_calls[]`
# and the tool spans, and counts entries that share an id as ONE call. So a
# record whose id is not its span's id is counted twice, and a handoff -- a
# function call the model makes, run by a handoff span rather than a function
# span -- stayed `running` forever. One real tool call plus one handoff read as
# "1 of 3 tool calls did not return".


def _joined_tool_calls(payload):
    """The backend's reading: records + tool spans, joined on id."""
    calls, by_id = [], {}
    for entry in [tc for c in payload.get("llm_calls") or [] for tc in c.get("tool_calls") or []] \
            + [s for s in payload.get("spans") or [] if s.get("span_type") in ("tool", "tool_call")]:
        ids = {str(entry[k]) for k in ("id", "span_id") if entry.get(k)}
        index = next((by_id[i] for i in ids if i in by_id), None)
        if index is None:
            index = len(calls)
            calls.append(dict(entry))
        else:
            calls[index].update({k: v for k, v in entry.items()
                                 if v is not None or k not in ("result", "output_preview")})
        by_id.update({i: index for i in ids})
    return calls


def _returned(call):
    return call.get("result") is not None or bool(call.get("output_preview")) \
        or call.get("status") == "success"


def _handoff_span(trace_id, from_agent, to_agent):
    return _MockSpan(
        trace_id=trace_id,
        span_data=_MockSpanData("handoff", from_agent=from_agent, to_agent=to_agent),
    )


class TestToolCallsJoinTheirSpans:
    def _run(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="support")
        tid = f"trace_{uuid4().hex[:16]}"
        args = '{"order": "A-1"}'
        function = _function_span(tid, "lookup_order", input=args, output="order A-1: refunded")
        handoff = _handoff_span(tid, "support", "Billing")
        trace = _run_response_spans(processor, tid, [
            _tool_turn(tid, _function_call("lookup_order", args, "call_lookup"),
                       _function_call("transfer_to_billing", "{}", "call_handoff")),
            function,
            handoff,
        ])
        return trace, function, handoff

    def test_a_tool_call_and_a_handoff_are_two_records_both_done(self):
        from decimalai.schema.common import Status

        trace, function, handoff = self._run()

        lookup, transfer = trace.llm_calls[0].tool_calls
        assert (lookup.tool_name, lookup.status) == ("lookup_order", Status.SUCCESS)
        assert (transfer.tool_name, transfer.status) == ("transfer_to_billing", Status.SUCCESS)
        assert "Billing" in transfer.result
        # Each record IS its span: same id.
        assert str(lookup.id) == function.span_id
        assert str(transfer.id) == handoff.span_id

    def test_the_payload_joins_records_and_spans_one_to_one(self):
        trace, _, _ = self._run()
        payload = trace.model_dump(mode="json")

        joined = _joined_tool_calls(payload)
        assert [c.get("tool_name") or c.get("name") for c in joined] == [
            "lookup_order", "transfer_to_billing",
        ]
        assert all(_returned(c) for c in joined)
        record_ids = {tc["id"] for c in payload["llm_calls"] for tc in c["tool_calls"]}
        tool_span_ids = {s["id"] for s in payload["spans"] if s["span_type"] == "tool"}
        assert tool_span_ids <= record_ids


def test_a_real_run_with_a_tool_and_a_handoff_joins_one_to_one(real_runner):
    """The same through the real Runner: support looks the order up, hands off
    to Billing, and Billing answers."""
    import json

    import httpx
    from agents import Agent, OpenAIResponsesModel, Runner, function_tool
    from openai import AsyncOpenAI

    import decimalai._config as cfg
    from decimalai._config import _sender
    from decimalai.schema.common import Status

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        outputs = [i.get("call_id") for i in body.get("input") or []
                   if isinstance(i, dict) and i.get("type") == "function_call_output"]
        if "You are billing." in (body.get("instructions") or ""):
            out = [_wire_text("Refund issued.")]
        elif "call_lookup" in outputs:
            out = [_wire_call("call_handoff", "transfer_to_billing", "{}")]
        else:
            out = [_wire_call("call_lookup", "lookup_order", json.dumps({"order": "A-1"}))]
        return httpx.Response(200, json=_responses_payload(out))

    client = AsyncOpenAI(
        api_key="sk-test", base_url="http://model.invalid/v1", max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    model = OpenAIResponsesModel(model="gpt-4o-mini", openai_client=client)

    @function_tool
    def lookup_order(order: str) -> str:
        """Look an order up."""
        return f"order {order}: refund requested"

    billing = Agent(name="Billing", instructions="You are billing.", model=model)
    support = Agent(name="support", instructions="You are support.", model=model,
                    tools=[lookup_order], handoffs=[billing])
    Runner.run_sync(support, "refund order A-1")
    _sender.flush()

    (trace,) = [c.args[0] for c in cfg._client.ingest_trace.call_args_list]
    records = [tc for c in trace.llm_calls for tc in c.tool_calls]
    assert [(r.tool_name, r.status) for r in records] == [
        ("lookup_order", Status.SUCCESS), ("transfer_to_billing", Status.SUCCESS),
    ]
    (tool_span,) = [s for s in trace.spans if s.span_type.value == "tool"]
    (handoff_span,) = [s for s in trace.spans if s.name.startswith("handoff:")]
    assert records[0].id == tool_span.id
    assert records[1].id == handoff_span.id
    assert "Billing" in records[1].result

    joined = _joined_tool_calls(trace.model_dump(mode="json"))
    assert len(joined) == 2 and all(_returned(c) for c in joined)
