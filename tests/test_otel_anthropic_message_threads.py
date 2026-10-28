"""anthropic >= 2's message threads: a later turn's call is recorded with the thread so far.

anthropic 2.x's Tool Runner sends each turn after the first on a message thread the API
keeps (``thread: {"type": "continue", "previous_message_id": ...}``, beta
``message-threads-2026-08-12``), with only what is new: a tool result, no ``system``, no
``tools``. The model is given the whole conversation; the request carries the new part. A
trace that records the request as sent shows no prompt and no history for any call after
the first, which is what the prod prober's latest-releases channel found on anthropic 2.3.0
(2026-10-26, ``tool-runner`` failing ``prompt_in_input``).

Attribute sets below are copied from a Tool Runner run on anthropic 2.3.0 and
openinference-instrumentation-anthropic 3.1.0 against a local Messages stub: two turns (a
tool call, then the answer), once on a thread and once with ``thread=False``. A live run on
claude-haiku-4-5 (2026-10-28) sent the same two requests.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import decimalai.otel as otel

SYSTEM = "You are a terse order-support agent."
QUESTION = "Where is order A-17?"
THREADS_BETA = ["message-threads-2026-08-12"]

OPENINFERENCE = "openinference.instrumentation.anthropic"
SDK = "com.anthropic.sdk.python"

#: The model's first reply: a sentence, then a tool call.
_ASKS_TOOL = {
    "llm.output_messages.0.message.contents.0.message_content.text": "Let me check.",
    "llm.output_messages.0.message.contents.0.message_content.type": "text",
    "llm.output_messages.0.message.contents.1.message_content.type": "tool_use",
    "llm.output_messages.0.message.contents.1.tool_call.function.arguments": '{"q": "A-17"}',
    "llm.output_messages.0.message.contents.1.tool_call.function.name": "lookup",
    "llm.output_messages.0.message.contents.1.tool_call.id": "toolu_1",
    "llm.output_messages.0.message.role": "assistant",
    "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"q": "A-17"}',
    "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup",
    "llm.output_messages.0.message.tool_calls.0.tool_call.id": "toolu_1",
}
#: The model's answer.
_ANSWERS = {
    "llm.output_messages.0.message.contents.0.message_content.text": "Shipped.",
    "llm.output_messages.0.message.contents.0.message_content.type": "text",
    "llm.output_messages.0.message.role": "assistant",
}
#: The first request: the whole of it.
_SYSTEM_AND_QUESTION = {
    "llm.input_messages.0.message.content": SYSTEM,
    "llm.input_messages.0.message.role": "system",
    "llm.input_messages.1.message.content": QUESTION,
    "llm.input_messages.1.message.role": "user",
}
#: A later request on the thread: the tool's result alone.
_TOOL_RESULT = {
    "llm.input_messages.0.message.content": "shipped",
    "llm.input_messages.0.message.role": "tool",
    "llm.input_messages.0.message.tool_call_id": "toolu_1",
}
#: The same later request with ``thread=False``: the whole conversation.
_WHOLE_SECOND_REQUEST = {
    **_SYSTEM_AND_QUESTION,
    "llm.input_messages.2.message.contents.0.message_content.text": "Let me check.",
    "llm.input_messages.2.message.contents.0.message_content.type": "text",
    "llm.input_messages.2.message.contents.1.message_content.type": "tool_use",
    "llm.input_messages.2.message.contents.1.tool_call.function.arguments": '{"q": "A-17"}',
    "llm.input_messages.2.message.contents.1.tool_call.function.name": "lookup",
    "llm.input_messages.2.message.contents.1.tool_call.id": "toolu_1",
    "llm.input_messages.2.message.role": "assistant",
    "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments": '{"q": "A-17"}',
    "llm.input_messages.2.message.tool_calls.0.tool_call.function.name": "lookup",
    "llm.input_messages.2.message.tool_calls.0.tool_call.id": "toolu_1",
    "llm.input_messages.3.message.content": "shipped",
    "llm.input_messages.3.message.role": "tool",
    "llm.input_messages.3.message.tool_call_id": "toolu_1",
}


def _invocation(thread):
    params = {"max_tokens": 50}
    if thread is not None:
        params.update(betas=THREADS_BETA, thread=thread)
    return json.dumps(params)


def _openinference(reply_id, request, reply, thread):
    """OpenInference's span of one ``beta.messages.create``: the request and the response."""
    return {
        **request, **reply,
        "llm.invocation_parameters": _invocation(thread),
        "llm.model_name": "claude-haiku-4-5",
        "llm.provider": "anthropic",
        "llm.system": "anthropic",
        "llm.token_count.completion": 3,
        "llm.token_count.prompt": 5,
        "openinference.span.kind": "LLM",
        "output.mime_type": "application/json",
        "output.value": json.dumps({"id": reply_id, "type": "message", "role": "assistant",
                                    "model": "claude-haiku-4-5", "content": []}),
    }


def _sdk(reply_id):
    """The SDK's own span of the same call: the wire, no content (``metadata_only``)."""
    return {
        "anthropic.beta": tuple(THREADS_BETA),
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.max_tokens": 50,
        "gen_ai.request.model": "claude-haiku-4-5",
        "gen_ai.response.id": reply_id,
        "gen_ai.response.model": "claude-haiku-4-5",
        "gen_ai.usage.input_tokens": 5,
        "gen_ai.usage.output_tokens": 3,
    }


class _Context:
    def __init__(self, trace_id, span_id):
        self.trace_id, self.span_id, self.trace_flags = trace_id, span_id, 1


class _Status:
    status_code = "OK"


class _Resource:
    attributes = {"service.name": "test-service"}


class _Scope:
    def __init__(self, name):
        self.name = name


class _Span:
    def __init__(self, trace_id, name, span_id, parent_id, attributes, scope=""):
        self.name = name
        self.context = _Context(trace_id, span_id)
        self.parent = _Context(trace_id, parent_id) if parent_id else None
        self.attributes = dict(attributes)
        self.instrumentation_scope = _Scope(scope)
        self.start_time = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.end_time = self.start_time + 100_000_000
        self.status = _Status()
        self.resource = _Resource()


ROOT = 0x1


def _turn(trace_id, n, reply_id, request, reply, thread, *, sdk_span=True):
    """One model call of a run, as its spans end: the SDK's inside OpenInference's."""
    wrapper = _Span(trace_id, "beta.messages.create", 0x100 + n, ROOT,
                    _openinference(reply_id, request, reply, thread), OPENINFERENCE)
    if not sdk_span:  # anthropic < 1.14, or the SDK's tracing off
        return [wrapper]
    return [_Span(trace_id, "anthropic.messages.create", 0x200 + n, 0x100 + n, _sdk(reply_id), SDK),
            wrapper]


def _root(trace_id, agent="support"):
    return _Span(trace_id, "agent.run", ROOT, None,
                 {"decimal.agent_name": agent, "gen_ai.operation.name": "invoke_agent"})


@pytest.fixture
def sent(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url="http://localhost:8000",
                                                      enabled=True))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "m", "status": "active"}
    monkeypatch.setattr(cfg, "_client", client)
    monkeypatch.setattr(otel, "_thread_turns", OrderedDict())
    traces: list = []
    monkeypatch.setattr(otel.DecimalSpanExporter, "_send", lambda self, trace: traces.append(trace))
    return traces


def _exporter():
    return otel.DecimalSpanExporter(agent_name="test-agent")


def _thread_record(trace, n):
    """The ``decimalai.message_thread`` record on the span of the trace's n-th call."""
    call = trace.llm_calls[n]
    span = next(s for s in trace.spans if s.id == call.span_id)
    return span.attributes.get("decimalai.message_thread")


def _first_turn(trace_id, **kw):
    return _turn(trace_id, 1, "msg_1", _SYSTEM_AND_QUESTION, _ASKS_TOOL, {"type": "create"}, **kw)


def _second_turn(trace_id, previous="msg_1", reply_id="msg_2", **kw):
    return _turn(trace_id, 2, reply_id, _TOOL_RESULT, _ANSWERS,
                 {"type": "continue", "previous_message_id": previous}, **kw)


def _unthreaded_rendering(sent_traces):
    """What the same two calls record with ``thread=False``: every request whole."""
    exporter = _exporter()
    exporter.export([*_turn(9, 1, "msg_a", _SYSTEM_AND_QUESTION, _ASKS_TOOL, None),
                     *_turn(9, 2, "msg_b", _WHOLE_SECOND_REQUEST, _ANSWERS, None), _root(9)])
    trace = sent_traces.pop()
    return [call.rendered_input for call in trace.llm_calls]


@pytest.mark.parametrize("sdk_span", [True, False], ids=["with-sdk-span", "openinference-only"])
def test_a_later_turn_is_recorded_with_the_thread_so_far(sent, sdk_span):
    expected = _unthreaded_rendering(sent)
    _exporter().export([*_first_turn(1, sdk_span=sdk_span), *_second_turn(1, sdk_span=sdk_span), _root(1)])
    (trace,) = sent
    rendered = [call.rendered_input for call in trace.llm_calls]
    assert [[m["role"] for m in r] for r in rendered] == [["system", "user"], ["system", "user", "assistant", "tool"]]
    assert rendered == expected, "what thread=False records for the same calls"
    assert _thread_record(trace, 0) is None, "the thread's first request was sent whole"
    assert _thread_record(trace, 1) == {"previous_message_id": "msg_1", "earlier_turns": "restored"}
    assert trace.user_input_preview == QUESTION


def test_each_turn_is_given_every_turn_before_it(sent):
    third = _turn(1, 3, "msg_3", _TOOL_RESULT, _ANSWERS,
                  {"type": "continue", "previous_message_id": "msg_2"})
    second_asks_again = _turn(1, 2, "msg_2", _TOOL_RESULT, _ASKS_TOOL,
                              {"type": "continue", "previous_message_id": "msg_1"})
    _exporter().export([*_first_turn(1), *second_asks_again, *third, _root(1)])
    (trace,) = sent
    assert [m["role"] for m in trace.llm_calls[2].rendered_input] == [
        "system", "user", "assistant", "tool", "assistant", "tool"]
    assert trace.llm_calls[2].rendered_input[0] == {"role": "system", "content": SYSTEM}


def test_a_thread_that_outlives_its_run_continues_in_the_next_runs_trace(sent):
    """A chat's ``client.beta.messages.thread()`` keeps one thread for every message: each
    message's run is a trace of its own."""
    exporter = _exporter()
    exporter.export([*_first_turn(1), _root(1)])
    exporter.export([*_second_turn(2), _root(2)])
    first, second = sent
    assert [m["role"] for m in second.llm_calls[0].rendered_input] == ["system", "user", "assistant", "tool"]
    assert _thread_record(second, 0)["earlier_turns"] == "restored"


def test_the_turn_before_is_known_while_its_own_trace_waits_for_its_root(sent):
    """A closed run's span can wait for its routing decision, so its trace is assembled
    after the next run's. Its calls reached the exporter first, and that is enough."""
    exporter = _exporter()
    exporter.export(_first_turn(1))           # the run's root has not ended yet
    exporter.export([*_second_turn(2), _root(2)])
    exporter.export([_root(1)])
    second, first = sent
    assert [m["role"] for m in second.llm_calls[0].rendered_input] == ["system", "user", "assistant", "tool"]
    assert [m["role"] for m in first.llm_calls[0].rendered_input] == ["system", "user"]


def test_a_turn_whose_earlier_turns_this_process_never_saw_is_recorded_as_sent_and_says_so(sent):
    """Another process sent the turns before (or they were evicted): nothing is made up."""
    later = _turn(1, 3, "msg_3", _TOOL_RESULT, _ANSWERS,
                  {"type": "continue", "previous_message_id": "msg_2"})
    _exporter().export([*_second_turn(1, previous="msg_elsewhere"), *later, _root(1)])
    (trace,) = sent
    assert [call.rendered_input for call in trace.llm_calls] == [
        [{"role": "tool", "content": "shipped"}]] * 2
    for n in (0, 1):
        assert _thread_record(trace, n)["earlier_turns"] == "not_seen"
    assert _thread_record(trace, 0)["previous_message_id"] == "msg_elsewhere"


def test_a_call_on_no_thread_is_left_as_it_was_sent(sent):
    rendered = _unthreaded_rendering(sent)
    assert [m["role"] for m in rendered[1]] == ["system", "user", "assistant", "tool"]
    assert otel._thread_turns == {}, "nothing is kept for calls on no thread"


def test_the_threads_kept_are_bounded(sent, monkeypatch):
    monkeypatch.setattr(otel, "_THREAD_TURNS_MAX", 2)
    exporter = _exporter()
    for n in range(3):
        exporter.export(_turn(n + 1, 1, f"msg_{n}", _SYSTEM_AND_QUESTION, _ASKS_TOOL, {"type": "create"}))
    assert list(otel._thread_turns) == ["msg_1", "msg_2"]
    exporter.export([*_second_turn(5, previous="msg_0", reply_id="msg_5"), _root(5)])
    assert _thread_record(sent[-1], 0)["earlier_turns"] == "not_seen"
