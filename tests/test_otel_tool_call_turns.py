"""A model turn that called tools keeps its calls in ``rendered_input`` and ``output``.

OpenInference records a call on the message that made it, sent or received:
``llm.{input,output}_messages.{i}.message.tool_calls.{j}.tool_call.{id,function.name,
function.arguments}``. The exporter read only role, content and tool_call_id, so a turn
that only called a tool reached the trace view as ``assistant (empty)``: on the tutorial's
OpenAI SDK tab the reader saw what ``lookup_order`` returned, but not that it was called,
or with ``ORD-1001`` (found by the tutorial's dashboard walk, 2026-11-16).

The attribute sets are what openinference-instrumentation-openai 0.1.65 recorded for a
Responses API loop on openai 3.31 that adds ``response.output`` back to its input, the
way Part 3's program does.
"""

from __future__ import annotations

import json

import decimalai.otel as otel

_SYSTEM_AND_QUESTION = {
    "llm.input_messages.0.message.content": "sys",
    "llm.input_messages.0.message.role": "system",
    "llm.input_messages.1.message.content": "How much back for ORD-1001?",
    "llm.input_messages.1.message.role": "user",
}

#: The second request: a reasoning item, the function call, and its output.
_SECOND_REQUEST = {
    **_SYSTEM_AND_QUESTION,
    "llm.input_messages.2.message.contents.0.message_content.type": "reasoning",
    "llm.input_messages.2.message.role": "assistant",
    "llm.input_messages.3.message.role": "assistant",
    "llm.input_messages.3.message.tool_calls.0.tool_call.function.arguments": '{"order_id":"ORD-1001"}',
    "llm.input_messages.3.message.tool_calls.0.tool_call.function.name": "lookup_order",
    "llm.input_messages.3.message.tool_calls.0.tool_call.id": "call_1",
    "llm.input_messages.4.message.content": "{'item': 'kettle'}",
    "llm.input_messages.4.message.role": "tool",
    "llm.input_messages.4.message.tool_call_id": "call_1",
}


def _call(name, arguments, call_id=None):
    call = {"id": call_id} if call_id else {}
    return {**call, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_a_turn_that_called_a_tool_names_the_tool_and_its_arguments():
    assert otel._rendered_input_from_attrs(_SECOND_REQUEST) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "How much back for ORD-1001?"},
        {"role": "assistant", "content": ""},  # the reasoning item: nothing to show
        {"role": "assistant", "content": "",
         "tool_calls": [_call("lookup_order", '{"order_id":"ORD-1001"}', "call_1")]},
        {"role": "tool", "content": "{'item': 'kettle'}"},
    ]


def test_the_question_is_still_the_ask_and_the_preview_still_text():
    rendered = otel._rendered_input_from_attrs(_SECOND_REQUEST)
    assert otel._ask_from_rendered_input(rendered) == "How much back for ORD-1001?"
    assert "tool_calls" not in (otel._preview_from_attrs(_SECOND_REQUEST, "input") or "")


def test_a_reply_of_parallel_calls_carries_every_call():
    """The Responses API returns each call as an output item, which OpenInference records as
    a message of its own; the reply carries them all, as LlmCallRecord.tool_calls does."""
    attrs = {
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup_order",
        "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"order_id":"ORD-1001"}',
        "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_1",
        "llm.output_messages.1.message.role": "assistant",
        "llm.output_messages.1.message.tool_calls.0.tool_call.function.name": "lookup_order",
        "llm.output_messages.1.message.tool_calls.0.tool_call.function.arguments": '{"order_id":"ORD-1002"}',
        "llm.output_messages.1.message.tool_calls.0.tool_call.id": "call_2",
    }
    assert otel._output_message_from_attrs(attrs) == {
        "role": "assistant", "content": "",
        "tool_calls": [_call("lookup_order", '{"order_id":"ORD-1001"}', "call_1"),
                       _call("lookup_order", '{"order_id":"ORD-1002"}', "call_2")],
    }
    assert [c.args for c in otel._extract_tool_calls(attrs)] == [
        {"order_id": "ORD-1001"}, {"order_id": "ORD-1002"}]


def test_a_reply_that_only_talks_is_unchanged():
    attrs = {"llm.output_messages.0.message.role": "assistant",
             "llm.output_messages.0.message.contents.0.message_content.text": "You get $49 back.",
             "llm.output_messages.0.message.contents.0.message_content.type": "text"}
    assert otel._output_message_from_attrs(attrs) == {"role": "assistant", "content": "You get $49 back."}


def test_a_call_with_no_arguments_reads_as_an_empty_object():
    """OpenInference leaves out ``{}`` (``lock_account()`` in the tutorial)."""
    attrs = {"llm.input_messages.0.message.role": "assistant",
             "llm.input_messages.0.message.tool_calls.0.tool_call.function.name": "lock_account",
             "llm.input_messages.0.message.tool_calls.0.tool_call.id": "call_9"}
    (message,) = otel._rendered_input_from_attrs(attrs)
    assert message["tool_calls"] == [_call("lock_account", "{}", "call_9")]
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {}


def test_a_message_with_only_calls_is_the_assistant_s():
    attrs = {**_SYSTEM_AND_QUESTION,
             "llm.input_messages.2.message.tool_calls.0.tool_call.function.name": "lookup_order",
             "llm.input_messages.2.message.tool_calls.0.tool_call.function.arguments": "{}"}
    assert otel._rendered_input_from_attrs(attrs)[2] == {
        "role": "assistant", "content": "", "tool_calls": [_call("lookup_order", "{}")]}


def test_calls_keep_their_order_past_ten_and_a_nameless_one_is_dropped():
    attrs = {"llm.input_messages.0.message.role": "assistant",
             "llm.input_messages.0.message.tool_calls.11.tool_call.id": "call_nameless"}
    for j in range(11):
        attrs[f"llm.input_messages.0.message.tool_calls.{j}.tool_call.function.name"] = f"t{j}"
    (message,) = otel._rendered_input_from_attrs(attrs)
    assert [c["function"]["name"] for c in message["tool_calls"]] == [f"t{j}" for j in range(11)]


def test_nameless_call_keys_alone_make_no_message():
    attrs = {**_SYSTEM_AND_QUESTION,
             "llm.input_messages.2.message.tool_calls.0.tool_call.id": "call_1"}
    assert otel._rendered_input_from_attrs(attrs) == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "How much back for ORD-1001?"},
    ]
