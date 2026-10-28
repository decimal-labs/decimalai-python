"""Pydantic AI's GenAI-semconv messages, as DecimalAI's OTel exporter records them.

Pydantic AI 2.x keeps its instructions in ``gen_ai.system_instructions`` (a JSON array of
parts) and sends a tool's result as a ``tool_call_response`` part with a ``result`` field,
inside a ``user`` message. The exporter read neither: a ``load_skill`` result was recorded
as an empty user message, and the instructions as JSON text. So no Pydantic AI trace showed
the skill text the model was given. Found by the live binding test
(``tests/integration/test_framework_live_binding_skill_loading.py``) on 2026-10-22.
"""

from __future__ import annotations

import json

import pytest

from decimalai.otel import _extract_system_prompt, _rendered_input_from_attrs

_INSTRUCTIONS = json.dumps([{"type": "text", "content": "You are a support agent."}])

#: The shape pydantic-ai 2.57 emits for a model request after one tool call.
_INPUT_AFTER_A_TOOL_CALL = json.dumps([
    {"role": "user", "parts": [{"type": "text", "content": "Where is order A1?"}]},
    {"role": "assistant", "parts": [
        {"type": "tool_call", "id": "c1", "name": "load_skill", "arguments": {"name": "orders"}},
    ]},
    {"role": "user", "parts": [
        {"type": "tool_call_response", "id": "c1", "name": "load_skill",
         "result": "## Skill: orders\n\nBODY"},
    ]},
])


def test_a_tool_result_is_a_tool_message_carrying_its_text():
    attrs = {"gen_ai.system_instructions": _INSTRUCTIONS,
             "gen_ai.input.messages": _INPUT_AFTER_A_TOOL_CALL}
    assert _rendered_input_from_attrs(attrs) == [
        {"role": "system", "content": "You are a support agent."},
        {"role": "user", "content": "Where is order A1?"},
        {"role": "assistant", "content": 'load_skill({"name": "orders"})'},
        {"role": "tool", "content": "## Skill: orders\n\nBODY"},
    ]


def test_the_semconv_spelling_response_still_reads():
    """The GenAI spec (and AG2) call the field ``response``; a dict is rendered as JSON."""
    attrs = {"gen_ai.input.messages": json.dumps([
        {"role": "tool", "parts": [
            {"type": "tool_call_response", "id": "c1", "response": {"temp_c": 21}},
        ]},
    ])}
    assert _rendered_input_from_attrs(attrs) == [{"role": "tool", "content": '{"temp_c": 21}'}]


def test_a_message_mixing_a_tool_result_and_text_keeps_its_role():
    attrs = {"gen_ai.input.messages": json.dumps([
        {"role": "user", "parts": [
            {"type": "tool_call_response", "id": "c1", "result": "done"},
            {"type": "text", "content": "Now summarize."},
        ]},
    ])}
    assert _rendered_input_from_attrs(attrs) == [{"role": "user", "content": "done\nNow summarize."}]


def test_a_system_message_in_the_messages_is_not_doubled():
    """A span carrying its instructions in both places records them once."""
    attrs = {
        "gen_ai.system_instructions": "You are terse.",
        "gen_ai.input.messages": json.dumps([
            {"role": "system", "parts": [{"type": "text", "content": "You are terse."}]},
            {"role": "user", "parts": [{"type": "text", "content": "2+2?"}]},
        ]),
    }
    assert _rendered_input_from_attrs(attrs) == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "2+2?"},
    ]


#: What Pydantic AI 2.63 adds to every request to a model with no mid-conversation system
#: role (Claude): a fixed system message saying what its ``<system>`` tag means.
_ANNOUNCEMENT = ("Mid-conversation system messages arrive inside user turns wrapped in <system> "
                 "tags, because this channel has no system role.")


def test_the_instructions_lead_a_system_message_of_the_messages():
    """Pydantic AI 2.63's own system message does not stand in for the instructions.

    The rule was "the attribute adds a system message when the messages carry none", so on
    2.63 the agent's prompt left every Claude call's rendered input, and the manifest took
    the announcement as the agent's prompt. Found by the prod prober on 2026-10-28.
    """
    attrs = {
        "gen_ai.system_instructions": _INSTRUCTIONS,
        "gen_ai.input.messages": json.dumps([
            {"role": "system", "parts": [{"type": "text", "content": _ANNOUNCEMENT}]},
            {"role": "user", "parts": [{"type": "text", "content": "Where is order A1?"}]},
        ]),
    }
    assert _rendered_input_from_attrs(attrs) == [
        {"role": "system", "content": "You are a support agent."},
        {"role": "system", "content": _ANNOUNCEMENT},
        {"role": "user", "content": "Where is order A1?"},
    ]
    assert _extract_system_prompt(attrs) == "You are a support agent."


def test_system_instructions_become_the_prompt_as_text():
    assert _extract_system_prompt({"gen_ai.system_instructions": _INSTRUCTIONS}) == (
        "You are a support agent.")
    assert _extract_system_prompt({"gen_ai.system_instructions": "Plain text."}) == "Plain text."


def test_a_real_pydantic_ai_run_records_the_tool_result():
    """The same, from the spans pydantic-ai itself emits, so a format change shows here."""
    pytest.importorskip("pydantic_ai")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    try:
        from pydantic_ai import InstrumentationSettings
    except ImportError:  # older pydantic-ai
        from pydantic_ai.models.instrumented import InstrumentationSettings

    spans = InMemorySpanExporter()
    provider = TracerProvider()  # private: never the global one
    provider.add_span_processor(SimpleSpanProcessor(spans))

    def model(messages, info):
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("lookup", {"q": "x"})])
        return ModelResponse(parts=[TextPart("done")])

    agent = Agent(FunctionModel(model), instructions="SYSTEM-TEXT")

    @agent.tool_plain
    def lookup(q: str) -> str:
        return "BODY-TOKEN-123"

    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    try:
        agent.run_sync("hello?")
    finally:
        Agent.instrument_all(False)

    chats = [dict(s.attributes or {}) for s in spans.get_finished_spans()
             if "gen_ai.input.messages" in (s.attributes or {})]
    assert len(chats) == 2, [s.name for s in spans.get_finished_spans()]
    rendered = _rendered_input_from_attrs(chats[-1])
    assert rendered[0] == {"role": "system", "content": "SYSTEM-TEXT"}
    assert rendered[-1] == {"role": "tool", "content": "BODY-TOKEN-123"}


def test_a_real_pydantic_ai_run_on_claude_records_its_instructions():
    """Pydantic AI's own Anthropic model, on a stub Messages API: every call's rendered input
    leads with the instructions, which every request sent. On 2.63 the request also carries
    the announcement, which used to push the instructions out of the trace."""
    pytest.importorskip("pydantic_ai")
    pytest.importorskip("pydantic_ai.models.anthropic")
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from pydantic_ai import Agent
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    try:
        from pydantic_ai import InstrumentationSettings
    except ImportError:  # older pydantic-ai
        from pydantic_ai.models.instrumented import InstrumentationSettings

    sent: list = []

    async def messages_api(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            content = [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}]
            stop = "tool_use"
        else:
            content = [{"type": "text", "text": "done"}]
            stop = "end_turn"
        return httpx2.Response(200, json={
            "id": f"msg_{len(sent)}", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    client = anthropic.AsyncAnthropic(
        api_key="sk-ant-test", http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(messages_api)))
    spans = InMemorySpanExporter()
    provider = TracerProvider()  # private: never the global one
    provider.add_span_processor(SimpleSpanProcessor(spans))
    # A max_tokens cap keeps the request unstreamed, so the stub answers in one JSON body.
    agent = Agent(AnthropicModel("claude-haiku-4-5", provider=AnthropicProvider(anthropic_client=client)),
                  instructions="SYSTEM-TEXT", model_settings={"max_tokens": 64})

    @agent.tool_plain
    def lookup(q: str) -> str:
        return "BODY-TOKEN-123"

    Agent.instrument_all(InstrumentationSettings(tracer_provider=provider))
    try:
        agent.run_sync("hello?")
    finally:
        Agent.instrument_all(False)

    def system_text(body):
        system = body.get("system")
        return system if isinstance(system, str) else " ".join(b.get("text", "") for b in system or ())

    assert len(sent) == 2 and all("SYSTEM-TEXT" in system_text(body) for body in sent)
    chats = [dict(s.attributes or {}) for s in spans.get_finished_spans()
             if "gen_ai.input.messages" in (s.attributes or {})]
    assert len(chats) == 2, [s.name for s in spans.get_finished_spans()]
    for chat in chats:
        rendered = _rendered_input_from_attrs(chat)
        assert rendered[0] == {"role": "system", "content": "SYSTEM-TEXT"}
        assert [m["content"] for m in rendered].count("SYSTEM-TEXT") == 1
        assert _extract_system_prompt(chat) == "SYSTEM-TEXT"
    assert _rendered_input_from_attrs(chats[-1])[-1] == {"role": "tool", "content": "BODY-TOKEN-123"}
