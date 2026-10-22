"""`decimalai.openai.agent_config()`: an agent's prompt and skills for the OpenAI SDK.

Driven through the real OpenAI SDK — a hand-written Chat Completions loop, a
Responses API loop, sync and async — against an in-process stub of both APIs (a
mock transport on the SDK's own HTTP client) that records each request and asks
for ``load_skill`` when told to. DecimalAI's answers come from the fake platform
the framework-neutral tests use, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import MagicMock

import pytest

openai = pytest.importorskip("openai")

import openai._base_client as _openai_base_client  # noqa: E402

import decimalai._agent as agent_mod  # noqa: E402
import decimalai._agent_binding as binding  # noqa: E402
import decimalai.openai as do  # noqa: E402
from decimalai._agent import SavedPrompt  # noqa: E402
from decimalai._agent_binding import ConversationRecord, MissingPromptVariable  # noqa: E402
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter  # noqa: E402
from tests.test_agent_binding import FakePlatform  # noqa: E402

#: The HTTP library the installed SDK is built on: httpx2 from openai 3, httpx before.
http = getattr(_openai_base_client, "httpx2", None) or _openai_base_client.httpx

BASE = "http://openai-agent-config.test"
SAM = {"customer": {"name": "Sam"}}

LOOKUP_ORDER = {"type": "function", "function": {
    "name": "lookup_order", "description": "Look up an order.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                   "required": ["order_id"], "additionalProperties": False}}}
LOOKUP_ORDER_RESPONSES = {"type": "function", **LOOKUP_ORDER["function"], "strict": True}


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=True))
    monkeypatch.setattr(cfg, "_client", MagicMock())
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    monkeypatch.setattr(do, "_tracing_turned_on", True)  # tracing is tested on its own below
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url=BASE, inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(do, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt="Hello {{customer.name}}.",
                             version_number=4, content_hash="c" * 64)
    monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kwargs: fake.saved)
    return fake


class OpenAIStub:
    """A stub of ``POST /v1/chat/completions`` and ``POST /v1/responses``.

    Each step answers one request: ``("call", skill)`` asks for ``load_skill``,
    ``("tool", name)`` asks for another tool, ``"answer"`` ends the turn.
    """

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests: list = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        step = self.steps.pop(0)
        n = len(self.requests)
        if request.url.path.endswith("/responses"):
            return http.Response(200, json=self._response(step, n))
        return http.Response(200, json=self._completion(step, n))

    @staticmethod
    def _call(step):
        kind, name = step
        tool = "load_skill" if kind == "call" else name
        arguments = {"name": name} if kind == "call" else {"order_id": "ORD-1"}
        return tool, json.dumps(arguments)

    def _completion(self, step, n):
        if step == "answer":
            message = {"role": "assistant", "content": "done"}
            finish = "stop"
        else:
            tool, arguments = self._call(step)
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{n}", "type": "function", "function": {"name": tool, "arguments": arguments}}]}
            finish = "tool_calls"
        return {"id": f"chatcmpl-{n}", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def _response(self, step, n):
        if step == "answer":
            output = [{"type": "message", "id": f"msg_{n}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": "done", "annotations": []}]}]
        else:
            tool, arguments = self._call(step)
            output = [{"type": "function_call", "id": f"fc_{n}", "call_id": f"call_{n}", "name": tool,
                       "arguments": arguments, "status": "completed"}]
        return {"id": f"resp_{n}", "object": "response", "created_at": 0, "model": "m",
                "status": "completed", "output": output, "parallel_tool_calls": True,
                "tool_choice": "auto", "tools": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}

    def client(self):
        return openai.OpenAI(api_key="sk-test", base_url=f"{BASE}/v1", max_retries=0,
                             http_client=http.Client(transport=http.MockTransport(self)))

    def async_client(self):
        async def handler(request):
            return self(request)

        return openai.AsyncOpenAI(api_key="sk-test", base_url=f"{BASE}/v1", max_retries=0,
                                  http_client=http.AsyncClient(transport=http.MockTransport(handler)))

    def chat_tool_messages(self, index):
        return [m for m in self.requests[index]["messages"] if m.get("role") == "tool"]

    def outputs(self, index):
        return [item for item in self.requests[index]["input"]
                if isinstance(item, dict) and item.get("type") == "function_call_output"]


def chat_loop(client, run, *, tools=(LOOKUP_ORDER,)):
    """The loop the docs teach: answer load_skill with run.tool_result, other tools yourself."""
    messages = [{"role": "system", "content": run.system}, run.message]
    while True:
        reply = client.chat.completions.create(
            model="gpt-5.4-mini", messages=messages, tools=[*tools, *run.tools]).choices[0].message
        messages.append(reply)
        if not reply.tool_calls:
            return reply.content
        for call in reply.tool_calls:
            messages.append(run.tool_result(call)
                            or {"role": "tool", "tool_call_id": call.id, "content": "{}"})


def responses_loop(client, run, *, tools=(LOOKUP_ORDER_RESPONSES,)):
    response = client.responses.create(model="gpt-5.4-mini", instructions=run.system,
                                       input=[run.message], tools=[*tools, *run.response_tools])
    while calls := [item for item in response.output if item.type == "function_call"]:
        response = client.responses.create(
            model="gpt-5.4-mini", instructions=run.system, previous_response_id=response.id,
            tools=[*tools, *run.response_tools],
            input=[run.tool_result(call)
                   or {"type": "function_call_output", "call_id": call.call_id, "output": "{}"}
                   for call in calls])
    return response.output_text


# ── Chat Completions ─────────────────────────────────────────


def test_a_chat_completions_loop_gets_the_prompt_the_skill_list_and_load_skill(platform):
    support = do.agent_config("support")
    assert type(support).__name__ == "OpenAIAgentConfig" and support.name == "support"
    api = OpenAIStub(("call", "damaged-item"), "answer")
    with support.run("My keyboard arrived cracked.", variables=SAM) as run:
        assert chat_loop(api.client(), run) == "done"
    first, second = api.requests
    system, user = first["messages"]
    assert system["role"] == "system" and system["content"].startswith("Hello Sam.\n\n## Available Skills")
    assert system["content"].endswith(LOAD_SKILL_PROMPT_HINT)
    assert user == {"role": "user", "content": "My keyboard arrived cracked."}
    assert [tool["function"]["name"] for tool in first["tools"]] == ["lookup_order", "load_skill"]
    load_skill = first["tools"][1]
    assert load_skill["type"] == "function" and "strict" not in load_skill["function"], \
        "no strict: some OpenAI-compatible endpoints refuse it in Chat Completions"
    assert load_skill["function"]["parameters"]["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    assert first["messages"][0] == second["messages"][0] and first["tools"] == second["tools"], \
        "the run's second request repeats the first one's prefix (the prompt cache reuses it)"
    (result,) = api.chat_tool_messages(1)
    assert result["tool_call_id"] == "call_1" and result["content"].startswith("## Skill: damaged-item")
    assert list(platform.queries.values()) == ["My keyboard arrived cracked."]


def test_another_tools_call_is_the_callers_to_answer(platform):
    support = do.agent_config("support")
    api = OpenAIStub(("tool", "lookup_order"), "answer")
    with support.run("Where is my order?", variables=SAM) as run:
        chat_loop(api.client(), run)
        assert run.tool_result({"id": "c", "type": "function",
                                "function": {"name": "lookup_order", "arguments": "{}"}}) is None
        assert run.tool_result({"type": "function_call", "call_id": "c", "name": "lookup_order",
                                "arguments": "{}"}) is None
        assert run.tool_result({"type": "message"}) is None and run.tool_result(None) is None
    (result,) = api.chat_tool_messages(1)
    assert result == {"role": "tool", "tool_call_id": "call_1", "content": "{}"}
    assert not [r for r in platform.requests if r[1].endswith("/body")]


def test_a_tool_call_given_as_a_dict_is_answered_in_its_own_shape(platform):
    with do.agent_config("support").run("cracked", variables=SAM) as run:
        chat = run.tool_result({"id": "call_9", "type": "function",
                                "function": {"name": "load_skill", "arguments": '{"name": "damaged-item"}'}})
    assert chat["role"] == "tool" and chat["tool_call_id"] == "call_9"
    assert chat["content"].startswith("## Skill: damaged-item")


def test_arguments_that_are_not_json_are_answered_with_a_sentence(platform):
    with do.agent_config("support").run("cracked", variables=SAM) as run:
        result = run.tool_result({"id": "c", "type": "function",
                                  "function": {"name": "load_skill", "arguments": "{not json"}})
    assert result["content"] == "load_skill needs the name of one skill from the list."
    assert not [r for r in platform.requests if r[1].endswith("/body")]


def test_a_name_not_on_the_list_is_answered_without_a_request(platform):
    """Outside strict mode Chat Completions does not hold a call to the schema's enum, so
    DecimalAI checks the name."""
    api = OpenAIStub(("call", "invented-skill"), "answer")
    with do.agent_config("support").run("cracked", variables=SAM) as run:
        chat_loop(api.client(), run)
    assert "not one of this agent's skills" in api.chat_tool_messages(1)[0]["content"]
    assert not [r for r in platform.requests if r[1].endswith("/body")]


def test_a_model_asking_with_other_capitals_gets_the_listed_skill(platform):
    api = OpenAIStub(("call", "Damaged-Item"), "answer")
    with do.agent_config("support").run("cracked", variables=SAM) as run:
        chat_loop(api.client(), run)
    assert api.chat_tool_messages(1)[0]["content"].startswith("## Skill: damaged-item")


def test_a_skill_loaded_once_in_a_run_is_not_sent_again(platform):
    api = OpenAIStub(("call", "damaged-item"), ("call", "damaged-item"), "answer")
    with do.agent_config("support").run("cracked", variables=SAM) as run:
        chat_loop(api.client(), run)
    assert "already loaded" in api.chat_tool_messages(2)[-1]["content"]
    assert len(platform.body_calls("damaged-item")) == 1


def test_each_run_is_routed_on_its_own_message(platform):
    support = do.agent_config("support")
    for question in ("My keyboard arrived cracked.", "Where is my order?"):
        with support.run(question, variables=SAM) as run:
            chat_loop(OpenAIStub("answer").client(), run)
    assert list(platform.queries.values()) == ["My keyboard arrived cracked.", "Where is my order?"]


def test_the_load_names_the_decimalai_agent_and_the_run_routes_once(platform):
    api = OpenAIStub(("call", "damaged-item"), "answer")
    with do.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        chat_loop(api.client(), run)
    (body,) = [params for _method, path, params in platform.requests if path.endswith("/body")]
    assert body.get("agent_name") == "support"
    assert len(platform.menu_calls()) == 1, "one decision per run, across its tool steps"


def test_load_skill_gives_the_text_alone_for_a_loop_or_runner_of_your_own(platform):
    with do.agent_config("support").run("cracked", variables=SAM) as run:
        assert run.load_skill("damaged-item").startswith("## Skill: damaged-item")
        assert "not one of this agent's skills" in run.load_skill("invented-skill")


# ── the Responses API ────────────────────────────────────────


def test_a_responses_loop_gets_instructions_input_and_a_flat_strict_tool(platform):
    support = do.agent_config("support")
    api = OpenAIStub(("call", "damaged-item"), "answer")
    with support.run("My keyboard arrived cracked.", variables=SAM) as run:
        assert responses_loop(api.client(), run) == "done"
        assert run.response_tools[0] == {"type": "function", **run.tools[0]["function"], "strict": True}, \
            "one definition, in each API's shape"
    first, second = api.requests
    assert first["instructions"].startswith("Hello Sam.\n\n## Available Skills")
    assert first["input"] == [{"role": "user", "content": "My keyboard arrived cracked."}]
    assert [tool["name"] for tool in first["tools"]] == ["lookup_order", "load_skill"]
    load_skill = first["tools"][1]
    assert load_skill["type"] == "function" and load_skill["strict"] is True and "function" not in load_skill
    assert load_skill["parameters"]["additionalProperties"] is False, "strict mode's rules hold"
    assert second["previous_response_id"] == "resp_1" and second["instructions"] == first["instructions"]
    (output,) = api.outputs(1)
    assert output["call_id"] == "call_1" and output["output"].startswith("## Skill: damaged-item")


def test_a_function_call_item_given_as_a_dict_is_answered_with_its_output(platform):
    with do.agent_config("support").run("cracked", variables=SAM) as run:
        result = run.tool_result({"type": "function_call", "call_id": "call_7", "name": "load_skill",
                                  "arguments": '{"name": "damaged-item"}'})
    assert set(result) == {"type", "call_id", "output"} and result["type"] == "function_call_output"
    assert result["call_id"] == "call_7" and result["output"].startswith("## Skill: damaged-item")


# ── async ─────────────────────────────────────────────────────


def test_an_async_run_serves_loads_off_the_event_loop(platform):
    support = do.agent_config("support")
    api = OpenAIStub(("call", "damaged-item"), "answer")

    async def main():
        client = api.async_client()
        async with support.run("My keyboard arrived cracked.", variables=SAM) as run:
            messages = [{"role": "system", "content": run.system}, run.message]
            while True:
                reply = (await client.chat.completions.create(
                    model="m", messages=messages, tools=[*run.tools])).choices[0].message
                messages.append(reply)
                if not reply.tool_calls:
                    return reply.content
                for call in reply.tool_calls:
                    messages.append(await run.atool_result(call))

    assert asyncio.run(main()) == "done"
    assert api.chat_tool_messages(1)[0]["content"].startswith("## Skill: damaged-item")


def test_an_async_run_answers_a_responses_call_and_a_direct_load(platform):
    support = do.agent_config("support")

    async def main():
        async with support.run("cracked", variables=SAM) as run:
            output = await run.atool_result({"type": "function_call", "call_id": "c1", "name": "load_skill",
                                             "arguments": '{"name": "damaged-item"}'})
            return output, await run.aload_skill("order-support")

    output, text = asyncio.run(main())
    assert output["call_id"] == "c1" and output["output"].startswith("## Skill: damaged-item")
    assert text.startswith("## Skill: order-support")


def test_the_sync_tool_result_inside_an_async_run_says_to_await_it(platform, caplog):
    async def main():
        async with do.agent_config("support").run("cracked", variables=SAM) as run:
            return run.tool_result({"id": "c", "type": "function",
                                    "function": {"name": "load_skill", "arguments": '{"name": "damaged-item"}'}})

    with caplog.at_level(logging.WARNING, logger="decimalai.openai"):
        result = asyncio.run(main())
    assert result["content"].startswith("## Skill: damaged-item")
    assert any("atool_result" in r.getMessage() for r in caplog.records)


# ── prompt, variables, skill list ───────────────────────────


def test_a_missing_variable_stops_the_run_before_any_model_call(platform):
    with pytest.raises(MissingPromptVariable, match="customer.name"):
        with do.agent_config("support").run("Hi"):
            raise AssertionError("the block must not run")


def test_no_skill_list_means_no_tool(platform):
    platform.skills = []
    with do.agent_config("support").run("Hi", variables=SAM) as run:
        assert run.system == "Hello Sam."
        assert run.tools == [] and run.response_tools == []


def test_always_include_text_ends_the_system_text(platform):
    """D-C: an always-include skill's full text follows the skill list in the system text."""
    platform.always = {"order-support": 1}
    with do.agent_config("support").run("Where is my order?", variables=SAM) as run:
        assert run.system.startswith("Hello Sam.\n\n## Available Skills")
        assert run.system.endswith("## Skill: order-support\n\n# Order support\n\nUse lookup_order for order facts.")


def test_a_cut_list_names_no_enum(platform):
    platform.truncated = True
    with do.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        assert "enum" not in run.tools[0]["function"]["parameters"]["properties"]["name"]
        assert "enum" not in run.response_tools[0]["parameters"]["properties"]["name"]


def test_decimalai_down_sends_the_prompt_alone_and_the_run_still_completes(platform):
    platform.down = True
    api = OpenAIStub("answer")
    with do.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        assert chat_loop(api.client(), run) == "done"
        assert run.system == "Hello Sam." and run.tools == []
    with pytest.raises(RuntimeError):
        with do.agent_config("support", on_unavailable="fail").run("cracked", variables=SAM):
            raise AssertionError("the block must not run")


def test_the_kill_switch_reaches_configs_built_before_it(platform, monkeypatch):
    import decimalai._config as cfg

    support = do.agent_config("support")
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    with support.run("cracked", variables=SAM) as run:
        assert run.system == "Hello Sam."
        assert run.tools == [] and run.response_tools == []


def test_with_load_skill_switched_off_the_routed_skills_text_goes_in_the_system_text(platform, monkeypatch):
    """init(load_skill_tool=False) / DECIMALAI_LOAD_SKILL_TOOL=0: no tool, the skills' text instead."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=True,
                                                      load_skill_tool=False))
    with do.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        assert run.tools == [] and run.response_tools == []
        assert "## Skill: damaged-item" in run.system


# ── conversations ────────────────────────────────────────────


def test_a_conversation_keeps_its_system_and_tools_when_the_prompt_and_list_change(platform):
    support = do.agent_config("support")
    api = OpenAIStub("answer", "answer")
    chat = support.conversation(variables=SAM)
    for index, question in enumerate(["Where is my order?", "It arrived cracked."]):
        if index == 1:
            support._prompt = SavedPrompt(agent_name="support", system_prompt="Changed, {{customer.name}}.",
                                          version_number=5, content_hash="d" * 64)
            platform.skills = ["damaged-item"]
        with chat.run(question) as run:
            chat_loop(api.client(), run)
    first, second = api.requests
    assert first["messages"][0] == second["messages"][0] and first["tools"] == second["tools"]
    assert list(platform.queries.values()) == ["Where is my order?", "It arrived cracked."], \
        "every message is still routed on its own"
    with support.run("Hi", variables=SAM) as run:
        assert run.system.startswith("Changed, Sam."), "the change reaches the next conversation"


def test_a_conversation_resumes_from_its_state_in_another_process(platform):
    support = do.agent_config("support")
    chat = support.conversation(variables=SAM)
    with chat.run("Where is my order?") as run:
        system, tools, response_tools = run.system, run.tools, run.response_tools
    state = json.loads(json.dumps(chat.state))
    assert ConversationRecord.from_state(state, "support") is not None, "the record every binding shares"
    later = do.agent_config("support")
    later._prompt = SavedPrompt(agent_name="support", system_prompt="Changed.", version_number=5,
                                content_hash="d" * 64)
    resumed = later.conversation(state=state)
    assert resumed.id == chat.id
    with resumed.run("It arrived cracked.") as run:
        assert (run.system, run.tools, run.response_tools) == (system, tools, response_tools)
    with pytest.raises(ValueError, match="not both"):
        later.conversation(state=state, variables=SAM)
    with pytest.raises(ValueError, match="belongs to agent"):
        later.conversation(state={**state, "agent": "billing"})


def test_a_cut_list_conversation_shows_new_skills_in_its_user_turn(platform):
    """A later message of a conversation whose list is cut is routed on its own text; the skills
    it picked that the conversation has not shown go in run.message, never in the system text."""
    platform.truncated = True
    support = do.agent_config("support")
    chat = support.conversation(variables=SAM)
    with chat.run("Where is my order?") as run:
        system = run.system
        assert "order-support" in system and "damaged-item" not in system
    with chat.run("It arrived cracked.") as run:
        assert run.system == system
        message = run.message
        assert message["role"] == "user" and message["content"].startswith("It arrived cracked.\n\n")
        assert "damaged-item" in message["content"]
        result = run.tool_result({"type": "function_call", "call_id": "c", "name": "load_skill",
                                  "arguments": '{"name": "damaged-item"}'})
        assert result["output"].startswith("## Skill: damaged-item")


# ── misuse ───────────────────────────────────────────────────


def test_system_read_without_tools_warns_at_exit(platform, caplog):
    with caplog.at_level(logging.WARNING, logger="decimalai.openai"):
        with do.agent_config("support").run("Hi", variables=SAM) as run:
            run.system
    assert any("never included it" in r.getMessage() and "run.response_tools" in r.getMessage()
               for r in caplog.records)


def test_run_pieces_outside_the_with_block_say_how_to_use_them(platform):
    run = do.agent_config("support").run("Hi", variables=SAM)
    for piece in ("system", "message", "tools", "response_tools"):
        with pytest.raises(RuntimeError, match="inside `with support.run"):
            getattr(run, piece)
    with pytest.raises(RuntimeError, match="inside `with support.run"):
        run.tool_result({"type": "function_call"})


def test_a_run_opens_once(platform):
    run = do.agent_config("support").run("Hi", variables=SAM)
    with run:
        pass
    with pytest.raises(RuntimeError, match="opened once"):
        with run:
            pass


def test_the_module_imports_without_the_openai_sdk(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "openai", None)
    module = importlib.reload(do)
    try:
        with pytest.raises(ImportError, match=r"decimalai\[openai\]"):
            module.agent_config("support")
    finally:
        monkeypatch.delitem(sys.modules, "openai")
        importlib.reload(do)


def test_agent_config_turns_on_openai_tracing_once(platform, monkeypatch):
    import decimalai.providers as providers

    calls = []
    monkeypatch.setattr(do, "_tracing_turned_on", False)
    monkeypatch.setattr(providers, "instrument_for_runs", calls.append)
    do.agent_config("support")
    do.agent_config("support")
    assert calls == ["openai"]


# ── the trace ────────────────────────────────────────────────


def test_the_run_is_one_trace_named_for_the_decimalai_agent_with_its_record(platform, monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel
    import decimalai.providers as providers

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(otel._AgentNameStamper())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(providers, "_last_provider", provider)
    with do.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        chat_loop(OpenAIStub(("call", "damaged-item"), "answer").client(), run)
    (span,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert span.name == otel.RUN_SPAN_NAME and span.attributes["decimal.agent_name"] == "support"
    rail = otel._pop_skill_rail(span.context.trace_id)
    assert rail["routing_id"] and rail["offered"] == ["damaged-item", "order-support"]
    assert rail["loaded"] == ["damaged-item"]
    attributes = otel._decimal_record_attributes(span)
    assert attributes["decimal.prompt.version"] == 4
    assert list(attributes["decimal.prompt.variables"]) == ["customer.name"], "names, never values"


def test_a_late_routing_decision_is_waited_for_and_never_left_behind(platform, monkeypatch):
    """A later run shows the local copy at once and routes in the background; its trace ends
    once that decision lands, off the caller's path, and decimalai.flush() waits for it."""
    pytest.importorskip("opentelemetry.sdk.trace")
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai
    import decimalai.otel as otel
    import decimalai.providers as providers

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(providers, "_last_provider", provider)
    support = do.agent_config("support")
    with support.run("Where is my order?", variables=SAM):
        pass  # the first run routes before it opens
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(binding, "_in_background",
                        lambda fn, *args: pool.submit(contextvars.copy_context().run, fn, *args))
    exporter.clear()
    with support.run("It arrived cracked.", variables=SAM) as run:
        assert run._state.get("pending") is not None
    decimalai.flush()  # drains the provider runs' finishing spans first
    (span,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert otel._pop_skill_rail(span.context.trace_id)["routing_id"], "the background decision landed"
    assert len(platform.queries) == 2


@pytest.fixture
def traced(platform, monkeypatch):
    """DecimalAI's real tracing for agent_config(): the OpenInference instrumentor and the
    exporter, with a provider standing in for the app's global one. Yields (that provider, sent
    traces)."""
    pytest.importorskip("openinference.instrumentation.openai")
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai.otel as otel
    import decimalai.providers as providers

    OpenAIInstrumentor().uninstrument()
    provider = TracerProvider()
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: provider)
    for name in ("_pipeline_provider", "_last_provider", "_pipeline_exporter"):
        monkeypatch.setattr(providers, name, None)
    for name in ("_instrumented", "_runs_via_another_tool", "_runs_only"):
        monkeypatch.setattr(providers, name, set())
    for name in ("_run_providers", "_explicitly_traced", "_openinference_provider"):
        monkeypatch.setattr(providers, name, {})
    # An earlier test's init(otel=True) / init(crewai=True) leaves its provider here, and
    # an exporter built while one is listed sends every trace, not only runs (as it should
    # after a real init(otel=True)), so the test would grade another test's setup.
    monkeypatch.setattr(otel, "_installed_providers", [])
    sent: list = []
    monkeypatch.setattr(otel.DecimalSpanExporter, "_send", lambda self, trace: sent.append(trace))
    monkeypatch.setattr(do, "_tracing_turned_on", False)
    yield provider, sent
    OpenAIInstrumentor().uninstrument()


def test_agent_config_sends_its_runs_and_nothing_else_from_the_process(traced, caplog):
    """agent_config() turns OpenAI tracing on for its runs: an OpenAI call made outside one and
    the app's own spans stay in the process; the run is one trace with its two calls."""
    provider, sent = traced
    support = do.agent_config("support")
    OpenAIStub("answer").client().chat.completions.create(
        model="m", messages=[{"role": "user", "content": "PRIVATE question"}])
    with provider.get_tracer("app").start_as_current_span("GET /health"):
        pass
    assert sent == [], "nothing outside a run is DecimalAI's to send"
    with caplog.at_level(logging.WARNING, logger="decimalai.openai"):
        with support.run("It arrived cracked.", variables=SAM) as run:
            chat_loop(OpenAIStub(("call", "damaged-item"), "answer").client(), run)
    (trace,) = sent
    assert trace.agent_name == "support" and len(trace.llm_calls) == 2
    assert not [r for r in caplog.records if "no OpenAI request" in r.getMessage()]


def test_a_responses_run_is_one_trace_with_its_calls(traced):
    _provider, sent = traced
    support = do.agent_config("support")
    with support.run("It arrived cracked.", variables=SAM) as run:
        responses_loop(OpenAIStub(("call", "damaged-item"), "answer").client(), run)
    (trace,) = sent
    assert trace.agent_name == "support" and len(trace.llm_calls) == 2


def test_a_run_whose_calls_were_made_after_the_block_warns_once(traced, caplog):
    _provider, sent = traced
    support = do.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai.openai"):
        with support.run("It arrived cracked.", variables=SAM) as run:
            system, tools = run.system, run.tools
        OpenAIStub("answer").client().chat.completions.create(
            model="m", messages=[{"role": "system", "content": system}], tools=tools)
    assert [r for r in caplog.records if "no OpenAI request was traced" in r.getMessage()]
