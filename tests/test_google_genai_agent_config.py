"""`decimalai.google_genai.agent_config()`: an agent's prompt and skills for the Gemini API.

Driven through the real google-genai SDK — automatic function calling in a chat, a
hand-written ``generate_content`` loop, sync and async — against an in-process stub of
``generateContent`` (a mock transport on the SDK's own httpx client) that records each
request and asks for ``load_skill`` when told to. DecimalAI's answers come from the fake
platform the framework-neutral tests use, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import warnings
from unittest.mock import MagicMock

import pytest

genai = pytest.importorskip("google.genai")
httpx = pytest.importorskip("httpx")

from google.genai import types  # noqa: E402

import decimalai._agent as agent_mod  # noqa: E402
import decimalai._agent_binding as binding  # noqa: E402
import decimalai.google_genai as dg  # noqa: E402
from decimalai._agent import SavedPrompt  # noqa: E402
from decimalai._agent_binding import ConversationRecord, MissingPromptVariable  # noqa: E402
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter  # noqa: E402
from tests.test_agent_binding import FakePlatform  # noqa: E402

BASE = "http://google-genai-agent-config.test"
SAM = {"customer": {"name": "Sam"}}
MODEL = "gemini-3.5-flash"

# google-genai recommends automatic function calling in chats; the warning it prints for
# generate_content is not what these tests are about.
pytestmark = pytest.mark.filterwarnings("ignore:Direct use of automatic function calling")


def lookup_order(order_id: str) -> str:
    """Look up an order's status by its id.

    Args:
        order_id: The order id, for example A1001.
    """
    return "{}"


LOOKUP_ORDER = {"function_declarations": [{
    "name": "lookup_order", "description": "Look up an order.",
    "parameters_json_schema": {"type": "object", "properties": {"order_id": {"type": "string"}},
                               "required": ["order_id"]}}]}


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=True))
    monkeypatch.setattr(cfg, "_client", MagicMock())
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    monkeypatch.setattr(dg, "_tracing_turned_on", True)  # tracing is tested on its own below
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url=BASE, inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(dg, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt="Hello {{customer.name}}.",
                             version_number=4, content_hash="c" * 64)
    monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kwargs: fake.saved)
    return fake


class GeminiStub:
    """A stub of ``POST /v1beta/models/{model}:generateContent``.

    Each step answers one request: ``("call", skill)`` asks for ``load_skill``,
    ``("fn", name)`` asks for another function, ``"answer"`` ends the turn.
    """

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests: list = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        step = self.steps.pop(0)
        n = len(self.requests)
        if step == "answer":
            parts = [{"text": "done"}]
        else:
            kind, name = step
            fn = "load_skill" if kind == "call" else name
            args = {"name": name} if kind == "call" else {"order_id": "A1001"}
            parts = [{"functionCall": {"name": fn, "args": args, "id": f"fc_{n}"}}]
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            "modelVersion": MODEL})

    def client(self):
        async def handler(request):
            return self(request)

        return genai.Client(api_key="test-key", http_options=types.HttpOptions(
            httpx_client=httpx.Client(transport=httpx.MockTransport(self)),
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))))

    def responses(self, index):
        """The function_response parts in request ``index``."""
        return [p["functionResponse"] for c in self.requests[index].get("contents", [])
                for p in c.get("parts", []) if "functionResponse" in p]


def chat(client, run, *, tools=(lookup_order,)):
    """The documented shape: a chat with automatic function calling."""
    session = client.chats.create(model=MODEL, config=types.GenerateContentConfig(
        system_instruction=run.system, tools=[*tools, *run.tools]))
    return session.send_message(run.message).text


def hand_loop(client, run, *, tools=(LOOKUP_ORDER,)):
    """A loop of your own: run.tool_params, and run.tool_result(call) for each function call."""
    contents = [{"role": "user", "parts": [{"text": run.message}]}]
    config = types.GenerateContentConfig(system_instruction=run.system, tools=[*tools, *run.tool_params])
    for _ in range(4):
        response = client.models.generate_content(model=MODEL, contents=contents, config=config)
        if not response.function_calls:
            return response.text
        contents.append(response.candidates[0].content)
        contents.append({"role": "user", "parts": [
            run.tool_result(call) or {"function_response": {"name": call.name, "response": {"result": "{}"}}}
            for call in response.function_calls]})
    return None


def system_text(request):
    return "".join(p.get("text", "") for p in (request.get("systemInstruction") or {}).get("parts", []))


def declarations(request):
    return [d for tool in request.get("tools", []) for d in tool.get("functionDeclarations", [])]


# ── automatic function calling ───────────────────────────────


def test_a_chat_with_automatic_function_calling_gets_the_prompt_the_list_and_load_skill(platform):
    support = dg.agent_config("support")
    assert type(support).__name__ == "GeminiAgentConfig" and support.name == "support"
    api = GeminiStub(("call", "damaged-item"), "answer")
    with support.run("My keyboard arrived cracked.", variables=SAM) as run:
        assert chat(api.client(), run) == "done"
    first, second = api.requests
    assert system_text(first).startswith("Hello Sam.\n\n## Available Skills")
    assert system_text(first).endswith(LOAD_SKILL_PROMPT_HINT)
    assert [d["name"] for d in declarations(first)] == ["lookup_order", "load_skill"]
    (load_skill,) = [d for d in declarations(first) if d["name"] == "load_skill"]
    schema = (load_skill.get("parameters_json_schema") or load_skill.get("parametersJsonSchema")
              or load_skill.get("parameters"))
    assert schema["required"] == ["name"] and "enum" not in schema["properties"]["name"], \
        "a plain string: automatic function calling cannot pass an argument through an enum"
    assert system_text(second) == system_text(first) and declarations(second) == declarations(first)
    (response,) = api.responses(1)
    assert response["name"] == "load_skill" and response["response"]["result"].startswith("## Skill: damaged-item")
    assert list(platform.queries.values()) == ["My keyboard arrived cracked."]


def test_run_tools_is_a_function_with_the_tools_name_and_description(platform):
    from decimalai.skill_router import LOAD_SKILL_TOOL_DESCRIPTION

    with dg.agent_config("support").run("cracked", variables=SAM) as run:
        (function,) = run.tools
        assert function.__name__ == "load_skill" and function.__doc__ == LOAD_SKILL_TOOL_DESCRIPTION
        assert function(name="damaged-item").startswith("## Skill: damaged-item")
        assert "already loaded" in function(name="damaged-item")


# ── a loop of your own ───────────────────────────────────────


def test_a_hand_written_loop_gets_a_declaration_and_answers_with_function_response_parts(platform):
    api = GeminiStub(("call", "damaged-item"), "answer")
    with dg.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        ((declaration,),) = [tool["function_declarations"] for tool in run.tool_params]
        assert set(declaration) == {"name", "description", "parameters_json_schema"}
        assert "additionalProperties" not in declaration["parameters_json_schema"]
        assert declaration["parameters_json_schema"]["properties"]["name"]["enum"] == \
            ["damaged-item", "order-support"], "your own loop's declaration names the listed skills"
        assert hand_loop(api.client(), run) == "done"
    (response,) = api.responses(1)
    assert response["name"] == "load_skill" and response["id"] == "fc_1"
    assert response["response"]["result"].startswith("## Skill: damaged-item")


def test_another_functions_call_is_the_callers_to_answer(platform):
    api = GeminiStub(("fn", "lookup_order"), "answer")
    with dg.agent_config("support").run("Where is my order?", variables=SAM) as run:
        hand_loop(api.client(), run)
        assert run.tool_result({"name": "lookup_order", "args": {}}) is None
        assert run.tool_result(types.FunctionCall(name="lookup_order", args={})) is None
        assert run.tool_result({"text": "hi"}) is None and run.tool_result(None) is None
    (response,) = api.responses(1)
    assert response == {"name": "lookup_order", "response": {"result": "{}"}}
    assert not [r for r in platform.requests if r[1].endswith("/body")]


def test_a_call_as_a_dict_or_a_part_is_answered_the_same_way(platform):
    with dg.agent_config("support").run("cracked", variables=SAM) as run:
        as_dict = run.tool_result({"name": "load_skill", "args": {"name": "damaged-item"}, "id": "c1"})
        part = types.Part(function_call=types.FunctionCall(name="load_skill", args={"name": "order-support"}))
        from_part = run.tool_result(part)
    assert as_dict["function_response"]["id"] == "c1"
    assert as_dict["function_response"]["response"]["result"].startswith("## Skill: damaged-item")
    assert "id" not in from_part["function_response"], "no id is sent for a call that had none"
    assert from_part["function_response"]["response"]["result"].startswith("## Skill: order-support")


def test_a_name_not_on_the_list_is_answered_as_an_error_without_a_request(platform):
    api = GeminiStub(("call", "invented-skill"), "answer")
    with dg.agent_config("support").run("cracked", variables=SAM) as run:
        hand_loop(api.client(), run)
    (response,) = api.responses(1)
    assert "not one of this agent's skills" in response["response"]["error"]
    assert not [r for r in platform.requests if r[1].endswith("/body")]


def test_a_model_asking_with_other_capitals_gets_the_listed_skill(platform):
    with dg.agent_config("support").run("cracked", variables=SAM) as run:
        result = run.tool_result({"name": "load_skill", "args": {"name": "Damaged-Item"}})
    assert result["function_response"]["response"]["result"].startswith("## Skill: damaged-item")


def test_each_run_is_routed_on_its_own_message_and_routes_once(platform):
    support = dg.agent_config("support")
    for question in ("My keyboard arrived cracked.", "Where is my order?"):
        with support.run(question, variables=SAM) as run:
            chat(GeminiStub(("call", "damaged-item"), "answer").client(), run)
    assert list(platform.queries.values()) == ["My keyboard arrived cracked.", "Where is my order?"]
    assert len(platform.menu_calls()) == 2, "one decision per run, across its function calls"
    bodies = [params for _method, path, params in platform.requests if path.endswith("/body")]
    assert bodies and all(params.get("agent_name") == "support" for params in bodies)


# ── async ─────────────────────────────────────────────────────


def test_an_async_run_gives_an_async_function_to_client_aio(platform):
    support = dg.agent_config("support")
    api = GeminiStub(("call", "damaged-item"), "answer")

    async def main():
        client = api.client()
        async with support.run("My keyboard arrived cracked.", variables=SAM) as run:
            (function,) = run.tools
            assert asyncio.iscoroutinefunction(function)
            response = await client.aio.models.generate_content(
                model=MODEL, contents=run.message,
                config=types.GenerateContentConfig(system_instruction=run.system, tools=[*run.tools]))
            return response.text

    assert asyncio.run(main()) == "done"
    assert api.responses(1)[0]["response"]["result"].startswith("## Skill: damaged-item")


def test_an_async_run_answers_a_call_and_a_direct_load_off_the_loop(platform):
    async def main():
        async with dg.agent_config("support").run("cracked", variables=SAM) as run:
            part = await run.atool_result({"name": "load_skill", "args": {"name": "damaged-item"}, "id": "x"})
            return part, await run.aload_skill("order-support")

    part, text = asyncio.run(main())
    assert part["function_response"]["response"]["result"].startswith("## Skill: damaged-item")
    assert text.startswith("## Skill: order-support")


def test_the_sync_tool_result_inside_an_async_run_says_to_await_it(platform, caplog):
    async def main():
        async with dg.agent_config("support").run("cracked", variables=SAM) as run:
            return run.tool_result({"name": "load_skill", "args": {"name": "damaged-item"}})

    with caplog.at_level(logging.WARNING, logger="decimalai.google_genai"):
        part = asyncio.run(main())
    assert part["function_response"]["response"]["result"].startswith("## Skill: damaged-item")
    assert any("atool_result" in r.getMessage() for r in caplog.records)


# ── prompt, variables, skill list ───────────────────────────


def test_a_missing_variable_stops_the_run_before_any_model_call(platform):
    with pytest.raises(MissingPromptVariable, match="customer.name"):
        with dg.agent_config("support").run("Hi"):
            raise AssertionError("the block must not run")


def test_no_skill_list_means_no_tool(platform):
    platform.skills = []
    with dg.agent_config("support").run("Hi", variables=SAM) as run:
        assert run.system == "Hello Sam."
        assert run.tools == [] and run.tool_params == []


def test_always_include_text_ends_the_system_instruction(platform):
    platform.always = {"order-support": 1}
    with dg.agent_config("support").run("Where is my order?", variables=SAM) as run:
        assert run.system.startswith("Hello Sam.\n\n## Available Skills")
        assert run.system.endswith("## Skill: order-support\n\n# Order support\n\nUse lookup_order for order facts.")


def test_a_cut_list_names_no_enum(platform):
    platform.truncated = True
    with dg.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        ((declaration,),) = [tool["function_declarations"] for tool in run.tool_params]
        assert "enum" not in declaration["parameters_json_schema"]["properties"]["name"]


def test_decimalai_down_sends_the_prompt_alone_and_the_run_still_completes(platform):
    platform.down = True
    api = GeminiStub("answer")
    with dg.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        assert chat(api.client(), run) == "done"
        assert run.system == "Hello Sam." and run.tools == []
    with pytest.raises(RuntimeError):
        with dg.agent_config("support", on_unavailable="fail").run("cracked", variables=SAM):
            raise AssertionError("the block must not run")


def test_the_kill_switch_reaches_configs_built_before_it(platform, monkeypatch):
    import decimalai._config as cfg

    support = dg.agent_config("support")
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    with support.run("cracked", variables=SAM) as run:
        assert run.system == "Hello Sam."
        assert run.tools == [] and run.tool_params == []


def test_with_load_skill_switched_off_the_routed_skills_text_goes_in_the_system_instruction(platform, monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=True,
                                                      load_skill_tool=False))
    with dg.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        assert run.tools == [] and run.tool_params == []
        assert "## Skill: damaged-item" in run.system


# ── conversations ────────────────────────────────────────────


def test_a_conversation_keeps_its_system_and_tools_when_the_prompt_and_list_change(platform):
    support = dg.agent_config("support")
    api = GeminiStub("answer", "answer")
    conversation = support.conversation(variables=SAM)
    for index, question in enumerate(["Where is my order?", "It arrived cracked."]):
        if index == 1:
            support._prompt = SavedPrompt(agent_name="support", system_prompt="Changed, {{customer.name}}.",
                                          version_number=5, content_hash="d" * 64)
            platform.skills = ["damaged-item"]
        with conversation.run(question) as run:
            hand_loop(api.client(), run)
    first, second = api.requests
    assert system_text(first) == system_text(second) and declarations(first) == declarations(second)
    assert list(platform.queries.values()) == ["Where is my order?", "It arrived cracked."]
    with support.run("Hi", variables=SAM) as run:
        assert run.system.startswith("Changed, Sam.")


def test_a_conversation_resumes_from_its_state_in_another_process(platform):
    support = dg.agent_config("support")
    conversation = support.conversation(variables=SAM)
    with conversation.run("Where is my order?") as run:
        system, params = run.system, run.tool_params
        names = run.tools[0].__annotations__["name"]
    state = json.loads(json.dumps(conversation.state))
    assert ConversationRecord.from_state(state, "support") is not None
    later = dg.agent_config("support")
    later._prompt = SavedPrompt(agent_name="support", system_prompt="Changed.", version_number=5,
                                content_hash="d" * 64)
    with later.conversation(state=state).run("It arrived cracked.") as run:
        assert (run.system, run.tool_params) == (system, params)
        assert run.tools[0].__annotations__["name"] == names
    with pytest.raises(ValueError, match="not both"):
        later.conversation(state=state, variables=SAM)


def test_a_cut_list_conversation_shows_new_skills_in_its_user_turn(platform):
    platform.truncated = True
    conversation = dg.agent_config("support").conversation(variables=SAM)
    with conversation.run("Where is my order?") as run:
        system = run.system
        assert "order-support" in system and "damaged-item" not in system
    with conversation.run("It arrived cracked.") as run:
        assert run.system == system
        assert run.message.startswith("It arrived cracked.\n\n") and "damaged-item" in run.message
        part = run.tool_result({"name": "load_skill", "args": {"name": "damaged-item"}})
        assert part["function_response"]["response"]["result"].startswith("## Skill: damaged-item")


# ── misuse ───────────────────────────────────────────────────


def test_system_read_without_tools_warns_at_exit(platform, caplog):
    with caplog.at_level(logging.WARNING, logger="decimalai.google_genai"):
        with dg.agent_config("support").run("Hi", variables=SAM) as run:
            run.system
    assert any("never included it" in r.getMessage() and "run.tool_params" in r.getMessage()
               for r in caplog.records)


def test_run_pieces_outside_the_with_block_say_how_to_use_them(platform):
    run = dg.agent_config("support").run("Hi", variables=SAM)
    for piece in ("system", "message", "tools", "tool_params"):
        with pytest.raises(RuntimeError, match="inside `with support.run"):
            getattr(run, piece)


def test_a_run_opens_once(platform):
    run = dg.agent_config("support").run("Hi", variables=SAM)
    with run:
        pass
    with pytest.raises(RuntimeError, match="opened once"):
        with run:
            pass


def test_the_module_imports_without_the_google_genai_sdk(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "google.genai", None)
    module = importlib.reload(dg)
    try:
        with pytest.raises(ImportError, match=r"decimalai\[google-genai\]"):
            module.agent_config("support")
    finally:
        monkeypatch.delitem(sys.modules, "google.genai")
        importlib.reload(dg)


def test_agent_config_turns_on_gemini_tracing_once(platform, monkeypatch):
    import decimalai.providers as providers

    calls = []
    monkeypatch.setattr(dg, "_tracing_turned_on", False)
    monkeypatch.setattr(providers, "instrument_for_runs", calls.append)
    dg.agent_config("support")
    dg.agent_config("support")
    assert calls == ["google"]


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
    with dg.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        chat(GeminiStub(("call", "damaged-item"), "answer").client(), run)
    (span,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert span.name == otel.RUN_SPAN_NAME and span.attributes["decimal.agent_name"] == "support"
    rail = otel._pop_skill_rail(span.context.trace_id)
    assert rail["routing_id"] and rail["offered"] == ["damaged-item", "order-support"]
    assert rail["loaded"] == ["damaged-item"]
    attributes = otel._decimal_record_attributes(span)
    assert attributes["decimal.prompt.version"] == 4
    assert list(attributes["decimal.prompt.variables"]) == ["customer.name"], "names, never values"


@pytest.fixture
def traced(platform, monkeypatch):
    """DecimalAI's real tracing for agent_config(): the OpenInference instrumentor and the
    exporter, with a provider standing in for the app's global one."""
    pytest.importorskip("openinference.instrumentation.google_genai")
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai.otel as otel
    import decimalai.providers as providers

    GoogleGenAIInstrumentor().uninstrument()
    provider = TracerProvider()
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: provider)
    for name in ("_pipeline_provider", "_last_provider", "_pipeline_exporter"):
        monkeypatch.setattr(providers, name, None)
    for name in ("_instrumented", "_runs_via_another_tool", "_runs_only"):
        monkeypatch.setattr(providers, name, set())
    for name in ("_run_providers", "_explicitly_traced", "_openinference_provider"):
        monkeypatch.setattr(providers, name, {})
    # An earlier test's init(otel=True) / init(crewai=True) leaves its provider here, and an
    # exporter built while one is listed sends every trace, not only runs.
    monkeypatch.setattr(otel, "_installed_providers", [])
    sent: list = []
    monkeypatch.setattr(otel.DecimalSpanExporter, "_send", lambda self, trace: sent.append(trace))
    monkeypatch.setattr(dg, "_tracing_turned_on", False)
    yield provider, sent
    GoogleGenAIInstrumentor().uninstrument()


def test_agent_config_sends_its_runs_and_nothing_else_from_the_process(traced, caplog):
    provider, sent = traced
    support = dg.agent_config("support")
    GeminiStub("answer").client().models.generate_content(model=MODEL, contents="PRIVATE question")
    with provider.get_tracer("app").start_as_current_span("GET /health"):
        pass
    assert sent == [], "nothing outside a run is DecimalAI's to send"
    with caplog.at_level(logging.WARNING, logger="decimalai.google_genai"):
        with support.run("It arrived cracked.", variables=SAM) as run:
            hand_loop(GeminiStub(("call", "damaged-item"), "answer").client(), run)
    (trace,) = sent
    assert trace.agent_name == "support" and len(trace.llm_calls) == 2
    assert not [r for r in caplog.records if "no Gemini request" in r.getMessage()]


def test_a_chat_with_automatic_function_calling_is_one_trace_with_its_calls(traced):
    _provider, sent = traced
    with dg.agent_config("support").run("It arrived cracked.", variables=SAM) as run:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chat(GeminiStub(("call", "damaged-item"), "answer").client(), run)
    (trace,) = sent
    assert trace.agent_name == "support" and trace.llm_calls
    assert trace.skills_loaded_by_agent == ["damaged-item"]
