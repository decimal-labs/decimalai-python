"""`decimalai.pydantic_ai.agent_config()`: an agent's prompt and skills as a Pydantic AI capability.

Driven through the real Pydantic AI with its offline ``FunctionModel``: the model
records what each request was given (instructions, tools, their schemas) and calls
``load_skill`` when told to. DecimalAI's answers come from the fake platform the
framework-neutral tests use, so nothing leaves the process.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart  # noqa: E402
from pydantic_ai.models.function import AgentInfo, FunctionModel  # noqa: E402

import decimalai  # noqa: E402
import decimalai._agent as agent_mod  # noqa: E402
import decimalai._agent_binding as binding  # noqa: E402
import decimalai.pydantic_ai as pai  # noqa: E402
from decimalai._agent import SavedPrompt  # noqa: E402
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter  # noqa: E402
from tests.test_agent_binding import FakePlatform  # noqa: E402

BASE = "http://pydantic-ai-load-agent.test"


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=True))
    monkeypatch.setattr(cfg, "_client", MagicMock())
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    # Tracing is tested on its own below (`traced`): here no pipeline, so no run is instrumented.
    monkeypatch.setattr(pai, "_tracing_turned_on", True)
    monkeypatch.setattr(pai, "_run_tracer_provider", None)
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url=BASE, inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(pai, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt="Hello {{customer.name}}.",
                             version_number=4, content_hash="c" * 64)
    monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kwargs: fake.saved)
    return fake


class Model:
    """A FunctionModel script: ("call", skill) loads a skill, "answer" ends the run."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests: list = []

    def __call__(self, messages, info: AgentInfo) -> ModelResponse:
        self.requests.append({
            "instructions": info.instructions,
            "tools": {t.name: t.parameters_json_schema for t in info.function_tools},
            "returns": [p.content for m in messages for p in getattr(m, "parts", [])
                        if isinstance(p, ToolReturnPart)],
        })
        step = self.steps.pop(0)
        if step == "answer":
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(tool_name="load_skill", args={"name": step[1]})])


def agent_with(support, model, **kwargs):
    return Agent(FunctionModel(model), name=support.name, capabilities=[support],
                 deps_type=dict, **kwargs)


SAM = {"customer": {"name": "Sam"}}


def test_the_capability_gives_the_agent_its_prompt_skill_list_and_load_skill(platform):
    support = pai.agent_config("support")
    assert type(support).__name__ == "PydanticAIAgentConfig" and support.name == "support"
    model = Model(("call", "damaged-item"), "answer")
    result = agent_with(support, model).run_sync("My keyboard arrived cracked.", deps=SAM)
    assert result.output == "done"
    first, second = model.requests
    assert first["instructions"].startswith("Hello Sam.\n\n## Available Skills")
    assert first["instructions"].endswith(LOAD_SKILL_PROMPT_HINT)
    assert first["tools"]["load_skill"]["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    assert second["returns"][0].startswith("## Skill: damaged-item")
    assert list(platform.queries.values()) == ["My keyboard arrived cracked."]


def test_each_run_is_routed_on_its_own_prompt(platform):
    support = pai.agent_config("support")
    agent = agent_with(support, Model("answer", "answer"))
    first = agent.run_sync("My keyboard arrived cracked.", deps=SAM)
    agent.run_sync("Where is my order?", deps=SAM, message_history=first.all_messages())
    assert list(platform.queries.values()) == ["My keyboard arrived cracked.", "Where is my order?"]


def test_a_missing_placeholder_says_the_values_come_from_deps(platform):
    support = pai.agent_config("support")
    with pytest.raises(decimalai.MissingPromptVariable, match=r"customer\.name\. Pass them as the run's deps="):
        agent_with(support, Model("answer")).run_sync("Hi", deps={})
    model = Model("answer")
    agent_with(support, model).run_sync("Hi", deps=SimpleNamespace(customer=SimpleNamespace(name="Ann")))
    assert model.requests[0]["instructions"].startswith("Hello Ann.")


def test_without_a_list_the_agent_gets_its_prompt_and_no_load_skill(platform):
    platform.down = True
    support = pai.agent_config("support")
    model = Model("answer")
    agent_with(support, model).run_sync("Hi", deps=SAM)
    assert model.requests[0]["instructions"] == "Hello Sam."
    assert "load_skill" not in model.requests[0]["tools"]


def test_an_agent_with_the_harness_skills_gets_no_decimal_skills_and_is_told_once(platform, monkeypatch, caplog):
    monkeypatch.setattr(pai, "_native_skills", lambda agent: True)
    support = pai.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai.pydantic_ai"):
        for _ in range(2):
            model = Model("answer")
            agent_with(support, model).run_sync("Hi", deps=SAM)
            assert model.requests[0]["instructions"] == "Hello Sam."
            assert "load_skill" not in model.requests[0]["tools"]
    assert sum("harness's Skills" in r.getMessage() for r in caplog.records) == 1
    assert platform.menu_calls() == []


def test_init_enabled_false_stops_skills_but_not_the_prompt(platform, monkeypatch):
    import decimalai._config as cfg

    support = pai.agent_config("support")
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    model = Model("answer")
    agent_with(support, model).run_sync("Hi", deps=SAM)
    assert model.requests[0]["instructions"] == "Hello Sam."
    assert platform.menu_calls() == []


# ── tracing: agent_config() alone ───────────────────────────


@pytest.fixture
def traced(platform, monkeypatch):
    """DecimalAI's real tracing for agent_config(): the exporter it puts on the global tracer
    provider, here a fresh one standing in for the app's. Yields (that provider, the traces
    sent, the spans the provider saw)."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel
    import decimalai.providers as providers

    provider = TracerProvider()
    spans = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: provider)
    for name in ("_pipeline_provider", "_last_provider", "_pipeline_exporter"):
        monkeypatch.setattr(providers, name, None)
    for name in ("_instrumented", "_runs_via_another_tool", "_runs_only"):
        monkeypatch.setattr(providers, name, set())
    for name in ("_run_providers", "_explicitly_traced", "_openinference_provider"):
        monkeypatch.setattr(providers, name, {})
    # An earlier test's init(otel=True) leaves its provider here, and an exporter built while
    # one is listed sends every trace, not only runs.
    monkeypatch.setattr(otel, "_installed_providers", [])
    sent: list = []
    monkeypatch.setattr(otel.DecimalSpanExporter, "_send", lambda self, trace: sent.append(trace))
    monkeypatch.setattr(pai, "_tracing_turned_on", False)
    monkeypatch.setattr(pai, "_instrumentation_settings", None)
    yield provider, sent, spans


def _names(spans) -> dict:
    """span name -> its parent's name (None for a root)."""
    by_id = {span.context.span_id: span.name for span in spans}
    return {span.name: (by_id.get(span.parent.span_id) if span.parent else None) for span in spans}


def test_agent_config_alone_traces_a_run_with_pydantic_ais_own_spans(traced):
    """No init(otel=True) and no Agent.instrument_all(): the run is one trace, rooted in
    DecimalAI's run span, carrying Pydantic AI's spans for its two model calls and its tool
    call, filed under the DecimalAI agent with its record."""
    import decimalai.otel as otel

    _provider, sent, spans = traced
    support = pai.agent_config("support")
    # An Agent with no name of its own: the trace is still the DecimalAI agent's.
    Agent(FunctionModel(Model(("call", "damaged-item"), "answer")), capabilities=[support],
          deps_type=dict).run_sync("My keyboard arrived cracked.", deps=SAM)
    (trace,) = sent
    assert trace.agent_name == "support" and len(trace.llm_calls) == 2
    assert trace.skills_loaded_by_agent == ["damaged-item"]
    assert trace.skills_offered_in_prompt == ["damaged-item", "order-support"] and trace.routing_id
    tree = _names(spans.get_finished_spans())
    assert tree[otel.RUN_SPAN_NAME] is None, "DecimalAI's run span is the trace's root"
    assert tree["invoke_agent agent"] == otel.RUN_SPAN_NAME
    assert tree["execute_tool load_skill"] == "invoke_agent agent"
    assert tree["chat function:Model:"] == "invoke_agent agent"
    run_span = next(s for s in spans.get_finished_spans() if s.name == otel.RUN_SPAN_NAME)
    assert run_span.attributes["decimal.agent_name"] == "support"
    assert any(otel._decimal_record_attributes(s).get("decimal.prompt.version") == 4
               for s in spans.get_finished_spans())


def test_a_streamed_run_is_one_trace_too(traced):
    """run_stream(): the capability's run span still wraps the whole run, streamed model call
    included, so the trace is one, rooted in it."""
    import asyncio

    import decimalai.otel as otel

    _provider, sent, spans = traced

    async def stream(messages, info):
        yield "streamed "
        yield "answer"

    support = pai.agent_config("support")
    agent = Agent(FunctionModel(stream_function=stream), name=support.name, capabilities=[support],
                  deps_type=dict)

    async def go():
        async with agent.run_stream("Hi", deps=SAM) as result:
            return await result.get_output()

    assert asyncio.run(go()) == "streamed answer"
    (trace,) = sent
    assert trace.agent_name == "support" and len(trace.llm_calls) == 1
    finished = spans.get_finished_spans()
    assert _names(finished)[otel.RUN_SPAN_NAME] is None
    root = next(s for s in finished if s.name == otel.RUN_SPAN_NAME)
    assert all(s.end_time <= root.end_time for s in finished), "nothing in the run ends after its root"


def test_only_its_runs_are_instrumented_and_sent(traced):
    """Another Pydantic AI agent in the process is not instrumented (no global switch), and
    the app's own spans on the shared provider stay in the process."""
    provider, sent, spans = traced
    support = pai.agent_config("support")
    Agent(FunctionModel(Model("answer"))).run_sync("Hi")  # an agent without the config
    with provider.get_tracer("app").start_as_current_span("GET /health"):
        pass
    assert [s.name for s in spans.get_finished_spans()] == ["GET /health"]
    assert sent == []
    agent_with(support, Model("answer")).run_sync("Hi", deps=SAM)
    (trace,) = sent
    assert trace.agent_name == "support"


def test_a_run_is_a_trace_of_its_own_linked_to_the_span_open_when_it_started(traced):
    """Inside the app's request span the run is still its own trace (DecimalAI's exporter
    finalizes a trace at its root), linked to that span."""
    import decimalai.otel as otel

    provider, sent, spans = traced
    support = pai.agent_config("support")
    with provider.get_tracer("app").start_as_current_span("POST /chat") as request:
        agent_with(support, Model("answer")).run_sync("Hi", deps=SAM)
    (trace,) = sent
    (run_span,) = [s for s in spans.get_finished_spans() if s.name == otel.RUN_SPAN_NAME]
    assert run_span.parent is None
    assert [link.context.span_id for link in run_span.links] == [request.get_span_context().span_id]


def test_a_conversations_runs_share_its_session(traced):
    _provider, sent, _spans = traced
    support = pai.agent_config("support")
    agent = agent_with(support, Model("answer", "answer"))
    first = agent.run_sync("My keyboard arrived cracked.", deps=SAM)
    agent.run_sync("Where is my order?", deps=SAM, message_history=first.all_messages())
    one, two = sent
    assert one.session_id and one.session_id == two.session_id == first.conversation_id


def test_a_later_runs_late_routing_decision_lands_on_its_own_trace(platform, traced, monkeypatch):
    """A later run shows the local copy at once and routes in the background. Its trace waits
    for that decision (off the run's path, as the Anthropic and OpenAI runs do), so the
    decision is on the run's own trace instead of an orphan record."""
    import contextvars
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from decimalai import _provider_run

    _provider, sent, _spans = traced
    support = pai.agent_config("support")
    agent = agent_with(support, Model("answer", "answer"))
    agent.run_sync("Where is my order?", deps=SAM)  # the first run routes before its model call
    landed = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)

    def late(fn, *args):
        def decide():
            landed.wait(5)
            return fn(*args)

        return pool.submit(contextvars.copy_context().run, decide)

    monkeypatch.setattr(binding, "_in_background", late)
    agent.run_sync("It arrived cracked.", deps=SAM)
    assert len(sent) == 1, "the run returned; its trace waits for the decision"
    landed.set()
    _provider_run._drain_finishing(timeout=5)
    first, second = sent
    assert second.routing_id and second.routing_id != first.routing_id
    assert platform.queries[second.routing_id] == "It arrived cracked."


def test_a_run_the_app_instruments_keeps_that_instrumentation_inside_decimalais_span(traced):
    """`Agent.instrument_all()` / `Agent(instrument=...)` / Logfire: the app's Instrumentation
    traces the run, once, inside DecimalAI's run span, and keeps Pydantic AI's own name."""
    from pydantic_ai import InstrumentationSettings

    import decimalai.otel as otel

    provider, sent, spans = traced
    support = pai.agent_config("support")
    agent = Agent(FunctionModel(Model("answer")), name="pydantic-agent", capabilities=[support],
                  deps_type=dict)
    # The per-agent form of Agent.instrument_all(), which would leave a global switch on.
    agent.instrument = InstrumentationSettings(tracer_provider=provider)
    agent.run_sync("Hi", deps=SAM)
    finished = spans.get_finished_spans()
    (framework_run,) = [s for s in finished if s.attributes.get("gen_ai.operation.name") == "invoke_agent"
                        and s.name != otel.RUN_SPAN_NAME]
    assert framework_run.attributes["gen_ai.agent.name"] == "pydantic-agent", "Pydantic AI's own name stays"
    assert _names(finished)[framework_run.name] == otel.RUN_SPAN_NAME
    (trace,) = sent
    assert trace.agent_name == "support" and len(trace.llm_calls) == 1


def test_a_run_whose_model_sdk_decimalai_traces_is_not_instrumented_again(traced, monkeypatch):
    """init(openai=True) beside agent_config(): OpenInference records the model calls, so
    Pydantic AI's spans would report each twice."""
    import decimalai.otel as otel
    import decimalai.providers as providers

    _provider, sent, spans = traced
    monkeypatch.setattr(pai, "_model_sdks", lambda model: {"openai"})
    monkeypatch.setattr(providers, "traces_calls_of", lambda name: name == "openai")
    support = pai.agent_config("support")
    agent_with(support, Model("answer")).run_sync("Hi", deps=SAM)
    assert [s.name for s in spans.get_finished_spans()] == [otel.RUN_SPAN_NAME]
    assert len(sent) == 1


def test_agent_config_turns_on_tracing_once_and_keyless_runs_are_not_traced(traced, monkeypatch):
    import decimalai.providers as providers

    _provider, sent, spans = traced
    calls = []
    real = providers.pipeline_for_runs
    monkeypatch.setattr(providers, "pipeline_for_runs", lambda: calls.append(1) or real())
    pai.agent_config("support")
    pai.agent_config("support")
    assert calls == [1]

    monkeypatch.setattr(pai, "_tracing_turned_on", False)
    monkeypatch.setattr(agent_mod, "_open_prompt", lambda name, fallback: SavedPrompt(
        agent_name=name, system_prompt=None, keyless=True))
    keyless = pai.agent_config("support")
    assert calls == [1] and pai._tracing_turned_on is False
    spans.clear()
    agent_with(keyless, Model("answer")).run_sync("Hi", deps=SAM)
    assert spans.get_finished_spans() == () and sent == []


def test_init_enabled_false_stops_the_runs_tracing(traced, monkeypatch):
    import decimalai._config as cfg

    _provider, sent, spans = traced
    support = pai.agent_config("support")
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    agent_with(support, Model("answer")).run_sync("Hi", deps=SAM)
    assert spans.get_finished_spans() == () and sent == []


def test_instructions_keep_the_developers_order_of_capabilities(platform):
    """The run span is outermost; the prompt and list stay where the developer listed them."""
    from dataclasses import dataclass

    from pydantic_ai.capabilities import AbstractCapability

    @dataclass
    class Note(AbstractCapability):
        text: str = ""

        def get_instructions(self):
            # A function, as the config's are: Pydantic AI puts plain strings first.
            async def note(ctx):
                return self.text

            return note

    support = pai.agent_config("support")
    model = Model("answer")
    Agent(FunctionModel(model), instructions="CALLER RULES.", deps_type=dict,
          capabilities=[Note(text="FIRST."), support, Note(text="LAST.")]).run_sync("Hi", deps=SAM)
    text = model.requests[0]["instructions"]
    assert text.startswith("CALLER RULES.\n\nFIRST.\n\nHello Sam.") and text.endswith("LAST.")


def test_two_configs_never_compare_equal(platform):
    first, second = pai.agent_config("support"), pai.agent_config("support")
    assert first != second and first == first
    assert len({first, second}) == 2


def test_the_load_names_the_decimalai_agent_and_the_run_routes_once(platform):
    support = pai.agent_config("support")
    Agent(FunctionModel(Model(("call", "damaged-item"), "answer")), capabilities=[support],
          deps_type=dict).run_sync("My keyboard arrived cracked.", deps=SAM)
    (body,) = [params for _method, path, params in platform.requests if path.endswith("/body")]
    assert body.get("agent_name") == "support"
    assert len(platform.menu_calls()) == 1, "one decision per run, across its tool steps"


def test_another_tool_named_load_skill_fails_the_run_loudly(platform):
    """Pydantic AI refuses two tools with one name; the run says so instead of
    serving one of them silently."""
    from pydantic_ai.exceptions import UserError

    def load_skill(name: str) -> str:
        return "theirs"

    support = pai.agent_config("support")
    with pytest.raises(UserError, match="load_skill"):
        agent_with(support, Model("answer"), tools=[load_skill]).run_sync("Hi", deps=SAM)


def test_the_callers_own_instructions_stay_a_leading_unedited_prefix(platform):
    support = pai.agent_config("support")
    model = Model("answer", "answer")
    agent = agent_with(support, model, instructions="CALLER RULES.")
    agent.run_sync("My keyboard arrived cracked.", deps=SAM)
    agent.run_sync("Where is my order?", deps=SAM)
    first, second = (request["instructions"] for request in model.requests)
    assert first.startswith("CALLER RULES.") and second.startswith("CALLER RULES.")
    assert "Hello Sam.\n\n## Available Skills" in first


# ── conversations on Claude ─────────────────────────────────


def claude_agent(support, model, model_name="claude-haiku-5-5"):
    return Agent(FunctionModel(model, model_name=model_name), name=support.name,
                 capabilities=[support], deps_type=dict)


def save_prompt(platform, support, text, version):
    platform.saved = SavedPrompt(agent_name="support", system_prompt=text, version_number=version,
                                 content_hash=str(version) * 64)
    support._refresh_prompt()


def test_on_claude_a_conversation_keeps_its_instructions_and_load_skill(platform):
    support = pai.agent_config("support")
    model = Model("answer", ("call", "order-support"), "answer", "answer")
    agent = claude_agent(support, model)
    first = agent.run_sync("My keyboard arrived cracked.", deps=SAM)
    save_prompt(platform, support, "Hi {{customer.name}}, be brief.", 5)
    platform.skills = ["damaged-item"]
    support._route_run(None, [], "anything")  # the local copy follows

    second = agent.run_sync("Where is my order?", deps=SAM, message_history=first.all_messages())
    one, two, three = model.requests[:3]
    assert two["instructions"] == three["instructions"] == one["instructions"], \
        "Claude rejects a conversation whose system prompt changed"
    assert two["tools"] == one["tools"] and one["tools"]["load_skill"]["properties"]["name"]["enum"] == [
        "damaged-item", "order-support"]
    assert three["returns"][-1].startswith("## Skill: order-support")
    assert list(platform.queries.values())[-1] == "Where is my order?", "still routed, for the record"

    agent.run_sync("Hello", deps=SAM)  # a new conversation
    assert model.requests[3]["instructions"].startswith("Hi Sam, be brief.")
    assert model.requests[3]["tools"]["load_skill"]["properties"]["name"]["enum"] == ["damaged-item"]
    assert second.all_messages()[0].metadata["decimalai.conversation"]["support"]["prompt"]["version"] == 4


def test_on_claude_the_record_travels_with_the_serialized_history(platform):
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    support = pai.agent_config("support")
    model = Model("answer", "answer")
    first = claude_agent(support, model).run_sync("My keyboard arrived cracked.", deps=SAM)
    stored = ModelMessagesTypeAdapter.dump_json(first.all_messages())
    save_prompt(platform, support, "Hi {{customer.name}}.", 5)

    elsewhere = pai.agent_config("support")  # another worker, which read version 5
    claude_agent(elsewhere, model).run_sync(
        "Where is my order?", deps=SAM, message_history=ModelMessagesTypeAdapter.validate_json(stored))
    assert model.requests[1]["instructions"] == model.requests[0]["instructions"]


def test_on_other_models_each_run_shows_the_newest_prompt_as_before(platform):
    support = pai.agent_config("support")
    model = Model("answer", "answer")
    agent = claude_agent(support, model, model_name="gpt-5.4-mini")
    first = agent.run_sync("My keyboard arrived cracked.", deps=SAM)
    assert "decimalai.conversation" not in (first.all_messages()[0].metadata or {})
    save_prompt(platform, support, "Hi {{customer.name}}.", 5)
    agent.run_sync("Where is my order?", deps=SAM, message_history=first.all_messages())
    assert model.requests[1]["instructions"].startswith("Hi Sam.")
