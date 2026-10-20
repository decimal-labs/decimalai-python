"""`decimalai.pydantic_ai.load_agent()`: an agent's prompt and skills as a Pydantic AI capability.

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
    support = pai.load_agent("support")
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
    support = pai.load_agent("support")
    agent = agent_with(support, Model("answer", "answer"))
    first = agent.run_sync("My keyboard arrived cracked.", deps=SAM)
    agent.run_sync("Where is my order?", deps=SAM, message_history=first.all_messages())
    assert list(platform.queries.values()) == ["My keyboard arrived cracked.", "Where is my order?"]


def test_a_missing_placeholder_says_the_values_come_from_deps(platform):
    support = pai.load_agent("support")
    with pytest.raises(decimalai.MissingPromptVariable, match=r"customer\.name\. Pass them as the run's deps="):
        agent_with(support, Model("answer")).run_sync("Hi", deps={})
    model = Model("answer")
    agent_with(support, model).run_sync("Hi", deps=SimpleNamespace(customer=SimpleNamespace(name="Ann")))
    assert model.requests[0]["instructions"].startswith("Hello Ann.")


def test_without_a_list_the_agent_gets_its_prompt_and_no_load_skill(platform):
    platform.down = True
    support = pai.load_agent("support")
    model = Model("answer")
    agent_with(support, model).run_sync("Hi", deps=SAM)
    assert model.requests[0]["instructions"] == "Hello Sam."
    assert "load_skill" not in model.requests[0]["tools"]


def test_an_agent_with_the_harness_skills_gets_no_decimal_skills_and_is_told_once(platform, monkeypatch, caplog):
    monkeypatch.setattr(pai, "_native_skills", lambda agent: True)
    support = pai.load_agent("support")
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

    support = pai.load_agent("support")
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    model = Model("answer")
    agent_with(support, model).run_sync("Hi", deps=SAM)
    assert model.requests[0]["instructions"] == "Hello Sam."
    assert platform.menu_calls() == []


def test_a_run_is_one_trace_named_for_the_decimalai_agent_with_its_record(platform, monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: provider)
    rails: dict = {}
    real_record = otel.record_skill_rail

    def record(**kwargs):
        rails.setdefault(otel.current_run_key(), []).append(kwargs)
        return real_record(**kwargs)

    monkeypatch.setattr(otel, "record_skill_rail", record)
    support = pai.load_agent("support")
    # An Agent with no name of its own: the trace is still the DecimalAI agent's.
    Agent(FunctionModel(Model(("call", "damaged-item"), "answer")), capabilities=[support],
          deps_type=dict).run_sync("My keyboard arrived cracked.", deps=SAM)
    (span,) = exporter.get_finished_spans()
    assert span.name == otel.RUN_SPAN_NAME and span.attributes["decimal.agent_name"] == "support"
    (run_key,) = rails
    assert run_key == span.context.trace_id
    rail = otel._pop_skill_rail(run_key)
    assert rail["routing_id"] and rail["offered"] == ["damaged-item", "order-support"]
    assert rail["loaded"] == ["damaged-item"]
    assert otel._decimal_record_attributes(span)["decimal.prompt.version"] == 4


def test_two_configs_never_compare_equal(platform):
    first, second = pai.load_agent("support"), pai.load_agent("support")
    assert first != second and first == first
    assert len({first, second}) == 2


def test_the_load_names_the_decimalai_agent_and_the_run_routes_once(platform):
    support = pai.load_agent("support")
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

    support = pai.load_agent("support")
    with pytest.raises(UserError, match="load_skill"):
        agent_with(support, Model("answer"), tools=[load_skill]).run_sync("Hi", deps=SAM)


def test_the_callers_own_instructions_stay_a_leading_unedited_prefix(platform):
    support = pai.load_agent("support")
    model = Model("answer", "answer")
    agent = agent_with(support, model, instructions="CALLER RULES.")
    agent.run_sync("My keyboard arrived cracked.", deps=SAM)
    agent.run_sync("Where is my order?", deps=SAM)
    first, second = (request["instructions"] for request in model.requests)
    assert first.startswith("CALLER RULES.") and second.startswith("CALLER RULES.")
    assert "Hello Sam.\n\n## Available Skills" in first
