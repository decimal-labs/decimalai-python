"""`decimalai.openai_agents.load_agent`: an agent's prompt and skills, attached explicitly.

Real OpenAI Agents SDK runner and the real SkillRouter; DecimalAI's HTTP answers
and the model are scripted. Nothing here may change an Agents SDK class.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

agents = pytest.importorskip("agents")

from agents import Agent, Model, ModelResponse, RunConfig, Runner, Usage, function_tool  # noqa: E402
from agents.tracing import get_trace_provider, set_trace_processors  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

import decimalai.openai_agents as oa  # noqa: E402
import decimalai.skill_router as sr  # noqa: E402
from decimalai._client import AgentNotFoundError  # noqa: E402
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter  # noqa: E402

BASE = "You help customers with orders. Treat tool data as facts."
DESCRIPTIONS = {"damaged-item": "Handle items that arrive broken.", "order-support": "Order status and tracking."}


def stable_menu(names):
    rows = "".join(f"| {n} | {DESCRIPTIONS[n]} |\n" for n in names)
    return f"## Available Skills\n\n| Skill | Description |\n|-------|-------------|\n{rows}" if names else ""


STABLE_MENU = stable_menu(["damaged-item", "order-support"])
MENU_INSTRUCTION = "When a skill above applies, follow its instructions and then answer the user directly."
PREFIX = f"{STABLE_MENU.rstrip()}\n\n{MENU_INSTRUCTION}"
BODIES = {
    "damaged-item": "# Damaged item\n\nAsk for photos of the item and packaging.",
    "order-support": "# Order support\n\nUse lookup_order for order facts.",
}
QUESTION = "My keyboard from ORD-1002 arrived cracked."


@function_tool
def lookup_order(order_id: str) -> str:
    """Look up an order."""
    return json.dumps({"order_id": order_id, "status": "delivered"})


class FakeDecimal:
    """DecimalAI's answers: the prompt (via the client) and the router's requests."""

    def __init__(self):
        self.prompt = {"system_prompt": BASE, "version_number": 2, "content_hash": "c" * 64}
        self.skills = ["damaged-item", "order-support"]
        self.menu_down = False
        self.requests: list = []
        self._next = 0

    def prompt_payload(self, agent_name, version=None):
        if agent_name == "missing-agent":
            import httpx

            raise AgentNotFoundError(
                httpx.Response(404, json={"detail": "Agent not found"},
                               request=httpx.Request("GET", "http://127.0.0.1:9/api/v1/agents/x/prompt")),
                agent_name=agent_name,
            )
        return {"agent_name": agent_name, "agent_id": "a-1", "resolved_from": None,
                "version_id": "v-1", "label": None, "provenance": None, "created_at": None,
                "updated_at": None, "version_mode": "latest", "pinned_version_number": None,
                **self.prompt}

    def routing_id(self):
        self._next += 1
        return f"rt_{self._next:024x}"

    def request(self, method, path, json=None, params=None, **kwargs):
        self.requests.append((method, path, dict(params or {})))
        if path == "/api/v1/skills/menu":
            if self.menu_down:
                raise sr.SkillRouterError("transport error: timed out")
            return {
                "skills": [{"name": n, "description": n} for n in self.skills],
                "stable_menu": stable_menu(self.skills),
                "menu_instruction": MENU_INSTRUCTION if self.skills else "",
                "routing_hint": "", "stable_menu_skills": list(self.skills),
                "routing_id": self.routing_id(), "strategy": "full_menu", "query_ranked": False,
            }
        if path.startswith("/api/v1/skills/") and path.endswith("/body"):
            name = path.split("/")[-2]
            body = BODIES[name]
            return {"body": body, "content_hash": hashlib.sha256(body.encode()).hexdigest(),
                    "skill_id": f"11111111-1111-4111-8111-{len(name):012d}",
                    "version_id": "22222222-2222-4222-8222-222222222222",
                    "content_hash_matches_body": True}
        raise AssertionError(f"unexpected DecimalAI request {method} {path}")

    def menu_calls(self):
        return [r for r in self.requests if r[1] == "/api/v1/skills/menu"]

    def body_calls(self):
        return [r for r in self.requests if r[1].endswith("/body")]


class ScriptedModel(Model):
    """Plays steps in order: ("call", name, args) or "answer"."""

    def __init__(self, steps):
        self.steps, self.instructions, self.tools, self.inputs = list(steps), [], [], []

    async def get_response(self, system_instructions, input, model_settings, tools, *args, **kwargs):
        self.instructions.append(system_instructions)
        self.tools.append({t.name: t for t in tools})
        self.inputs.append(list(input) if isinstance(input, list) else input)
        step = self.steps.pop(0)
        if step == "answer":
            text = "done"
            output = [ResponseOutputMessage(
                id="msg", type="message", role="assistant", status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])])]
        else:
            _, name, arguments = step
            n = len(self.instructions)
            output = [ResponseFunctionToolCall(
                id=f"fc{n}", type="function_call", call_id=f"call{n}", name=name,
                arguments=json.dumps(arguments))]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    async def stream_response(self, *args, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover


ORIGINALS = {}


@pytest.fixture
def decimal(monkeypatch):
    """A configured SDK with a mock client, a router answering from FakeDecimal,
    background work run inline, and only DecimalAI's trace processor registered."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    from agents.run_internal import turn_resolution

    ORIGINALS.update({
        "Agent.__init__": Agent.__init__, "Agent.get_system_prompt": Agent.get_system_prompt,
        "Agent.get_all_tools": Agent.get_all_tools,
        "check_for_final_output_from_tools": turn_resolution.check_for_final_output_from_tools,
    })
    fake = FakeDecimal()
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True))
    client = MagicMock()
    client.get_agent_prompt.side_effect = fake.prompt_payload
    client.register_manifest.return_value = {"manifest_id": "m-1", "status": "active"}
    monkeypatch.setattr(cfg, "_client", client)
    router = SkillRouter(api_key="dai_sk_test", base_url="http://127.0.0.1:9", inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(oa, "_skill_router_singleton", router)
    monkeypatch.setattr(oa, "_tracing_processor", None, raising=False)
    monkeypatch.setattr(oa, "_tracing_processor_added", False, raising=False)
    monkeypatch.setattr(oa, "_run_rails", OrderedDict())
    monkeypatch.setattr(oa, "_in_background", lambda fn, *args: fn(*args))
    provider = get_trace_provider()
    saved = list(provider._multi_processor._processors)
    set_trace_processors([])
    fake.client = client
    try:
        yield fake
    finally:
        set_trace_processors(saved)


def sent(fake) -> list:
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in fake.client.ingest_trace.call_args_list]


def build(support, model, **kwargs):
    return Agent(name=support.name, instructions=support.instructions,
                 tools=[lookup_order, *support.tools], model=model, **kwargs)


LOAD = ("call", "load_skill", {"name": "damaged-item"})
LOOKUP = ("call", "lookup_order", {"order_id": "ORD-1002"})


def test_one_run_gets_prompt_skill_list_tool_and_a_full_record(decimal):
    support = oa.load_agent("support")
    model = ScriptedModel([LOAD, LOOKUP, "answer"])
    Runner.run_sync(build(support, model), QUESTION)

    assert model.instructions[0] == f"{BASE}\n\n{PREFIX}\n{LOAD_SKILL_PROMPT_HINT}"
    assert all(text == model.instructions[0] for text in model.instructions)
    tool = model.tools[0]["load_skill"]
    assert tool.params_json_schema["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    assert any(BODIES["damaged-item"] in str(item.get("output", ""))
               for item in model.inputs[1] if isinstance(item, dict))
    assert len(decimal.menu_calls()) == 1, "the startup decision is the first run's"

    [trace] = sent(decimal)
    assert trace.agent_name == "support"
    assert trace.routing_id == "rt_" + f"{1:024x}"
    assert trace.skills_offered_in_prompt == ["damaged-item", "order-support"]
    assert trace.skills_loaded_by_agent == ["damaged-item"]
    digest = hashlib.sha256(BODIES["damaged-item"].encode()).hexdigest()
    assert any(v["name"] == "damaged-item" and v["hash"] == digest for v in trace.skills_delivered_versions)
    span = next(s for s in trace.spans if s.name == "support" and s.attributes.get("decimal.prompt.version"))
    assert span.attributes["decimal.prompt.version"] == 2
    assert span.attributes["decimal.prompt.source"] == "decimal"
    assert span.attributes["decimal.prompt.content_hash"] == "c" * 64


def test_each_later_run_records_its_own_routing_decision(decimal):
    support = oa.load_agent("support")
    for _ in range(3):
        Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    traces = sent(decimal)
    assert [t.routing_id for t in traces] == ["rt_" + f"{n:024x}" for n in (1, 2, 3)]
    assert len(decimal.menu_calls()) == 3


def test_a_decision_offering_a_different_list_is_not_stamped(decimal):
    support = oa.load_agent("support")
    Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    decimal.skills = ["order-support"]  # unassigned between the run's read and its decision
    Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    second = sent(decimal)[1]
    assert second.routing_id is None
    assert second.skills_offered_in_prompt == ["damaged-item", "order-support"]
    model = ScriptedModel(["answer"])
    Runner.run_sync(build(support, model), "Hi")
    assert "| damaged-item |" not in model.instructions[0], "later runs use the refreshed list"
    assert model.tools[0]["load_skill"].params_json_schema["properties"]["name"]["enum"] == ["order-support"]


@dataclass
class Customer:
    name: str
    tier: str


@dataclass
class OrderContext:
    store: str
    customer: Customer


def test_template_is_filled_from_the_run_context_and_only_names_are_recorded(decimal):
    decimal.prompt = {"system_prompt": "Help customers of {{store}}.\n\n## Customer\n- {{customer.name}} ({{customer.tier}})",
                      "version_number": 4, "content_hash": "d" * 64}
    support = oa.load_agent("support")
    assert support.variables == ["customer.name", "customer.tier", "store"]
    model = ScriptedModel(["answer"])
    Runner.run_sync(build(support, model), "Hi", context=OrderContext("Demo Store", Customer("Ana", "gold")))
    assert model.instructions[0].startswith("Help customers of Demo Store.\n\n## Customer\n- Ana (gold)\n\n## Available Skills")
    [trace] = sent(decimal)
    span = next(s for s in trace.spans if s.attributes.get("decimal.prompt.version") == 4)
    assert span.attributes["decimal.prompt.variables"] == ["customer.name", "customer.tier", "store"]
    assert "Ana" not in json.dumps(span.attributes)


def test_a_missing_template_value_stops_the_run_before_any_model_call(decimal):
    decimal.prompt = {"system_prompt": "Help customers of {{store}}.", "version_number": 4, "content_hash": "d" * 64}
    support = oa.load_agent("support")
    model = ScriptedModel(["answer"])
    with pytest.raises(oa.MissingPromptVariable, match="store"):
        Runner.run_sync(build(support, model), "Hi", context={"other": 1})
    assert model.instructions == []


def test_no_saved_prompt_uses_the_fallback_and_says_so(decimal):
    decimal.prompt = {"system_prompt": None, "version_number": None, "content_hash": None}
    support = oa.load_agent("support", fallback=BASE)
    model = ScriptedModel(["answer"])
    Runner.run_sync(build(support, model), "Hi")
    assert model.instructions[0].startswith(BASE)
    [trace] = sent(decimal)
    span = next(s for s in trace.spans if s.attributes.get("decimal.prompt.source"))
    assert span.attributes["decimal.prompt.source"] == "code"


def test_an_unknown_agent_fails_at_load_agent(decimal):
    with pytest.raises(AgentNotFoundError):
        oa.load_agent("missing-agent", fallback=BASE)


def test_skills_unavailable_continue_or_fail(decimal):
    decimal.menu_down = True
    support = oa.load_agent("support")
    model = ScriptedModel(["answer"])
    Runner.run_sync(build(support, model), QUESTION)
    assert model.instructions[0] == BASE
    assert "load_skill" not in model.tools[0], "no skills, so no loader"
    [trace] = sent(decimal)
    span = next(s for s in trace.spans if s.attributes.get("decimal.prompt.source"))
    assert "did not return" in span.attributes["decimal.skills.unavailable"]

    strict = oa.load_agent("support", on_unavailable="fail")
    with pytest.raises(RuntimeError, match="unavailable"):
        Runner.run_sync(build(strict, ScriptedModel(["answer"])), QUESTION)


def test_a_skill_loaded_twice_in_one_run_is_sent_once(decimal):
    support = oa.load_agent("support")
    model = ScriptedModel([LOAD, LOAD, "answer"])
    Runner.run_sync(build(support, model), QUESTION)
    assert len(decimal.body_calls()) == 1
    assert "already loaded" in str(model.inputs[2][-1].get("output"))


def test_stop_on_first_tool_agents_get_no_loader_and_a_warning(decimal, caplog):
    support = oa.load_agent("support")
    model = ScriptedModel([LOOKUP])
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        result = Runner.run_sync(build(support, model, tool_use_behavior="stop_on_first_tool"), QUESTION)
    assert "load_skill" not in model.tools[0]
    assert '"order_id": "ORD-1002"' in str(result.final_output)
    assert sum("StopAtTools" in r.getMessage() for r in caplog.records) == 1


def test_the_tool_without_the_instructions_warns(decimal, caplog):
    support = oa.load_agent("support")
    agent = Agent(name="support", instructions=BASE, tools=[lookup_order, *support.tools],
                  model=ScriptedModel(["answer"]))
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        Runner.run_sync(agent, "Hi")
    assert any("no skill list" in r.getMessage() for r in caplog.records)


def test_an_agent_with_another_name_is_filed_under_the_decimal_name(decimal):
    support = oa.load_agent("support")
    agent = Agent(name="Assistant", instructions=support.instructions, tools=[*support.tools],
                  model=ScriptedModel(["answer"]))
    Runner.run_sync(agent, "Hi")
    assert sent(decimal)[0].agent_name == "support"


def test_nothing_in_the_sdk_changes_and_other_agents_are_untouched(decimal):
    from agents.run_internal import turn_resolution

    tools = [lookup_order]
    support = oa.load_agent("support")
    agent = Agent(name="support", instructions=support.instructions, tools=[*tools, *support.tools],
                  model=ScriptedModel(["answer"]))
    Runner.run_sync(agent, "Hi")
    helper_model = ScriptedModel(["answer"])
    Runner.run_sync(Agent(name="pii-check", instructions="Answer yes or no.", model=helper_model), "Hi")

    assert helper_model.instructions == ["Answer yes or no."]
    assert list(helper_model.tools[0]) == []
    assert [t.name for t in tools] == ["lookup_order"]
    assert Agent.__init__ is ORIGINALS["Agent.__init__"]
    assert Agent.get_system_prompt is ORIGINALS["Agent.get_system_prompt"]
    assert Agent.get_all_tools is ORIGINALS["Agent.get_all_tools"]
    assert turn_resolution.check_for_final_output_from_tools is ORIGINALS["check_for_final_output_from_tools"]


class WaitingModel(ScriptedModel):
    """Answers only once DecimalAI has seen ``menu_calls`` menu requests."""

    def __init__(self, fake, menu_calls):
        super().__init__(["answer"])
        self.fake, self.menu_calls = fake, menu_calls

    async def get_response(self, *args, **kwargs):
        import asyncio

        for _ in range(200):
            if len(self.fake.menu_calls()) >= self.menu_calls:
                break
            await asyncio.sleep(0.01)
        return await super().get_response(*args, **kwargs)


def test_the_background_decision_reaches_its_own_run_from_a_worker_thread(decimal, monkeypatch):
    monkeypatch.setattr(oa, "_in_background", ORIGINAL_IN_BACKGROUND)
    support = oa.load_agent("support")
    Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    Runner.run_sync(build(support, WaitingModel(decimal, 2)), "Hi")
    oa._config_pool.shutdown(wait=True)
    monkeypatch.setattr(oa, "_config_pool", None)
    assert [t.routing_id for t in sent(decimal)] == ["rt_" + f"{n:024x}" for n in (1, 2)]


ORIGINAL_IN_BACKGROUND = oa._in_background
