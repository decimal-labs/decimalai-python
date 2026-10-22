"""`decimalai.openai_agents.agent_config`: an agent's prompt and skills, attached explicitly.

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

from agents import Agent, Model, ModelResponse, Runner, Usage, function_tool  # noqa: E402
from agents.tracing import get_trace_provider, set_trace_processors  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

import decimalai._agent_binding as binding  # noqa: E402
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
        #: A catalog bigger than the menu: which skills make it depends on the message.
        self.truncated = False
        #: ...answered as the platform does over its caps: by retrieval, with
        #: strategy "smart_routing" and no `truncated` field at all.
        self.smart_routing = False
        self.ranking = {"cracked": ["damaged-item"], "where": ["order-support"]}
        self.requests: list = []
        self.queries: dict = {}  # routing id -> the message it was minted for
        self._next = 0
        #: Names the workspace has no agent by, and the agents created since.
        self.missing = {"missing-agent"}
        self.created: list = []

    def created_names(self):
        return {name for name, _ in self.created}

    def create_agent(self, agent_name, **fields):
        self.created.append((agent_name, fields))
        self.missing.discard(agent_name)

    def prompt_payload(self, agent_name, version=None):
        if agent_name in self.created_names():
            return {"agent_name": agent_name, "system_prompt": None, "version_number": None,
                    "content_hash": None}
        if agent_name in self.missing:
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
        if path in ("/api/v1/skills/menu", "/api/v1/skills/route"):
            if self.menu_down:
                raise sr.SkillRouterError("transport error: timed out")
            query = (json or {}).get("query") if path.endswith("/route") else None
            shown = list(self.skills)
            if self.truncated:
                words = (query or "").lower()
                shown = next((names for word, names in self.ranking.items() if word in words), shown[:1])
            routing_id = self.routing_id()
            self.queries[routing_id] = query
            if self.truncated and self.smart_routing:
                return {
                    "skills": [{"name": n, "description": n} for n in shown],
                    "stable_menu": stable_menu(shown), "menu_instruction": MENU_INSTRUCTION,
                    "routing_hint": "", "stable_menu_skills": shown, "routing_id": routing_id,
                    "strategy": "smart_routing", "query_ranked": bool(shown),
                }
            return {
                "skills": [{"name": n, "description": n} for n in shown],
                "stable_menu": stable_menu(shown),
                "menu_instruction": MENU_INSTRUCTION if shown else "",
                "routing_hint": "", "stable_menu_skills": shown,
                "routing_id": routing_id, "strategy": "full_menu" if not query else "semantic",
                "query_ranked": bool(query), "truncated": self.truncated,
                "rows_total": len(self.skills), "skill_count": len(shown),
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
        """Every routing request: the agent's menu, or a route on a message."""
        return [r for r in self.requests if r[1] in ("/api/v1/skills/menu", "/api/v1/skills/route")]

    def body_calls(self):
        return [r for r in self.requests if r[1].endswith("/body")]


class ScriptedModel(Model):
    """Plays steps in order: ("call", name, args) or "answer"."""

    def __init__(self, steps):
        self.steps, self.instructions, self.tools, self.inputs = list(steps), [], [], []
        #: Each request's tool parameters, read when the request was made (as a model does).
        self.schemas: list = []

    async def get_response(self, system_instructions, input, model_settings, tools, *args, **kwargs):
        self.instructions.append(system_instructions)
        self.tools.append({t.name: t for t in tools})
        self.schemas.append({t.name: json.loads(json.dumps(getattr(t, "params_json_schema", None)))
                             for t in tools})
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
    client.create_agent.side_effect = fake.create_agent
    client.register_manifest.return_value = {"manifest_id": "m-1", "status": "active"}
    monkeypatch.setattr(cfg, "_client", client)
    router = SkillRouter(api_key="dai_sk_test", base_url="http://127.0.0.1:9", inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(oa, "_skill_router_singleton", router)
    monkeypatch.setattr(oa, "_tracing_processor", None, raising=False)
    monkeypatch.setattr(oa, "_tracing_processor_added", False, raising=False)
    monkeypatch.setattr(oa, "_run_rails", OrderedDict())
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    monkeypatch.setattr(binding, "_config_pool", None)
    provider = get_trace_provider()
    saved = list(provider._multi_processor._processors)
    set_trace_processors([])
    fake.client = client
    try:
        yield fake
    finally:
        set_trace_processors(saved)


def trace_routing_id(fake) -> str:
    return "rt_" + f"{1:024x}"


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
    support = oa.agent_config("support")
    model = ScriptedModel([LOAD, LOOKUP, "answer"])
    Runner.run_sync(build(support, model), QUESTION)

    assert model.instructions[0] == f"{BASE}\n\n{PREFIX}\n{LOAD_SKILL_PROMPT_HINT}"
    assert all(text == model.instructions[0] for text in model.instructions)
    tool = model.tools[0]["load_skill"]
    assert tool.params_json_schema["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    assert any(BODIES["damaged-item"] in str(item.get("output", ""))
               for item in model.inputs[1] if isinstance(item, dict))
    assert len(decimal.menu_calls()) == 1, "the first run routes; nothing was read at startup"
    assert decimal.queries[trace_routing_id(decimal)] == QUESTION, "routed on the run's own message"

    [trace] = sent(decimal)
    assert trace.agent_name == "support"
    assert trace.routing_id == "rt_" + f"{1:024x}"
    assert trace.skills_offered_in_prompt == ["damaged-item", "order-support"]
    assert trace.skills_loaded_by_agent == ["damaged-item"]
    digest = hashlib.sha256(BODIES["damaged-item"].encode()).hexdigest()
    assert any(v["name"] == "damaged-item" and v["hash"] == digest for v in trace.skills_delivered_versions)
    span = next(s for s in trace.spans if s.name == "support" and s.attributes.get("decimal.prompt.version"))
    assert span.attributes["decimal.prompt.version"] == 2
    assert span.attributes["decimal.prompt.source"] == "decimalai"
    assert span.attributes["decimal.prompt.content_hash"] == "c" * 64


def test_each_later_run_records_its_own_routing_decision(decimal):
    support = oa.agent_config("support")
    for _ in range(3):
        Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    traces = sent(decimal)
    assert [t.routing_id for t in traces] == ["rt_" + f"{n:024x}" for n in (1, 2, 3)]
    assert len(decimal.menu_calls()) == 3


def test_a_decision_offering_a_different_list_is_not_stamped(decimal):
    support = oa.agent_config("support")
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
    support = oa.agent_config("support")
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
    support = oa.agent_config("support")
    model = ScriptedModel(["answer"])
    with pytest.raises(oa.MissingPromptVariable, match="store"):
        Runner.run_sync(build(support, model), "Hi", context={"other": 1})
    assert model.instructions == []


def test_no_saved_prompt_uses_the_fallback_and_says_so(decimal):
    decimal.prompt = {"system_prompt": None, "version_number": None, "content_hash": None}
    support = oa.agent_config("support", fallback=BASE)
    model = ScriptedModel(["answer"])
    Runner.run_sync(build(support, model), "Hi")
    assert model.instructions[0].startswith(BASE)
    [trace] = sent(decimal)
    span = next(s for s in trace.spans if s.attributes.get("decimal.prompt.source"))
    assert span.attributes["decimal.prompt.source"] == "fallback (none saved)"


def test_an_unknown_agent_is_created_without_the_codes_prompt(decimal, monkeypatch, tmp_path, capsys):
    """A name the workspace has not seen is a new agent (owner decision 2026-10-21): it is
    created, with no prompt (the code's is never saved), and runs on its fallback."""
    monkeypatch.chdir(tmp_path)
    support = oa.agent_config("missing-agent", fallback=BASE)
    assert decimal.created == [("missing-agent", {})]
    assert (support.name, support.prompt_source) == ("missing-agent", "fallback (none saved)")
    assert "created agent 'missing-agent'" in capsys.readouterr().err
    oa.agent_config("missing-agent", fallback=BASE)
    assert len(decimal.created) == 1, "created once; the next process finds it"


def test_skills_unavailable_continue_or_fail(decimal):
    decimal.menu_down = True
    support = oa.agent_config("support")
    model = ScriptedModel(["answer"])
    Runner.run_sync(build(support, model), QUESTION)
    assert model.instructions[0] == BASE
    assert "load_skill" not in model.tools[0], "no skills, so no loader"
    [trace] = sent(decimal)
    span = next(s for s in trace.spans if s.attributes.get("decimal.prompt.source"))
    assert "did not return" in span.attributes["decimal.skills.unavailable"]

    strict = oa.agent_config("support", on_unavailable="fail")
    with pytest.raises(RuntimeError, match="unavailable"):
        Runner.run_sync(build(strict, ScriptedModel(["answer"])), QUESTION)


def test_a_skill_loaded_twice_in_one_run_is_sent_once(decimal):
    support = oa.agent_config("support")
    model = ScriptedModel([LOAD, LOAD, "answer"])
    Runner.run_sync(build(support, model), QUESTION)
    assert len(decimal.body_calls()) == 1
    assert "already loaded" in str(model.inputs[2][-1].get("output"))


def test_stop_on_first_tool_agents_get_no_loader_and_a_warning(decimal, caplog):
    support = oa.agent_config("support")
    model = ScriptedModel([LOOKUP])
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        result = Runner.run_sync(build(support, model, tool_use_behavior="stop_on_first_tool"), QUESTION)
    assert "load_skill" not in model.tools[0]
    assert '"order_id": "ORD-1002"' in str(result.final_output)
    assert sum("StopAtTools" in r.getMessage() for r in caplog.records) == 1


def test_the_tool_without_the_instructions_warns(decimal, caplog):
    support = oa.agent_config("support")
    agent = Agent(name="support", instructions=BASE, tools=[lookup_order, *support.tools],
                  model=ScriptedModel(["answer"]))
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        Runner.run_sync(agent, "Hi")
    assert any("no skill list" in r.getMessage() for r in caplog.records)


def test_the_instructions_without_the_tool_get_the_prompt_alone_and_a_warning(decimal, caplog):
    """The likeliest slip in the pull shape: ``*support.tools`` left out. A
    skill list would tell the model to call a tool it does not have, and the
    Agents SDK ends the run with "tool not found" when it tries."""
    support = oa.agent_config("support")
    model = ScriptedModel(["answer", "answer"])
    agent = Agent(name="support", instructions=support.instructions, tools=[lookup_order], model=model)
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        Runner.run_sync(agent, QUESTION)
        Runner.run_sync(agent, QUESTION)
    assert model.instructions[0] == BASE
    assert decimal.menu_calls() == []
    assert sum("*support.tools" in r.getMessage() for r in caplog.records) == 1


def test_an_agent_with_another_name_is_filed_under_the_decimal_name(decimal):
    support = oa.agent_config("support")
    agent = Agent(name="Assistant", instructions=support.instructions, tools=[*support.tools],
                  model=ScriptedModel(["answer"]))
    Runner.run_sync(agent, "Hi")
    assert sent(decimal)[0].agent_name == "support"


def test_nothing_in_the_sdk_changes_and_other_agents_are_untouched(decimal):
    from agents.run_internal import turn_resolution

    tools = [lookup_order]
    support = oa.agent_config("support")
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
    monkeypatch.setattr(binding, "_in_background", ORIGINAL_IN_BACKGROUND)
    support = oa.agent_config("support")
    Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    Runner.run_sync(build(support, WaitingModel(decimal, 2)), "Hi")
    binding._config_pool.shutdown(wait=True)
    assert [t.routing_id for t in sent(decimal)] == ["rt_" + f"{n:024x}" for n in (1, 2)]


def test_a_run_faster_than_its_routing_request_still_ships_its_routing_id(decimal, monkeypatch):
    import threading

    monkeypatch.setattr(binding, "_in_background", ORIGINAL_IN_BACKGROUND)
    support = oa.agent_config("support")
    release = threading.Event()
    answer = decimal.request

    def slow_menu(method, path, **kwargs):
        if path.endswith(("/skills/menu", "/skills/route")) and len(decimal.menu_calls()) >= 1:
            release.wait(0.3)  # the run ends long before this returns
        return answer(method, path, **kwargs)

    monkeypatch.setattr(oa._skill_router_singleton, "_request", slow_menu)
    Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    binding._config_pool.shutdown(wait=True)
    assert [t.routing_id for t in sent(decimal)] == ["rt_" + f"{n:024x}" for n in (1, 2)]
    assert not oa._run_rails, "a late routing result must not leave a rail behind"


def test_a_shut_down_worker_pool_never_fails_a_run(decimal, monkeypatch):
    monkeypatch.setattr(binding, "_in_background", ORIGINAL_IN_BACKGROUND)
    support = oa.agent_config("support")
    for _ in range(2):  # the second run is the first with background work
        Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    binding._config_pool.shutdown(wait=True)  # e.g. torn down at interpreter exit
    Runner.run_sync(build(support, ScriptedModel(["answer"])), "Hi")
    binding._config_pool.shutdown(wait=True)
    assert len(sent(decimal)) == 3


def test_every_run_is_routed_on_its_own_message(decimal):
    support = oa.agent_config("support")
    messages = ["Where is ORD-1001?", "My keyboard from ORD-1002 arrived cracked.", "Thanks!"]
    for message in messages:
        Runner.run_sync(build(support, ScriptedModel(["answer"])), message)
    traces = sent(decimal)
    assert [decimal.queries[t.routing_id] for t in traces] == messages
    assert all(t.skills_offered_in_prompt == ["damaged-item", "order-support"] for t in traces)


@pytest.mark.parametrize("answer", ["whole_menu_cut", "smart_routing"])
def test_a_catalog_bigger_than_the_menu_is_routed_before_each_model_call(decimal, answer):
    """Either way the platform says it: a whole-menu answer marked `truncated`,
    or (over its caps) a `smart_routing` answer, which carries no such field."""
    decimal.truncated = True
    decimal.smart_routing = answer == "smart_routing"
    support = oa.agent_config("support")
    cracked, where = ScriptedModel(["answer"]), ScriptedModel(["answer"])
    Runner.run_sync(build(support, cracked), "My keyboard arrived cracked.")
    Runner.run_sync(build(support, where), "Where is my order?")
    assert "| damaged-item |" in cracked.instructions[0] and "| order-support |" not in cracked.instructions[0]
    assert "| order-support |" in where.instructions[0] and "| damaged-item |" not in where.instructions[0]
    assert "enum" not in where.tools[0]["load_skill"].params_json_schema["properties"]["name"], (
        "one tool serves every run, so a per-message list must not be its name list")
    traces = sent(decimal)
    assert [decimal.queries[t.routing_id] for t in traces] == ["My keyboard arrived cracked.", "Where is my order?"]
    assert len(decimal.menu_calls()) == 2, "no background request: each run's route was its own"


def test_without_turn_input_skills_keep_a_fixed_order_and_it_says_so_once(decimal, caplog):
    from types import SimpleNamespace

    support = oa.agent_config("support")
    agent = build(support, ScriptedModel(["answer"]))
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        assert support._message(SimpleNamespace(context=None), agent) is None
        assert support._message(SimpleNamespace(context=None), agent) is None
    assert sum("turn_input" in r.getMessage() for r in caplog.records) == 1
    assert support._message(SimpleNamespace(turn_input=[{"role": "user", "content": "Hi there"}]), agent) == "Hi there"


ORIGINAL_IN_BACKGROUND = binding._in_background


# ── the documented channel settings ──────────────────────────────────────────


def configure(monkeypatch, **settings):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True, **settings))


def test_with_the_tool_switched_off_each_run_gets_its_skills_text_in_the_instructions(decimal, monkeypatch):
    """The tool's kill switch (DECIMALAI_LOAD_SKILL_TOOL=0 / init(load_skill_tool=False))
    leaves pasted text as the channel: every run is routed on its own message
    and the picked skill's text follows the list. Delivered, never "loaded"."""
    configure(monkeypatch, load_skill_tool=False)
    support = oa.agent_config("support")
    model = ScriptedModel(["answer", "answer"])
    Runner.run_sync(build(support, model), QUESTION)
    Runner.run_sync(build(support, model), "Where is ORD-1003?")

    assert all("load_skill" not in tools for tools in model.tools)
    first = model.instructions[0]
    assert first.startswith(f"{BASE}\n\n{PREFIX}")
    assert f"## Skill: damaged-item\n\n{BODIES['damaged-item']}" in first
    assert LOAD_SKILL_PROMPT_HINT not in first, "the usage sentence names a tool this agent lacks"
    assert len(decimal.menu_calls()) == 2, "each run routed on its own message"
    for trace in sent(decimal):
        assert trace.skills_delivered == ["damaged-item"]
        assert trace.skills_loaded_by_agent == []
        assert [v["name"] for v in trace.skills_delivered_versions] == ["damaged-item"]


def test_with_the_tool_and_pasting_both_off_skills_are_off_and_it_says_so(decimal, monkeypatch, caplog):
    configure(monkeypatch, load_skill_tool=False, inject_skill_body=False)
    support = oa.agent_config("support")
    model = ScriptedModel(["answer"])
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        Runner.run_sync(build(support, model), QUESTION)
    assert model.instructions[0] == BASE
    assert "load_skill" not in model.tools[0]
    assert decimal.menu_calls() == []
    assert sum("inject_skill_body" in r.getMessage() for r in caplog.records) == 1


def test_text_pasted_into_the_instructions_is_not_sent_again_by_the_tool(decimal, monkeypatch):
    """Both channels on, by asking for both: the tool answers for a skill whose
    text is already in the instructions without fetching it again."""
    configure(monkeypatch, inject_skill_body=True)
    support = oa.agent_config("support")
    model = ScriptedModel([LOAD, "answer"])
    Runner.run_sync(build(support, model), QUESTION)
    assert BODIES["damaged-item"] in model.instructions[0]
    assert "load_skill" in model.tools[0]
    assert "already loaded" in str(model.inputs[1][-1].get("output"))
    assert len(decimal.body_calls()) == 1


class ClaudeModel(ScriptedModel):
    """A scripted model with Claude's name, as LiteLLM's model carries it."""

    model = "anthropic/claude-haiku-5-5"


REPLAYED_THINKING = [
    {"role": "user", "content": "My keyboard arrived cracked."},
    {"type": "reasoning", "id": "rs_1", "summary": [],
     "provider_data": {"model": "anthropic/claude-haiku-5-5", "thinking_blocks": [
         {"type": "thinking", "thinking": "", "signature": "sig"}]}},
    {"role": "assistant", "content": "Sorry to hear that."},
    {"role": "user", "content": "Where is my order?"},
]


def _enum(schemas, n=0):
    return schemas[n]["load_skill"]["properties"]["name"].get("enum")


def _save_prompt(fake, support, text, version):
    """A version saved in the dashboard, and the config's local copy re-read."""
    fake.prompt = {"system_prompt": text, "version_number": version, "content_hash": str(version) * 64}
    support._refresh_prompt()


def test_a_claude_chat_keeps_its_first_runs_instructions_and_load_skill(decimal):
    """Runs sharing a RunConfig(group_id=...) on Claude: a prompt saved and a skill removed
    mid-chat reach the next chat, not this one (Claude 400s on a changed system or tool
    once thinking is in the history). Each message is still routed, for the record."""
    from agents import RunConfig

    support = oa.agent_config("support")
    model = ClaudeModel(["answer", "answer", "answer"])
    agent = build(support, model)
    chat = RunConfig(group_id="thread-1")
    Runner.run_sync(agent, "My keyboard arrived cracked.", run_config=chat)
    _save_prompt(decimal, support, "Be brief.", 3)
    decimal.skills = ["damaged-item"]
    support._route_run(None, [], "anything")  # the local copy follows

    Runner.run_sync(agent, REPLAYED_THINKING, run_config=chat)
    assert model.instructions[1] == model.instructions[0], "Claude rejects a changed system prompt"
    assert model.instructions[0].startswith(BASE)
    assert _enum(model.schemas, 1) == _enum(model.schemas, 0) == ["damaged-item", "order-support"]
    assert list(decimal.queries.values())[-1] == "Where is my order?", "still routed, for the record"

    Runner.run_sync(agent, "Hello", run_config=RunConfig(group_id="thread-2"))  # a new chat
    assert model.instructions[2].startswith("Be brief.")
    assert _enum(model.schemas, 2) == ["damaged-item"]


def test_a_claude_chat_without_a_group_id_follows_the_newest_prompt_and_says_so_once(decimal, caplog):
    support = oa.agent_config("support")
    model = ClaudeModel(["answer", "answer"])
    agent = build(support, model)
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        Runner.run_sync(agent, "My keyboard arrived cracked.")
        _save_prompt(decimal, support, "Be brief.", 3)
        Runner.run_sync(agent, REPLAYED_THINKING)
    assert model.instructions[1].startswith("Be brief.")
    warned = [r.getMessage() for r in caplog.records if "bound to a different conversation" in r.getMessage()]
    assert len(warned) == 1 and "RunConfig(group_id=...)" in warned[0]
    assert "drop_block" not in warned[0], "no workaround: the group id is the fix"


def test_on_other_models_a_group_id_leaves_each_run_on_the_newest_prompt(decimal):
    from agents import RunConfig

    support = oa.agent_config("support")
    model = ScriptedModel(["answer", "answer"])
    agent = build(support, model)
    chat = RunConfig(group_id="thread-1")
    Runner.run_sync(agent, "My keyboard arrived cracked.", run_config=chat)
    _save_prompt(decimal, support, "Be brief.", 3)
    Runner.run_sync(agent, "Where is my order?", run_config=chat)
    assert model.instructions[1].startswith("Be brief.")


def test_a_claude_chat_carries_to_another_process_through_its_state(decimal):
    from agents import RunConfig

    support = oa.agent_config("support")
    model = ClaudeModel(["answer", "answer"])
    chat = support.conversation("thread-1")
    assert chat.state is None, "nothing kept before its first run"
    Runner.run_sync(build(support, model), "My keyboard arrived cracked.",
                    run_config=RunConfig(group_id=chat.id))
    saved = json.loads(json.dumps(chat.state))
    assert saved["id"] == "thread-1" and saved["agents"]["support"]["prompt"]["version"] == 2

    decimal.prompt = {"system_prompt": "Be brief.", "version_number": 3, "content_hash": "3" * 64}
    elsewhere = oa.agent_config("support")  # another worker, which read version 3
    resumed = elsewhere.conversation("thread-1", state=saved)
    Runner.run_sync(build(elsewhere, model), REPLAYED_THINKING, run_config=RunConfig(group_id=resumed.id))
    assert model.instructions[1] == model.instructions[0]
    assert _enum(model.schemas, 1) == _enum(model.schemas, 0)


def test_conversation_refuses_another_chats_or_agents_state(decimal):
    from agents import RunConfig

    support = oa.agent_config("support")
    chat = support.conversation("thread-1")
    Runner.run_sync(build(support, ClaudeModel(["answer"])), "Hi", run_config=RunConfig(group_id="thread-1"))
    with pytest.raises(ValueError, match="thread-2"):
        support.conversation("thread-2", state=chat.state)
    other = oa.agent_config("billing")
    with pytest.raises(ValueError, match="billing"):
        other.conversation("thread-1", state=chat.state)
    with pytest.raises(ValueError, match="group_id"):
        support.conversation("  ")
    with pytest.raises(ValueError, match="state="):
        support.conversation("thread-1", state=["not", "a", "state"])


def test_a_runs_load_skill_keeps_the_names_it_was_shown_when_the_list_changes_mid_run(decimal):
    """One Claude run replays its thinking between tool calls, so a decision landing mid-run
    (the agent's list changed) must not change the tool its next request carries."""
    support = oa.agent_config("support")

    @function_tool
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        support._store_menu(stable_menu(["damaged-item"]), ["damaged-item"], "rt_late", False)
        return "delivered"

    model = ClaudeModel([LOOKUP, "answer"])
    agent = Agent(name=support.name, instructions=support.instructions,
                  tools=[lookup_order, *support.tools], model=model)
    Runner.run_sync(agent, QUESTION)
    assert _enum(model.schemas, 1) == _enum(model.schemas, 0) == ["damaged-item", "order-support"]
    assert support.load_skill.params_json_schema["properties"]["name"]["enum"] == ["damaged-item"], \
        "outside a run, the agent's whole list as it now stands"


@pytest.mark.parametrize("model, items", [
    (ScriptedModel, REPLAYED_THINKING),                     # not Claude
    (ClaudeModel, [{"role": "user", "content": "Hi"}]),     # Claude, nothing sent back
])
def test_no_warning_where_nothing_binds(decimal, caplog, model, items):
    support = oa.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        Runner.run_sync(build(support, model(["answer"])), items)
    assert not any("bound to a different conversation" in r.getMessage() for r in caplog.records)
