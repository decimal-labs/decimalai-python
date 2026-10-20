"""`decimalai.adk.agent_config()`: an agent's prompt and skills as ADK objects.

Driven through the real Google ADK runner with a scripted ``BaseLlm``: the model
records what each request was given (the system instruction, the tools and the
load_skill schema, the tool results) and calls ``load_skill`` when told to. Traces
come from the real ``DecimalAIPlugin`` in ``App(plugins=[...])``, captured with its
``on_trace`` observer. DecimalAI's answers come from the fake platform the
framework-neutral tests use, so nothing leaves the process.
"""

from __future__ import annotations

import copy
import logging
from unittest.mock import MagicMock

import pytest

pytest.importorskip("google.adk")

from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.apps import App  # noqa: E402
from google.adk.models import BaseLlm, LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

import decimalai  # noqa: E402
import decimalai._agent as agent_mod  # noqa: E402
import decimalai._agent_binding as binding  # noqa: E402
import decimalai.adk as adk  # noqa: E402
from decimalai._agent import SavedPrompt  # noqa: E402
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter  # noqa: E402
from tests.test_agent_binding import FakePlatform  # noqa: E402

BASE = "http://adk-load-agent.test"
SAM = {"customer": {"name": "Sam"}}


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "adk-manifest", "status": "active"}
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=True))
    monkeypatch.setattr(cfg, "_client", client)
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    for name, value in (("_PluginClass", None), ("_manifest_ids", {}), ("_manifest_trackers", {}),
                        ("_pending_manifests", {}), ("_live_runs", {})):
        monkeypatch.setattr(adk, name, value)
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url=BASE, inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(adk, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt="Hello {{customer.name}}.",
                             version_number=4, content_hash="c" * 64)
    monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kwargs: fake.saved)
    fake.client = client
    return fake


class Model(BaseLlm):
    """A scripted model: ("call", skill) asks load_skill for it, "answer" ends the turn."""

    steps: list = []
    requests: list = []

    async def generate_content_async(self, llm_request, stream=False):
        tools = {}
        for tool in llm_request.config.tools or ():
            for declaration in tool.function_declarations or ():
                tools[declaration.name] = declaration.parameters_json_schema
        returns = [part.function_response.response for content in llm_request.contents
                   for part in content.parts or () if part.function_response]
        self.requests.append({"instruction": llm_request.config.system_instruction or "",
                              "tools": tools, "returns": returns})
        step = self.steps.pop(0)
        if step == "answer":
            part = types.Part(text="done")
        else:
            part = types.Part(function_call=types.FunctionCall(name="load_skill", args={"name": step[1]}))
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def model(*steps):
    return Model(model="scripted", steps=list(steps), requests=[])


def ours(request):
    """The request's system instruction up to the line ADK adds about the agent itself."""
    return request["instruction"].split("\n\nYou are an agent.")[0]


def run(support, llm, question, *, state=None, tools=(), plugin=None, traces=None, session=None):
    """One invocation of an LlmAgent built from ``support``, through the real runner."""
    agent = LlmAgent(name="support_agent", model=llm, instruction=support.instruction,
                     tools=[*tools, *support.tools])
    plugins = [plugin] if plugin is not None else (
        [adk.DecimalAIPlugin(on_trace=traces.append)] if traces is not None else [])
    runner = InMemoryRunner(app=App(name="support_app", root_agent=agent, plugins=plugins))
    if session is None:
        session = runner.session_service.create_session_sync(
            app_name="support_app", user_id="u1", state=dict(SAM if state is None else state))
    else:
        runner.session_service = session[1]
        session = session[0]
    text = []
    for event in runner.run(user_id="u1", session_id=session.id, new_message=types.Content(
            role="user", parts=[types.Part(text=question)])):
        for part in (event.content.parts if event.content else None) or ():
            if part.text:
                text.append(part.text)
    decimalai.flush()
    return "".join(text), (session, runner.session_service)


def test_the_agent_gets_its_prompt_skill_list_and_load_skill(platform):
    support = adk.agent_config("support")
    assert type(support).__name__ == "ADKAgentConfig" and support.name == "support"
    llm = model(("call", "damaged-item"), "answer")
    answer, _ = run(support, llm, "My keyboard arrived cracked.")
    assert answer == "done"
    first, second = llm.requests
    assert ours(first).startswith("Hello Sam.\n\n## Available Skills")
    assert ours(first).endswith(LOAD_SKILL_PROMPT_HINT)
    assert first["tools"]["load_skill"]["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    assert second["returns"][0]["result"].startswith("## Skill: damaged-item")
    assert list(platform.queries.values()) == ["My keyboard arrived cracked."]


def test_each_invocation_is_routed_on_its_own_message(platform):
    support = adk.agent_config("support")
    _, session = run(support, model("answer"), "My keyboard arrived cracked.")
    run(support, model("answer"), "Where is my order?", session=session)
    assert list(platform.queries.values()) == ["My keyboard arrived cracked.", "Where is my order?"]


def test_a_missing_placeholder_says_the_values_come_from_session_state(platform):
    support = adk.agent_config("support")
    with pytest.raises(decimalai.MissingPromptVariable, match=r"customer\.name\. Put them in the session's state"):
        run(support, model("answer"), "Hi", state={})


def test_without_a_list_the_agent_gets_its_prompt_and_no_load_skill(platform):
    platform.down = True
    support = adk.agent_config("support")
    llm = model("answer")
    run(support, llm, "Hi")
    assert ours(llm.requests[0]) == "Hello Sam."
    assert "load_skill" not in llm.requests[0]["tools"]


def test_an_agent_with_adks_skill_toolset_gets_no_decimal_skills_and_is_told_once(platform, monkeypatch, caplog):
    monkeypatch.setattr(adk, "_has_skill_toolset", lambda agent: True)
    support = adk.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai.adk"):
        for _ in range(2):
            llm = model("answer")
            run(support, llm, "Hi")
            assert ours(llm.requests[0]) == "Hello Sam."
            assert "load_skill" not in llm.requests[0]["tools"]
    assert sum("SkillToolset" in r.getMessage() for r in caplog.records) == 1
    assert platform.menu_calls() == []


def test_another_tool_named_load_skill_gets_no_decimal_skills(platform, caplog):
    def load_skill(name: str) -> str:
        """Theirs."""
        return "theirs"

    support = adk.agent_config("support")
    llm = model("answer")
    with caplog.at_level(logging.WARNING, logger="decimalai.adk"):
        run(support, llm, "Hi", tools=[load_skill])
    assert ours(llm.requests[0]) == "Hello Sam."
    assert any("another tool named load_skill" in r.getMessage() for r in caplog.records)
    assert platform.menu_calls() == []


def test_init_enabled_false_stops_skills_but_not_the_prompt(platform, monkeypatch):
    import decimalai._config as cfg

    support = adk.agent_config("support")
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    llm = model("answer")
    run(support, llm, "Hi")
    assert ours(llm.requests[0]) == "Hello Sam."
    assert platform.menu_calls() == []


def test_an_invocation_is_one_trace_filed_under_the_decimalai_agent_with_its_record(platform):
    traces: list = []
    support = adk.agent_config("support")
    # The plugin's own label and ADK's agent name both lose to the agent_config() name.
    plugin = adk.DecimalAIPlugin(agent_name="runner-label", on_trace=traces.append)
    run(support, model(("call", "damaged-item"), "answer"), "My keyboard arrived cracked.", plugin=plugin)
    (trace,) = traces
    assert trace.agent_name == "support"
    assert trace.routing_id and trace.routing_id in platform.queries
    assert trace.skills_offered_in_prompt == ["damaged-item", "order-support"]
    assert trace.skills_loaded_by_agent == ["damaged-item"]
    assert "damaged-item" in trace.skills_delivered
    assert [entry["name"] for entry in trace.active_skills] == ["damaged-item"]
    (span,) = [s for s in trace.spans if s.name == "support_agent" and s.span_type.value == "agent"]
    assert span.attributes["decimal.prompt.version"] == 4
    assert span.attributes["decimal.prompt.variables"] == ["customer.name"]
    assert span.attributes["decimal.agent"] == "support"


def test_the_load_names_the_decimalai_agent_and_the_invocation_routes_once(platform):
    support = adk.agent_config("support")
    run(support, model(("call", "damaged-item"), "answer"), "My keyboard arrived cracked.", traces=[])
    (body,) = [params for _method, path, params in platform.requests if path.endswith("/body")]
    assert body.get("agent_name") == "support"
    assert len(platform.menu_calls()) == 1, "one decision per invocation, across its tool steps"


def test_the_manifest_declares_the_saved_prompt_template(platform):
    support = adk.agent_config("support")
    run(support, model("answer"), "Hi", traces=[])
    (snapshot,) = [c.args[0] for c in platform.client.register_manifest.call_args_list]
    assert snapshot.agent_name == "support"
    prompts = {c.component_name: c.schema_json["content"]
               for c in snapshot.components if c.component_type == "prompt"}
    assert prompts == {"system": "Hello {{customer.name}}."}


def test_without_the_plugin_the_agent_still_gets_its_prompt_and_skills(platform):
    support = adk.agent_config("support")
    llm = model(("call", "damaged-item"), "answer")
    run(support, llm, "My keyboard arrived cracked.")
    assert llm.requests[1]["returns"][0]["result"].startswith("## Skill: damaged-item")
    assert adk._live_runs == {}


def test_a_copied_agent_shares_its_config(platform):
    support = adk.agent_config("support")
    assert copy.copy(support) is support and copy.deepcopy(support) is support
    (tool,) = support.tools
    assert support.tools == [tool], "one load_skill per config"


# ── sessions on Claude ──────────────────────────────────────


def claude(*steps):
    return Model(model="claude-haiku-5-5", steps=list(steps), requests=[])


def save_prompt(platform, support, text, version):
    platform.saved = SavedPrompt(agent_name="support", system_prompt=text, version_number=version,
                                 content_hash=str(version) * 64)
    support._refresh_prompt()


def change_everything(platform, support):
    """A version saved in the dashboard, a skill unassigned, and both reach the local copy."""
    save_prompt(platform, support, "Hi {{customer.name}}, be brief.", 5)
    platform.skills = ["damaged-item"]
    support._route_run(None, [], "anything")


def test_on_claude_a_session_keeps_its_instruction_and_load_skill(platform):
    support = adk.agent_config("support")
    llm = claude("answer", ("call", "order-support"), "answer", "answer")
    traces: list = []
    _, session = run(support, llm, "My keyboard arrived cracked.", traces=traces)
    change_everything(platform, support)
    run(support, llm, "Where is my order?", traces=traces, session=session)
    one, two, three = llm.requests[:3]
    assert ours(two) == ours(three) == ours(one), "Claude rejects a conversation whose system prompt changed"
    assert two["tools"] == one["tools"] and one["tools"]["load_skill"]["properties"]["name"]["enum"] == [
        "damaged-item", "order-support"]
    assert three["returns"][-1]["result"].startswith("## Skill: order-support")
    assert list(platform.queries.values())[-1] == "Where is my order?", "still routed, for the record"

    run(support, llm, "Hello", traces=traces)  # a new session
    assert ours(llm.requests[3]).startswith("Hi Sam, be brief.")
    assert llm.requests[3]["tools"]["load_skill"]["properties"]["name"]["enum"] == ["damaged-item"]


def test_on_claude_the_plugin_keeps_the_session_in_its_state_for_another_process(platform):
    support = adk.agent_config("support")
    llm = claude("answer", "answer")
    _, (session, service) = run(support, llm, "My keyboard arrived cracked.", traces=[])
    stored = service.get_session_sync(app_name="support_app", user_id="u1", session_id=session.id)
    assert stored.state[adk._STATE_KEY]["support"]["prompt"]["version"] == 4
    change_everything(platform, support)

    elsewhere = adk.agent_config("support")  # another worker: no memory of the session
    run(elsewhere, llm, "Where is my order?", traces=[], session=(session, service))
    assert ours(llm.requests[1]) == ours(llm.requests[0])


def test_on_claude_without_the_plugin_this_process_keeps_the_session(platform):
    support = adk.agent_config("support")
    llm = claude("answer", "answer")
    _, session = run(support, llm, "My keyboard arrived cracked.")
    change_everything(platform, support)
    run(support, llm, "Where is my order?", session=session)
    assert ours(llm.requests[1]) == ours(llm.requests[0])


def test_on_gemini_each_invocation_shows_the_newest_prompt_as_before(platform):
    support = adk.agent_config("support")
    llm = model("answer", "answer")
    _, session = run(support, llm, "My keyboard arrived cracked.", traces=[])
    save_prompt(platform, support, "Hi {{customer.name}}.", 5)
    run(support, llm, "Where is my order?", traces=[], session=session)
    assert ours(llm.requests[1]).startswith("Hi Sam.")
    assert adk._STATE_KEY not in session[1].get_session_sync(
        app_name="support_app", user_id="u1", session_id=session[0].id).state
