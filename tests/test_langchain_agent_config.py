"""``decimalai.langchain.agent_config()``: one agent's prompt and skills as a LangChain middleware.

Drives real ``create_agent`` graphs (and a hand-built LangGraph graph) with a scripted chat model,
the real SkillRouter against the fake platform from ``test_agent_binding.py``, and the real
``CallbackHandler`` as the tracer. Nothing leaves the process.

What the binding promises, and these tests hold it to:

- the first run's model calls get the saved prompt, filled from ``context=``, then the skill list,
  and ``load_skill``, whose result is the skill's text;
- a thread keeps its first message's system text and ``load_skill`` definition byte for byte, even
  after a newer saved prompt (Claude rejects a changed one once a thinking block is in the history);
- a later message of a cut-list thread is routed on its own text, and the skills it picks that the
  thread has not shown arrive as one tagged human message, after the user's;
- the trace is filed under the DecimalAI agent, with its prompt record and routing decision.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langchain.agents.middleware")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

import decimalai._agent_binding as binding  # noqa: E402
import decimalai._config as cfg  # noqa: E402
import decimalai._langchain_config as lcc  # noqa: E402
import decimalai.langchain as lc  # noqa: E402
from decimalai._agent_binding import MissingPromptVariable  # noqa: E402
from decimalai._config import DecimalConfig  # noqa: E402
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter  # noqa: E402

from tests.test_agent_binding import FakePlatform  # noqa: E402

SAM = {"customer": {"name": "Sam"}}


# ── a scripted model: load the first listed skill once per user turn, then answer ─────
#: Every model call the scripted model answered, and the skill it asks for next.
SEEN: List[Dict[str, Any]] = []
WANT: Dict[str, Any] = {"skill": "damaged-item"}


class Scripted(BaseChatModel):
    """Loads ``WANT["skill"]`` once per user turn when ``load_skill`` is offered, then answers."""

    @property
    def _llm_type(self) -> str:
        return "decimalai-test-scripted"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[t if isinstance(t, dict) else convert_to_openai_tool(t) for t in tools],
                         **kwargs)

    def _reply(self, messages, tools):
        names = [t["function"]["name"] for t in tools or []]
        humans = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
        after = messages[humans[-1] + 1:] if humans else []
        results = [m.content for m in after if isinstance(m, ToolMessage)]
        schema = next((t["function"] for t in tools or [] if t["function"]["name"] == "load_skill"), None)
        SEEN.append({
            "system": messages[0].content if isinstance(messages[0], SystemMessage) else None,
            "messages": list(messages), "tools": names, "load_skill": schema, "results": results,
        })
        again = WANT.get("twice") and len(results) == 1
        if "load_skill" in names and WANT["skill"] and (not results or again):
            return AIMessage(content="", tool_calls=[{
                "name": "load_skill", "args": {"name": WANT["skill"]},
                "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}])
        return AIMessage(content="done")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages, kwargs.get("tools")))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)


@tool
def lookup_order(order_id: str) -> str:
    """Look an order up."""
    return f"order {order_id}: shipped"


def saved_prompt(text: str = "Hello {{customer.name}}. You help with orders.", version: int = 4):
    return SimpleNamespace(agent_name="support", system_prompt=text, version_number=version,
                           content_hash=f"{version:064d}", is_fallback=False)


@pytest.fixture()
def platform(monkeypatch):
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://lc-config.test", enabled=True))
    monkeypatch.setattr(cfg, "_client", MagicMock())
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url="http://lc-config.test", inject_body=False)
    router._request = fake.request
    monkeypatch.setattr(lc, "_skill_router_singleton", router)
    SEEN.clear()
    WANT.clear()
    WANT["skill"] = "damaged-item"
    fake.router = router
    return fake


def make_agent(support, **kwargs):
    return create_agent(Scripted(), tools=[lookup_order], middleware=[support.middleware],
                        checkpointer=InMemorySaver(), name="support-graph", **kwargs)


def ask(agent, text, thread="t1", handler=None, context=SAM):
    config: Dict[str, Any] = {"configurable": {"thread_id": thread}}
    if handler is not None:
        config["callbacks"] = [handler]
    return agent.invoke({"messages": [{"role": "user", "content": text}]}, config, context=context)


def bodies_requested(platform) -> List[Dict[str, Any]]:
    return [params for method, path, params in platform.requests if path.endswith("/body")]


# ── one run ────────────────────────────────────────────────────────────────────────────
def test_the_first_run_gets_the_prompt_then_the_list_and_load_skill(platform):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    handler = lc.CallbackHandler(auto_send=False)
    ask(make_agent(support), "My keyboard arrived cracked.", handler=handler)

    first, second = SEEN
    assert first["system"].startswith("Hello Sam. You help with orders.\n\n")
    assert first["system"].endswith(LOAD_SKILL_PROMPT_HINT)
    assert sorted(first["tools"]) == ["load_skill", "lookup_order"]
    assert first["load_skill"]["parameters"]["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    assert second["results"][0].startswith("## Skill: damaged-item")
    assert list(platform.queries.values()) == ["My keyboard arrived cracked."]

    trace = handler.get_trace()
    assert trace.agent_name == "support"          # the DecimalAI agent, not the graph's name
    assert trace.routing_id and trace.routing_id.startswith("rt_")
    assert sorted(trace.skills_offered_in_prompt) == ["damaged-item", "order-support"]
    assert trace.skills_loaded_by_agent == ["damaged-item"]
    prompt_spans = [s for s in trace.spans if (s.attributes or {}).get("decimal.prompt.version") == 4]
    assert prompt_spans and prompt_spans[0].attributes["decimal.prompt.variables"] == ["customer.name"]


def test_a_missing_variable_stops_the_run_before_any_model_call(platform):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    with pytest.raises(MissingPromptVariable, match="customer.name"):
        ask(make_agent(support), "hi", context={"nobody": 1})
    assert SEEN == []


# ── a thread ───────────────────────────────────────────────────────────────────────────
def test_a_thread_keeps_its_system_text_and_tool_through_a_newer_prompt(platform):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = make_agent(support)
    ask(agent, "My keyboard arrived cracked.", thread="chat")
    first = SEEN[0]

    # A newer prompt is saved between the messages.
    support._prompt = saved_prompt("Hi {{customer.name}}, version five.", version=5)
    SEEN.clear()
    ask(agent, "Where is my order?", thread="chat")
    later = SEEN[0]
    assert later["system"] == first["system"]
    assert later["load_skill"] == first["load_skill"]

    # The next thread gets the newer prompt.
    SEEN.clear()
    ask(agent, "Where is my order?", thread="other")
    assert SEEN[0]["system"].startswith("Hi Sam, version five.")


def test_a_cut_list_threads_new_skills_arrive_as_a_tagged_message(platform):
    platform.truncated = True
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = make_agent(support)
    ask(agent, "My keyboard arrived cracked.", thread="cut")
    first = SEEN[0]
    assert "damaged-item" in first["system"] and "order-support" not in first["system"]
    first_routing = list(platform.queries)[0]

    SEEN.clear()
    WANT["skill"] = "order-support"
    ask(agent, "Where is my order?", thread="cut")
    later = SEEN[0]
    assert later["system"] == first["system"]                  # the system text is the thread's
    assert later["load_skill"] == first["load_skill"]
    tagged = [m for m in later["messages"] if isinstance(m, HumanMessage)
              and m.additional_kwargs.get("lc_source") == "decimalai_skills"]
    assert len(tagged) == 1 and "order-support" in tagged[0].content
    assert later["messages"].index(tagged[0]) == len(later["messages"]) - 1   # after the user's
    assert SEEN[1]["results"][0].startswith("## Skill: order-support")
    # The load names the decision that showed the skill: the second message's own.
    second_routing = list(platform.queries)[1]
    assert second_routing != first_routing
    assert bodies_requested(platform)[-1].get("routing_id") == second_routing

    # A third message routes to a skill the thread has already shown, so it adds no message;
    # its load of a skill an EARLIER message showed (and its own decision did not offer) names
    # the decision that showed it.
    SEEN.clear()
    WANT["skill"] = "damaged-item"
    ask(agent, "Where is it now?", thread="cut")
    third = SEEN[0]["messages"]
    assert sum(1 for m in third if isinstance(m, HumanMessage)
               and m.additional_kwargs.get("lc_source") == "decimalai_skills") == 1
    assert bodies_requested(platform)[-1].get("routing_id") == first_routing


# ── several runs at once, and the other ways in ─────────────────────────────────────────
def test_concurrent_runs_fill_the_prompt_from_their_own_context(platform):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = make_agent(support)

    async def many():
        return await asyncio.gather(*(
            agent.ainvoke({"messages": [{"role": "user", "content": f"cracked #{i}"}]},
                          {"configurable": {"thread_id": f"c{i}"}},
                          context={"customer": {"name": f"user{i}"}})
            for i in range(4)))

    asyncio.run(many())
    systems = [s["system"] for s in SEEN if not s["results"]]
    assert sorted(s.split(".")[0] for s in systems) == [f"Hello user{i}" for i in range(4)]
    assert all(s["results"][0].startswith("## Skill: damaged-item") for s in SEEN if s["results"])


def test_stream_and_astream_run_the_same_way(platform):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = make_agent(support)
    list(agent.stream({"messages": [{"role": "user", "content": "stream: cracked"}]},
                      {"configurable": {"thread_id": "s1"}}, context=SAM, stream_mode="messages"))

    async def astream():
        async for _ in agent.astream({"messages": [{"role": "user", "content": "astream: cracked"}]},
                                     {"configurable": {"thread_id": "s2"}}, context=SAM,
                                     stream_mode="updates"):
            pass

    asyncio.run(astream())
    assert len(SEEN) == 4
    assert all(s["system"].startswith("Hello Sam") for s in SEEN)
    assert sum(1 for s in SEEN if s["results"]) == 2


def test_a_skill_loaded_twice_in_one_run_is_served_once(platform):
    WANT["twice"] = True
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    ask(make_agent(support), "My keyboard arrived cracked.")

    first_load, second_load = SEEN[2]["results"]
    assert first_load.startswith("## Skill: damaged-item")
    assert "already loaded earlier in this run" in second_load
    assert len(bodies_requested(platform)) == 1


def test_an_async_run_before_python_3_11_warns_once(platform, monkeypatch, caplog):
    monkeypatch.setattr(lcc, "_ASYNC_RUN_LOSES_ITS_CALLS", True)
    monkeypatch.setattr(lcc, "_async_warned", False)
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = make_agent(support)

    def warnings() -> List[str]:
        return [r.getMessage() for r in caplog.records if "Python 3.10" in r.getMessage()]

    with caplog.at_level(logging.WARNING):
        ask(agent, "My keyboard arrived cracked.", thread="sync")
        assert warnings() == [], "a sync run keeps its model calls"
        for thread in ("a1", "a2"):
            asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "cracked"}]},
                                      {"configurable": {"thread_id": thread}}, context=SAM))
    assert len(warnings()) == 1
    assert sum(1 for s in SEEN if s["results"]) == 3, "skills still reach the model"


def test_a_hand_built_graph_uses_system_message_and_tools(platform):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    model = Scripted().bind_tools([lookup_order, *support.tools])

    def call_model(state: MessagesState, runtime):
        system = support.system_message(state["messages"], context=runtime.context)
        return {"messages": [model.invoke([system, *state["messages"]])]}

    def route(state):
        return "tools" if state["messages"][-1].tool_calls else END

    graph = StateGraph(MessagesState, context_schema=dict)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([lookup_order, *support.tools]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")
    handler = lc.CallbackHandler(auto_send=False)
    graph.compile().invoke({"messages": [HumanMessage("Hand-built: it arrived cracked")]},
                           {"callbacks": [handler]}, context=SAM)

    first, second = SEEN
    assert first["system"].startswith("Hello Sam") and second["system"] == first["system"]
    assert second["results"][0].startswith("## Skill: damaged-item")
    trace = handler.get_trace()
    assert trace.agent_name == "support" and trace.skills_loaded_by_agent == ["damaged-item"]


# ── what a run leaves behind, and the switches ──────────────────────────────────────────
def test_an_untraced_run_leaves_nothing_on_the_router(platform):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = make_agent(support)
    for i in range(3):
        ask(agent, f"cracked #{i}", thread=f"u{i}")
    router = lc._skill_router_singleton
    assert dict(router._scoped_loaded_names) == {}
    assert dict(router._scoped_loaded_hashes) == {}


def test_with_the_tool_off_the_skill_text_is_in_the_system_text(platform, monkeypatch):
    monkeypatch.setattr(cfg._config, "load_skill_tool", False)
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    handler = lc.CallbackHandler(auto_send=False)
    ask(make_agent(support), "My keyboard arrived cracked.", handler=handler)
    (only,) = SEEN
    assert "load_skill" not in only["tools"]
    assert "## Skill: damaged-item" in only["system"]
    assert handler.get_trace().skills_delivered == ["damaged-item"]


def test_a_code_system_prompt_is_replaced_with_a_warning(platform, caplog):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = make_agent(support, system_prompt="You are a pirate.")
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        ask(agent, "My keyboard arrived cracked.")
    assert SEEN[0]["system"].startswith("Hello Sam")
    assert "fallback=" in caplog.text


def test_another_load_skill_tool_is_left_in_place_with_a_warning(platform, caplog):
    @tool("load_skill")
    def their_loader(name: str) -> str:
        """Their own loader."""
        return f"theirs: {name}"

    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = create_agent(Scripted(), tools=[their_loader], middleware=[support.middleware],
                         checkpointer=InMemorySaver())
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        ask(agent, "My keyboard arrived cracked.")
    assert SEEN[1]["results"][0] == "theirs: damaged-item"
    assert "already has a tool named load_skill" in caplog.text


def test_a_late_background_decision_reaches_the_trace(platform, monkeypatch):
    support = lcc.LangChainAgentConfig(saved_prompt(), fallback=None, on_unavailable="continue")
    agent = make_agent(support)
    ask(agent, "My keyboard arrived cracked.", thread="w1")   # the first run routes first

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def slow(fn, *args):
        def later():
            time.sleep(0.3)
            return fn(*args)
        return pool.submit(later)

    monkeypatch.setattr(binding, "_in_background", slow)
    # One model call and no load (a load waits for its run's decision itself): only the trace's
    # wait can bring the decision in.
    WANT["skill"] = None
    SEEN.clear()
    handler = lc.CallbackHandler(auto_send=False)
    ask(agent, "Where is my order?", thread="w2", handler=handler)  # the local copy; routed later
    assert len(SEEN) == 1
    trace = handler.get_trace()
    assert trace.routing_id == list(platform.queries)[-1]
    pool.shutdown()


def test_agent_config_checks_on_unavailable(platform):
    with pytest.raises(ValueError, match="on_unavailable"):
        lc.agent_config("support", on_unavailable="maybe")
