"""Two smaller LangChain adapter regressions.

* Every ToolCallRecord was dropped from a LangGraph / `create_agent` trace.
  A tool record only reaches the wire through `LlmCallRecord.tool_calls`,
  and the only rule for attaching one was `tool_span.parent_span_id ==
  llm_call.span_id`. In a graph the model runs under the `agent` node and
  the tool under a sibling `tools` node, so that equality never held.

* The skill loader stamped a routing_id and the full offered-skill menu onto
  LCEL runs it had provably injected nothing into: `prompt | llm` hands the
  model a PromptValue, which the old injector could not build a system
  message into — but it had already consulted the Router and recorded the
  telemetry by the time it found that out.
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate


class NamedFakeChat(FakeListChatModel):
    model_name: str = "fake-model-1"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name, "temperature": 0.0}


@pytest.fixture(autouse=True)
def reset_sdk_state(monkeypatch):
    import decimalai._config as cfg
    import decimalai.langchain as lc_mod
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {}
    cfg._client.list_manifests.return_value = {"manifests": []}
    cfg._sender._pending = []
    monkeypatch.setattr(lc_mod, "_manifest_id", None)
    monkeypatch.setattr(lc_mod, "_manifest_tracker", ManifestTracker())
    monkeypatch.setattr(lc_mod, "_skill_router_singleton", None)
    yield
    cfg._config = None
    cfg._client = None


def _llm_response(*tool_names):
    """A minimal LLMResult stand-in carrying an assistant tool-call message."""
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": {"i": i}, "id": f"call_{i}"}
            for i, name in enumerate(tool_names)
        ],
    )
    generation = type("Gen", (), {"text": "", "message": message})()
    return type("Res", (), {"generations": [[generation]], "llm_output": {}})()


class TestToolRecordsSurviveAGraph:
    def _drive_graph(self, handler, tool_names):
        """Replay the callback shape a LangGraph agent emits.

        The model runs under the `agent` node; the tools run under a
        SEPARATE `tools` node, which is exactly why parent-matching failed.
        """
        root, agent_node, llm_run, tools_node = (uuid4() for _ in range(4))
        handler.on_chain_start({"name": "LangGraph"}, {"messages": []}, run_id=root)
        handler.on_chain_start(
            {"name": "agent"}, {"messages": []}, run_id=agent_node, parent_run_id=root
        )
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"}, [[]], run_id=llm_run, parent_run_id=agent_node,
            invocation_params={"model_name": "gpt-4o-mini"},
        )
        handler.on_llm_end(_llm_response(*tool_names), run_id=llm_run,
                           parent_run_id=agent_node)
        handler.on_chain_end({}, run_id=agent_node, parent_run_id=root)
        handler.on_chain_start({"name": "tools"}, {}, run_id=tools_node,
                               parent_run_id=root)
        for name in tool_names:
            tool_run = uuid4()
            handler.on_tool_start({"name": name}, '{"i": 0}', run_id=tool_run,
                                  parent_run_id=tools_node)
            handler.on_tool_end(f"{name}-result", run_id=tool_run,
                                parent_run_id=tools_node)
        handler.on_chain_end({}, run_id=tools_node, parent_run_id=root)
        handler.on_chain_end({"messages": []}, run_id=root, parent_run_id=None)

    def test_graph_tool_call_reaches_the_model_turn_that_asked_for_it(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="graph-agent", auto_send=False)
        self._drive_graph(handler, ["get_weather"])
        trace = handler.build_trace()

        attached = [tc for call in trace.llm_calls for tc in call.tool_calls]
        assert len(attached) == 1, (
            "the tool record was dropped — a ToolCallRecord only reaches the "
            "platform through LlmCallRecord.tool_calls"
        )
        assert attached[0].tool_name == "get_weather"
        assert attached[0].result == "get_weather-result"

    def test_parallel_tool_calls_each_get_their_own_record(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="graph-agent", auto_send=False)
        self._drive_graph(handler, ["get_weather", "get_time"])
        trace = handler.build_trace()

        attached = [tc for call in trace.llm_calls for tc in call.tool_calls]
        assert sorted(tc.tool_name for tc in attached) == ["get_time", "get_weather"]

    def test_flat_parent_match_still_wins(self):
        """The exact-parent rule keeps working for a flat agent."""
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="flat-agent", auto_send=False)
        root, llm_run, tool_run = (uuid4() for _ in range(3))
        handler.on_chain_start({"name": "AgentExecutor"}, {"input": "x"}, run_id=root)
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"}, [[]], run_id=llm_run, parent_run_id=root,
            invocation_params={"model_name": "gpt-4o-mini"},
        )
        handler.on_llm_end(_llm_response("search"), run_id=llm_run, parent_run_id=root)
        handler.on_tool_start({"name": "search"}, "q", run_id=tool_run,
                              parent_run_id=root)
        handler.on_tool_end("hits", run_id=tool_run, parent_run_id=root)
        trace = handler.build_trace()

        attached = [tc for call in trace.llm_calls for tc in call.tool_calls]
        assert [tc.tool_name for tc in attached] == ["search"]


class _StubRouter:
    """Stands in for the SkillRouter singleton, counting consultations."""

    def __init__(self):
        self.calls = 0
        self._routing_id = None
        self._offered = []

    def build_prompt_fragment(self, query=None, **kwargs):
        self.calls += 1
        self._routing_id = "rt_" + "a" * 24
        self._offered = [f"skill-{i}" for i in range(30)]
        return "== SKILL MENU ==", self._routing_id

    def consume_routing_id(self):
        value, self._routing_id = self._routing_id, None
        return value

    def consume_offered_names(self):
        value, self._offered = self._offered, []
        return value

    def consume_delivered_names(self):
        return []

    def consume_loaded_names(self):
        return []


@pytest.fixture
def stub_router(monkeypatch):
    import decimalai.langchain as lc_mod
    import decimalai.skill_router as sr

    router = _StubRouter()
    monkeypatch.setattr(lc_mod, "_skill_router_singleton", router)
    monkeypatch.setattr(sr, "consume_last_offered_names", lambda: [])
    monkeypatch.setattr(sr, "consume_last_delivered_names", lambda: [])
    return router


class TestSkillLoaderOnlyReportsWhatItInjected:
    def test_lcel_prompt_value_is_injected_into(self, stub_router, monkeypatch):
        """`prompt | llm` hands the model a PromptValue.

        The old injector recognised only str and list, so it returned the
        PromptValue untouched — after stamping a routing_id and 30 offered
        names on the trace. The menu now actually reaches the model, which is
        what makes that telemetry true.
        """
        from langchain_core.language_models.chat_models import BaseChatModel

        import decimalai.langchain as lc_mod

        # `_install_skill_loader` rebinds BaseChatModel.invoke/ainvoke for the
        # whole process. Record them with monkeypatch FIRST so they are put
        # back at teardown — otherwise every later test in the session routes
        # its model calls through the injector, which builds a real SkillRouter
        # and talks to the network.
        monkeypatch.setattr(BaseChatModel, "invoke", BaseChatModel.invoke)
        monkeypatch.setattr(BaseChatModel, "ainvoke", BaseChatModel.ainvoke)
        monkeypatch.setattr(lc_mod, "_skill_loader_installed", False)
        lc_mod._install_skill_loader()

        seen = {}
        original_generate = NamedFakeChat._generate

        def spy(self, messages, *args, **kwargs):
            seen["messages"] = [str(getattr(m, "content", m)) for m in messages]
            return original_generate(self, messages, *args, **kwargs)

        monkeypatch.setattr(NamedFakeChat, "_generate", spy)

        handler = lc_mod.CallbackHandler(agent_name="lcel-agent", auto_send=False)
        prompt = ChatPromptTemplate.from_messages([("human", "{q}")])
        chain = (prompt | NamedFakeChat(responses=["A"])).with_config(
            run_name="LCEL", callbacks=[handler]
        )
        chain.invoke({"q": "hello"})
        trace = handler.build_trace()

        assert any("SKILL MENU" in m for m in seen["messages"]), (
            "the menu never reached the model, so the trace must not claim it"
        )
        assert trace.routing_id == "rt_" + "a" * 24
        assert len(trace.skills_offered_in_prompt) == 30

    def test_uninjectable_shape_consults_no_router_and_stamps_nothing(
        self, stub_router
    ):
        import decimalai.langchain as lc_mod

        lc_mod._set_routing_id(None)
        unsupported = object()

        result = lc_mod._inject_skills_into_input(unsupported)

        assert result is unsupported
        assert stub_router.calls == 0, (
            "consulting the Router is what mints a routing_id and fills its "
            "offered rail — a call we cannot inject into must not do it"
        )
        assert lc_mod._routing_id_ctx.get() is None

    @pytest.mark.parametrize(
        "shape",
        ["string", "message-list", "prompt-value"],
    )
    def test_supported_shapes_all_receive_the_menu(self, stub_router, shape):
        from langchain_core.messages import HumanMessage

        import decimalai.langchain as lc_mod

        value = {
            "string": "hello",
            "message-list": [HumanMessage(content="hello")],
            "prompt-value": ChatPromptTemplate.from_messages(
                [("human", "hello")]
            ).invoke({}),
        }[shape]

        result = lc_mod._inject_skills_into_input(value)

        assert isinstance(result, list)
        # Index 0 is still right HERE, and only here: none of these three
        # shapes carries a caller system message, so there is no cacheable
        # prefix in front for the menu to displace. The injector now inserts
        # after the caller's leading system messages when there are any —
        # that ordering contract lives in
        # tests/test_prompt_cache_prefix_ordering.py, which states it in cache
        # terms (bytes before the fragment are stable across queries) rather
        # than index terms. This test is about DELIVERY: every supported input
        # shape gets the menu at all.
        assert "SKILL MENU" in str(result[0].content)
        assert stub_router.calls == 1
