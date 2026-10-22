"""Every ToolCallRecord was dropped from a LangGraph / `create_agent` trace.

A tool record only reaches the wire through `LlmCallRecord.tool_calls`, and the only rule for
attaching one was `tool_span.parent_span_id == llm_call.span_id`. In a graph the model runs under
the `agent` node and the tool under a sibling `tools` node, so that equality never held.

(This file also held the skill loader's honesty tests: it stamped a routing decision onto LCEL
runs it had injected nothing into. The loader is gone since 0.25.)
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage


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
