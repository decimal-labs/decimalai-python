"""Two LangChain adapter record defects.

* A tool's result was recorded as ``str(output)``. From a LangGraph tool node
  (``create_agent``, ``create_react_agent``) the output is a ``ToolMessage``, so
  the record held its repr (``content='...' name=... tool_call_id=...``)
  instead of what the tool said.

* An LLM call record took its model name from the invocation params only. A
  chat model that names no model there (a custom model class, or a wrapper
  that drops the field) produced ``model_name=None``, and ingest rejects the
  whole trace for that (``llm_calls[i]: 'model_name' is required``).
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, ToolMessage


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


def _tool_turn(handler, output):
    """One model turn asking for `lookup`, then the tool returning `output`."""
    root, llm_run, tool_run = (uuid4() for _ in range(3))
    handler.on_chain_start({"name": "LangGraph"}, {"messages": []}, run_id=root)
    handler.on_chat_model_start(
        {"name": "ChatOpenAI"}, [[]], run_id=llm_run, parent_run_id=root,
        invocation_params={"model_name": "gpt-4o-mini"},
    )
    message = AIMessage(
        content="", tool_calls=[{"name": "lookup", "args": {}, "id": "call_0"}],
    )
    generation = type("Gen", (), {"text": "", "message": message})()
    response = type("Res", (), {"generations": [[generation]], "llm_output": {}})()
    handler.on_llm_end(response, run_id=llm_run, parent_run_id=root)
    handler.on_tool_start({"name": "lookup"}, "{}", run_id=tool_run, parent_run_id=root)
    handler.on_tool_end(output, run_id=tool_run, parent_run_id=root)
    trace = handler.build_trace()
    [record] = [tc for call in trace.llm_calls for tc in call.tool_calls]
    [span] = [s for s in trace.spans if s.name == "lookup"]
    return record, span


class TestToolResultIsWhatTheToolSaid:
    def test_tool_message_records_its_content_not_its_repr(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="graph-agent", auto_send=False)
        record, span = _tool_turn(
            handler, ToolMessage(content="72F and sunny", tool_call_id="call_0"),
        )

        assert record.result == "72F and sunny"
        assert span.output_preview == "72F and sunny"

    def test_content_blocks_are_joined_by_their_text(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="graph-agent", auto_send=False)
        output = ToolMessage(
            content=[{"type": "text", "text": "first"}, "second"],
            tool_call_id="call_0",
        )
        record, _ = _tool_turn(handler, output)

        assert record.result == "first\nsecond"

    def test_a_plain_string_is_unchanged(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="flat-agent", auto_send=False)
        record, span = _tool_turn(handler, "hits")

        assert record.result == "hits"
        assert span.output_preview == "hits"


class TestEveryLlmCallHasAModelName:
    def test_unnamed_chat_model_falls_back_to_its_class_name(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="custom-agent", auto_send=False)
        root, llm_run = uuid4(), uuid4()
        handler.on_chain_start({"name": "Chain"}, {"input": "x"}, run_id=root)
        handler.on_chat_model_start(
            {"name": "MyChatModel", "id": ["my_pkg", "MyChatModel"]}, [[]],
            run_id=llm_run, parent_run_id=root, invocation_params={"_type": "custom"},
        )
        trace = handler.build_trace()

        assert trace.llm_calls[0].model_name == "MyChatModel"

    def test_completion_model_without_serialized_name_still_has_one(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="custom-agent", auto_send=False)
        root, llm_run = uuid4(), uuid4()
        handler.on_chain_start({"name": "Chain"}, {"input": "x"}, run_id=root)
        handler.on_llm_start({}, ["hi"], run_id=llm_run, parent_run_id=root)
        trace = handler.build_trace()

        assert trace.llm_calls[0].model_name == "unknown"

    def test_a_named_model_keeps_its_name(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="named-agent", auto_send=False)
        root, llm_run = uuid4(), uuid4()
        handler.on_chain_start({"name": "Chain"}, {"input": "x"}, run_id=root)
        handler.on_chat_model_start(
            {"name": "ChatOpenAI"}, [[]], run_id=llm_run, parent_run_id=root,
            invocation_params={"model_name": "gpt-4o-mini"},
        )
        trace = handler.build_trace()

        assert trace.llm_calls[0].model_name == "gpt-4o-mini"

    def test_a_real_unnamed_model_run_passes_the_ingest_rule(self):
        """Drive LangChain itself, so `serialized` has the shape it really sends."""
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="fake-agent", auto_send=False)
        FakeListChatModel(responses=["A"]).invoke("hi", config={"callbacks": [handler]})
        trace = handler.build_trace()

        payload = trace.model_dump(mode="json")
        # The rule ingest applies (platform trace_service): every LLM call needs one.
        assert payload["llm_calls"]
        assert all(call.get("model_name") for call in payload["llm_calls"])
        assert payload["llm_calls"][0]["model_name"] == "FakeListChatModel"
