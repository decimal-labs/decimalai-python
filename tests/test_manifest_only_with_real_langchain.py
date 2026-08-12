"""Behavior verification: manifest_only mode + real LangChain primitives.

Previously, `flush_manifest_for_ci(chain=...)` was tested only with a
hand-rolled FakeChain. This adds a verification using real
`langchain_core` primitives (StructuredTool, ChatPromptTemplate,
FakeListChatModel) composed in a Runnable shape closer to what
`create_react_agent` returns.

Scope: this file stays on `langchain_core` primitives only. Coverage
against a real `create_react_agent` output lives in
`test_langgraph_create_react_agent.py`, which needs the `langgraph` dev
extra and skips when it is absent.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

import decimalai
from decimalai import _config as _cfg


class _SearchArgs(BaseModel):
    query: str = Field(description="Search query")


class _CalcArgs(BaseModel):
    a: int
    b: int


def _build_realistic_chain():
    """Construct a Runnable wrapping real LangChain primitives in a shape
    that mimics `create_react_agent` output: an outer Runnable exposing
    `.bound` containing tools/prompt/llm.
    """
    search_tool = StructuredTool.from_function(
        func=lambda query: f"results for {query}",
        name="search_docs",
        description="Search the docs",
        args_schema=_SearchArgs,
    )
    calc_tool = StructuredTool.from_function(
        func=lambda a, b: a + b,
        name="add",
        description="Add two integers",
        args_schema=_CalcArgs,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        ("human", "{question}"),
    ])
    llm = FakeListChatModel(responses=["hello"])
    # The introspector looks for model_name/model/deployment_name to identify
    # the LLM. Real chat models (ChatOpenAI etc.) provide these; the test
    # stub doesn't, so set it explicitly to exercise the model-extraction path.
    object.__setattr__(llm, "model_name", "gpt-4o")
    object.__setattr__(llm, "temperature", 0.1)

    # Mimic create_react_agent: outer Runnable with `.bound` exposing
    # tools/prompt/llm. The introspector walks `bound` as one of its
    # known nesting attributes.
    class FakeBound:
        def __init__(self):
            self.tools = [search_tool, calc_tool]
            self.prompt = prompt
            self.llm = llm

    class FakeAgentRunnable(Runnable):
        def __init__(self):
            self.bound = FakeBound()

        def invoke(self, input, config=None, **kwargs):
            return input  # never actually called in CI path

    return FakeAgentRunnable()


@pytest.fixture
def clean_env(monkeypatch):
    # Strip any DECIMAL_* / DECIMALAI_* vars to isolate the test
    for key in list(__import__("os").environ.keys()):
        if key.startswith(("DECIMAL_", "DECIMALAI_")):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_flush_manifest_for_ci_with_realistic_chain_in_manifest_only_mode(
    clean_env, tmp_path, monkeypatch,
):
    """Build a Runnable with real langchain_core primitives, call
    flush_manifest_for_ci with manifest_only mode enabled. Expect:

      1. The manifest registration call captures all 3 surfaces (tools,
         prompt, llm).
      2. No trace-ingestion HTTP calls were made (manifest_only bouncer
         engaged).
    """
    clean_env.setenv("DECIMAL_API_KEY", "dai_sk_test")
    clean_env.setenv("DECIMALAI_MODE", "manifest_only")
    decimalai.init()
    monkeypatch.chdir(tmp_path)

    captured = {}
    mock_client = MagicMock()

    def capture_register(snapshot):
        captured["agent_name"] = snapshot.agent_name
        captured["components"] = [
            (c.component_type, c.component_name) for c in snapshot.components
        ]
        return {"manifest_id": "mfst_realistic"}

    # Track ingestion-side calls too — they should NOT happen in manifest_only.
    mock_client.register_manifest.side_effect = capture_register
    mock_client.ingest_trace = MagicMock()
    mock_client.ingest_traces_batch = MagicMock()
    _cfg._client = mock_client

    chain = _build_realistic_chain()
    result = decimalai.flush_manifest_for_ci(
        agent_name="real-langchain-agent",
        chain=chain,
    )

    assert result["manifest_id"] == "mfst_realistic"
    assert captured["agent_name"] == "real-langchain-agent"

    # All 3 surfaces were found by introspection
    comp_types = {ctype for ctype, _ in captured["components"]}
    comp_names = {name for _, name in captured["components"]}

    assert "tool" in comp_types, f"tool surface missing from {captured['components']}"
    assert "prompt" in comp_types, f"prompt surface missing from {captured['components']}"
    assert "model" in comp_types, f"model surface missing from {captured['components']}"

    # Both tools picked up
    assert "search_docs" in comp_names
    assert "add" in comp_names

    # The bouncer worked: no trace ingestion happened during the flush
    mock_client.ingest_trace.assert_not_called()
    mock_client.ingest_traces_batch.assert_not_called()
