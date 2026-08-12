"""Tests for langchain_introspect — static manifest extraction from chain objects.

Uses real langchain-core primitives (it's a SDK dependency anyway) so the
tests exercise the actual shapes the introspection will see in customers'
init scripts.
"""

from __future__ import annotations

from typing import Type

import pytest
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from decimalai.integrations.langchain_introspect import (
    introspect_chain,
    _extract_models,
    _extract_prompts,
    _extract_tools,
    _infer_provider,
)


# ─────────────────────────────────────────────────────────────────────
# Fixture builders
# ─────────────────────────────────────────────────────────────────────


class SearchArgs(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10)


class RefundArgs(BaseModel):
    order_id: str
    reason: str


def _search_fn(query: str, limit: int = 10) -> str:
    return f"results for {query}"


def _refund_fn(order_id: str, reason: str) -> str:
    return "refunded"


def make_search_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_search_fn,
        name="search_docs",
        description="Search the docs",
        args_schema=SearchArgs,
    )


def make_refund_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_refund_fn,
        name="refund_order",
        description="Issue a refund",
        args_schema=RefundArgs,
    )


class FakeChatOpenAI:
    """Minimal stand-in for ChatOpenAI for tests that don't need a real client.

    The introspection checks attribute presence via `getattr`, so this is
    structurally indistinguishable from the real class for our purposes.
    """
    def __init__(self, model="gpt-4o", temperature=0.0, max_tokens=None):
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens


class FakeAnthropic:
    def __init__(self, model="claude-4-7"):
        self.model = model
        self.temperature = 0.2


class FakeReactAgent:
    """Mimics the shape of `create_react_agent(llm, tools, prompt)` results."""
    def __init__(self, tools, llm, prompt):
        self.tools = tools
        self.llm = llm
        self.prompt = prompt


class FakeAgentExecutor:
    """Mimics AgentExecutor wrapping a deeper agent."""
    def __init__(self, agent):
        self.agent = agent


# ─────────────────────────────────────────────────────────────────────
# Tool extraction
# ─────────────────────────────────────────────────────────────────────


class TestExtractTools:
    def test_extracts_tools_from_top_level(self):
        chain = FakeReactAgent(
            tools=[make_search_tool(), make_refund_tool()],
            llm=FakeChatOpenAI(),
            prompt=PromptTemplate.from_template("Be helpful."),
        )
        tools = _extract_tools(chain)
        assert {t["name"] for t in tools} == {"search_docs", "refund_order"}

    def test_extracts_tools_via_agent_wrapper(self):
        inner = FakeReactAgent(
            tools=[make_search_tool()],
            llm=FakeChatOpenAI(),
            prompt=PromptTemplate.from_template("x"),
        )
        executor = FakeAgentExecutor(agent=inner)
        tools = _extract_tools(executor)
        assert [t["name"] for t in tools] == ["search_docs"]

    def test_returns_empty_when_no_tools_attribute(self):
        class NoTools:
            llm = FakeChatOpenAI()
        assert _extract_tools(NoTools()) == []

    def test_tool_schema_includes_field_definitions(self):
        chain = FakeReactAgent(
            tools=[make_search_tool()],
            llm=FakeChatOpenAI(),
            prompt=PromptTemplate.from_template("x"),
        )
        tools = _extract_tools(chain)
        schema = tools[0]["schema"]
        # Pydantic-generated JSON schema has properties → query + limit
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]

    def test_deduplicates_tools_by_name(self):
        t = make_search_tool()
        chain = FakeReactAgent(tools=[t, t, t], llm=FakeChatOpenAI(), prompt=None)
        tools = _extract_tools(chain)
        assert len(tools) == 1


# ─────────────────────────────────────────────────────────────────────
# Prompt extraction
# ─────────────────────────────────────────────────────────────────────


class TestExtractPrompts:
    def test_extracts_from_prompt_template_template_attribute(self):
        chain = FakeReactAgent(
            tools=[],
            llm=FakeChatOpenAI(),
            prompt=PromptTemplate.from_template("You are a research assistant."),
        )
        prompts = _extract_prompts(chain)
        assert prompts == {"system": "You are a research assistant."}

    def test_extracts_from_chat_prompt_template_messages(self):
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are helpful."),
            ("human", "{query}"),
        ])
        chain = FakeReactAgent(tools=[], llm=FakeChatOpenAI(), prompt=chat_prompt)
        prompts = _extract_prompts(chain)
        assert "system" in prompts
        # The combined message texts should mention both segments
        text = prompts["system"]
        assert "helpful" in text
        assert "{query}" in text

    def test_returns_empty_when_no_prompt(self):
        class NoPrompt:
            tools = []
            llm = FakeChatOpenAI()
        assert _extract_prompts(NoPrompt()) == {}

    def test_walks_agent_wrapper(self):
        inner = FakeReactAgent(
            tools=[],
            llm=FakeChatOpenAI(),
            prompt=PromptTemplate.from_template("Wrapped prompt."),
        )
        executor = FakeAgentExecutor(agent=inner)
        prompts = _extract_prompts(executor)
        assert prompts["system"] == "Wrapped prompt."


# ─────────────────────────────────────────────────────────────────────
# Model extraction
# ─────────────────────────────────────────────────────────────────────


class TestExtractModels:
    def test_extracts_openai_model(self):
        chain = FakeReactAgent(
            tools=[], llm=FakeChatOpenAI(model="gpt-4o", temperature=0.3), prompt=None,
        )
        models = _extract_models(chain)
        assert "default" in models
        m = models["default"]
        assert m["provider"] == "openai"
        assert m["model"] == "gpt-4o"
        assert m["temperature"] == 0.3

    def test_extracts_anthropic_model(self):
        chain = FakeReactAgent(
            tools=[], llm=FakeAnthropic(model="claude-4-7"), prompt=None,
        )
        models = _extract_models(chain)
        assert models["default"]["provider"] == "anthropic"
        assert models["default"]["model"] == "claude-4-7"

    def test_walks_to_find_llm_through_wrapper(self):
        inner = FakeReactAgent(tools=[], llm=FakeChatOpenAI(), prompt=None)
        executor = FakeAgentExecutor(agent=inner)
        models = _extract_models(executor)
        assert "default" in models

    def test_returns_empty_when_no_llm(self):
        class NoLLM:
            tools = []
            prompt = None
        assert _extract_models(NoLLM()) == {}


class TestInferProvider:
    @pytest.mark.parametrize("class_name,expected_provider", [
        ("ChatOpenAI", "openai"),
        ("OpenAI", "openai"),
        ("ChatAnthropic", "anthropic"),
        ("ChatVertexAI", "google"),
        ("ChatGoogleGenerativeAI", "google"),
        ("ChatGemini", "google"),
        ("ChatMistralAI", "mistral"),
        ("ChatCohere", "cohere"),
        ("ChatGroq", "groq"),
        ("ChatOllama", "ollama"),
        ("MysteryLLM", "unknown"),
    ])
    def test_infers_provider_from_class_name(self, class_name, expected_provider):
        fake_class = type(class_name, (), {})
        instance = fake_class()
        assert _infer_provider(instance) == expected_provider


# ─────────────────────────────────────────────────────────────────────
# Full introspect_chain (composite of the above)
# ─────────────────────────────────────────────────────────────────────


class TestIntrospectChain:
    def test_extracts_full_react_agent(self):
        chain = FakeReactAgent(
            tools=[make_search_tool(), make_refund_tool()],
            llm=FakeChatOpenAI(model="gpt-4o", temperature=0.2),
            prompt=ChatPromptTemplate.from_messages([
                ("system", "You are a customer support assistant."),
            ]),
        )
        tools, prompts, models = introspect_chain(chain)
        assert {t["name"] for t in tools} == {"search_docs", "refund_order"}
        assert "support" in prompts["system"]
        assert models["default"]["model"] == "gpt-4o"

    def test_handles_partial_chains_gracefully(self):
        """A chain with tools but no prompt or model should still extract tools."""
        class ToolsOnly:
            tools = [make_search_tool()]
        tools, prompts, models = introspect_chain(ToolsOnly())
        assert len(tools) == 1
        assert prompts == {}
        assert models == {}

    def test_handles_empty_chain(self):
        class Empty:
            pass
        tools, prompts, models = introspect_chain(Empty())
        assert tools == []
        assert prompts == {}
        assert models == {}
