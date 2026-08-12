"""Tests for OpenAI Agents SDK introspection.

Verifies the `(tools, prompts, models)` extraction shape matches what
`flush_manifest_for_ci` expects, using real `agents` SDK primitives.
"""

import pytest

# Skip the whole module if the SDK isn't installed
agents = pytest.importorskip("agents")

from agents import Agent, function_tool  # noqa: E402

from decimalai.integrations.openai_agents_introspect import (
    introspect_agent,
    _extract_models,
    _extract_prompts,
    _extract_tools,
)


# ─────────────────────────────────────────────────────────────────────
# Tool extraction
# ─────────────────────────────────────────────────────────────────────


@function_tool
def search_docs(query: str) -> str:
    """Search the documentation for the given query."""
    return query


@function_tool
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_introspect_extracts_function_tools_with_schemas():
    agent = Agent(
        name="docs-bot",
        instructions="Help users search the docs.",
        tools=[search_docs, add_numbers],
        model="gpt-4o",
    )

    tools, _, _ = introspect_agent(agent)

    assert len(tools) == 2
    names = {t["name"] for t in tools}
    assert names == {"search_docs", "add_numbers"}

    # FunctionTool.params_json_schema flows through unchanged
    by_name = {t["name"]: t for t in tools}
    search_schema = by_name["search_docs"]["schema"]
    assert search_schema["type"] == "object"
    assert "query" in search_schema["properties"]


def test_introspect_handles_agent_with_no_tools():
    agent = Agent(name="empty", instructions="Just chat.", model="gpt-4o")
    tools, _, _ = introspect_agent(agent)
    assert tools == []


def test_extract_tools_dedupes_by_name():
    """If somehow the same tool appears twice, only count it once."""
    agent = Agent(
        name="dupe",
        instructions="x",
        tools=[search_docs, search_docs],  # same tool twice
        model="gpt-4o",
    )
    tools = _extract_tools(agent)
    assert len(tools) == 1
    assert tools[0]["name"] == "search_docs"


def test_extract_tools_safely_handles_non_list_value():
    """If someone passes a non-list (corrupted state), don't crash."""

    class BadAgent:
        tools = "not-a-list"

    tools = _extract_tools(BadAgent())
    assert tools == []


# ─────────────────────────────────────────────────────────────────────
# Prompt extraction
# ─────────────────────────────────────────────────────────────────────


def test_introspect_extracts_static_instructions_as_system_prompt():
    agent = Agent(
        name="static",
        instructions="You are a helpful documentation assistant.",
        model="gpt-4o",
    )
    _, prompts, _ = introspect_agent(agent)
    assert prompts == {"system": "You are a helpful documentation assistant."}


def test_introspect_extracts_callable_instructions_as_dynamic_marker():
    """Dynamic prompt callables get a placeholder so structural diff can
    detect the change — we can't capture the actual text without invoking.
    """
    def my_dynamic_prompt(ctx, agent):
        """Compute the system prompt at runtime."""
        return "dynamic"

    agent = Agent(
        name="dynamic-agent",
        instructions=my_dynamic_prompt,
        model="gpt-4o",
    )
    _, prompts, _ = introspect_agent(agent)
    assert "system_dynamic" in prompts
    assert "my_dynamic_prompt" in prompts["system_dynamic"]


def test_introspect_returns_empty_prompts_when_instructions_none():
    # Agent allows instructions=None — bare agent
    agent = Agent(name="bare", model="gpt-4o")
    _, prompts, _ = introspect_agent(agent)
    assert prompts == {}


# ─────────────────────────────────────────────────────────────────────
# Model extraction
# ─────────────────────────────────────────────────────────────────────


def test_introspect_extracts_model_from_string_value():
    agent = Agent(name="m1", instructions="x", model="gpt-4o")
    _, _, models = introspect_agent(agent)
    assert "default" in models
    assert models["default"]["model"] == "gpt-4o"
    assert models["default"]["provider"] == "openai"


def test_introspect_handles_no_model():
    agent = Agent(name="no-model", instructions="x", model=None)
    _, _, models = introspect_agent(agent)
    assert models == {}


def test_extract_models_infers_anthropic_provider_from_claude_name():
    """If the model name hints at Anthropic, provider should reflect it."""

    class FakeAgent:
        model = "claude-3-5-sonnet"
        model_settings = None

    models = _extract_models(FakeAgent())
    assert models["default"]["provider"] == "anthropic"


# ─────────────────────────────────────────────────────────────────────
# End-to-end shape
# ─────────────────────────────────────────────────────────────────────


def test_introspect_returns_three_tuple_with_correct_shapes():
    """Smoke test: result is unpacked the same way `flush_manifest_for_ci`
    would consume it.
    """
    agent = Agent(
        name="full",
        instructions="Be helpful.",
        tools=[search_docs],
        model="gpt-4o",
    )
    result = introspect_agent(agent)
    assert isinstance(result, tuple) and len(result) == 3
    tools, prompts, models = result
    assert isinstance(tools, list)
    assert isinstance(prompts, dict)
    assert isinstance(models, dict)
    assert tools[0]["name"] == "search_docs"
    assert prompts["system"] == "Be helpful."
    assert models["default"]["model"] == "gpt-4o"
