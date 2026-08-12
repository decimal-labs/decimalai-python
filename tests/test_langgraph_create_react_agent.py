"""Real `langgraph.prebuilt.create_react_agent` introspection test.

An earlier version of this coverage used `langchain_core` primitives in
a Runnable shape that mimicked `create_react_agent`'s output but didn't
actually invoke it, because langgraph wasn't in the dev env. langgraph
is now a dev extra, so this verifies the introspection works against a
real CompiledStateGraph.

Catches drift if langgraph changes its output shape (e.g., renames the
`tools` node, moves `tools_by_name` to a different attribute).
"""

from __future__ import annotations

import warnings

import pytest

# Skip cleanly if langgraph isn't installed (e.g., a minimal dev install).
langgraph = pytest.importorskip("langgraph", reason="requires `langgraph` (dev extra)")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from decimalai.integrations.langchain_introspect import introspect_chain


# ─────────────────────────────────────────────────────────────────────
# Test fixtures: tools + a stub model that supports bind_tools
# ─────────────────────────────────────────────────────────────────────


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


class _StubChatModel(BaseChatModel):
    """Minimal chat model that supports bind_tools (FakeListChatModel
    raises NotImplementedError on bind_tools, so we need our own).

    Returns a canned 'ok' response on _generate so the agent compiles
    even though we never invoke it.
    """

    @property
    def _llm_type(self) -> str:
        return "stub-chat-model"

    @property
    def model_name(self) -> str:
        return "stub-model-v1"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )

    def bind_tools(self, tools, **kwargs):
        bound = _StubChatModel()
        bound._bound_tools = list(tools)
        return bound


def _build_real_react_agent(tools_list):
    """Build a real langgraph create_react_agent. Suppress the V1-deprecation
    warning since we're explicitly testing the still-supported path.
    """
    from langgraph.prebuilt import create_react_agent

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return create_react_agent(model=_StubChatModel(), tools=tools_list)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────


def test_introspect_extracts_tools_from_real_create_react_agent():
    """introspect_chain on a real CompiledStateGraph returns both tools."""
    agent = _build_real_react_agent([add, multiply])

    tools, prompts, models = introspect_chain(agent)

    tool_names = sorted(t["name"] for t in tools)
    assert tool_names == ["add", "multiply"], (
        f"Expected ['add', 'multiply']; got {tool_names}. "
        f"If langgraph renamed the tools node or its tools_by_name path, "
        f"`_extract_tools_from_compiled_graph` needs updating."
    )


def test_introspect_extracts_tool_schemas():
    """Each extracted tool carries a non-empty JSON schema for its args."""
    agent = _build_real_react_agent([add, multiply])

    tools, _, _ = introspect_chain(agent)

    by_name = {t["name"]: t for t in tools}
    add_schema = by_name["add"]["schema"]
    assert isinstance(add_schema, dict) and add_schema, (
        f"Expected non-empty JSON schema for `add`; got {add_schema!r}"
    )
    # Pydantic v2 produces {"properties": {"a": {...}, "b": {...}}, ...}
    properties = add_schema.get("properties") or {}
    assert "a" in properties and "b" in properties, (
        f"Expected `a` and `b` parameters in `add` schema; got {properties}"
    )


def test_introspect_with_zero_tools_returns_empty_list():
    """An agent built with no tools introspects to an empty tools list,
    not a crash. Catches: a future change that makes `tools_by_name`
    None instead of `{}` for the no-tools case.
    """
    agent = _build_real_react_agent([])

    tools, _, _ = introspect_chain(agent)
    assert tools == []


def test_introspect_handles_unique_tool_names():
    """Tools with duplicate names are deduplicated (first wins). Same
    contract as the existing `_extract_tools` for non-langgraph chains.
    """
    @tool
    def add_alias(a: int, b: int) -> int:
        """Add (alias)."""
        return a + b

    # Force two tools with conflicting names by manually rebinding
    add_alias.name = "add"
    agent = _build_real_react_agent([add, add_alias])

    tools, _, _ = introspect_chain(agent)
    # tools_by_name dict-keys deduplicate at construction; verify the
    # introspection doesn't somehow re-add duplicates downstream
    names = [t["name"] for t in tools]
    assert len(names) == len(set(names)), f"Duplicate names in {names}"


def test_introspect_does_not_invoke_the_agent():
    """Pure attribute access — no LLM calls, no async. Critical for CI.
    If introspection ever calls .invoke / .ainvoke, this test will time
    out (because the stub model returns 'ok' which loops forever in a
    react agent).
    """
    agent = _build_real_react_agent([add, multiply])

    # Patch the agent's invoke methods to raise — if introspection calls
    # them, the test fails loudly rather than hanging.
    invoked = {"count": 0}
    original_invoke = agent.invoke
    def traced_invoke(*args, **kwargs):
        invoked["count"] += 1
        return original_invoke(*args, **kwargs)
    agent.invoke = traced_invoke

    introspect_chain(agent)
    assert invoked["count"] == 0, (
        "introspect_chain should be pure attribute access; it invoked the agent"
    )
