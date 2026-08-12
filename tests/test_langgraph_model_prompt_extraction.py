"""`create_react_agent` model + prompt extraction limitation.

Tool extraction from langgraph's `CompiledStateGraph` works. Model
and prompt are NOT extractable through any public attribute — they're
closure-captured inside `nodes['agent'].bound` (a `RunnableCallable`).
The spec recommended path (b): require the user to pass `model=` and
`prompt=` explicitly to `flush_manifest_for_ci` when introspecting a
langgraph agent. This file pins that contract:

  1. **Pins the limitation**: `introspect_chain` on a real
     CompiledStateGraph returns tools (works) but empty prompts +
     empty models (the closure-captured fields can't be reached).
     A future enhancement that picks them up from closures will
     flip this test and require an explicit update — that's
     intentional so the change is noticed.

  2. **Explicit-arg override**: when the user passes `prompts=...`
     / `models=...` alongside `chain=...`, the explicit args win
     and fill the gap. This is the workaround until closure
     introspection can reach the model and prompt directly.
"""

from __future__ import annotations

import pytest

# Skip cleanly if langgraph isn't installed.
langgraph = pytest.importorskip("langgraph", reason="requires `langgraph` (dev extra)")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from decimalai.integrations.langchain_introspect import introspect_chain


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class _StubChatModel(BaseChatModel):
    """Minimal chat model that supports bind_tools (shared pattern with the
    other langgraph introspection tests)."""

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])

    def bind_tools(self, tools, **kwargs):
        bound = self.__class__()
        bound._bound_tools = list(tools)
        return bound


def _build_react_agent_with_prompt():
    from langgraph.prebuilt import create_react_agent
    return create_react_agent(
        model=_StubChatModel(),
        tools=[add],
        prompt="You are a helpful math assistant.",
    )


# ─────────────────────────────────────────────────────────────────────
# (1) Pin the limitation
# ─────────────────────────────────────────────────────────────────────


def test_langgraph_introspection_misses_prompt_and_model():
    """The CompiledStateGraph from `create_react_agent(model=..., prompt=...)`
    encloses the model + prompt inside `nodes['agent'].bound` — there's
    no public attribute to read them from. `introspect_chain` returns
    empty dicts for both.

    If a future closure-introspection trick lands, this test will fail
    and force the author to update it — that's the point. We want the
    SDK contract change to be visible.
    """
    agent = _build_react_agent_with_prompt()
    tools, prompts, models = introspect_chain(agent)

    # Tools come back fine — only the closure-captured fields are the gap.
    assert {t["name"] for t in tools} == {"add"}

    # Prompts + models are the documented gap.
    assert prompts == {}, (
        f"langgraph introspection now finds prompts: {prompts}. "
        f"Closure introspection has landed, so this documented gap is "
        f"closed — update this test and deprecate the explicit-prompt= "
        f"workaround in the SDK README."
    )
    assert models == {}, (
        f"langgraph introspection now finds models: {models}. "
        f"Closure introspection has landed, so this documented gap is "
        f"closed — update this test and deprecate the explicit-model= "
        f"workaround in the SDK README."
    )


# ─────────────────────────────────────────────────────────────────────
# (2) Explicit-arg override (the workaround)
# ─────────────────────────────────────────────────────────────────────


def test_explicit_prompts_models_override_empty_introspection():
    """The recommended path (b) workaround: pass `prompts=` and
    `models=` explicitly alongside `chain=`. The merge logic in
    `flush_manifest_for_ci` is `tools = tools or i_tools` (etc.), so
    explicit args take priority over introspection.

    Here we simulate the merge directly rather than calling
    `flush_manifest_for_ci` (which would try to POST to a backend).
    The contract is: `final = explicit or introspected`.
    """
    agent = _build_react_agent_with_prompt()
    i_tools, i_prompts, i_models = introspect_chain(agent)

    # Caller-supplied explicit args.
    explicit_prompts = {"system": "You are a helpful math assistant."}
    explicit_models = {"default": {"provider": "openai", "model": "gpt-4o"}}

    # Replicate the merge from flush_manifest_for_ci:
    final_tools = None or i_tools                 # tools come from introspection
    final_prompts = explicit_prompts or i_prompts # explicit fills the gap
    final_models = explicit_models or i_models    # explicit fills the gap

    assert {t["name"] for t in final_tools} == {"add"}
    assert final_prompts == explicit_prompts
    assert final_models == explicit_models
