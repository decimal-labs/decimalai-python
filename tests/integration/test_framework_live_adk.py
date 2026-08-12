"""Live-LLM — Google ADK through the native DecimalAI plugin.

ADK (the Agent Development Kit) is Gemini-native and ships its own plugin
system: a ``BasePlugin`` registered on the ``Runner`` receives before/after
callbacks for the run, every model generation, and every tool call. The
DecimalAI adapter (``decimalai.adk.DecimalaiPlugin`` / ``install()``) builds
**one** :class:`RunTrace` per ADK invocation from those callbacks — capturing
the LLM steps, the tool calls, and an auto-detected manifest of the root
agent (model / instruction / tools / sub-agents).

This test proves that path end-to-end with a **real Gemini-backed ADK agent**:

  * Build a 2-tool (`get_price`, `calculate`) shopping agent on a Gemini model.
  * Attach ``DecimalaiPlugin`` to an ``InMemoryRunner`` and actually run it.
  * Assert the assembled backend trace carries ≥1 LLM step, ≥1 tool call, and
    records the Gemini model — i.e. the native-plugin trace build works, not
    just that the agent ran.

ADK pairs with the ``google`` provider only (the release gate enforces this),
so the matrix is Gemini-only.

Marker: live_llm + adk. Install the extra with ``pip install -e ".[adk-tests]"``.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from . import _live_helpers as h


@pytest.mark.live_llm
@pytest.mark.adk
@pytest.mark.parametrize("provider, model", h.matrix("adk"))
def test_adk_agent_traces_through_native_plugin(provider, model):
    """A real ADK agent on Gemini → DecimalaiPlugin callbacks → one backend
    trace with the LLM step(s), the tool call(s), and the Gemini model."""
    h.require_key_for(provider)
    pytest.importorskip("google.adk")

    # ADK's google-genai client authenticates with GOOGLE_API_KEY (falling back
    # to GEMINI_API_KEY on newer google-genai). Mirror the key across both names
    # and force API-key mode (not Vertex) so the run works with just the gate's
    # GEMINI_API_KEY. setdefault → never clobber an explicitly-set value.
    if os.environ.get("GEMINI_API_KEY"):
        os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")

    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from decimalai.adk import DecimalaiPlugin

    # DecimalAI agent label (hyphenated, what we poll for) vs. ADK's node name,
    # which must be a valid Python identifier — keep them parallel but distinct.
    agent_name = h.unique_agent("adk-gemini-shopping")
    adk_node_name = agent_name.replace("-", "_")

    def get_price(item: str) -> dict:
        """Return the unit price in dollars for a given item name."""
        return {"price": h.lookup_price(item)}

    def calculate(expression: str) -> dict:
        """Evaluate a simple arithmetic expression like '3*10+2*25'."""
        return {"result": h.safe_calculate(expression)}

    agent = LlmAgent(
        name=adk_node_name,
        model=model,
        instruction=(
            "You are a shopping assistant. Use get_price to look up unit prices "
            "and calculate to total them. Never guess a price — call the tool."
        ),
        tools=[get_price, calculate],
    )

    app_name = "decimal-adk-test"
    runner = InMemoryRunner(
        agent=agent,
        app_name=app_name,
        plugins=[DecimalaiPlugin(agent_name=agent_name)],
    )

    user_msg = types.Content(role="user", parts=[types.Part(text=(
        "A cart has 3 widgets and 2 gadgets. Use get_price to look up the unit "
        "price of 'widget' and of 'gadget', then use calculate with the "
        "expression '3*<widget_price>+2*<gadget_price>'. Reply with only the "
        "final total number."
    ))])

    async def _run() -> str:
        session = await runner.session_service.create_session(
            app_name=app_name, user_id="u1",
        )
        final_text = ""
        async for event in runner.run_async(
            user_id="u1", session_id=session.id, new_message=user_msg,
        ):
            if event.is_final_response() and event.content and event.content.parts:
                final_text = "".join(p.text or "" for p in event.content.parts)
        return final_text

    result = asyncio.run(_run())

    assert str(h.SHOPPING_EXPECTED_TOTAL) in result.replace(",", "").replace("$", ""), (
        f"ADK agent didn't compute the expected total {h.SHOPPING_EXPECTED_TOTAL}: {result!r}"
    )

    h.flush_sdk_sender()

    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])

    # The load-bearing assertion: the native-plugin trace build captured the
    # model generation(s) and tool call(s), plus an auto-detected manifest.
    h.assert_rich_agent_trace(
        detail, min_llm_calls=1, min_tool_calls=1, min_distinct_tools=1,
    )

    # The Gemini model id must survive into the LlmCallRecord.
    llm_models = " ".join(
        str(c.get("model_name") or c.get("model") or "")
        for c in detail.get("llm_calls", [])
    ).lower()
    assert "gemini" in llm_models, (
        f"ADK trace didn't record the Gemini model. llm_calls={detail.get('llm_calls')}"
    )
