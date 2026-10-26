"""Live-LLM — Google ADK through the native DecimalAI plugin.

ADK (the Agent Development Kit) is Gemini-native and ships its own plugin
system: a ``BasePlugin`` registered on the ``Runner`` receives before/after
callbacks for the run, every model generation, and every tool call. The
DecimalAI adapter (``decimalai.adk.DecimalAIPlugin``, put in ``App(plugins=[...])``)
builds **one** :class:`RunTrace` per ADK invocation from those callbacks — capturing
the LLM steps, the tool calls, and an auto-detected manifest of the root
agent (model / instruction / tools / sub-agents).

This test proves that path end-to-end with a **real model behind an ADK agent**:

  * Build a 2-tool (`get_price`, `calculate`) shopping agent.
  * Put ``DecimalAIPlugin`` in the ``App`` an ``InMemoryRunner`` runs, and run it.
  * Assert the assembled backend trace carries ≥1 LLM step, ≥1 tool call, and
    records the model — i.e. the native-plugin trace build works, not just that
    the agent ran.

ADK pairs with ``google`` and ``anthropic`` (the release gate's FRAMEWORK_PROVIDERS).
On Gemini the model is the plain id. On Claude it is ADK's own ``AnthropicLlm``
(``h.adk_model``), and the cell also checks tokens on a thinking model: Anthropic
counts thinking inside ``output_tokens``, ADK's converter splits it out as
``thoughts_token_count``, and the trace's output tokens must add it back. So the
cell asks for effort ``max``, watches the raw responses through its own
``AsyncAnthropic`` client, and requires that at least one call thought and that the
trace's output tokens equal what Anthropic billed. decimalai 0.26.0 and earlier
recorded the answer's tokens alone (fixed in 9bb3792).

Marker: live_llm + adk. Install the extra with ``pip install -e ".[adk-tests]"``.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from . import _live_helpers as h

#: Substring the recorded model id must carry, per provider.
_MODEL_SUBSTR = {"google": "gemini", "anthropic": "claude"}

#: Runs of the Claude cell before it gives up on getting a thinking block: thinking is
#: adaptive, so a call can come back without one, and then there is nothing to check.
_THINKING_ATTEMPTS = 3


def _thinking_tokens(usage) -> int:
    return getattr(getattr(usage, "output_tokens_details", None), "thinking_tokens", None) or 0


def _billing_client(billed: list):
    """An ``AsyncAnthropic`` that keeps each response's ``usage``: what Anthropic billed."""
    import anthropic

    client = anthropic.AsyncAnthropic(max_retries=2)
    create = client.messages.create

    async def create_and_keep_usage(**kwargs):
        message = await create(**kwargs)
        if getattr(message, "usage", None) is not None:
            billed.append(message.usage)
        return message

    client.messages.create = create_and_keep_usage
    return client


def _run_shopping(provider: str, model: str, agent_name: str, billed: list) -> str:
    """One ADK invocation of the shopping agent, traced by the plugin; returns the answer."""
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from decimalai.adk import DecimalAIPlugin

    def get_price(item: str) -> dict:
        """Return the unit price in dollars for a given item name."""
        return {"price": h.lookup_price(item)}

    def calculate(expression: str) -> dict:
        """Evaluate a simple arithmetic expression like '3*10+2*25'."""
        return {"result": h.safe_calculate(expression)}

    extra = {}
    if provider == "anthropic":
        from google.adk.models.anthropic_llm import AnthropicGenerateContentConfig

        # Haiku 5.5 thinks by default at effort medium, but not on a turn this easy: on
        # this prompt it thought 0 of 3 times at high and 3 of 3 at max (2026-10-26, about
        # 40 thinking tokens on the first call). The token check needs the thinking.
        extra["generate_content_config"] = AnthropicGenerateContentConfig(effort="max")

    # DecimalAI agent label (hyphenated, what we poll for) vs. ADK's node name,
    # which must be a valid Python identifier — keep them parallel but distinct.
    agent = LlmAgent(
        name=agent_name.replace("-", "_"),
        model=h.adk_model(provider, model,
                          client=_billing_client(billed) if provider == "anthropic" else None),
        instruction=(
            "You are a shopping assistant. Use get_price to look up unit prices "
            "and calculate to total them. Never guess a price — call the tool."
        ),
        tools=[get_price, calculate],
        **extra,
    )

    app_name = "decimal_adk_test"
    runner = InMemoryRunner(app=App(
        name=app_name, root_agent=agent, plugins=[DecimalAIPlugin(agent_name=agent_name)],
    ))

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

    return asyncio.run(_run())


@pytest.mark.live_llm
@pytest.mark.adk
@pytest.mark.parametrize("provider, model", h.matrix("adk"))
def test_adk_agent_traces_through_native_plugin(provider, model):
    """A real ADK agent → DecimalAIPlugin callbacks → one backend trace with the LLM
    step(s), the tool call(s), and the model; on Claude, the output tokens billed."""
    h.require_key_for(provider)
    pytest.importorskip("google.adk")
    if provider == "anthropic":
        pytest.importorskip("anthropic")

    # ADK's google-genai client authenticates with GOOGLE_API_KEY (falling back
    # to GEMINI_API_KEY on newer google-genai). Mirror the key across both names
    # and force API-key mode (not Vertex) so the run works with just the gate's
    # GEMINI_API_KEY. setdefault → never clobber an explicitly-set value.
    if os.environ.get("GEMINI_API_KEY"):
        os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")

    attempts = _THINKING_ATTEMPTS if provider == "anthropic" else 1
    for attempt in range(attempts):
        agent_name = h.unique_agent(f"adk-{provider}-shopping")
        billed: list = []
        result = _run_shopping(provider, model, agent_name, billed)
        if provider != "anthropic" or any(_thinking_tokens(u) for u in billed):
            break

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

    # The model id must survive into the LlmCallRecord.
    calls = detail.get("llm_calls") or []
    llm_models = " ".join(
        str(c.get("model_name") or c.get("model") or "") for c in calls
    ).lower()
    assert _MODEL_SUBSTR[provider] in llm_models, (
        f"ADK trace didn't record the {_MODEL_SUBSTR[provider]} model. llm_calls={calls}"
    )

    if provider != "anthropic":
        return

    # Tokens on a thinking model: what the trace says each call produced is what
    # Anthropic billed for it, thinking included.
    thinking = [_thinking_tokens(u) for u in billed]
    print(f"\n[adk-anthropic] attempt {attempt + 1}: {len(billed)} Claude calls, output "
          f"{[u.output_tokens for u in billed]}, thinking {thinking}, trace "
          f"{[c.get('output_tokens') for c in calls]}")
    assert any(thinking), (
        f"{attempts} runs at effort max came back without thinking, so the trace's "
        f"output tokens could not be checked against thinking. output_tokens per call: "
        f"{[u.output_tokens for u in billed]}"
    )
    assert len(calls) == len(billed), (
        f"the trace has {len(calls)} llm_calls for {len(billed)} Claude responses"
    )
    recorded = sum(int(c.get("output_tokens") or 0) for c in calls)
    paid = sum(u.output_tokens for u in billed)
    assert recorded == paid, (
        f"the trace records {recorded} output tokens; Anthropic billed {paid}, of which "
        f"{sum(thinking)} were thinking (per call: trace "
        f"{[c.get('output_tokens') for c in calls]}, billed {[u.output_tokens for u in billed]})"
    )
