"""Live-LLM — a skill's BODY reaches a real Gemini-backed ADK agent's ANSWER.

An agent built from ``decimalai.adk.load_agent()`` gets its saved prompt, the
list of skills it may load, and a ``load_skill`` tool. This proves the whole
path against a real model and a real backend: the model reads the list, calls
``load_skill``, and answers from the body it got back.

The conformance suite's C14 grades the PROMPT — it proves a body was put in
front of the model, which is the right bar for a hermetic stub run and is
deliberately not this one. Here a real model has to USE it.

WHAT MAKES IT NON-CIRCULAR
--------------------------
The fact is minted per run: a random hex token that has never existed before,
embedded in the skill body and asked for by a question that supplies no other
route to it. The agent's saved prompt does not mention it and no other tool
returns it. A correct answer is therefore only reachable through the body
``load_skill`` served.

The negative control is a plain ADK agent with the same instruction and no
``load_agent()``: same question, no skills. If the control also answers with
the token, the token leaked from somewhere other than the skill and the
positive result means nothing — so the control failing to produce it is part
of the proof, not a nicety.

Marker: live_llm + adk. Needs a funded Gemini key and a reachable backend.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from . import _live_helpers as h

#: Long enough that the model cannot stumble onto it, and shaped so a refusal
#: ("I don't know the code") can never be mistaken for a pass.
_TOKEN = uuid.uuid4().hex[:10].upper()

_SKILL_BODY = f"""# Damaged-shipment settlement procedure

This is the internal settlement procedure for shipments that arrive damaged.
It is not published anywhere outside this document.

## The settlement code

When a shipment arrives damaged, the settlement code to quote on the claim is
`SETTLE-{_TOKEN}`. Quote it exactly, including the prefix.

## Amounts

A damaged shipment is refunded in full, plus a 15% goodwill credit applied to
the customer's next order. Do not offer a replacement unless asked.
"""

#: Deliberately says nothing about settlement codes. The only path to the token
#: is the skill body ``load_skill`` serves.
_INSTRUCTION = (
    "You are a customer support agent. Answer using the procedures "
    "available to you. If you do not know something, say so."
)

_QUERY = (
    "A customer's shipment arrived damaged. What is the settlement code I "
    "should quote on the claim? Reply with the code and nothing else."
)

_APP = "decimal_adk_skill_delivery"


def _adk_env() -> None:
    """ADK's google-genai client reads GOOGLE_API_KEY; mirror the gate's key."""
    if os.environ.get("GEMINI_API_KEY"):
        os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")


def _run_agent(model: str, agent_name: str, *, skills: bool) -> str:
    """One ADK invocation, traced by the plugin; returns the model's final text."""
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from decimalai.adk import DecimalaiPlugin, load_agent

    node = agent_name.replace("-", "_")
    if skills:
        support = load_agent(agent_name)
        agent = LlmAgent(name=node, model=model, instruction=support.instruction,
                         tools=[*support.tools])
        plugin = DecimalaiPlugin()
    else:
        agent = LlmAgent(name=node, model=model, instruction=_INSTRUCTION, tools=[])
        plugin = DecimalaiPlugin(agent_name=agent_name)
    runner = InMemoryRunner(app=App(name=_APP, root_agent=agent, plugins=[plugin]))

    async def _go() -> str:
        session = await runner.session_service.create_session(app_name=_APP, user_id="u1")
        out = ""
        async for event in runner.run_async(
            user_id="u1",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=_QUERY)]),
        ):
            content = getattr(event, "content", None)
            for part in (getattr(content, "parts", None) or []):
                if getattr(part, "text", None):
                    out += part.text
        return out

    return asyncio.run(_go())


@pytest.mark.live_llm
@pytest.mark.adk
@pytest.mark.parametrize("provider, model", h.matrix("adk"))
def test_adk_skill_body_reaches_the_models_answer(provider, model):
    """``load_skill`` serves a body a real Gemini answer then quotes back."""
    h.require_key_for(provider)
    pytest.importorskip("google.adk")
    _adk_env()

    import decimalai
    from decimalai.skill_router import SkillRouter

    agent_name = h.unique_agent("adk-skill-delivery")
    # A STABLE name, deliberately not uniquified per run. `sync_skills` with
    # local_wins then REPLACES the body, so exactly one settlement skill ever
    # exists and it always carries this run's token.
    #
    # The first cut minted a unique name per run and left every one behind. The
    # router answers a QUERY, not a name, so run N+1's list offered run N's
    # skill and the model dutifully quoted a stale code:
    #   assert '60599C8B99' in 'SETTLE-3123A63F2F'
    # which looks exactly like a delivery failure and is not one. A live test
    # that seeds registry rows has to be idempotent or it poisons its successor.
    skill_name = "adk-live-settlement-procedure"

    decimalai.init(api_key=h.API_KEY, base_url=h.BACKEND_URL, enabled=True)
    router = SkillRouter(
        api_key=h.API_KEY, base_url=h.BACKEND_URL, agent_name=agent_name,
    )
    router.sync_skills(
        [{
            "name": skill_name,
            "description": (
                "Internal settlement procedure for damaged shipments, including "
                "the settlement code to quote on a claim."
            ),
            "body_markdown": _SKILL_BODY,
            "category": "support",
            "trigger_phrases": ["damaged", "settlement", "claim"],
        }],
        author=f"live-adk/{agent_name}",
        conflict_policy="local_wins",
    )
    # load_agent() reads the agent's saved prompt, so the agent has to exist.
    h.create_agent(agent_name, _INSTRUCTION)

    # ── The negative control runs FIRST, so a leak cannot be explained away
    #    as contamination from the positive run's session.
    control_name = h.unique_agent("adk-skill-delivery-control")
    control_text = _run_agent(model, control_name, skills=False)
    assert _TOKEN not in control_text.upper(), (
        "the control answered with the token WITHOUT load_agent(), so the "
        "token reached the model by some other route and this test proves "
        f"nothing about delivery. Control answer: {control_text!r}"
    )

    answer = _run_agent(model, agent_name, skills=True)

    assert _TOKEN in answer.upper(), (
        "the agent was built from load_agent() and the model's answer does not "
        f"contain the token that exists only in the skill body. Answer: {answer!r}"
    )

    # Delivery must also be OBSERVABLE, not merely effective: a body that
    # reaches the model and leaves no trace of having done so is the one
    # failure mode indistinguishable from never delivering at all.
    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    assert skill_name in (detail.get("skills_loaded_by_agent") or []), (
        f"the model loaded {skill_name} and the trace does not say so: "
        f"skills_loaded_by_agent={detail.get('skills_loaded_by_agent')!r}"
    )

    # ⚠ Read the rendered INPUT specifically, never `str(detail)`. The trace
    # also carries the model's own answer, which by now contains the token —
    # so a whole-blob substring check passes whether or not the input was ever
    # recorded. The body arrives as load_skill's result, a tool-role message in
    # the next model call's input.
    rendered_inputs = " ".join(
        str(part.get("content", ""))
        for call in (detail.get("llm_calls") or [])
        for part in (call.get("rendered_input") or [])
    ).upper()
    assert _TOKEN in rendered_inputs, (
        "the model used the skill body but the trace's rendered_input does not "
        "contain it — delivery happened and is unwitnessable. Every downstream "
        "delivery signal (infer_prompt_rungs, the platform's activation "
        "detection, conformance C8/C14) reads this field."
    )
