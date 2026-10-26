"""Live-LLM: every ``agent_config()`` binding loads a skill that a real model then answers from.

One test, one cell per shipped binding and funded provider, against a real backend.
Each cell builds the agent the way ``decimalai init`` and the docs build it:

    cell                          the agent
    openai_agents-openai          Agent(instructions=support.instructions, tools=[*support.tools])
    pydantic_ai-openai            Agent(model, capabilities=[support])
    pydantic_ai-anthropic         the same, on Claude
    anthropic_messages-anthropic  with support.run(q) as run: the Tool Runner with run.system, run.tools
    neutral-openai                decimalai.agent_config(): run.system, run.tool_spec and
                                  run.load_skill() in a hand-written chat-completions loop
    openai_chat-openai            decimalai.openai.agent_config(): with support.run(q) as run:
                                  a Chat Completions loop with run.system, run.message,
                                  run.tools and run.tool_result(call), as `decimalai init
                                  --framework openai` writes it
    openai_responses-openai       the same run on the Responses API: instructions=run.system,
                                  run.response_tools, previous_response_id
    google_genai-google           decimalai.google_genai.agent_config(): a chat with automatic
                                  function calling, system_instruction=run.system and
                                  tools=[*run.tools]; needs a Gemini key (GEMINI_API_KEY)
    adk-openai                    LlmAgent(model=LiteLlm("openai/..."), instruction=support.instruction,
                                  tools=[*support.tools]); runs only in the openai2 environment,
                                  because litellm pins openai<3 (skips elsewhere)
    adk-anthropic                 the same on Claude, through ADK's own AnthropicLlm (needs only
                                  `anthropic`, so it runs in the openai3 environment)
    langchain-openai              create_agent("openai:...", middleware=[support.middleware])
    langchain-anthropic           the same, on Claude
    langgraph_stategraph-openai   a hand-built LangGraph StateGraph: the model node takes
                                  support.system_message(...), the graph binds support.tools
                                  and runs them in a ToolNode (the LangChain page's "A
                                  hand-built graph"), the most common LangGraph shape on GitHub

ADK on Gemini has its own file, ``test_framework_live_adk_skill_delivery.py``, which this one
generalizes.

The release gate runs these cells in its provider lanes: each id carries the provider, and the
lanes select by ``-k`` (``platform/release_gate/config.py``). ``adk-openai`` runs in the gate's
openai2 leg, ``adk-anthropic`` in its anthropic lane.

WHAT MAKES IT NON-CIRCULAR
--------------------------
The fact is minted per session: a random hex token that has never existed before, in the skill
body and asked for by a question that gives no other route to it. The agent's saved prompt does
not mention it and no other tool returns it, so a correct answer is only reachable through the
body ``load_skill`` served.

Each cell first runs a control: the same framework, model, instruction and question, with no
``agent_config()``. If the control also answers with the token, the token reached the model some
other way and the cell proves nothing, so the control failing to produce it is part of the proof.

Delivery must also be observable: the trace filed under the agent says the skill was loaded,
and the body appears in a model call's recorded input (the ``load_skill`` result).

Marker: live_llm + skill_routing. Needs ``RUN_LIVE_LLM_TESTS=1``, the provider's key and a
reachable backend.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest

from . import _live_helpers as h

#: Long enough that a model cannot stumble onto it, and shaped so a refusal ("I don't
#: have that code") can never be mistaken for a pass.
_TOKEN = uuid.uuid4().hex[:10].upper()

#: A STABLE name, not one per run: ``sync_skills`` with ``local_wins`` replaces the body,
#: so exactly one such skill exists and it always carries this session's token. A name
#: per run leaves the earlier ones behind with their old tokens (see the ADK file).
#: The topic differs from the ADK file's settlement procedure, so neither skill can be
#: offered for the other's question.
_SKILL_NAME = "binding-live-rma-procedure"

_SKILL_BODY = f"""# Cracked-screen return authorization

This is the internal return procedure for phones that arrive with a cracked screen.
It is not published anywhere outside this document.

## The return authorization code

For a phone that arrived with a cracked screen, the return authorization code to give
the customer is `RMA-{_TOKEN}`. Quote it exactly, including the prefix.

## What the customer gets

A full refund once the phone is back, and a prepaid shipping label. Do not offer a repair.
"""

#: Says nothing about return codes: the only path to the token is the skill body.
_INSTRUCTION = (
    "You are a customer support agent. Answer using the procedures available to you. "
    "If you do not know something, say so."
)

_QUERY = (
    "A customer's new phone arrived with a cracked screen. What return authorization code "
    "should I give them? Reply with the code and nothing else."
)

_CELLS = [
    pytest.param("openai_agents", "openai", id="openai_agents-openai"),
    pytest.param("pydantic_ai", "openai", id="pydantic_ai-openai"),
    pytest.param("neutral", "openai", id="neutral-openai"),
    pytest.param("openai_chat", "openai", id="openai_chat-openai"),
    pytest.param("openai_responses", "openai", id="openai_responses-openai"),
    pytest.param("google_genai", "google", id="google_genai-google"),
    pytest.param("adk", "openai", id="adk-openai"),
    pytest.param("adk", "anthropic", id="adk-anthropic"),
    pytest.param("langchain", "openai", id="langchain-openai"),
    pytest.param("langchain", "anthropic", id="langchain-anthropic"),
    pytest.param("langgraph_stategraph", "openai", id="langgraph_stategraph-openai"),
    pytest.param("pydantic_ai", "anthropic", id="pydantic_ai-anthropic"),
    pytest.param("anthropic_messages", "anthropic", id="anthropic_messages-anthropic"),
]


# ── leave the process as found ───────────────────────────────────────────────


def _openinference_instrumentors() -> list:
    found = []
    for module, cls in (("openinference.instrumentation.openai", "OpenAIInstrumentor"),
                        ("openinference.instrumentation.anthropic", "AnthropicInstrumentor"),
                        ("openinference.instrumentation.google_genai", "GoogleGenAIInstrumentor")):
        try:
            found.append(getattr(__import__(module, fromlist=[cls]), cls))
        except ImportError:
            pass
    return found


def _unpatch() -> None:
    """Turn off the process-wide switches these cells turn on.

    ``init(openai=True)`` and ``decimalai.anthropic.agent_config()`` turn OpenInference's
    instrumentors on, and Pydantic AI's ``Agent.instrument_all()`` is global too (no cell
    calls it now; one left on by another file would trace the controls). Each
    cell starts and ends with them off, so no cell traces another's calls, and a later
    file meets the SDK unpatched (the 0.23.0 gate failed on one left on, f426799).
    """
    for cls in _openinference_instrumentors():
        if cls().is_instrumented_by_opentelemetry:
            cls().uninstrument()
    for name in ("decimalai.anthropic", "decimalai.openai", "decimalai.google_genai",
                 "decimalai.pydantic_ai"):
        try:
            # So the next agent_config() turns its provider's tracing on again.
            __import__(name, fromlist=["_tracing_turned_on"])._tracing_turned_on = False
        except ImportError:
            pass
    try:
        from pydantic_ai import Agent
        Agent.instrument_all(False)
    except ImportError:
        pass


@pytest.fixture(autouse=True)
def _unpatched():
    _unpatch()
    yield
    _unpatch()


# ── one runner per binding ───────────────────────────────────────────────────


def _text_of(message) -> str:
    return "".join(getattr(b, "text", "") for b in message.content if getattr(b, "type", "") == "text")


def _openai_agents(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    pytest.importorskip("agents")
    from agents import Agent, RunConfig, Runner

    model_spec = h.openai_agents_model(provider, model)
    if not skills:
        control = Agent(name="control", model=model_spec, instructions=_INSTRUCTION)
        result = Runner.run_sync(control, _QUERY, max_turns=6,
                                 run_config=RunConfig(tracing_disabled=True))
        return str(result.final_output or "")

    from decimalai.openai_agents import agent_config

    support = agent_config(agent_name)
    agent = Agent(name=support.name, model=model_spec, instructions=support.instructions,
                  tools=[*support.tools])
    return str(Runner.run_sync(agent, _QUERY, max_turns=6).final_output or "")


def _pydantic_ai(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent

    model_id = f"{provider}:{model}"
    if not skills:
        # Instrumentation is off here (the fixture), so the control is not traced.
        return str(Agent(model_id, instructions=_INSTRUCTION).run_sync(_QUERY).output)

    from decimalai.pydantic_ai import agent_config

    # As `decimalai init --framework pydantic-ai` writes it: agent_config() alone, which
    # traces its own runs (no init(otel=True), no Agent.instrument_all()).
    support = agent_config(agent_name)
    agent = Agent(model_id, name=support.name, capabilities=[support])
    return str(agent.run_sync(_QUERY).output)


def _anthropic_messages(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": _QUERY}]
    if not skills:
        # A call outside a run: nothing traces it (the instrumentors are off).
        return _text_of(client.messages.create(model=model, max_tokens=512,
                                               system=_INSTRUCTION, messages=messages))

    from decimalai.anthropic import agent_config

    support = agent_config(agent_name)
    with support.run(_QUERY) as run:
        answer = client.beta.messages.tool_runner(
            model=model, max_tokens=1024, system=run.system, tools=[*run.tools],
            messages=messages,
        ).until_done()
    return _text_of(answer)


def _neutral(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    openai = pytest.importorskip("openai")
    client = openai.OpenAI()
    if not skills:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _INSTRUCTION},
                      {"role": "user", "content": _QUERY}],
        ).choices[0].message.content or ""

    import decimalai
    from decimalai.providers import agent_run

    # The docs' shape for your own model calls: a provider flag, and agent_run() around the
    # run, so the run's record goes on one trace filed under the agent.
    decimalai.init(api_key=h.API_KEY, base_url=h.BACKEND_URL, openai=True)
    support = decimalai.agent_config(agent_name)
    with agent_run(support.name):
        run = support.run(_QUERY)
        spec = run.tool_spec
        tools = [{"type": "function", "function": {
            "name": spec["name"], "description": spec["description"],
            "parameters": spec["parameters"],
        }}]
        messages: list = [{"role": "system", "content": run.system},
                          {"role": "user", "content": _QUERY}]
        for _ in range(4):
            reply = client.chat.completions.create(model=model, messages=messages,
                                                   tools=tools).choices[0].message
            if not reply.tool_calls:
                return reply.content or ""
            messages.append({"role": "assistant", "content": reply.content, "tool_calls": [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in reply.tool_calls]})
            for call in reply.tool_calls:
                wanted = json.loads(call.function.arguments or "{}").get("name", "")
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": run.load_skill(wanted)})
    return ""


def _openai_chat(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    openai = pytest.importorskip("openai")
    client = openai.OpenAI()
    if not skills:
        # A call outside a run: nothing traces it (the instrumentors are off).
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": _INSTRUCTION},
                      {"role": "user", "content": _QUERY}],
        ).choices[0].message.content or ""

    from decimalai.openai import agent_config

    # As `decimalai init --framework openai` writes it: no init() line, since agent_config()
    # traces its own runs, and the loop inside the run's `with` block.
    support = agent_config(agent_name)
    with support.run(_QUERY) as run:
        messages: list = [{"role": "system", "content": run.system}, run.message]
        for _ in range(4):
            reply = client.chat.completions.create(model=model, messages=messages,
                                                   tools=[*run.tools]).choices[0].message
            messages.append(reply)
            if not reply.tool_calls:
                return reply.content or ""
            for call in reply.tool_calls:
                messages.append(run.tool_result(call) or {
                    "role": "tool", "tool_call_id": call.id, "content": "No such tool."})
    return ""


def _openai_responses(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    openai = pytest.importorskip("openai")
    client = openai.OpenAI()
    if not skills:
        return client.responses.create(model=model, instructions=_INSTRUCTION,
                                       input=_QUERY).output_text or ""

    from decimalai.openai import agent_config

    support = agent_config(agent_name)
    with support.run(_QUERY) as run:
        response = client.responses.create(model=model, instructions=run.system,
                                           input=[run.message], tools=[*run.response_tools])
        for _ in range(4):
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                break
            # A previous_response_id does not carry instructions over: send them again.
            response = client.responses.create(
                model=model, instructions=run.system, previous_response_id=response.id,
                tools=[*run.response_tools],
                input=[run.tool_result(call) or {"type": "function_call_output",
                                                 "call_id": call.call_id, "output": "No such tool."}
                       for call in calls])
    return response.output_text or ""


def _google_genai(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    pytest.importorskip("google.genai")
    import os

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if not skills:
        # A call outside a run: nothing traces it (the instrumentors are off).
        return client.models.generate_content(
            model=model, contents=_QUERY,
            config=types.GenerateContentConfig(system_instruction=_INSTRUCTION)).text or ""

    from decimalai.google_genai import agent_config

    # The module's documented shape: a chat with automatic function calling, no init() line,
    # since agent_config() traces its own runs.
    support = agent_config(agent_name)
    with support.run(_QUERY) as run:
        chat = client.chats.create(model=model, config=types.GenerateContentConfig(
            system_instruction=run.system, tools=[*run.tools]))
        return chat.send_message(run.message).text or ""


def _adk(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    pytest.importorskip("google.adk")
    if provider == "anthropic":
        pytest.importorskip("anthropic")
    else:
        pytest.importorskip(
            "litellm",
            reason="ADK on an OpenAI model needs litellm, which pins openai<3: this cell runs in "
                   "the openai2 environment",
        )
    from google.adk.agents import LlmAgent
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from decimalai.adk import DecimalAIPlugin, agent_config

    if provider == "anthropic":
        llm = h.adk_model(provider, model)
    else:
        from google.adk.models.lite_llm import LiteLlm

        llm = LiteLlm(model=f"{provider}/{model}")
    if skills:
        support = agent_config(agent_name)
        agent = LlmAgent(name=agent_name.replace("-", "_"), model=llm,
                         instruction=support.instruction, tools=[*support.tools])
        plugins = [DecimalAIPlugin()]
    else:
        agent = LlmAgent(name="control", model=llm, instruction=_INSTRUCTION, tools=[])
        plugins = []
    app_name = "decimal_binding_skill_loading"
    runner = InMemoryRunner(app=App(name=app_name, root_agent=agent, plugins=plugins))

    async def _go() -> str:
        session = await runner.session_service.create_session(app_name=app_name, user_id="u1")
        out = ""
        async for event in runner.run_async(
            user_id="u1", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=_QUERY)]),
        ):
            for part in (getattr(getattr(event, "content", None), "parts", None) or []):
                if getattr(part, "text", None):
                    out += part.text
        return out

    return asyncio.run(_go())


def _langchain(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    pytest.importorskip("langchain.agents")
    pytest.importorskip(f"langchain_{provider}")
    from langchain.agents import create_agent

    def final_text(result) -> str:
        return str(result["messages"][-1].text or "")   # a property in langchain-core 1.x

    model_id = f"{provider}:{model}"
    ask = {"messages": [{"role": "user", "content": _QUERY}]}
    if not skills:
        return final_text(create_agent(model_id, tools=[], system_prompt=_INSTRUCTION).invoke(ask))

    from decimalai.langchain import agent_config

    # As `decimalai init --framework langchain` writes it: the middleware, and no init() line,
    # since agent_config() turns LangChain tracing on.
    support = agent_config(agent_name)
    agent = create_agent(model_id, tools=[], middleware=[support.middleware])
    return final_text(agent.invoke(ask))


def _langgraph_stategraph(provider: str, model: str, agent_name: str, *, skills: bool) -> str:
    pytest.importorskip("langgraph.graph")
    pytest.importorskip(f"langchain_{provider}")
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import SystemMessage
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    def final_text(result) -> str:
        return str(result["messages"][-1].text or "")   # a property in langchain-core 1.x

    ask = {"messages": [{"role": "user", "content": _QUERY}]}
    llm = init_chat_model(f"{provider}:{model}")
    if not skills:
        def plain(state: MessagesState):
            return {"messages": [llm.invoke([SystemMessage(_INSTRUCTION), *state["messages"]])]}

        graph = StateGraph(MessagesState)
        graph.add_node("model", plain)
        graph.add_edge(START, "model")
        return final_text(graph.compile().invoke(ask))

    from decimalai.langchain import agent_config

    # As the LangChain page's "A hand-built graph" writes it: the model node takes the run's
    # system message, and the graph binds the config's tools (load_skill) and runs them.
    support = agent_config(agent_name)
    tools = [*support.tools]
    bound = llm.bind_tools(tools)

    def call_model(state: MessagesState, runtime):
        system = support.system_message(state["messages"], context=runtime.context)
        return {"messages": [bound.invoke([system, *state["messages"]])]}

    builder = StateGraph(MessagesState, context_schema=dict)
    builder.add_node("model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")
    return final_text(builder.compile().invoke(ask, context={}))


_RUNNERS = {
    "langchain": _langchain,
    "langgraph_stategraph": _langgraph_stategraph,
    "openai_agents": _openai_agents,
    "pydantic_ai": _pydantic_ai,
    "anthropic_messages": _anthropic_messages,
    "neutral": _neutral,
    "openai_chat": _openai_chat,
    "openai_responses": _openai_responses,
    "google_genai": _google_genai,
    "adk": _adk,
}


# ── the test ─────────────────────────────────────────────────────────────────


def _seed_agent(agent_name: str) -> None:
    from decimalai.skill_router import SkillRouter

    # The agent first: agent_config() reads its saved prompt, and a sync naming an agent
    # that does not exist yet makes it, except a `live-` name, which the platform keeps
    # for test fixtures. That sync adds the skill to no agent (added_to_agent None), and
    # the cell then passed only because the skill was offered to every agent.
    h.create_agent(agent_name, _INSTRUCTION)
    router = SkillRouter(api_key=h.API_KEY, base_url=h.BACKEND_URL, agent_name=agent_name)
    router.sync_skills(
        [{
            "name": _SKILL_NAME,
            "description": (
                "Internal return procedure for phones that arrive with a cracked screen, "
                "including the return authorization code to give the customer."
            ),
            "body_markdown": _SKILL_BODY,
            "category": "support",
            "trigger_phrases": ["cracked screen", "return authorization", "RMA code"],
        }],
        author=f"live-binding/{agent_name}",
        conflict_policy="local_wins",
        agent_name=agent_name,
        # Only the agents it is added to. Left out, the platform offered it to every agent in
        # the workspace (seen 2026-10-22: offer_scope "workspace" on the local backend).
        offer_scope="restricted",
    )


@pytest.mark.live_llm
@pytest.mark.skill_routing
@pytest.mark.parametrize("binding, provider", _CELLS)
def test_binding_skill_body_reaches_the_models_answer(binding, provider):
    """``load_skill`` serves a body that a real model's answer then quotes back."""
    h.require_key_for(provider)
    model = h.resolve_model(provider)
    run_agent = _RUNNERS[binding]

    # The control runs FIRST, so a leak cannot be explained away as contamination from
    # the agent's own run.
    control_text = run_agent(provider, model, h.unique_agent(f"{binding}-control"), skills=False)
    assert _TOKEN not in control_text.upper(), (
        "the control answered with the token WITHOUT agent_config(), so the token reached "
        f"the model by some other route and this cell proves nothing. Control: {control_text!r}"
    )

    agent_name = h.unique_agent(f"{binding}-skill-loading")
    _seed_agent(agent_name)
    answer = run_agent(provider, model, agent_name, skills=True)
    assert _TOKEN in answer.upper(), (
        f"the {binding} agent was built from agent_config() and the model's answer does not "
        f"contain the token that exists only in the skill body. Answer: {answer!r}"
    )

    import decimalai

    decimalai.flush()
    h.flush_sdk_sender()
    detail = h.get_trace_detail(h.poll_for_trace(agent_name)[0]["id"])
    assert _SKILL_NAME in (detail.get("skills_loaded_by_agent") or []), (
        f"the model loaded {_SKILL_NAME} and the trace does not say so: "
        f"skills_loaded_by_agent={detail.get('skills_loaded_by_agent')!r}"
    )
    # Read the recorded INPUT, never the whole trace: the trace also carries the answer,
    # which by now contains the token, so a whole-blob check passes whether or not the
    # body was recorded. The body arrives as load_skill's result, in the next call's input.
    rendered_inputs = " ".join(
        str(part.get("content", ""))
        for call in (detail.get("llm_calls") or [])
        for part in (call.get("rendered_input") or [])
    ).upper()
    assert _TOKEN in rendered_inputs, (
        f"the model used the skill body but no recorded model input of the {binding} trace "
        "contains it, so delivery happened and cannot be seen. Every downstream delivery "
        "signal reads this field."
    )
