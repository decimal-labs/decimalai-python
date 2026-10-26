"""Live-LLM: on a prompt-cache hit, a Claude trace's tokens are what Anthropic billed.

Anthropic bills a cached prompt in three parts: ``input_tokens`` (the uncached remainder),
``cache_creation_input_tokens`` (written to the cache) and ``cache_read_input_tokens`` (read
from it), each at its own price. LlmCallRecord keeps that split for a Claude call. Most
frameworks hand DecimalAI a single input count that already includes the cache (LangChain's
``usage_metadata``, Pydantic AI's and OpenInference's spans, ADK's ``prompt_token_count``). A
binding that records that count as the input bills every cached token at the full input price,
and nothing shows it while the cache is cold: with no cache, every convention gives the same
number. So each cell here runs twice with the same long prompt, marked for caching:

    cell                          the run
    anthropic_messages-anthropic  decimalai.anthropic.agent_config(): the Messages API with
                                  run.system, which carries its own cache breakpoint
    langchain-anthropic           ChatAnthropic, a system message with cache_control,
                                  decimalai.langchain.CallbackHandler
    pydantic_ai-anthropic         decimalai.pydantic_ai.agent_config() on Agent(...,
                                  model_settings={"anthropic_cache": True}): Anthropic's
                                  automatic caching (anthropic_cache_instructions marks only
                                  static instructions, and the capability's are a function)
    adk-anthropic                 ADK's AnthropicLlm, App(context_cache_config=...), DecimalAIPlugin

The second run must read the cache. Each run's trace must equal what Anthropic billed for it
(``h.assert_trace_tokens_match_billed``): llm_calls, input, cache read, cache write and
output (thinking included). The billed numbers come from the wire (``h.billed_anthropic_usage``),
not from the framework.

The prompt starts with a token minted per session, so the first run writes the cache instead of
reading an entry an earlier session left. The model is the budget tier's Claude (Haiku 5.5).

Marker: live_llm + skill_routing. Needs ``RUN_LIVE_LLM_TESTS=1``, ``ANTHROPIC_API_KEY`` and a
reachable backend.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from . import _live_helpers as h

_TOKEN = uuid.uuid4().hex[:10].upper()

#: About 2,500 tokens: past Claude Haiku 5.5's minimum cacheable prefix, with room to spare.
_LONG_PROMPT = (
    f"You are Acme's customer support agent (session {_TOKEN}).\n\n"
    + "\n".join(
        f"Rule {i}: For order question type {i}, check the order record first, quote only facts "
        f"the tools returned, and never promise a refund amount the policy does not state."
        for i in range(1, 61))
)

_QUESTIONS = ("Reply with the single word: ready.", "Reply with the single word: steady.")

_CELLS = [
    pytest.param("anthropic_messages", id="anthropic_messages-anthropic"),
    pytest.param("langchain", id="langchain-anthropic"),
    pytest.param("pydantic_ai", id="pydantic_ai-anthropic"),
    pytest.param("adk", id="adk-anthropic"),
]


@pytest.fixture(autouse=True)
def _unpatched():
    h.unpatch_provider_tracing()
    yield
    h.unpatch_provider_tracing()


# ── one run per binding: the long prompt, cached, and one question ──────────


def _anthropic_messages(model: str, agent_name: str, question: str) -> None:
    anthropic = pytest.importorskip("anthropic")
    from decimalai.anthropic import agent_config

    support = agent_config(agent_name)
    client = anthropic.Anthropic()
    with support.run(question) as run:
        assert "cache_control" in run.system[-1], "agent_config() marks the system for caching"
        client.messages.create(model=model, max_tokens=1024, system=run.system,
                               messages=[{"role": "user", "content": question}])


def _langchain(model: str, agent_name: str, question: str) -> None:
    pytest.importorskip("langchain_anthropic")
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    from decimalai.langchain import CallbackHandler

    system = SystemMessage(content=[{"type": "text", "text": _LONG_PROMPT,
                                     "cache_control": {"type": "ephemeral"}}])
    ChatAnthropic(model=model, max_tokens=1024).invoke(
        [system, HumanMessage(question)],
        config={"callbacks": [CallbackHandler(agent_name=agent_name)]})


def _pydantic_ai(model: str, agent_name: str, question: str) -> None:
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent

    from decimalai.pydantic_ai import agent_config

    support = agent_config(agent_name)
    Agent(f"anthropic:{model}", name=support.name, capabilities=[support],
          model_settings={"anthropic_cache": True}).run_sync(question)


def _adk(model: str, agent_name: str, question: str) -> None:
    pytest.importorskip("google.adk")
    from google.adk.agents import LlmAgent
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from decimalai.adk import DecimalAIPlugin

    # ADK writes the agent's name into the system prompt, so both runs share one (with a name
    # per run, the second run wrote the cache again and read nothing). The plugin files each
    # run's trace under its own name.
    agent = LlmAgent(name="support_agent", model=h.adk_model("anthropic", model),
                     instruction=_LONG_PROMPT)
    app_name = "decimal_billed_tokens"
    runner = InMemoryRunner(app=App(
        name=app_name, root_agent=agent, plugins=[DecimalAIPlugin(agent_name=agent_name)],
        context_cache_config=ContextCacheConfig(min_tokens=0)))

    async def _go() -> None:
        session = await runner.session_service.create_session(app_name=app_name, user_id="u1")
        async for _ in runner.run_async(
            user_id="u1", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=question)]),
        ):
            pass

    asyncio.run(_go())


_RUNNERS = {
    "anthropic_messages": _anthropic_messages,
    "langchain": _langchain,
    "pydantic_ai": _pydantic_ai,
    "adk": _adk,
}

#: Bindings whose prompt is the agent's saved prompt (agent_config()), not one passed in.
_SAVED_PROMPT = {"anthropic_messages", "pydantic_ai"}


# ── the test ─────────────────────────────────────────────────────────────────


@pytest.mark.live_llm
@pytest.mark.skill_routing
@pytest.mark.parametrize("binding", _CELLS)
def test_a_cache_hit_is_billed_and_traced_alike(binding):
    """Two runs on one long cached prompt: the second reads the cache, and each run's trace
    carries Anthropic's bill for it, cache split included."""
    h.require_key_for("anthropic")
    model = h.resolve_model("anthropic")
    run = _RUNNERS[binding]

    import decimalai

    decimalai.init(api_key=h.API_KEY, base_url=h.BACKEND_URL, enabled=True)
    shared = h.unique_agent(f"{binding}-billed")
    if binding in _SAVED_PROMPT:
        h.create_agent(shared, _LONG_PROMPT)

    reads = []
    seen_traces: set = set()
    for i, question in enumerate(_QUESTIONS):
        # agent_config() reads one agent's saved prompt for both runs; the others file each
        # run under its own name.
        agent_name = shared if binding in _SAVED_PROMPT else f"{shared}-{i + 1}"
        with h.billed_anthropic_usage() as seen:
            run(model, agent_name, question)
        decimalai.flush()
        h.flush_sdk_sender()
        expected = len(seen_traces) + 1 if binding in _SAVED_PROMPT else 1
        new = [t for t in h.poll_for_trace(agent_name, expected_count=expected)
               if t["id"] not in seen_traces]
        assert len(new) == 1, f"run {i + 1} should file one new trace under {agent_name}: {new}"
        seen_traces.add(new[0]["id"])
        detail = h.get_trace_detail(new[0]["id"])
        h.assert_trace_tokens_match_billed(detail.get("llm_calls") or [], seen,
                                           what=f"{binding} run {i + 1}")
        reads.append(seen.total("cache_read_input_tokens"))

    assert reads[1] > 0, (
        f"the second run read nothing from the cache (cache reads per run: {reads}), so this "
        f"cell checked no cache arithmetic")
