"""`DecimalAIPlugin` in ``App(plugins=[...])``: the only way ADK is traced since 0.20.

google-adk 2.12.0 drives every test here, with a ``BaseLlm`` subclass that
answers offline. ``App`` is ADK's own place for plugins (``Runner(plugins=...)`` is
deprecated, and ``adk web`` / ``adk run`` / ``api_server`` build
``Runner(app=...)``). Until 0.20 ``instrument()`` and ``init(adk=True)`` rewrote
``Runner.__init__`` to add the plugin to every runner in the process; that patch is
gone, and so is anything else this SDK did to ADK's classes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List
from unittest.mock import MagicMock

import pytest

pytest.importorskip("google.adk")

CALLER_PROMPT = "You are a terse support agent."


@pytest.fixture(autouse=True)
def client(monkeypatch):
    import decimalai
    import decimalai._config as cfg
    import decimalai.adk as adk
    from decimalai._config import DecimalConfig

    mock = MagicMock()
    mock.register_manifest.return_value = {"manifest_id": "plugin-manifest", "status": "active"}
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    ))
    monkeypatch.setattr(cfg, "_client", mock)
    monkeypatch.setattr(decimalai, "_global_client", mock, raising=False)
    monkeypatch.delenv("DECIMAL_AGENT_NAME", raising=False)
    for name, value in (("_PluginClass", None), ("_manifest_ids", {}), ("_manifest_trackers", {}),
                        ("_pending_manifests", {}), ("_skill_router_singleton", None), ("_live_runs", {})):
        monkeypatch.setattr(adk, name, value)
    return mock


def _flushed(mock: MagicMock) -> List[Any]:
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in mock.ingest_trace.call_args_list]


def _agent(name: str, response: Any = None, model: str = "offline-stub") -> Any:
    """An agent whose model answers offline, with `response` (an LlmResponse) when given."""
    from google.adk.agents import LlmAgent
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class OfflineLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            yield response or LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="offline answer")]))

    return LlmAgent(name=name, model=OfflineLlm(model=model), instruction=CALLER_PROMPT)


async def _one_turn(runner: Any, text: str = "What fee applies to an opened box?") -> None:
    from google.genai import types

    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="u")
    async for _ in runner.run_async(
        user_id="u", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        pass


def _runner(plugin: Any, name: str = "app_probe", response: Any = None,
            model: str = "offline-stub") -> Any:
    from google.adk.apps import App
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    return Runner(app=App(name=name, root_agent=_agent(name, response, model), plugins=[plugin]),
                  session_service=InMemorySessionService())


def test_a_plugin_in_the_app_traces_each_invocation_once(client):
    import decimalai.adk as adk

    runner = _runner(adk.DecimalAIPlugin(agent_name="explicit_label"))
    asyncio.run(_one_turn(runner))
    plugins = [p for p in runner.plugin_manager.plugins if isinstance(p, adk._plugin_class())]
    assert len(plugins) == 1
    assert [t.agent_name for t in _flushed(client)] == ["explicit_label"]


def test_the_trace_is_filed_under_init_s_agent_else_adk_s(client, monkeypatch):
    import decimalai._config as cfg
    import decimalai.adk as adk
    from decimalai._config import DecimalConfig

    asyncio.run(_one_turn(_runner(adk.DecimalAIPlugin(), name="adk_node")))
    monkeypatch.setattr(cfg, "_config", DecimalConfig(  # init(agent_name="from-init")
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True, agent_name="from-init",
    ))
    asyncio.run(_one_turn(_runner(adk.DecimalAIPlugin(), name="adk_node")))
    assert [t.agent_name for t in _flushed(client)] == ["adk_node", "from-init"]


def test_nothing_in_adk_is_patched(client):
    """Building a plugin, a config and a runner leaves ADK's classes as they were."""
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner, Runner

    import decimalai.adk as adk

    before = (Runner.__init__, Runner.run, Runner.run_async, InMemoryRunner.__init__,
              LlmAgent.canonical_instruction)
    asyncio.run(_one_turn(_runner(adk.DecimalAIPlugin())))
    adk._load_skill_tool_class()
    assert (Runner.__init__, Runner.run, Runner.run_async, InMemoryRunner.__init__,
            LlmAgent.canonical_instruction) == before


def test_a_runner_without_the_plugin_is_not_traced(client):
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner

    import decimalai.adk  # noqa: F401  (importing the adapter installs nothing)

    runner = InMemoryRunner(app=App(name="untraced", root_agent=_agent("untraced")))
    asyncio.run(_one_turn(runner))
    assert _flushed(client) == []


def test_a_sync_run_links_its_parent_through_adks_own_context_copy(client):
    """ADK 2.12's Runner.run starts its worker thread under
    ``contextvars.copy_context().run``, so the parent trace reaches before_run."""
    from google.adk.apps import App
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    import decimalai.adk as adk
    from decimalai.generic import start_trace

    runner = InMemoryRunner(app=App(name="sync_probe", root_agent=_agent("sync_probe"),
                                    plugins=[adk.DecimalAIPlugin(agent_name="sync_probe")]))
    runner.session_service.create_session_sync(app_name="sync_probe", user_id="u", session_id="s")
    with start_trace(agent_name="outer", auto_send=False) as outer:
        parent_id = outer.get_trace_id()
        assert list(runner.run(user_id="u", session_id="s", new_message=types.Content(
            role="user", parts=[types.Part(text="hello")],
        )))
    runner.session_service.create_session_sync(app_name="sync_probe", user_id="u", session_id="root")
    assert list(runner.run(user_id="u", session_id="root", new_message=types.Content(
        role="user", parts=[types.Part(text="standalone")],
    )))
    assert {t.user_input_preview: t.parent_trace_id for t in _flushed(client)} == {
        "hello": parent_id, "standalone": None,
    }


def _traced_call(client: MagicMock, response: Any, model: str = "offline-stub") -> Any:
    """The LlmCallRecord the plugin records for one model call that answers `response`."""
    import decimalai.adk as adk

    asyncio.run(_one_turn(_runner(adk.DecimalAIPlugin(agent_name="tokens"), response=response,
                                  model=model)))
    (trace,) = _flushed(client)
    (call,) = trace.llm_calls
    return call


def _traced_output_tokens(client: MagicMock, response: Any) -> Any:
    """The output tokens the plugin records for one model call that answers `response`."""
    return _traced_call(client, response).output_tokens


def _answer(usage: Any) -> Any:
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text="answer")]),
                       usage_metadata=usage)


def test_a_model_calls_output_tokens_include_its_thinking(client):
    """Gemini's usage keeps the thinking apart from the answer and bills both as output;
    the call's output is their sum, not the answer alone (0.25.1 recorded 40 here)."""
    from google.genai import types

    assert _traced_output_tokens(client, _answer(types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10, candidates_token_count=40, thoughts_token_count=80,
        total_token_count=130,
    ))) == 120


def test_a_model_call_with_no_thinking_records_its_answer(client):
    from google.genai import types

    assert _traced_output_tokens(client, _answer(types.GenerateContentResponseUsageMetadata(
        prompt_token_count=10, candidates_token_count=7, total_token_count=17,
    ))) == 7


def test_a_claude_calls_output_tokens_are_what_anthropic_billed(client):
    """ADK's AnthropicLlm subtracts the thinking from Anthropic's output_tokens (the billed
    output) to fill candidates_token_count; the trace puts it back."""
    pytest.importorskip("anthropic")
    from anthropic.types import (
        Message,
        OutputTokensDetails,
        TextBlock,
        ThinkingBlock,
        Usage,
    )
    from google.adk.models.anthropic_llm import message_to_generate_content_response

    message = Message(
        id="msg_1", type="message", role="assistant", model="claude-sonnet-5-5",
        content=[ThinkingBlock(type="thinking", thinking="weighing it", signature="sig"),
                 TextBlock(type="text", text="answer")],
        stop_reason="end_turn", stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=120,
                    output_tokens_details=OutputTokensDetails(thinking_tokens=80)),
    )
    response = message_to_generate_content_response(message)
    assert response.usage_metadata.candidates_token_count == 40  # what 0.25.1 recorded
    assert _traced_output_tokens(client, response) == message.usage.output_tokens == 120


def test_a_claude_calls_input_keeps_anthropics_cache_split(client):
    """ADK's AnthropicLlm folds the cache into prompt_token_count; Anthropic bills its
    input_tokens (the uncached remainder), the cache read and the cache write apart, and a
    Claude call's record keeps that split (live, 2026-10-26: 2892 recorded where Anthropic
    billed 4, with the 2888-token cache write missing)."""
    pytest.importorskip("anthropic")
    from anthropic.types import Message, TextBlock, Usage
    from google.adk.models.anthropic_llm import message_to_generate_content_response

    response = message_to_generate_content_response(Message(
        id="msg_1", type="message", role="assistant", model="claude-haiku-5-5",
        content=[TextBlock(type="text", text="steady")], stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=4, output_tokens=4, cache_read_input_tokens=2865,
                    cache_creation_input_tokens=13),
    ))
    assert response.usage_metadata.prompt_token_count == 2882
    call = _traced_call(client, response, model="claude-haiku-5-5")
    assert (call.input_tokens, call.cache_read_tokens, call.cache_creation_tokens,
            call.output_tokens) == (4, 2865, 13, 4)


def test_a_gemini_calls_prompt_stays_whole(client):
    """Gemini's cached tokens are part of its prompt count, which the record keeps whole."""
    from google.genai import types

    call = _traced_call(client, _answer(types.GenerateContentResponseUsageMetadata(
        prompt_token_count=1000, cached_content_token_count=800, candidates_token_count=7,
        total_token_count=1007,
    )), model="gemini-3.5-flash")
    assert (call.input_tokens, call.cache_read_tokens, call.cache_creation_tokens) == (1000, None, None)


def test_a_litellm_calls_reasoning_is_not_counted_twice(client):
    """ADK's LiteLlm keeps LiteLLM's completion_tokens, reasoning included, as
    candidates_token_count and repeats the reasoning as thoughts_token_count."""
    pytest.importorskip("litellm")
    from google.adk.models.lite_llm import _model_response_to_generate_content_response
    from litellm import ModelResponse

    response = _model_response_to_generate_content_response(ModelResponse(
        model="gpt-5",
        choices=[{"index": 0, "message": {"role": "assistant", "content": "answer"},
                  "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60,
               "completion_tokens_details": {"reasoning_tokens": 30}},
    ))
    assert response.usage_metadata.thoughts_token_count == 30
    assert _traced_output_tokens(client, response) == 50


def test_the_process_wide_install_is_gone(client, monkeypatch, caplog):
    """`instrument()` and `init(adk=True)` patched Runner.__init__; 0.20 removed both.
    `DECIMAL_AUTO_TRACE=adk` cannot raise, so it says what to do instead."""
    import decimalai
    import decimalai.adk as adk

    assert not hasattr(adk, "instrument")
    with pytest.raises(TypeError, match="adk"):
        decimalai.init(adk=True)  # type: ignore[call-arg]
    monkeypatch.setenv("DECIMAL_AUTO_TRACE", "adk")
    monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_test")
    init = MagicMock()
    monkeypatch.setattr(decimalai, "init", init)
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        decimalai._auto_init_from_env()
    assert any("DECIMAL_AUTO_TRACE=adk traces nothing" in r.getMessage() and "DecimalAIPlugin()" in r.getMessage()
               for r in caplog.records)
    assert "adk" not in init.call_args.kwargs
