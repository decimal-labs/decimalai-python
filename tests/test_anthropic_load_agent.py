"""`decimalai.anthropic.agent_config()`: an agent's prompt and skills for the Anthropic Messages API.

Driven through the real Anthropic SDK — the beta Tool Runner, a hand-written
``messages.create`` loop, sync and async — against an in-process stub of the
Messages API (an ``httpx2`` mock transport) that records each request and asks
for ``load_skill`` when told to. DecimalAI's answers come from the fake platform
the framework-neutral tests use, so nothing leaves the process.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

pytest.importorskip("anthropic")
httpx2 = pytest.importorskip("httpx2")

import anthropic  # noqa: E402
from anthropic import beta_async_tool, beta_tool  # noqa: E402

import decimalai._agent as agent_mod  # noqa: E402
import decimalai._agent_binding as binding  # noqa: E402
import decimalai._provider_run as provider_run  # noqa: E402
import decimalai.anthropic as da  # noqa: E402
from decimalai._agent import SavedPrompt  # noqa: E402
from decimalai._agent_binding import MissingPromptVariable  # noqa: E402
from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT, SkillRouter  # noqa: E402
from tests.test_agent_binding import FakePlatform  # noqa: E402

BASE = "http://anthropic-load-agent.test"
SAM = {"customer": {"name": "Sam"}}


@pytest.fixture
def platform(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=True))
    monkeypatch.setattr(cfg, "_client", MagicMock())
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    monkeypatch.setattr(da, "_tracing_turned_on", True)  # tracing is tested on its own below
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url=BASE, inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(da, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt="Hello {{customer.name}}.",
                             version_number=4, content_hash="c" * 64)
    monkeypatch.setattr(agent_mod, "read_prompt", lambda name, **kwargs: fake.saved)
    return fake


class Messages:
    """A stub of ``POST /v1/messages``: ("call", skill) asks for load_skill, "answer" ends the turn."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests: list = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        step = self.steps.pop(0)
        if step == "answer":
            content = [{"type": "text", "text": "done"}]
            stop = "end_turn"
        else:
            content = [{"type": "tool_use", "id": f"toolu_{len(self.requests)}", "name": "load_skill",
                        "input": {"name": step[1]}}]
            stop = "tool_use"
        return httpx2.Response(200, json={
            "id": f"msg_{len(self.requests)}", "type": "message", "role": "assistant", "model": "m",
            "content": content, "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}})

    def client(self):
        return anthropic.Anthropic(api_key="sk-ant-test",
                                   http_client=httpx2.Client(transport=httpx2.MockTransport(self)))

    def async_client(self):
        async def handler(request):
            return self(request)

        return anthropic.AsyncAnthropic(api_key="sk-ant-test",
                                        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)))

    def tool_results(self, index):
        return [c for m in self.requests[index]["messages"] if isinstance(m.get("content"), list)
                for c in m["content"] if c.get("type") == "tool_result"]


@beta_tool
def lookup_order(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: The order id, for example ORD-1001.
    """
    return "{}"


@beta_async_tool
async def lookup_order_async(order_id: str) -> str:
    """Look up an order.

    Args:
        order_id: The order id, for example ORD-1001.
    """
    return "{}"


def tool_runner(client, run, question, *, tools=(lookup_order,)):
    return client.beta.messages.tool_runner(
        model="claude-haiku-5-5", max_tokens=64, system=run.system,
        tools=[*tools, *run.tools], messages=[{"role": "user", "content": question}],
    ).until_done()


def test_a_run_gives_the_tool_runner_the_prompt_the_skill_list_and_load_skill(platform):
    support = da.agent_config("support")
    assert type(support).__name__ == "AnthropicAgentConfig" and support.name == "support"
    api = Messages(("call", "damaged-item"), "answer")
    with support.run("My keyboard arrived cracked.", variables=SAM) as run:
        assert tool_runner(api.client(), run, "My keyboard arrived cracked.").stop_reason == "end_turn"
    first, second = api.requests
    prompt, menu = first["system"]
    assert prompt == {"type": "text", "text": "Hello Sam."}, "the prompt block is the prompt, as filled"
    assert menu["text"].startswith("\n\n## Available Skills") and menu["text"].endswith(LOAD_SKILL_PROMPT_HINT)
    assert menu["cache_control"] == {"type": "ephemeral"}, "one breakpoint, at the end of the system text"
    if "thread" in second:
        # anthropic >= 2: the Tool Runner's message thread keeps the system text, and a
        # later request names the reply it follows (the exporter restores what it left out).
        assert first["thread"] == {"type": "create"} and "system" not in second
        assert second["thread"] == {"type": "continue", "previous_message_id": "msg_1"}
    else:
        assert first["system"] == second["system"], "the run's second request reads the first one's cache"
    assert [tool["name"] for tool in first["tools"]] == ["lookup_order", "load_skill"]
    assert first["tools"][1]["input_schema"]["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    (result,) = api.tool_results(1)
    assert result["content"].startswith("## Skill: damaged-item")
    assert list(platform.queries.values()) == ["My keyboard arrived cracked."]


def test_a_hand_written_loop_gets_plain_tool_definitions_and_call(platform):
    support = da.agent_config("support")
    api = Messages(("call", "damaged-item"), "answer")
    client = api.client()
    with support.run("My keyboard arrived cracked.", variables=SAM) as run:
        (definition,) = run.tool_params
        assert isinstance(definition, dict) and set(definition) == {"name", "description", "input_schema"}
        messages = [{"role": "user", "content": "My keyboard arrived cracked."}]
        while True:
            response = client.messages.create(model="m", max_tokens=64, system=run.system,
                                              tools=[lookup_order.to_dict(), *run.tool_params], messages=messages)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            results = [run.tool_result(block) for block in response.content if block.type == "tool_use"]
            messages.append({"role": "user", "content": results})
    (result,) = api.tool_results(1)
    assert result["tool_use_id"] == "toolu_1" and result["content"].startswith("## Skill: damaged-item")
    assert run.tool_result({"type": "tool_use", "id": "x", "name": "lookup_order", "input": {}}) is None, \
        "another tool's block is the caller's to answer"


def test_the_tool_runners_tool_and_tool_params_are_the_same_definition(platform):
    support = da.agent_config("support")
    with support.run("Hi", variables=SAM) as run:
        (tool,) = run.tools
        assert tool.to_dict() == run.tool_params[0]


def test_each_run_is_routed_on_its_own_message(platform):
    support = da.agent_config("support")
    for question in ("My keyboard arrived cracked.", "Where is my order?"):
        with support.run(question, variables=SAM) as run:
            tool_runner(Messages("answer").client(), run, question)
    assert list(platform.queries.values()) == ["My keyboard arrived cracked.", "Where is my order?"]


def test_a_model_asking_with_other_capitals_gets_the_listed_skill(platform):
    support = da.agent_config("support")
    api = Messages(("call", "Damaged-Item"), "answer")
    with support.run("cracked", variables=SAM) as run:
        tool_runner(api.client(), run, "cracked")
    assert api.tool_results(1)[0]["content"].startswith("## Skill: damaged-item")


def test_a_name_not_on_the_list_is_answered_without_a_request(platform):
    """The SDK does not hold a tool call to its schema's enum, so DecimalAI checks the name."""
    support = da.agent_config("support")
    api = Messages(("call", "invented-skill"), "answer")
    with support.run("cracked", variables=SAM) as run:
        tool_runner(api.client(), run, "cracked")
    assert "not one of this agent's skills" in api.tool_results(1)[0]["content"]
    assert not [path for _method, path, _params in platform.requests if path.endswith("/body")]


def test_a_conversation_keeps_its_system_and_tools_when_the_prompt_and_list_change(platform):
    """Anthropic rejects a conversation whose system prompt or tool definitions change once
    thinking blocks are in its history (HTTP 400, checked live). A conversation keeps the first
    message's; the change reaches the next conversation."""
    support = da.agent_config("support")
    api = Messages("answer", "answer")
    chat = support.conversation(variables=SAM)
    history = []
    for index, question in enumerate(["Where is my order?", "It arrived cracked."]):
        if index == 1:
            support._prompt = SavedPrompt(agent_name="support", system_prompt="Changed, {{customer.name}}.",
                                          version_number=5, content_hash="d" * 64)
            platform.skills = ["damaged-item"]
        history.append({"role": "user", "content": question})
        with chat.run(question) as run:
            final = api.client().beta.messages.tool_runner(
                model="m", max_tokens=64, system=run.system, tools=[lookup_order, *run.tools],
                messages=list(history)).until_done()
        history.append({"role": "assistant", "content": [block.model_dump() for block in final.content]})
    first, second = api.requests
    assert first["system"] == second["system"] and first["tools"] == second["tools"]
    assert list(platform.queries.values()) == ["Where is my order?", "It arrived cracked."], \
        "every message is still routed on its own"
    with support.run("Hi", variables=SAM) as run:
        assert run.system[0]["text"].startswith("Changed, Sam.")


def test_a_missing_placeholder_value_stops_the_run_before_any_model_call(platform):
    support = da.agent_config("support")
    with pytest.raises(MissingPromptVariable, match="customer.name"):
        with support.run("Hi"):
            raise AssertionError("the block must not run")


def test_async_runs_give_async_tools_to_the_async_tool_runner(platform):
    support = da.agent_config("support")
    api = Messages(("call", "damaged-item"), "answer")

    async def main():
        async with support.run("My keyboard arrived cracked.", variables=SAM) as run:
            tools = run.tools
            await api.async_client().beta.messages.tool_runner(
                model="m", max_tokens=64, system=run.system, tools=[lookup_order_async, *tools],
                messages=[{"role": "user", "content": "My keyboard arrived cracked."}]).until_done()
            return tools

    (tool,) = asyncio.run(main())
    assert type(tool).__name__ == "AsyncLoadSkillTool"
    assert api.tool_results(1)[0]["content"].startswith("## Skill: damaged-item")


def test_acall_serves_a_load_in_an_async_hand_written_loop(platform):
    support = da.agent_config("support")

    async def main():
        async with support.run("cracked", variables=SAM) as run:
            return await run.atool_result({"type": "tool_use", "id": "t1", "name": "load_skill",
                                    "input": {"name": "damaged-item"}})

    result = asyncio.run(main())
    assert result["tool_use_id"] == "t1" and result["content"].startswith("## Skill: damaged-item")


def test_no_skill_list_means_no_tool(platform):
    platform.skills = []
    support = da.agent_config("support")
    with support.run("Hi", variables=SAM) as run:
        assert run.system == [{"type": "text", "text": "Hello Sam.", "cache_control": {"type": "ephemeral"}}]
        assert run.tools == [] and run.tool_params == []


def test_always_include_text_ends_the_cached_skill_block(platform):
    """D-C: an always-include skill's full text is part of the skill block, so the one cache
    breakpoint covers it and a conversation keeps it with the list."""
    platform.always = {"order-support": 1}
    with da.agent_config("support").run("Where is my order?", variables=SAM) as run:
        prompt, menu = run.system
        assert prompt == {"type": "text", "text": "Hello Sam."}
        assert menu["text"].startswith("\n\n## Available Skills")
        assert menu["text"].endswith("## Skill: order-support\n\n# Order support\n\nUse lookup_order for order facts.")
        assert menu["cache_control"] == {"type": "ephemeral"}


def test_cache_control_none_sends_no_breakpoint_and_a_ttl_is_kept(platform):
    platform.saved = SavedPrompt(agent_name="support", system_prompt="You help.", version_number=4,
                                 content_hash="c" * 64)
    with da.agent_config("support", cache_control=None).run("Hi", variables=SAM) as run:
        assert not any("cache_control" in block for block in run.system)
    with da.agent_config("support", cache_control={"type": "ephemeral", "ttl": "1h"}).run("Hi", variables=SAM) as run:
        assert run.system[-1]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    with pytest.raises(ValueError, match="cache_control"):
        da.agent_config("support", cache_control="ephemeral")


def test_a_cut_list_names_no_enum_and_the_breakpoint_still_ends_the_system_text(platform):
    platform.truncated = True
    support = da.agent_config("support")
    with support.run("My keyboard arrived cracked.", variables=SAM) as run:
        prompt, menu = run.system
        assert "cache_control" not in prompt and menu["cache_control"] == {"type": "ephemeral"}
        assert "enum" not in run.tool_params[0]["input_schema"]["properties"]["name"]


def test_the_kill_switch_reaches_configs_built_before_it(platform, monkeypatch):
    import decimalai._config as cfg

    support = da.agent_config("support")
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    with support.run("cracked", variables=SAM) as run:
        assert [block["text"] for block in run.system] == ["Hello Sam."]
        assert run.tools == [] and run.tool_params == []


def test_the_kill_switch_keeps_an_open_conversations_system_and_refuses_loads(platform, monkeypatch):
    import decimalai._config as cfg

    support = da.agent_config("support")
    chat = support.conversation(variables=SAM)
    with chat.run("Where is my order?") as run:
        system, tools = run.system, run.tool_params
    monkeypatch.setattr(cfg, "_disabled_by_init", lambda: True)
    with chat.run("cracked") as run:
        assert run.system == system and run.tool_params == tools, "a changed system would break the chat"
        result = run.tool_result({"type": "tool_use", "id": "t", "name": "load_skill", "input": {"name": "damaged-item"}})
        assert "switched off" in result["content"]


def test_system_read_without_tools_warns_at_exit(platform, caplog):
    support = da.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        with support.run("Hi", variables=SAM) as run:
            run.system
    assert any("never included it" in record.getMessage() for record in caplog.records)


def test_run_pieces_outside_the_with_block_say_how_to_use_them(platform):
    run = da.agent_config("support").run("Hi", variables=SAM)
    with pytest.raises(RuntimeError, match="inside `with support.run"):
        run.system


def test_a_run_opens_once(platform):
    run = da.agent_config("support").run("Hi", variables=SAM)
    with run:
        pass
    with pytest.raises(RuntimeError, match="opened once"):
        with run:
            pass


def test_the_run_is_one_trace_named_for_the_decimalai_agent_with_its_record(platform, monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel
    import decimalai.providers as providers

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(otel._AgentNameStamper())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(providers, "_last_provider", provider)
    support = da.agent_config("support")
    with support.run("My keyboard arrived cracked.", variables=SAM) as run:
        tool_runner(Messages(("call", "damaged-item"), "answer").client(), run, "My keyboard arrived cracked.")
    (span,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert span.name == otel.RUN_SPAN_NAME and span.attributes["decimal.agent_name"] == "support"
    rail = otel._pop_skill_rail(span.context.trace_id)
    assert rail["routing_id"] and rail["offered"] == ["damaged-item", "order-support"]
    assert rail["loaded"] == ["damaged-item"]
    attributes = otel._decimal_record_attributes(span)
    assert attributes["decimal.prompt.version"] == 4
    assert list(attributes["decimal.prompt.variables"]) == ["customer.name"], "names, never values"


def test_a_late_routing_decision_is_waited_for_and_never_left_behind(platform, monkeypatch):
    """A later run shows the local copy at once and routes in the background. Its trace is
    ended once that decision lands (off the caller's path), so the decision is on the run's
    own trace instead of an orphan record."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from concurrent.futures import ThreadPoolExecutor

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel
    import decimalai.providers as providers

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(providers, "_last_provider", provider)
    support = da.agent_config("support")
    with support.run("Where is my order?", variables=SAM):
        pass  # the first run routes before it opens
    pool = ThreadPoolExecutor(max_workers=1)
    import contextvars

    monkeypatch.setattr(binding, "_in_background",
                        lambda fn, *args: pool.submit(contextvars.copy_context().run, fn, *args))
    exporter.clear()
    with support.run("It arrived cracked.", variables=SAM) as run:
        assert run._state.get("pending") is not None
    da._drain_finishing(timeout=5)  # the span ends when the decision lands, not at the block's end
    (span,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    rail = otel._pop_skill_rail(span.context.trace_id)
    assert rail["routing_id"], "the background decision landed on its own run"
    assert len(platform.queries) == 2


def test_load_agent_turns_on_anthropic_tracing_once(platform, monkeypatch):
    import decimalai.providers as providers

    calls = []
    monkeypatch.setattr(da, "_tracing_turned_on", False)
    monkeypatch.setattr(providers, "instrument_for_runs", calls.append)
    da.agent_config("support")
    da.agent_config("support")
    assert calls == ["anthropic"]


def test_the_load_names_the_decimalai_agent_and_the_run_routes_once(platform):
    support = da.agent_config("support")
    with support.run("My keyboard arrived cracked.", variables=SAM) as run:
        tool_runner(Messages(("call", "damaged-item"), "answer").client(), run, "My keyboard arrived cracked.")
    (body,) = [params for _method, path, params in platform.requests if path.endswith("/body")]
    assert body.get("agent_name") == "support"
    assert len(platform.menu_calls()) == 1, "one decision per run, across its tool steps"


def test_a_skill_loaded_once_in_a_run_is_not_sent_again(platform):
    support = da.agent_config("support")
    api = Messages(("call", "damaged-item"), ("call", "damaged-item"), "answer")
    with support.run("cracked", variables=SAM) as run:
        tool_runner(api.client(), run, "cracked")
    assert "already loaded" in api.tool_results(2)[-1]["content"]


def test_the_removed_instrument_says_what_replaced_it():
    with pytest.raises(TypeError, match="agent_config"):
        da.instrument(enable_skill_loader=True)
    assert not hasattr(da, "skill_system") and not hasattr(da, "get_current_routing_id")


def test_the_module_imports_without_the_anthropic_sdk(monkeypatch):
    import importlib
    import sys

    monkeypatch.setitem(sys.modules, "anthropic", None)
    module = importlib.reload(da)
    try:
        with pytest.raises(ImportError, match=r"decimalai\[anthropic\]"):
            module.agent_config("support")
    finally:
        monkeypatch.delitem(sys.modules, "anthropic")
        importlib.reload(da)


# ── ported from the removed loader's tests (test_anthropic_phase1 / _phase1d) ──

_ROUTING_PATHS = ("/api/v1/skills/menu", "/api/v1/skills/route")


def test_decimalai_down_sends_the_prompt_alone_and_the_run_still_completes(platform):
    """Skills are best-effort: with the platform unreachable the run still opens, the
    model gets the saved prompt with no list and no load_skill, and the Tool Runner
    finishes. Only ``on_unavailable="fail"`` makes it an error, when the run opens."""
    platform.down = True
    api = Messages("answer")
    with da.agent_config("support").run("My keyboard arrived cracked.", variables=SAM) as run:
        assert tool_runner(api.client(), run, "My keyboard arrived cracked.").stop_reason == "end_turn"
    (request,) = api.requests
    assert request["system"] == [{"type": "text", "text": "Hello Sam.", "cache_control": {"type": "ephemeral"}}]
    assert [tool["name"] for tool in request["tools"]] == ["lookup_order"]
    with pytest.raises(RuntimeError, match="unavailable"):
        with da.agent_config("support", on_unavailable="fail").run("Hi", variables=SAM):
            raise AssertionError("the block must not run")


def test_two_async_runs_route_at_once_off_the_loop_each_on_its_own_message(platform, monkeypatch):
    """``async with`` routes on a worker thread. Two runs opened together on one event
    loop have both routing requests in flight at once, which a request made on the loop
    thread cannot allow (the second run could not start until the first returned, so
    the barrier would break). Each run's trace keeps its own decision."""
    pytest.importorskip("opentelemetry.sdk.trace")
    import threading

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel
    import decimalai.providers as providers

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(providers, "_last_provider", provider)
    both_routing = threading.Barrier(2, timeout=5)
    one_at_a_time = threading.Lock()

    def request(method, path, **kwargs):
        if path in _ROUTING_PATHS:
            both_routing.wait()
        with one_at_a_time:  # the fake numbers its decisions; keep the count exact
            return platform.request(method, path, **kwargs)

    monkeypatch.setattr(da._skill_router_singleton, "_request", request)
    support = da.agent_config("support")
    questions = ["My keyboard arrived cracked.", "Where is my order?"]

    async def one(question):
        async with support.run(question, variables=SAM) as run:
            return run.system

    async def main():
        return await asyncio.gather(*(one(question) for question in questions))

    for system in asyncio.run(main()):
        assert system[-1]["text"].lstrip().startswith("## Available Skills"), "both runs were routed"
    roots = [span for span in exporter.get_finished_spans() if span.parent is None]
    decided_for = [platform.queries[otel._pop_skill_rail(span.context.trace_id)["routing_id"]]
                   for span in roots]
    assert sorted(decided_for) == sorted(questions), "each run's trace names its own decision"


def test_cancelling_a_run_while_it_routes_cancels_at_once(platform, monkeypatch):
    """A cancelled request (a client that hung up, a timeout around the call) stops
    while the routing request is still in flight, not after the platform answers, and
    neither the block nor the model is reached. The run closes on the way out."""
    import threading

    entered, release, answered = threading.Event(), threading.Event(), threading.Event()

    def request(method, path, **kwargs):
        if path in _ROUTING_PATHS:
            entered.set()
            release.wait(5)
        try:
            return platform.request(method, path, **kwargs)
        finally:
            answered.set()

    monkeypatch.setattr(da._skill_router_singleton, "_request", request)
    support = da.agent_config("support")
    reached = []

    async def main():
        async def one():
            async with support.run("My keyboard arrived cracked.", variables=SAM):
                reached.append(True)

        task = asyncio.create_task(one())
        while not entered.is_set():
            await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cancelled_while_routing = not answered.is_set()
        release.set()  # let the abandoned request finish before the loop closes
        return cancelled_while_routing

    assert asyncio.run(main()) is True
    assert reached == []
    assert not support._open_runs, "the cancelled run was closed"


def test_the_docstring_does_not_call_agent_skills_beta():
    """Anthropic's Agent Skills went GA on 2026-08-19 (``client.skills`` is a non-beta
    resource in anthropic 1.13); the module docstring names them as the other way to
    give an agent skills."""
    doc = da.__doc__ or ""
    assert "Agent Skills" in doc and "Skills (beta)" not in doc


# ── what the design review asked for ─────────────────────────────────────────


def _memory_pipeline(monkeypatch):
    """An in-memory OTel pipeline standing in for DecimalAI's: (exporter, provider)."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import decimalai.otel as otel
    import decimalai.providers as providers

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(otel._AgentNameStamper())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(providers, "_last_provider", provider)
    return exporter, provider


def test_a_conversation_resumes_from_its_state_in_another_process(platform):
    """A server that keeps no chat in memory stores `chat.state` beside the history. A new
    config (another worker, a restart) resumes it with the same system and tools, byte for
    byte, although the saved prompt changed in between."""
    first_process = da.agent_config("support")
    chat = first_process.conversation(variables=SAM)
    assert chat.state is None, "nothing is fixed before the first run"
    with chat.run("Where is my order?") as run:
        system, tools = run.system, run.tool_params
    saved = json.loads(json.dumps(chat.state))  # it is JSON-ready
    platform.saved = SavedPrompt(agent_name="support", system_prompt="Changed.", version_number=5,
                                 content_hash="d" * 64)
    second_process = da.agent_config("support")
    resumed = second_process.conversation(state=saved)
    assert resumed.id == chat.id
    with resumed.run("It arrived cracked.") as run:
        assert run.system == system and run.tool_params == tools
    with pytest.raises(ValueError, match="not both"):
        second_process.conversation(state=saved, variables=SAM)
    with pytest.raises(ValueError, match="belongs to agent 'billing', not to 'support'") as refused:
        second_process.conversation(state={**saved, "agent": "billing"})
    assert not re.search(r"'\w+''s", str(refused.value))  # no possessive on a quoted name


def test_a_cut_list_conversation_routes_each_message_and_shows_new_skills_in_its_user_turn(
        platform, monkeypatch):
    """With a list cut to fit, which skills fit depends on the message, but a conversation's
    system cannot change. A later message is routed on its own text before its call; the skills
    it picks that the conversation has not shown ride in that message's user turn
    (`run.message`), the history only growing; its trace carries its own decision; and a load
    names the decision that showed the skill."""
    import decimalai.otel as otel

    exporter, _provider = _memory_pipeline(monkeypatch)
    platform.truncated = True
    support = da.agent_config("support")
    chat = support.conversation(variables=SAM)
    with chat.run("My keyboard arrived cracked.") as run:  # the cut list: damaged-item only
        first_decision = chat._current().routing_id
        assert run.message == {"role": "user", "content": "My keyboard arrived cracked."}
        system, tools = run.system, run.tool_params
    with chat.run("Where is my order?") as run:  # this message's pick: order-support
        assert run.system == system and run.tool_params == tools
        question, extra = run.message["content"]
        assert question == {"type": "text", "text": "Where is my order?"}
        assert "| order-support |" in extra["text"] and "damaged-item" not in extra["text"]
        for name in ("order-support", "damaged-item"):
            assert run.tool_result({"type": "tool_use", "id": name, "name": "load_skill",
                                    "input": {"name": name}})["content"].startswith(f"## Skill: {name}")
    second_decision = [rid for rid, q in platform.queries.items() if q == "Where is my order?"]
    assert len(second_decision) == 1 and second_decision[0] != first_decision
    bodies = {path.split("/")[-2]: params.get("routing_id")
              for _m, path, params in platform.requests if path.endswith("/body")}
    assert bodies == {"order-support": second_decision[0], "damaged-item": first_decision}
    runs = sorted((s for s in exporter.get_finished_spans() if s.parent is None),
                  key=lambda s: s.start_time)
    rails = [otel._pop_skill_rail(span.context.trace_id) for span in runs]
    assert [rail["routing_id"] for rail in rails] == [first_decision, second_decision[0]]
    assert chat.state["skills_shown"] == {"damaged-item": first_decision,
                                          "order-support": second_decision[0]}


def test_a_message_whose_new_skills_were_not_sent_files_no_decision(platform, monkeypatch, caplog):
    import decimalai.otel as otel

    exporter, _provider = _memory_pipeline(monkeypatch)
    platform.truncated = True
    chat = da.agent_config("support").conversation(variables=SAM)
    with chat.run("My keyboard arrived cracked."):
        pass
    exporter.clear()
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        with chat.run("Where is my order?") as run:
            run.system  # the developer sent their own user dict, not run.message
    (root,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    rail = otel._pop_skill_rail(root.context.trace_id) or {}
    assert not rail.get("routing_id"), "the model was not shown what that decision offered"
    assert any("run.message" in r.getMessage() for r in caplog.records)


def test_a_load_from_a_worker_thread_is_recorded_on_the_run(platform, monkeypatch):
    """Tool calls often run in a thread pool; the load still lands on its run's trace."""
    from concurrent.futures import ThreadPoolExecutor

    import decimalai.otel as otel

    exporter, _provider = _memory_pipeline(monkeypatch)
    support = da.agent_config("support")
    with support.run("cracked", variables=SAM) as run:
        block = {"type": "tool_use", "id": "t", "name": "load_skill", "input": {"name": "damaged-item"}}
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(run.tool_result, block).result()
    assert result["content"].startswith("## Skill: damaged-item")
    (root,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert otel._pop_skill_rail(root.context.trace_id)["loaded"] == ["damaged-item"]


def test_closing_a_run_does_not_wait_for_a_routing_decision_in_flight(platform, monkeypatch):
    """The decision a later run records in the background never holds up the caller: the block
    exits at once, and the trace is sent when the decision lands, carrying it."""
    import contextvars
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    import decimalai.otel as otel

    exporter, _provider = _memory_pipeline(monkeypatch)
    support = da.agent_config("support")
    with support.run("Where is my order?", variables=SAM):
        pass  # the agent's first run routes before it opens
    gate = threading.Event()
    real_request = platform.request

    def slow(method, path, **kwargs):
        if path.endswith("/route"):
            gate.wait(5)
        return real_request(method, path, **kwargs)

    monkeypatch.setattr(support._router(), "_request", slow)
    pool = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(binding, "_in_background",
                        lambda fn, *args: pool.submit(contextvars.copy_context().run, fn, *args))
    exporter.clear()
    started = time.monotonic()
    with support.run("It arrived cracked.", variables=SAM):
        pass
    assert time.monotonic() - started < 1.0, "the caller is not held for the decision"
    assert not [s for s in exporter.get_finished_spans() if s.parent is None], "the trace waits"
    threading.Timer(0.3, gate.set).start()
    import decimalai

    decimalai.flush()  # a short script's flush waits for it, as it does at interpreter exit
    (root,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    assert otel._pop_skill_rail(root.context.trace_id)["routing_id"], "the late decision is on it"


def test_a_run_is_its_own_trace_inside_a_web_request_span(platform, monkeypatch):
    """Two runs under one open span (a web server's request span, say) are two traces, each
    linked to that span, never one merged trace."""
    exporter, provider = _memory_pipeline(monkeypatch)
    support = da.agent_config("support")
    with provider.get_tracer("app").start_as_current_span("POST /chat") as request_span:
        for question in ("Where is my order?", "It arrived cracked."):
            with support.run(question, variables=SAM):
                pass
    runs = [s for s in exporter.get_finished_spans() if s.name == "agent.run"]
    assert len(runs) == 2 and all(s.parent is None for s in runs)
    assert len({s.context.trace_id for s in runs}) == 2
    assert all(s.links and s.links[0].context.span_id == request_span.get_span_context().span_id
               for s in runs)


def test_the_run_trace_carries_the_conversation_as_its_session(platform, monkeypatch):
    import decimalai.otel as otel

    exporter, _provider = _memory_pipeline(monkeypatch)
    chat = da.agent_config("support").conversation(variables=SAM)
    for question in ("Where is my order?", "It arrived cracked."):
        with chat.run(question):
            pass
    runs = [s for s in exporter.get_finished_spans() if s.name == "agent.run"]
    assert {s.attributes[otel.SESSION_ID_ATTRIBUTE] for s in runs} == {chat.id}


def test_outside_a_run_what_is_held_back_and_from_whom(monkeypatch):
    from types import SimpleNamespace

    import decimalai.otel as otel
    import decimalai.providers as providers

    def span(scope, trace_id):
        return SimpleNamespace(instrumentation_scope=SimpleNamespace(name=scope),
                               context=SimpleNamespace(trace_id=trace_id))

    run = SimpleNamespace(instrumentation_scope=SimpleNamespace(name="decimalai"), parent=None,
                          get_span_context=lambda: SimpleNamespace(trace_id=0xAB))
    otel._note_run_trace(run)
    monkeypatch.setattr(providers, "_runs_only", {"anthropic"})
    exporter = otel.DecimalSpanExporter()
    exporter.runs_only = True
    assert not otel._held_back(span("openinference.instrumentation.anthropic", 0xAB), exporter)
    assert otel._held_back(span("app", 0xCD), exporter), "a runs-only exporter sends only runs"
    exporter.runs_only = False
    assert not otel._held_back(span("app", 0xCD), exporter), "init(otel=True): the app's spans go"
    assert otel._held_back(span("openinference.instrumentation.anthropic", 0xCD), exporter)
    assert otel._held_back(span("com.anthropic.sdk.python", 0xCD), exporter)
    monkeypatch.setattr(providers, "_runs_only", set())
    assert not otel._held_back(span("openinference.instrumentation.anthropic", 0xCD), exporter)


def test_a_load_agent_runs_rendered_prompt_never_becomes_the_manifest_prompt(platform, monkeypatch):
    """The text the model saw has placeholder values (personal data, D8) and the skill list in
    it; the agent's prompt is saved in DecimalAI already. Only names leave the process."""
    import decimalai.otel as otel

    exporter, _provider = _memory_pipeline(monkeypatch)
    support = da.agent_config("support")
    with support.run("Where is my order?", variables=SAM):
        pass
    spans = list(exporter.get_finished_spans())
    seen = {}
    dex = otel.DecimalSpanExporter(agent_name="support")
    monkeypatch.setattr(dex, "_maybe_register_manifest",
                        lambda name, model, tools, prompts: seen.setdefault("prompts", prompts))
    monkeypatch.setattr(dex, "_send", lambda trace: seen.setdefault("trace", trace))
    real = dex._assemble_trace

    def with_a_rendered_prompt(group):
        trace, model, tools, prompts = real(group)
        return trace, model, tools, {"system": "Hello Sam. ## Available Skills"}

    monkeypatch.setattr(dex, "_assemble_trace", with_a_rendered_prompt)
    dex._finalize_trace(spans[0].context.trace_id, spans)
    assert seen["prompts"] == {}
    assert seen["trace"].session_id


def test_load_agent_after_the_kill_switch_runs_on_its_fallback_without_asking(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url=BASE, enabled=False))
    monkeypatch.setattr(cfg, "_client", None)
    called = []
    monkeypatch.setattr("decimalai.init", lambda *a, **k: called.append(k))
    support = da.agent_config("support", fallback="You help.")
    assert called == [], "the kill switch is not turned back off by agent_config()"
    assert support.prompt_source == "fallback (unreachable)"
    with support.run("Hi") as run:
        assert [block["text"] for block in run.system] == ["You help."] and run.tools == []
    with pytest.raises(RuntimeError, match="switched off"):
        da.agent_config("support")


def test_a_prompt_with_placeholders_still_caches_within_a_run(platform):
    """A filled prompt differs per customer, but a run's model calls (and a conversation's
    turns) send the same system: the breakpoint at its end makes every request after the first
    a cache read (measured live: 3,047 tokens at full price without it)."""
    platform.saved = SavedPrompt(agent_name="support", version_number=4, content_hash="c" * 64,
                                 system_prompt="You are Acme's support agent. The customer is {{customer.name}}.")
    with da.agent_config("support").run("Hi", variables=SAM) as run:
        prompt, menu = run.system
    assert prompt == {"type": "text", "text": "You are Acme's support agent. The customer is Sam."}
    assert menu["cache_control"] == {"type": "ephemeral"}


def test_refusals_are_marked_as_errors(platform):
    support = da.agent_config("support")
    with support.run("cracked", variables=SAM) as run:
        refused = run.tool_result({"type": "tool_use", "id": "t", "name": "load_skill",
                                   "input": {"name": "invented"}})
        served = run.tool_result({"type": "tool_use", "id": "u", "name": "load_skill",
                                  "input": {"name": "damaged-item"}})
        repeat = run.tool_result({"type": "tool_use", "id": "v", "name": "load_skill",
                                  "input": {"name": "damaged-item"}})
    assert refused["is_error"] is True
    assert "is_error" not in served and "is_error" not in repeat
    api = Messages(("call", "invented"), "answer")
    with support.run("cracked", variables=SAM) as run:
        tool_runner(api.client(), run, "cracked")
    assert api.tool_results(1)[0]["is_error"] is True


def test_the_exporter_shows_an_anthropic_tool_result_as_a_tool_message():
    import decimalai.otel as otel

    messages = otel._messages_from_attrs({
        "llm.input_messages.0.message.role": "user",
        "llm.input_messages.0.message.content": "cracked",
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.content": "## Skill: damaged-item",
        "llm.input_messages.1.message.tool_call_id": "toolu_1",
    }, "input")
    assert [m["role"] for m in messages] == ["user", "tool"]


def test_the_provider_pipeline_reuses_an_otel_exporter_already_on_the_provider(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai.otel as otel
    import decimalai.providers as providers

    provider = TracerProvider()
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(otel, "_installed_providers", [provider])
    monkeypatch.setattr(providers, "_pipeline_provider", None)
    added = []
    monkeypatch.setattr(provider, "add_span_processor", lambda p: added.append(p))
    assert providers._ensure_pipeline("support") is provider
    assert added == [], "a second DecimalAI exporter would send every trace twice"


def test_the_conversation_state_is_the_shared_record_every_binding_keeps(platform):
    """One mechanism for every binding: the state is ConversationRecord's, with the skill-list
    part kept apart (so the system blocks come back exactly) and which decision showed each
    skill, plus this module's conversation id."""
    from decimalai._agent_binding import ConversationRecord

    chat = da.agent_config("support").conversation(variables=SAM)
    with chat.run("Where is my order?") as run:
        system = run.system
    state = chat.state
    record = ConversationRecord.from_state(state, "support")
    assert record is not None and state["id"] == chat.id
    assert state["system"] == "".join(block["text"] for block in system)
    assert state["skills_menu"] and state["system"].endswith(state["skills_menu"])
    assert set(state["skills_shown"]) == {"damaged-item", "order-support"}


def test_a_state_without_the_newer_fields_still_resumes(platform):
    """A record saved before `skills_menu` and `skills_shown` existed loads: the list then
    stays inside one system block, and each listed skill is shown by the record's decision."""
    from decimalai._agent_binding import ConversationRecord

    support = da.agent_config("support")
    chat = support.conversation(variables=SAM)
    with chat.run("Where is my order?"):
        pass
    old = {key: value for key, value in chat.state.items() if key not in ("skills_menu", "skills_shown")}
    record = ConversationRecord.from_state(old, "support")
    assert record.menu == "" and set(record.shown) == set(record.names)
    with support.conversation(state=old).run("It arrived cracked.") as run:
        (block,) = run.system
        assert block["text"] == old["system"]


# ── what the pre-landing review found (2026-10-21) ───────────────────────────


@pytest.fixture
def traced(platform, monkeypatch):
    """DecimalAI's real tracing for agent_config(): the OpenInference instrumentor and the
    exporter, with a provider standing in for the app's global one. Yields (that provider, sent
    traces)."""
    pytest.importorskip("openinference.instrumentation.anthropic")
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai.otel as otel
    import decimalai.providers as providers

    AnthropicInstrumentor().uninstrument()
    provider = TracerProvider()
    monkeypatch.setattr(trace_api, "get_tracer_provider", lambda: provider)
    for name in ("_pipeline_provider", "_last_provider", "_pipeline_exporter"):
        monkeypatch.setattr(providers, name, None)
    for name in ("_instrumented", "_runs_via_another_tool", "_runs_only"):
        monkeypatch.setattr(providers, name, set())
    for name in ("_run_providers", "_explicitly_traced", "_openinference_provider"):
        monkeypatch.setattr(providers, name, {})
    # An earlier test's init(otel=True) / init(crewai=True) leaves its provider here, and
    # an exporter built while one is listed sends every trace, not only runs (as it should
    # after a real init(otel=True)), so the test would grade another test's setup.
    monkeypatch.setattr(otel, "_installed_providers", [])
    # The stub's reply ids repeat from test to test; a message thread is followed by them.
    monkeypatch.setattr(otel, "_thread_turns", OrderedDict())
    sent: list = []
    monkeypatch.setattr(otel.DecimalSpanExporter, "_send", lambda self, trace: sent.append(trace))
    monkeypatch.setattr(da, "_tracing_turned_on", False)
    yield provider, sent
    AnthropicInstrumentor().uninstrument()


def _ask(client, **kwargs):
    return client.messages.create(model="claude-haiku-5-5", max_tokens=16,
                                  messages=[{"role": "user", "content": "PRIVATE question"}], **kwargs)


def test_agent_config_sends_its_runs_and_nothing_else_from_the_process(traced, caplog):
    """A11: agent_config() turns Anthropic tracing on for its runs. A Claude call made outside
    one, the app's own spans on the tracer provider DecimalAI shares, and the SDK's own span of
    a call outside a run (anthropic >= 1.14: the Tool Runner's root, count_tokens) stay in the
    process; the run is one trace with its call."""
    provider, sent = traced
    support = da.agent_config("support")
    _ask(Messages("answer").client())
    with provider.get_tracer("app").start_as_current_span("GET /health"):
        pass
    with provider.get_tracer("com.anthropic.sdk.python").start_as_current_span("anthropic.messages.tool_runner"):
        pass
    assert sent == [], "nothing outside a run is DecimalAI's to send"
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        with support.run("It arrived cracked.", variables=SAM) as run:
            tool_runner(Messages(("call", "damaged-item"), "answer").client(), run, "It arrived cracked.")
    (trace,) = sent
    assert trace.agent_name == "support" and len(trace.llm_calls) == 2
    assert not [r for r in caplog.records if "no Claude request" in r.getMessage()]


def test_every_call_of_a_tool_runner_run_is_recorded_with_the_prompt_and_the_question(traced):
    """anthropic >= 2 sends the Tool Runner's later requests on a message thread, with the
    tool result alone; each call's record still holds what the model was given."""
    _provider, sent = traced
    support = da.agent_config("support")
    with support.run("It arrived cracked.", variables=SAM) as run:
        tool_runner(Messages(("call", "damaged-item"), "answer").client(), run, "It arrived cracked.")
    (trace,) = sent
    assert len(trace.llm_calls) == 2
    for call in trace.llm_calls:
        system, question = call.rendered_input[:2]
        assert system["role"] == "system" and system["content"].startswith("Hello Sam.")
        assert question == {"role": "user", "content": "It arrived cracked."}
    assert [m["role"] for m in trace.llm_calls[1].rendered_input] == ["system", "user", "assistant", "tool"]


def test_an_explicit_init_anthropic_sends_every_call_again(traced):
    import decimalai.providers as providers

    provider, sent = traced
    da.agent_config("support")
    providers.instrument(anthropic=True)  # what init(anthropic=True) runs
    _ask(Messages("answer").client())
    (trace,) = sent
    assert len(trace.llm_calls) == 1, "every call is sent, as that caller asked"


def test_runs_whose_calls_were_made_after_the_block_warn_once(traced, caplog):
    _provider, sent = traced
    support = da.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        with support.run("It arrived cracked.", variables=SAM) as run:
            lazy = run.system  # the developer read the run, and called the model after it
        _ask(Messages("answer").client(), system=lazy)
    assert any("no Claude request was traced" in r.getMessage() for r in caplog.records)


def test_another_tools_instrumentor_still_gets_decimalai_its_runs(traced, caplog):
    """Phoenix, Arize and Langfuse turn the same OpenInference instrumentor on. agent_config()
    then adds its exporter to the global provider for its runs instead of tracing nothing."""
    from openinference.instrumentation.anthropic import AnthropicInstrumentor

    provider, sent = traced
    AnthropicInstrumentor().instrument(tracer_provider=provider)  # the other tool, first
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        support = da.agent_config("support")
    assert any("turned on outside DecimalAI" in r.getMessage() for r in caplog.records)
    _ask(Messages("answer").client())
    with support.run("It arrived cracked.", variables=SAM) as run:
        _ask(Messages("answer").client(), system=run.system)
    (trace,) = sent
    assert trace.agent_name == "support" and len(trace.llm_calls) == 1


def test_a_provider_attached_later_does_not_split_a_run_from_its_calls(traced):
    """A later providers.instrument(tracer_provider=X) moved run spans to X while the Claude
    calls stayed on the provider their instrumentor was bound to: a run with no calls."""
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai.providers as providers

    openai_instrumentor = pytest.importorskip("openinference.instrumentation.openai").OpenAIInstrumentor()
    was_on = openai_instrumentor.is_instrumented_by_opentelemetry
    _provider, sent = traced
    support = da.agent_config("support")
    providers.instrument(openai=True, tracer_provider=TracerProvider())
    try:
        with support.run("It arrived cracked.", variables=SAM) as run:
            _ask(Messages("answer").client(), system=run.system)
    finally:
        if not was_on:  # process-wide: left on, it traces every later test's OpenAI calls
            openai_instrumentor.uninstrument()
    (trace,) = sent
    assert len(trace.llm_calls) == 1


def _load(run, name):
    return run.tool_result({"type": "tool_use", "id": name, "name": "load_skill", "input": {"name": name}})


def _body_decisions(platform):
    return [(path.split("/")[-2], params.get("routing_id"))
            for _m, path, params in platform.requests if path.endswith("/body")]


def _cut_chat_two_messages(platform):
    """A cut-list chat: message 1 shows damaged-item, message 2 order-support. (chat, rt1, rt2)."""
    platform.truncated = True
    chat = da.agent_config("support").conversation(variables=SAM)
    with chat.run("My keyboard arrived cracked."):
        pass
    rt1 = chat._current().routing_id
    return chat, rt1


def test_each_load_names_the_decision_that_showed_its_skill_in_either_order(platform):
    """The platform serves a public-registry skill only to a load whose decision offered it.
    Loading a skill an earlier message showed first must not make this message's own new skill
    name that earlier decision."""
    chat, rt1 = _cut_chat_two_messages(platform)
    with chat.run("Where is my order?") as run:
        for name in ("damaged-item", "order-support"):
            assert _load(run, name)["content"].startswith(f"## Skill: {name}")
    (rt2,) = [rid for rid, q in platform.queries.items() if q == "Where is my order?"]
    assert _body_decisions(platform) == [("damaged-item", rt1), ("order-support", rt2)]


def test_a_skill_shown_by_a_later_message_names_that_decision_when_this_routing_fails(platform):
    chat, rt1 = _cut_chat_two_messages(platform)
    with chat.run("Where is my order?") as run:
        run.message
    (rt2,) = [rid for rid, q in platform.queries.items() if q == "Where is my order?"]
    platform.down = True  # message 3's own routing fails
    with chat.run("Any update?") as run:
        platform.down = False
        assert _load(run, "order-support")["content"].startswith("## Skill: order-support")
    assert _body_decisions(platform) == [("order-support", rt2)]


def test_a_resumed_cut_list_chat_still_names_the_decision_that_showed_each_skill(platform):
    """chat.state carries which decision showed which skill (skills_shown) to the next worker."""
    chat, rt1 = _cut_chat_two_messages(platform)
    with chat.run("Where is my order?") as run:
        run.message
    (rt2,) = [rid for rid, q in platform.queries.items() if q == "Where is my order?"]
    saved = json.loads(json.dumps(chat.state))
    resumed = da.agent_config("support").conversation(state=saved)
    with resumed.run("It arrived cracked.") as run:
        assert _load(run, "order-support")["content"].startswith("## Skill: order-support")
    assert _body_decisions(platform) == [("order-support", rt2)]


def test_an_async_cut_list_chat_routes_each_message_too(platform):
    platform.truncated = True
    chat = da.agent_config("support").conversation(variables=SAM)

    async def go():
        async with chat.run("My keyboard arrived cracked."):
            pass
        async with chat.run("Where is my order?") as run:
            _question, extra = run.message["content"]
            assert "| order-support |" in extra["text"]
            return await run.atool_result({"type": "tool_use", "id": "t", "name": "load_skill",
                                           "input": {"name": "order-support"}})

    result = asyncio.run(go())
    assert result["content"].startswith("## Skill: order-support")
    (rt2,) = [rid for rid, q in platform.queries.items() if q == "Where is my order?"]
    assert _body_decisions(platform) == [("order-support", rt2)]


def test_without_an_asyncio_loop_anyio_carries_the_routing(platform, monkeypatch):
    """AsyncAnthropic also runs under trio (it is built on anyio), where there is no asyncio loop
    to hand the routing to; anyio's worker thread takes it there. Exercised on anyio's asyncio
    backend with the asyncio branch switched off."""
    import threading

    monkeypatch.setattr(binding, "_running_asyncio_loop", lambda: None)

    async def no_asyncio_loop(*args, **kwargs):  # what asyncio.to_thread does under trio
        raise RuntimeError("no running event loop")

    monkeypatch.setattr(asyncio, "to_thread", no_asyncio_loop)
    routed_in = []
    real_route = da.AnthropicAgentConfig._route

    def route(self, *args):
        routed_in.append(threading.get_ident())
        return real_route(self, *args)

    monkeypatch.setattr(da.AnthropicAgentConfig, "_route", route)

    async def go():
        async with da.agent_config("support").run("It arrived cracked.", variables=SAM) as run:
            return threading.get_ident(), run.tools, run.system

    loop_thread, tools, system = asyncio.run(go())
    assert tools and "## Available Skills" in system[-1]["text"], "the run got its skills"
    assert routed_in and routed_in[0] != loop_thread


def test_an_async_load_that_stops_waiting_leaves_the_decision_running(platform, monkeypatch):
    """`asyncio.wait_for(wrap_future(...))` cancelled a decision still queued in the pool when
    the load stopped waiting for it; the decision was then lost for good."""
    import contextvars
    import threading
    from concurrent.futures import ThreadPoolExecutor

    support = da.agent_config("support")
    with support.run("Where is my order?", variables=SAM):
        pass  # the first run routes before it opens; later ones in the background
    pool = ThreadPoolExecutor(max_workers=1)
    gate = threading.Event()
    pool.submit(gate.wait, 5)  # the one worker is busy: the decision queues
    monkeypatch.setattr(binding, "_in_background",
                        lambda fn, *args: pool.submit(contextvars.copy_context().run, fn, *args))
    monkeypatch.setattr(binding, "_LOAD_WAITS_FOR_ROUTING_S", 0.2)

    async def go():
        async with support.run("It arrived cracked.", variables=SAM) as run:
            pending = run._state["pending"]
            await run.atool_result({"type": "tool_use", "id": "t", "name": "load_skill",
                                    "input": {"name": "damaged-item"}})
            return pending

    pending = asyncio.run(go())
    assert not pending.cancelled(), "the decision is still on its way"
    gate.set()
    pending.result(timeout=5)
    pool.shutdown()


def test_a_long_run_keeps_its_own_state_after_many_later_runs(platform, monkeypatch):
    """The binding keeps only its latest runs' states; a run still open uses its own."""
    monkeypatch.setattr(binding, "_MAX_CONFIG_RUNS", 3)
    support = da.agent_config("support")
    with support.run("It arrived cracked.", variables=SAM) as run:
        assert _load(run, "damaged-item")["content"].startswith("## Skill: damaged-item")
        for _ in range(5):
            with support.run("Where is my order?", variables=SAM):
                pass
        assert "already loaded earlier in this run" in _load(run, "damaged-item")["content"]


def test_a_load_after_its_run_closed_is_not_credited_to_the_run_open_now(platform, monkeypatch):
    import decimalai.otel as otel

    exporter, _provider = _memory_pipeline(monkeypatch)
    support = da.agent_config("support")
    with support.run("It arrived cracked.", variables=SAM) as first:
        pass
    with support.run("Where is my order?", variables=SAM):
        _load(first, "damaged-item")  # a Tool Runner consumed after its run closed
    for root in (s for s in exporter.get_finished_spans() if s.parent is None):
        assert "damaged-item" not in ((otel._pop_skill_rail(root.context.trace_id) or {}).get("loaded") or [])


def test_a_flush_waits_for_the_runs_closed_before_it_and_leaves_later_ones_to_their_timer(monkeypatch):
    import threading
    import time

    ended: list = []

    class FakeRun:
        _exit_ns = 1

        def __init__(self, name):
            self.name = name

        def _end(self, end_ns):
            ended.append(self.name)

    class Never:
        def add_done_callback(self, fn):
            pass

    monkeypatch.setattr(provider_run, "_EXIT_WAITS_FOR_ROUTING_S", 30.0)
    monkeypatch.setattr(provider_run, "_register_drain_at_exit", lambda: None)
    before, during = FakeRun("before"), FakeRun("during")
    da._finish_later(before, Never(), 1)
    threading.Timer(0.1, lambda: da._finish_later(during, Never(), 1)).start()
    started = time.monotonic()
    da._drain_finishing(timeout=0.4)
    assert time.monotonic() - started < 1.0
    assert ended == ["before"], "a run closed during the flush keeps its own timer"
    with provider_run._finishing_lock:
        provider_run._finishing.discard(during)


def test_a_claude_call_under_the_apps_own_span_outside_a_run_is_not_sent(traced):
    """Its root is the app's request span, which DecimalAI's provider never sees: the group has
    no run at its root even when it is flushed at exit."""
    import decimalai.providers as providers

    provider, sent = traced
    da.agent_config("support")
    with provider.get_tracer("app").start_as_current_span("POST /summarize"):
        _ask(Messages("answer").client())
    providers._pipeline_exporter.force_flush()
    assert sent == []


def test_asking_for_another_providers_calls_does_not_send_claude_calls_outside_runs(traced):
    from opentelemetry.sdk.trace import TracerProvider

    import decimalai.providers as providers

    openai_instrumentor = pytest.importorskip("openinference.instrumentation.openai").OpenAIInstrumentor()
    was_on = openai_instrumentor.is_instrumented_by_opentelemetry
    _provider, sent = traced
    da.agent_config("support")
    try:
        providers.instrument(openai=True, tracer_provider=TracerProvider())
        _ask(Messages("answer").client())
    finally:
        if not was_on:
            openai_instrumentor.uninstrument()
    assert sent == [], "init(openai=True) asks for OpenAI's calls, not Claude's"


def test_agent_config_traces_on_the_apps_global_provider_when_there_is_one(traced, monkeypatch):
    """One pipeline for the process: the app's spans, another tool's and DecimalAI's runs
    share the global provider, so no span points at a parent its backend never gets."""
    from opentelemetry import trace as trace_api

    import decimalai.providers as providers

    provider, _sent = traced
    set_calls = []
    monkeypatch.setattr(trace_api, "set_tracer_provider", set_calls.append)
    da.agent_config("support")
    assert set_calls == [], "the app's provider stays the global one"
    assert providers._run_provider("anthropic") is provider
    assert providers._openinference_provider["anthropic"] is provider


def test_agent_run_and_a_later_init_anthropic_use_agent_configs_pipeline(traced):
    import decimalai.providers as providers

    _provider, sent = traced
    da.agent_config("support")
    with providers.agent_run("billing"):
        _ask(Messages("answer").client())
    (grouped,) = sent
    assert grouped.agent_name == "billing" and len(grouped.llm_calls) == 1
    sent.clear()
    providers.instrument(anthropic=True)  # what init(anthropic=True) runs
    _ask(Messages("answer").client())
    (alone,) = sent
    assert len(alone.llm_calls) == 1, "every Claude call, as init(anthropic=True) asks"


def test_asking_for_the_apps_spans_still_holds_back_claude_calls_outside_runs(traced):
    """init(otel=True) asks for the app's spans; a Claude call outside a run still needs
    init(anthropic=True)."""
    import decimalai.providers as providers

    provider, sent = traced
    da.agent_config("support")
    providers._all_traces_wanted()  # what init(otel=True) runs
    with provider.get_tracer("app").start_as_current_span("GET /health"):
        pass
    with provider.get_tracer("app").start_as_current_span("POST /summarize"):
        _ask(Messages("answer").client())
    names = sorted(span.name for trace in sent for span in trace.spans)
    assert names == ["GET /health", "POST /summarize"], names


def test_a_run_another_tools_tracing_never_reached_says_so(traced, caplog, monkeypatch):
    """Another tool turned the instrumentor on, onto a provider of its own that is not the
    global one: the run cannot get its calls, and the warning names that, not the `with` block.
    (anthropic >= 1.14 also traces its own calls onto the global provider, where the run then
    gets a record of each call without its content; switched off here, as on 1.13.)"""
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from opentelemetry.sdk.trace import TracerProvider

    monkeypatch.setenv("ANTHROPIC_OPEN_TELEMETRY", "false")
    _provider, _sent = traced
    AnthropicInstrumentor().instrument(tracer_provider=TracerProvider())  # not the global one
    support = da.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        with support.run("It arrived cracked.", variables=SAM) as run:
            _ask(Messages("answer").client(), system=run.system)
    assert any("turned on outside DecimalAI" in r.getMessage() for r in caplog.records)


class _NeverDecided:
    def add_done_callback(self, fn):
        pass


class _FinishingRun:
    _exit_ns = 1

    def __init__(self, name, ended):
        self.name, self.ended = name, ended

    def _end(self, end_ns):
        self.ended.append(self.name)


def test_at_exit_the_drain_waits_for_runs_other_threads_close_during_it(monkeypatch):
    """No run's own timer fires after exit, so the exit drain takes every run still finishing,
    not only those closed before it started (as a flush does)."""
    import threading

    ended: list = []
    monkeypatch.setattr(provider_run, "_EXIT_WAITS_FOR_ROUTING_S", 30.0)
    monkeypatch.setattr(provider_run, "_register_drain_at_exit", lambda: None)
    before, during = _FinishingRun("before", ended), _FinishingRun("during", ended)
    da._finish_later(before, _NeverDecided(), 1)
    threading.Timer(0.1, lambda: da._finish_later(during, _NeverDecided(), 1)).start()
    da._drain_finishing(timeout=0.4, at_exit=True)
    assert sorted(ended) == ["before", "during"]


def test_the_drain_is_registered_for_exit_as_the_exit_drain(monkeypatch):
    import threading

    registered = []
    monkeypatch.setattr(threading, "_register_atexit", registered.append, raising=False)
    provider_run._register_drain_at_exit()
    (drain,) = registered
    assert drain.func is provider_run._drain_finishing is da._drain_finishing and drain.keywords == {"at_exit": True}


def test_interpreter_exit_ends_a_run_still_waiting_for_its_decision(tmp_path):
    """The exit drain really runs at exit (threading's atexit, before the OTel pipelines
    shut down): a run whose decision never landed is ended, so its trace is sent."""
    import subprocess
    import sys

    out = tmp_path / "ended.txt"
    script = f"""
import sys
sys.path.insert(0, {str(__import__("pathlib").Path(da.__file__).parents[1])!r})
import decimalai.anthropic as da
da._drain_finishing.__defaults__ = (0.3,)
class Never:
    def add_done_callback(self, fn): pass
class Run:
    _exit_ns = 1
    def _end(self, end_ns):
        open({str(out)!r}, "a").write("ended\\n")
da._finish_later(Run(), Never(), 1)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=30)
    assert out.read_text() == "ended\n"


def test_a_load_after_its_run_closed_says_to_consume_the_runner_inside_the_block(platform, caplog):
    support = da.agent_config("support")
    with support.run("It arrived cracked.", variables=SAM) as run:
        pass
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        _load(run, "damaged-item")
    assert any("load_skill was called after its run closed" in r.getMessage() for r in caplog.records)


def test_an_evicted_span_count_is_not_read_as_no_claude_request(traced, caplog, monkeypatch):
    """The count is kept for the latest traces only; a long run may outlive its entry."""
    import decimalai.otel as otel

    _provider, _sent = traced
    support = da.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        with support.run("It arrived cracked.", variables=SAM) as run:
            run.system
            monkeypatch.setattr(otel, "spans_started", lambda trace_id: None)
    assert not [r for r in caplog.records if "no Claude request" in r.getMessage()]


def test_an_async_run_names_each_loads_own_decision_too(platform):
    chat, rt1 = _cut_chat_two_messages(platform)

    async def go():
        async with chat.run("Where is my order?") as run:
            for name in ("damaged-item", "order-support"):
                result = await run.atool_result({"type": "tool_use", "id": name, "name": "load_skill",
                                                 "input": {"name": name}})
                assert result["content"].startswith(f"## Skill: {name}")

    asyncio.run(go())
    (rt2,) = [rid for rid, q in platform.queries.items() if q == "Where is my order?"]
    assert _body_decisions(platform) == [("damaged-item", rt1), ("order-support", rt2)]


def test_after_init_otel_agent_configs_exporter_is_not_runs_only(traced, monkeypatch):
    """init(otel=True) asked for every trace before agent_config(): its own provider lost the
    global slot to the app's, so the exporter agent_config() puts there is the one that sends."""
    import decimalai.otel as otel
    import decimalai.providers as providers

    provider, sent = traced
    monkeypatch.setattr(otel, "_installed_providers", [object()])
    da.agent_config("support")
    assert providers._pipeline_exporter.runs_only is False
    with provider.get_tracer("app").start_as_current_span("GET /health"):
        pass
    assert [s.name for t in sent for s in t.spans] == ["GET /health"]


def test_a_sync_run_inside_trio_code_is_seen_as_inside_an_event_loop(monkeypatch):
    sniffio = pytest.importorskip("sniffio")
    monkeypatch.setattr(sniffio, "current_async_library", lambda: "trio")
    assert da._event_loop_running()
    monkeypatch.setattr(sniffio, "current_async_library",
                        lambda: (_ for _ in ()).throw(sniffio.AsyncLibraryNotFoundError()))
    assert not da._event_loop_running()


def test_a_run_stays_a_run_however_many_later_runs_start_while_it_is_open(traced, monkeypatch, caplog):
    """The run registry and the span counts are bounded; an open run is never evicted from
    either, or its late Claude calls would be held back as outside a run."""
    import decimalai.otel as otel

    monkeypatch.setattr(otel, "_RUN_TRACES_MAX", 3)
    monkeypatch.setattr(otel, "_SPANS_STARTED_MAX", 3)
    _provider, sent = traced
    support = da.agent_config("support")
    with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
        with support.run("It arrived cracked.", variables=SAM) as long_run:
            for _ in range(6):
                with support.run("Where is my order?", variables=SAM):
                    pass
            _ask(Messages("answer").client(), system=long_run.system)
    (trace,) = [t for t in sent if t.llm_calls]
    assert len(trace.llm_calls) == 1
    assert not [r for r in caplog.records if "no Claude request" in r.getMessage()]
    with support.run("Any update?", variables=SAM):
        pass  # closed runs are evictable again: the tables stay bounded
    assert len(otel._run_traces) <= 3 and len(otel._spans_started) <= 3


def test_what_was_buffered_under_a_held_back_call_goes_with_it(traced):
    """init(otel=True) beside agent_config(): a Claude call outside a run is held back, and
    the spans recorded under it (an HTTP client's) are dropped too, not buffered without end."""
    import decimalai.providers as providers

    provider, sent = traced
    da.agent_config("support")
    providers._all_traces_wanted()  # what init(otel=True) runs
    exporter = providers._pipeline_exporter
    oi = provider.get_tracer("openinference.instrumentation.anthropic")
    with oi.start_as_current_span("messages.create") as call:
        with provider.get_tracer("opentelemetry.instrumentation.httpx").start_as_current_span("POST"):
            pass
    tid = call.get_span_context().trace_id
    assert tid not in exporter._pending
    exporter.force_flush()
    assert sent == []


def test_two_first_agent_configs_at_once_both_get_tracing(platform, monkeypatch):
    import threading
    import time

    import decimalai.providers as providers

    done = []

    def slow_setup(name):
        time.sleep(0.3)
        done.append(time.monotonic())

    monkeypatch.setattr(da, "_tracing_turned_on", False)
    monkeypatch.setattr(providers, "instrument_for_runs", slow_setup)
    returned = []
    threads = [threading.Thread(target=lambda: (da.agent_config("support"), returned.append(time.monotonic())))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(done) == 1 and all(at >= done[0] for at in returned), "neither returned before tracing was up"
