"""Config-to-Claude boundary, tool delivery and actual RunTrace, without a CLI/model call.

The provider surface is a tiny deterministic SDK stand-in; no installed Claude
version is needed to exercise config semantics. A separate smoke test checks
the installed SDK's real MCP helper when that optional extra is present.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import decimalai._agent as core
import decimalai._agent_binding as binding
import decimalai._config as cfg
import decimalai.claude_agent_sdk as claude
from decimalai._agent import SavedPrompt
from decimalai._config import DecimalConfig
from decimalai.schema.manifest import ManifestTracker
from decimalai.skill_router import SkillRouter
from tests.test_agent_binding import FakePlatform
from tests.test_claude_agent_sdk_handler import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

SAM = {"customer": {"name": "Sam"}}


@dataclass
class Options:
    system_prompt: object = None
    model: str = "claude-test"
    max_turns: int = 6
    max_budget_usd: float = 0.1
    mcp_servers: object = field(default_factory=dict)
    allowed_tools: list = field(default_factory=list)
    disallowed_tools: list = field(default_factory=list)
    plugins: list = field(default_factory=list)
    resume: object = None
    continue_conversation: bool = False
    session_id: object = None
    skills: object = None


class SDK:
    ClaudeAgentOptions = Options

    def __init__(self):
        self.calls = []
        self.barrier = None
        self.load = "damaged-item"

    @staticmethod
    def tool(name, description, input_schema):
        return lambda handler: SimpleNamespace(name=name, description=description,
                                               input_schema=input_schema, handler=handler)

    @staticmethod
    def create_sdk_mcp_server(name, tools):
        return {"type": "sdk", "name": name, "tools": tools}

    async def query(self, *, prompt, options):
        self.calls.append((prompt, options))
        if self.barrier is not None:
            await self.barrier.wait()
        server = options.mcp_servers.get("decimalai")
        if server is not None and self.load:
            call = ToolUseBlock("call-1", "mcp__decimalai__load_skill", {"name": self.load})
            yield AssistantMessage([call], model=options.model)
            result = await server["tools"][0].handler(call.input)
            yield UserMessage([ToolResultBlock(call.id, result["content"],
                                              is_error=result.get("is_error", False))])
        yield AssistantMessage([TextBlock("done")], model=options.model)
        yield ResultMessage(result="done", usage={"input_tokens": 5, "output_tokens": 3},
                            session_id=options.resume or f"session-{len(self.calls)}")


@pytest.fixture
def platform(monkeypatch):
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url="http://config.test"))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "man_test"}
    monkeypatch.setattr(cfg, "_client", client)
    monkeypatch.setattr(claude, "_manifest_tracker", ManifestTracker())
    monkeypatch.setattr(claude, "_manifest_ids", {})
    monkeypatch.setattr(claude, "_manifest_hashes", {})
    monkeypatch.setattr(claude, "_pending_manifests", {})
    monkeypatch.setattr(claude, "_query_patched", False)
    monkeypatch.setattr(binding, "_in_background", lambda fn, *args: fn(*args))
    fake = FakePlatform()
    router = SkillRouter(api_key="dai_sk_test", base_url="http://config.test", inject_body=False)
    monkeypatch.setattr(router, "_request", fake.request)
    monkeypatch.setattr(core, "_skill_router_singleton", router)
    fake.saved = SavedPrompt(agent_name="support", system_prompt="Hello {{customer.name}}.",
                             version_number=4, content_hash="c" * 64)
    monkeypatch.setattr(core, "read_prompt", lambda name, **kwargs: fake.saved)
    fake.client = client
    fake.sdk = SDK()
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake.sdk)
    return fake


async def ask(config, options=None, **kwargs):
    async with config.run("My keyboard arrived cracked.", **kwargs) as run:
        return [message async for message in run.query(options=options)]


def traces(platform):
    cfg._sender.flush()
    return [call.args[0] for call in platform.client.ingest_trace.call_args_list]


def test_config_passes_saved_prompt_and_mcp_load_skill_and_records_one_trace(platform):
    support = claude.agent_config("support", fallback="fallback")
    messages = asyncio.run(ask(support, variables=SAM))
    assert len(messages) == 4
    prompt, options = platform.sdk.calls[0]
    assert prompt == "My keyboard arrived cracked."
    assert options.system_prompt.startswith("Hello Sam.\n\n## Available Skills")
    assert "mcp__decimalai__load_skill" in options.allowed_tools
    tool = options.mcp_servers["decimalai"]["tools"][0]
    assert tool.input_schema["properties"]["name"]["enum"] == ["damaged-item", "order-support"]
    assert "Ask for photos" in messages[1].content[0].content[0]["text"]
    trace, = traces(platform)
    assert trace.agent_name == "support"
    assert trace.routing_id in platform.queries
    assert trace.skills_offered_in_prompt == ["damaged-item", "order-support"]
    assert trace.skills_loaded_by_agent == trace.skills_delivered == ["damaged-item"]
    assert trace.skills_delivered_versions[0]["name"] == "damaged-item"
    assert trace.active_skills[0]["name"] == "damaged-item" and trace.active_skills[0]["hash"]
    record = next(s.attributes for s in trace.spans if s.attributes.get("decimal.prompt.source"))
    assert record["decimal.prompt.version"] == 4
    assert record["decimal.prompt.variables"] == ["customer.name"]
    assert "Sam" not in str(record), "records retain variable names, never customer values"


def test_config_preserves_callers_options_and_does_not_mutate_them(platform):
    external = {"type": "stdio", "command": "some-server"}
    plugin = {"type": "local", "path": "/my/plugin"}
    options = Options(system_prompt="old", model="another-model", max_turns=3,
                      mcp_servers={"external": external}, allowed_tools=["Read"], plugins=[plugin])
    asyncio.run(ask(claude.agent_config("support"), options, variables=SAM))
    used = platform.sdk.calls[0][1]
    assert used.model == "another-model" and used.max_turns == 3
    assert used.mcp_servers["external"] is external and used.plugins == [plugin]
    assert used.allowed_tools == ["Read", "mcp__decimalai__load_skill"]
    assert options.system_prompt == "old" and options.mcp_servers == {"external": external}
    assert options.allowed_tools == ["Read"]


def test_manifest_declares_the_template_and_not_customer_values_or_routed_menu(platform):
    support = claude.agent_config("support")
    asyncio.run(ask(support, variables=SAM))
    asyncio.run(ask(support, variables={"customer": {"name": "Alex"}}))
    assert len(traces(platform)) == 2
    platform.client.register_manifest.assert_called_once()
    snapshot = platform.client.register_manifest.call_args.args[0]
    rendered = str(snapshot.model_dump(mode="json"))
    assert "Hello {{customer.name}}." in rendered
    assert "Sam" not in rendered and "Alex" not in rendered
    assert "Available Skills" not in rendered and "mcp__decimalai__load_skill" not in rendered


@pytest.mark.parametrize("changes, reason", [
    ({"mcp_servers": {"decimalai": {}}}, "reserved"),
    ({"mcp_servers": "servers.json"}, "as a dict"),
    ({"disallowed_tools": ["mcp__decimalai__load_skill"]}, "disallowed"),
    ({"resume": "session-id"}, "bypass config"),
    ({"continue_conversation": True}, "bypass config"),
    ({"session_id": "fixed-session"}, "bypass config"),
    ({"skills": "all"}, "native skills"),
    ({"skills": ["decimalai-skills:damaged-item"]}, "native skills"),
])
def test_invalid_options_stop_before_query(platform, changes, reason):
    with pytest.raises(ValueError, match=reason):
        asyncio.run(ask(claude.agent_config("support"), Options(**changes), variables=SAM))
    assert platform.sdk.calls == [] and traces(platform) == []


def test_missing_variables_stop_before_query_and_release_run(platform):
    support = claude.agent_config("support")
    with pytest.raises(binding.MissingPromptVariable, match="customer.name"):
        asyncio.run(ask(support))
    assert support._active_runs == {} and support._runs == {}
    assert platform.sdk.calls == [] and traces(platform) == []


def test_two_agents_and_concurrent_runs_keep_their_prompt_and_loads(platform):
    first = claude.agent_config("support")
    platform.saved = SavedPrompt(agent_name="shipping", system_prompt="Shipping only.",
                                 version_number=9, content_hash="d" * 64)
    second = claude.agent_config("shipping")

    async def run():
        class Barrier:
            def __init__(self):
                self.count = 0
                self.ready = asyncio.Event()

            async def wait(self):
                self.count += 1
                if self.count == 2:
                    self.ready.set()
                await self.ready.wait()
        platform.sdk.barrier = Barrier()
        await asyncio.gather(ask(first, variables=SAM), ask(second))
    asyncio.run(run())
    by_agent = {trace.agent_name: trace for trace in traces(platform)}
    assert set(by_agent) == {"support", "shipping"}
    assert by_agent["support"].routing_id != by_agent["shipping"].routing_id
    for trace in by_agent.values():
        assert trace.skills_loaded_by_agent == ["damaged-item"]
        assert trace.skills_delivered_versions[0]["name"] == "damaged-item"
    versions = {name: next(s.attributes["decimal.prompt.version"] for s in trace.spans
                           if s.attributes.get("decimal.prompt.source"))
                for name, trace in by_agent.items()}
    assert versions == {"support": 4, "shipping": 9}
    assert first._active_runs == second._active_runs == {}


def test_existing_global_tracing_does_not_duplicate_config_trace(platform):
    claude.instrument(agent_name="global-name")
    asyncio.run(ask(claude.agent_config("support"), variables=SAM))
    trace, = traces(platform)
    assert trace.agent_name == "support"


def test_a_run_queries_once_and_its_tool_stops_after_exit(platform):
    async def run():
        pending = claude.agent_config("support").run("hello", variables=SAM)
        with pytest.raises(RuntimeError, match="inside"):
            await anext(pending.query())
        async with pending as current:
            _ = [message async for message in current.query()]
            with pytest.raises(RuntimeError, match="only once"):
                await anext(current.query())
        answer = await pending._load_skill({"name": "order-support"})
        assert answer["is_error"]
    asyncio.run(run())
    assert len(traces(platform)) == 1


def test_no_skills_does_not_install_mcp_tool(platform):
    platform.skills = []
    asyncio.run(ask(claude.agent_config("support"), variables=SAM))
    options = platform.sdk.calls[0][1]
    assert options.mcp_servers == {} and options.allowed_tools == []
    assert options.system_prompt == "Hello Sam."
    trace, = traces(platform)
    assert trace.skills_loaded_by_agent == []


def test_disabled_load_tool_uses_shared_body_injection_instead(platform, monkeypatch):
    monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key="dai_sk_test", base_url="http://config.test",
                                                      load_skill_tool=False))
    asyncio.run(ask(claude.agent_config("support"), variables=SAM))
    options = platform.sdk.calls[0][1]
    assert options.mcp_servers == {}
    assert "Ask for photos" in options.system_prompt
    trace, = traces(platform)
    assert trace.skills_loaded_by_agent == []


def test_keyless_run_keeps_fallback_and_does_not_send_a_trace(platform, monkeypatch):
    from decimalai import _keyless
    platform.saved = SimpleNamespace(agent_name="support", system_prompt="Local fallback.",
                                     is_fallback=True, keyless=True)
    monkeypatch.setattr(_keyless, "keyless_router", lambda: None)
    asyncio.run(ask(claude.agent_config("support", fallback="Local fallback.")))
    assert platform.sdk.calls[0][1].system_prompt == "Local fallback."
    assert traces(platform) == []


def test_actual_sdk_can_construct_the_config_mcp_server_without_cli(platform, monkeypatch):
    monkeypatch.delitem(sys.modules, "claude_agent_sdk")
    sdk = pytest.importorskip("claude_agent_sdk")
    platform.sdk.load = None

    async def run():
        async with claude.agent_config("support").run("hello", variables=SAM) as current:
            options = current._options(sdk.ClaudeAgentOptions(max_turns=2))
            assert options.mcp_servers["decimalai"]["type"] == "sdk"
            assert options.mcp_servers["decimalai"]["instance"] is not None
            assert options.max_turns == 2
    asyncio.run(run())


@pytest.mark.parametrize("text", [
    "load_skill error: skill router unavailable.",
    "load_skill: no skill named 'unknown'.",
    "load_skill: that skill isn't available for this run.",
])
def test_skill_load_refusals_are_mcp_errors_without_delivery_credit(platform, monkeypatch, text):
    support = claude.agent_config("support")
    monkeypatch.setattr(support, "_serve_load", lambda name: text)
    messages = asyncio.run(ask(support, variables=SAM))
    assert messages[1].content[0].is_error is True
    trace, = traces(platform)
    assert trace.skills_loaded_by_agent == trace.skills_delivered == []
    assert trace.active_skills == []
    tool_span = next(span for span in trace.spans if span.name == "mcp__decimalai__load_skill")
    assert tool_span.status.value == "error"


def test_exiting_after_first_message_closes_the_original_sdk_query(platform, monkeypatch):
    closed = []

    async def source(*, prompt, options):
        try:
            yield AssistantMessage([TextBlock("first")])
            yield ResultMessage(result="done")
        finally:
            closed.append(True)
    monkeypatch.setattr(platform.sdk, "query", source)

    async def run():
        async with claude.agent_config("support").run("hello", variables=SAM) as current:
            iterator = current.query()
            await anext(iterator)
        assert closed == [True], "exit awaits provider cleanup in the caller's task"
        await iterator.aclose()
    asyncio.run(run())
    assert len(traces(platform)) == 1


def test_a_conversation_keeps_prompt_and_tool_schema_and_resumes_its_own_session(platform):
    support = claude.agent_config("support")

    async def run():
        chat = support.conversation(variables=SAM)
        async with chat.run("First question") as first:
            _ = [message async for message in first.query()]
        platform.saved = SavedPrompt(agent_name="support", system_prompt="Changed prompt.",
                                     version_number=5, content_hash="d" * 64)
        support._refresh_prompt()
        async with chat.run("Second question") as second:
            _ = [message async for message in second.query()]
        async with support.run("Fresh question") as fresh:
            _ = [message async for message in fresh.query()]
    asyncio.run(run())
    first, second, fresh = [options for _prompt, options in platform.sdk.calls]
    assert first.resume is None and second.resume == "session-1"
    assert first.system_prompt == second.system_prompt
    assert first.mcp_servers["decimalai"]["tools"][0].input_schema == second.mcp_servers["decimalai"]["tools"][0].input_schema
    assert fresh.system_prompt.startswith("Changed prompt.") and fresh.resume is None
    first_trace, second_trace, _ = traces(platform)
    assert first_trace.session_id == second_trace.session_id == "session-1"
    assert first_trace.routing_id and second_trace.routing_id


def test_a_cut_list_chat_routes_each_message_and_sends_new_skills_in_the_user_turn(platform):
    platform.truncated = True
    support = claude.agent_config("support")

    async def run():
        chat = support.conversation(variables=SAM)
        async with chat.run("The keyboard is cracked") as first:
            _ = [message async for message in first.query()]
        platform.sdk.load = "order-support"
        async with chat.run("Where is the next order?") as second:
            _ = [message async for message in second.query()]
    asyncio.run(run())
    (first_prompt, first), (second_prompt, second) = platform.sdk.calls
    assert first.system_prompt == second.system_prompt
    assert "order-support" not in first.system_prompt and "order-support" in second_prompt
    assert first_prompt == "The keyboard is cracked"
    first_trace, second_trace = traces(platform)
    assert first_trace.routing_id != second_trace.routing_id
    assert platform.queries[second_trace.routing_id] == "Where is the next order?"
    assert second_trace.skills_loaded_by_agent == ["order-support"]
    assert second_trace.user_input_preview == "Where is the next order?"
    assert "order-support" in str(second_trace.llm_calls[0].rendered_input)


def test_actual_sdk_query_executes_the_real_mcp_load_over_local_control_transport(platform, monkeypatch):
    monkeypatch.delitem(sys.modules, "claude_agent_sdk")
    sdk = pytest.importorskip("claude_agent_sdk")
    from tests.conformance.drivers import Ctx
    from tests.conformance.drivers.claude_agent_sdk import _frames, _stub_transport

    ctx = Ctx(base_url="http://config.test", api_key="dai_sk_test", agent_name="support",
              prompt_sentinel="question", reply_sentinel="answer", tool_name="lookup",
              tool_sentinel="tool-result", workdir=".")

    async def run():
        async with claude.agent_config("support").run("question", variables=SAM) as current:
            return [message async for message in current.query(
                options=sdk.ClaudeAgentOptions(model="claude-test", max_turns=2),
                transport=_stub_transport(_frames(ctx, skill="damaged-item")),
            )]
    messages = asyncio.run(run())
    assert any(type(message).__name__ == "ResultMessage" for message in messages)
    trace, = traces(platform)
    assert trace.skills_loaded_by_agent == ["damaged-item"]
    assert "Ask for photos" in str([call.rendered_input for call in trace.llm_calls])
