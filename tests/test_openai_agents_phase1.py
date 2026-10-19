"""OpenAI Agents SDK adapter: one tracing processor, and `load_agent()` agents in
the places the old process-wide skill loader broke.

Driven through the REAL Agents SDK (no stand-in ``agents`` module) with a
scripted in-process model, so nothing here contacts a provider or the DecimalAI
backend. Until 0.18 the adapter patched the Agents SDK; the 2026-10-13 audit
found three silent failures in that loader (``SandboxAgent`` got no skills, a
second ``instrument()`` sent every run twice, routing ran on the event loop).
These tests keep the guarantees, for the explicit `load_agent()` shape:

1. ``instrument()`` is idempotent: one ``DecimalTracingProcessor`` however often
   it is called, and beside ``load_agent()``.
2. ``SandboxAgent``: a ``load_agent()`` menu reaches a sandbox run; an agent
   whose own ``Skills`` capability serves its skills gets no second catalog.
3. Routing never runs on the event loop.
4. ``tool_use_behavior`` stays the developer's: ``stop_on_first_tool`` turns
   DecimalAI's skills off for that agent (a skill's text would be the answer),
   and a tool of the developer's own named ``load_skill`` is never shadowed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

agents = pytest.importorskip("agents")

from agents import Runner  # noqa: E402
from agents.items import ModelResponse  # noqa: E402
from agents.models.interface import Model  # noqa: E402
from agents.run_config import RunConfig  # noqa: E402
from agents.sandbox import SandboxAgent, SandboxRunConfig  # noqa: E402
from agents.sandbox.capabilities import LocalDirLazySkillSource, Skills  # noqa: E402
from agents.sandbox.entries import LocalDir  # noqa: E402
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient  # noqa: E402
from agents.tracing import get_trace_provider, set_trace_processors  # noqa: E402
from agents.usage import Usage  # noqa: E402
from openai.types.responses import (  # noqa: E402
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

import decimalai.openai_agents as oa  # noqa: E402
from decimalai import skill_router as sr  # noqa: E402

SKILL = "refund-policy"
MENU = f"## Recommended Skills\n| {SKILL} | how refunds are handled |"
BODY = f"## Skill: {SKILL}\n\nREFUND BODY SENTINEL"
ROUTING_ID = "rt_0123456789abcdef01234567"
QUESTION = "how do refunds work?"
PROMPT = "You are helpful."


# ── a scripted model ─────────────────────────────────────────────────────────


def _text(text: str) -> ModelResponse:
    message = ResponseOutputMessage(
        id="msg_1", type="message", role="assistant", status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return ModelResponse(output=[message], usage=Usage(), response_id=None)


def _call(name: str, arguments: dict) -> ModelResponse:
    call = ResponseFunctionToolCall(
        id="fc_1", call_id="call_1", type="function_call",
        name=name, arguments=json.dumps(arguments),
    )
    return ModelResponse(output=[call], usage=Usage(), response_id=None)


def _turn(*calls: tuple) -> ModelResponse:
    """One model turn that calls each ``(name, arguments)`` in order."""
    return ModelResponse(output=[
        ResponseFunctionToolCall(
            id=f"fc_{i}", call_id=f"call_{name}_{i}", type="function_call",
            name=name, arguments=json.dumps(arguments),
        )
        for i, (name, arguments) in enumerate(calls)
    ], usage=Usage(), response_id=None)


class _ScriptedModel(Model):
    """Answers from a script and records what each model call was shown."""

    def __init__(self, script: List[ModelResponse]):
        self.script = list(script)
        self.calls: List[dict] = []

    async def get_response(self, system_instructions, input, model_settings, tools,
                           output_schema, handoffs, tracing, *, previous_response_id=None,
                           conversation_id=None, prompt=None, **_):
        self.calls.append({"instructions": system_instructions or "", "tools": list(tools),
                           "input": input})
        return self.script.pop(0) if self.script else _text("done")

    def stream_response(self, *args, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError


# ── a router stand-in that records where and how it was called ──────────────


class _Router:
    """Answers like a SkillRouter on a 200: one menu, one body."""

    def __init__(self, wait: Optional[threading.Event] = None):
        self.routes: List[dict] = []
        self.loads: List[dict] = []
        self.wait = wait
        self.released_while_routing: Optional[bool] = None

    def build_prompt_fragment(self, query=None, *, agent_name=None, scope=None, **_):
        self.routes.append({"query": query, "agent_name": agent_name, "scope": scope,
                            "thread": threading.get_ident()})
        if self.wait is not None:
            # On the loop thread this can only time out: the coroutine that
            # sets the event cannot run while this call holds the loop.
            self.released_while_routing = self.wait.wait(timeout=2)
        sr._last_offered_names_ctx.set([SKILL])
        return MENU, ROUTING_ID

    def build_prompt_parts(self, query=None, **kwargs):
        fragment, routing_id = self.build_prompt_fragment(query, **kwargs)
        return fragment, "", routing_id

    def load_skill(self, name, *, scope=None, agent_name=None):
        self.loads.append({"name": name, "scope": scope, "agent_name": agent_name,
                           "thread": threading.get_ident()})
        return BODY if name == SKILL else f"load_skill: no skill named {name!r}"

    # The rails the adapter drains. Nothing is held here between calls.
    def consume_routing_id(self, scope=None):
        return None

    def consume_offered_names(self, scope=None):
        return []

    def consume_delivered_names(self, scope=None):
        return []

    def consume_loaded_names(self, scope=None):
        return []

    def consume_loaded_hashes(self, scope=None):
        return {}

    def consume_delivered_versions(self, scope=None):
        return []


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sdk(monkeypatch):
    """A configured SDK whose client is a mock (saved prompt: ``PROMPT``), no
    router, no processor, and inline background work."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    monkeypatch.setenv("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True,
    ))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "m-1", "status": "active"}
    client.get_agent_prompt.side_effect = lambda name, version=None: {
        "agent_name": name, "system_prompt": PROMPT, "version_number": 1, "content_hash": "h" * 64,
    }
    monkeypatch.setattr(cfg, "_client", client)
    for name, value in (
        ("_skill_router_singleton", None),
        ("_manifest_id", None),
        ("_manifest_tracker", ManifestTracker()),
        ("_tracing_processor", None),
        ("_tracing_processor_added", False),
        ("_run_rails", OrderedDict()),
        ("_config_pool", None),
    ):
        monkeypatch.setattr(oa, name, value)
    monkeypatch.setattr(oa, "_in_background", lambda fn, *args: fn(*args))
    monkeypatch.setattr(oa, "_two_catalogs_warned", False, raising=False)

    provider = get_trace_provider()
    saved = list(provider._multi_processor._processors)
    set_trace_processors([])
    sr._last_offered_names_ctx.set(None)
    try:
        yield client
    finally:
        set_trace_processors(saved)


def _sent(client) -> list:
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in client.ingest_trace.call_args_list]


def _ours() -> list:
    return [p for p in get_trace_provider()._multi_processor._processors
            if isinstance(p, oa.DecimalTracingProcessor)]


def _sandbox_run(agent, text: str = QUESTION):
    return Runner.run_sync(
        agent, text,
        run_config=RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient())),
    )


@pytest.fixture
def lazy_skills_dir(tmp_path: Path) -> Path:
    skill = tmp_path / "native-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: native-skill\ndescription: a skill the sandbox loads itself\n---\nbody\n"
    )
    return tmp_path


@pytest.fixture
def router(sdk):
    router = _Router()
    oa._skill_router_singleton = router
    return router


# ── 1. instrument() is idempotent ────────────────────────────────────────────


class TestInstrumentIsIdempotent:
    def test_instrument_then_load_agent_leaves_one_processor(self, router):
        other = MagicMock(name="users-own-processor")
        set_trace_processors([other])

        oa.instrument(agent_name="support", disk_sync=False)        # what init() does
        oa.load_agent("support")

        assert len(_ours()) == 1
        assert other in get_trace_provider()._multi_processor._processors
        assert _ours()[0].default_agent_name == "support"

    def test_no_mix_of_calls_stacks_a_second_processor(self, sdk):
        oa.instrument(disk_sync=False)
        oa.instrument(disk_sync=False, exclusive=True)
        oa.instrument(disk_sync=False)
        oa.instrument(agent_name="b", disk_sync=False)
        assert len(_ours()) == 1
        assert _ours()[0].default_agent_name == "b"

    def test_a_processor_the_user_removed_is_registered_again(self, sdk):
        oa.instrument(disk_sync=False)
        set_trace_processors([])
        oa.instrument(disk_sync=False)
        assert len(_ours()) == 1

    def test_one_run_sends_one_trace_after_instrument_and_load_agent(self, router, sdk):
        oa.instrument(agent_name="support", disk_sync=False)
        support = oa.load_agent("support")

        Runner.run_sync(agents.Agent(name=support.name, instructions=support.instructions,
                                     tools=[*support.tools],
                                     model=_ScriptedModel([_text("ok")])), QUESTION)

        traces = _sent(sdk)
        assert len(traces) == 1
        assert traces[0].routing_id == ROUTING_ID


# ── 2. SandboxAgent ──────────────────────────────────────────────────────────


class TestSandboxAgents:
    def test_a_sandbox_run_shows_the_menu_and_can_load(self, router):
        support = oa.load_agent("support")
        model = _ScriptedModel([_call("load_skill", {"name": SKILL}), _text("ok")])
        _sandbox_run(SandboxAgent(name="support", instructions=support.instructions,
                                  tools=[*support.tools], model=model))

        assert [r["agent_name"] for r in router.routes] == ["support"]
        shown = model.calls[0]["instructions"]
        assert PROMPT in shown and shown.count(MENU) == 1
        assert "load_skill" in [t.name for t in model.calls[0]["tools"]]
        assert [load["name"] for load in router.loads] == [SKILL]

    def test_an_agent_with_its_own_skills_capability_gets_no_second_catalog(
        self, router, lazy_skills_dir, caplog,
    ):
        support = oa.load_agent("support")
        model = _ScriptedModel([_text("ok")])
        agent = SandboxAgent(
            name="support", instructions=support.instructions, tools=[*support.tools],
            model=model, capabilities=[Skills(lazy_from=LocalDirLazySkillSource(
                source=LocalDir(src=str(lazy_skills_dir))))],
        )
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            _sandbox_run(agent)
            _sandbox_run(agent)

        assert router.routes == [], "the sandbox's own catalog serves this agent"
        assert MENU not in model.calls[0]["instructions"]
        assert [t.name for t in model.calls[0]["tools"]].count("load_skill") == 1  # the sandbox's
        assert sum("sandbox Skills capability" in r.getMessage() for r in caplog.records) == 1


# ── 3. routing is never made on the event loop ───────────────────────────────


class TestRoutingDoesNotBlockTheEventLoop:
    def test_the_first_runs_routing_is_made_on_a_worker_thread(self, sdk):
        router = _Router(wait=threading.Event())
        oa._skill_router_singleton = router
        support = oa.load_agent("support")
        agent = agents.Agent(name="support", instructions=support.instructions, tools=[*support.tools])

        async def resolve():
            async def release():
                await asyncio.sleep(0)
                router.wait.set()

            ctx = agents.RunContextWrapper(context=None, turn_input=[{"role": "user", "content": QUESTION}])
            text, _ = await asyncio.gather(support.instructions(ctx, agent), release())
            return text, threading.get_ident()

        text, loop_thread = asyncio.run(resolve())
        assert router.released_while_routing is True, "the routing call held the event loop"
        assert router.routes[0]["thread"] != loop_thread
        assert router.routes[0]["query"] == QUESTION
        assert MENU in text


# ── 4. tool_use_behavior stays the developer's ───────────────────────────────


ORDER = "ORDER 42: refunded"


def _lookup_order():
    @agents.function_tool
    def lookup_order(order_id: str) -> str:
        """Look up an order."""
        return f"ORDER {order_id}: refunded"

    return lookup_order


class TestToolUseBehavior:
    def _agent(self, support, script, behavior, *, tools=None):
        return agents.Agent(
            name="support", instructions=support.instructions,
            model=_ScriptedModel(script),
            tools=(tools if tools is not None else [_lookup_order()]) + list(support.tools),
            tool_use_behavior=behavior,
        )

    def test_stop_on_first_tool_turns_skills_off_for_that_agent(self, router, caplog):
        support = oa.load_agent("support")
        agent = self._agent(support, [_turn(("lookup_order", {"order_id": "42"}))], "stop_on_first_tool")
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            result = Runner.run_sync(agent, "refund status for order 42?")
        assert result.final_output == ORDER
        assert [t.name for t in agent.model.calls[0]["tools"]] == ["lookup_order"]
        assert MENU not in agent.model.calls[0]["instructions"]
        assert router.routes == []
        assert sum("StopAtTools" in r.getMessage() for r in caplog.records) == 1

    def test_run_llm_again_reads_the_body_then_answers(self, router):
        support = oa.load_agent("support")
        agent = self._agent(support, [_turn(("load_skill", {"name": SKILL})), _text("final answer")],
                            "run_llm_again")
        assert Runner.run_sync(agent, QUESTION).final_output == "final answer"
        assert "REFUND BODY SENTINEL" in json.dumps(agent.model.calls[1]["input"], default=str)

    def test_stop_at_tool_names_still_stops_at_the_named_tool(self, router):
        support = oa.load_agent("support")
        agent = self._agent(
            support,
            [_turn(("load_skill", {"name": SKILL})), _turn(("lookup_order", {"order_id": "42"}))],
            {"stop_at_tool_names": ["lookup_order"]},
        )
        assert Runner.run_sync(agent, QUESTION).final_output == ORDER
        assert [load["name"] for load in router.loads] == [SKILL]

    def test_a_custom_function_is_warned_that_it_sees_load_skill(self, router, caplog):
        support = oa.load_agent("support")
        seen: List[List[str]] = []

        def first_order_lookup(context, tool_results):
            seen.append([r.tool.name for r in tool_results])
            for r in tool_results:
                if r.tool.name == "lookup_order":
                    return agents.ToolsToFinalOutputResult(is_final_output=True, final_output=r.output)
            return agents.ToolsToFinalOutputResult(is_final_output=False, final_output=None)

        agent = self._agent(
            support,
            [_turn(("load_skill", {"name": SKILL})), _turn(("lookup_order", {"order_id": "42"}))],
            first_order_lookup,
        )
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            assert Runner.run_sync(agent, "refund status for order 42?").final_output == ORDER
        assert seen == [["load_skill"], ["lookup_order"]]
        assert any("tool_use_behavior function" in r.getMessage() for r in caplog.records)

    def test_the_developers_own_load_skill_tool_is_never_shadowed(self, router, caplog):
        support = oa.load_agent("support")

        @agents.function_tool
        def load_skill(name: str) -> str:
            """The developer's own loader."""
            return f"USER LOADER: {name}"

        agent = self._agent(support, [_turn(("load_skill", {"name": "x"})), _text("done")],
                            "run_llm_again", tools=[load_skill])
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            Runner.run_sync(agent, QUESTION)
        tools = [t.name for t in agent.model.calls[0]["tools"]]
        assert tools.count("load_skill") == 1
        assert "USER LOADER: x" in json.dumps(agent.model.calls[1]["input"], default=str)
        assert MENU not in agent.model.calls[0]["instructions"]
        assert any("already has a tool named load_skill" in r.getMessage() for r in caplog.records)
