"""Phase 1 of the 2026-10-13 integration audit, OpenAI Agents SDK adapter.

Three silent failures, each driven through the REAL Agents SDK (no stand-in
``agents`` module) with a scripted in-process model, so nothing here contacts a
provider or the DecimalAI backend:

1. ``SandboxAgent`` got no skills at all. It is a ``@dataclass`` subclass with
   its own generated ``__init__``, so the loader's ``Agent.__init__`` patch never
   ran on it; and at run time the sandbox executes a CLONE whose
   ``instructions`` is the sandbox's own builder callable, which the run-time
   retrofit mistook for "a callable the user wrote". Router calls: zero. With a
   lazy native ``Skills`` capability the sandbox also registers its OWN tool
   called ``load_skill``, so DecimalAI either backed off entirely or, had both
   been registered, lost dispatch to the native one (last name wins, no error).
2. ``instrument()`` stacked a second ``DecimalTracingProcessor`` on every call.
   ``init(openai_agents=True)`` followed by ``instrument(enable_skill_loader=True)``
   -- the only way to add skills after the documented one-liner -- sent every
   run twice, with the skill telemetry on only one of the two copies.
3. The routing call -- a blocking HTTP request -- ran ON the event loop, from
   the async ``get_system_prompt``. In an async server that stalls every other
   conversation in the process while the platform answers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import types
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
NAMESPACED = "decimalai_load_skill"


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
    """A configured SDK whose client is a mock, and an adapter with nothing
    installed. Every patch the loader makes to the Agent classes is undone."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    monkeypatch.setenv("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True,
    ))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "m-1", "status": "active"}
    monkeypatch.setattr(cfg, "_client", client)

    for method in ("__init__", "get_system_prompt", "get_all_tools"):
        monkeypatch.setattr(agents.Agent, method, getattr(agents.Agent, method))
    for name, value in (
        ("_skill_loader_installed", False),
        ("_agent_hooks_installed", False),
        ("_retrofit_notice_emitted", False),
        ("_skill_router_singleton", None),
        ("_load_skill_tool_registration_failed", False),
        ("_manifest_id", None),
        ("_manifest_tracker", ManifestTracker()),
    ):
        monkeypatch.setattr(oa, name, value)
    # Introduced by the fix; absent before it, hence raising=False.
    monkeypatch.setattr(oa, "_namespace_notice_emitted", False, raising=False)
    monkeypatch.setattr(oa, "_tracing_processor", None, raising=False)
    monkeypatch.setattr(oa, "_run_rails", OrderedDict())
    monkeypatch.setattr(oa, "_load_skill_tool_enabled", lambda: True)

    provider = get_trace_provider()
    saved = list(provider._multi_processor._processors)
    set_trace_processors([])
    sr._last_offered_names_ctx.set(None)
    oa._routing_id_ctx.set(None)
    oa._skills_offered_ctx.set(None)
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


# ── 1. SandboxAgent ──────────────────────────────────────────────────────────


class TestSandboxAgentGetsSkills:
    @pytest.mark.parametrize("built", ["before_instrument", "after_instrument"])
    def test_a_sandbox_run_routes_and_shows_the_menu(self, sdk, built):
        router = _Router()
        oa._skill_router_singleton = router
        model = _ScriptedModel([_text("ok")])

        def build():
            return SandboxAgent(name="sbx", instructions="You are helpful.", model=model)

        agent = build() if built == "before_instrument" else None
        oa.instrument(enable_skill_loader=True, disk_sync=False, exclusive=True)
        agent = agent or build()
        _sandbox_run(agent)

        assert [r["agent_name"] for r in router.routes] == ["sbx"]
        assert router.routes[0]["query"] == QUESTION
        shown = model.calls[0]["instructions"]
        assert shown.count(MENU) == 1
        assert "You are helpful." in shown
        # The menu points at a tool that is really there, and it is ours.
        assert "call the load_skill tool" in shown
        tools = [t for t in model.calls[0]["tools"] if t.name == "load_skill"]
        assert len(tools) == 1 and hasattr(tools[0], oa._SKILL_TOOL_AGENT_ATTR)
        # The user's declaration is untouched: the skills arrive per run.
        assert agent.instructions == "You are helpful."

    def test_a_native_lazy_load_skill_is_kept_and_ours_is_namespaced(
        self, sdk, lazy_skills_dir, caplog,
    ):
        """The sandbox's lazy ``Skills`` registers its own ``load_skill``. Both
        catalogs must stay loadable, through two tools with distinct names."""
        router = _Router()
        oa._skill_router_singleton = router
        model = _ScriptedModel([_call(NAMESPACED, {"name": SKILL}), _text("ok")])
        oa.instrument(enable_skill_loader=True, disk_sync=False, exclusive=True)
        agent = SandboxAgent(
            name="sbx", instructions="You are helpful.", model=model,
            capabilities=[Skills(lazy_from=LocalDirLazySkillSource(
                source=LocalDir(src=lazy_skills_dir)))],
        )

        with caplog.at_level(logging.WARNING, logger="decimalai.openai_agents"):
            _sandbox_run(agent)

        names = [t.name for t in model.calls[0]["tools"]]
        assert sorted(names) == sorted(set(names)), f"duplicate tool names: {names}"
        assert {"load_skill", NAMESPACED} <= set(names)
        native = next(t for t in model.calls[0]["tools"] if t.name == "load_skill")
        assert not hasattr(native, oa._SKILL_TOOL_AGENT_ATTR)  # still the sandbox's own
        shown = model.calls[0]["instructions"]
        assert shown.count(MENU) == 1
        assert f"call the {NAMESPACED} tool" in shown
        # The registry skill went through DecimalAI, as this agent.
        assert [(load["name"], load["agent_name"]) for load in router.loads] == [(SKILL, "sbx")]
        assert any("REFUND BODY SENTINEL" in str(item) for item in model.calls[1]["input"])
        # ...and the trace says so: offered, loaded, under this run's routing id.
        (trace,) = _sent(sdk)
        assert trace.routing_id == ROUTING_ID
        assert trace.skills_loaded_by_agent == [SKILL]
        assert SKILL in trace.skills_offered_in_prompt
        # The rename is announced, once.
        notices = [r for r in caplog.records if NAMESPACED in r.getMessage()]
        assert len(notices) == 1

    def test_eager_native_skills_add_no_tool_so_ours_keeps_its_name(self, sdk):
        """``Skills(skills=[...])`` mounts the skills as files and registers no
        tool, so there is nothing to collide with: one ``load_skill``, ours."""
        from agents.sandbox.capabilities import Skill

        router = _Router()
        oa._skill_router_singleton = router
        model = _ScriptedModel([_text("ok")])
        oa.instrument(enable_skill_loader=True, disk_sync=False, exclusive=True)
        _sandbox_run(SandboxAgent(
            name="sbx", instructions="You are helpful.", model=model,
            capabilities=[Skills(skills=[Skill(
                name="native-skill", description="mounted by the sandbox",
                content="---\nname: native-skill\ndescription: d\n---\nbody\n",
            )])],
        ))

        tools = [t for t in model.calls[0]["tools"] if t.name in ("load_skill", NAMESPACED)]
        assert [t.name for t in tools] == ["load_skill"]
        assert hasattr(tools[0], oa._SKILL_TOOL_AGENT_ATTR)
        assert model.calls[0]["instructions"].count(MENU) == 1
        assert len(router.routes) == 1

    def test_the_namespace_notice_is_emitted_once_per_process(self, sdk, lazy_skills_dir, caplog):
        oa._skill_router_singleton = _Router()
        oa.instrument(enable_skill_loader=True, disk_sync=False, exclusive=True)
        caps = [Skills(lazy_from=LocalDirLazySkillSource(source=LocalDir(src=lazy_skills_dir)))]
        with caplog.at_level(logging.WARNING, logger="decimalai.openai_agents"):
            for _ in range(2):
                _sandbox_run(SandboxAgent(name="sbx", instructions="x",
                                          model=_ScriptedModel([_text("ok")]),
                                          capabilities=caps))
        assert len([r for r in caplog.records if NAMESPACED in r.getMessage()]) == 1

    def test_a_user_written_sandbox_instructions_callable_is_left_alone(self, sdk):
        """Same policy as a plain Agent: the user's own callable decides."""
        router = _Router()
        oa._skill_router_singleton = router
        model = _ScriptedModel([_text("ok")])
        oa.instrument(enable_skill_loader=True, disk_sync=False, exclusive=True)
        _sandbox_run(SandboxAgent(name="sbx", instructions=lambda ctx, agent: "MINE",
                                  model=model))

        assert router.routes == []
        assert MENU not in model.calls[0]["instructions"]
        assert "MINE" in model.calls[0]["instructions"]

    def test_a_plain_agent_keeps_the_plain_tool_name(self, sdk):
        router = _Router()
        oa._skill_router_singleton = router
        loop_threads = []

        class _Recording(_ScriptedModel):
            async def get_response(self, *args, **kwargs):
                loop_threads.append(threading.get_ident())
                return await super().get_response(*args, **kwargs)

        model = _Recording([_call("load_skill", {"name": SKILL}), _text("ok")])
        oa.instrument(enable_skill_loader=True, disk_sync=False, exclusive=True)
        Runner.run_sync(agents.Agent(name="plain", instructions="You are helpful.",
                                     model=model), QUESTION)

        assert "load_skill" in [t.name for t in model.calls[0]["tools"]]
        assert NAMESPACED not in [t.name for t in model.calls[0]["tools"]]
        assert "call the load_skill tool" in model.calls[0]["instructions"]
        assert [load["name"] for load in router.loads] == [SKILL]
        # The SDK runs a sync function tool on a worker thread; the body fetch
        # must stay off the loop too.
        assert router.loads[0]["thread"] not in loop_threads
        (trace,) = _sent(sdk)
        assert trace.skills_loaded_by_agent == [SKILL]

    def test_the_namespaced_tool_stays_out_of_the_manifest(self, sdk):
        """Like `load_skill`, it is this adapter's affordance, not the agent's
        declared contract: listing it would version the agent on whether the
        loader was on."""
        declared = types.SimpleNamespace(name="lookup", params_json_schema={"type": "object"})
        agent = types.SimpleNamespace(tools=[declared, types.SimpleNamespace(name=NAMESPACED)])
        assert [t["name"] for t in oa._introspect_agent(agent)["tools"]] == ["lookup"]

        processor = oa.DecimalTracingProcessor(agent_name="sbx")
        trace = types.SimpleNamespace(trace_id="trace_ns", name="w")
        processor.on_trace_start(trace)
        span_data = types.SimpleNamespace(type="agent", name="sbx", output_type=None,
                                          tools=["lookup", NAMESPACED, "load_skill"], handoffs=[])
        processor.on_span_end(types.SimpleNamespace(
            trace_id="trace_ns", span_id="span_1", parent_id=None, span_data=span_data,
            started_at=None, ended_at=None, error=None,
        ))
        assert list(processor._traces["trace_ns"].seen_tools) == ["lookup"]


# ── 2. instrument() is idempotent ────────────────────────────────────────────


class TestInstrumentIsIdempotent:
    def test_init_flag_then_skill_loader_leaves_one_processor(self, sdk):
        """``init(openai_agents=True)`` runs ``instrument(agent_name=...)``; the
        documented way to add skills afterwards is a second ``instrument``."""
        other = MagicMock(name="users-own-processor")
        set_trace_processors([other])

        oa.instrument(agent_name="support", disk_sync=False)        # what init() does
        oa.instrument(enable_skill_loader=True, disk_sync=False)

        assert len(_ours()) == 1
        assert other in get_trace_provider()._multi_processor._processors
        assert oa._skill_loader_installed is True
        # A second call that names no agent keeps the name the first one set.
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

    def test_one_run_sends_one_trace_after_two_instrument_calls(self, sdk):
        router = _Router()
        oa._skill_router_singleton = router
        oa.instrument(agent_name="support", disk_sync=False)
        oa.instrument(enable_skill_loader=True, disk_sync=False)

        Runner.run_sync(agents.Agent(name="support", instructions="You are helpful.",
                                     model=_ScriptedModel([_text("ok")])), QUESTION)

        traces = _sent(sdk)
        assert len(traces) == 1
        assert traces[0].routing_id == ROUTING_ID


# ── 3. the routing call is made off the event loop ───────────────────────────


async def _prompt_while_the_loop_is_needed(agent, router):
    """Resolve ``agent``'s prompt while another coroutine must run to release
    the router. Returns (prompt, loop thread id)."""
    async def release():
        await asyncio.sleep(0)
        router.wait.set()

    ctx = agents.RunContextWrapper(context=None, turn_input=[{"role": "user", "content": QUESTION}])
    prompt, _ = await asyncio.gather(agent.get_system_prompt(ctx), release())
    return prompt, threading.get_ident()


class TestRoutingDoesNotBlockTheEventLoop:
    @pytest.mark.parametrize("built", ["before_instrument", "after_instrument"])
    def test_get_system_prompt_routes_on_a_worker_thread(self, sdk, built):
        router = _Router(wait=threading.Event())
        oa._skill_router_singleton = router
        agent = agents.Agent(name="a", instructions="be terse") if built == "before_instrument" else None
        oa._install_skill_loader()
        agent = agent or agents.Agent(name="a", instructions="be terse")

        prompt, loop_thread = asyncio.run(_prompt_while_the_loop_is_needed(agent, router))

        assert MENU in prompt and prompt.startswith("be terse")
        assert router.routes[0]["thread"] != loop_thread
        assert router.released_while_routing is True

    def test_a_sandbox_run_routes_on_a_worker_thread(self, sdk):
        router = _Router()
        oa._skill_router_singleton = router
        oa.instrument(enable_skill_loader=True, disk_sync=False, exclusive=True)
        loop_threads = []

        class _Recording(_ScriptedModel):
            async def get_response(self, *args, **kwargs):
                loop_threads.append(threading.get_ident())
                return await super().get_response(*args, **kwargs)

        _sandbox_run(SandboxAgent(name="sbx", instructions="x", model=_Recording([_text("ok")])))
        assert router.routes and router.routes[0]["thread"] not in loop_threads

    def test_the_callers_context_sees_what_an_inline_call_left(self, sdk):
        """Off the loop, but not off the books: the context-variable rails the
        routing call writes still land in the awaiting coroutine's context."""
        oa._skill_router_singleton = _Router()
        oa._install_skill_loader()
        agent = agents.Agent(name="a", instructions="be terse")

        async def resolve():
            await agent.get_system_prompt(agents.RunContextWrapper(context=None))
            return oa._consume_routing_id(), oa._consume_skills_offered()

        assert asyncio.run(resolve()) == (ROUTING_ID, [SKILL])

    def test_called_without_a_running_loop_it_still_returns_a_string(self, sdk):
        oa._skill_router_singleton = _Router()
        oa._install_skill_loader()
        agent = agents.Agent(name="a", instructions="be terse")
        out = agent.instructions(agents.RunContextWrapper(context=None), agent)
        assert isinstance(out, str) and MENU in out
