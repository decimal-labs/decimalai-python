"""A run that names itself to the router gives that slot back.

``SkillRouter.build_prompt_fragment(scope=...)`` files every routing decision a
second time under the run's scope, in ``_scoped_routing_rails``, for an adapter
that drains by run — LangChain, whose per-call ContextVars never reach its
trace-send path. ``load_skill(scope=...)`` does the same in
``_scoped_loaded_names``. Both stores hold ``_MAX_SCOPED_RAILS`` (4096) runs and
WARN on every eviction, because for LangChain an eviction drops the routing
decision or the activation of a run that is still in flight.

So an adapter that passes a scope and never drains it holds one slot per run for
the life of the process. From run 4,097 on, every new run evicts the oldest and
logs "skill rail overflow", and a LangChain run in flight in the same process
loses its slot to that churn.

LangChain (``_drain_scoped_router_rails``, ``_discard_scoped_router_rails``) and
CrewAI release what they file. ADK and Anthropic read the per-call ContextVars
instead and never released theirs — found reading the code on 2026-10-08, beside
commit 02704a3. Reproducing it found the same leak in openai_agents and
pydantic_ai, and in openai_agents' loaded rail too. All five now release through
``skill_router._release_scoped_routing_rail``.

Every test here runs ``_MAX_SCOPED_RAILS + 4`` runs through the adapter's own
entry point — ADK's plugin callbacks, Anthropic's patched ``messages.create``,
the openai_agents instructions callable and trace processor, the pydantic_ai
system-prompt hook, CrewAI's ``before_llm_call`` hook — against a REAL
``SkillRouter`` whose two network calls are stubbed. The adapters' delivery
tests use a stand-in router, which has no slots to leak; that is how this went
unseen. The frameworks themselves are faked: each adapter is reached exactly the
way the framework reaches it, and nothing else.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import types
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

from decimalai.skill_router import (
    SkillRouter,
    consume_last_delivered_names,
    consume_last_offered_names,
)

pytest.importorskip("opentelemetry.sdk.trace")

#: Four runs past the cap: before the fix, every one of them evicts another
#: run's slot and warns, which is what "warns on every new run" looks like.
RUNS = SkillRouter._MAX_SCOPED_RAILS + 4

ROUTING_ID = "rt_" + "c" * 24
# A menu row could never contain this; only a delivered BODY can.
SENTINEL = "SENTINEL-SKILLBODY-SCOPE-RELEASE-41c7"
QUESTION = "What fee applies to an opened box return?"
CALLER_PROMPT = "You are a terse support agent."
#: A run of ANOTHER adapter, still in flight while the flood goes through. Its
#: routing decision has to be there to drain when its trace is finally built.
IN_FLIGHT = "langchain-run-still-in-flight"


# ── plumbing ─────────────────────────────────────────────────────────────────


@pytest.fixture
def router(monkeypatch) -> SkillRouter:
    """A real router — the slots under test are its own — off the network."""
    router = SkillRouter(
        api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=True,
    )
    result = {
        "prompt_fragment": "Available skills:\n- refund-policy: how refunds work",
        "routing_id": ROUTING_ID,
        "skills": [{"name": "refund-policy"}],
        "strategy": "semantic",
    }
    monkeypatch.setattr(router, "smart_route", lambda *a, **k: dict(result))
    monkeypatch.setattr(
        router, "get_skill_body",
        lambda name, **k: f"Opened boxes carry a 23.5% restocking fee. {SENTINEL}",
    )
    # Filed BEFORE the flood and drained after it — the way a LangChain run's
    # decision waits on the router until its trace is built.
    router.build_prompt_parts(query=QUESTION, agent_name="support", scope=IN_FLIGHT)
    consume_last_offered_names()  # that call's per-call rails are not under test
    consume_last_delivered_names()
    return router


@pytest.fixture
def overflow_warnings(caplog):
    caplog.set_level(logging.WARNING, logger="decimalai")

    def _read() -> List[str]:
        return [
            r.getMessage() for r in caplog.records
            if "skill rail overflow" in r.getMessage()
        ]

    return _read


class _Client:
    """The ingest client, keeping the count and the last trace — 4,100 traces
    held by a MagicMock's call list would be memory spent on nothing."""

    def __init__(self) -> None:
        self.traces = 0
        self.last: Any = None

    def ingest_trace(self, trace: Any) -> None:
        self.traces += 1
        self.last = trace

    def register_manifest(self, snapshot: Any) -> dict:
        return {"manifest_id": "test-manifest-id", "status": "active"}


@pytest.fixture
def client(monkeypatch) -> _Client:
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    client = _Client()
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    ))
    monkeypatch.setattr(cfg, "_client", client)
    return client


@pytest.fixture
def pipeline(client):
    """A caller-owned TracerProvider wired exactly as ``_ensure_pipeline`` wires
    one, so each run's root span becomes a trace and pops its OTel rail — the
    way it does in production."""
    from opentelemetry.sdk.trace import TracerProvider

    from decimalai import providers
    from decimalai.otel import _reset_skill_rails

    saved = (providers._pipeline_provider, providers._last_provider)
    _reset_skill_rails()
    provider = TracerProvider()
    providers._ensure_pipeline("support", provider)
    yield provider
    _reset_skill_rails()
    providers._pipeline_provider, providers._last_provider = saved


def _assert_every_slot_released(router: SkillRouter, overflow: List[str]) -> None:
    assert not overflow, (
        f"{len(overflow)} runs evicted an older run's slot (the {RUNS - router._MAX_SCOPED_RAILS} "
        f"runs past the cap, and the run in flight). First: {overflow[0]}"
    )
    leaked = [s for s in router._scoped_routing_rails if s != IN_FLIGHT]
    assert not leaked, (
        f"{len(leaked)} runs still hold a routing slot after finishing"
    )
    assert not router._scoped_loaded_names, (
        f"{len(router._scoped_loaded_names)} runs still hold a loaded-skills slot"
    )
    # The precise half: each run gave back ITS slot, not everybody's.
    assert router.consume_routing_id(scope=IN_FLIGHT) == ROUTING_ID, (
        "the run in flight lost its routing decision"
    )


# ── Google ADK ───────────────────────────────────────────────────────────────


def _real_adk_installed() -> bool:
    """Asked at import time, as tests/test_adk_skill_delivery.py does, so a stub
    left in sys.modules by a fixture is never mistaken for the real thing."""
    try:
        return importlib.util.find_spec("google.adk.plugins.base_plugin") is not None
    except (ImportError, ValueError):
        return False


HAS_REAL_ADK = _real_adk_installed()


class _LlmRequest:
    """``google.adk.models.llm_request.LlmRequest`` on the members the adapter
    touches — the stand-in tests/test_adk_skill_delivery.py pins to the real
    class wherever google-adk is installed."""

    def __init__(self) -> None:
        self.model = "gemini-3.6-flash"
        self.config = SimpleNamespace(system_instruction=CALLER_PROMPT)
        self.contents = [SimpleNamespace(role="user", parts=[SimpleNamespace(text=QUESTION)])]

    def append_instructions(self, instructions: List[str]) -> None:
        self.config.system_instruction += "\n\n" + "\n\n".join(instructions)


@pytest.fixture
def adk_plugin(monkeypatch, router, client):
    import decimalai.adk as adk

    monkeypatch.setattr(adk, "_skill_router_singleton", router)
    monkeypatch.setattr(adk, "_manifest_ids", {})
    monkeypatch.setattr(adk, "_manifest_trackers", {})
    monkeypatch.setattr(adk, "_pending_manifests", {})
    if not HAS_REAL_ADK:
        # Stub the one import `_plugin_class()` makes; the real BasePlugin only
        # stores its name.
        monkeypatch.setattr(adk, "_PluginClass", None)
        base_plugin = types.ModuleType("google.adk.plugins.base_plugin")

        class BasePlugin:
            def __init__(self, name):
                self.name = name

        base_plugin.BasePlugin = BasePlugin
        for name in ("google", "google.adk", "google.adk.plugins"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, "google.adk.plugins.base_plugin", base_plugin)
    return adk.DecimalaiPlugin(agent_name="support", enable_skill_loader=True)


class TestAdk:
    def test_runs_past_the_cap_hold_no_slot(self, router, adk_plugin, client, overflow_warnings):
        agent = SimpleNamespace(
            name="root_agent", model="gemini-3.6-flash", instruction=CALLER_PROMPT,
            tools=[], sub_agents=[],
        )
        last_request: List[Any] = []

        async def _runs() -> None:
            # One event loop for every run; ADK runs many invocations on one.
            for i in range(RUNS):
                ic = SimpleNamespace(agent=agent, invocation_id=f"inv-{i}", user_content=QUESTION)
                cc = SimpleNamespace(invocation_id=f"inv-{i}", agent_name=None)
                request = _LlmRequest()
                await adk_plugin.before_run_callback(invocation_context=ic)
                await adk_plugin.before_agent_callback(agent=agent, callback_context=cc)
                await adk_plugin.before_model_callback(callback_context=cc, llm_request=request)
                await adk_plugin.after_agent_callback(agent=agent, callback_context=cc)
                last_request[:] = [request]

        asyncio.run(_runs())

        # Releasing must not cost the delivery itself.
        assert SENTINEL in last_request[0].config.system_instruction
        assert client.traces == RUNS
        assert client.last.routing_id == ROUTING_ID
        assert client.last.skills_delivered == ["refund-policy"]
        _assert_every_slot_released(router, overflow_warnings())


# ── Anthropic ────────────────────────────────────────────────────────────────


class _Messages:
    """``anthropic.resources.messages.Messages``: ``create`` hands back the
    request it would have sent."""

    def create(self, *args: Any, **kwargs: Any) -> dict:
        return kwargs


class _AsyncMessages:
    async def create(self, *args: Any, **kwargs: Any) -> dict:
        return kwargs


@pytest.fixture
def anthropic_messages(monkeypatch, router):
    """The documented one-liner, ``instrument(enable_skill_loader=True)``,
    installed onto a fake ``Messages`` class instead of the SDK's."""
    import decimalai.anthropic as anthropic_adapter

    module = types.ModuleType("anthropic.resources.messages")
    module.Messages = _Messages
    module.AsyncMessages = _AsyncMessages
    monkeypatch.setitem(sys.modules, "anthropic.resources.messages", module)
    monkeypatch.setattr(_Messages, "create", _Messages.create)  # undone after
    monkeypatch.setattr(_AsyncMessages, "create", _AsyncMessages.create)
    monkeypatch.setattr(anthropic_adapter, "_skill_loader_installed", False)
    monkeypatch.setattr(anthropic_adapter, "_install_agent_name", None)
    monkeypatch.setattr(anthropic_adapter, "_skill_router_singleton", router)
    anthropic_adapter.instrument(enable_skill_loader=True, agent_name="support")
    return _Messages()


class TestAnthropic:
    def test_runs_past_the_cap_hold_no_slot(
        self, router, anthropic_messages, pipeline, client, overflow_warnings,
    ):
        from decimalai.providers import agent_run

        tracer = pipeline.get_tracer("fake-anthropic-instrumentor")
        sent: Optional[dict] = None
        for _ in range(RUNS):
            with agent_run("support", tracer_provider=pipeline):
                sent = anthropic_messages.create(
                    model="claude-sonnet-5-5",
                    system=CALLER_PROMPT,
                    messages=[{"role": "user", "content": QUESTION}],
                )
                # The span the OpenInference instrumentor emits for the call.
                with tracer.start_as_current_span("Messages", attributes={
                    "gen_ai.system": "anthropic", "gen_ai.request.model": "claude-sonnet-5-5",
                }):
                    pass

        assert sent is not None and SENTINEL in sent["system"]
        assert sent["system"].startswith(CALLER_PROMPT)
        assert client.traces == RUNS
        assert client.last.routing_id == ROUTING_ID
        assert client.last.skills_delivered == ["refund-policy"]
        _assert_every_slot_released(router, overflow_warnings())


# ── OpenAI Agents ────────────────────────────────────────────────────────────


class TestOpenAIAgents:
    def test_runs_past_the_cap_hold_no_slot(
        self, monkeypatch, router, client, overflow_warnings,
    ):
        """Both rails: the routing decision the instructions callable files, and
        the load the ``load_skill`` tool files, each under the Agents-SDK trace id."""
        import decimalai.openai_agents as oa

        current: dict = {"trace_id": None}
        monkeypatch.setattr(oa, "_skill_router_singleton", router)
        monkeypatch.setattr(oa, "_manifest_id", None)
        # The Agents SDK's own `get_current_trace()`, faked: the run key is
        # all the adapter reads off it.
        monkeypatch.setattr(oa, "_current_run_key", lambda: current["trace_id"])

        processor = oa.DecimalTracingProcessor(agent_name="support")
        instructions = oa._make_skill_aware_instructions(CALLER_PROMPT)
        agent = SimpleNamespace(name="support", tools=[])
        ctx = SimpleNamespace(turn_input=[{"role": "user", "content": QUESTION}])
        prompt = loaded = ""
        for i in range(RUNS):
            trace = MagicMock(trace_id=f"trace_{i:032x}")
            trace.name = "Agent workflow"
            current["trace_id"] = trace.trace_id
            processor.on_trace_start(trace)
            prompt = instructions(ctx, agent)
            loaded = oa._handle_load_skill("refund-policy")
            processor.on_trace_end(trace)
            current["trace_id"] = None

        assert SENTINEL in prompt and prompt.startswith(CALLER_PROMPT)
        assert loaded.startswith("## Skill: refund-policy")
        assert client.traces == RUNS
        assert client.last.routing_id == ROUTING_ID
        assert client.last.skills_loaded_by_agent == ["refund-policy"]
        _assert_every_slot_released(router, overflow_warnings())


# ── Pydantic AI ──────────────────────────────────────────────────────────────


class TestPydanticAI:
    def test_runs_past_the_cap_hold_no_slot(
        self, monkeypatch, router, pipeline, client, overflow_warnings,
    ):
        import decimalai.pydantic_ai as pa
        from decimalai.providers import agent_run

        monkeypatch.setattr(pa, "_skill_router_singleton", router)
        tracer = pipeline.get_tracer("fake-model")
        ctx = SimpleNamespace(prompt=QUESTION, messages=[], agent=SimpleNamespace(name="support"))
        prompts: List[str] = []

        async def _runs() -> None:
            for _ in range(RUNS):
                # `instrument()` opens this span around every agent run.
                with agent_run("support", tracer_provider=pipeline):
                    prompts[:] = [await pa._skills_system_prompt(ctx)]
                    pa._handle_load_skill("refund-policy")
                    with tracer.start_as_current_span("chat", attributes={
                        "gen_ai.system": "openai", "gen_ai.request.model": "stub-model-1",
                    }):
                        pass

        asyncio.run(_runs())

        assert SENTINEL in prompts[0]
        assert client.traces == RUNS
        assert client.last.routing_id == ROUTING_ID
        assert client.last.skills_loaded_by_agent == ["refund-policy"]
        _assert_every_slot_released(router, overflow_warnings())


# ── CrewAI ───────────────────────────────────────────────────────────────────


class TestCrewAI:
    """Released before this file existed; graded here because it now releases
    through the helper the four above share."""

    def test_runs_past_the_cap_hold_no_slot(
        self, monkeypatch, router, pipeline, client, overflow_warnings,
    ):
        import decimalai.crewai as dc
        from decimalai.providers import agent_run

        monkeypatch.setattr(dc, "_skill_router_singleton", router)
        tracer = pipeline.get_tracer("fake-crewai-instrumentor")
        context: Any = None
        for _ in range(RUNS):
            # `agent_run` stands in for the crew span OpenInference opens.
            with agent_run("support", tracer_provider=pipeline):
                # `crewai.hooks.LLMCallHookContext` on the members the hook reads.
                context = SimpleNamespace(
                    executor="executor",
                    messages=[
                        {"role": "system", "content": CALLER_PROMPT},
                        {"role": "user", "content": QUESTION},
                    ],
                    agent=SimpleNamespace(role="Support"),
                    task=SimpleNamespace(description=QUESTION),
                    crew=None, llm=None, iterations=0,
                )
                dc._skill_hook(context)
                with tracer.start_as_current_span("chat", attributes={
                    "gen_ai.system": "openai", "gen_ai.request.model": "stub-model-1",
                }):
                    pass

        assert any(SENTINEL in str(m.get("content")) for m in context.messages)
        assert client.traces == RUNS
        assert client.last.routing_id == ROUTING_ID
        assert client.last.skills_delivered == ["refund-policy"]
        _assert_every_slot_released(router, overflow_warnings())
