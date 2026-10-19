"""``init()``'s agent name and API key are the default wherever one is taken.

The Phase 4 audit (2026-10-18): "the agent name must match in five or more
places; a mismatch silently files traces under another agent and drops that
agent's skills". ``init(agent_name=...)`` was stored nowhere — only the adapters
whose flag was set in the same call ever saw it — so a later ``instrument()``,
``start_trace()``, ``@trace`` or ``SkillRouter`` each needed the name again, and
``SkillRouter`` needed the key again too. Design D6 makes ``init()`` (else
``DECIMAL_AGENT_NAME``) the default everywhere, with an explicit argument still
winning.

On OpenAI Agents the DecimalAI name now also wins over ``Agent(name=...)`` for
the trace AND for routing: it is the agent the skills are attached to, and the
two halves must keep naming the same agent (conformance C18).

Every case restores the global config and every adapter global it writes.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
from collections import OrderedDict
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

import decimalai
import decimalai._config as cfg
from decimalai._config import DecimalConfig, DecimalConfigError, default_agent_name

KEY = "dai_sk_init_key"
NAME = "support"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    """No ambient name or key, and the global config/client put back after."""
    for var in ("DECIMAL_AGENT_NAME", "DECIMAL_API_KEY", "DECIMALAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")
    monkeypatch.setattr(cfg, "_config", None)
    monkeypatch.setattr(cfg, "_client", None)
    monkeypatch.setattr(decimalai, "_global_client", None, raising=False)


@pytest.fixture
def configured(monkeypatch):
    """``configured(name)``: the state ``init(agent_name=name)`` leaves, minus a real client."""

    def _configure(agent_name: Optional[str] = NAME) -> MagicMock:
        monkeypatch.setattr(cfg, "_config", DecimalConfig(
            api_key=KEY, base_url="http://127.0.0.1:9", enabled=True, agent_name=agent_name,
        ))
        client = MagicMock()
        client.register_manifest.return_value = {"manifest_id": "m-1", "status": "active"}
        monkeypatch.setattr(cfg, "_client", client)
        return client

    return _configure


def _init(**kwargs: Any) -> None:
    decimalai.init(api_key=KEY, base_url="http://127.0.0.1:9", verify=False, **kwargs)


# ── the config itself ─────────────────────────────────────────────────────────


class TestInitStoresTheName:
    def test_init_keeps_the_name(self):
        _init(agent_name=NAME)
        assert cfg._config.agent_name == NAME
        assert default_agent_name() == NAME

    def test_the_env_var_is_the_default(self, monkeypatch):
        monkeypatch.setenv("DECIMAL_AGENT_NAME", "from-env")
        _init()
        assert default_agent_name() == "from-env"

    def test_the_argument_beats_the_env_var(self, monkeypatch):
        monkeypatch.setenv("DECIMAL_AGENT_NAME", "from-env")
        _init(agent_name=NAME)
        assert default_agent_name() == NAME

    def test_a_blank_env_var_names_nothing(self, monkeypatch):
        monkeypatch.setenv("DECIMAL_AGENT_NAME", "   ")
        _init()
        assert default_agent_name() is None

    def test_before_init_the_env_var_is_read_directly(self, monkeypatch):
        monkeypatch.setenv("DECIMAL_AGENT_NAME", "from-env")
        assert cfg._config is None
        assert default_agent_name() == "from-env"

    def test_an_explicit_name_always_wins(self, configured):
        configured(NAME)
        assert default_agent_name("other") == "other"

    def test_the_flags_install_under_the_env_name(self, monkeypatch):
        """``DECIMAL_AUTO_TRACE`` calls the flags with no ``agent_name``; the env
        var is the only way to name that agent."""
        import decimalai.langchain as lc

        seen = []
        monkeypatch.setattr(lc, "instrument", lambda **kw: seen.append(kw))
        monkeypatch.setenv("DECIMAL_AGENT_NAME", "from-env")
        _init(langchain=True)
        assert seen and seen[0]["agent_name"] == "from-env"


# ── start_trace and @trace ────────────────────────────────────────────────────


class TestGenericTracing:
    def test_start_trace_uses_the_init_name(self, configured):
        configured(NAME)
        with decimalai.start_trace(auto_send=False) as t:
            assert t.agent_name == NAME

    def test_start_trace_explicit_name_wins(self, configured):
        configured(NAME)
        with decimalai.start_trace(agent_name="other", auto_send=False) as t:
            assert t.agent_name == "other"

    def test_start_trace_reads_the_env_var(self, configured, monkeypatch):
        monkeypatch.setenv("DECIMAL_AGENT_NAME", "from-env")
        configured(None)  # what init() leaves when the env var was set after it
        with decimalai.start_trace(auto_send=False) as t:
            assert t.agent_name is None
        monkeypatch.setattr(cfg, "_config", DecimalConfig(api_key=KEY))
        with decimalai.start_trace(auto_send=False) as t:
            assert t.agent_name == "from-env"

    def test_the_decorator_resolves_at_call_time(self, configured):
        """Decorated at import, before ``init()`` — the usual order."""
        from decimalai.generic import _get_current_trace

        @decimalai.trace(auto_send=False)
        def answer() -> Optional[str]:
            return _get_current_trace().agent_name

        @decimalai.trace(agent_name="explicit", auto_send=False)
        def named() -> Optional[str]:
            return _get_current_trace().agent_name

        configured(None)
        assert answer() == "answer"
        configured(NAME)
        assert answer() == NAME
        assert named() == "explicit"


# ── SkillRouter ───────────────────────────────────────────────────────────────


class TestSkillRouterDefaults:
    def test_key_and_agent_come_from_init(self, configured):
        from decimalai.skill_router import SkillRouter

        configured(NAME)
        router = SkillRouter()
        assert router.api_key == KEY
        assert router.agent_name == NAME

    def test_explicit_arguments_win_positional_included(self, configured):
        from decimalai.skill_router import SkillRouter

        configured(NAME)
        router = SkillRouter("dai_sk_other", "http://example.test", "other")
        assert (router.api_key, router.base_url, router.agent_name) == (
            "dai_sk_other", "http://example.test", "other",
        )

    def test_an_explicit_none_still_routes_unscoped(self, configured):
        """The adapters build theirs this way: they name the agent per call."""
        from decimalai.skill_router import SkillRouter

        configured(NAME)
        assert SkillRouter(agent_name=None).agent_name is None

    @pytest.mark.parametrize("var", ["DECIMAL_API_KEY", "DECIMALAI_API_KEY"])
    def test_without_init_the_env_vars_are_read(self, monkeypatch, var):
        from decimalai.skill_router import SkillRouter

        monkeypatch.setenv(var, "dai_sk_env")
        monkeypatch.setenv("DECIMAL_AGENT_NAME", "from-env")
        router = SkillRouter()
        assert router.api_key == "dai_sk_env"
        assert router.agent_name == "from-env"

    def test_init_key_beats_the_env_var(self, configured, monkeypatch):
        from decimalai.skill_router import SkillRouter

        monkeypatch.setenv("DECIMAL_API_KEY", "dai_sk_env")
        configured(NAME)
        assert SkillRouter().api_key == KEY

    def test_init_key_goes_to_the_host_init_was_given(self, configured, monkeypatch):
        from decimalai.skill_router import SkillRouter

        monkeypatch.setenv("DECIMAL_BASE_URL", "http://elsewhere.test")
        configured(NAME)
        assert SkillRouter().base_url == "http://127.0.0.1:9"
        # A key passed explicitly keeps the resolution it always had.
        assert SkillRouter(api_key="dai_sk_other").base_url == "http://elsewhere.test"

    def test_no_key_anywhere_is_a_clear_error(self):
        from decimalai.skill_router import SkillRouter

        with pytest.raises(DecimalConfigError, match="DECIMAL_API_KEY"):
            SkillRouter()

    @pytest.mark.parametrize("module", [
        "langchain", "openai_agents", "adk", "crewai", "anthropic", "pydantic_ai",
    ])
    def test_adapter_routers_carry_no_default_agent(self, configured, monkeypatch, module):
        """An adapter names the agent on every routing call — the one its trace
        is filed under. A router default would route a run it could not name as
        an agent the trace never mentions."""
        import importlib

        mod = importlib.import_module(f"decimalai.{module}")
        configured(NAME)
        monkeypatch.setattr(mod, "_skill_router_singleton", None)
        router = mod._get_skill_router()
        assert router is not None
        assert router.agent_name is None


# ── every adapter's instrument() ──────────────────────────────────────────────


class TestInstrumentDefaults:
    def _langchain(self, monkeypatch):
        import decimalai.langchain as lc

        for attr in (
            "_installed", "_install_agent_name", "_skill_body_top_k", "_priority_skills",
            "_runtime_policy", "_evals", "_builtin_evals_enabled", "_explicit_manifest_config",
        ):
            monkeypatch.setattr(lc, attr, getattr(lc, attr))
        monkeypatch.setattr(lc._global_handler, "on_trace", lc._global_handler.on_trace)
        monkeypatch.setattr(
            lc._global_handler, "_explicit_agent_name", lc._global_handler._explicit_agent_name,
        )
        token = lc._decimal_callback_var.set(None)
        monkeypatch.setattr(lc, "_installed", False)
        return lc, token

    def test_langchain(self, configured, monkeypatch):
        pytest.importorskip("langchain_core")
        lc, token = self._langchain(monkeypatch)
        try:
            configured(NAME)
            lc.instrument(disk_sync=False)
            assert lc._install_agent_name == NAME
            assert lc._global_handler.agent_name == NAME
        finally:
            lc._decimal_callback_var.reset(token)

    def test_langchain_explicit_wins(self, configured, monkeypatch):
        pytest.importorskip("langchain_core")
        lc, token = self._langchain(monkeypatch)
        try:
            configured(NAME)
            lc.instrument(agent_name="other", disk_sync=False)
            assert lc._install_agent_name == "other"
        finally:
            lc._decimal_callback_var.reset(token)

    def test_langchain_repeat_call_without_a_name_is_silent(self, configured, monkeypatch, caplog):
        """The repeat-call rule is untouched: a second call that names nothing
        keeps the installed name and has nothing to warn about."""
        pytest.importorskip("langchain_core")
        lc, token = self._langchain(monkeypatch)
        try:
            configured(NAME)
            lc.instrument(agent_name="first", disk_sync=False)
            with caplog.at_level(logging.WARNING, logger="decimalai"):
                lc.instrument(disk_sync=False)
            assert lc._install_agent_name == "first"
            assert "ignoring agent_name" not in caplog.text
        finally:
            lc._decimal_callback_var.reset(token)

    def test_adk_and_its_repeat_call(self, configured, monkeypatch):
        """Each ``adk.instrument()`` replaces the name, so a bare repeat used to
        put the install back on ADK's node name."""
        import decimalai.adk as adk

        for attr in ("_install_agent_name", "_skill_loader_enabled", "_instrument_config"):
            monkeypatch.setattr(adk, attr, getattr(adk, attr))
        monkeypatch.setattr(adk, "_runner_patched", True)  # configure only; patch nothing
        configured(NAME)
        adk.instrument()
        assert adk._install_agent_name == NAME
        assert adk._instrument_config.agent_name == NAME
        adk.instrument(agent_name="other")
        assert adk._install_agent_name == "other"

    def test_crewai(self, configured, monkeypatch):
        import decimalai.crewai as dc

        installed: List[Optional[str]] = []
        monkeypatch.setattr(dc, "_install_agent_name", None)
        monkeypatch.setattr(dc, "_instrument_config", dc._instrument_config)
        monkeypatch.setattr(dc, "_install_tracing", installed.append)
        configured(NAME)
        dc.instrument()
        assert dc._install_agent_name == NAME
        assert installed == [NAME]
        # An earlier explicit name is not replaced by init()'s.
        monkeypatch.setattr(dc, "_install_agent_name", "earlier")
        dc.instrument()
        assert dc._install_agent_name == "earlier"

    def test_anthropic(self, configured, monkeypatch):
        import decimalai.anthropic as da

        monkeypatch.setattr(da, "_install_agent_name", None)
        configured(NAME)
        da.instrument()
        assert da._install_agent_name == NAME
        assert da._routing_agent_name() == NAME
        da.instrument(agent_name="other")
        assert da._install_agent_name == "other"

    def test_pydantic_ai_fills_the_fallback_only(self, configured, monkeypatch):
        """Pydantic AI keeps its documented rule: ``Agent(name=...)`` wins, and
        the DecimalAI name is the fallback for an Agent built without one."""
        import decimalai.pydantic_ai as pa

        monkeypatch.setattr(pa, "_default_agent_name", None)
        configured(NAME)
        pa.instrument(trace_runs=False)
        assert pa._default_agent_name == NAME

        class _Agent:
            name = "agent"  # what `_infer_name` leaves on an unnamed Agent

        unnamed, named = _Agent(), _Agent()
        setattr(unnamed, pa._CONSTRUCTED_NAME_ATTR, None)
        setattr(named, pa._CONSTRUCTED_NAME_ATTR, "billing")
        assert pa._run_agent_name(unnamed) == NAME
        assert pa._run_agent_name(named) == "billing"

    def test_llamaindex(self, configured, monkeypatch):
        pytest.importorskip("llama_index.core")
        import llama_index.core.instrumentation as li_inst

        import decimalai.llamaindex as dl

        dispatcher = MagicMock()
        monkeypatch.setattr(li_inst, "get_dispatcher", lambda *a, **k: dispatcher)
        configured(NAME)
        assert dl.instrument().agent_name == NAME
        assert dl.instrument(agent_name="other").agent_name == "other"

    def test_claude_agent_sdk(self, configured, monkeypatch):
        import decimalai.claude_agent_sdk as cas

        monkeypatch.setattr(cas, "_install_agent_name", None)
        configured(NAME)
        # A stream wrapped with no instrument() at all still files under it.
        state = cas._RunState(
            agent_name=None, project=None, parent_trace_id=None, user_input=None, options=None,
        )
        assert state.agent_name == NAME
        monkeypatch.setattr(cas, "_query_patched", True)  # configure only; patch nothing
        cas.instrument()
        assert cas._install_agent_name == NAME

    def test_otel(self, configured, monkeypatch, tmp_path):
        from opentelemetry import trace as trace_api

        import decimalai.otel as dotel

        monkeypatch.chdir(tmp_path)  # no SKILL.md discovery from the repo
        monkeypatch.setattr(trace_api, "set_tracer_provider", lambda provider: None)
        monkeypatch.setattr(dotel, "_register_flush_atexit", lambda provider: None)
        monkeypatch.setattr(dotel, "_installed_providers", [])
        configured(NAME)
        run = contextvars.copy_context()
        provider = run.run(dotel.instrument)
        try:
            exporters = [
                getattr(p, "span_exporter", None)
                for p in provider._active_span_processor._span_processors
            ]
            [exporter] = [e for e in exporters if isinstance(e, dotel.DecimalSpanExporter)]
            assert exporter.default_agent_name == NAME
            assert run[dotel._active_agent_name] == NAME
        finally:
            provider.shutdown()

    def test_providers(self, configured):
        import decimalai.otel as dotel
        import decimalai.providers as dp

        configured(NAME)
        run = contextvars.copy_context()
        run.run(dp.instrument, openai=False, anthropic=False, google=False)
        assert run[dotel._active_agent_name] == NAME


# ── OpenAI Agents: the DecimalAI name wins over Agent(name=) ─────────────────


class _Span:
    def __init__(self, trace_id: str, data: Any):
        from datetime import datetime, timezone
        from uuid import uuid4

        self.trace_id, self.span_data = trace_id, data
        self.span_id, self.parent_id, self.error = str(uuid4()), None, None
        self.started_at = self.ended_at = datetime.now(timezone.utc).isoformat()


class _Data:
    def __init__(self, type_: str, **kw: Any):
        self.type = type_
        self.__dict__.update(kw)


class _Trace:
    def __init__(self, trace_id: str):
        self.trace_id, self.name = trace_id, "Agent workflow"


def _one_trace(processor: Any, client: MagicMock, *, agent_span: Optional[str]) -> Any:
    """Drive one trace through ``processor``: an agent span (if any) and a model call."""
    from uuid import uuid4

    from decimalai._config import _sender

    tid = f"trace_{uuid4().hex[:16]}"
    processor.on_trace_start(_Trace(tid))
    if agent_span is not None:
        processor.on_span_end(_Span(tid, _Data(
            "agent", name=agent_span, tools=[], handoffs=[], output_type="str",
        )))
    processor.on_span_end(_Span(tid, _Data(
        "generation", model="gpt-4o", input=[{"role": "user", "content": "hi"}],
        output=[{"role": "assistant", "content": "hello"}], model_config={},
        usage={"input_tokens": 1, "output_tokens": 1},
    )))
    processor.on_trace_end(_Trace(tid))
    _sender.flush()
    return client.ingest_trace.call_args[0][0]


class TestOpenAIAgentsTraceName:
    @pytest.fixture(autouse=True)
    def _adapter(self, monkeypatch):
        oa = pytest.importorskip("decimalai.openai_agents")
        pytest.importorskip("agents")
        monkeypatch.setattr(oa, "_manifest_id", None)
        monkeypatch.setattr(oa, "_tracing_processor", None)
        monkeypatch.setattr(oa, "_name_overrides_noted", set())
        return oa

    def test_the_decimalai_name_beats_the_agent_span(self, configured, caplog):
        from decimalai.openai_agents import DecimalTracingProcessor

        client = configured(None)
        with caplog.at_level(logging.INFO, logger="decimalai"):
            trace = _one_trace(
                DecimalTracingProcessor(agent_name=NAME), client, agent_span="Assistant",
            )
        assert trace.agent_name == NAME
        assert [c.agent_name for c in trace.llm_calls] == [NAME]
        assert "Agent(name='Assistant')" in caplog.text  # said once, at INFO

    def test_with_no_decimalai_name_the_agent_span_names_it(self, configured):
        from decimalai.openai_agents import DecimalTracingProcessor

        client = configured(None)
        trace = _one_trace(DecimalTracingProcessor(), client, agent_span="Assistant")
        assert trace.agent_name == "Assistant"

    def test_instrument_agent_is_a_fallback_not_a_decimalai_name(self, configured, monkeypatch):
        """``instrument(agent=a)`` names a trace only when no agent span does —
        as a DecimalAI name it would outrank every other Agent in the process."""
        from agents.tracing import get_trace_provider, set_trace_processors

        import decimalai.openai_agents as oa

        client = configured(None)
        monkeypatch.setattr(oa, "_register_manifest_from_agent", lambda *a, **k: None)
        saved = list(get_trace_provider()._multi_processor._processors)
        try:
            agent = MagicMock()
            agent.name = "triage"
            oa.instrument(agent=agent, exclusive=True, disk_sync=False)
            processor = oa._tracing_processor
            assert processor.default_agent_name is None
            assert processor.fallback_agent_name == "triage"
            assert _one_trace(processor, client, agent_span="refunds").agent_name == "refunds"
            assert _one_trace(processor, client, agent_span=None).agent_name == "triage"
        finally:
            set_trace_processors(saved)

    def test_a_name_given_with_an_agent_is_that_agents(self, configured, monkeypatch):
        """``instrument(agent=a, agent_name=X)`` once per agent — a process that
        serves several — names each Agent's runs, not the whole process's: as a
        process-wide name the last install would file every run (conformance C9).
        Routing is named by ``load_agent(name)`` alone."""
        from agents.tracing import get_trace_provider, set_trace_processors

        import decimalai.openai_agents as oa

        client = configured("process-wide")
        monkeypatch.setattr(oa, "_register_manifest_from_agent", lambda *a, **k: None)
        saved = list(get_trace_provider()._multi_processor._processors)

        def _agent(name: str) -> MagicMock:
            agent = MagicMock()
            agent.name = name
            return agent

        try:
            triage, refunds, other = _agent("triage"), _agent("refunds"), _agent("other")
            oa.instrument(agent=triage, agent_name="support", exclusive=True, disk_sync=False)
            oa.instrument(agent=refunds, agent_name="billing", exclusive=True, disk_sync=False)
            processor = oa._tracing_processor
            assert processor.default_agent_name == "process-wide"  # init()'s, untouched
            for agent, expected in ((triage, "support"), (refunds, "billing"),
                                    (other, "process-wide")):
                assert _one_trace(processor, client, agent_span=agent.name).agent_name == expected
        finally:
            set_trace_processors(saved)

    def test_instrument_defaults_and_keeps_the_name(self, configured):
        from agents.tracing import get_trace_provider, set_trace_processors

        import decimalai.openai_agents as oa

        configured(NAME)
        saved = list(get_trace_provider()._multi_processor._processors)
        try:
            oa.instrument(exclusive=True, disk_sync=False)
            assert oa._tracing_processor.default_agent_name == NAME
            oa.instrument(agent_name="other", exclusive=True, disk_sync=False)
            assert oa._tracing_processor.default_agent_name == "other"
            # A later call that names nothing keeps "other" — not init()'s name.
            oa.instrument(exclusive=True, disk_sync=False)
            assert oa._tracing_processor.default_agent_name == "other"
        finally:
            set_trace_processors(saved)


class TestOpenAIAgentsRunEndToEnd:
    """A real ``Runner.run_sync`` with a scripted model: the routing call, the
    ``load_skill`` call and the trace all name the ``load_agent()`` agent."""

    SKILL = "refund-policy"

    @pytest.fixture
    def run(self, configured, monkeypatch):
        agents = pytest.importorskip("agents")
        from agents import Runner
        from agents.items import ModelResponse
        from agents.models.interface import Model
        from agents.tracing import get_trace_provider, set_trace_processors
        from agents.usage import Usage
        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseOutputMessage,
            ResponseOutputText,
        )

        import decimalai.openai_agents as oa
        from decimalai import skill_router as sr
        from decimalai.schema.manifest import ManifestTracker

        for name, value in (
            ("_skill_router_singleton", None), ("_manifest_id", None),
            ("_manifest_tracker", ManifestTracker()), ("_two_catalogs_warned", False),
            ("_tracing_processor", None), ("_tracing_processor_added", False),
            ("_run_rails", OrderedDict()), ("_name_overrides_noted", set()),
            ("_config_pool", None),
        ):
            monkeypatch.setattr(oa, name, value)
        monkeypatch.setattr(oa, "_in_background", lambda fn, *args: fn(*args))
        saved = list(get_trace_provider()._multi_processor._processors)
        set_trace_processors([])
        sr._last_offered_names_ctx.set(None)
        skill = self.SKILL

        def text(value: str) -> Any:
            message = ResponseOutputMessage(
                id="msg_1", type="message", role="assistant", status="completed",
                content=[ResponseOutputText(type="output_text", text=value, annotations=[])],
            )
            return ModelResponse(output=[message], usage=Usage(), response_id=None)

        def load(name: str) -> Any:
            call = ResponseFunctionToolCall(
                id="fc_1", call_id="call_1", type="function_call",
                name="load_skill", arguments=f'{{"name": "{name}"}}',
            )
            return ModelResponse(output=[call], usage=Usage(), response_id=None)

        class Scripted(Model):
            def __init__(self, *, loads: bool, together: Optional[List[str]] = None) -> None:
                self.script = [load(skill), text("done")] if loads else [text("done")]
                self.together = together

            async def get_response(self, *args: Any, **kwargs: Any) -> Any:
                if self.together is not None and len(self.script) == 2:
                    # Hold the first answer until every run in the group has
                    # resolved its instructions: they all overlap.
                    self.together.append("here")
                    for _ in range(500):
                        if len(self.together) >= 2:
                            break
                        await asyncio.sleep(0.01)
                return self.script.pop(0) if self.script else text("done")

            def stream_response(self, *args: Any, **kwargs: Any):  # pragma: no cover
                raise NotImplementedError

        class Router:
            def __init__(self) -> None:
                self.routes: List[Optional[str]] = []
                self.loads: List[Optional[str]] = []
                self.lock = threading.Lock()

            def build_prompt_parts(self, query=None, *, agent_name=None, scope=None, **_):
                with self.lock:
                    self.routes.append(agent_name)
                sr._last_offered_names_ctx.set([skill])
                return (f"## Recommended Skills\n| {skill} | refunds |", "",
                        "rt_0123456789abcdef01234567")

            def load_skill(self, name, *, scope=None, agent_name=None):
                with self.lock:
                    self.loads.append(agent_name)
                return f"## Skill: {name}\n\nBODY"

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

        def _run(*, init_name: Optional[str], load_names: List[Optional[str]]) -> tuple:
            """One run per entry, concurrently: ``load_agent(name)``'s agent, or
            for None a plain agent with its own instructions. Every Agent is
            ``Agent(name="Assistant")``."""
            client = configured(init_name)
            client.get_agent_prompt.side_effect = lambda name, version=None: {
                "agent_name": name, "system_prompt": "Be helpful.", "version_number": 1,
            }
            router = Router()
            oa._skill_router_singleton = router
            oa.instrument(disk_sync=False, exclusive=True)
            together: Optional[List[str]] = [] if len(load_names) > 1 else None
            runs = []
            for load_name in load_names:
                if load_name:
                    support = oa.load_agent(load_name)
                    runs.append(agents.Agent(name="Assistant", instructions=support.instructions,
                                             tools=[*support.tools],
                                             model=Scripted(loads=True, together=together)))
                else:
                    runs.append(agents.Agent(name="Assistant", instructions="Be helpful.",
                                             model=Scripted(loads=False, together=together)))

            async def all_runs():
                await asyncio.gather(*(Runner.run(agent, "how do refunds work?") for agent in runs))

            asyncio.run(all_runs())
            from decimalai._config import _sender

            _sender.flush()
            traces = [c.args[0] for c in client.ingest_trace.call_args_list]
            return router, traces

        try:
            yield _run
        finally:
            set_trace_processors(saved)

    @pytest.mark.parametrize("init_name", [None, NAME, "other"])
    def test_the_load_agent_name_wins_everywhere(self, run, init_name):
        router, [trace] = run(init_name=init_name, load_names=[NAME])
        assert router.routes and set(router.routes) == {NAME}
        assert router.loads == [NAME]
        assert trace.agent_name == NAME

    def test_two_agents_with_one_openai_name_each_keep_their_own(self, run):
        """Both Agents are ``name="Assistant"``, and their runs overlap. Each
        trace is filed under the config that served it, not under whichever
        config's run resolved its instructions last."""
        router, traces = run(init_name=None, load_names=["support", "billing"])
        assert sorted(router.loads) == ["billing", "support"]
        assert sorted(trace.agent_name for trace in traces) == ["billing", "support"]

    def test_with_no_decimalai_name_the_agent_names_its_trace(self, run):
        """No ``load_agent()`` and no ``init(agent_name=...)``: ``Agent(name=...)``
        is the only name, and nothing is routed."""
        router, [trace] = run(init_name=None, load_names=[None])
        assert router.routes == [] and router.loads == []
        assert trace.agent_name == "Assistant"
