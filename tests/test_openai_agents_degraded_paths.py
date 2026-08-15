"""Regressions for the four ways the OpenAI Agents adapter lost data.

Each class here pins one mechanism the execution audit found, and each
assertion is the thing that was measurably wrong against a live backend
before the fix:

1. MANIFEST — a run that never completes a model call had nothing to
   register, so `_maybe_register_manifest` returned early, the trace went
   out with `manifest_id=None`, and ingest answered 400. Two of two such
   runs were lost.
2. CONCURRENCY — the routing rails were instance state on a process-global
   router, so of two parallel `Runner.run` calls one came back with
   `routing_id=None` and the other with an id minted by a *different
   agent's earlier run*.
3. CONSTRUCTOR — the skill loader patched `Agent.__init__`, so an Agent
   built before `instrument()` (i.e. any module-level agent) got 0 skills
   offered and no `load_skill` tool.
4. EXTRACTION — `_extract_query` probed `ctx.input` / `user_input` /
   `query`, none of which `RunContextWrapper` defines; the turn is on
   `ctx.turn_input`. Routing logged `strategy: full_menu`, 30 skills
   offered, an empty query.

Plus the two smaller ones in the same file: Responses-API input items
flattened to `{"role": "user", "content": ""}`, and a `_manifest_id`
global shared by every agent in the process.
"""

from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest


# ── Mocks for the SDK objects the processor is handed ───────


class _SpanData:
    def __init__(self, span_type: str, **kwargs):
        self._type = span_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def type(self) -> str:
        return self._type


class _Span:
    def __init__(self, trace_id, span_data, span_id=None, parent_id=None, error=None):
        self.trace_id = trace_id
        self.span_id = span_id or str(uuid4())
        self.parent_id = parent_id
        self.span_data = span_data
        now = datetime.now(timezone.utc).isoformat()
        self.started_at = now
        self.ended_at = now
        self.error = error


class _Trace:
    def __init__(self, trace_id, name="test-workflow"):
        self.trace_id = trace_id
        self.name = name


class _FakeAgent:
    """Stands in for `agents.Agent` — only the attributes we introspect."""

    def __init__(self, name, instructions=None, model=None, tools=(), handoffs=()):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = list(tools)
        self.handoffs = list(handoffs)


@pytest.fixture(autouse=True)
def _reset_sdk(monkeypatch):
    """Fresh config + a clean slate for every module-global this file touches."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig
    import decimalai.openai_agents as oa

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    client = MagicMock()
    client.register_manifest.side_effect = lambda snap: {
        "manifest_id": f"mf-{snap.agent_name}", "status": "active",
    }
    client.list_manifests.return_value = {"manifests": []}
    cfg._client = client

    monkeypatch.setattr(oa, "_manifest_id", None)
    monkeypatch.setattr(oa, "_manifest_ids", {})
    monkeypatch.setattr(oa, "_manifest_hashes", {})
    monkeypatch.setattr(oa, "_declared", {})
    monkeypatch.setattr(oa, "_run_rails", oa.OrderedDict())
    yield client


def _ingested(client):
    """Every RunTrace handed to ingest_trace, oldest first."""
    return [c.args[0] for c in client.ingest_trace.call_args_list]


# ── 4. Query extraction ─────────────────────────────────────


class TestExtractQueryFromTurnInput:
    def test_reads_turn_input_the_runner_actually_stamps(self):
        from decimalai.openai_agents import _extract_query

        ctx = types.SimpleNamespace(
            context=None,
            usage=None,
            turn_input=[{"role": "user", "content": "Review this SQL query"}],
        )
        assert _extract_query(ctx) == "Review this SQL query"

    def test_pre_fix_attributes_are_absent_on_a_real_wrapper_shape(self):
        """The old probe list (input/user_input/query) finds nothing on a
        wrapper carrying only the fields the SDK defines — which is why
        routing fell through to full_menu on every single call."""
        from decimalai.openai_agents import _extract_query

        ctx = types.SimpleNamespace(context=None, usage=None, tool_input=None)
        assert _extract_query(ctx) is None

    def test_multi_turn_history_routes_on_the_LATEST_user_message(self):
        """`turn_input` is the whole conversation, not just this turn's ask.
        Routing on the first item would pin every turn to the opening line."""
        from decimalai.openai_agents import _extract_query

        ctx = types.SimpleNamespace(turn_input=[
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "4"}]},
            {"role": "user", "content": "Now help me tune this Postgres index."},
        ])
        assert _extract_query(ctx) == "Now help me tune this Postgres index."

    def test_skips_tool_plumbing_items_to_find_the_user_turn(self):
        from decimalai.openai_agents import _extract_query

        ctx = types.SimpleNamespace(turn_input=[
            {"role": "user", "content": "Weather in Paris?"},
            {"type": "function_call", "name": "get_weather", "arguments": '{"city":"Paris"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "sunny"},
            {"type": "reasoning", "id": "r1", "summary": []},
        ])
        assert _extract_query(ctx) == "Weather in Paris?"

    def test_flattens_content_parts(self):
        from decimalai.openai_agents import _extract_query

        ctx = types.SimpleNamespace(turn_input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Summarize "},
                {"type": "input_image", "image_url": "http://x/y.png"},
                {"type": "input_text", "text": "this chart."},
            ],
        }])
        assert _extract_query(ctx) == "Summarize this chart."

    def test_bare_string_turn_input(self):
        from decimalai.openai_agents import _extract_query

        assert _extract_query(types.SimpleNamespace(turn_input="hello")) == "hello"

    def test_legacy_duck_typed_contexts_still_work(self):
        from decimalai.openai_agents import _extract_query

        ctx = types.SimpleNamespace(turn_input=[], user_input="from a custom wrapper")
        assert _extract_query(ctx) == "from a custom wrapper"


# ── Responses-API input items ───────────────────────────────


class TestNormalizeResponsesItems:
    def test_tool_items_keep_their_role_and_body(self):
        """Pre-fix these three items all became {"role":"user","content":""}
        — 6 of 7 recorded messages on a tool-using turn were blank."""
        from decimalai.openai_agents import _normalize_messages

        out = _normalize_messages([
            {"role": "user", "content": "Weather in Paris?"},
            {"type": "function_call", "name": "get_weather",
             "arguments": '{"city":"Paris"}', "call_id": "c1"},
            {"type": "function_call_output", "call_id": "c1", "output": "21C sunny"},
            {"type": "reasoning", "id": "r1",
             "summary": [{"type": "summary_text", "text": "check the tool"}]},
            {"id": "m1", "type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "It is 21C."}]},
        ])

        assert [m["role"] for m in out] == [
            "user", "assistant", "tool", "assistant", "assistant",
        ]
        assert [m["content"] for m in out] == [
            "Weather in Paris?", '{"city":"Paris"}', "21C sunny",
            "check the tool", "It is 21C.",
        ]
        assert out[1]["name"] == "get_weather"
        assert all(m["content"] for m in out), "no message may come back empty"

    def test_chat_completions_shape_is_unchanged(self):
        from decimalai.openai_agents import _normalize_messages

        assert _normalize_messages([{"role": "system", "content": "be terse"}]) == [
            {"role": "system", "content": "be terse"},
        ]

    def test_none_and_bare_string(self):
        from decimalai.openai_agents import _normalize_messages

        assert _normalize_messages(None) is None
        assert _normalize_messages("hi") == [{"role": "user", "content": "hi"}]


# ── 1. Manifest / trace loss ────────────────────────────────


class TestManifestNeverDropsATrace:
    def test_run_with_no_model_and_no_tools_still_gets_a_manifest(self, _reset_sdk):
        """The guardrail-tripwire shape: an agent span with an empty tool
        list and no response span. Pre-fix `_maybe_register_manifest`
        returned on `not tools and not models`, the trace carried
        manifest_id=None and ingest 400'd it away."""
        from decimalai.openai_agents import DecimalTracingProcessor

        proc = DecimalTracingProcessor(agent_name="guarded")
        trace = _Trace("trace_guard")
        proc.on_trace_start(trace)
        proc.on_span_end(_Span("trace_guard", _SpanData(
            "guardrail", name="block", triggered=True)))
        proc.on_span_end(_Span("trace_guard", _SpanData(
            "agent", name="guarded", tools=[], handoffs=[], output_type="str")))
        proc.on_trace_end(trace)

        (sent,) = _ingested(_reset_sdk)
        assert sent.manifest_id, "a trace without a manifest_id is a 400"

    def test_the_live_agent_supplies_what_the_spans_could_not(self, _reset_sdk):
        """The `get_system_prompt` hook records the Agent object against the
        run, so even a run that never reached a model call declares the
        model and tool schemas the author actually wrote."""
        import decimalai.openai_agents as oa
        from decimalai.openai_agents import DecimalTracingProcessor

        tool = types.SimpleNamespace(name="get_weather", params_json_schema={"type": "object"})
        agent = _FakeAgent("guarded", instructions="be terse",
                           model="gpt-4.1-mini", tools=[tool])
        oa._rails_for("trace_guard")["agent"] = agent

        proc = DecimalTracingProcessor(agent_name="guarded")
        trace = _Trace("trace_guard")
        proc.on_trace_start(trace)
        proc.on_span_end(_Span("trace_guard", _SpanData(
            "agent", name="guarded", tools=[], handoffs=[], output_type="str")))
        proc.on_trace_end(trace)

        snapshot = _reset_sdk.register_manifest.call_args.args[0]
        kinds = {c.component_type for c in snapshot.components}
        assert kinds == {"tool", "model", "prompt"}

    def test_nothing_to_declare_adopts_the_manifest_already_on_file(self, _reset_sdk):
        """A run that declares nothing must NOT register an empty manifest.
        The diff engine reads empty→populated as `provider: '' → 'openai'`,
        which is breaking/major — so one unlucky run would fabricate a
        "replay everything" bump on the next healthy one. Point at the
        contract already on file instead."""
        from decimalai.openai_agents import DecimalTracingProcessor

        _reset_sdk.list_manifests.return_value = {"manifests": [
            {"id": "superseded-one", "status": "superseded"},
            {"id": "the-active-one", "status": "active"},
        ]}

        proc = DecimalTracingProcessor(agent_name="known-agent")
        trace = _Trace("trace_bare")
        proc.on_trace_start(trace)
        proc.on_trace_end(trace)

        (sent,) = _ingested(_reset_sdk)
        assert sent.manifest_id == "the-active-one"
        _reset_sdk.register_manifest.assert_not_called()

    def test_a_declared_surface_never_shrinks_between_runs(self, _reset_sdk):
        """Run 1 sees the tools; run 2 (which dies early) does not. Run 2
        must re-register run 1's declaration, not a tool-less one that the
        diff would read as `get_weather removed` — breaking/major."""
        from decimalai.openai_agents import DecimalTracingProcessor

        proc = DecimalTracingProcessor(agent_name="flappy")

        t1 = _Trace("t1")
        proc.on_trace_start(t1)
        proc.on_span_end(_Span("t1", _SpanData(
            "agent", name="flappy", tools=["get_weather"], handoffs=[])))
        proc.on_span_end(_Span("t1", _SpanData(
            "generation", model="gpt-4.1-mini", usage={}, input=None, output=None)))
        proc.on_trace_end(t1)

        t2 = _Trace("t2")
        proc.on_trace_start(t2)
        proc.on_span_end(_Span("t2", _SpanData(
            "agent", name="flappy", tools=[], handoffs=[])))
        proc.on_trace_end(t2)

        snapshots = [c.args[0] for c in _reset_sdk.register_manifest.call_args_list]
        assert snapshots, "run 1 must have registered"
        tool_names = {
            c.component_name for c in snapshots[-1].components
            if c.component_type == "tool"
        }
        assert tool_names == {"get_weather"}
        # Same declaration → same hash → one registration, no version churn.
        assert len(snapshots) == 1

    def test_load_skill_is_not_part_of_the_declared_contract(self, _reset_sdk):
        """The adapter attaches load_skill itself. Counting it would make
        the manifest — and therefore the version history — depend on
        whether the skill loader happened to be switched on."""
        from decimalai.openai_agents import DecimalTracingProcessor

        proc = DecimalTracingProcessor(agent_name="skilled")
        trace = _Trace("t")
        proc.on_trace_start(trace)
        proc.on_span_end(_Span("t", _SpanData(
            "agent", name="skilled", tools=["get_weather", "load_skill"], handoffs=[])))
        proc.on_trace_end(trace)

        snapshot = _reset_sdk.register_manifest.call_args.args[0]
        assert {c.component_name for c in snapshot.components
                if c.component_type == "tool"} == {"get_weather"}

    def test_the_declared_model_beats_the_resolved_snapshot(self, _reset_sdk):
        """`model="gpt-4.1-mini"` comes back as `gpt-4.1-mini-2025-04-14`.
        Letting the resolved snapshot into the contract means a silent
        rotation at the provider mints a breaking bump nobody caused — and,
        concretely, made an early-failing run and a healthy run of the SAME
        agent register as v1 and v2."""
        import decimalai.openai_agents as oa
        from decimalai.openai_agents import DecimalTracingProcessor

        agent = _FakeAgent("pinned", instructions="hi", model="gpt-4.1-mini")
        proc = DecimalTracingProcessor(agent_name="pinned")

        for trace_id in ("t1", "t2"):
            oa._rails_for(trace_id)["agent"] = agent
            trace = _Trace(trace_id)
            proc.on_trace_start(trace)
            proc.on_span_end(_Span(trace_id, _SpanData(
                "generation", model="gpt-4.1-mini-2025-04-14", usage={},
                input=None, output=None)))
            proc.on_trace_end(trace)

        snapshots = [c.args[0] for c in _reset_sdk.register_manifest.call_args_list]
        models = [
            c.schema_json["model"] for s in snapshots for c in s.components
            if c.component_type == "model"
        ]
        assert models == ["gpt-4.1-mini"]
        assert len(snapshots) == 1, "the second run must dedupe, not mint v2"

    def test_install_time_introspection_and_the_first_trace_agree(self, _reset_sdk):
        """`instrument(agent=...)` registers at install; the first trace then
        registers again from span data. If the two disagree on any surface —
        the install one declares skills, the trace one does not — the second
        reads as `skill_registry removed`, which is moderate/breaking. They
        must produce the SAME snapshot and dedupe."""
        import decimalai.openai_agents as oa
        from decimalai.openai_agents import DecimalTracingProcessor

        tool = types.SimpleNamespace(name="get_weather", params_json_schema={"type": "object"})
        agent = _FakeAgent("dual", instructions="hi", model="gpt-4.1-mini", tools=[tool])
        skills = [{"name": "sql-optimizer", "hash": "h1", "description": "d"}]

        oa._register_manifest_from_agent(agent, "dual", skills)

        proc = DecimalTracingProcessor(agent_name="dual", skills_registry=skills)
        oa._rails_for("t")["agent"] = agent
        trace = _Trace("t")
        proc.on_trace_start(trace)
        proc.on_span_end(_Span("t", _SpanData(
            "agent", name="dual", tools=["get_weather"], handoffs=[])))
        proc.on_span_end(_Span("t", _SpanData(
            "generation", model="gpt-4.1-mini", usage={}, input=None, output=None)))
        proc.on_trace_end(trace)

        assert _reset_sdk.register_manifest.call_count == 1
        (sent,) = _ingested(_reset_sdk)
        assert sent.manifest_id == "mf-dual"

    def test_each_agent_gets_its_own_manifest(self, _reset_sdk):
        """`_manifest_id` was one process-global slot: in a process running
        two agents, the second agent's traces were stamped with the FIRST
        agent's manifest, so the manifest→trace join attributed one agent's
        runs to another's contract."""
        from decimalai.openai_agents import DecimalTracingProcessor

        for name in ("alpha", "beta"):
            proc = DecimalTracingProcessor(agent_name=name)
            trace = _Trace(f"trace_{name}")
            proc.on_trace_start(trace)
            proc.on_span_end(_Span(f"trace_{name}", _SpanData(
                "agent", name=name, tools=[f"{name}_tool"], handoffs=[])))
            proc.on_span_end(_Span(f"trace_{name}", _SpanData(
                "generation", model="gpt-4.1-mini", usage={}, input=None, output=None)))
            proc.on_trace_end(trace)

        a, b = _ingested(_reset_sdk)
        assert (a.manifest_id, b.manifest_id) == ("mf-alpha", "mf-beta")

    def test_a_failed_model_call_keeps_a_usable_model_name(self, _reset_sdk):
        """Ingest rejects an llm_call with no model_name, so an errored
        response span (no Response object → no model) took the whole trace
        down with it — a second, independent 400 on the same runs."""
        import decimalai.openai_agents as oa
        from decimalai.openai_agents import DecimalTracingProcessor

        oa._rails_for("t")["agent"] = _FakeAgent(
            "broken", instructions="hi", model="gpt-does-not-exist-9x")

        proc = DecimalTracingProcessor(agent_name="broken")
        trace = _Trace("t")
        proc.on_trace_start(trace)
        proc.on_span_end(_Span(
            "t", _SpanData("response", response=None, input=None, usage=None),
            error={"message": "Error getting response"},
        ))
        proc.on_trace_end(trace)

        (sent,) = _ingested(_reset_sdk)
        assert [c.model_name for c in sent.llm_calls] == ["gpt-does-not-exist-9x"]

    def test_an_unknowable_model_drops_the_record_not_the_trace(self, _reset_sdk):
        from decimalai.openai_agents import DecimalTracingProcessor

        proc = DecimalTracingProcessor(agent_name="broken")
        trace = _Trace("t")
        proc.on_trace_start(trace)
        proc.on_span_end(_Span(
            "t", _SpanData("response", response=None, input=None, usage=None),
            error={"message": "Error getting response"},
        ))
        proc.on_trace_end(trace)

        (sent,) = _ingested(_reset_sdk)
        assert sent.llm_calls == []
        assert any(s.status.value == "error" for s in sent.spans), \
            "the error must survive on the span"


# ── 2. Per-run rails ────────────────────────────────────────


class TestRailsArePerRun:
    def test_two_live_runs_do_not_steal_each_others_routing(self, _reset_sdk):
        """Both runs are open at once, exactly as two parallel
        `Runner.run` calls are. Pre-fix both drained one process-global
        router rail: the first trace to end took everything and the second
        reported `routing_id=None`."""
        import decimalai.openai_agents as oa
        from decimalai.openai_agents import DecimalTracingProcessor

        proc = DecimalTracingProcessor()
        a, b = _Trace("run_a", "a"), _Trace("run_b", "b")
        proc.on_trace_start(a)
        proc.on_trace_start(b)

        # Interleaved, as the runner would produce them.
        oa._rails_for("run_a").update(
            {"routing_id": "rt_a", "offered": ["sql-optimizer"]})
        oa._rails_for("run_b").update(
            {"routing_id": "rt_b", "offered": ["code-reviewer"], "loaded": ["code-reviewer"]})

        proc.on_trace_end(b)
        proc.on_trace_end(a)

        sent_b, sent_a = _ingested(_reset_sdk)
        assert (sent_a.routing_id, sent_b.routing_id) == ("rt_a", "rt_b")
        assert sent_a.skills_offered_in_prompt == ["sql-optimizer"]
        assert sent_b.skills_offered_in_prompt == ["code-reviewer"]
        assert sent_b.skills_loaded_by_agent == ["code-reviewer"]
        assert sent_a.skills_loaded_by_agent == []

    def test_a_finished_run_leaves_no_rail_behind(self, _reset_sdk):
        import decimalai.openai_agents as oa
        from decimalai.openai_agents import DecimalTracingProcessor

        proc = DecimalTracingProcessor()
        trace = _Trace("run_x")
        proc.on_trace_start(trace)
        oa._rails_for("run_x").update({"routing_id": "rt_x", "offered": ["s"]})
        proc.on_trace_end(trace)

        assert "run_x" not in oa._run_rails

        proc.on_trace_start(_Trace("run_y"))
        proc.on_trace_end(_Trace("run_y"))
        _, second = _ingested(_reset_sdk)
        assert second.routing_id is None
        assert second.skills_offered_in_prompt == []

    def test_the_rail_registry_is_bounded(self):
        import decimalai.openai_agents as oa

        for i in range(oa._RUN_RAILS_MAX + 25):
            oa._rails_for(f"leaked_{i}")
        assert len(oa._run_rails) == oa._RUN_RAILS_MAX

    def test_the_router_budget_is_kept_per_run(self):
        """Two concurrent runs sharing one router singleton each get their
        own per-turn body budget — `_last_budget` is a single slot they
        used to overwrite for one another."""
        from decimalai.skill_router import SkillRouter

        router = SkillRouter(api_key="k", base_url="http://localhost:8000")
        router.get_skill_body = lambda name, **kw: "x" * 100  # noqa: ARG005

        router.load_skill("s1", scope="run_a")
        router.load_skill("s2", scope="run_b")

        assert list(router._budget_for("run_a").loaded) == ["s1"]
        assert list(router._budget_for("run_b").loaded) == ["s2"]


# ── 3. Constructor / retrofit ───────────────────────────────


def _fake_agents_module_with_class_hooks():
    """An `agents` stand-in whose Agent has the two async class methods the
    real SDK resolves per turn."""
    mod = types.ModuleType("agents")

    class Agent:
        def __init__(self, name=None, instructions=None, tools=None):
            self.name = name
            self.instructions = instructions
            self.tools = list(tools or [])

        async def get_system_prompt(self, run_context):
            if callable(self.instructions):
                return self.instructions(run_context, self)
            return self.instructions

        async def get_all_tools(self, run_context):
            return list(self.tools)

    def function_tool(fn):
        return types.SimpleNamespace(name=fn.__name__, fn=fn)

    mod.Agent = Agent
    mod.function_tool = function_tool
    mod.tracing = types.ModuleType("agents.tracing")
    mod.tracing.get_current_trace = lambda: None
    return mod, Agent


@pytest.fixture
def fake_agents(monkeypatch):
    import decimalai.openai_agents as oa
    from decimalai import skill_router as sr

    mod, Agent = _fake_agents_module_with_class_hooks()
    monkeypatch.setitem(sys.modules, "agents", mod)
    monkeypatch.setitem(sys.modules, "agents.tracing", mod.tracing)
    monkeypatch.setattr(oa, "_skill_loader_installed", False)
    monkeypatch.setattr(oa, "_agent_hooks_installed", False)
    monkeypatch.setattr(oa, "_retrofit_notice_emitted", False)
    monkeypatch.setattr(oa, "_skill_router_singleton", None)
    sr._last_offered_names_ctx.set(None)
    sr._last_delivered_names_ctx.set(None)
    yield mod, Agent, oa


class TestRetrofitsAgentsBuiltBeforeInstrument:
    def test_a_pre_existing_agent_gets_skills_and_the_tool(self, fake_agents):
        """The failing shape from the audit: an agent defined at module
        level, i.e. before `instrument()` ran. `Agent.__init__` patching
        can only reach objects built after the patch, so this agent got 0
        skills offered and no load_skill tool — silently."""
        _, Agent, oa = fake_agents

        prebuilt = Agent(name="early", instructions="be terse")  # BEFORE install

        router = MagicMock()
        router.build_prompt_fragment.return_value = ("## Recommended Skills\n| a |", "rt_1")
        oa._install_skill_loader()
        oa._skill_router_singleton = router

        prompt = asyncio.run(prebuilt.get_system_prompt(types.SimpleNamespace()))
        tools = asyncio.run(prebuilt.get_all_tools(types.SimpleNamespace()))

        assert "## Recommended Skills" in prompt
        assert prompt.endswith("be terse")
        assert [t.name for t in tools] == ["load_skill"]
        # The retrofit must not mutate what the user declared.
        assert prebuilt.instructions == "be terse"
        assert prebuilt.tools == []

    def test_agents_built_after_install_are_not_double_injected(self, fake_agents):
        """Both layers are live at once; the fragment must appear once."""
        _, Agent, oa = fake_agents

        router = MagicMock()
        router.build_prompt_fragment.return_value = ("## Recommended Skills\n| a |", "rt_1")
        oa._install_skill_loader()
        oa._skill_router_singleton = router

        later = Agent(name="late", instructions="be terse")
        prompt = asyncio.run(later.get_system_prompt(types.SimpleNamespace()))

        assert prompt.count("## Recommended Skills") == 1
        tools = asyncio.run(later.get_all_tools(types.SimpleNamespace()))
        assert [t.name for t in tools] == ["load_skill"]

    def test_a_user_written_instructions_callable_is_left_alone(self, fake_agents):
        _, Agent, oa = fake_agents

        def mine(ctx, agent):
            return "MY PROMPT"

        prebuilt = Agent(name="early", instructions=mine)
        oa._skill_router_singleton = MagicMock()
        oa._install_skill_loader()

        assert asyncio.run(prebuilt.get_system_prompt(types.SimpleNamespace())) == "MY PROMPT"

    def test_plain_instrument_hooks_the_class_without_loading_skills(self, fake_agents):
        """The observer hook is always on — it is what hands the trace
        processor the live Agent — but it must not inject skills."""
        _, Agent, oa = fake_agents

        prebuilt = Agent(name="early", instructions="be terse")
        assert oa._install_agent_hooks() is True

        assert asyncio.run(prebuilt.get_system_prompt(types.SimpleNamespace())) == "be terse"
        assert asyncio.run(prebuilt.get_all_tools(types.SimpleNamespace())) == []


class TestSkillLoaderKeepsThePromptComponent:
    def test_wrapped_instructions_still_yield_a_prompt_component(self, fake_agents):
        """Turning the loader on replaced `instructions` with a callable,
        and `_introspect_agent` only recorded `isinstance(..., str)` — so
        the prompt component vanished from the manifest and prompt-drift
        detection went blind for exactly the installs using skills."""
        _, Agent, oa = fake_agents

        oa._install_skill_loader()
        agent = Agent(name="a", instructions="You are a careful reviewer.")
        assert callable(agent.instructions)

        data = oa._introspect_agent(agent)
        assert data["prompts"] == {"system": "You are a careful reviewer."}
