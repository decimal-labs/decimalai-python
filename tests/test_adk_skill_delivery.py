"""Google ADK can deliver a skill BODY — proven at the model boundary.

``decimalai/cli/scaffold.py::NO_PROMPT_SEAM`` listed ``adk`` among the
frameworks whose "generated file would trace correctly and deliver none of the
agent's skills". That was wrong, and this file is the refutation.

The seam, read out of google-adk 2.8.0 rather than remembered:

* ``BasePlugin.before_model_callback(*, callback_context, llm_request)`` gets
  the live ``LlmRequest`` BY REFERENCE. ``flows/llm_flows/base_llm_flow.py``
  runs it at line 1735 and passes that same object to
  ``llm.generate_content_async(llm_request, ...)`` at line 1801, with no copy in
  between.
* ``LlmRequest.append_instructions(list[str])`` is public API and appends to
  ``config.system_instruction`` with a ``"\\n\\n"`` join
  (``models/llm_request.py:262-279``).
* ADK's OWN ``GlobalInstructionPlugin`` writes
  ``llm_request.config.system_instruction`` from this very hook
  (``plugins/global_instruction_plugin.py:86-121``) — first-party proof that
  this is a supported extension point, not an internal we poked at.

Two layers, because google-adk is not in the SDK's own dev environment:

* ``TestSeamWithoutAdk`` drives the plugin callbacks directly against a
  synthetic request, so the rails are graded on every run of the suite.
* ``TestRealAdkRun`` walks a real ``Runner.run_async`` with a real ``LlmAgent``
  and asserts the sentinel in the request the model was handed. Gemini keys on
  this machine are quota-dead, so the model is a ``BaseLlm`` subclass that
  CAPTURES the outbound request instead of sending it — the payload is the real
  one ADK built, the network is the only thing missing.
  ``test_the_synthetic_request_has_not_drifted`` pins the first layer's
  stand-in to the second layer's real object so the cheap tests cannot go green
  on a fiction.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


def _real_adk_installed() -> bool:
    """Is google-adk actually here?

    Asked at IMPORT time, deliberately: this file (and ``test_adk_error_path``)
    put a stub ``google.adk`` in ``sys.modules`` inside a fixture, so an
    ``importorskip`` asked from inside a test would find the stub and the real
    run would fail on ``google.adk.models`` instead of skipping.
    """
    try:
        return importlib.util.find_spec("google.adk.models.llm_request") is not None
    except (ImportError, ValueError):
        return False


HAS_REAL_ADK = _real_adk_installed()

# A menu row could never contain this; only a delivered BODY can.
SENTINEL = "SENTINEL-SKILLBODY-ADK-7f3a2c"
CALLER_PROMPT = "You are a terse support agent."
PREFIX = (
    "== DecimalAI skills ==\n"
    "| refund-policy | how refunds work |\n\n"
    "### refund-policy\n"
    f"Opened boxes carry a 23.5% restocking fee. {SENTINEL}\n"
)
TAIL = "Most relevant for this request: refund-policy."
ROUTING_ID = "rt_" + "a" * 24
USER_QUESTION = "What fee applies to an opened box return?"


# ── routers ─────────────────────────────────────────────────


class _SplitRouter:
    """Stand-in for the platform: a constant PREFIX carrying the body, and a
    TAIL that varies with the query. Records every query it is asked."""

    def __init__(self) -> None:
        self.queries: List[Optional[str]] = []
        self.scopes: List[Optional[str]] = []
        self.agent_names: List[Optional[str]] = []

    def _rails(self) -> None:
        import decimalai.skill_router as sr

        sr._last_offered_names_ctx.set(["refund-policy", "returns-window"])
        sr._last_delivered_names_ctx.set(["refund-policy"])

    def build_prompt_parts(self, query=None, *, agent_name=None, scope=None, **kw):
        self.queries.append(query)
        self.scopes.append(scope)
        self.agent_names.append(agent_name)
        self._rails()
        return PREFIX, TAIL, ROUTING_ID


class _LegacyRouter(_SplitRouter):
    """A router object built before the prefix/tail split existed."""

    build_prompt_parts = None  # not callable -> the adapter must degrade

    def build_prompt_fragment(self, query=None, *, agent_name=None, scope=None, **kw):
        self.queries.append(query)
        self._rails()
        return PREFIX, ROUTING_ID


class _EmptyRouter(_SplitRouter):
    """The platform had nothing to offer for this query."""

    def build_prompt_parts(self, query=None, **kw):
        self.queries.append(query)
        import decimalai.skill_router as sr

        sr._last_offered_names_ctx.set(None)
        sr._last_delivered_names_ctx.set(None)
        return "", "", None


class _VersionedRouter(_SplitRouter):
    def __init__(self, hashes=None):
        super().__init__()
        self.hashes = iter(hashes or ["a" * 64])

    def _rails(self):
        super()._rails()
        import decimalai.skill_router as sr

        sr._last_delivered_versions_ctx.set([
            {"name": "refund-policy", "hash": next(self.hashes)},
        ])


# ── a stand-in for ADK's LlmRequest ─────────────────────────


class _FakeConfig:
    def __init__(self, system_instruction: Any = None) -> None:
        self.system_instruction = system_instruction


class _FakeLlmRequest:
    """Mirrors ``google.adk.models.llm_request.LlmRequest`` on the two members
    the adapter touches. ``test_the_synthetic_request_has_not_drifted`` holds it
    to the real thing wherever google-adk is installed."""

    def __init__(self, system_instruction: Any = None, contents: Any = None) -> None:
        self.model = "gemini-3.6-flash"
        self.config = _FakeConfig(system_instruction)
        self.contents = list(contents or [])

    def append_instructions(self, instructions: List[str]) -> List[Any]:
        if not instructions:
            return []
        new_text = "\n\n".join(instructions)
        si = self.config.system_instruction
        if not si:
            self.config.system_instruction = new_text
        elif isinstance(si, str):
            self.config.system_instruction = si + "\n\n" + new_text
        # A non-str system_instruction is DROPPED with a log warning by the real
        # ADK — that silent no-op is the whole reason `_append_system_text` has
        # a non-str branch and a read-back check.
        return []


def _text_content(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, parts=[SimpleNamespace(text=text)])


def _function_response_content(name: str) -> SimpleNamespace:
    """What ADK appends after a tool runs: role="user", NO text.

    Verified against a real two-step run — see the module docstring's second
    layer, and ``TestRealAdkRun.test_a_tool_turn_keeps_the_body_and_the_query``.
    """
    return SimpleNamespace(
        role="user",
        parts=[SimpleNamespace(text=None, function_response=SimpleNamespace(name=name))],
    )


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_sdk(monkeypatch):
    """Fresh SDK globals + a stubbed google-adk BasePlugin, per test."""
    import decimalai._config as cfg
    from decimalai import skill_router as sr
    from decimalai._config import DecimalConfig

    sr._last_delivered_versions_ctx.set(None)

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }

    import decimalai.adk as adk

    if HAS_REAL_ADK:
        from google.adk.runners import Runner

        # The plugin factory now preserves parents at Runner.run's caller
        # boundary as well as patching the constructor during instrument().
        monkeypatch.setattr(Runner, "run", Runner.run)
        monkeypatch.setattr(Runner, "__init__", Runner.__init__)

    saved = (
        adk._PluginClass,
        dict(adk._manifest_ids),
        dict(adk._manifest_trackers),
        adk._skill_loader_enabled,
        adk._skill_router_singleton,
        adk._instrument_config,
        adk._runner_patched,
        adk._install_agent_name,
    )
    adk._manifest_ids = {}
    adk._manifest_trackers = {}
    adk._skill_loader_enabled = False
    adk._skill_router_singleton = None
    adk._instrument_config = adk._InstrumentationConfig()
    adk._runner_patched = False
    adk._install_agent_name = None

    if not HAS_REAL_ADK:
        # No real google-adk here: stub the one import `_plugin_class()` does.
        adk._PluginClass = None
        base_plugin_mod = types.ModuleType("google.adk.plugins.base_plugin")

        class BasePlugin:  # the real one just stores the name
            def __init__(self, name):
                self.name = name

        base_plugin_mod.BasePlugin = BasePlugin
        plugins_mod = types.ModuleType("google.adk.plugins")
        plugins_mod.base_plugin = base_plugin_mod
        adk_pkg_mod = types.ModuleType("google.adk")
        adk_pkg_mod.plugins = plugins_mod
        google_mod = types.ModuleType("google")
        google_mod.adk = adk_pkg_mod
        for name, mod in (
            ("google", google_mod),
            ("google.adk", adk_pkg_mod),
            ("google.adk.plugins", plugins_mod),
            ("google.adk.plugins.base_plugin", base_plugin_mod),
        ):
            monkeypatch.setitem(sys.modules, name, mod)

    yield

    sr._last_delivered_versions_ctx.set(None)

    (
        adk._PluginClass,
        adk._manifest_ids,
        adk._manifest_trackers,
        adk._skill_loader_enabled,
        adk._skill_router_singleton,
        adk._instrument_config,
        adk._runner_patched,
        adk._install_agent_name,
    ) = saved


@pytest.fixture
def use_router(monkeypatch):
    def _use(router):
        import decimalai.adk as adk

        monkeypatch.setattr(adk, "_skill_router_singleton", router)
        return router

    return _use


def _flushed_traces() -> List[Any]:
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in cfg._client.ingest_trace.call_args_list]


def _agent(name: str = "root_agent") -> SimpleNamespace:
    return SimpleNamespace(
        name=name, model="gemini-3.6-flash", instruction=CALLER_PROMPT,
        tools=[], sub_agents=[],
    )


def _drive_one_turn(
    plugin: Any,
    request: Any,
    *,
    inv_id: str = "inv-1",
    user_content: Any = USER_QUESTION,
) -> Any:
    """before_run -> before_agent -> before_model -> after_agent."""
    agent = _agent()
    ic = SimpleNamespace(agent=agent, invocation_id=inv_id, user_content=user_content)
    cc = SimpleNamespace(invocation_id=inv_id, agent_name=None)

    async def _go():
        await plugin.before_run_callback(invocation_context=ic)
        await plugin.before_agent_callback(agent=agent, callback_context=cc)
        await plugin.before_model_callback(callback_context=cc, llm_request=request)
        await plugin.after_agent_callback(agent=agent, callback_context=cc)

    asyncio.run(_go())
    return request


# ── layer 1: the seam, graded on every run of the suite ─────


class TestSeamWithoutAdk:
    def test_the_loader_is_off_by_default(self, use_router):
        """Nobody who did not ask for skills gets their prompt rewritten."""
        from decimalai.adk import DecimalaiPlugin

        router = use_router(_SplitRouter())
        req = _drive_one_turn(
            DecimalaiPlugin(agent_name="support"), _FakeLlmRequest(CALLER_PROMPT),
        )
        assert req.config.system_instruction == CALLER_PROMPT
        assert router.queries == [], "the Router was consulted with the loader off"
        assert _flushed_traces()[0].skills_offered_in_prompt == []

    def test_the_body_reaches_the_system_instruction(self, use_router):
        from decimalai.adk import DecimalaiPlugin

        use_router(_SplitRouter())
        req = _drive_one_turn(
            DecimalaiPlugin(agent_name="support", enable_skill_loader=True),
            _FakeLlmRequest(CALLER_PROMPT),
        )
        si = req.config.system_instruction
        assert SENTINEL in si, "the skill BODY never reached the request"
        assert TAIL in si

    def test_the_callers_prompt_leads_and_the_varying_half_trails(self, use_router):
        """The cacheable ordering: caller's stable prompt, then the stable
        menu+bodies, then the one sentence that depends on this query."""
        from decimalai.adk import DecimalaiPlugin

        use_router(_SplitRouter())
        req = _drive_one_turn(
            DecimalaiPlugin(agent_name="support", enable_skill_loader=True),
            _FakeLlmRequest(CALLER_PROMPT),
        )
        si = req.config.system_instruction
        assert si.index(CALLER_PROMPT) < si.index(SENTINEL) < si.index(TAIL)

    def test_the_trace_carries_the_delivery_rails(self, use_router):
        from decimalai.adk import DecimalaiPlugin

        use_router(_SplitRouter())
        _drive_one_turn(
            DecimalaiPlugin(agent_name="support", enable_skill_loader=True),
            _FakeLlmRequest(CALLER_PROMPT),
        )
        trace = _flushed_traces()[0]
        assert trace.routing_id == ROUTING_ID
        assert trace.skills_offered_in_prompt == ["refund-policy", "returns-window"]
        assert trace.skills_delivered == ["refund-policy"]
        # Delivered is NOT an activation. ADK has no load_skill tool, so nothing
        # here can be a model-initiated pull; stamping the delivered body (or its
        # hash) onto active_skills would fabricate an activation — the platform
        # turns every active_skills entry into a TraceSkillActivation row, and
        # conformance C13 fails that payload. Proposed on 2026-09-04 as "stamp
        # drained hashes onto active_skills"; refused, and pinned here.
        assert trace.active_skills == []
        assert trace.skills_loaded_by_agent == []

    def test_the_router_is_scoped_to_this_run_and_named_for_this_agent(self, use_router):
        """Two concurrent runs sharing the singleton must not share a routing
        decision, and an agent-scope skill only resolves if the name is sent."""
        from decimalai.adk import DecimalaiPlugin

        router = use_router(_SplitRouter())
        plugin = DecimalaiPlugin(agent_name="support", enable_skill_loader=True)
        _drive_one_turn(plugin, _FakeLlmRequest(CALLER_PROMPT), inv_id="inv-1")
        _drive_one_turn(plugin, _FakeLlmRequest(CALLER_PROMPT), inv_id="inv-2")
        assert router.agent_names == ["support", "support"]
        assert len(set(router.scopes)) == 2, "two runs shared one routing scope"
        assert all(s for s in router.scopes)

    def test_a_tool_turn_routes_on_the_users_question_not_the_tool_result(
        self, use_router,
    ):
        """ADK files a tool result as a role="user" content with NO text. Taking
        the last user content naively makes step 2 of one turn route on that
        instead — an empty query, which means full-menu mode, a different cache
        key and a second routing decision for one user turn."""
        from decimalai.adk import DecimalaiPlugin

        router = use_router(_SplitRouter())
        plugin = DecimalaiPlugin(agent_name="support", enable_skill_loader=True)
        agent = _agent()
        ic = SimpleNamespace(agent=agent, invocation_id="inv-1", user_content=USER_QUESTION)
        cc = SimpleNamespace(invocation_id="inv-1", agent_name=None)
        step1 = _FakeLlmRequest(CALLER_PROMPT, [_text_content("user", USER_QUESTION)])
        step2 = _FakeLlmRequest(
            CALLER_PROMPT,
            [
                _text_content("user", USER_QUESTION),
                _text_content("model", ""),
                _function_response_content("lookup_order"),
            ],
        )

        async def _go():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_agent_callback(agent=agent, callback_context=cc)
            await plugin.before_model_callback(callback_context=cc, llm_request=step1)
            await plugin.before_model_callback(callback_context=cc, llm_request=step2)
            await plugin.after_agent_callback(agent=agent, callback_context=cc)

        asyncio.run(_go())
        assert router.queries == [USER_QUESTION, USER_QUESTION], router.queries
        assert SENTINEL in step2.config.system_instruction, (
            "the second step of one turn lost the body"
        )
        trace = _flushed_traces()[0]
        assert trace.skills_delivered == ["refund-policy"], "a skill counted twice"

    def test_a_router_without_the_split_still_delivers(self, use_router):
        """The silent no-op guard: swallowing the AttributeError and returning
        would trace perfectly and inject nothing."""
        from decimalai.adk import DecimalaiPlugin

        use_router(_LegacyRouter())
        req = _drive_one_turn(
            DecimalaiPlugin(agent_name="support", enable_skill_loader=True),
            _FakeLlmRequest(CALLER_PROMPT),
        )
        assert SENTINEL in req.config.system_instruction
        assert _flushed_traces()[0].skills_delivered == ["refund-policy"]

    def test_an_empty_route_claims_nothing(self, use_router):
        from decimalai.adk import DecimalaiPlugin

        use_router(_EmptyRouter())
        req = _drive_one_turn(
            DecimalaiPlugin(agent_name="support", enable_skill_loader=True),
            _FakeLlmRequest(CALLER_PROMPT),
        )
        assert req.config.system_instruction == CALLER_PROMPT
        trace = _flushed_traces()[0]
        assert trace.skills_offered_in_prompt == []
        assert trace.skills_delivered == []
        assert trace.routing_id is None

    def test_an_append_that_silently_no_ops_claims_nothing(self, use_router):
        """The honesty rail. A request whose append does nothing must produce a
        trace that says nothing was offered or delivered — reporting reach that
        never happened is worse than reporting none."""
        from decimalai.adk import DecimalaiPlugin

        class _RefusingRequest(_FakeLlmRequest):
            def append_instructions(self, instructions):
                return []  # exactly what real ADK does for a non-str

        use_router(_SplitRouter())
        req = _drive_one_turn(
            DecimalaiPlugin(agent_name="support", enable_skill_loader=True),
            _RefusingRequest(CALLER_PROMPT),
        )
        assert SENTINEL not in (req.config.system_instruction or "")
        trace = _flushed_traces()[0]
        assert trace.skills_delivered == [], "claimed delivery of a body the model never saw"
        assert trace.skills_offered_in_prompt == []
        assert trace.routing_id is None

    def test_a_router_that_raises_never_breaks_the_model_call(self, use_router):
        from decimalai.adk import DecimalaiPlugin

        class _Exploding:
            def build_prompt_parts(self, *a, **k):
                raise RuntimeError("platform down")

        use_router(_Exploding())
        req = _drive_one_turn(
            DecimalaiPlugin(agent_name="support", enable_skill_loader=True),
            _FakeLlmRequest(CALLER_PROMPT),
        )
        assert req.config.system_instruction == CALLER_PROMPT
        assert _flushed_traces()[0].skills_delivered == []

    def test_instrument_turns_the_loader_on_after_a_bare_instrument(self):
        """`instrument()` is idempotent on the Runner patch, but the flag must
        still take: a second call that asks for skills cannot be swallowed by
        the early return."""
        import decimalai.adk as adk

        adk.instrument(agent_name="support")
        assert adk._skill_loader_enabled is False
        adk.instrument(agent_name="support", enable_skill_loader=True)
        assert adk._skill_loader_enabled is True

    def test_bodies_are_on_by_default_because_adk_has_no_load_skill_tool(
        self, monkeypatch,
    ):
        """ADK registers no `load_skill` tool, so injection is the ONLY body
        channel — the `has_tool_loop=False` case `resolve_inject_body` answers
        True for. Built with bodies off, the model gets a menu of titles it has
        no mechanism to read."""
        import decimalai.adk as adk

        captured: Dict[str, Any] = {}

        class _Router:
            def __init__(self, **kw):
                captured.update(kw)

        monkeypatch.setattr(adk, "_skill_router_singleton", None)
        monkeypatch.setattr("decimalai.skill_router.SkillRouter", _Router)
        adk._get_skill_router()
        assert captured["inject_body"] is True


class TestInvocationObserver:
    def test_observer_gets_one_detached_delivery_snapshot_and_cannot_mutate_export(self, use_router):
        from decimalai.adk import DecimalaiPlugin

        use_router(_SplitRouter())
        seen = []

        def observe(trace):
            seen.append(trace.model_copy(deep=True))
            trace.skills_delivered.append("observer-mutation")
            trace.agent_name = "observer-mutation"

        plugin = DecimalaiPlugin(agent_name="support", enable_skill_loader=True, on_trace=observe)
        _drive_one_turn(plugin, _FakeLlmRequest(CALLER_PROMPT))
        asyncio.run(plugin.after_run_callback(invocation_context=SimpleNamespace(invocation_id="inv-1")))
        asyncio.run(plugin.on_run_error_callback(
            invocation_context=SimpleNamespace(invocation_id="inv-1"), error=RuntimeError("late"),
        ))
        sent = _flushed_traces()
        assert len(seen) == len(sent) == 1
        assert seen[0].id == sent[0].id
        assert seen[0].skills_delivered == sent[0].skills_delivered == ["refund-policy"]
        assert seen[0].active_skills == sent[0].active_skills == []
        assert sent[0].agent_name == "support"

    def test_an_error_invocation_observes_delivery_once_and_observer_failure_does_not_drop_it(self, use_router):
        from decimalai.adk import DecimalaiPlugin
        from decimalai.schema.common import Status

        use_router(_SplitRouter())
        seen = []

        def broken_observer(trace):
            seen.append(trace)
            raise RuntimeError("observer failed")

        plugin = DecimalaiPlugin(enable_skill_loader=True, on_trace=broken_observer)
        ic = SimpleNamespace(agent=_agent(), invocation_id="error", user_content=USER_QUESTION)
        cc = SimpleNamespace(invocation_id="error")
        req = _FakeLlmRequest(CALLER_PROMPT)

        async def drive():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_model_callback(callback_context=cc, llm_request=req)
            await plugin.on_model_error_callback(callback_context=cc, llm_request=req, error=RuntimeError("429"))
            await plugin.on_run_error_callback(invocation_context=ic, error=RuntimeError("429"))
            await plugin.on_run_error_callback(invocation_context=ic, error=RuntimeError("late"))

        asyncio.run(drive())
        sent = _flushed_traces()
        assert len(seen) == len(sent) == 1
        assert seen[0].status == sent[0].status == Status.ERROR
        assert seen[0].skills_delivered == ["refund-policy"]
        assert seen[0].active_skills == []

    def test_parent_is_captured_at_invocation_start_before_executor_finalization(self):
        from decimalai.adk import DecimalaiPlugin
        from decimalai.generic import start_trace

        seen = []
        plugin = DecimalaiPlugin(on_trace=seen.append)
        ic = SimpleNamespace(agent=_agent(), invocation_id="parent", user_content=USER_QUESTION)
        with start_trace(agent_name="outer", auto_send=False) as outer:
            parent_id = outer.get_trace_id()
            asyncio.run(plugin.before_run_callback(invocation_context=ic))
        # The generic context has already closed; finalization's executor has no
        # ContextVar inheritance. Reading the parent here would silently unlink.
        asyncio.run(plugin.after_run_callback(invocation_context=ic))
        assert seen[0].parent_trace_id == _flushed_traces()[0].parent_trace_id == parent_id

    def test_explicit_parent_wins_and_standalone_invocations_stay_roots(self):
        from uuid import uuid4

        from decimalai.adk import DecimalaiPlugin
        from decimalai.generic import start_trace

        parent_id = str(uuid4())
        seen = []
        with start_trace(agent_name="outer", auto_send=False):
            _drive_one_turn(DecimalaiPlugin(parent_trace_id=parent_id, on_trace=seen.append), _FakeLlmRequest())
        _drive_one_turn(DecimalaiPlugin(on_trace=seen.append), _FakeLlmRequest(), inv_id="standalone")
        assert [trace.parent_trace_id for trace in seen] == [parent_id, None]

    def test_concurrent_generic_contexts_keep_their_own_parent(self):
        from decimalai.adk import DecimalaiPlugin
        from decimalai.generic import start_trace

        seen, parents = [], {}
        plugin = DecimalaiPlugin(on_trace=seen.append)

        async def run(invocation_id):
            with start_trace(agent_name=invocation_id, auto_send=False) as outer:
                parents[invocation_id] = outer.get_trace_id()
                ic = SimpleNamespace(agent=_agent(invocation_id), invocation_id=invocation_id)
                await plugin.before_run_callback(invocation_context=ic)
                await asyncio.sleep(0)  # overlap starts before either finalizes
                await plugin.after_run_callback(invocation_context=ic)

        async def drive():
            await asyncio.gather(run("first"), run("second"))

        asyncio.run(drive())
        assert len(seen) == 2
        assert {trace.agent_name: trace.parent_trace_id for trace in seen} == parents
        assert len({trace.id for trace in seen}) == 2

    def test_reinstrument_updates_new_invocations_without_moving_inflight_observers(self, monkeypatch):
        import decimalai.adk as adk

        runners = types.ModuleType("google.adk.runners")

        class Runner:
            def __init__(self, **kwargs):
                self.plugins = kwargs["plugins"]

        runners.Runner = Runner
        monkeypatch.setitem(sys.modules, "google.adk.runners", runners)
        first, second = [], []
        adk.instrument(agent_name="first", on_trace=first.append)
        plugin = Runner().plugins[0]
        original_patch = Runner.__init__
        ic1 = SimpleNamespace(agent=_agent(), invocation_id="first")
        ic2 = SimpleNamespace(agent=_agent(), invocation_id="second")

        async def drive():
            await plugin.before_run_callback(invocation_context=ic1)
            adk.instrument(agent_name="second", on_trace=second.append)
            await plugin.before_run_callback(invocation_context=ic2)
            await asyncio.gather(plugin.after_run_callback(invocation_context=ic2),
                                 plugin.after_run_callback(invocation_context=ic1))

        asyncio.run(drive())
        assert Runner.__init__ is original_patch
        assert Runner().plugins[0] is plugin
        assert [trace.agent_name for trace in first] == ["first"]
        assert [trace.agent_name for trace in second] == ["second"]
        assert {trace.agent_name for trace in _flushed_traces()} == {"first", "second"}


class TestDeliveredVersions:
    def test_a_delivery_hash_survives_worker_finalization_and_observer_mutation(self, use_router):
        from decimalai.adk import DecimalaiPlugin

        use_router(_VersionedRouter())
        seen = []

        def observe(trace):
            seen.append(trace.model_copy(deep=True))
            trace.skills_delivered_versions[0]["hash"] = "observer mutation"

        _drive_one_turn(
            DecimalaiPlugin(agent_name="support", enable_skill_loader=True, on_trace=observe),
            _FakeLlmRequest(CALLER_PROMPT),
        )
        sent = _flushed_traces()[0]
        expected = [{"name": "refund-policy", "hash": "a" * 64}]
        assert seen[0].skills_delivered_versions == sent.skills_delivered_versions == expected
        assert sent.model_dump(mode="json")["skills_delivered_versions"] == expected
        assert sent.active_skills == sent.skills_loaded_by_agent == []

    @pytest.mark.parametrize("failure", ["append", "router"])
    def test_failed_insertion_does_not_export_a_version_or_leak_into_next_run(self, use_router, failure):
        from decimalai.adk import DecimalaiPlugin

        class RefusingRequest(_FakeLlmRequest):
            def append_instructions(self, instructions):
                return []

        class FailingRouter(_VersionedRouter):
            def build_prompt_parts(self, **kwargs):
                self._rails()
                raise RuntimeError("body assembly failed")

        plugin = DecimalaiPlugin(agent_name="support", enable_skill_loader=True)
        agent = _agent()

        async def run(invocation_id, request):
            ic = SimpleNamespace(agent=agent, invocation_id=invocation_id, user_content=USER_QUESTION)
            cc = SimpleNamespace(invocation_id=invocation_id, agent_name=None)
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_agent_callback(agent=agent, callback_context=cc)
            await plugin.before_model_callback(callback_context=cc, llm_request=request)
            await plugin.after_agent_callback(agent=agent, callback_context=cc)

        async def drive():
            use_router(FailingRouter() if failure == "router" else _VersionedRouter())
            request = _FakeLlmRequest(CALLER_PROMPT) if failure == "router" else RefusingRequest(CALLER_PROMPT)
            await run("refused", request)
            # Keep the same async context: a new asyncio.run would isolate
            # the stale ContextVar by itself and hide a missing drain.
            use_router(_SplitRouter())  # an older router without a body hash
            await run("next", _FakeLlmRequest(CALLER_PROMPT))

        asyncio.run(drive())
        refused, next_trace = _flushed_traces()
        assert refused.skills_delivered == refused.skills_delivered_versions == []
        assert next_trace.skills_delivered == ["refund-policy"]
        assert next_trace.skills_delivered_versions == []

    def test_distinct_versions_in_one_invocation_are_retained_and_identical_deliveries_deduped(self, use_router):
        from decimalai.adk import DecimalaiPlugin

        # A third call repeats the newest version: it must not erase the old
        # version or duplicate the same witnessed name/hash pair.
        use_router(_VersionedRouter(["a" * 64, "b" * 64, "b" * 64]))
        plugin = DecimalaiPlugin(agent_name="support", enable_skill_loader=True)
        agent = _agent()
        ic = SimpleNamespace(agent=agent, invocation_id="multi", user_content=USER_QUESTION)
        cc = SimpleNamespace(invocation_id="multi", agent_name=None)

        async def drive():
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_agent_callback(agent=agent, callback_context=cc)
            for _ in range(3):
                await plugin.before_model_callback(callback_context=cc, llm_request=_FakeLlmRequest(CALLER_PROMPT))
            await plugin.after_agent_callback(agent=agent, callback_context=cc)

        asyncio.run(drive())
        trace = _flushed_traces()[0]
        assert trace.skills_delivered == ["refund-policy"]
        assert trace.skills_delivered_versions == [
            {"name": "refund-policy", "hash": "a" * 64},
            {"name": "refund-policy", "hash": "b" * 64},
        ]
        assert trace.active_skills == trace.skills_loaded_by_agent == []

    def test_concurrent_invocations_keep_separate_delivery_witnesses(self, use_router):
        from decimalai.adk import DecimalaiPlugin

        use_router(_VersionedRouter(["a" * 64, "b" * 64]))
        plugin = DecimalaiPlugin(enable_skill_loader=True)

        async def run(invocation_id):
            agent = _agent(invocation_id)
            ic = SimpleNamespace(agent=agent, invocation_id=invocation_id, user_content=USER_QUESTION)
            cc = SimpleNamespace(invocation_id=invocation_id, agent_name=None)
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_agent_callback(agent=agent, callback_context=cc)
            await plugin.before_model_callback(callback_context=cc, llm_request=_FakeLlmRequest(CALLER_PROMPT))
            await asyncio.sleep(0)
            await plugin.after_agent_callback(agent=agent, callback_context=cc)

        async def drive():
            await asyncio.gather(run("first"), run("second"))

        asyncio.run(drive())
        assert {trace.agent_name: trace.skills_delivered_versions for trace in _flushed_traces()} == {
            "first": [{"name": "refund-policy", "hash": "a" * 64}],
            "second": [{"name": "refund-policy", "hash": "b" * 64}],
        }


# ── layer 2: the real thing ─────────────────────────────────


@pytest.mark.skipif(
    not HAS_REAL_ADK,
    reason="google-adk is an optional extra (pip install 'decimalai[adk]'); "
           "the seam is still graded by TestSeamWithoutAdk on every run",
)
class TestRealAdkRun:
    """A real ``Runner.run_async`` over a real ``LlmAgent``.

    Gemini keys on this machine are quota-dead (429), so rather than call one,
    the model is a ``BaseLlm`` subclass that captures the outbound
    ``LlmRequest``. Everything upstream of the wire — ADK's request processors,
    the plugin manager, the flow — is the real code path.
    """

    @staticmethod
    def _build(router, *, tools=(), turns=None, on_trace=None, instrumented=False):
        from google.adk.agents import LlmAgent
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_response import LlmResponse
        from google.adk.runners import InMemoryRunner, Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types

        import decimalai.adk as adk

        adk._skill_router_singleton = router
        adk._skill_loader_enabled = True
        seen: List[Any] = []

        class _CapturingLlm(BaseLlm):
            async def generate_content_async(self, llm_request, stream=False):
                seen.append(llm_request)
                si = llm_request.config.system_instruction or ""
                step = len(seen) - 1
                if turns and step < len(turns):
                    yield LlmResponse(content=turns[step](types))
                    return
                # The C14 shape: the answer is the skill's fact only if the
                # skill's body was actually in front of the model.
                answer = "23.5%" if SENTINEL in si else "I do not know."
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text=answer)])
                )

        agent = LlmAgent(
            name="support_agent",
            model=_CapturingLlm(model="capturing-stub"),
            instruction=CALLER_PROMPT,
            tools=list(tools),
        )
        if instrumented:
            # Fleet uses global instrumentation and this synchronous runner,
            # rather than explicitly constructing the tracing plugin.
            adk.instrument(agent_name="support", enable_skill_loader=True, on_trace=on_trace)
            runner = InMemoryRunner(agent=agent, app_name="support")
            runner.session_service.create_session_sync(
                app_name="support", user_id="u1", session_id="s1",
            )
            out = []
            for ev in runner.run(
                user_id="u1", session_id="s1",
                new_message=types.Content(
                    role="user", parts=[types.Part(text=USER_QUESTION)],
                ),
            ):
                for part in (getattr(ev.content, "parts", None) or []) if ev.content else []:
                    if part.text:
                        out.append(part.text)
            return "".join(out), seen

        svc = InMemorySessionService()
        runner = Runner(
            agent=agent, app_name="support", session_service=svc,
            plugins=[adk.DecimalaiPlugin(agent_name="support", on_trace=on_trace)],
        )

        async def _run() -> str:
            await svc.create_session(app_name="support", user_id="u1", session_id="s1")
            out: List[str] = []
            async for ev in runner.run_async(
                user_id="u1", session_id="s1",
                new_message=types.Content(
                    role="user", parts=[types.Part(text=USER_QUESTION)]
                ),
            ):
                for part in (getattr(ev.content, "parts", None) or []) if ev.content else []:
                    if part.text:
                        out.append(part.text)
            return "".join(out)

        return asyncio.run(_run()), seen

    def test_the_sentinel_is_in_the_request_the_model_was_handed(self, use_router):
        router = use_router(_SplitRouter())
        answer, seen = self._build(router)

        assert seen, "the model was never called"
        si = seen[0].config.system_instruction
        assert isinstance(si, str)
        assert SENTINEL in si, (
            "ADK built a request with no skill body in it — the seam does not work"
        )
        assert si.index(CALLER_PROMPT) < si.index(SENTINEL) < si.index(TAIL)
        assert router.queries == [USER_QUESTION]
        # And the fact reached the answer, not just the payload.
        assert "23.5%" in answer

    def test_the_trace_of_a_real_run_carries_the_rails(self, use_router):
        self._build(use_router(_SplitRouter()))
        traces = _flushed_traces()
        assert traces, "no trace was sent for a real ADK run"
        trace = traces[-1]
        assert trace.routing_id == ROUTING_ID
        assert trace.skills_offered_in_prompt == ["refund-policy", "returns-window"]
        assert trace.skills_delivered == ["refund-policy"]

    @pytest.mark.parametrize("instrumented", [False, True], ids=["explicit-async", "fleet-sync"])
    def test_actual_get_body_hash_reaches_the_real_model_request_and_trace(self, use_router, monkeypatch, instrumented):
        from google.adk.runners import Runner

        import decimalai.adk as adk
        from decimalai.skill_router import SkillRouter

        # Register restoration before instrument() replaces the real constructor.
        monkeypatch.setattr(Runner, "__init__", Runner.__init__)
        if instrumented:
            # Exercise the factory Fleet's global instrumentation actually
            # uses, so an adapter-specific legacy cap cannot bypass this test.
            router = use_router(adk._get_skill_router())
            assert isinstance(router, SkillRouter)
        else:
            router = use_router(SkillRouter(
                api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=True,
            ))
        body = f"Opened boxes carry a 23.5% restocking fee. {SENTINEL}\n" + (
            "Verify the original order and apply the documented restocking fee.\n" * 220
        )
        digest = hashlib.sha256(body.encode()).hexdigest()
        monkeypatch.setattr(router, "smart_route", MagicMock(return_value={
            "prompt_fragment": "Available skill: refund-policy",
            "routing_id": ROUTING_ID,
            "skills": [{"name": "refund-policy"}],
            "stable_menu": "Available skill: refund-policy",
            "stable_menu_skills": ["refund-policy"],
            "routing_hint": TAIL,
        }))
        monkeypatch.setattr(router, "get_skill_body_record", MagicMock(return_value={
            "body": body, "version": 3, "content_hash": digest,
        }))
        observed = []
        from decimalai.generic import start_trace

        with start_trace(agent_name="outer", auto_send=False) as parent:
            parent_id = parent.get_trace_id()
            answer, requests = self._build(
                router, on_trace=observed.append, instrumented=instrumented,
            )
        assert "23.5%" in answer
        assert body in requests[0].config.system_instruction
        assert requests[0].config.system_instruction.index(body) < requests[0].config.system_instruction.index(TAIL)
        trace = _flushed_traces()[0]
        assert trace.parent_trace_id == observed[0].parent_trace_id == parent_id
        self._assert_retained_system_inputs(trace, observed, requests, [body], [TAIL])
        assert trace.skills_delivered_versions == [{"name": "refund-policy", "hash": digest, "routing_id": ROUTING_ID}]
        assert trace.active_skills == trace.skills_loaded_by_agent == []

    @pytest.mark.parametrize("instrumented", [False, True], ids=["explicit-async", "fleet-sync"])
    def test_two_real_model_calls_preserve_both_complete_body_versions(self, use_router, monkeypatch, instrumented):
        from google.adk.runners import Runner

        from decimalai.skill_router import SkillRouter

        monkeypatch.setattr(Runner, "__init__", Runner.__init__)
        router = use_router(SkillRouter(
            api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=True,
        ))
        first_body = f"Opened boxes carry a 23.5% restocking fee. {SENTINEL}\n" + (
            "Verify the original order and apply the documented restocking fee.\n" * 220
        )
        second_body = f"Opened boxes carry a 25% restocking fee. {SENTINEL}\n" + (
            "Apply the revised fee after verifying the original order.\n" * 25
        )
        first_hash = hashlib.sha256(first_body.encode()).hexdigest()
        second_hash = hashlib.sha256(second_body.encode()).hexdigest()
        first_route = {
            "prompt_fragment": "Available skill: refund-policy",
            "routing_id": ROUTING_ID,
            "skills": [{"name": "refund-policy"}],
            "stable_menu": "Available skill: refund-policy",
            "stable_menu_skills": ["refund-policy"],
            "routing_hint": TAIL,
        }
        second_id = "rt_" + "b" * 24
        second_tail = "Use refund-policy for the follow-up request."
        second_route = {**first_route, "routing_id": second_id, "routing_hint": second_tail}
        monkeypatch.setattr(router, "smart_route", MagicMock(side_effect=[first_route, second_route]))
        monkeypatch.setattr(router, "get_skill_body_record", MagicMock(side_effect=[
            {"body": first_body, "version": 3, "content_hash": first_hash},
            {"body": second_body, "version": 4, "content_hash": second_hash},
        ]))
        # A long-running tool can outlive the fragment cache; force that
        # expiry instead of waiting 30s or switching to a synthetic router.
        monkeypatch.setattr(router._fragment_cache, "get", lambda key: None)

        def lookup_order(order_id: str) -> dict:
            """Look up an order."""
            return {"order_id": order_id, "state": "opened"}

        def first_turn(types):
            return types.Content(role="model", parts=[types.Part(
                function_call=types.FunctionCall(name="lookup_order", args={"order_id": "A1"}),
            )])

        def second_turn(types):
            return types.Content(role="model", parts=[types.Part(text="25%")])

        observed = []
        _, requests = self._build(
            router, tools=[lookup_order], turns=[first_turn, second_turn],
            on_trace=observed.append, instrumented=instrumented,
        )
        assert len(requests) == 2
        assert first_body in requests[0].config.system_instruction
        assert TAIL in requests[0].config.system_instruction
        assert second_tail not in requests[0].config.system_instruction
        assert second_body in requests[1].config.system_instruction
        assert second_tail in requests[1].config.system_instruction
        trace = _flushed_traces()[0]
        self._assert_retained_system_inputs(
            trace, observed, requests, [first_body, second_body], [TAIL, second_tail],
        )
        assert trace.skills_delivered_versions == sorted([
            {"name": "refund-policy", "hash": first_hash, "routing_id": ROUTING_ID},
            {"name": "refund-policy", "hash": second_hash, "routing_id": second_id},
        ], key=lambda entry: (entry["name"], entry["hash"], entry["routing_id"]))
        assert trace.routing_id == second_id
        assert len(trace.llm_calls) == 2
        assert trace.active_skills == trace.skills_loaded_by_agent == []

    @staticmethod
    def _assert_retained_system_inputs(trace, observed, requests, bodies, tails):
        """Delivery metadata must remain gradeable against each actual prompt.

        An ADK trace can claim delivery while retaining only user turns. Check
        the complete exported and observer records against the real model
        requests, including bodies that exceed the trace preview limit.
        """
        exported = trace.model_dump(mode="json")
        # Exercise the real ingest serializer and httpx JSON encoding too:
        # observer equality alone misses a later exporter truncation.
        import json

        import httpx

        from decimalai._client import DecimalAIClient

        captured = []

        def capture(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "accepted"}, request=request)

        client = DecimalAIClient(api_key="dai_sk_test", base_url="http://sdk-proof.test")
        client._http.close()
        client._http = httpx.Client(
            base_url="http://sdk-proof.test", transport=httpx.MockTransport(capture),
        )
        try:
            client.ingest_trace(trace)
        finally:
            client._http.close()
        assert len(captured) == 1
        assert captured[0]["llm_calls"] == exported["llm_calls"]
        assert captured[0]["skills_delivered_versions"] == exported["skills_delivered_versions"]
        assert captured[0].get("parent_trace_id") == exported.get("parent_trace_id")
        assert len(observed) == 1
        assert observed[0].model_dump(mode="json")["llm_calls"] == exported["llm_calls"]
        assert len(exported["llm_calls"]) == len(requests) == len(bodies) == len(tails)
        assert len(bodies[0]) > 13_000, "the real body must exceed the former 8192-char cap"
        for call, request, body, tail in zip(exported["llm_calls"], requests, bodies, tails):
            model_system = request.config.system_instruction
            assert isinstance(model_system, str), "the real LlmAgent instruction shape changed"
            systems = [entry["content"] for entry in call["rendered_input"] if entry["role"] == "system"]
            assert systems == [model_system]
            assert len(body) > 500, "the full body must exceed the input_preview cap"
            assert "## Skill: refund-policy" in systems[0]
            assert body in systems[0]
            assert systems[0].index(body) < systems[0].index(tail)
            assert USER_QUESTION in str(call["rendered_input"])

    def test_a_tool_turn_keeps_the_body_and_the_query(self, use_router):
        """Two model turns in one invocation. The second one's contents end in
        a role="user" function_response with no text — the case the query
        extractor has to skip."""
        router = use_router(_SplitRouter())

        def lookup_order(order_id: str) -> dict:
            """Look up an order."""
            return {"order_id": order_id, "state": "opened"}

        def _turn0(types):
            return types.Content(role="model", parts=[types.Part(
                function_call=types.FunctionCall(
                    name="lookup_order", args={"order_id": "A1"}))])

        def _turn1(types):
            return types.Content(role="model", parts=[types.Part(text="23.5%")])

        observed = []
        _answer, seen = self._build(
            router, tools=[lookup_order], turns=[_turn0, _turn1], on_trace=observed.append,
        )
        assert len(seen) == 2, f"expected a tool round-trip, got {len(seen)} model turns"
        for step, req in enumerate(seen):
            assert SENTINEL in (req.config.system_instruction or ""), (
                f"model turn {step} of one invocation lost the skill body"
            )
        assert router.queries == [USER_QUESTION, USER_QUESTION], router.queries
        exported = _flushed_traces()
        assert len(observed) == len(exported) == 1
        assert observed[0].id == exported[0].id
        assert len(observed[0].llm_calls) == 2
        assert observed[0].skills_delivered == ["refund-policy"]
        assert observed[0].active_skills == []

    def test_the_synthetic_request_has_not_drifted(self):
        """Pin ``_FakeLlmRequest`` to the real ``LlmRequest``.

        The cheap layer above is only worth anything if its stand-in appends
        the way ADK does. Same three inputs, same resulting system instruction —
        including the non-str case, where BOTH must refuse.
        """
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types

        for start in (None, CALLER_PROMPT):
            real = LlmRequest(config=types.GenerateContentConfig(system_instruction=start))
            fake = _FakeLlmRequest(start)
            real.append_instructions([PREFIX, TAIL])
            fake.append_instructions([PREFIX, TAIL])
            assert real.config.system_instruction == fake.config.system_instruction

        # Non-str: real ADK logs a warning and drops the text.
        content = types.Content(role="user", parts=[types.Part(text=CALLER_PROMPT)])
        real = LlmRequest(config=types.GenerateContentConfig(system_instruction=content))
        real.append_instructions([PREFIX])
        assert PREFIX not in str(real.config.system_instruction), (
            "ADK started honouring a non-str system_instruction — "
            "decimalai/adk.py::_append_system_text's non-str branch needs a re-read"
        )

    def test_a_content_shaped_system_instruction_is_still_delivered(self):
        """The case ``append_instructions`` drops on the floor. The adapter's
        own branch has to carry it, or a caller who set
        ``GenerateContentConfig(system_instruction=Content(...))`` gets a
        perfectly traced run with none of their skills in it."""
        from google.adk.models.llm_request import LlmRequest
        from google.genai import types

        from decimalai.adk import _append_system_text, _system_instruction_text

        req = LlmRequest(
            config=types.GenerateContentConfig(
                system_instruction=types.Content(
                    role="user", parts=[types.Part(text=CALLER_PROMPT)]
                )
            )
        )
        assert _append_system_text(req, [PREFIX, TAIL]) is True
        flat = _system_instruction_text(req)
        assert flat.index(CALLER_PROMPT) < flat.index(SENTINEL) < flat.index(TAIL)


@pytest.mark.skipif(not HAS_REAL_ADK, reason="google-adk not installed")
class TestRealAdkParentLink:
    """The sync caller boundary must work before ADK starts its worker thread."""

    def test_explicit_plugin_and_global_instrument_share_one_idempotent_parent_patch(self, monkeypatch):
        from google.adk.runners import Runner

        import decimalai.adk as adk

        original = getattr(Runner.run, "__wrapped__", Runner.run)
        monkeypatch.setattr(Runner, "run", original)
        monkeypatch.setattr(Runner, "__init__", Runner.__init__)
        adk.DecimalaiPlugin()
        wrapped = Runner.run
        assert wrapped is not original
        assert wrapped.__wrapped__ is original
        adk.DecimalaiPlugin()
        adk.instrument()
        adk.instrument()
        assert Runner.run is wrapped

    @staticmethod
    def _runner(monkeypatch, observed, *, instrumented, explicit_parent=None, barrier=None):
        from google.adk.agents import LlmAgent
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_response import LlmResponse
        from google.adk.plugins.base_plugin import BasePlugin
        from google.adk.runners import InMemoryRunner, Runner
        from google.genai import types

        import decimalai.adk as adk

        configs = []

        class CaptureConfig(BasePlugin):
            def __init__(self):
                super().__init__(name="capture_config")

            async def before_run_callback(self, *, invocation_context):
                configs.append(invocation_context.run_config)

        class StubLlm(BaseLlm):
            async def generate_content_async(self, llm_request, stream=False):
                if barrier is not None:
                    # Each caller has its own event loop. Force both actual
                    # model invocations to overlap before either can finalize.
                    barrier.wait(timeout=10)
                yield LlmResponse(content=types.Content(
                    role="model", parts=[types.Part(text="Provider-free answer")],
                ))

        monkeypatch.setattr(Runner, "__init__", Runner.__init__)
        plugins = [CaptureConfig()]
        if instrumented:
            adk.instrument(agent_name="parent_probe", on_trace=observed.append)
        else:
            plugins.insert(0, adk.DecimalaiPlugin(
                agent_name="parent_probe", parent_trace_id=explicit_parent, on_trace=observed.append,
            ))
        runner = InMemoryRunner(
            agent=LlmAgent(name="parent_probe", model=StubLlm(model="capturing-stub"), instruction=CALLER_PROMPT),
            app_name="parent_probe", plugins=plugins,
        )
        return runner, configs

    @staticmethod
    def _run(runner, session_id, *, synchronous, run_config=None):
        from google.genai import types

        arguments = {
            "user_id": "u", "session_id": session_id,
            "new_message": types.Content(role="user", parts=[types.Part(text=session_id)]),
            "run_config": run_config,
        }
        if synchronous:
            return list(runner.run(**arguments))

        async def run():
            return [event async for event in runner.run_async(**arguments)]

        return asyncio.run(run())

    @staticmethod
    def _session(runner, session_id):
        runner.session_service.create_session_sync(app_name="parent_probe", user_id="u", session_id=session_id)

    @staticmethod
    def _assert_parents(observed, expected):
        import json

        import httpx

        from decimalai._client import DecimalAIClient

        sent = _flushed_traces()
        assert len(observed) == len(sent) == len(expected)
        assert {trace.user_input_preview: trace.parent_trace_id for trace in observed} == expected
        assert {trace.user_input_preview: trace.parent_trace_id for trace in sent} == expected
        assert {trace.id for trace in observed} == {trace.id for trace in sent}
        assert len({trace.id for trace in sent}) == len(expected)
        payloads = []

        def capture(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={"status": "accepted"}, request=request)

        client = DecimalAIClient(api_key="dai_sk_test", base_url="http://sdk-proof.test")
        client._http.close()
        client._http = httpx.Client(base_url="http://sdk-proof.test", transport=httpx.MockTransport(capture))
        try:
            for trace in sent:
                client.ingest_trace(trace)
        finally:
            client._http.close()
        assert {trace["user_input_preview"]: trace.get("parent_trace_id") for trace in payloads} == expected
        assert "_decimalai_invocation_parent_trace_id" not in json.dumps(payloads)

    @pytest.mark.parametrize("instrumented", [False, True], ids=["explicit-plugin", "global-instrument"])
    @pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
    def test_reused_runner_and_config_capture_each_parent_and_standalone_root(self, monkeypatch, instrumented, synchronous):
        from google.adk.agents.run_config import RunConfig

        from decimalai.generic import start_trace

        observed = []
        runner, configs = self._runner(monkeypatch, observed, instrumented=instrumented)
        config = RunConfig(max_llm_calls=5, custom_metadata={"caller": "preserved"})
        original = config.model_dump(mode="json")
        expected = {}
        for session_id in ("first_parent", "second_parent", "standalone"):
            self._session(runner, session_id)
            if session_id == "standalone":
                expected[session_id] = None
                self._run(runner, session_id, synchronous=synchronous, run_config=config)
            else:
                with start_trace(agent_name=session_id, auto_send=False) as outer:
                    expected[session_id] = outer.get_trace_id()
                    self._run(runner, session_id, synchronous=synchronous, run_config=config)
        self._assert_parents(observed, expected)
        assert config.model_dump(mode="json") == original
        assert not hasattr(config, "_decimalai_invocation_parent_trace_id")
        assert len(configs) == 3
        assert all(value.model_dump(mode="json") == original for value in configs)
        if synchronous:
            assert all(value is not config for value in configs)
            assert len({id(value) for value in configs}) == 3

    @pytest.mark.parametrize("instrumented", [False, True], ids=["explicit-plugin", "global-instrument"])
    @pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
    def test_concurrent_invocations_on_one_runner_keep_separate_parents(self, monkeypatch, instrumented, synchronous):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        from google.adk.agents.run_config import RunConfig

        from decimalai.generic import start_trace

        observed = []
        runner, configs = self._runner(monkeypatch, observed, instrumented=instrumented, barrier=Barrier(2))
        config = RunConfig(custom_metadata={"caller": "shared input config"})
        original = config.model_dump(mode="json")
        for session_id in ("concurrent_first", "concurrent_second"):
            self._session(runner, session_id)

        def invoke(session_id):
            with start_trace(agent_name=session_id, auto_send=False) as parent:
                parent_id = parent.get_trace_id()
                self._run(runner, session_id, synchronous=synchronous, run_config=config)
                return session_id, parent_id

        with ThreadPoolExecutor(max_workers=2) as callers:
            expected = dict(callers.map(invoke, ("concurrent_first", "concurrent_second")))
        self._assert_parents(observed, expected)
        assert len(set(expected.values())) == 2
        assert config.model_dump(mode="json") == original
        assert not hasattr(config, "_decimalai_invocation_parent_trace_id")
        if synchronous:
            assert len(configs) == 2 and configs[0] is not configs[1]
            assert all(value is not config for value in configs)

    @pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
    def test_explicit_plugin_parent_wins_over_caller_and_standalone(self, monkeypatch, synchronous):
        from uuid import uuid4

        from decimalai.generic import start_trace

        observed = []
        explicit = str(uuid4())
        runner, _ = self._runner(monkeypatch, observed, instrumented=False, explicit_parent=explicit)
        for session_id in ("explicit_nested", "explicit_standalone"):
            self._session(runner, session_id)
        with start_trace(agent_name="ignored_parent", auto_send=False) as outer:
            assert outer.get_trace_id() != explicit
            self._run(runner, "explicit_nested", synchronous=synchronous)
        self._run(runner, "explicit_standalone", synchronous=synchronous)
        self._assert_parents(observed, {"explicit_nested": explicit, "explicit_standalone": explicit})

    def test_deferred_sync_iterator_captures_parent_when_iteration_starts(self, monkeypatch):
        from google.genai import types

        from decimalai.generic import start_trace

        observed = []
        runner, _ = self._runner(monkeypatch, observed, instrumented=True)
        self._session(runner, "deferred")
        events = runner.run(user_id="u", session_id="deferred", new_message=types.Content(
            role="user", parts=[types.Part(text="deferred")],
        ))
        with start_trace(agent_name="iteration_parent", auto_send=False) as outer:
            expected = outer.get_trace_id()
            assert list(events)
        self._assert_parents(observed, {"deferred": expected})
