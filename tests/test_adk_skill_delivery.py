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
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }

    import decimalai.adk as adk

    saved = (
        adk._PluginClass,
        dict(adk._manifest_ids),
        dict(adk._manifest_trackers),
        adk._skill_loader_enabled,
        adk._skill_router_singleton,
    )
    adk._manifest_ids = {}
    adk._manifest_trackers = {}
    adk._skill_loader_enabled = False
    adk._skill_router_singleton = None

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

    (
        adk._PluginClass,
        adk._manifest_ids,
        adk._manifest_trackers,
        adk._skill_loader_enabled,
        adk._skill_router_singleton,
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
    def _build(router, *, tools=(), turns=None):
        from google.adk.agents import LlmAgent
        from google.adk.models.base_llm import BaseLlm
        from google.adk.models.llm_response import LlmResponse
        from google.adk.runners import Runner
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
        svc = InMemorySessionService()
        runner = Runner(
            agent=agent, app_name="support", session_service=svc,
            plugins=[adk.DecimalaiPlugin(agent_name="support")],
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

        _answer, seen = self._build(router, tools=[lookup_order], turns=[_turn0, _turn1])
        assert len(seen) == 2, f"expected a tool round-trip, got {len(seen)} model turns"
        for step, req in enumerate(seen):
            assert SENTINEL in (req.config.system_instruction or ""), (
                f"model turn {step} of one invocation lost the skill body"
            )
        assert router.queries == [USER_QUESTION, USER_QUESTION], router.queries

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
