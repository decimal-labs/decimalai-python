"""ADK adapter, Phase 1 (2026-10-14): no crash on ``App``, no blocked event loop.

google-adk 2.12.0 is the only supported version. Every test here drives the
REAL package — a real ``Runner`` / ``InMemoryRunner`` / ``App`` and a real
``LlmRequest`` — with a ``BaseLlm`` subclass that captures the outbound request
instead of sending it. No key, no network, no backend.

1. ``App`` is ADK's own way to attach plugins (``Runner(plugins=...)`` is
   deprecated, and ``adk web`` / ``adk run`` / ``api_server`` always build
   ``Runner(app=...)``). The global install used to pass ``plugins=`` on every
   construction, and ADK refuses ``app=`` together with ``plugins=``, so after
   ``init(adk=True)`` every App-shaped runner raised ``ValueError``.
2. ``before_model_callback`` is a coroutine on ADK's event loop. The skill
   router it calls is a synchronous HTTP round-trip, so calling it inline froze
   every other coroutine on that loop for the length of the request.
3. ADK 2.12's ``Runner.run`` carries the caller's contextvars into its worker
   thread itself, so the adapter no longer wraps ``Runner.run`` to smuggle the
   parent trace across that boundary.
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest


def _real_adk_installed() -> bool:
    # Asked at import time: sibling ADK test files stub `google.adk` in
    # sys.modules inside fixtures, and a stub must never satisfy this.
    try:
        return importlib.util.find_spec("google.adk.runners") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(
    not _real_adk_installed(), reason="google-adk is not installed in this environment",
)

CALLER_PROMPT = "You are a terse support agent."
SENTINEL = "SENTINEL-PHASE1-ADK-5d1e9b"
PREFIX = "### refund-policy\n" f"Opened boxes carry a restocking fee. {SENTINEL}\n"
TAIL = "Most relevant for this request: refund-policy."
ROUTING_ID = "rt_" + "c" * 24
BODY_HASH = "b" * 64
# Long enough that a blocked loop is unmistakable, short enough that the red
# run of this file stays cheap.
GATE_TIMEOUT_S = 3.0

_CALLER_MARKER: ContextVar[Optional[str]] = ContextVar("adk_phase1_caller_marker", default=None)


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def client(monkeypatch):
    """A fresh adapter, a mock DecimalAI client, and ADK's Runner restored after."""
    from google.adk.runners import Runner

    import decimalai
    import decimalai._config as cfg
    import decimalai.adk as adk
    from decimalai import skill_router as sr
    from decimalai._config import DecimalConfig

    # instrument() patches the constructor process-wide; put it back.
    monkeypatch.setattr(Runner, "__init__", Runner.__init__)
    monkeypatch.setattr(Runner, "run", Runner.run)
    mock = MagicMock()
    mock.register_manifest.return_value = {"manifest_id": "phase1-manifest", "status": "active"}
    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    ))
    monkeypatch.setattr(cfg, "_client", mock)
    monkeypatch.setattr(decimalai, "_global_client", mock, raising=False)
    for name, value in (
        ("_PluginClass", None),
        ("_manifest_ids", {}),
        ("_manifest_trackers", {}),
        ("_pending_manifests", {}),
        ("_skill_loader_enabled", False),
        ("_skill_router_singleton", None),
        ("_instrument_config", adk._InstrumentationConfig()),
        ("_runner_patched", False),
        ("_install_agent_name", None),
    ):
        monkeypatch.setattr(adk, name, value)
    sr._last_delivered_versions_ctx.set(None)
    yield mock
    sr._last_delivered_versions_ctx.set(None)


def _flushed(mock: MagicMock) -> List[Any]:
    from decimalai._config import _sender

    _sender.flush()
    return [call.args[0] for call in mock.ingest_trace.call_args_list]


def _stub_model(requests: List[Any]) -> Any:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    class CapturingLlm(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            requests.append(llm_request.config.system_instruction)
            yield LlmResponse(content=types.Content(
                role="model", parts=[types.Part(text="offline answer")],
            ))

    return CapturingLlm(model="offline-stub")


def _agent(name: str, requests: List[Any]) -> Any:
    from google.adk.agents import LlmAgent

    return LlmAgent(name=name, model=_stub_model(requests), instruction=CALLER_PROMPT)


async def _one_turn(runner: Any, text: str = "What fee applies to an opened box?") -> None:
    from google.genai import types

    session = await runner.session_service.create_session(app_name=runner.app_name, user_id="u")
    async for _ in runner.run_async(
        user_id="u", session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=text)]),
    ):
        pass


def _decimalai_plugins(runner: Any) -> List[Any]:
    import decimalai.adk as adk

    cls = adk._plugin_class()
    return [p for p in runner.plugin_manager.plugins if isinstance(p, cls)]


# ── 1. App-shaped runners ───────────────────────────────────


class TestAppShapedRunners:
    def test_init_adk_true_then_runner_with_app_constructs_and_traces_once(self, client, monkeypatch):
        """The audit's exact reproduction: ``init(adk=True)`` at import, then
        the Runner shape ADK's own dev tooling builds."""
        import decimalai
        import decimalai._config as cfg
        from google.adk.apps import App
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        decimalai.init(
            api_key="dai_sk_test", base_url="http://localhost:8000",
            adk=True, agent_name="app_probe", verify=False,
        )
        # init() built a real client; capture the export instead.
        monkeypatch.setattr(cfg, "_client", client)
        requests: List[Any] = []
        app = App(name="app_probe", root_agent=_agent("app_probe", requests))

        runner = Runner(app=app, session_service=InMemorySessionService())
        asyncio.run(_one_turn(runner))

        traces = _flushed(client)
        assert len(requests) == 1
        assert len(_decimalai_plugins(runner)) == 1
        assert [t.agent_name for t in traces] == ["app_probe"]

    def test_instrument_then_runner_with_app_constructs_and_traces_once(self, client):
        import decimalai.adk as adk
        from google.adk.apps import App
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        adk.instrument(agent_name="app_probe")
        requests: List[Any] = []
        runner = Runner(
            app=App(name="app_probe", root_agent=_agent("app_probe", requests)),
            session_service=InMemorySessionService(),
        )
        asyncio.run(_one_turn(runner))

        assert len(_decimalai_plugins(runner)) == 1
        assert [t.agent_name for t in _flushed(client)] == ["app_probe"]

    def test_instrument_then_in_memory_runner_with_app_constructs_and_traces_once(self, client):
        import decimalai.adk as adk
        from google.adk.apps import App
        from google.adk.runners import InMemoryRunner

        adk.instrument(agent_name="app_probe")
        requests: List[Any] = []
        runner = InMemoryRunner(app=App(name="app_probe", root_agent=_agent("app_probe", requests)))
        asyncio.run(_one_turn(runner))

        assert len(_decimalai_plugins(runner)) == 1
        assert [t.agent_name for t in _flushed(client)] == ["app_probe"]

    def test_app_plugins_decimalai_plugin_without_instrument_traces_once(self, client):
        """ADK's own non-deprecated way to add a plugin, with no global install."""
        import decimalai.adk as adk
        from google.adk.apps import App
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        requests: List[Any] = []
        runner = Runner(
            app=App(
                name="app_probe", root_agent=_agent("app_probe", requests),
                plugins=[adk.DecimalaiPlugin(agent_name="explicit_label")],
            ),
            session_service=InMemorySessionService(),
        )
        asyncio.run(_one_turn(runner))

        assert len(_decimalai_plugins(runner)) == 1
        assert [t.agent_name for t in _flushed(client)] == ["explicit_label"]

    def test_app_plugins_decimalai_plugin_with_instrument_is_not_doubled(self, client):
        """Global install AND an explicit plugin on the App: the explicit one
        answers for this runner, and the invocation is traced exactly once."""
        import decimalai.adk as adk
        from google.adk.apps import App
        from google.adk.runners import InMemoryRunner

        adk.instrument(agent_name="global_label")
        requests: List[Any] = []
        explicit = adk.DecimalaiPlugin(agent_name="explicit_label")
        runner = InMemoryRunner(app=App(
            name="app_probe", root_agent=_agent("app_probe", requests), plugins=[explicit],
        ))
        asyncio.run(_one_turn(runner))

        assert _decimalai_plugins(runner) == [explicit]
        assert [t.agent_name for t in _flushed(client)] == ["explicit_label"]

    @pytest.mark.parametrize("shape", ["app", "agent"])
    def test_a_custom_named_explicit_plugin_is_not_doubled_by_the_global_one(self, client, shape):
        """Presence is "a DecimalAI plugin is attached", not "a plugin named
        'decimalai' is attached": a renamed explicit plugin used to get the
        shared one added beside it, and every invocation shipped twice."""
        import decimalai.adk as adk
        from google.adk.apps import App
        from google.adk.runners import InMemoryRunner

        adk.instrument(agent_name="global_label")
        requests: List[Any] = []
        explicit = adk.DecimalaiPlugin(agent_name="explicit_label", name="decimalai_billing")
        if shape == "app":
            runner = InMemoryRunner(app=App(
                name="app_probe", root_agent=_agent("app_probe", requests), plugins=[explicit],
            ))
        else:
            runner = InMemoryRunner(
                agent=_agent("agent_probe", requests), app_name="agent_probe", plugins=[explicit],
            )
        asyncio.run(_one_turn(runner))

        assert _decimalai_plugins(runner) == [explicit]
        assert [t.agent_name for t in _flushed(client)] == ["explicit_label"]

    def test_the_callers_app_is_left_untouched_and_its_plugins_keep_their_order(self, client):
        import decimalai.adk as adk
        from google.adk.apps import App
        from google.adk.plugins.base_plugin import BasePlugin
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        class First(BasePlugin):
            def __init__(self):
                super().__init__(name="first")

        class Second(BasePlugin):
            def __init__(self):
                super().__init__(name="second")

        adk.instrument(agent_name="app_probe")
        requests: List[Any] = []
        first, second = First(), Second()
        app = App(name="app_probe", root_agent=_agent("app_probe", requests), plugins=[first, second])
        runner_a = Runner(app=app, session_service=InMemorySessionService())
        runner_b = Runner(app=app, session_service=InMemorySessionService())

        assert app.plugins == [first, second], "the caller's App was mutated"
        for runner in (runner_a, runner_b):
            plugins = runner.plugin_manager.plugins
            assert len(_decimalai_plugins(runner)) == 1
            assert plugins[1:] == [first, second]
            assert runner.app.root_agent is app.root_agent
            assert runner.app_name == "app_probe"

    def test_app_with_plugins_kwarg_is_still_adks_own_refusal(self, client):
        """Passing both is the caller's error. The patch must hand back ADK's
        own message, not swallow it and not replace it with a different one."""
        import decimalai.adk as adk
        from google.adk.apps import App
        from google.adk.plugins.base_plugin import BasePlugin
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        class Extra(BasePlugin):
            def __init__(self):
                super().__init__(name="extra")

        app = App(name="app_probe", root_agent=_agent("app_probe", []))
        with pytest.raises(ValueError) as native:
            Runner(app=app, plugins=[Extra()], session_service=InMemorySessionService())
        adk.instrument(agent_name="app_probe")
        with pytest.raises(ValueError) as patched:
            Runner(app=app, plugins=[Extra()], session_service=InMemorySessionService())
        assert str(patched.value) == str(native.value)

    def test_the_agent_shaped_runner_still_gets_the_shared_plugin(self, client):
        """The path that already worked keeps working."""
        import decimalai.adk as adk
        from google.adk.runners import InMemoryRunner

        adk.instrument(agent_name="agent_probe")
        runner = InMemoryRunner(agent=_agent("agent_probe", []), app_name="agent_probe")
        asyncio.run(_one_turn(runner))

        assert len(_decimalai_plugins(runner)) == 1
        assert [t.agent_name for t in _flushed(client)] == ["agent_probe"]


# ── 2. the event loop is never blocked by the router ────────


class _GatedRouter:
    """A router whose HTTP call cannot finish until a coroutine on the
    caller's event loop releases it — so it can only ever finish in time if
    the loop is free while the call is in flight."""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.entered = threading.Event()
        self.released: List[bool] = []
        self.thread_ids: List[int] = []
        self.caller_markers: List[Optional[str]] = []

    def build_prompt_parts(self, query=None, *, agent_name=None, scope=None, **kw):
        from decimalai import skill_router as sr

        self.thread_ids.append(threading.get_ident())
        self.caller_markers.append(_CALLER_MARKER.get())
        self.entered.set()
        self.released.append(self.gate.wait(timeout=GATE_TIMEOUT_S))
        sr._last_offered_names_ctx.set(["refund-policy", "returns-window"])
        sr._last_delivered_names_ctx.set(["refund-policy"])
        sr._last_delivered_versions_ctx.set([{"name": "refund-policy", "hash": BODY_HASH}])
        return PREFIX, TAIL, ROUTING_ID


async def _release_once_entered(router: _GatedRouter) -> None:
    # Polls with real awaits: it can only observe `entered` and open the gate
    # if the loop gets control back while the router call is still running.
    while not router.entered.is_set():
        await asyncio.sleep(0.005)
    router.gate.set()


class TestRouterRunsOffTheEventLoop:
    def test_a_slow_router_call_does_not_block_a_concurrent_coroutine(self, client, monkeypatch):
        import decimalai.adk as adk
        from google.adk.models.llm_request import LlmRequest
        from google.adk.models.llm_response import LlmResponse
        from google.genai import types

        router = _GatedRouter()
        monkeypatch.setattr(adk, "_skill_router_singleton", router)
        plugin = adk.DecimalaiPlugin(agent_name="loop_probe", enable_skill_loader=True)
        agent = _agent("loop_probe", [])
        request = LlmRequest(
            model="offline-stub",
            contents=[types.Content(role="user", parts=[types.Part(text="opened box fee?")])],
            config=types.GenerateContentConfig(system_instruction=CALLER_PROMPT),
        )
        ic = SimpleNamespace(agent=agent, invocation_id="inv-loop", user_content=None, run_config=None)
        cc = SimpleNamespace(invocation_id="inv-loop", agent_name="loop_probe")
        loop_thread: List[int] = []

        async def go():
            loop_thread.append(threading.get_ident())
            _CALLER_MARKER.set("set-on-the-loop")
            await plugin.before_run_callback(invocation_context=ic)
            await plugin.before_agent_callback(agent=agent, callback_context=cc)
            releaser = asyncio.create_task(_release_once_entered(router))
            await plugin.before_model_callback(callback_context=cc, llm_request=request)
            await releaser
            await plugin.after_model_callback(callback_context=cc, llm_response=LlmResponse(
                content=types.Content(role="model", parts=[types.Part(text="offline answer")]),
            ))
            await plugin.after_agent_callback(agent=agent, callback_context=cc)

        asyncio.run(go())

        assert router.released == [True], (
            "the router call held the event loop: a coroutine scheduled beside "
            "it could not run until the call gave up"
        )
        assert router.thread_ids and router.thread_ids[0] != loop_thread[0]
        # asyncio.to_thread copies the caller's context into the worker.
        assert router.caller_markers == ["set-on-the-loop"]
        # Delivery still lands on the request and is still witnessed.
        assert SENTINEL in request.config.system_instruction
        trace = _flushed(client)[0]
        assert trace.routing_id == ROUTING_ID
        assert trace.skills_offered_in_prompt == ["refund-policy", "returns-window"]
        assert trace.skills_delivered == ["refund-policy"]
        assert trace.skills_delivered_versions == [{"name": "refund-policy", "hash": BODY_HASH}]
        systems = [m["content"] for m in trace.llm_calls[0].rendered_input if m["role"] == "system"]
        assert SENTINEL in systems[0], "the rendered input missed the routed body"

    def test_a_real_turn_with_a_slow_router_keeps_the_loop_responsive(self, client, monkeypatch):
        import decimalai.adk as adk
        from google.adk.runners import InMemoryRunner

        router = _GatedRouter()
        monkeypatch.setattr(adk, "_skill_router_singleton", router)
        adk.instrument(agent_name="loop_probe", enable_skill_loader=True)
        requests: List[Any] = []
        runner = InMemoryRunner(agent=_agent("loop_probe", requests), app_name="loop_probe")

        async def go():
            releaser = asyncio.create_task(_release_once_entered(router))
            await _one_turn(runner)
            await releaser

        asyncio.run(go())

        assert router.released == [True], "the router call held ADK's event loop"
        assert len(requests) == 1 and SENTINEL in requests[0]
        trace = _flushed(client)[0]
        assert trace.skills_delivered == ["refund-policy"]
        assert trace.routing_id == ROUTING_ID

    def test_a_router_that_raises_off_the_loop_still_degrades_to_an_unskilled_turn(
        self, client, monkeypatch,
    ):
        import decimalai.adk as adk
        from decimalai import skill_router as sr
        from google.adk.runners import InMemoryRunner

        class Exploding:
            def build_prompt_parts(self, **kw):
                # A partial witness set before the failure must not survive it.
                sr._last_delivered_versions_ctx.set([{"name": "refund-policy", "hash": BODY_HASH}])
                raise ConnectionError("router offline")

        monkeypatch.setattr(adk, "_skill_router_singleton", Exploding())
        adk.instrument(agent_name="loop_probe", enable_skill_loader=True)
        requests: List[Any] = []
        runner = InMemoryRunner(agent=_agent("loop_probe", requests), app_name="loop_probe")
        asyncio.run(_one_turn(runner))

        assert len(requests) == 1 and requests[0].startswith(CALLER_PROMPT)
        assert SENTINEL not in requests[0]
        trace = _flushed(client)[0]
        assert trace.skills_offered_in_prompt == []
        assert trace.skills_delivered == []
        assert trace.skills_delivered_versions == []
        assert trace.routing_id is None


# ── 3. no compatibility shims for google-adk < 2.12 ─────────


class TestNoShimsBelowTheFloor:
    def test_neither_entry_point_wraps_runner_run(self):
        """ADK 2.12's Runner.run runs its worker thread in a copy of the
        caller's context (runners.py, ``contextvars.copy_context().run``), so
        the parent trace reaches before_run without help."""
        import decimalai.adk as adk
        from google.adk.runners import Runner

        original = Runner.run
        adk.DecimalaiPlugin()
        adk.instrument()
        assert Runner.run is original

    def test_a_sync_run_still_links_its_parent_through_adks_own_context_copy(self, client):
        import decimalai.adk as adk
        from decimalai.generic import start_trace
        from google.adk.runners import InMemoryRunner
        from google.genai import types

        adk.instrument(agent_name="sync_probe")
        runner = InMemoryRunner(agent=_agent("sync_probe", []), app_name="sync_probe")
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
        traces = _flushed(client)
        assert {t.user_input_preview: t.parent_trace_id for t in traces} == {
            "hello": parent_id, "standalone": None,
        }
