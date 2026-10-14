"""LangChain Phase 1: route as the agent the trace is filed under, and never
block the event loop to do it.

Two defects, both silent.

1. **Routing name != trace name.** The skill injection sent
   ``agent_name=_install_agent_name`` to the Router, and that is ``None``
   unless ``instrument(agent_name=...)`` was called. The trace, meanwhile, is
   filed under handler name -> name detected from the root chain -> install
   name. So ``CallbackHandler(agent_name="joke-bot")`` (the docs' per-call form)
   and every named ``create_agent(name=...)`` graph shipped traces under one
   agent while routing as no agent at all: agent-scope assignments, version
   pins and restricted skills for that agent were never offered, and nothing
   said so.

2. **The async entry points ran the synchronous Router on the event loop.**
   ``ainvoke`` / ``astream`` called the blocking HTTP client inline, so one
   slow routing call (5 s route budget, 30 s body budget) stalled every other
   coroutine in the process.

Every test here drives real LangChain runnables (``create_agent`` graphs, bare
model calls, threads and asyncio), because both seams live in how LangChain
propagates config and context, which a hand-called callback sequence does not
reproduce.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain")

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel

SKILLS_BY_AGENT = {
    None: ["ambient-skill"],
    "agent-a": ["alpha"],
    "agent-b": ["beta"],
}
ROUTING_IDS = {
    None: "rt_" + "0" * 24,
    "agent-a": "rt_" + "a" * 24,
    "agent-b": "rt_" + "b" * 24,
}


class _RecordingRouter:
    """A real ``SkillRouter`` whose network is a canned, per-agent menu.

    Subclassing the real class keeps everything under ``build_prompt_parts``
    real: the cache key, the ambient run scope, the scoped rails, the
    ContextVar rails. Only ``_request`` is replaced, and it records what the
    adapter asked for and from where.
    """

    def __new__(cls, *, on_request=None, **kwargs):
        from decimalai.skill_router import SkillRouter, _ambient_scope

        class _R(SkillRouter):
            def _request(self, method, path, json=None, params=None, **_kw):
                agent = (params or {}).get("agent_name")
                self.requests.append({
                    "path": path,
                    "agent_name": agent,
                    "thread": threading.get_ident(),
                    "scope": _ambient_scope(),
                })
                if on_request is not None:
                    on_request(agent)
                if path.endswith("/skills/menu"):
                    names = SKILLS_BY_AGENT.get(agent, SKILLS_BY_AGENT[None])
                    return {
                        "skills": [{"name": n, "description": f"{n} desc"} for n in names],
                        "prompt_fragment": "## Skills\n" + "\n".join(
                            f"- {n}: {n} desc" for n in names
                        ),
                        "strategy": "menu",
                        "routing_id": ROUTING_IDS.get(agent, ROUTING_IDS[None]),
                    }
                return {}

        router = _R(**kwargs)
        router.requests = []
        return router


@pytest.fixture(autouse=True)
def sdk(monkeypatch):
    import decimalai._config as cfg
    import decimalai.langchain as lc
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "mf_test"}
    cfg._client.list_manifests.return_value = {"manifests": []}
    cfg._sender._pending = []
    monkeypatch.setattr(lc, "_manifest_ids", {})
    monkeypatch.setattr(lc, "_manifest_hashes", {})
    monkeypatch.setattr(lc, "_manifest_adoption_probed", set(), raising=False)
    monkeypatch.setattr(lc, "_explicit_manifest_config", None)
    monkeypatch.setattr(lc, "_install_agent_name", None)
    # A handler an earlier module published process-wide would trace these
    # runs a second time AND answer the routing-name fallback.
    token = lc._decimal_callback_var.set(None)
    # A bare `llm.invoke()` on the test thread routes in pytest's own Context,
    # so the Router's per-turn body budget would outlive the test and be
    # inherited by every later test's runs (a later `load_skill("alpha")` is
    # then refused as already loaded). Put them back.
    import decimalai.skill_router as sr

    router_vars = (
        sr._body_budget_ctx, sr._last_split_parts_ctx, sr._last_offered_names_ctx,
        sr._last_delivered_names_ctx, sr._last_delivered_versions_ctx,
    )
    router_tokens = [var.set(None) for var in router_vars]
    yield
    for var, router_token in zip(router_vars, router_tokens):
        var.reset(router_token)
    lc._decimal_callback_var.reset(token)
    cfg._config = None
    cfg._client = None


def _install_router(monkeypatch, **kwargs):
    import decimalai.langchain as lc

    router = _RecordingRouter(
        api_key="dai_sk_test", base_url="http://localhost:8000", strategy="menu",
        **kwargs,
    )
    monkeypatch.setattr(lc, "_skill_router_singleton", router)
    return router


@pytest.fixture
def router(monkeypatch):
    return _install_router(monkeypatch)


@pytest.fixture
def loader(monkeypatch):
    """The real invoke/ainvoke/stream/astream patch, undone at teardown."""
    import decimalai.langchain as lc

    for name in ("invoke", "ainvoke", "stream", "astream"):
        monkeypatch.setattr(BaseChatModel, name, getattr(BaseChatModel, name))
    monkeypatch.setattr(lc, "_skill_loader_installed", False)
    lc._install_skill_loader()
    yield


def _sent_traces() -> Dict[str, Any]:
    import decimalai._config as cfg

    cfg._sender.flush()
    return {
        c[1][0].agent_name: c[1][0]
        for c in cfg._client.method_calls
        if c[0] == "ingest_trace"
    }


def _menu_requests(router) -> List[Dict[str, Any]]:
    return [r for r in router.requests if r["path"].endswith("/skills/menu")]


def _graph(name: Optional[str] = None):
    llm = FakeListChatModel(responses=["done"] * 8)
    return create_agent(llm, tools=[], system_prompt="You are helpful.", name=name)


def _user(text: str) -> Dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


# ── 1. routing uses the name the trace is filed under ────────────────────────


class TestRoutingNameIsTheTraceName:
    def test_per_call_handler_name_routes_a_bare_invoke(self, router, loader):
        """The docs' per-call form: ``CallbackHandler(agent_name=...)``."""
        from decimalai.langchain import CallbackHandler

        llm = FakeListChatModel(responses=["ok"])
        llm.invoke(
            "refund my order",
            config={"callbacks": [CallbackHandler(agent_name="joke-bot")]},
        )

        [request] = _menu_requests(router)
        assert request["agent_name"] == "joke-bot", (
            "the trace is filed under 'joke-bot' but routing asked for "
            f"{request['agent_name']!r}: that agent's assigned, pinned and "
            "restricted skills were never offered"
        )
        assert set(_sent_traces()) == {"joke-bot"}

    def test_detected_graph_name_routes_a_create_agent_run(self, router, loader):
        """No name anywhere but the graph's own: the trace detects it, so
        routing must too."""
        from decimalai.langchain import CallbackHandler

        _graph("refund-agent").invoke(
            _user("refund my order"), config={"callbacks": [CallbackHandler()]},
        )

        [request] = _menu_requests(router)
        assert set(_sent_traces()) == {"refund-agent"}
        assert request["agent_name"] == "refund-agent"

    def test_handler_name_beats_the_detected_name(self, router, loader):
        """Same order as the trace: handler first, detection second."""
        from decimalai.langchain import CallbackHandler

        _graph("refund-agent").invoke(
            _user("refund my order"),
            config={"callbacks": [CallbackHandler(agent_name="joke-bot")]},
        )

        [request] = _menu_requests(router)
        assert set(_sent_traces()) == {"joke-bot"}
        assert request["agent_name"] == "joke-bot"

    def test_install_name_is_the_fallback(self, router, loader, monkeypatch):
        """Nothing on the run names it: ``instrument(agent_name=...)`` does."""
        import decimalai.langchain as lc
        from decimalai.langchain import CallbackHandler

        monkeypatch.setattr(lc, "_install_agent_name", "support")
        FakeListChatModel(responses=["ok"]).invoke(
            "refund my order", config={"callbacks": [CallbackHandler()]},
        )

        [request] = _menu_requests(router)
        assert request["agent_name"] == "support"
        assert set(_sent_traces()) == {"support"}

    def test_no_name_anywhere_still_routes_unscoped(self, router, loader):
        """A run that nothing names keeps the org-wide menu (agent_name None),
        and its trace keeps the ``langchain-agent`` label."""
        from decimalai.langchain import DEFAULT_AGENT_NAME, CallbackHandler

        FakeListChatModel(responses=["ok"]).invoke(
            "refund my order", config={"callbacks": [CallbackHandler()]},
        )

        [request] = _menu_requests(router)
        assert request["agent_name"] is None
        assert set(_sent_traces()) == {DEFAULT_AGENT_NAME}

    def test_global_handler_routes_under_the_detected_name(
        self, router, loader, monkeypatch,
    ):
        """``instrument()`` with no name: the process-wide handler detects the
        graph's name for the trace, and routing must follow it."""
        import decimalai.langchain as lc

        _instrument(monkeypatch, None)
        try:
            _graph("refund-agent").invoke(_user("refund my order"))
            traces = _sent_traces()
        finally:
            _unpublish()

        [request] = _menu_requests(router)
        assert set(traces) == {"refund-agent"}
        assert request["agent_name"] == "refund-agent"
        assert lc._global_handler.agent_name is None, "detection leaked onto the handler"

    def test_global_install_name_beats_detection(self, router, loader, monkeypatch):
        """``instrument(agent_name=...)`` names the global handler itself, so
        it wins over the graph name, exactly as on the trace."""
        _instrument(monkeypatch, "support")
        try:
            _graph("refund-agent").invoke(_user("refund my order"))
            traces = _sent_traces()
        finally:
            _unpublish()

        [request] = _menu_requests(router)
        assert set(traces) == {"support"}
        assert request["agent_name"] == "support"

    def test_concurrent_threads_route_under_their_own_names(
        self, monkeypatch, loader,
    ):
        """Two graphs in flight at once, through one handler type: each
        routing call is resolved from ITS run, never from "the latest run"."""
        from decimalai.langchain import CallbackHandler

        both_routing = threading.Barrier(2, timeout=10)
        router = _install_router(monkeypatch, on_request=lambda _a: both_routing.wait())

        def run(name: str):
            _graph(name).invoke(
                _user(f"question for {name}"),
                config={"callbacks": [CallbackHandler()]},
            )

        threads = [threading.Thread(target=run, args=(n,)) for n in ("agent-a", "agent-b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)

        assert sorted((r["agent_name"] for r in _menu_requests(router)), key=str) == [
            "agent-a", "agent-b",
        ]
        traces = _sent_traces()
        assert set(traces) == {"agent-a", "agent-b"}
        assert traces["agent-a"].routing_id == ROUTING_IDS["agent-a"]
        assert traces["agent-a"].skills_offered_in_prompt == ["alpha"]
        assert traces["agent-b"].routing_id == ROUTING_IDS["agent-b"]
        assert traces["agent-b"].skills_offered_in_prompt == ["beta"]


def _instrument(monkeypatch, agent_name: Optional[str]) -> None:
    """Run the real ``instrument()`` and restore every global it writes."""
    import decimalai.langchain as lc

    for attr in (
        "_installed", "_install_agent_name", "_skill_body_top_k", "_priority_skills",
        "_runtime_policy", "_evals", "_builtin_evals_enabled",
    ):
        monkeypatch.setattr(lc, attr, getattr(lc, attr))
    monkeypatch.setattr(lc._global_handler, "on_trace", lc._global_handler.on_trace)
    lc._installed = False
    lc.instrument(agent_name=agent_name, disk_sync=False)


def _unpublish() -> None:
    import decimalai.langchain as lc

    lc._decimal_callback_var.set(None)
    lc._global_handler.agent_name = None
    lc._global_handler.reset()


# ── 2. the async entry points route off the event loop ──────────────────────


async def _while_router_blocks(entered: threading.Event, released: threading.Event):
    """Set ``released`` from the event loop once the router is inside its
    request. If the router is running ON the loop, this never gets to run."""
    while not entered.is_set():
        await asyncio.sleep(0.005)
    released.set()


def _blocking_router(monkeypatch):
    entered, released = threading.Event(), threading.Event()
    outcome: Dict[str, Any] = {}

    def on_request(_agent):
        entered.set()
        outcome["released_in_time"] = released.wait(2.0)

    return _install_router(monkeypatch, on_request=on_request), entered, released, outcome


class TestAsyncRoutingRunsOffTheLoop:
    def test_ainvoke_does_not_block_the_loop(self, monkeypatch, loader):
        import decimalai.langchain as lc
        from decimalai.langchain import CallbackHandler

        router, entered, released, outcome = _blocking_router(monkeypatch)
        llm = FakeListChatModel(responses=["ok"])
        call_rails: Dict[str, Any] = {}

        class _CallRailProbe(CallbackHandler):
            """Reads the per-call rails where the handler reads them: inside
            the model call's own callbacks. Without the copy-back from the
            worker thread they are empty here, and the trace only gets its
            decision from the Router's last-resort scoped drain."""

            def on_chat_model_start(self, *args, **kwargs):
                call_rails["routing_id"] = lc._routing_id_ctx.get()
                call_rails["offered"] = sorted(lc._skills_offered_ctx.get() or ())
                return super().on_chat_model_start(*args, **kwargs)

        async def main():
            loop_thread = threading.get_ident()
            waiter = asyncio.create_task(_while_router_blocks(entered, released))
            await llm.ainvoke(
                "refund my order",
                config={"callbacks": [_CallRailProbe(agent_name="agent-a")]},
            )
            await waiter
            return loop_thread

        loop_thread = asyncio.run(main())

        assert outcome.get("released_in_time") is True, (
            "the event loop could not run a single other task while the "
            "router was inside its HTTP call"
        )
        [request] = _menu_requests(router)
        assert request["thread"] != loop_thread
        assert call_rails == {"routing_id": ROUTING_IDS["agent-a"], "offered": ["alpha"]}
        trace = _sent_traces()["agent-a"]
        assert trace.routing_id == ROUTING_IDS["agent-a"]
        assert trace.skills_offered_in_prompt == ["alpha"]

    def test_astream_does_not_block_the_loop(self, monkeypatch, loader):
        """``astream`` on a model with a native ``_stream`` is its own entry
        point, and had the same inline call."""
        from decimalai.langchain import CallbackHandler

        router, entered, released, outcome = _blocking_router(monkeypatch)
        llm = FakeListChatModel(responses=["streamed"])
        assert type(llm)._stream is not BaseChatModel._stream  # really streams

        async def main():
            waiter = asyncio.create_task(_while_router_blocks(entered, released))
            chunks = [
                c async for c in llm.astream(
                    "refund my order",
                    config={"callbacks": [CallbackHandler(agent_name="agent-b")]},
                )
            ]
            await waiter
            return chunks

        chunks = asyncio.run(main())

        assert chunks
        assert outcome.get("released_in_time") is True
        [request] = _menu_requests(router)
        assert request["agent_name"] == "agent-b"
        trace = _sent_traces()["agent-b"]
        assert trace.routing_id == ROUTING_IDS["agent-b"]
        assert trace.skills_offered_in_prompt == ["beta"]

    def test_two_async_runs_route_concurrently_and_keep_their_own_rails(
        self, monkeypatch, loader,
    ):
        """Both routing calls must be in flight at the same time (only
        possible off the loop), and each must still be filed under its own
        run: the ambient scope seen inside the worker thread is the run's own
        LangChain id, and each trace carries only its own decision."""
        from decimalai.langchain import CallbackHandler

        both_routing = threading.Barrier(2, timeout=3)
        router = _install_router(monkeypatch, on_request=lambda _a: both_routing.wait())
        model_parents: Dict[str, str] = {}

        class _Recording(CallbackHandler):
            def on_chat_model_start(self, serialized, messages, *, run_id=None,
                                    parent_run_id=None, **kwargs):
                model_parents[self._agent_name_or_default(
                    self._state_for(run_id, parent_run_id),
                )] = str(parent_run_id)
                return super().on_chat_model_start(
                    serialized, messages, run_id=run_id,
                    parent_run_id=parent_run_id, **kwargs,
                )

        async def main():
            await asyncio.gather(*(
                _graph(name).ainvoke(
                    _user(f"question for {name}"), config={"callbacks": [_Recording()]},
                )
                for name in ("agent-a", "agent-b")
            ))

        asyncio.run(main())

        requests = {r["agent_name"]: r for r in _menu_requests(router)}
        assert set(requests) == {"agent-a", "agent-b"}
        for name in ("agent-a", "agent-b"):
            assert requests[name]["scope"] == model_parents[name], (
                f"{name}'s routing decision was filed under "
                f"{requests[name]['scope']!r}, not its own model node's run id"
            )
        traces = _sent_traces()
        assert traces["agent-a"].routing_id == ROUTING_IDS["agent-a"]
        assert traces["agent-a"].skills_offered_in_prompt == ["alpha"]
        assert traces["agent-b"].routing_id == ROUTING_IDS["agent-b"]
        assert traces["agent-b"].skills_offered_in_prompt == ["beta"]

    def test_router_context_writes_still_reach_the_caller(self, router, loader):
        """Moving the call to a thread must not lose what it wrote into the
        caller's Context. The per-turn body budget is the one a later
        ``router.load_skill()`` in the same coroutine reads; inline it landed
        in the caller's Context, and it still must."""
        import decimalai.skill_router as sr

        llm = FakeListChatModel(responses=["ok"])

        async def main():
            sr._body_budget_ctx.set(None)
            await llm.ainvoke("refund my order")
            return sr._body_budget_ctx.get()

        assert asyncio.run(main()) is not None

    def test_sync_invoke_still_routes_on_the_calling_thread(self, router, loader):
        """Only the async entry points hop threads; a sync caller keeps its
        call on its own thread."""
        FakeListChatModel(responses=["ok"]).invoke("refund my order")

        [request] = _menu_requests(router)
        assert request["thread"] == threading.get_ident()
