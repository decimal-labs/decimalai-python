"""One LangChain run's skill rails must land on THAT run's trace.

What was wrong
--------------
``decimalai.langchain._drain_router_rails()`` read the ``SkillRouter``
singleton's instance rails with no scope at all, and ``build_trace`` used
whatever came back whenever the run had captured nothing of its own. Those
rails are process-global and clear-on-read, so under concurrency the first
trace to send took every lane's names:

* run A's user tool calls ``router.load_skill("alpha")``; run B calls
  nothing; B ships first, and ``alpha`` is stamped on **B's**
  ``skills_loaded_by_agent`` while A's real load is dropped. That is a
  fabricated ACTIVATION — the one rung on the ladder that is supposed to
  mean "the model reached for this skill" — plus a lost true one.
* a run that never called a model at all still claimed the routing_id and
  the offered set of a concurrent run that did.

Why every test here gives the two runs DIFFERENT skills
-------------------------------------------------------
Same reason ``tests/test_skill_rail_per_run.py`` does it: if both lanes are
offered the same names, an implementation that hands every trace the union
of everything still satisfies a set-equality assertion. Only different names
per lane can tell "each run got its own" apart from "everyone got everything".

These tests drive REAL LangChain runnables (threads and asyncio both), not a
hand-called sequence of callbacks, because the seam under test is exactly the
one a hand-called sequence has no version of: LangChain dispatches this
handler's callbacks under ``copy_context()``, which is why the ContextVar
rails could not carry the answer and the unscoped singleton drain existed in
the first place.
"""

from __future__ import annotations

import asyncio
import threading
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda

SKILL_BODIES = {
    "alpha": "ALPHA BODY LINE that only the alpha body carries.",
    "beta": "BETA BODY LINE that only the beta body carries.",
}


class _CannedRouter:
    """A real ``SkillRouter`` with the network replaced by a canned payload.

    Subclassing the real class (rather than stubbing ``get_menu`` or
    ``build_prompt_fragment``) is deliberate: the rails under test — the
    per-scope stores, the unscoped mirrors, the fragment cache key — all live
    BELOW those methods, so a stub above them would test nothing.
    """

    def __new__(cls, **kwargs):
        from decimalai.skill_router import SkillRouter

        class _R(SkillRouter):
            def _request(self, method, path, json=None, params=None):
                if path.endswith("/skills/menu"):
                    return {
                        "skills": [
                            {"name": n, "description": f"{n} desc"}
                            for n in SKILL_BODIES
                        ],
                        "prompt_fragment": "## Skills\n"
                        + "\n".join(f"- {n}: {n} desc" for n in SKILL_BODIES),
                        "strategy": "menu",
                        "routing_id": "rt_" + "0" * 24,
                    }
                if "/body" in path:
                    name = path.rsplit("/", 2)[-2]
                    return {"body": SKILL_BODIES.get(name, "")}
                return {}

        return _R(**kwargs)


@pytest.fixture(autouse=True)
def sdk(monkeypatch):
    import decimalai._config as cfg
    import decimalai.langchain as lc_mod
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "mf_test"}
    cfg._client.list_manifests.return_value = {"manifests": []}
    cfg._sender._pending = []
    monkeypatch.setattr(lc_mod, "_manifest_ids", {})
    monkeypatch.setattr(lc_mod, "_manifest_hashes", {})
    monkeypatch.setattr(lc_mod, "_manifest_adoption_probed", set(), raising=False)
    monkeypatch.setattr(lc_mod, "_explicit_manifest_config", None)
    yield
    cfg._config = None
    cfg._client = None


@pytest.fixture
def router(monkeypatch):
    """The singleton every run in the process shares — the whole hazard."""
    import decimalai.langchain as lc_mod

    r = _CannedRouter(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        strategy="menu",  # full-menu path: no /route endpoint needed
    )
    monkeypatch.setattr(lc_mod, "_skill_router_singleton", r)
    return r


@pytest.fixture
def loader(monkeypatch):
    """Install the invoke/ainvoke skill-injection patch, restored at teardown.

    ``_install_skill_loader`` rebinds ``BaseChatModel.invoke`` for the whole
    process; recording the originals with monkeypatch FIRST is what puts them
    back, so later tests in the session do not route model calls through the
    injector (and out to the network).
    """
    import decimalai.langchain as lc_mod

    monkeypatch.setattr(BaseChatModel, "invoke", BaseChatModel.invoke)
    monkeypatch.setattr(BaseChatModel, "ainvoke", BaseChatModel.ainvoke)
    monkeypatch.setattr(lc_mod, "_skill_loader_installed", False)
    lc_mod._install_skill_loader()
    yield


def _sent_traces():
    import decimalai._config as cfg

    cfg._sender.flush()
    return {
        c[1][0].agent_name: c[1][0]
        for c in cfg._client.method_calls
        if c[0] == "ingest_trace"
    }


def _handler(name: str):
    from decimalai.langchain import CallbackHandler

    return CallbackHandler(agent_name=name)


# ── activation: the rung the product claim rests on ──────────────────────────


class TestLoadsStayOnTheRunThatMadeThem:
    def test_two_threads_each_keep_their_own_load(self, router, loader):
        """A loads ``alpha``, B loads ``beta``, and B ships first.

        Against the unscoped drain B's ``build_trace`` took BOTH names and A
        got none.
        """
        llm = FakeListChatModel(responses=list("ABCDEFGH"))
        both_loaded = threading.Barrier(2, timeout=20)
        b_shipped = threading.Event()

        def make_tool(skill: str, wait_for_b: bool):
            def tool(_x):
                body = router.load_skill(skill)
                assert SKILL_BODIES[skill].split()[0] in body, body
                both_loaded.wait()
                if wait_for_b:
                    # A finishes only after B's trace has already shipped —
                    # this is the "whichever trace sends first" race, made
                    # deterministic.
                    b_shipped.wait(20)
                return f"question for {skill}"

            return tool

        ha, hb = _handler("run-A"), _handler("run-B")
        chain_a = (RunnableLambda(make_tool("alpha", True)) | llm).with_config(
            run_name="ChainA", callbacks=[ha],
        )
        chain_b = (RunnableLambda(make_tool("beta", False)) | llm).with_config(
            run_name="ChainB", callbacks=[hb],
        )

        ta = threading.Thread(target=lambda: chain_a.invoke("a"))
        tb = threading.Thread(target=lambda: chain_b.invoke("b"))
        ta.start()
        tb.start()
        tb.join(30)
        b_shipped.set()
        ta.join(30)

        traces = _sent_traces()
        assert set(traces) == {"run-A", "run-B"}, traces
        assert traces["run-A"].skills_loaded_by_agent == ["alpha"]
        assert traces["run-B"].skills_loaded_by_agent == ["beta"]

    def test_a_run_that_loaded_nothing_reports_no_activation(self, router, loader):
        """The fabrication half of the defect, on its own.

        B never calls ``load_skill``. Nothing may put ``alpha`` on B's
        activation rung — an inferred or borrowed activation is exactly the
        number the ladder is not allowed to invent.
        """
        llm = FakeListChatModel(responses=list("ABCDEFGH"))
        a_loaded = threading.Event()
        b_shipped = threading.Event()

        def a_tool(_x):
            router.load_skill("alpha")
            a_loaded.set()
            b_shipped.wait(20)
            return "question A"

        def b_tool(_x):
            a_loaded.wait(20)  # B routes strictly AFTER A's load is on the rail
            return "question B"

        ha, hb = _handler("run-A"), _handler("run-B")
        chain_a = (RunnableLambda(a_tool) | llm).with_config(
            run_name="ChainA", callbacks=[ha],
        )
        chain_b = (RunnableLambda(b_tool) | llm).with_config(
            run_name="ChainB", callbacks=[hb],
        )

        ta = threading.Thread(target=lambda: chain_a.invoke("a"))
        tb = threading.Thread(target=lambda: chain_b.invoke("b"))
        ta.start()
        tb.start()
        tb.join(30)
        b_shipped.set()
        ta.join(30)

        traces = _sent_traces()
        assert set(traces) == {"run-A", "run-B"}, traces
        assert traces["run-B"].skills_loaded_by_agent == []
        assert "alpha" not in traces["run-B"].skills_delivered
        assert traces["run-A"].skills_loaded_by_agent == ["alpha"]

    def test_two_asyncio_tasks_each_keep_their_own_load(self, router, loader):
        """Same defect over ``ainvoke``: LangChain runs concurrent tasks in
        copied contexts on ONE thread, so a thread-local fix would not hold
        here."""
        llm = FakeListChatModel(responses=list("ABCDEFGH"))

        async def main():
            a_loaded = asyncio.Event()
            b_shipped = asyncio.Event()

            async def a_tool(_x):
                router.load_skill("alpha")
                a_loaded.set()
                await asyncio.wait_for(b_shipped.wait(), 20)
                return "question A"

            async def b_tool(_x):
                await asyncio.wait_for(a_loaded.wait(), 20)
                router.load_skill("beta")
                return "question B"

            ha, hb = _handler("run-A"), _handler("run-B")
            chain_a = (RunnableLambda(a_tool) | llm).with_config(
                run_name="ChainA", callbacks=[ha],
            )
            chain_b = (RunnableLambda(b_tool) | llm).with_config(
                run_name="ChainB", callbacks=[hb],
            )

            task_a = asyncio.create_task(chain_a.ainvoke("a"))
            task_b = asyncio.create_task(chain_b.ainvoke("b"))
            await task_b
            b_shipped.set()
            await task_a

        asyncio.run(main())

        traces = _sent_traces()
        assert set(traces) == {"run-A", "run-B"}, traces
        assert traces["run-A"].skills_loaded_by_agent == ["alpha"]
        assert traces["run-B"].skills_loaded_by_agent == ["beta"]


# ── offered / delivered / routing_id: the rungs below activation ─────────────


class TestRoutingRailsStayOnTheRunThatEarnedThem:
    def test_a_run_with_no_model_call_claims_no_routing_decision(
        self, router, loader,
    ):
        """B is a chain with no model call at all, so it captures nothing of
        its own and falls through to the last-resort rail. It must still not
        report A's routing_id or A's offered set — the model in run B was
        never shown anything, and there is no model in run B to show it to."""
        llm = FakeListChatModel(responses=list("ABCDEFGH"))
        a_routed = threading.Event()
        b_shipped = threading.Event()

        def a_tool(_x):
            return "question A"

        def a_after(_x):
            a_routed.set()
            b_shipped.wait(20)
            return _x

        def b_only(_x):
            a_routed.wait(20)
            return "B never called a model"

        ha, hb = _handler("run-A"), _handler("run-B")
        chain_a = (
            RunnableLambda(a_tool) | llm | RunnableLambda(a_after)
        ).with_config(run_name="ChainA", callbacks=[ha])
        chain_b = (RunnableLambda(b_only) | RunnableLambda(lambda x: x)).with_config(
            run_name="ChainB", callbacks=[hb],
        )

        ta = threading.Thread(target=lambda: chain_a.invoke("a"))
        tb = threading.Thread(target=lambda: chain_b.invoke("b"))
        ta.start()
        tb.start()
        tb.join(30)
        b_shipped.set()
        ta.join(30)

        traces = _sent_traces()
        assert set(traces) == {"run-A", "run-B"}, traces
        b = traces["run-B"]
        assert b.llm_calls == []
        assert b.routing_id is None
        assert b.skills_offered_in_prompt == []
        assert b.skills_delivered == []
        # A still reports its own decision — the fix removes a false claim,
        # it does not silence a true one.
        a = traces["run-A"]
        assert a.routing_id == "rt_" + "0" * 24
        assert a.skills_offered_in_prompt == ["alpha", "beta"]


    def test_a_bare_llm_invoke_does_not_donate_its_decision(
        self, router, loader,
    ):
        """The one case LangChain's own run ids cannot cover.

        ``llm.invoke()`` emits no chain callbacks, and its injection runs
        BEFORE the callback manager has minted any run id — so the routing
        decision it puts on the Router has no LangChain id to be filed under.
        Left unowned, a concurrent run that captured nothing of its own would
        happily claim it. The per-model-call token is what closes that.
        """
        blocked = threading.Event()
        b_shipped = threading.Event()

        class _BlockingChat(FakeListChatModel):
            def _generate(self, messages, *args, **kwargs):
                blocked.set()
                b_shipped.wait(20)
                return super()._generate(messages, *args, **kwargs)

        llm = _BlockingChat(responses=["ok"])
        ha, hb = _handler("bare-A"), _handler("chain-B")

        def run_a():
            llm.invoke("bare question", config={"callbacks": [ha]})

        def run_b():
            blocked.wait(20)  # B drains strictly AFTER A's injection wrote
            (
                RunnableLambda(lambda x: "no model here")
                | RunnableLambda(lambda x: x)
            ).with_config(run_name="ChainB", callbacks=[hb]).invoke("b")

        ta = threading.Thread(target=run_a)
        tb = threading.Thread(target=run_b)
        ta.start()
        tb.start()
        tb.join(30)
        b_shipped.set()
        ta.join(30)

        traces = _sent_traces()
        assert set(traces) == {"bare-A", "chain-B"}, traces
        assert traces["chain-B"].routing_id is None
        assert traces["chain-B"].skills_offered_in_prompt == []
        # A keeps its own decision.
        assert traces["bare-A"].routing_id == "rt_" + "0" * 24
        assert traces["bare-A"].skills_offered_in_prompt == ["alpha", "beta"]


# ── the fallback the unscoped drain existed to provide still works ───────────


class TestTheSingleRunFallbackSurvives:
    """LangChain dispatches callbacks under ``copy_context()``, so a rail the
    injection wrote inside the runnable's context is invisible to the handler.
    That is why the singleton drain exists, and a fix that simply deleted it
    would regress a working path. A lone run must still get its names.
    """

    def test_a_lone_run_still_reports_offered_and_delivered(self, router, loader):
        import decimalai.skill_router as sr

        # Blind the ContextVar rails, which is what `copy_context()` does to
        # them in the frameworks this fallback was added for. Everything the
        # trace reports now has to come off the router itself.
        sr._last_offered_names_ctx.set(None)
        sr._last_delivered_names_ctx.set(None)

        llm = FakeListChatModel(responses=["ok"])
        h = _handler("solo")
        chain = (RunnableLambda(lambda x: "q") | llm).with_config(
            run_name="Solo", callbacks=[h],
        )
        chain.invoke("x")

        traces = _sent_traces()
        assert traces["solo"].routing_id == "rt_" + "0" * 24
        assert traces["solo"].skills_offered_in_prompt == ["alpha", "beta"]

    def test_a_lone_run_still_reports_a_load(self, router, loader):
        llm = FakeListChatModel(responses=["ok"])
        h = _handler("solo")

        def tool(_x):
            router.load_skill("alpha")
            return "q"

        chain = (RunnableLambda(tool) | llm).with_config(
            run_name="Solo", callbacks=[h],
        )
        chain.invoke("x")

        traces = _sent_traces()
        assert traces["solo"].skills_loaded_by_agent == ["alpha"]

    def test_a_router_with_no_scoped_rails_at_all_still_works(self, monkeypatch):
        """A router object from an older SDK (or a user's stand-in) carries
        only the unscoped ``consume_*`` methods. The adapter must degrade to
        them rather than raising or reporting nothing."""
        import decimalai.langchain as lc_mod

        class _LegacyRouter:
            def __init__(self):
                self._rid: Optional[str] = "rt_" + "c" * 24
                self._offered: List[str] = ["legacy-skill"]

            def consume_routing_id(self):
                v, self._rid = self._rid, None
                return v

            def consume_offered_names(self):
                v, self._offered = self._offered, []
                return v

            def consume_delivered_names(self):
                return []

            def consume_loaded_names(self):
                return []

        monkeypatch.setattr(lc_mod, "_skill_router_singleton", _LegacyRouter())
        from uuid import uuid4

        h = lc_mod.CallbackHandler(agent_name="legacy", auto_send=False)
        root = uuid4()
        h.on_chain_start({"name": "AgentExecutor"}, {"input": "x"}, run_id=root)
        h.on_chat_model_start(
            {"name": "ChatOpenAI"}, [[]], run_id=uuid4(), parent_run_id=root,
            invocation_params={"model_name": "gpt-4o-mini"},
        )
        trace = h.build_trace()

        assert trace.routing_id == "rt_" + "c" * 24
        assert trace.skills_offered_in_prompt == ["legacy-skill"]


# ── bookkeeping cannot grow without bound ────────────────────────────────────


class TestPerRunStateIsCleanedUp:
    """A dict entry left on the process-wide singleton per run is a slow leak
    in a long-lived server — and these entries are keyed by run, so they are
    exactly the kind that accumulates one per request forever."""

    def test_a_scoped_rail_is_drained_by_the_run_that_owns_it(
        self, router, loader,
    ):
        """After the owning run ships, nothing of its is left on the router
        for the next run to pick up."""
        llm = FakeListChatModel(responses=["ok"])

        def tool(_x):
            router.load_skill("alpha")
            return "q"

        h = _handler("owner")
        (RunnableLambda(tool) | llm).with_config(
            run_name="Owner", callbacks=[h],
        ).invoke("x")

        assert router._scoped_loaded_names == {}
        assert router._scoped_routing_rails == {}
        assert router._unscoped_rail_owners == {}
        assert router.consume_loaded_names() == []
        assert router.consume_offered_names() == []

    def test_a_run_that_raises_leaves_nothing_behind(self, router, loader):
        """The exception path is the one that matters: a run that blew up on
        its way out is exactly the run whose rails nobody would come back
        for."""
        llm = FakeListChatModel(responses=["ok"])

        def boom(_x):
            router.load_skill("alpha")
            raise RuntimeError("tool exploded")

        h = _handler("kaboom")
        with pytest.raises(RuntimeError):
            (RunnableLambda(boom) | llm).with_config(
                run_name="Boom", callbacks=[h],
            ).invoke("x")

        assert router._scoped_loaded_names == {}
        assert router._scoped_routing_rails == {}

    def test_a_run_that_never_builds_a_trace_still_releases_its_rails(
        self, router, loader,
    ):
        """The leak with no other reader: tracing switched off.

        The Router keeps working — the menu is still injected, `load_skill`
        still serves bodies — but `_auto_send` returns before assembling
        anything, so nothing ever drains what those calls filed. One entry per
        request, forever, in a process that is doing no tracing at all.
        """
        import decimalai._config as cfg

        llm = FakeListChatModel(responses=["ok", "ok2"])

        def tool(_x):
            router.load_skill("alpha")
            return "q"

        cfg._config.enabled = False
        try:
            for _ in range(3):
                (RunnableLambda(tool) | llm).with_config(
                    run_name="Dark", callbacks=[_handler("dark")],
                ).invoke("x")
        finally:
            cfg._config.enabled = True

        assert _sent_traces() == {}
        assert router._scoped_loaded_names == {}
        assert router._scoped_routing_rails == {}

    def test_a_manual_caller_still_gets_the_rails_after_the_chain_returns(
        self, router, loader,
    ):
        """The other side of that cleanup. With ``auto_send=False`` the run's
        only reader is the caller's own ``build_trace()``, which happens after
        the chain has already ended — releasing on chain end would hand her an
        empty trace."""
        from decimalai.langchain import CallbackHandler

        llm = FakeListChatModel(responses=["ok"])
        h = CallbackHandler(agent_name="manual", auto_send=False)

        def tool(_x):
            router.load_skill("alpha")
            return "q"

        (RunnableLambda(tool) | llm).with_config(
            run_name="Manual", callbacks=[h],
        ).invoke("x")
        trace = h.build_trace()

        assert trace.skills_loaded_by_agent == ["alpha"]

    def test_an_evicted_run_releases_its_rails(self, router, loader):
        """A root chain that starts and never ends is evicted at the
        in-flight cap; its trace is already lost, so its rails have no reader
        left and must not sit on the singleton either."""
        import decimalai.langchain as lc_mod
        from uuid import uuid4

        h = _handler("evict")
        h.auto_send = False
        first_root = uuid4()
        h.on_chain_start({"name": "AgentExecutor"}, {"input": "x"},
                         run_id=first_root)
        router.load_skill("alpha", scope=str(first_root))
        assert str(first_root) in router._scoped_loaded_names

        for _ in range(lc_mod._MAX_LIVE_RUNS):
            h.on_chain_start({"name": "AgentExecutor"}, {"input": "x"},
                             run_id=uuid4())

        assert str(first_root) not in router._scoped_loaded_names


# ── ownership is REQUIRED, not merely un-contradicted ────────────────────────


class TestRailOwnership:
    """Who may take the shared rails, and what happens when they may not.

    Two properties, both of which the original peek-then-drain pair got wrong:
    the check and the take are ATOMIC (a write landing between them was taken
    while the snapshot still said clear), and a refusal LEAVES the rail rather
    than draining it (the original defect lost the true activation from the run
    that earned it, not just misplaced it).

    The third property — refusing rails nobody owns — was tried and reverted;
    see the test below for why.
    """

    def _router(self):
        from decimalai.skill_router import SkillRouter

        r = SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")
        r._loaded_names.append("ghost-skill")
        return r

    def test_an_unowned_rail_is_still_claimable_and_that_is_deliberate(self):
        """Documents a trade, so the next reader does not "fix" it back.

        Requiring an owner looks strictly safer and is not: the documented
        "assemble the fragment yourself, then invoke" pattern calls
        build_prompt_fragment outside any runnable, so nothing can name a run
        for it. Requiring ownership sent four regression tests red — including
        one guarding a real reported failure where the trace shipped a NULL
        routing_id.

        The residual, stated plainly: a load_skill from a plain thread running
        concurrently with a run is unowned, and the run may claim it. The fix
        for that is to give the outside-a-run pattern an owner of its own, not
        to refuse unowned rails.
        """
        router = self._router()
        assert router.unscoped_rail_owners() == [], "fixture wrote an owner by accident"

        _, _, _, loaded = router.drain_unscoped_rails_for({"run-A"})
        assert loaded == ["ghost-skill"]

    def test_an_owned_rail_still_reaches_its_own_run(self):
        """The other direction, so the guard cannot pass by refusing everything."""
        router = self._router()
        router._note_unscoped_writer("run-A")

        _, _, _, loaded = router.drain_unscoped_rails_for({"run-A"})
        assert loaded == ["ghost-skill"]
        assert router._loaded_names == [], "an accepted rail must be consumed"

    def test_a_rail_owned_by_another_run_is_refused_and_kept(self):
        router = self._router()
        router._note_unscoped_writer("run-B")

        _, _, _, loaded = router.drain_unscoped_rails_for({"run-A"})
        assert loaded == []
        assert router._loaded_names == ["ghost-skill"], (
            "run-A discarded run-B's activation instead of leaving it"
        )
