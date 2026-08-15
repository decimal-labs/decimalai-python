"""Regressions for the four ways the LangChain adapter used to lose traces.

Every test here corresponds to a failure reproduced against a live local
backend before the fix:

1. `chain.batch([a, b, c])` and `asyncio.gather(ainvoke x3)` each collapsed
   three runs into ONE trace the backend rejected with
   ``spans[N]: 'ended_at' is required`` — 0 of 3 runs persisted. `instrument()`
   publishes exactly one handler process-wide, LangChain copies the context
   for parallel work, and ALL trace state lived in single slots on that
   handler.
2. Four chains on four `threading.Thread`s produced ZERO traces and no
   warning: a thread starts with a fresh empty Context, so the module
   ContextVar fell back to `default=None` and LangChain's configure hook
   installed nothing.
3. A run with no model, tool or prompt registered no manifest, and ingest
   requires one — every such trace 400'd.
4. A bare `llm.invoke()` emitted no trace at all, and its stranded LLM
   record was then shipped inside an unrelated later run's trace.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda


class NamedFakeChat(FakeListChatModel):
    """A fake chat model that reports a model name, like a real provider."""

    model_name: str = "fake-model-1"

    @property
    def _identifying_params(self):
        return {"model_name": self.model_name, "temperature": 0.0}


@pytest.fixture(autouse=True)
def reset_sdk_state(monkeypatch):
    import decimalai._config as cfg
    import decimalai.langchain as lc_mod
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {}
    cfg._client.list_manifests.return_value = {"manifests": []}
    cfg._sender._pending = []
    monkeypatch.setattr(lc_mod, "_manifest_id", None)
    monkeypatch.setattr(lc_mod, "_manifest_tracker", ManifestTracker())
    # raising=False so this fixture still builds against a build that predates
    # the probe — the tests then fail on their own assertions, which is what
    # makes them readable as regressions.
    monkeypatch.setattr(lc_mod, "_manifest_adoption_probed", set(), raising=False)
    monkeypatch.setattr(lc_mod, "_explicit_manifest_config", None)
    yield
    cfg._config = None
    cfg._client = None


def _sent_traces():
    import decimalai._config as cfg

    cfg._sender.flush()
    return [c[1][0] for c in cfg._client.method_calls if c[0] == "ingest_trace"]


def _unpublish_global_handler():
    """Undo an `instrument()` for the rest of the session.

    The configure hook LangChain registers is never unregistered, and the
    ContextVar's default IS the global handler — so clearing the var only
    silences it in contexts that inherit this one. Reset the handler itself
    too, or a later test that runs a chain on a worker thread inherits this
    one's agent name and in-flight state.
    """
    import decimalai.langchain as lc_mod

    lc_mod._decimal_callback_var.set(None)
    lc_mod._global_handler.agent_name = None
    lc_mod._global_handler.reset()


def _make_chain(handler, name="Chain"):
    llm = NamedFakeChat(responses=list("ABCDEFGHIJ"))
    prompt = ChatPromptTemplate.from_messages([("system", "sys"), ("human", "{q}")])
    return (prompt | llm).with_config(run_name=name, callbacks=[handler])


# ── 1. Concurrency: one handler, many simultaneous runs ─────


class TestConcurrentRunsGetTheirOwnTrace:
    def test_batch_of_three_sends_three_complete_traces(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="batch-agent")
        _make_chain(handler).batch([{"q": "one"}, {"q": "two"}, {"q": "three"}])

        traces = _sent_traces()
        assert len(traces) == 3, (
            f"chain.batch([3]) must produce 3 traces, got {len(traces)}. One "
            f"trace here means the runs shared a single set of state slots."
        )
        # Every span must be closed. An open span is what the backend rejects
        # with `spans[N]: 'ended_at' is required`, losing the whole trace.
        for trace in traces:
            assert trace.spans, "trace has no spans"
            assert all(s.ended_at is not None for s in trace.spans), (
                "a sibling run's still-open span was shipped in this trace"
            )
            assert len(trace.llm_calls) == 1
        assert len({t.id for t in traces}) == 3
        assert {t.user_input_preview for t in traces} == {
            '{"q": "one"}', '{"q": "two"}', '{"q": "three"}',
        }

    def test_asyncio_gather_of_three_sends_three_complete_traces(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="gather-agent")
        chain = _make_chain(handler)

        async def run_all():
            await asyncio.gather(*[chain.ainvoke({"q": f"q{i}"}) for i in range(3)])

        asyncio.run(run_all())

        traces = _sent_traces()
        assert len(traces) == 3
        for trace in traces:
            assert all(s.ended_at is not None for s in trace.spans)
            assert len(trace.llm_calls) == 1

    def test_threads_each_get_their_own_trace(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="thread-agent")
        chain = _make_chain(handler)
        threads = [
            threading.Thread(target=lambda i=i: chain.invoke({"q": f"t{i}"}))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        traces = _sent_traces()
        assert len(traces) == 4
        assert {t.user_input_preview for t in traces} == {
            '{"q": "t0"}', '{"q": "t1"}', '{"q": "t2"}', '{"q": "t3"}',
        }

    def test_interleaved_raw_callbacks_do_not_cross_contaminate(self):
        """The event ORDER `.batch()` actually emits: all roots open first."""
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="interleaved", auto_send=False)
        roots = [uuid4() for _ in range(3)]
        children = [uuid4() for _ in range(3)]

        for i, root in enumerate(roots):
            handler.on_chain_start({"name": f"Root{i}"}, {"q": i}, run_id=root)
        for i, (root, child) in enumerate(zip(roots, children)):
            handler.on_chain_start(
                {"name": f"Child{i}"}, {}, run_id=child, parent_run_id=root
            )
            handler.on_chain_end({}, run_id=child, parent_run_id=root)

        states = [handler._runs[r] for r in roots]
        assert [len(s.spans) for s in states] == [2, 2, 2], (
            f"each root should own exactly its own 2 spans, got "
            f"{[len(s.spans) for s in states]}"
        )
        assert len({s.trace_id for s in states}) == 3

    def test_closing_one_run_leaves_the_others_intact(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="partial", auto_send=False)
        first, second = uuid4(), uuid4()
        handler.on_chain_start({"name": "First"}, {"q": 1}, run_id=first)
        handler.on_chain_start({"name": "Second"}, {"q": 2}, run_id=second)
        handler.on_chain_end({"out": 1}, run_id=first, parent_run_id=None)

        assert second in handler._runs, (
            "ending the first root wiped the second root's state — the exact "
            "shape that made batch() lose runs 2 and 3"
        )
        assert len(handler._runs[second].spans) == 1


# ── 2. Threading: the global handler must reach worker threads ──


class TestGlobalHandlerReachesWorkerThreads:
    def test_callback_var_default_is_the_global_handler(self):
        """A fresh Context must still resolve the handler.

        LangChain's configure hook only installs a handler when
        `context_var.get()` is non-None, and a `threading.Thread` starts with
        a Context in which nothing was ever set — so `.get()` can only return
        the var's DEFAULT. With the old `default=None` this returned None and
        threaded runs were silently untraced.
        """
        import decimalai.langchain as lc_mod

        resolved = {}

        def read_in_fresh_context():
            resolved["handler"] = lc_mod._decimal_callback_var.get()

        thread = threading.Thread(target=read_in_fresh_context)
        thread.start()
        thread.join()

        assert resolved["handler"] is lc_mod._global_handler
        assert resolved["handler"] is not None

    def test_instrument_publishes_the_default_handler(self, monkeypatch):
        import decimalai.langchain as lc_mod

        monkeypatch.setattr(lc_mod, "_installed", False)
        lc_mod.instrument(agent_name="published-agent", disk_sync=False)
        try:
            assert lc_mod._decimal_callback_var.get() is lc_mod._global_handler
            assert lc_mod._global_handler.agent_name == "published-agent"
            assert lc_mod._global_handler.auto_send is True
        finally:
            _unpublish_global_handler()

    def test_an_explicit_handler_suppresses_the_global_one(self, monkeypatch):
        """`instrument()` + a per-call handler must not double-trace.

        Both handlers used to run, and both shipped a trace. Because a span's
        id IS the LangChain run_id, the two traces carried identical span
        ids — the backend's id-dedup kept the first and stored the second
        with zero spans and zero llm_calls, putting a phantom empty trace on
        the agent's timeline for every run.
        """
        import decimalai.langchain as lc_mod

        monkeypatch.setattr(lc_mod, "_installed", False)
        lc_mod.instrument(agent_name="global-agent", disk_sync=False)
        try:
            explicit = lc_mod.CallbackHandler(agent_name="explicit-agent")
            _make_chain(explicit, "Doubled").invoke({"q": "hi"})
            traces = _sent_traces()
        finally:
            _unpublish_global_handler()

        assert len(traces) == 1, (
            f"expected exactly one trace, got {len(traces)} "
            f"(agents: {[t.agent_name for t in traces]})"
        )
        assert traces[0].agent_name == "explicit-agent"
        assert traces[0].spans and traces[0].llm_calls


# ── 3. The manifest gate ────────────────────────────────────


class TestManifestForRunsWithNothingToDeclare:
    def test_run_with_no_model_or_tool_still_ships_a_manifest_id(self):
        """Ingest requires manifest_id; this run has nothing to introspect."""
        import decimalai._config as cfg
        from decimalai.langchain import CallbackHandler

        cfg._client.register_manifest.return_value = {"manifest_id": "mf-placeholder"}
        handler = CallbackHandler(agent_name="lambda-agent")
        RunnableLambda(lambda x: {"r": x["q"].upper()}).with_config(
            run_name="Shout", callbacks=[handler]
        ).invoke({"q": "hi"})

        traces = _sent_traces()
        assert len(traces) == 1
        assert traces[0].manifest_id == "mf-placeholder", (
            "a run that declares no model/tool/prompt still needs a manifest_id "
            "or ingest rejects it with 400"
        )

    def test_placeholder_manifest_declares_nothing_rather_than_faking_it(self):
        import decimalai._config as cfg
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="lambda-agent")
        RunnableLambda(lambda x: x).with_config(
            run_name="Noop", callbacks=[handler]
        ).invoke({"q": "hi"})
        _sent_traces()

        (snapshot,), _ = cfg._client.register_manifest.call_args
        assert snapshot.components == [], (
            "the placeholder must declare NOTHING. A fabricated "
            "'unknown/unknown' model would both lie and still diff as a major "
            "model_runtime change once the real model shows up."
        )
        assert snapshot.agent_name == "lambda-agent"

    def test_empty_run_after_a_populated_one_never_re_registers(self):
        """The poison case: an empty manifest must not supersede a real one.

        Measured against the local backend before the guard: registering a
        zero-component manifest after a populated one produces
        `tool_registry: search removed` + `model_runtime: provider 'openai' →
        ''`, breaking/major, recommended_decision "replay" — the platform
        telling the user they deleted their tools when they did no such thing.
        """
        import decimalai._config as cfg
        from decimalai.langchain import CallbackHandler

        cfg._client.register_manifest.return_value = {"manifest_id": "mf-real"}
        handler = CallbackHandler(agent_name="mixed-agent")

        _make_chain(handler, "WithModel").invoke({"q": "hi"})
        _sent_traces()
        assert cfg._client.register_manifest.call_count == 1
        (real_snapshot,), _ = cfg._client.register_manifest.call_args
        assert real_snapshot.components, "the model run should declare components"

        RunnableLambda(lambda x: x).with_config(
            run_name="NothingToDeclare", callbacks=[handler]
        ).invoke({"q": "hi"})
        traces = _sent_traces()

        assert cfg._client.register_manifest.call_count == 1, (
            "the empty run re-registered — that supersedes the populated "
            "manifest and the diff reads every absent surface as a deletion"
        )
        assert traces[-1].manifest_id == "mf-real"

    def test_fresh_process_adopts_the_platforms_active_manifest(self):
        """A process-local guard alone is not enough.

        A replica that boots and only ever runs model-less chains has an empty
        local tracker, so without this probe it would register the empty
        manifest over whatever a sibling process already declared — the
        regression returns on every restart.
        """
        import decimalai._config as cfg
        from decimalai.langchain import CallbackHandler

        cfg._client.list_manifests.return_value = {
            "manifests": [
                {"id": "mf-superseded", "status": "superseded"},
                {"id": "mf-active", "status": "active"},
            ]
        }
        handler = CallbackHandler(agent_name="restarted-agent")
        RunnableLambda(lambda x: x).with_config(
            run_name="Noop", callbacks=[handler]
        ).invoke({"q": "hi"})

        traces = _sent_traces()
        assert traces[0].manifest_id == "mf-active"
        cfg._client.register_manifest.assert_not_called()

    def test_adoption_is_probed_at_most_once_per_agent(self):
        import decimalai._config as cfg
        from decimalai.langchain import CallbackHandler

        cfg._client.list_manifests.return_value = {"manifests": []}
        cfg._client.register_manifest.return_value = {"manifest_id": "mf-new"}
        handler = CallbackHandler(agent_name="probe-agent")
        noop = RunnableLambda(lambda x: x).with_config(
            run_name="Noop", callbacks=[handler]
        )
        for _ in range(3):
            noop.invoke({"q": "hi"})
        _sent_traces()

        assert cfg._client.list_manifests.call_count == 1
        assert cfg._client.register_manifest.call_count == 1

    def test_completion_model_is_a_declared_model(self):
        """`on_llm_start` (non-chat LLM) records the model too.

        Only `on_chat_model_start` used to, so a completion model produced an
        empty manifest and lost its traces to the same 400.
        """
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="completion-agent", auto_send=False)
        root, llm_run = uuid4(), uuid4()
        handler.on_chain_start({"name": "Root"}, {"q": "x"}, run_id=root)
        handler.on_llm_start(
            {"name": "OpenAI"}, ["hello"], run_id=llm_run, parent_run_id=root,
            invocation_params={"model_name": "gpt-3.5-turbo-instruct"},
        )
        state = handler._runs[root]
        assert state.seen_model is not None
        assert state.seen_model["model"] == "gpt-3.5-turbo-instruct"


# ── 4. Bare llm.invoke(): its own trace, and no leaking ─────


class TestBareModelCall:
    def test_bare_llm_invoke_produces_its_own_trace(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="bare-agent")
        NamedFakeChat(responses=["Z"]).invoke("hello", config={"callbacks": [handler]})

        traces = _sent_traces()
        assert len(traces) == 1, (
            "a bare llm.invoke() emits no chain callbacks, so nothing used to "
            "close or send its trace"
        )
        assert len(traces[0].llm_calls) == 1
        assert traces[0].started_at is not None

    def test_bare_call_does_not_leak_into_a_later_unrelated_run(self):
        """The stranded record used to ride out on someone else's trace.

        A bare `RunnableLambda` is in `_SKIP_CHAIN_TYPES`, so the old
        `on_chain_start` returned before the state reset — leaving the bare
        call's LLM record in place — and the lambda's `on_chain_end` then
        found a non-empty `_llm_calls` and shipped it as that run's own work.
        """
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="leak-agent")
        NamedFakeChat(responses=["Z"]).invoke(
            "SECRET-ORPHAN-PROMPT", config={"callbacks": [handler]}
        )
        RunnableLambda(lambda x: x).invoke(
            {"q": "unrelated"}, config={"callbacks": [handler]}
        )

        traces = _sent_traces()
        for trace in traces:
            carries_orphan = any(
                "SECRET-ORPHAN-PROMPT" in str(call.rendered_input)
                for call in trace.llm_calls
            )
            is_the_lambda_run = "unrelated" in (
                (trace.final_output_preview or "") + (trace.user_input_preview or "")
            )
            assert not (carries_orphan and is_the_lambda_run), (
                "the bare call's LLM record was shipped inside the unrelated "
                "lambda run's trace"
            )
        assert sum(len(t.llm_calls) for t in traces) == 1, (
            "the bare call's record must appear exactly once, on its own trace"
        )
