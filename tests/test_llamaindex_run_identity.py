"""Whose run is this? — per-run identity for the LlamaIndex adapter.

``decimalai.llamaindex.instrument()`` installs ONE ``DecimalSpanHandler`` on
LlamaIndex's root dispatcher, and the dispatcher never gives a handler back, so
re-installing is not a way to run a second agent (it double-traces every query
from then on). That made the install-time ``agent_name`` the answer for every
run in the process, which is wrong in two ways that this file pins down:

  1. **Identity.** A second agent's traces shipped under the FIRST agent's
     name. Eight concurrent lanes arrived as eight traces of one agent, so a
     multi-tenant service had one tenant's history and seven silent ones.
  2. **Manifest.** ``manifest_hash`` does not cover the agent name (see
     ``schema.manifest._compute_overall_hash``), so a second agent with the
     same structure hashed identically, deduped against the first, never
     registered — and carried the first agent's ``manifest_id``. Fixing (1)
     alone turns "wrong name" into "right name, someone else's manifest",
     which is why both halves are tested together.

The fix ports LangChain's shape (``decimalai/langchain.py``): per-run state
keyed by the run's own id — here the root SPAN id, which LlamaIndex mints as
``f"{Class}.{method}-{uuid4()}"`` and passes as an argument, so it survives
threads and asyncio tasks unconditionally — plus per-agent manifest maps
(``_manifest_ids`` / ``_manifest_hashes``). The name is supplied by the SDK's
existing run scope, ``decimalai.providers.agent_run`` (the same ContextVar the
raw-provider and Pydantic AI rails read), or by LlamaIndex's own
``instrument_tags``.

Same mock-client capture pattern as test_llamaindex_tree_fidelity.py.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from decimalai.llamaindex import DecimalSpanHandler
from decimalai.providers import agent_run


# ── Fake LlamaIndex instances — classification keys off the type name ────

class QueryEngine:  # → "query" → the tree root
    pass


class Retriever:  # → "retrieve"
    pass


class OpenAIEmbedding:  # → "embed"
    pass


class OpenAI:  # fake LLM → is_llm_call, provider=openai
    def __init__(self, model=None, temperature=None):
        self.model = model
        self.temperature = temperature


def _openai_result(model="gpt-4o-mini"):
    usage = SimpleNamespace(prompt_tokens=129, completion_tokens=11)
    return SimpleNamespace(raw=SimpleNamespace(usage=usage, model=model), response="ok")


#: manifest_id the fake backend minted → the agent_name it was registered FOR.
#: This is the platform's own answer to "whose manifest is this", and it is what
#: the conformance probe's `manifest_owner()` checks.
_MANIFEST_OWNERS: dict = {}


@pytest.fixture(autouse=True)
def _reset_sdk():
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    _MANIFEST_OWNERS.clear()
    counter = iter(f"m{i}" for i in range(1, 10_000))
    lock = threading.Lock()

    def _register(snapshot):
        # Every registration mints a DISTINCT id, so "agent B carries agent A's
        # manifest" is visible rather than hidden behind one constant.
        with lock:
            manifest_id = next(counter)
            _MANIFEST_OWNERS[manifest_id] = snapshot.agent_name
        return {"manifest_id": manifest_id}

    cfg._client.register_manifest.side_effect = _register
    # The adoption probe must find nothing, or a model-less run rides an
    # existing manifest instead of registering its own.
    cfg._client.list_manifests.return_value = {"manifests": []}
    yield


def _flush_and_get_traces():
    """Drain the background sender and return every captured RunTrace."""
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    assert cfg._client.ingest_trace.called, "ingest_trace was never called"
    return [c.args[0] for c in cfg._client.ingest_trace.call_args_list]


def _llm_tree(handler, root, *, model="gpt-4o-mini", tags=None):
    """A query tree with one wrapper+inner LLM pair under it.

    Span ids are derived from ``root`` so two trees never collide — which is
    also true of the real dispatcher, whose ids carry a uuid4.
    """
    handler.new_span(root, SimpleNamespace(query_str="Q"), instance=QueryEngine(),
                     parent_span_id=None, tags=tags)
    handler.new_span(f"{root}/predict", None,
                     instance=OpenAI(model=model, temperature=0.0),
                     parent_span_id=root)
    handler.new_span(f"{root}/chat", None,
                     instance=OpenAI(model=model, temperature=0.0),
                     parent_span_id=f"{root}/predict")
    handler.prepare_to_exit_span(f"{root}/chat", None, instance=OpenAI(),
                                 result=_openai_result(model=model))
    handler.prepare_to_exit_span(f"{root}/predict", None, instance=OpenAI(), result="A")
    handler.prepare_to_exit_span(root, None, instance=QueryEngine(),
                                 result=SimpleNamespace(response="A"))


def _retrieval_tree(handler, root):
    """A model-less RUN: `as_retriever().retrieve()`."""
    handler.new_span(root, None, instance=Retriever(), parent_span_id=None)
    handler.new_span(f"{root}/emb", None, instance=OpenAIEmbedding(), parent_span_id=root)
    handler.prepare_to_exit_span(f"{root}/emb", None, instance=OpenAIEmbedding(),
                                 result=[0.1])
    handler.prepare_to_exit_span(root, None, instance=Retriever(), result=["doc"])


def _owner_of(manifest_id):
    """Which agent ``manifest_id`` was registered for, per the fake backend."""
    return _MANIFEST_OWNERS.get(str(manifest_id))


def _by_agent(traces):
    out = {}
    for t in traces:
        out.setdefault(t.agent_name, []).append(t)
    return out


# ── 1. Two agents, one process ───────────────────────────────────────

class TestTwoAgentsInOneProcess:
    def test_each_run_is_filed_under_the_agent_that_ran_it(self):
        """The second agent's trace used to ship under the FIRST agent's name.

        One installed handler, two runs, two names — the shape of any service
        that serves more than one tenant from one process.
        """
        h = DecimalSpanHandler(agent_name="installed-default")

        with agent_run("support-rag"):
            _llm_tree(h, "RetrieverQueryEngine-A")
        with agent_run("billing-rag"):
            _llm_tree(h, "RetrieverQueryEngine-B")

        names = [t.agent_name for t in _flush_and_get_traces()]
        assert names == ["support-rag", "billing-rag"], (
            "the run scope was ignored; every trace carried the install-time name"
        )

    def test_identically_shaped_second_agent_still_gets_its_own_manifest(self):
        """The trap that fixing the NAME alone walks straight into.

        `manifest_hash` covers the components, not the agent name, so two
        agents on the same model hash the same. With one shared hash slot the
        second agent's snapshot looked like a repeat: it never registered, and
        its trace carried the first agent's manifest_id.
        """
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="installed-default")

        with agent_run("support-rag"):
            _llm_tree(h, "Q-A")           # same model...
        with agent_run("billing-rag"):
            _llm_tree(h, "Q-B")           # ...same structure, different agent

        traces = _flush_and_get_traces()
        assert cfg._client.register_manifest.call_count == 2, (
            "the second agent deduped against the first agent's hash"
        )
        by_agent = _by_agent(traces)
        support = by_agent["support-rag"][0].manifest_id
        billing = by_agent["billing-rag"][0].manifest_id
        assert support != billing
        assert _owner_of(support) == "support-rag"
        assert _owner_of(billing) == "billing-rag"

    def test_one_agents_model_is_not_declared_for_the_other(self):
        """`_seen_model` was sticky across the whole handler.

        Agent A queries with a model; agent B only ever retrieves. B's
        model-less run must not inherit A's model and declare a config it
        never ran.
        """
        h = DecimalSpanHandler(agent_name="installed-default")

        with agent_run("has-a-model"):
            _llm_tree(h, "Q-A", model="gpt-4o-mini")
        with agent_run("retrieval-only"):
            _retrieval_tree(h, "R-B")

        _flush_and_get_traces()
        import decimalai._config as cfg

        snapshots = {c.args[0].agent_name: c.args[0]
                     for c in cfg._client.register_manifest.call_args_list}
        assert sorted(snapshots) == ["has-a-model", "retrieval-only"], (
            f"each agent must declare its own manifest; the backend saw "
            f"{sorted(snapshots)}"
        )
        assert [c for c in snapshots["has-a-model"].components
                if c.component_type == "model"]
        assert not [c for c in snapshots["retrieval-only"].components
                    if c.component_type == "model"], (
            "the retrieval-only agent declared the other agent's model"
        )

    def test_the_install_time_name_is_still_the_default(self):
        """No run scope, no tags — nothing changes for a one-agent process."""
        h = DecimalSpanHandler(agent_name="my-rag-agent")
        _llm_tree(h, "Q-plain")

        assert [t.agent_name for t in _flush_and_get_traces()] == ["my-rag-agent"]

    def test_llamaindex_instrument_tags_can_name_the_run(self):
        """LlamaIndex's own per-run channel already reached `span_enter(tags=)`
        and was dropped on the floor. Honouring it costs nothing and is what a
        LlamaIndex user reaches for first."""
        h = DecimalSpanHandler(agent_name="installed-default")
        _llm_tree(h, "Q-tagged", tags={"agent_name": "tagged-rag"})

        assert [t.agent_name for t in _flush_and_get_traces()] == ["tagged-rag"]


# ── 2. The name is captured at ENTER, never re-read at flush ─────────

class TestNameCapturedAtEnter:
    def test_a_tree_that_flushes_outside_its_scope_keeps_its_own_name(self):
        """Three of the four flush paths run outside the caller's context.

        A streamed root flushes from the pass-through tee (whoever drains
        `response_gen`), `close()` flushes at dispatcher shutdown, and the
        dispatcher's asyncio done-callback calls `span_exit` in a COPIED
        context. Reading the run scope at flush time is therefore right by
        luck on the sync path and wrong on the other three — so it is read
        when the tree OPENS.
        """
        h = DecimalSpanHandler(agent_name="installed-default")

        with agent_run("opener"):
            h.new_span("Q", SimpleNamespace(query_str="Q"), instance=QueryEngine(),
                       parent_span_id=None)
            h.new_span("Q/chat", None, instance=OpenAI(model="gpt-4o-mini"),
                       parent_span_id="Q")
            h.prepare_to_exit_span("Q/chat", None, instance=OpenAI(),
                                   result=_openai_result())

        # The scope is gone, and somebody ELSE's scope is open when the root
        # exits — the shape of a stream drained by another request's handler.
        with agent_run("whoever-drained-the-stream"):
            h.prepare_to_exit_span("Q", None, instance=QueryEngine(),
                                   result=SimpleNamespace(response="A"))

        assert [t.agent_name for t in _flush_and_get_traces()] == ["opener"]


# ── 3. N concurrent runs never see each other's state ────────────────

_LANES = 8


def _assert_lanes_are_clean(traces, expected_names):
    assert sorted(t.agent_name for t in traces) == sorted(expected_names), (
        f"agent_names on the wire {sorted(t.agent_name for t in traces)} "
        f"!= lanes {sorted(expected_names)}"
    )
    # Each lane's manifest was registered FOR that lane.
    for t in traces:
        assert t.manifest_id is not None
        assert _owner_of(t.manifest_id) == t.agent_name, (
            f"{t.agent_name} carries a manifest registered for "
            f"{_owner_of(t.manifest_id)}"
        )
    assert len({t.manifest_id for t in traces}) == len(traces), (
        "lanes shared a manifest id"
    )
    # No span belongs to two traces.
    seen = {}
    for t in traces:
        for span in t.spans:
            assert seen.setdefault(span.id, t.agent_name) == t.agent_name, (
                f"span {span.id} appears in two lanes"
            )


class TestConcurrentRunsAreIsolated:
    def test_threads(self):
        """Eight OS threads, eight agents, one process-wide handler.

        Every lane runs the SAME model on purpose: identical structure is what
        made the per-agent manifest keying necessary, so a fix that only
        threads the name through would still fail here.
        """
        h = DecimalSpanHandler(agent_name="installed-default")
        start = threading.Barrier(_LANES)
        names = [f"lane{i}" for i in range(_LANES)]

        def _lane(i):
            start.wait(timeout=10)
            # The scope opens INSIDE the worker: a ThreadPoolExecutor worker
            # starts from a fresh context, so a scope entered before submit
            # would not reach it.
            with agent_run(names[i]):
                _llm_tree(h, f"Q-{i}")

        threads = [threading.Thread(target=_lane, args=(i,)) for i in range(_LANES)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        traces = _flush_and_get_traces()
        assert len(traces) == _LANES
        _assert_lanes_are_clean(traces, names)

    def test_asyncio_tasks(self):
        """The same eight lanes as asyncio tasks.

        Reachable because `agent_run` scopes a ContextVar, and a task copies
        the context at creation — so the scope has to be opened inside the
        coroutine, not around `gather`.
        """
        h = DecimalSpanHandler(agent_name="installed-default")
        names = [f"task{i}" for i in range(_LANES)]

        async def _lane(i):
            with agent_run(names[i]):
                h.new_span(f"Q-{i}", SimpleNamespace(query_str="Q"),
                           instance=QueryEngine(), parent_span_id=None)
                await asyncio.sleep(0)  # force interleaving
                h.new_span(f"Q-{i}/chat", None, instance=OpenAI(model="gpt-4o-mini"),
                           parent_span_id=f"Q-{i}")
                await asyncio.sleep(0)
                h.prepare_to_exit_span(f"Q-{i}/chat", None, instance=OpenAI(),
                                       result=_openai_result())
                await asyncio.sleep(0)
                h.prepare_to_exit_span(f"Q-{i}", None, instance=QueryEngine(),
                                       result=SimpleNamespace(response="A"))

        async def _main():
            await asyncio.gather(*(_lane(i) for i in range(_LANES)))

        asyncio.run(_main())

        traces = _flush_and_get_traces()
        assert len(traces) == _LANES
        _assert_lanes_are_clean(traces, names)


# ── 4. Run state is released when the run ends — however it ends ─────

def _buffers(h):
    """Everything the handler holds ON BEHALF OF a run in flight.

    Read with getattr so these assertions fail on the leak itself rather than
    on a missing attribute when they are pointed at an older handler.
    """
    return {
        name: getattr(h, name, ())
        for name in ("_spans", "_parents", "_trees", "_tree_agents", "_deferred_roots")
    }


class TestNoRunStateLeaks:
    def test_a_completed_run_leaves_nothing_behind(self):
        h = DecimalSpanHandler(agent_name="installed-default")
        for i in range(5):
            with agent_run(f"agent{i}"):
                _llm_tree(h, f"Q-{i}")

        _flush_and_get_traces()
        assert all(not v for v in _buffers(h).values()), _buffers(h)

    def test_a_run_that_ends_by_raising_leaves_nothing_behind(self):
        """The error path — `span_drop` — still releases the tree."""
        h = DecimalSpanHandler(agent_name="installed-default")
        with agent_run("boom"):
            h.new_span("Q", SimpleNamespace(query_str="Q"), instance=QueryEngine(),
                       parent_span_id=None)
            h.new_span("Q/chat", None, instance=OpenAI(model="gpt-4o-mini"),
                       parent_span_id="Q")
            h.prepare_to_drop_span("Q/chat", None, instance=OpenAI(),
                                   err=RuntimeError("model down"))
            h.prepare_to_drop_span("Q", None, instance=QueryEngine(),
                                   err=RuntimeError("model down"))

        assert [t.agent_name for t in _flush_and_get_traces()] == ["boom"]
        assert all(not v for v in _buffers(h).values()), _buffers(h)

    def test_a_flush_that_itself_raises_still_releases_the_tree(self):
        """A tree that dies on the way to the wire must not stay in memory.

        These adapters run inside long-lived web servers, so "the flush
        raised" is a whole span tree leaked per request, not a one-off. The
        dispatcher swallows the exception (`except BaseException: pass`), so
        nothing else would ever come back for these buffers.
        """
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="installed-default")
        with patch.object(cfg._sender, "submit", side_effect=RuntimeError("wire down")):
            with pytest.raises(RuntimeError):
                with agent_run("doomed"):
                    _llm_tree(h, "Q-doomed")

        assert all(not v for v in _buffers(h).values()), _buffers(h)

    def test_a_setup_tree_is_released_too(self):
        """Index construction ships nothing — and must buffer nothing after."""
        h = DecimalSpanHandler(agent_name="installed-default")

        class SentenceSplitter:
            pass

        h.new_span("S", None, instance=SentenceSplitter(), parent_span_id=None)
        h.prepare_to_exit_span("S", None, instance=SentenceSplitter(), result=["node"])

        assert all(not v for v in _buffers(h).values()), _buffers(h)
