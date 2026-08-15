"""Regression tests for what the LlamaIndex adapter puts ON the wire.

test_llamaindex_handler.py covers "does a tree become a trace at all";
test_llamaindex_dispatcher_integration.py covers "does the dispatcher reach
the handler". This file covers the three fidelity defects found once the
adapter started emitting (2026-08-15), each of which was live against the
local backend:

  1. A tree with NO LLM call (index construction, a bare retrieve) got no
     manifest, so ingest 400'd with "manifest_id is required" and the trace
     was lost — a retrieval-only flow scored sent=0 failed=3.
  2. `_flush_tree` minted a fresh TraceSpan per span and never carried the
     parent link, so a real 14-span RAG tree stored 12 flat spans.
  3. LlamaIndex instruments both the public wrapper (`OpenAI.predict`) and
     the inner call (`OpenAI.chat`), so ONE gpt-4o-mini request produced two
     LlmCallRecords — {in=None,out=None} and {in=129,out=11}.

Same mock-client capture pattern as test_llamaindex_handler.py.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from decimalai.llamaindex import DecimalSpanHandler

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Fake LlamaIndex instances — classification keys off type name ────

class QueryEngine:  # → "query" → SpanType.AGENT (root)
    pass


class Retriever:  # → "retrieve" → SpanType.RETRIEVAL
    pass


class SentenceSplitter:  # → "other"; the whole index-construction tree
    pass


class TokenTextSplitter:  # → "other"
    pass


class OpenAIEmbedding:  # → "embed" → SpanType.OTHER
    pass


class OpenAI:  # fake LLM → is_llm_call, provider=openai
    def __init__(self, model=None, temperature=None):
        self.model = model
        self.temperature = temperature


def _openai_result(prompt_tokens=129, completion_tokens=11, model="gpt-4o-mini"):
    """A LlamaIndex-style ChatResponse with .raw.usage (OpenAI token names)."""
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(raw=SimpleNamespace(usage=usage, model=model), response="ok")


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
    cfg._client.register_manifest.return_value = {"manifest_id": "m1"}
    yield


def _flush_and_get_traces():
    """Drain the background sender and return every captured RunTrace."""
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    assert cfg._client.ingest_trace.called, "ingest_trace was never called"
    return [c.args[0] for c in cfg._client.ingest_trace.call_args_list]


def _index_tree(handler, root="SentenceSplitter-0"):
    """An index-construction tree: nested splitters, not a model in sight."""
    handler.new_span(root, None, instance=SentenceSplitter(), parent_span_id=None)
    handler.new_span("SentenceSplitter-1", None, instance=SentenceSplitter(),
                     parent_span_id=root)
    handler.prepare_to_exit_span("SentenceSplitter-1", None,
                                 instance=SentenceSplitter(), result=["node"])
    handler.prepare_to_exit_span(root, None, instance=SentenceSplitter(),
                                 result=["node"])


def _llm_tree(handler, root="RetrieverQueryEngine-0", model="gpt-4o-mini"):
    """A query tree with one wrapper+inner LLM pair under it."""
    handler.new_span(root, SimpleNamespace(query_str="Q"), instance=QueryEngine(),
                     parent_span_id=None)
    handler.new_span("OpenAI.predict-0", None, instance=OpenAI(model=model, temperature=0.0),
                     parent_span_id=root)
    handler.new_span("OpenAI.chat-0", None, instance=OpenAI(model=model, temperature=0.0),
                     parent_span_id="OpenAI.predict-0")
    handler.prepare_to_exit_span("OpenAI.chat-0", None, instance=OpenAI(),
                                 result=_openai_result(model=model))
    handler.prepare_to_exit_span("OpenAI.predict-0", None, instance=OpenAI(),
                                 result="A")
    handler.prepare_to_exit_span(root, None, instance=QueryEngine(),
                                 result=SimpleNamespace(response="A"))


# ── 1. An LLM-free tree still satisfies ingest ───────────────────────

class TestLlmFreeTreeGetsManifest:
    def test_index_tree_registers_manifest_and_stamps_trace(self):
        """Index construction has no model, hence nothing to auto-detect —
        but ingest requires a manifest_id, so the model-less manifest must be
        registered anyway. Skipping it lost the trace to a 400."""
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="rag")
        _index_tree(h)

        trace = _flush_and_get_traces()[0]
        cfg._client.register_manifest.assert_called_once()
        assert trace.manifest_id == "m1"
        snap = cfg._client.register_manifest.call_args[0][0]
        assert snap.agent_name == "rag"
        assert not [c for c in snap.components if c.component_type == "model"]

    def test_retrieval_only_tree_gets_manifest(self):
        """The reported repro: `as_retriever().retrieve()` — retrieval spans,
        no model anywhere."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("ret", None, instance=Retriever(), parent_span_id=None)
        h.new_span("emb", None, instance=OpenAIEmbedding(), parent_span_id="ret")
        h.prepare_to_exit_span("emb", None, instance=OpenAIEmbedding(), result=[0.1])
        h.prepare_to_exit_span("ret", None, instance=Retriever(), result=["doc"])

        trace = _flush_and_get_traces()[0]
        assert trace.manifest_id == "m1"
        assert not trace.llm_calls

    def test_seen_model_is_sticky_so_llm_free_trees_dont_churn_the_manifest(self):
        """Once a model has been seen, a later model-less tree reuses that
        manifest instead of re-registering a model-less one — otherwise an app
        that interleaves `retrieve()` and `query()` bumps the manifest version
        on every single tree."""
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="rag")
        _llm_tree(h)
        _index_tree(h)

        traces = _flush_and_get_traces()
        assert len(traces) == 2
        # One registration total — the second tree hashed identically.
        cfg._client.register_manifest.assert_called_once()
        assert [t.manifest_id for t in traces] == ["m1", "m1"]
        snap = cfg._client.register_manifest.call_args[0][0]
        assert [c for c in snap.components if c.component_type == "model"]

    def test_first_model_promotes_the_manifest_exactly_once(self):
        """Model-less tree first, then a query: the manifest gains the model
        (one bump), and a third model-less tree does NOT drop it again."""
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="rag")
        _index_tree(h)
        _llm_tree(h)
        _index_tree(h, root="SentenceSplitter-2")

        traces = _flush_and_get_traces()
        assert len(traces) == 3
        assert cfg._client.register_manifest.call_count == 2
        assert all(t.manifest_id == "m1" for t in traces)

    def test_registration_failure_is_surfaced_and_retried(self):
        """A failed registration must (a) surface on export_status() as a
        MANIFEST error, not just the confusing trace-side 400, and (b) reset
        the hash tracker so the next tree retries — one blip must not poison
        every later trace in the process."""
        import decimalai._config as cfg

        class Boom(Exception):
            pass

        boom = Boom("backend down")
        cfg._client.register_manifest.side_effect = [boom, {"manifest_id": "m1"}]

        h = DecimalSpanHandler(agent_name="rag")
        _index_tree(h)
        _index_tree(h, root="SentenceSplitter-2")

        traces = _flush_and_get_traces()
        assert cfg._client.register_manifest.call_count == 2, (
            "the tracker was not reset, so registration was never retried"
        )
        assert traces[1].manifest_id == "m1"
        assert cfg._sender._last_manifest_error is boom


# ── 2. Parent hierarchy survives the flush ───────────────────────────

class TestSpanHierarchy:
    def test_nested_tree_carries_parent_span_ids(self):
        """root → retrieve → embed must store as a tree, not three flat spans."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("ret", None, instance=Retriever(), parent_span_id="root")
        h.new_span("emb", None, instance=OpenAIEmbedding(), parent_span_id="ret")
        h.prepare_to_exit_span("emb", None, instance=OpenAIEmbedding(), result=[0.1])
        h.prepare_to_exit_span("ret", None, instance=Retriever(), result=["doc"])
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="A")

        trace = _flush_and_get_traces()[0]
        by_name = {s.name: s for s in trace.spans}
        assert len(trace.spans) == 3
        assert by_name["QueryEngine"].parent_span_id is None
        assert by_name["Retriever"].parent_span_id == by_name["QueryEngine"].id
        assert by_name["OpenAIEmbedding"].parent_span_id == by_name["Retriever"].id

    def test_no_dangling_parent_references(self):
        """Every non-null parent_span_id resolves to a span in the same trace —
        an LLM span becomes an LlmCallRecord, so its children re-parent to the
        nearest ancestor that IS a span rather than pointing at nothing."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o-mini"), parent_span_id="root")
        h.new_span("split", None, instance=TokenTextSplitter(), parent_span_id="llm")
        h.prepare_to_exit_span("split", None, instance=TokenTextSplitter(), result="x")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(), result=_openai_result())
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="A")

        trace = _flush_and_get_traces()[0]
        ids = {s.id for s in trace.spans}
        by_name = {s.name: s for s in trace.spans}
        assert len(trace.spans) == 2  # the LLM span left for llm_calls
        assert by_name["TokenTextSplitter"].parent_span_id == by_name["QueryEngine"].id
        for s in trace.spans:
            assert s.parent_span_id is None or s.parent_span_id in ids

    def test_llm_call_links_to_its_enclosing_span(self):
        """The LlmCallRecord points at the span it ran under, so the call
        rejoins the tree it was lifted out of."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("syn", None, instance=TokenTextSplitter(), parent_span_id="root")
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o-mini"), parent_span_id="syn")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(), result=_openai_result())
        h.prepare_to_exit_span("syn", None, instance=TokenTextSplitter(), result="x")
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="A")

        trace = _flush_and_get_traces()[0]
        by_name = {s.name: s for s in trace.spans}
        assert trace.llm_calls[0].span_id == by_name["TokenTextSplitter"].id

    def test_only_the_root_span_is_parentless(self):
        """A 12-span RAG-shaped tree stores 11 parent links, not 0."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        parent = "root"
        for i in range(5):
            sid = f"n{i}"
            h.new_span(sid, None, instance=TokenTextSplitter(), parent_span_id=parent)
            parent = sid
        for i in reversed(range(5)):
            h.prepare_to_exit_span(f"n{i}", None, instance=TokenTextSplitter(), result="x")
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="A")

        trace = _flush_and_get_traces()[0]
        assert len(trace.spans) == 6
        assert sum(1 for s in trace.spans if s.parent_span_id is None) == 1


# ── 3. One real request → one LlmCallRecord ──────────────────────────

class TestLlmCallDeduplication:
    def test_wrapper_and_inner_call_collapse_into_one_record(self):
        """`OpenAI.predict` wrapping `OpenAI.chat` is ONE gpt-4o-mini request.
        The wrapper never sees the raw response, so the merged record must take
        its token counts from the inner call and its wall clock from the
        wrapper."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("OpenAI.predict-0", None,
                   instance=OpenAI(model="gpt-4o-mini", temperature=0.0),
                   parent_span_id="root")
        h.new_span("OpenAI.chat-0", None,
                   instance=OpenAI(model="gpt-4o-mini", temperature=0.0),
                   parent_span_id="OpenAI.predict-0")
        wrapper_started = h._spans["OpenAI.predict-0"]["started_at"]
        inner_started = h._spans["OpenAI.chat-0"]["started_at"]

        # The inner call carries the usage block; the wrapper returns a bare
        # string, exactly as OpenAI.predict does.
        h.prepare_to_exit_span("OpenAI.chat-0", None, instance=OpenAI(),
                               result=_openai_result(129, 11))
        h.prepare_to_exit_span("OpenAI.predict-0", None, instance=OpenAI(), result="A")
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="A")

        trace = _flush_and_get_traces()[0]
        assert len(trace.llm_calls) == 1, (
            f"one request must yield one record; got "
            f"{[(c.input_tokens, c.output_tokens) for c in trace.llm_calls]}"
        )
        call = trace.llm_calls[0]
        assert call.input_tokens == 129
        assert call.output_tokens == 11
        assert call.model_name == "gpt-4o-mini"
        assert call.provider == "openai"
        assert call.temperature == 0.0
        assert call.started_at == wrapper_started
        assert wrapper_started <= inner_started

    def test_sibling_llm_calls_are_not_collapsed(self):
        """Two genuine calls under a refine loop stay two records — dedup keys
        off nesting, never off 'looks similar'."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("refine", None, instance=TokenTextSplitter(), parent_span_id="root")
        for i in range(2):
            h.new_span(f"OpenAI.predict-{i}", None,
                       instance=OpenAI(model="gpt-4o-mini"), parent_span_id="refine")
            h.new_span(f"OpenAI.chat-{i}", None, instance=OpenAI(model="gpt-4o-mini"),
                       parent_span_id=f"OpenAI.predict-{i}")
            h.prepare_to_exit_span(f"OpenAI.chat-{i}", None, instance=OpenAI(),
                                   result=_openai_result(100 + i, 5 + i))
            h.prepare_to_exit_span(f"OpenAI.predict-{i}", None, instance=OpenAI(),
                                   result="A")
        h.prepare_to_exit_span("refine", None, instance=TokenTextSplitter(), result="A")
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="A")

        trace = _flush_and_get_traces()[0]
        assert len(trace.llm_calls) == 2
        assert sorted(c.input_tokens for c in trace.llm_calls) == [100, 101]

    def test_error_on_either_half_marks_the_merged_call_failed(self):
        """The inner call is what raises; the wrapper is what the user sees.
        Either erroring must produce one ERROR record, not a success."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("OpenAI.predict-0", None, instance=OpenAI(model="gpt-4o-mini"),
                   parent_span_id="root")
        h.new_span("OpenAI.chat-0", None, instance=OpenAI(model="gpt-4o-mini"),
                   parent_span_id="OpenAI.predict-0")
        h.prepare_to_drop_span("OpenAI.chat-0", None, instance=OpenAI(),
                               err=ValueError("rate limited"))
        h.prepare_to_exit_span("OpenAI.predict-0", None, instance=OpenAI(), result="A")
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="A")

        trace = _flush_and_get_traces()[0]
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].status.value == "error"


# ── 4. The declared llama-index-core floor is one number ─────────────


def test_llamaindex_core_floor_is_stated_consistently():
    """The runtime extra, the test extra, and the adapter's own ImportError
    must name the same floor. They said 0.10.20 / 0.12.0 / 0.12.0 — and at
    0.10.24 the dispatcher calls span_enter(instance, id=...), which raises
    TypeError straight through the user's query."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    floors = set(re.findall(r'"llama-index-core>=([0-9][^,"]*)"', pyproject))
    assert floors == {"0.12.0"}, f"pyproject states llama-index-core floors {floors}"

    adapter = (REPO_ROOT / "decimalai" / "llamaindex.py").read_text()
    assert set(re.findall(r"llama-index-core>=([0-9][0-9.]*)", adapter)) == {"0.12.0"}


# ── 5. End to end through the real dispatcher ────────────────────────

class TestRealDispatcherFidelity:
    def test_index_and_query_trees_all_ingestible_and_nested(self):
        """The full documented flow with llama-index installed: EVERY tree it
        produces — including the LLM-free index-construction ones — carries a
        manifest_id, the query tree nests, and one MockLLM call yields one
        record."""
        pytest.importorskip("llama_index.core")
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.llms import MockLLM

        from decimalai.llamaindex import instrument

        dispatcher = get_dispatcher()
        saved_handlers = list(dispatcher.span_handlers)
        instrument(agent_name="fidelity-rag")
        try:
            index = VectorStoreIndex.from_documents(
                [Document(text="The Eiffel Tower is 330 meters tall.")],
                embed_model=MockEmbedding(embed_dim=8),
            )
            index.as_query_engine(llm=MockLLM(), similarity_top_k=1).query(
                "How tall is the Eiffel Tower?"
            )
        finally:
            dispatcher.span_handlers = saved_handlers

        traces = _flush_and_get_traces()
        assert all(t.manifest_id for t in traces), (
            "an LLM-free tree shipped without a manifest_id — the backend "
            "rejects it with a 400 and the trace is lost"
        )

        query_traces = [t for t in traces if t.llm_calls]
        assert query_traces, "no trace captured the MockLLM call"
        trace = query_traces[-1]
        assert len(trace.llm_calls) == 1, (
            "MockLLM.predict wrapping MockLLM.chat is one call, not "
            f"{len(trace.llm_calls)}"
        )

        ids = {s.id for s in trace.spans}
        assert sum(1 for s in trace.spans if s.parent_span_id is None) == 1, (
            f"tree stored flat: {[(s.name, s.parent_span_id) for s in trace.spans]}"
        )
        for s in trace.spans:
            assert s.parent_span_id is None or s.parent_span_id in ids
