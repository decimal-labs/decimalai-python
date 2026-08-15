"""Regression tests for the LlamaIndex follow-on defects found 2026-08-15,
once traces from the adapter actually started reaching the backend.

Each test names a defect that was live against the local dev backend:

  1. P0 — `_safe_preview` fell through to `str(obj)` on a `StreamingResponse`,
     whose `__str__` DRAINS `response_gen`. Tracing ate the user's stream and
     the app received nothing.
  2. `_classify_span` decided "is this an LLM?" from substrings of the class
     name, so Ollama / Cohere / Mistral / Groq / Bedrock / Vertex calls were
     never recorded and those agents' manifests stayed permanently model-less.
  3. `VectorStoreIndex.from_documents` shipped TWO `source_type="production"`
     traces per build (a splitter tree and an embedding tree) that then showed
     up as replay episodes in the compatibility report.
  4. `dispatcher.shutdown()` flushes the root while its children are still
     open, and ingest rejects the whole trace: "spans[1]: 'ended_at' is
     required".
  5. Every span was named for its class alone, so a 12-span RAG tree read as
     `RetrieverQueryEngine, RetrieverQueryEngine, VectorIndexRetriever, ...`.
  6. A model-less run registered a model-less manifest, which the diff engine
     read as `provider: '' → 'openai'` — breaking/major, "replay everything" —
     once a query in the same process observed a model.

Same mock-client capture pattern as test_llamaindex_handler.py.
"""

from __future__ import annotations

import gc
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from decimalai.llamaindex import (
    DecimalSpanHandler,
    _classify_span,
    _classify_span_by_name,
    _detect_provider,
    _get_span_name,
    _safe_preview,
)


# ── Fake LlamaIndex instances (name-based classification fallback) ───

class QueryEngine:
    pass


class Retriever:
    pass


class SentenceSplitter:
    pass


class OpenAIEmbedding:
    pass


class OpenAI:
    def __init__(self, model=None, temperature=None):
        self.model = model
        self.temperature = temperature


class NamelessLLM:
    """An LLM wrapper exposing neither `.model` nor `LLMMetadata.model_name`."""


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
    cfg._client.list_manifests.return_value = {"manifests": []}
    yield


def _flush_and_get_traces():
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    return [c.args[0] for c in cfg._client.ingest_trace.call_args_list]


# ── 1. A tracer must never consume what it observes ──────────────────

class TestStreamingIsNotConsumed:
    def test_preview_does_not_drain_a_streaming_response(self):
        """THE P0. `str(StreamingResponse)` drains response_gen into
        response_txt; the app then gets an empty stream."""
        pytest.importorskip("llama_index.core")
        from llama_index.core.base.response.schema import StreamingResponse

        resp = StreamingResponse(response_gen=iter(["Hello", " ", "world"]))
        assert _safe_preview(resp) is None

        assert "".join(resp.response_gen) == "Hello world"

    def test_preview_does_not_drain_a_bare_generator(self):
        """`llm.stream_chat` returns a generator straight into span_exit."""
        def chunks():
            yield "a"
            yield "b"

        gen = chunks()
        assert _safe_preview(gen) is None
        assert list(gen) == ["a", "b"]

    def test_preview_reads_already_materialized_stream_text(self):
        """Once the owner has consumed the stream, the text is free to read."""
        pytest.importorskip("llama_index.core")
        from llama_index.core.base.response.schema import StreamingResponse

        resp = StreamingResponse(response_gen=iter([]), response_txt="the answer")
        assert _safe_preview(resp) == "the answer"

    def test_bound_args_holding_a_stream_are_not_drained(self):
        """A synthesizer is CALLED with a streaming response — the input
        preview must not eat it either."""
        pytest.importorskip("llama_index.core")
        from llama_index.core.base.response.schema import StreamingResponse

        resp = StreamingResponse(response_gen=iter(["x", "y"]))
        bound = SimpleNamespace(arguments={"response": resp})
        assert _safe_preview(bound) is None
        assert "".join(resp.response_gen) == "xy"

    def test_streamed_query_flushes_after_delivery_with_the_text(self):
        """The trace waits for the stream, then carries what was streamed —
        flushing at query() return shipped an empty answer."""
        pytest.importorskip("llama_index.core")
        import decimalai._config as cfg
        from decimalai._config import _sender
        from llama_index.core.base.response.schema import StreamingResponse

        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", SimpleNamespace(query_str="What is the revenue?"),
                   instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o-mini"),
                   parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(), result=iter(()))

        resp = StreamingResponse(response_gen=iter(["Revenue ", "was ", "$1M"]))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result=resp)

        _sender.flush()
        assert not cfg._client.ingest_trace.called, (
            "flushed before the stream was delivered — no output, no tokens"
        )

        assert "".join(resp.response_gen) == "Revenue was $1M"

        trace = _flush_and_get_traces()[0]
        assert trace.final_output_preview == "Revenue was $1M"

    def test_abandoned_stream_still_ships_its_trace(self):
        """The caller drops the response without reading it — the trace must
        not be held forever."""
        pytest.importorskip("llama_index.core")
        from llama_index.core.base.response.schema import StreamingResponse

        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        resp = StreamingResponse(response_gen=iter(["never read"]))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result=resp)

        del resp
        gc.collect()

        assert len(_flush_and_get_traces()) == 1

    def test_close_flushes_a_stream_still_in_flight(self):
        """Dispatcher shutdown while a stream is undelivered still ships."""
        pytest.importorskip("llama_index.core")
        from llama_index.core.base.response.schema import StreamingResponse

        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        resp = StreamingResponse(response_gen=iter(["a", "b"]))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result=resp)

        h.close()
        assert len(_flush_and_get_traces()) == 1

    def test_shutdown_drop_of_a_waiting_tree_keeps_it_successful(self):
        """`shutdown()` span-drops everything the handler holds, which includes
        the finished spans of a tree merely waiting on its stream. That must
        not restamp a successful run as failed — nor ship it twice."""
        pytest.importorskip("llama_index.core")
        from llama_index.core.base.response.schema import StreamingResponse

        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        resp = StreamingResponse(response_gen=iter(["a"]))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result=resp)

        h.prepare_to_drop_span("root", None, instance=None,
                               err=RuntimeError("dispatcher shutdown"))
        h.close()
        del resp
        gc.collect()

        traces = _flush_and_get_traces()
        assert len(traces) == 1
        assert traces[0].status.value == "success"

    def test_real_streaming_query_delivers_every_token_and_one_trace(self):
        """The documented flow, through the real dispatcher."""
        pytest.importorskip("llama_index.core")
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.llms import MockLLM

        from decimalai.llamaindex import instrument

        dispatcher = get_dispatcher()
        saved_span, saved_event = (list(dispatcher.span_handlers),
                                   list(dispatcher.event_handlers))
        instrument(agent_name="stream-rag")
        try:
            index = VectorStoreIndex.from_documents(
                [Document(text="The Eiffel Tower is 330 meters tall.")],
                embed_model=MockEmbedding(embed_dim=8),
            )
            resp = index.as_query_engine(
                llm=MockLLM(max_tokens=6), streaming=True, similarity_top_k=1,
            ).query("How tall is it?")
            streamed = "".join(resp.response_gen)
        finally:
            dispatcher.span_handlers = saved_span
            dispatcher.event_handlers = saved_event

        assert streamed.strip(), "tracing ate the stream — the app got nothing"

        traces = _flush_and_get_traces()
        assert len(traces) == 1, f"expected the query tree only, got {len(traces)}"
        assert traces[0].final_output_preview == streamed[:300]


# ── 2. Classification is structural, not a name match ────────────────

class TestStructuralClassification:
    @pytest.mark.parametrize("name", ["Ollama", "Cohere", "MistralAI", "Groq",
                                      "BedrockConverse", "Vertex"])
    def test_non_matching_llm_class_names_are_still_llms(self, name):
        """"ollama" contains no "llm" — every one of these was filed as a
        generic span, so the call never reached llm_calls and the agent's
        manifest never learned it had a model."""
        pytest.importorskip("llama_index.core")
        from llama_index.core.llms import MockLLM

        cls = type(name, (MockLLM,), {})
        instance = cls()

        assert _classify_span_by_name(instance) == "other", (
            f"{name} unexpectedly matches the name heuristic — pick another"
        )
        assert _classify_span(instance) == "llm"

    def test_embedding_models_are_not_llms(self):
        """BaseEmbedding is resolved before BaseLLM so an embedding model can
        never be filed as a completion."""
        pytest.importorskip("llama_index.core")
        from llama_index.core.embeddings import MockEmbedding

        assert _classify_span(MockEmbedding(embed_dim=4)) == "embed"

    def test_real_query_engine_and_retriever_classify_structurally(self):
        pytest.importorskip("llama_index.core")
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.llms import MockLLM

        index = VectorStoreIndex.from_documents(
            [Document(text="x")], embed_model=MockEmbedding(embed_dim=4),
        )
        assert _classify_span(index.as_retriever()) == "retrieve"
        assert _classify_span(index.as_query_engine(llm=MockLLM())) == "query"

    def test_provider_falls_back_to_the_integration_package_name(self):
        """`llama_index.llms.<provider>` names any provider the explicit list
        has never heard of."""
        cls = type("Foo", (), {})
        cls.__module__ = "llama_index.llms.groq.base"
        assert _detect_provider(cls()) == "groq"

    def test_known_provider_strings_are_unchanged(self):
        """The explicit list wins over the module path: flipping "google" to
        "google_genai" under a live agent would itself diff as a breaking
        provider change."""
        cls = type("GoogleGenAI", (), {})
        cls.__module__ = "llama_index.llms.google_genai.base"
        assert _detect_provider(cls()) == "google"


# ── 3. Index construction is setup, not a run ────────────────────────

class TestSetupTreesAreNotRuns:
    def test_index_construction_tree_is_not_traced(self):
        """A splitter tree carries no query, no retrieval and no model. It
        used to ship as a production run and then be replayed as a
        compatibility episode."""
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("SentenceSplitter.__call__-0", None, instance=SentenceSplitter(),
                   parent_span_id=None)
        h.new_span("SentenceSplitter._parse_nodes-0", None, instance=SentenceSplitter(),
                   parent_span_id="SentenceSplitter.__call__-0")
        h.prepare_to_exit_span("SentenceSplitter._parse_nodes-0", None,
                               instance=SentenceSplitter(), result=["node"])
        h.prepare_to_exit_span("SentenceSplitter.__call__-0", None,
                               instance=SentenceSplitter(), result=["node"])

        assert _flush_and_get_traces() == []
        cfg._client.register_manifest.assert_not_called()
        assert h._spans == {}

    def test_index_embedding_batch_tree_is_not_traced(self):
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("emb-batch", None, instance=OpenAIEmbedding(), parent_span_id=None)
        h.new_span("emb-one", None, instance=OpenAIEmbedding(), parent_span_id="emb-batch")
        h.prepare_to_exit_span("emb-one", None, instance=OpenAIEmbedding(), result=[0.1])
        h.prepare_to_exit_span("emb-batch", None, instance=OpenAIEmbedding(), result=[[0.1]])

        assert _flush_and_get_traces() == []

    def test_bare_retrieval_is_still_a_run(self):
        """One retrieval span anywhere keeps the tree — `retrieve()` is a run
        even though it uses no model."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("ret", None, instance=Retriever(), parent_span_id=None)
        h.new_span("emb", None, instance=OpenAIEmbedding(), parent_span_id="ret")
        h.prepare_to_exit_span("emb", None, instance=OpenAIEmbedding(), result=[0.1])
        h.prepare_to_exit_span("ret", None, instance=Retriever(), result=["doc"])

        assert len(_flush_and_get_traces()) == 1

    def test_real_index_build_emits_no_traces(self):
        pytest.importorskip("llama_index.core")
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.instrumentation import get_dispatcher

        from decimalai.llamaindex import instrument

        dispatcher = get_dispatcher()
        saved_span, saved_event = (list(dispatcher.span_handlers),
                                   list(dispatcher.event_handlers))
        instrument(agent_name="setup-rag")
        try:
            VectorStoreIndex.from_documents(
                [Document(text="The Eiffel Tower is 330 meters tall.")],
                embed_model=MockEmbedding(embed_dim=8),
            )
        finally:
            dispatcher.span_handlers = saved_span
            dispatcher.event_handlers = saved_event

        assert _flush_and_get_traces() == [], (
            "index construction shipped junk production traces"
        )


# ── 4. Nothing an unfinished span does may sink the whole trace ──────

class TestIngestibleUnderShutdown:
    def test_shutdown_with_open_children_closes_every_span(self):
        """`dispatcher.shutdown()` drops the ROOT first (open_spans is
        insertion-ordered), flushing while the children still run. Ingest
        rejected the whole trace: "spans[1]: 'ended_at' is required"."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("child", None, instance=Retriever(), parent_span_id="root")

        h.prepare_to_drop_span("root", None, instance=None,
                               err=RuntimeError("dispatcher shutdown"))

        trace = _flush_and_get_traces()[0]
        assert len(trace.spans) == 2
        assert all(s.ended_at is not None for s in trace.spans), (
            "an unfinished span shipped with ended_at=None — ingest 400s"
        )
        assert trace.status.value == "error"

    def test_nameless_model_does_not_sink_the_trace(self):
        """Ingest rejects the whole trace over one nameless call
        ("llm_calls[i]: 'model_name' is required")."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=NamelessLLM(), parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=NamelessLLM(), result="hi")
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="hi")

        trace = _flush_and_get_traces()[0]
        assert [c.model_name for c in trace.llm_calls] == ["unknown"]
        # ...and the placeholder must not leak into the declared contract.
        import decimalai._config as cfg
        snap = cfg._client.register_manifest.call_args[0][0]
        assert not [c for c in snap.components if c.component_type == "model"]


# ── 5. Span names carry the method ───────────────────────────────────

class TestSpanNames:
    def test_name_comes_from_the_dispatcher_span_id(self):
        assert _get_span_name(
            QueryEngine(), None,
            span_id="RetrieverQueryEngine.query-47e3dd42-d703-4346-9d8e-1bec2b66a346",
        ) == "RetrieverQueryEngine.query"

    def test_name_falls_back_to_the_class_without_a_uuid_suffix(self):
        assert _get_span_name(QueryEngine(), None, span_id="root") == "QueryEngine"
        assert _get_span_name(QueryEngine(), None) == "QueryEngine"

    def test_sibling_spans_on_one_instance_are_distinguishable(self):
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111",
                   None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("RetrieverQueryEngine._query-22222222-2222-2222-2222-222222222222",
                   None, instance=QueryEngine(),
                   parent_span_id="RetrieverQueryEngine.query-11111111-1111-1111-1111-111111111111")
        h.new_span("VectorIndexRetriever.retrieve-33333333-3333-3333-3333-333333333333",
                   None, instance=Retriever(),
                   parent_span_id="RetrieverQueryEngine._query-22222222-2222-2222-2222-222222222222")
        for sid in reversed(list(h._spans)):
            h.prepare_to_exit_span(sid, None, instance=None, result="x")

        trace = _flush_and_get_traces()[0]
        names = sorted(s.name for s in trace.spans)
        assert names == [
            "RetrieverQueryEngine._query",
            "RetrieverQueryEngine.query",
            "VectorIndexRetriever.retrieve",
        ]


# ── 6. "Nothing to declare" must not read as "the model was removed" ─

class TestManifestDoesNotChurn:
    def test_model_less_run_adopts_the_agents_existing_manifest(self):
        """A fresh process whose first run declares no model must NOT register
        a model-less manifest: it hash-matches the agent's own earlier
        model-less version, the backend treats that as a REVERT, and the next
        query flips it straight back — a round trip per process."""
        import decimalai._config as cfg

        cfg._client.list_manifests.return_value = {
            "manifests": [
                {"id": "m-live", "status": "active"},
                {"id": "m-old", "status": "superseded"},
            ]
        }

        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("ret", None, instance=Retriever(), parent_span_id=None)
        h.prepare_to_exit_span("ret", None, instance=Retriever(), result=["doc"])

        trace = _flush_and_get_traces()[0]
        assert trace.manifest_id == "m-live"
        cfg._client.register_manifest.assert_not_called()

    def test_adoption_is_attempted_once_then_the_manifest_is_reused(self):
        import decimalai._config as cfg

        cfg._client.list_manifests.return_value = {
            "manifests": [{"id": "m-live", "status": "active"}]
        }

        h = DecimalSpanHandler(agent_name="rag")
        for i in range(3):
            h.new_span(f"ret{i}", None, instance=Retriever(), parent_span_id=None)
            h.prepare_to_exit_span(f"ret{i}", None, instance=Retriever(), result=["d"])

        traces = _flush_and_get_traces()
        assert [t.manifest_id for t in traces] == ["m-live"] * 3
        assert cfg._client.list_manifests.call_count == 1

    def test_brand_new_agent_still_gets_a_manifest(self):
        """Nothing to adopt: register the minimal manifest rather than lose the
        trace to the ingest gate."""
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("ret", None, instance=Retriever(), parent_span_id=None)
        h.prepare_to_exit_span("ret", None, instance=Retriever(), result=["doc"])

        trace = _flush_and_get_traces()[0]
        assert trace.manifest_id == "m1"
        cfg._client.register_manifest.assert_called_once()

    def test_a_real_model_change_is_still_registered(self):
        """The anti-churn rule must not blind the adapter to actual drift."""
        import decimalai._config as cfg

        cfg._client.register_manifest.side_effect = [
            {"manifest_id": "m1"}, {"manifest_id": "m2"},
        ]

        h = DecimalSpanHandler(agent_name="rag")
        for i, model in enumerate(("gpt-4o-mini", "gpt-4o")):
            h.new_span(f"root{i}", None, instance=QueryEngine(), parent_span_id=None)
            h.new_span(f"llm{i}", None, instance=OpenAI(model=model),
                       parent_span_id=f"root{i}")
            h.prepare_to_exit_span(f"llm{i}", None, instance=OpenAI(), result="ok")
            h.prepare_to_exit_span(f"root{i}", None, instance=QueryEngine(), result="ok")

        traces = _flush_and_get_traces()
        assert [t.manifest_id for t in traces] == ["m1", "m2"]
        assert cfg._client.register_manifest.call_count == 2


# ── 7. Token usage arrives with the LLM end event ────────────────────

class TestStreamedTokenUsage:
    def test_end_event_stamps_usage_on_a_streamed_call(self):
        """A streamed span exits with a generator, so its return value carries
        no usage at all; the dispatcher's LLM end event is the only place the
        finished response appears."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o-mini"),
                   parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(), result=iter(()))

        h.record_llm_result("llm", SimpleNamespace(
            raw=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=31,
                                                      completion_tokens=7)),
            additional_kwargs={},
        ))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="done")

        call = _flush_and_get_traces()[0].llm_calls[0]
        assert (call.input_tokens, call.output_tokens) == (31, 7)

    def test_return_value_does_not_overwrite_event_usage_with_none(self):
        """Both sources reach the same span; whichever lands second must fill
        gaps, not clobber real numbers."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o-mini"),
                   parent_span_id="root")
        h.record_llm_result("llm", SimpleNamespace(
            raw=SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12,
                                                      completion_tokens=3)),
        ))
        # The traced function returns a usage-free object afterwards.
        h.prepare_to_exit_span("llm", None, instance=OpenAI(),
                               result=SimpleNamespace(raw=None, additional_kwargs={}))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="done")

        call = _flush_and_get_traces()[0].llm_calls[0]
        assert (call.input_tokens, call.output_tokens) == (12, 3)

    def test_usage_in_additional_kwargs_is_read(self):
        """OpenAI puts the counts flat on additional_kwargs; the old `elif`
        chain could never reach that branch when `.raw` existed."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o-mini"),
                   parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(), result=SimpleNamespace(
            raw=SimpleNamespace(model="gpt-4o-mini"),  # no .usage
            additional_kwargs={"prompt_tokens": 14, "completion_tokens": 5},
        ))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="done")

        call = _flush_and_get_traces()[0].llm_calls[0]
        assert (call.input_tokens, call.output_tokens) == (14, 5)

    def test_event_handler_is_registered_on_the_dispatcher(self):
        pytest.importorskip("llama_index.core")
        from llama_index.core.instrumentation import get_dispatcher

        from decimalai.llamaindex import instrument

        dispatcher = get_dispatcher()
        saved_span, saved_event = (list(dispatcher.span_handlers),
                                   list(dispatcher.event_handlers))
        try:
            instrument(agent_name="events-rag")
            assert any(type(h).__name__ == "DecimalLLMEventHandler"
                       for h in dispatcher.event_handlers)
        finally:
            dispatcher.span_handlers = saved_span
            dispatcher.event_handlers = saved_event
