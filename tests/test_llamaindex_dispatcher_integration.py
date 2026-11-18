"""Integration tests: DecimalSpanHandler driven by LlamaIndex's REAL dispatcher.

test_llamaindex_handler.py drives the handler's new_span / prepare_to_* methods
directly — which stayed green while the shipped adapter captured nothing: the
dispatcher only ever invokes ``span_enter`` / ``span_exit`` / ``span_drop``
(each wrapped in ``except BaseException: pass``, so the AttributeError was
swallowed silently), and the handler didn't implement them. These tests close
that gap from both sides:

* with llama-index-core installed — a real MockLLM/MockEmbedding query-engine
  run through the root dispatcher must produce at least one ingested trace.
* without llama-index — the dispatcher-facing trio, called with the exact
  keyword convention the dispatcher uses (``parent_id=``, never
  ``parent_span_id=``), must delegate into the buffering logic.

Same mock-client capture pattern as test_llamaindex_handler.py.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from decimalai.llamaindex import DecimalSpanHandler


# ── Fakes for the no-llama-index tests (classification keys off type name) ──

class QueryEngine:  # → "query" → SpanType.AGENT (root)
    pass


class FakeLLM:  # "llm" in the name → is_llm_call; no .model attr on purpose
    # Mirrors llama_index.core.llms.MockLLM: no .model / .model_name attrs,
    # only the LLMMetadata every LlamaIndex LLM carries.
    metadata = SimpleNamespace(model_name="unknown")


# ── SDK reset (mirror test_llamaindex_handler._reset_sdk) ────────────

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
    assert cfg._client.ingest_trace.called, (
        "ingest_trace was never called — the dispatcher-facing "
        "span_enter/span_exit/span_drop methods are missing or broken"
    )
    return [c.args[0] for c in cfg._client.ingest_trace.call_args_list]


# ── Real dispatcher → real query engine → trace sent ─────────────────

class TestRealDispatcherQueryEngine:
    def test_mock_query_engine_sends_trace(self):
        """The documented flow — instrument() then a query-engine query — must
        send at least one trace through the REAL root dispatcher.

        This is the test whose absence let the adapter ship never having
        worked: the handler implemented only the prepare_* names, the
        dispatcher only calls the span_* trio, and it swallows the
        AttributeError — so the docs flow logged "installed", answered the
        query, and exported nothing.
        """
        pytest.importorskip("llama_index.core")
        from llama_index.core import Document, VectorStoreIndex
        from llama_index.core.embeddings import MockEmbedding
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.llms import MockLLM

        from decimalai.llamaindex import instrument

        # instrument() appends to the global root dispatcher and there is no
        # public uninstall — snapshot and restore so this handler can't capture
        # other tests' spans (same pattern as the live llamaindex test).
        dispatcher = get_dispatcher()
        saved_handlers = list(dispatcher.span_handlers)
        handler = instrument(agent_name="dispatcher-rag")
        try:
            index = VectorStoreIndex.from_documents(
                [Document(text="The Eiffel Tower is 330 meters tall.")],
                embed_model=MockEmbedding(embed_dim=8),
            )
            engine = index.as_query_engine(llm=MockLLM(), similarity_top_k=1)
            engine.query("How tall is the Eiffel Tower?")
        finally:
            dispatcher.span_handlers = saved_handlers

        # Index construction flushes its own (LLM-less) trees too; the query
        # trace is the one that captured the MockLLM synthesis call.
        traces = _flush_and_get_traces()
        query_traces = [t for t in traces if t.llm_calls]
        assert query_traces, (
            f"No trace captured the MockLLM call; got {len(traces)} trace(s) "
            f"with spans {[[s.name for s in t.spans] for t in traces]}"
        )
        trace = query_traces[-1]
        assert trace.agent_name == "dispatcher-rag"

        span_types = {s.span_type.value for s in trace.spans}
        assert "agent" in span_types      # RetrieverQueryEngine root
        assert "retrieval" in span_types  # VectorIndexRetriever

        # MockLLM has no .model attr — model_name comes via LLMMetadata,
        # which is also what lets the manifest register for this run.
        assert all(c.model_name == "unknown" for c in trace.llm_calls)
        assert trace.manifest_id == "m1"

        # Every tree flushed — nothing left for Dispatcher.shutdown() to drop.
        assert handler.open_spans == {}


# ── Real dispatcher → a bare streamed LLM call → its counts ──────────


def _usage_llm_class():
    """A CustomLLM that streams like a provider binding: the usage arrives on the
    LAST chunk only. No network; the real dispatcher and LLM callbacks drive it."""
    from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
    from llama_index.core.llms import CustomLLM
    from llama_index.core.llms.callbacks import llm_completion_callback

    def chunks():
        return [
            CompletionResponse(text="3", delta="3"),
            CompletionResponse(text="330", delta="30",
                               raw={"usage": {"input_tokens": 60, "output_tokens": 35,
                                              "cache_read_input_tokens": 200,
                                              "cache_creation_input_tokens": 0}}),
        ]

    class UsageLLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(model_name="claude-haiku-5-5")

        @llm_completion_callback()
        def complete(self, prompt, formatted=False, **kwargs):
            return chunks()[-1]

        @llm_completion_callback()
        def stream_complete(self, prompt, formatted=False, **kwargs):
            def gen():
                yield from chunks()
            return gen()

        # Natively async, as the provider bindings are. CustomLLM's own default
        # starts the sync stream lazily, from inside the async generator.
        @llm_completion_callback()
        async def astream_complete(self, prompt, formatted=False, **kwargs):
            async def gen():
                for chunk in chunks():
                    yield chunk
            return gen()

    return UsageLLM


class TestRealDispatcherBareStream:
    @pytest.fixture
    def instrumented(self):
        pytest.importorskip("llama_index.core")
        from llama_index.core.instrumentation import get_dispatcher

        from decimalai.llamaindex import instrument

        dispatcher = get_dispatcher()
        saved = (list(dispatcher.span_handlers), list(dispatcher.event_handlers))
        handler = instrument(agent_name="bare-stream")
        try:
            yield handler
        finally:
            dispatcher.span_handlers, dispatcher.event_handlers = saved

    @pytest.mark.parametrize(
        "call", ["stream_complete", "stream_chat", "astream_complete", "astream_chat"])
    def test_bare_stream_trace_carries_the_counts(self, instrumented, call):
        """One bare streamed call, one trace, with the counts from its last chunk.

        ``astream_chat`` is CustomLLM's default, which starts the sync stream lazily
        from inside the async generator, outside every span: that inner call must
        join the held tree rather than ship as a second trace with the same counts.
        """
        import asyncio

        from llama_index.core.llms import ChatMessage

        llm = _usage_llm_class()()
        messages = [ChatMessage(role="user", content="hi")]
        if call == "stream_complete":
            deltas = [r.delta for r in llm.stream_complete("hi")]
        elif call == "stream_chat":
            deltas = [r.delta for r in llm.stream_chat(messages)]
        else:
            async def consume():
                stream = await (llm.astream_complete("hi") if call == "astream_complete"
                                else llm.astream_chat(messages))
                return [r.delta async for r in stream]
            deltas = asyncio.run(consume())
        assert deltas == ["3", "30"]  # what the LLM yielded, unchanged

        traces = _flush_and_get_traces()
        assert len(traces) == 1, [[c.model_name for c in t.llm_calls] for t in traces]
        [call_record] = traces[0].llm_calls
        assert (call_record.input_tokens, call_record.output_tokens) == (60, 35)
        assert (call_record.cache_read_tokens, call_record.cache_creation_tokens) == (200, 0)
        assert instrumented.open_spans == {}


# ── Dispatcher calling convention (no llama-index needed) ────────────

class TestDispatcherCallingConvention:
    def test_span_trio_delegates_with_dispatcher_kwargs(self):
        """span_enter/span_exit accept the dispatcher's keyword convention
        (``parent_id=``, ``tags=``) and feed the buffering logic, so the tree
        assembles and the root exit flushes one trace."""
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="convention")
        h.span_enter(id_="root", bound_args=SimpleNamespace(query_str="Q"),
                     instance=QueryEngine(), parent_id=None, tags={})
        h.span_enter(id_="llm", bound_args=None, instance=FakeLLM(),
                     parent_id="root", tags=None)
        h.span_exit(id_="llm", bound_args=None, instance=FakeLLM(), result=None)
        h.span_exit(id_="root", bound_args=None, instance=QueryEngine(),
                    result=SimpleNamespace(response="A"))

        traces = _flush_and_get_traces()
        cfg._client.ingest_trace.assert_called_once()  # llm was a CHILD of root
        trace = traces[0]
        assert trace.agent_name == "convention"
        assert trace.user_input_preview == "Q"
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].model_name == "unknown"  # metadata fallback

    def test_span_drop_marks_error_and_flushes(self):
        h = DecimalSpanHandler(agent_name="convention")
        h.span_enter(id_="root", bound_args=None, instance=QueryEngine(),
                     parent_id=None)
        h.span_drop(id_="root", bound_args=None, instance=QueryEngine(),
                    err=RuntimeError("boom"))

        trace = _flush_and_get_traces()[0]
        assert trace.status.value == "error"

    def test_open_spans_and_close_shutdown_contract(self):
        """Dispatcher.shutdown() iterates ``handler.open_spans`` with no
        exception guard, span-drops each id, then calls ``close()`` — walk
        that exact sequence."""
        h = DecimalSpanHandler()
        h.span_enter(id_="root", bound_args=None, instance=QueryEngine(),
                     parent_id=None)
        assert "root" in h.open_spans

        for span_id in list(h.open_spans.keys()):
            h.span_drop(id_=span_id, bound_args=None, instance=None,
                        err=RuntimeError("dispatcher shutdown"))
        h.close()
        assert h.open_spans == {}
