"""Unit tests for the LlamaIndex span handler (decimalai.llamaindex).

Drives DecimalSpanHandler with *synthetic* spans — no llama-index install,
no network, no API key. The handler's new_span / prepare_to_exit_span /
prepare_to_drop_span methods take plain ids + duck-typed instances, so we can
reconstruct a query→retrieve→llm tree by hand and assert on the RunTrace that
lands at the (mocked) backend client.

Mirrors the trace-capture pattern in test_otel_exporter.py: a mock client on
the global config, then read the RunTrace from ingest_trace.call_args.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from decimalai.llamaindex import (
    DecimalSpanHandler,
    _classify_span,
    _detect_provider,
    _get_span_name,
    _safe_preview,
)


# ── Fake LlamaIndex instances — classification keys off type name ────
#
# The handler classifies a span from type(instance).__name__.lower(), so the
# class *names* here are what matter (a "Retriever" → retrieval span, etc.).

class QueryEngine:  # → "query" → SpanType.AGENT (root)
    pass


class Retriever:  # → "retrieve" → SpanType.RETRIEVAL
    pass


class ResponseSynthesizer:  # → "synthesize" → SpanType.OTHER
    pass


class OpenAIEmbedding:  # → "embed" (embed wins over openai) → SpanType.OTHER
    pass


class OpenAI:  # fake LLM → is_llm_call, provider=openai
    def __init__(self, model=None, temperature=None):
        self.model = model
        self.temperature = temperature


class Anthropic:  # fake LLM → is_llm_call, provider=anthropic
    def __init__(self, model=None, temperature=None):
        self.model = model
        self.temperature = temperature


class GoogleGenAI:  # current LlamaIndex Google LLM class → is_llm_call, provider=google
    def __init__(self, model=None, temperature=None):
        self.model = model
        self.temperature = temperature


def _openai_result(prompt_tokens=10, completion_tokens=20, model="gpt-4o"):
    """A LlamaIndex-style ChatResponse with .raw.usage (OpenAI token names)."""
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(raw=SimpleNamespace(usage=usage, model=model), response="ok")


def _anthropic_result(input_tokens=100, output_tokens=50, model="claude-haiku-4-5"):
    """A response whose .raw.usage uses Anthropic token names
    (input_tokens/output_tokens) — the shape the OpenAI fallback must catch."""
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(raw=SimpleNamespace(usage=usage, model=model))


# ── SDK reset (mirror test_otel_exporter._reset_sdk) ─────────────────

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
    # The handler registers a manifest from the run's model;
    # return a real dict so the manifest_id stamped on the trace is valid.
    cfg._client.register_manifest.return_value = {"manifest_id": "m1"}
    yield


def _flush_and_get_trace():
    """Drain the background sender and return the single captured RunTrace."""
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    assert cfg._client.ingest_trace.called, "ingest_trace was never called"
    return cfg._client.ingest_trace.call_args[0][0]


# ── Tree assembly → RunTrace ─────────────────────────────────────────

class TestSpanHandlerTree:
    def test_query_tree_produces_one_trace(self):
        """root QueryEngine + Retriever + LLM → one RunTrace with the LLM in
        llm_calls and the query/retrieve spans typed correctly.

        This is the regression guard for the eager-dict bug: before the fix,
        flushing any non-LLM span raised AttributeError on SpanType.RETRIEVER.
        """
        h = DecimalSpanHandler(agent_name="rag-agent")

        h.new_span("root", SimpleNamespace(query_str="What is the revenue?"),
                   instance=QueryEngine(), parent_span_id=None)
        h.new_span("ret", None, instance=Retriever(), parent_span_id="root")
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o", temperature=0.7),
                   parent_span_id="root")

        # Exit children first, root last (root exit triggers the flush).
        h.prepare_to_exit_span("llm", None, instance=OpenAI(),
                               result=_openai_result(10, 20, "gpt-4o"))
        h.prepare_to_exit_span("ret", None, instance=Retriever(), result=["doc1", "doc2"])
        h.prepare_to_exit_span("root", None, instance=QueryEngine(),
                               result=SimpleNamespace(response="Revenue was $1M"))

        import decimalai._config as cfg
        trace = _flush_and_get_trace()
        cfg._client.ingest_trace.assert_called_once()

        assert trace.agent_name == "rag-agent"
        assert trace.status.value == "success"
        assert trace.user_input_preview == "What is the revenue?"
        assert trace.final_output_preview == "Revenue was $1M"

        # One LLM call, captured with model/provider/tokens/temperature.
        assert len(trace.llm_calls) == 1
        llm = trace.llm_calls[0]
        assert llm.model_name == "gpt-4o"
        assert llm.provider == "openai"
        assert llm.input_tokens == 10
        assert llm.output_tokens == 20
        assert llm.temperature == 0.7

        # Non-LLM spans: the query root (agent) + the retriever (retrieval).
        span_types = {s.span_type.value for s in trace.spans}
        assert "agent" in span_types
        assert "retrieval" in span_types
        assert len(trace.spans) == 2

    def test_llm_run_registers_manifest_and_stamps_trace(self):
        """A run with an LLM model registers a manifest and stamps
        the resulting manifest_id on the emitted trace, so manifest diff/compat
        gating engages for LlamaIndex agents."""
        import decimalai._config as cfg

        h = DecimalSpanHandler(agent_name="rag-agent")
        h.new_span("root", SimpleNamespace(query_str="Q"),
                   instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o", temperature=0.0),
                   parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(),
                               result=_openai_result(5, 7, "gpt-4o"))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(),
                               result=SimpleNamespace(response="A"))

        trace = _flush_and_get_trace()
        cfg._client.register_manifest.assert_called()
        assert trace.manifest_id == "m1"
        snap = cfg._client.register_manifest.call_args[0][0]
        assert any(c.component_type == "model" for c in snap.components)

    def test_synthesize_span_maps_to_other(self):
        """A ResponseSynthesizer span flushes as SpanType.OTHER (it used to
        reference the nonexistent SpanType.CHAIN and crash the flush)."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("syn", None, instance=ResponseSynthesizer(), parent_span_id="root")
        h.prepare_to_exit_span("syn", None, instance=ResponseSynthesizer(), result="answer")
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="answer")

        trace = _flush_and_get_trace()
        syn = [s for s in trace.spans if s.name == "ResponseSynthesizer"]
        assert len(syn) == 1
        assert syn[0].span_type.value == "other"

    def test_embed_span_maps_to_other(self):
        """An embedding span flushes as OTHER (no embedding SpanType exists)."""
        h = DecimalSpanHandler()
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("emb", None, instance=OpenAIEmbedding(), parent_span_id="root")
        h.prepare_to_exit_span("emb", None, instance=OpenAIEmbedding(), result=[0.1, 0.2])
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="x")

        trace = _flush_and_get_trace()
        emb = [s for s in trace.spans if s.name == "OpenAIEmbedding"]
        assert emb and emb[0].span_type.value == "other"

    def test_anthropic_token_shape_extracted(self):
        """LLM result with .raw.usage.input_tokens/output_tokens (Anthropic
        naming) is captured via the OpenAI→Anthropic fallback in
        _extract_llm_result."""
        h = DecimalSpanHandler(agent_name="claude-rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=Anthropic(model="claude-haiku-4-5"),
                   parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=Anthropic(),
                               result=_anthropic_result(100, 50))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="done")

        trace = _flush_and_get_trace()
        assert len(trace.llm_calls) == 1
        llm = trace.llm_calls[0]
        assert llm.provider == "anthropic"
        assert llm.input_tokens == 100
        assert llm.output_tokens == 50

    def test_google_genai_llm_lands_in_llm_calls(self):
        """A `GoogleGenAI` LLM span is classified as an LLM call (not 'other')
        and records provider='google'. Regression guard for the live google
        native cell: without the classification fix this span would flush as a
        generic OTHER TraceSpan and the trace would carry zero llm_calls."""
        h = DecimalSpanHandler(agent_name="gemini-rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=GoogleGenAI(model="gemini-3.5-flash"),
                   parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=GoogleGenAI(),
                               result=_openai_result(7, 11, "gemini-3.5-flash"))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="done")

        trace = _flush_and_get_trace()
        assert len(trace.llm_calls) == 1, (
            f"GoogleGenAI span did not land in llm_calls; spans={trace.spans}"
        )
        llm = trace.llm_calls[0]
        assert llm.provider == "google"
        assert llm.model_name == "gemini-3.5-flash"

    def test_llm_model_falls_back_to_raw_model(self):
        """When the instance carries no model attr, model_name comes from
        result.raw.model."""
        h = DecimalSpanHandler()
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model=None), parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(),
                               result=_openai_result(model="gpt-4o-mini"))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="x")

        trace = _flush_and_get_trace()
        assert trace.llm_calls[0].model_name == "gpt-4o-mini"

    def test_error_span_marks_trace_error(self):
        """A dropped (errored) child span propagates ERROR to the whole trace."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o"), parent_span_id="root")
        h.prepare_to_drop_span("llm", None, instance=OpenAI(),
                               err=ValueError("boom"))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="x")

        trace = _flush_and_get_trace()
        assert trace.status.value == "error"
        assert trace.llm_calls[0].status.value == "error"

    def test_errored_root_still_flushes(self):
        """Dropping the root span flushes the (failed) trace rather than
        silently discarding it."""
        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.prepare_to_drop_span("root", None, instance=QueryEngine(),
                               err=RuntimeError("query failed"))

        trace = _flush_and_get_trace()
        assert trace.status.value == "error"

    def test_nested_three_level_tree_single_flush(self):
        """root → synthesize → llm: only the root exit flushes, and the flush
        carries every span in the tree exactly once."""
        h = DecimalSpanHandler(agent_name="deep")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("syn", None, instance=ResponseSynthesizer(), parent_span_id="root")
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o"), parent_span_id="syn")

        h.prepare_to_exit_span("llm", None, instance=OpenAI(), result=_openai_result())
        import decimalai._config as cfg
        from decimalai._config import _sender
        # Child exits must NOT flush — only the root does.
        _sender.flush()
        assert not cfg._client.ingest_trace.called
        h.prepare_to_exit_span("syn", None, instance=ResponseSynthesizer(), result="a")
        _sender.flush()
        assert not cfg._client.ingest_trace.called

        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="a")
        trace = _flush_and_get_trace()
        cfg._client.ingest_trace.assert_called_once()
        assert len(trace.llm_calls) == 1
        assert len(trace.spans) == 2  # root (agent) + synthesize (other)

    def test_buffers_cleaned_after_flush(self):
        """After a tree flushes, the handler's internal buffers are empty so
        it doesn't leak spans across runs."""
        h = DecimalSpanHandler()
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o"), parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(), result=_openai_result())
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="x")
        _flush_and_get_trace()

        assert h._spans == {}
        assert h._trees == {}
        assert h._parents == {}

    def test_disabled_sdk_skips_send(self):
        """With the SDK disabled, no trace is sent and buffers still clear."""
        import decimalai._config as cfg
        cfg._config.enabled = False

        h = DecimalSpanHandler(agent_name="rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=OpenAI(model="gpt-4o"), parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=OpenAI(), result=_openai_result())
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="x")

        from decimalai._config import _sender
        _sender.flush()
        cfg._client.ingest_trace.assert_not_called()
        assert h._spans == {}  # cleanup still ran


# ── Pure helper functions ────────────────────────────────────────────

class TestSpanHandlerHelpers:
    def test_classify_span(self):
        assert _classify_span(QueryEngine()) == "query"
        assert _classify_span(Retriever()) == "retrieve"
        assert _classify_span(ResponseSynthesizer()) == "synthesize"
        assert _classify_span(OpenAIEmbedding()) == "embed"
        assert _classify_span(OpenAI()) == "llm"
        assert _classify_span(Anthropic()) == "llm"
        # Current Google LLM class `GoogleGenAI` — has neither "llm" nor "gemini"
        # in its name, so this guards the google-cell classification fix.
        assert _classify_span(GoogleGenAI()) == "llm"
        assert _classify_span(None) == "other"
        assert _classify_span(object()) == "other"

    def test_detect_provider(self):
        assert _detect_provider(OpenAI()) == "openai"
        assert _detect_provider(Anthropic()) == "anthropic"
        # GoogleGenAI is the real current class; detection must work off the class
        # name alone (the fake's __module__ isn't llama_index.llms.google_genai).
        assert _detect_provider(GoogleGenAI()) == "google"

        class Gemini:
            pass

        class Cohere:
            pass

        assert _detect_provider(Gemini()) == "google"
        assert _detect_provider(Cohere()) == "cohere"
        assert _detect_provider(object()) is None

    def test_get_span_name(self):
        assert _get_span_name(QueryEngine(), None) == "QueryEngine"

        def my_func():
            pass

        assert _get_span_name(None, my_func) == "my_func"
        assert _get_span_name(None, None) == "LlamaIndexOperation"

    def test_safe_preview(self):
        assert _safe_preview(None) is None
        assert _safe_preview(SimpleNamespace(query_str="hello")) == "hello"
        assert _safe_preview(SimpleNamespace(response="the answer")) == "the answer"
        assert _safe_preview(SimpleNamespace(content="a message")) == "a message"

        # Truncation at max_len with an ellipsis.
        long = _safe_preview("x" * 500, max_len=50)
        assert long is not None and long.endswith("…") and len(long) == 51

    def test_safe_preview_bound_args_first_value(self):
        """A BoundArguments-like object previews its first argument value."""
        ba = SimpleNamespace(arguments={"query": "find docs", "k": 5})
        assert _safe_preview(ba) == "find docs"
