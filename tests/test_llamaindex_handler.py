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
    (input_tokens/output_tokens) — the shape the OpenAI fallback must catch.

    An object ``raw``. llama-index-llms-anthropic itself hands over a dict
    (``raw=dict(message)``): that shape is ``_claude_chat_response`` below."""
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


# ── Token usage, in the shapes the real provider bindings hand over ──
#
# Built from the providers' own response types, not look-alikes: the bug these
# guard was a look-alike (an object `raw`) standing in for what
# llama-index-llms-anthropic really returns (a dict `raw` holding a Usage
# OBJECT), so every Claude run traced 0 tokens while the fakes passed.


def _claude_message(input_tokens=260, output_tokens=35, **cache):
    """An ``anthropic.types.Message``. ``cache`` takes ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``, which Anthropic reports beside ``input_tokens``
    (the uncached remainder)."""
    anthropic_types = pytest.importorskip("anthropic.types")
    return anthropic_types.Message(
        id="msg_1", type="message", role="assistant", model="claude-haiku-5-5",
        content=[anthropic_types.TextBlock(type="text", text="330")],
        stop_reason="end_turn", stop_sequence=None,
        usage=anthropic_types.Usage(input_tokens=input_tokens, output_tokens=output_tokens,
                                    **cache),
    )


def _claude_chat_response(input_tokens=260, output_tokens=35, **cache):
    """What llama-index-llms-anthropic 0.13.0's ``chat()`` returns:
    ``AnthropicChatResponse(message=..., raw=dict(response))``. ``dict()`` of a
    pydantic model is shallow, so ``raw["usage"]`` is an ``anthropic.types.Usage``."""
    raw = dict(_claude_message(input_tokens, output_tokens, **cache))
    assert not isinstance(raw["usage"], dict)  # the shape that broke the dict branch
    return SimpleNamespace(
        raw=raw, additional_kwargs={},
        message=SimpleNamespace(content="330", additional_kwargs={}),
    )


def _openai_chat_response(prompt_tokens=120, completion_tokens=8, cached_tokens=None,
                          model="gpt-5.4-mini"):
    """What llama-index-llms-openai's ``chat()`` returns: ``raw`` is the
    ``openai.types.chat.ChatCompletion`` itself, and ``additional_kwargs``
    repeats the counts flat. ``cached_tokens`` goes where OpenAI reports a
    prompt-cache hit, ``usage.prompt_tokens_details`` — a part of ``prompt_tokens``."""
    chat = pytest.importorskip("openai.types.chat")
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens}
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    completion = chat.ChatCompletion.model_validate({
        "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": model,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "330"}}],
        "usage": usage,
    })
    return SimpleNamespace(
        raw=completion,
        additional_kwargs={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                           "total_tokens": prompt_tokens + completion_tokens},
        message=SimpleNamespace(content="330", additional_kwargs={}),
    )


def _extract(result) -> dict:
    span_data = {"input_tokens": None, "output_tokens": None, "model_name": None,
                 "output_text": None, "cache_read_tokens": None, "cache_creation_tokens": None}
    DecimalSpanHandler()._extract_llm_result(span_data, result)
    return span_data


class TestUsageShapes:
    def test_claude_dict_raw_with_usage_object(self):
        """llama-index-llms-anthropic's non-streamed response: the counts are read
        through a dict ``raw`` into the Usage object it holds."""
        got = _extract(_claude_chat_response(260, 35))
        assert (got["input_tokens"], got["output_tokens"]) == (260, 35)
        assert got["model_name"] == "claude-haiku-5-5"

    def test_claude_rag_run_trace_carries_the_counts(self):
        """The prober's LlamaIndex-on-Claude shape end to end: query → Claude
        synthesis call. The trace's one LLM call has what Anthropic billed."""
        h = DecimalSpanHandler(agent_name="claude-rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("llm", None, instance=Anthropic(model="claude-haiku-5-5"),
                   parent_span_id="root")
        h.prepare_to_exit_span("llm", None, instance=Anthropic(),
                               result=_claude_chat_response(260, 35))
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="330")

        trace = _flush_and_get_trace()
        assert len(trace.llm_calls) == 1
        call = trace.llm_calls[0]
        assert (call.input_tokens, call.output_tokens) == (260, 35)

    def test_bare_claude_call_still_sends_its_trace(self):
        """A bare ``llm.chat()``: the Claude call is the root of its own tree, so
        the flush follows the usage read in the same exit. Reading this shape used
        to raise there, and the trace was never sent."""
        h = DecimalSpanHandler(agent_name="claude-bare")
        h.new_span("llm", None, instance=Anthropic(model="claude-haiku-5-5"),
                   parent_span_id=None)
        h.prepare_to_exit_span("llm", None, instance=Anthropic(),
                               result=_claude_chat_response(41, 7))

        trace = _flush_and_get_trace()
        assert [(c.input_tokens, c.output_tokens) for c in trace.llm_calls] == [(41, 7)]
        assert h._spans == {}

    def test_claude_stream_last_event(self):
        """A streamed Claude call: the LLM end event carries the last chunk, whose
        ``raw`` is ``dict(RawMessageDeltaEvent)`` holding a MessageDeltaUsage
        object, and whose message repeats the counts as a dict."""
        anthropic_types = pytest.importorskip("anthropic.types")
        from anthropic.types.raw_message_delta_event import Delta

        event = anthropic_types.RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="end_turn", stop_sequence=None),
            usage=anthropic_types.MessageDeltaUsage(input_tokens=260, output_tokens=35),
        )
        chunk = SimpleNamespace(
            raw=dict(event), additional_kwargs={}, delta="",
            message=SimpleNamespace(
                content="330",
                additional_kwargs={"usage": {"input_tokens": 260, "output_tokens": 35},
                                   "stop_reason": "end_turn"}),
        )
        got = _extract(chunk)
        assert (got["input_tokens"], got["output_tokens"]) == (260, 35)

        # The message's own copy alone is enough, for a binding that drops raw.
        chunk.raw = None
        got = _extract(chunk)
        assert (got["input_tokens"], got["output_tokens"]) == (260, 35)

    def test_openai_chat_completion_object(self):
        """llama-index-llms-openai: an object ``raw`` (ChatCompletion → CompletionUsage)."""
        response = _openai_chat_response(120, 8)
        got = _extract(response)
        assert (got["input_tokens"], got["output_tokens"]) == (120, 8)
        assert got["model_name"] == "gpt-5.4-mini"

        # And from additional_kwargs alone, flat.
        response.raw = None
        got = _extract(response)
        assert (got["input_tokens"], got["output_tokens"]) == (120, 8)

    @pytest.mark.parametrize("usage", [
        {"prompt_tokens": 12, "completion_tokens": 3},
        {"input_tokens": 12, "output_tokens": 3},
        SimpleNamespace(prompt_tokens=12, completion_tokens=3),
        SimpleNamespace(input_tokens=12, output_tokens=3),
    ], ids=["dict-openai-names", "dict-anthropic-names", "object-openai-names",
            "object-anthropic-names"])
    def test_additional_kwargs_usage_dict_or_object(self, usage):
        """``additional_kwargs["usage"]`` is read whether it is a dict or an object."""
        got = _extract(SimpleNamespace(raw=None, additional_kwargs={"usage": usage}))
        assert (got["input_tokens"], got["output_tokens"]) == (12, 3)

    def test_additional_kwargs_holding_a_real_usage_object(self):
        anthropic_types = pytest.importorskip("anthropic.types")
        usage = anthropic_types.Usage(input_tokens=9, output_tokens=4)
        got = _extract(SimpleNamespace(raw=None, additional_kwargs={"usage": usage}))
        assert (got["input_tokens"], got["output_tokens"]) == (9, 4)

    @pytest.mark.parametrize("usage", [
        {"input_tokens": 5, "output_tokens": 2},
        SimpleNamespace(input_tokens=5, output_tokens=2),
    ], ids=["usage-dict", "usage-object"])
    def test_dict_response(self, usage):
        """A raw HTTP payload as the response, its ``usage`` a dict or an object."""
        got = _extract({"model": "claude-haiku-5-5", "usage": usage})
        assert (got["input_tokens"], got["output_tokens"]) == (5, 2)
        assert got["model_name"] == "claude-haiku-5-5"

    def test_zero_is_a_count(self):
        """0 output tokens is a real count, kept rather than skipped for the next name."""
        got = _extract(SimpleNamespace(raw={"usage": {"input_tokens": 7, "output_tokens": 0}}))
        assert (got["input_tokens"], got["output_tokens"]) == (7, 0)

    def test_first_source_wins_and_later_ones_only_fill_gaps(self):
        span_data = {"input_tokens": 260, "output_tokens": None, "model_name": None,
                     "output_text": None}
        DecimalSpanHandler()._extract_llm_result(
            span_data, SimpleNamespace(raw={"usage": {"input_tokens": 1, "output_tokens": 35}}))
        assert (span_data["input_tokens"], span_data["output_tokens"]) == (260, 35)

    @pytest.mark.parametrize("result", [
        SimpleNamespace(raw={"usage": "260 in"}),
        SimpleNamespace(raw={"usage": [260, 35]}),
        SimpleNamespace(raw=["not", "a", "response"]),
        SimpleNamespace(raw={"usage": {"input_tokens": "260", "output_tokens": True}}),
        SimpleNamespace(raw=MagicMock()),
        SimpleNamespace(raw=None, additional_kwargs={"usage": 260}),
        {"usage": 260, "model": 5},
    ], ids=["usage-str", "usage-list", "raw-list", "non-int-counts", "raw-mock",
            "ak-usage-int", "dict-junk"])
    def test_unexpected_shapes_give_no_counts_and_never_raise(self, result):
        got = _extract(result)
        assert (got["input_tokens"], got["output_tokens"]) == (None, None)
        assert got["model_name"] is None

    def test_a_raising_response_costs_the_counts_not_the_trace(self):
        """Whatever a response does on read, the run's trace still ships."""

        class Exploding:
            message = None

            @property
            def raw(self):
                raise RuntimeError("provider object broke on read")

        h = DecimalSpanHandler(agent_name="claude-bare")
        h.new_span("llm", None, instance=Anthropic(model="claude-haiku-5-5"),
                   parent_span_id=None)
        h.prepare_to_exit_span("llm", None, instance=Anthropic(), result=Exploding())

        trace = _flush_and_get_trace()
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].input_tokens is None


# ── Prompt-cache counts ──────────────────────────────────────────────
#
# A cached Claude prompt is billed in three parts: the uncached remainder
# (`input_tokens`), the cache read, and the cache write, each at its own price.
# Without the last two the trace prices a cached run as if only the remainder
# was sent. LlmCallRecord keeps each provider's own convention (see
# decimalai/schema/trace.py): Anthropic's cache counts are ADDITIONAL to
# input_tokens; OpenAI's cached_tokens is a PART of prompt_tokens.


def _prompt_counts(got) -> tuple:
    return (got["input_tokens"], got["cache_read_tokens"], got["cache_creation_tokens"])


class TestPromptCacheCounts:
    def test_claude_cache_counts_sit_beside_the_uncached_input(self):
        got = _extract(_claude_chat_response(
            60, 35, cache_read_input_tokens=200, cache_creation_input_tokens=0))
        assert _prompt_counts(got) == (60, 200, 0)  # a reported 0 stays 0: a cold cache
        assert got["output_tokens"] == 35

        # Not reported is not zero: no cache fields leaves both unmeasured.
        assert _prompt_counts(_extract(_claude_chat_response(60, 35))) == (60, None, None)

    def test_bare_claude_call_trace_carries_cache_counts(self):
        h = DecimalSpanHandler(agent_name="claude-bare")
        h.new_span("llm", None, instance=Anthropic(model="claude-haiku-5-5"),
                   parent_span_id=None)
        h.prepare_to_exit_span("llm", None, instance=Anthropic(), result=_claude_chat_response(
            60, 35, cache_read_input_tokens=200, cache_creation_input_tokens=1500))

        call = _flush_and_get_trace().llm_calls[0]
        assert (call.input_tokens, call.output_tokens) == (60, 35)
        assert (call.cache_read_tokens, call.cache_creation_tokens) == (200, 1500)

    def test_claude_stream_last_event_cache_counts(self):
        """A streamed Claude call's last chunk: ``raw`` is ``dict(RawMessageDeltaEvent)``
        and its MessageDeltaUsage carries the cumulative cache counts."""
        anthropic_types = pytest.importorskip("anthropic.types")
        from anthropic.types.raw_message_delta_event import Delta

        event = anthropic_types.RawMessageDeltaEvent(
            type="message_delta",
            delta=Delta(stop_reason="end_turn", stop_sequence=None),
            usage=anthropic_types.MessageDeltaUsage(
                input_tokens=60, output_tokens=35,
                cache_read_input_tokens=200, cache_creation_input_tokens=0),
        )
        chunk = SimpleNamespace(
            raw=dict(event), additional_kwargs={}, delta="",
            message=SimpleNamespace(
                content="330",
                additional_kwargs={"usage": {"input_tokens": 60, "output_tokens": 35}}),
        )
        assert _prompt_counts(_extract(chunk)) == (60, 200, 0)

    def test_wrapper_and_inner_call_record_the_inner_calls_cache_counts(self):
        """``complete()`` wraps ``chat()``: two spans, one call. The wrapper's own result
        has no usage, so the record's prompt counts come from the inner call."""
        h = DecimalSpanHandler(agent_name="claude-rag")
        h.new_span("root", None, instance=QueryEngine(), parent_span_id=None)
        h.new_span("outer", None, instance=Anthropic(model="claude-haiku-5-5"),
                   parent_span_id="root")
        h.new_span("inner", None, instance=Anthropic(model="claude-haiku-5-5"),
                   parent_span_id="outer")
        h.prepare_to_exit_span("inner", None, instance=Anthropic(), result=_claude_chat_response(
            60, 35, cache_read_input_tokens=200, cache_creation_input_tokens=7))
        h.prepare_to_exit_span("outer", None, instance=Anthropic(), result="330")
        h.prepare_to_exit_span("root", None, instance=QueryEngine(), result="330")

        calls = _flush_and_get_trace().llm_calls
        assert [(c.input_tokens, c.cache_read_tokens, c.cache_creation_tokens)
                for c in calls] == [(60, 200, 7)]

    def test_prompt_counts_come_from_one_usage(self):
        """Input and cache counts are taken together from one usage, never paired
        across two: an input count that already includes the cache (OpenAI's
        convention) beside Anthropic-style cache counts would bill the cache twice."""
        span_data = {"input_tokens": 264, "output_tokens": None, "model_name": None,
                     "output_text": None, "cache_read_tokens": None,
                     "cache_creation_tokens": None}
        DecimalSpanHandler()._extract_llm_result(span_data, _claude_chat_response(
            64, 35, cache_read_input_tokens=200, cache_creation_input_tokens=0))
        assert _prompt_counts(span_data) == (264, None, None)
        assert span_data["output_tokens"] == 35

        # The same response on a fresh span does carry the cache counts.
        assert _prompt_counts(_extract(_claude_chat_response(
            64, 35, cache_read_input_tokens=200, cache_creation_input_tokens=0))) == (64, 200, 0)

    def test_openai_cached_tokens_are_a_part_of_the_input(self):
        """generic.py's documented convention: ``prompt_tokens_details.cached_tokens``
        → cache_read_tokens, a SUBSET of input_tokens; no cache write is reported."""
        got = _extract(_openai_chat_response(120, 8, cached_tokens=64))
        assert _prompt_counts(got) == (120, 64, None)

        # A response that reports a cold cache says 0; one that reports nothing, None.
        assert _prompt_counts(_extract(_openai_chat_response(120, 8, cached_tokens=0))) == (
            120, 0, None)
        assert _prompt_counts(_extract(_openai_chat_response(120, 8))) == (120, None, None)

    def test_claude_through_an_openai_shaped_usage_keeps_claudes_convention(self):
        """Claude behind an OpenAI-compatible proxy reports OpenAI's shape: a
        prompt count with the cache inside it. A Claude record keeps Anthropic's
        convention (the platform prices ``claude-*`` that way), so the cache read
        comes OUT of the input instead of being billed a second time."""
        got = _extract(_openai_chat_response(
            264, 8, cached_tokens=200, model="anthropic/claude-haiku-5-5"))
        assert _prompt_counts(got) == (64, 200, None)


# ── A bare streamed call: held until its end event ───────────────────
#
# `llm.stream_chat()` with no query engine around it makes the LLM span the
# root, and it exits as soon as the generator is handed back. The counts come
# later, on LlamaIndex's LLMChatEndEvent / LLMCompletionEndEvent, which the
# dispatcher delivers to `record_llm_result` once the caller has consumed the
# stream. Flushing at exit shipped every such trace with no counts.


def _stream(*chunks):
    def gen():
        yield from chunks
    return gen()


def _sent():
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    return [c.args[0] for c in cfg._client.ingest_trace.call_args_list]


class TestBareLlmStream:
    def _start(self, h, span_id="llm", parent=None, result=None):
        h.new_span(span_id, None, instance=Anthropic(model="claude-haiku-5-5"),
                   parent_span_id=parent)
        h.prepare_to_exit_span(span_id, None, instance=Anthropic(), result=result)

    def test_trace_waits_for_the_end_event_and_carries_the_counts(self):
        h = DecimalSpanHandler(agent_name="claude-stream")
        stream = _stream("3", "30")
        self._start(h, result=stream)
        assert _sent() == []  # held: the counts have not arrived yet

        assert list(stream) == ["3", "30"]  # the caller's stream, untouched
        h.record_llm_result("llm", _claude_chat_response(
            60, 35, cache_read_input_tokens=200, cache_creation_input_tokens=0))

        [trace] = _sent()
        [call] = trace.llm_calls
        assert (call.input_tokens, call.output_tokens) == (60, 35)
        assert (call.cache_read_tokens, call.cache_creation_tokens) == (200, 0)
        assert trace.final_output_preview == "330"
        assert trace.ended_at >= call.started_at
        assert h._spans == {} and h._deferred_roots == set()

    @pytest.mark.parametrize("started", [False, True], ids=["never-started", "half-read"])
    def test_an_abandoned_stream_still_ships(self, started):
        """No end event ever comes for a stream the caller walks away from. The
        trace ships when the generator is collected — without counts, but not lost."""
        import gc

        h = DecimalSpanHandler(agent_name="claude-stream")
        stream = _stream("3", "30")
        self._start(h, result=stream)
        if started:
            assert next(stream) == "3"
        assert _sent() == []

        del stream
        gc.collect()
        [trace] = _sent()
        assert [(c.input_tokens, c.output_tokens) for c in trace.llm_calls] == [(None, None)]
        assert h._spans == {} and h._deferred_roots == set()

    def test_stream_complete_wrapping_stream_chat_sends_one_trace(self):
        """Anthropic's ``stream_complete`` wraps its ``stream_chat``: two LLM spans, both
        exiting with generators. The inner end event (it carries the usage) ships the
        trace; the outer one that follows finds nothing left to do."""
        h = DecimalSpanHandler(agent_name="claude-stream")
        outer, inner = _stream("330"), _stream("330")
        h.new_span("outer", None, instance=Anthropic(model="claude-haiku-5-5"),
                   parent_span_id=None)
        self._start(h, "inner", parent="outer", result=inner)
        h.prepare_to_exit_span("outer", None, instance=Anthropic(), result=outer)
        assert _sent() == []

        response = _claude_chat_response(60, 35)
        h.record_llm_result("inner", response)
        h.record_llm_result("outer", response)

        [trace] = _sent()
        assert [(c.input_tokens, c.output_tokens) for c in trace.llm_calls] == [(60, 35)]

    def test_a_stream_that_raises_ships_as_an_error(self):
        h = DecimalSpanHandler(agent_name="claude-stream")
        stream = _stream("3")
        self._start(h, result=stream)

        h.record_llm_exception("llm", RuntimeError("overloaded_error"))

        [trace] = _sent()
        assert trace.status.value == "error"
        assert trace.llm_calls[0].status.value == "error"
        assert h._spans == {}

    def test_a_stream_the_caller_closes_ships_as_a_success(self):
        """Closing a stream early raises GeneratorExit inside it, which LlamaIndex
        reports as an ExceptionEvent. The caller chose to stop: not an error."""
        h = DecimalSpanHandler(agent_name="claude-stream")
        stream = _stream("3", "30")
        self._start(h, result=stream)

        h.record_llm_exception("llm", GeneratorExit())

        [trace] = _sent()
        assert trace.status.value == "success"

    def test_async_stream_waits_for_the_end_event(self):
        import asyncio

        async def agen():
            yield "3"
            yield "30"

        h = DecimalSpanHandler(agent_name="claude-stream")
        stream = agen()
        self._start(h, result=stream)
        assert _sent() == []

        async def consume():
            return [chunk async for chunk in stream]

        assert asyncio.run(consume()) == ["3", "30"]
        h.record_llm_result("llm", _claude_chat_response(60, 35))

        [trace] = _sent()
        assert [(c.input_tokens, c.output_tokens) for c in trace.llm_calls] == [(60, 35)]

    def test_a_call_the_stream_starts_while_being_read_joins_its_tree(self):
        """A parentless span opened from inside the held generator's body is work
        the stream started lazily (CustomLLM's default ``astream_chat``): it joins
        the tree, so its counts are not shipped a second time as another trace."""
        h = DecimalSpanHandler(agent_name="claude-stream")

        def body():
            h.new_span("lazy", None, instance=Anthropic(model="claude-haiku-5-5"),
                       parent_span_id=None)
            h.prepare_to_exit_span("lazy", None, instance=Anthropic(),
                                   result=_claude_chat_response(60, 35))
            yield "330"

        stream = body()
        self._start(h, result=stream)
        assert list(stream) == ["330"]
        assert h._parents["lazy"] == "llm"
        h.record_llm_result("lazy", _claude_chat_response(60, 35))

        [trace] = _sent()
        assert [(c.input_tokens, c.output_tokens) for c in trace.llm_calls] == [(60, 35)]

    def test_another_threads_stream_does_not_adopt_this_threads_span(self):
        """On a server, thread A may be inside its held stream while thread B opens
        an unrelated call. "Being read" is judged on B's own stack, not the
        generator's running flag alone."""
        import threading

        h = DecimalSpanHandler(agent_name="claude-stream")
        started, release = threading.Event(), threading.Event()

        def body():
            started.set()
            release.wait(5)
            yield "330"

        stream = body()
        self._start(h, result=stream)
        reader = threading.Thread(target=lambda: next(stream))
        reader.start()
        try:
            assert started.wait(5)
            assert stream.gi_running  # thread A is inside it right now
            h.new_span("other", None, instance=Anthropic(model="claude-haiku-5-5"),
                       parent_span_id=None)
            assert h._parents["other"] is None
        finally:
            release.set()
            reader.join(5)
        h.prepare_to_exit_span("other", None, instance=Anthropic(),
                               result=_claude_chat_response(41, 7))
        assert [[(c.input_tokens, c.output_tokens) for c in t.llm_calls]
                for t in _sent()] == [[(41, 7)]]

    def test_close_ships_a_stream_still_in_flight(self):
        """Dispatcher shutdown calls ``close()``; a stream still being read ships then."""
        h = DecimalSpanHandler(agent_name="claude-stream")
        stream = _stream("3")
        self._start(h, result=stream)
        assert _sent() == []

        h.close()
        assert len(_sent()) == 1
        assert h._spans == {}
        del stream  # the finalizer has nothing left to send
        assert len(_sent()) == 1


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
