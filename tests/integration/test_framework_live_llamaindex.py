"""Live-LLM — LlamaIndex through BOTH adapter paths.

LlamaIndex is a RAG-first framework, so the workload here is the idiom the
``DecimalSpanHandler`` was actually built to classify: a real
``VectorStoreIndex`` query engine. The query tree the handler sees is
``RetrieverQueryEngine`` (query → AGENT root) → ``VectorIndexRetriever``
(retrieve → RETRIEVAL) → embedding (OTHER) → synthesizer (OTHER) → the provider
LLM (llm → an ``LlmCallRecord``). Embeddings use ``MockEmbedding`` so the test
stays provider-agnostic and key-free on the embedding side; with
``similarity_top_k == len(docs)`` every doc is always retrieved, so the answer
is deterministic regardless of the (random) embedding vectors. Only the
*synthesis* LLM call is real — that's the call whose token shape + model id the
adapter must capture.

Two cells:
  * native — ``decimalai.llamaindex.install()`` registers the span handler on
    LlamaIndex's root dispatcher. Runs on all three providers, because the
    synthesis LLM is provider-specific and the handler must classify each
    vendor's LlamaIndex LLM class (OpenAI / Anthropic / GoogleGenAI) as an LLM
    call — the GoogleGenAI class in particular contains neither "llm" nor
    "gemini", so this is the live guard for that classification.
  * otel — the same query engine bridged through
    ``LlamaIndexInstrumentor`` (OpenInference) → ``DecimalSpanExporter``.
    OpenAI-only, mirroring the CrewAI OTEL cell: it bounds the OpenInference
    dependency surface to one canonical pairing.

Marker: live_llm + llamaindex (+ otel for the exporter cell).
Install the extra with ``pip install -e ".[llamaindex-tests]"``.
"""

from __future__ import annotations

import os

import pytest

from . import _live_helpers as h


# Three tiny facts; the Eiffel height is the one the query asks for. Short
# enough that CompactAndRefine fits them in a single synthesis prompt.
RAG_DOCS = [
    "The Eiffel Tower is located in Paris, France, and stands 330 meters tall.",
    "The Great Wall of China stretches for more than 21000 kilometers.",
    "Mount Everest rises to 8849 meters above sea level.",
]
RAG_QUERY = "How tall is the Eiffel Tower in meters? Reply with the number."
RAG_EXPECTED = "330"

# Provider → the import that must be present for that provider's LlamaIndex LLM.
_LLM_IMPORT = {
    "openai": "llama_index.llms.openai",
    "anthropic": "llama_index.llms.anthropic",
    "google": "llama_index.llms.google_genai",
}
# Provider → a substring that must appear in the recorded model id.
_MODEL_HINT = {"openai": "gpt", "anthropic": "claude", "google": "gemini"}


def _make_llamaindex_llm(provider: str, model: str):
    """Construct the provider's LlamaIndex LLM. Caller has already
    importorskip-ed the binding and checked the key."""
    if provider == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(model=model)
    if provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        # Anthropic requires max_tokens; LlamaIndex's wrapper surfaces it.
        return Anthropic(model=model, max_tokens=1024)
    if provider == "google":
        # google-genai authenticates with GOOGLE_API_KEY (newer builds fall back
        # to GEMINI_API_KEY). Mirror the gate's GEMINI_API_KEY across both and
        # force API-key mode (not Vertex). setdefault → never clobber.
        if os.environ.get("GEMINI_API_KEY"):
            os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "0")
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(model=model)
    raise ValueError(f"unknown provider {provider!r}")


def _build_query_engine(llm):
    """A tiny VectorStoreIndex query engine over RAG_DOCS, embeddings mocked so
    the test needs no embedding key and stays deterministic."""
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding

    docs = [Document(text=t) for t in RAG_DOCS]
    index = VectorStoreIndex.from_documents(docs, embed_model=MockEmbedding(embed_dim=8))
    # top_k == len(docs): every doc is retrieved, so the answer doesn't depend on
    # the (random) mock embedding similarity — only on the real synthesis LLM.
    return index.as_query_engine(llm=llm, similarity_top_k=len(RAG_DOCS))


def _assert_rag_trace(detail: dict, provider: str) -> None:
    """Shared across both adapter paths: the synthesis LLM call landed in
    llm_calls, carries the right vendor's model id, and the trace got an
    auto-detected manifest."""
    llm_calls = detail.get("llm_calls", [])
    assert llm_calls, (
        f"Trace {detail['id']} has no llm_calls — the synthesis LLM call wasn't "
        f"captured/classified. For google this is the GoogleGenAI-classification "
        f"regression. spans={detail.get('spans')}"
    )
    models = " ".join(
        str(c.get("model_name") or c.get("model") or "") for c in llm_calls
    ).lower()
    hint = _MODEL_HINT[provider]
    assert hint in models, (
        f"Expected model hint {hint!r} in recorded llm_calls models {models!r}. "
        f"Trace id={detail['id']}"
    )
    assert detail.get("manifest_id"), "manifest_id missing — auto-detection failed"


# ═══════════════════════════════════════════════════════════════════
# Native path — decimalai.llamaindex span handler
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.llamaindex
@pytest.mark.parametrize("provider, model", h.matrix("llamaindex"))
def test_llamaindex_query_engine_native(provider, model):
    """A real LlamaIndex RAG query on each provider → the native span handler →
    one backend trace whose synthesis LLM call is captured with the vendor's
    model id."""
    h.require_key_for(provider)
    pytest.importorskip("llama_index.core")
    pytest.importorskip(_LLM_IMPORT[provider])

    from llama_index.core.instrumentation import get_dispatcher
    from decimalai.llamaindex import install

    agent_name = h.unique_agent(f"llamaindex-{provider}-rag")

    # install() appends a handler to the *global* root dispatcher and there's no
    # public uninstall, so snapshot the handler list and restore it after — keeps
    # this cell's handler from capturing the next cell's spans under the wrong
    # agent_name.
    dispatcher = get_dispatcher()
    saved_handlers = list(dispatcher.span_handlers)
    install(agent_name=agent_name)
    try:
        llm = _make_llamaindex_llm(provider, model)
        query_engine = _build_query_engine(llm)
        answer = str(query_engine.query(RAG_QUERY))
    finally:
        dispatcher.span_handlers = saved_handlers

    assert RAG_EXPECTED in answer.replace(",", ""), (
        f"Query engine didn't surface {RAG_EXPECTED!r} from context: {answer!r}"
    )

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    _assert_rag_trace(detail, provider)


# ═══════════════════════════════════════════════════════════════════
# OTEL path — OpenInference LlamaIndexInstrumentor → DecimalSpanExporter
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.live_llm
@pytest.mark.llamaindex
@pytest.mark.otel
@pytest.mark.parametrize("provider, model", h.matrix("llamaindex", only=("openai",)))
def test_llamaindex_query_engine_otel(provider, model):
    """The same RAG query, bridged through OpenInference's LlamaIndexInstrumentor
    into the DecimalSpanExporter → one backend trace with the synthesis LLM call.

    OpenAI-only (like the CrewAI OTEL cell): bounds the OpenInference dependency
    surface to a single canonical pairing — the OTEL ingest path itself is
    provider-independent, proven across vendors by the CrewAI/ADK/generic cells.
    """
    h.require_key_for(provider)
    pytest.importorskip("llama_index.core")
    pytest.importorskip(_LLM_IMPORT[provider])
    pytest.importorskip("openinference.instrumentation.llama_index")
    pytest.importorskip("opentelemetry.sdk")

    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from decimalai.otel import DecimalSpanExporter

    agent_name = h.unique_agent(f"llamaindex-{provider}-otel")

    # Local tracer provider with our exporter — same wiring decimalai.init(otel=True)
    # installs globally, but scoped to this test so it can't leak across the matrix
    # (OTEL honors set_tracer_provider only once per process).
    provider_otel = TracerProvider(
        resource=Resource.create({SERVICE_NAME: "decimal-agent"})
    )
    exporter = DecimalSpanExporter(agent_name=agent_name)
    provider_otel.add_span_processor(BatchSpanProcessor(exporter))

    instrumentor = LlamaIndexInstrumentor()
    instrumentor.instrument(tracer_provider=provider_otel)
    try:
        llm = _make_llamaindex_llm(provider, model)
        query_engine = _build_query_engine(llm)
        answer = str(query_engine.query(RAG_QUERY))
        # Multi-span RAG run may straddle batch-flush intervals — force a flush so
        # the exporter buffers + finalizes the whole tree into one RunTrace.
        provider_otel.force_flush()
    finally:
        instrumentor.uninstrument()
        provider_otel.shutdown()

    assert RAG_EXPECTED in answer.replace(",", ""), (
        f"Query engine didn't surface {RAG_EXPECTED!r} from context: {answer!r}"
    )

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])
    _assert_rag_trace(detail, provider)
