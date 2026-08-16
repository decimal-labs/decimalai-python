"""LlamaIndex driver.

Runs the snippet documented at ``decimalai-docs/sdk/python/frameworks/llamaindex.mdx``:
``decimalai.init(..., llamaindex=True)`` — i.e. ``decimalai.llamaindex.instrument()``
— followed by a ``VectorStoreIndex`` query engine call. Embeddings are
``MockEmbedding`` and the LLM is a stub, so every phase runs with no key and no
network, exactly like the LangChain reference driver.

Three things about this adapter shape the driver, and all three are stated here
rather than being quietly worked around:

**The span handler is process-wide, so a second agent is named per RUN, not per
install.** ``instrument()`` appends a ``DecimalSpanHandler`` to LlamaIndex's
ROOT dispatcher; there is no per-call handler. So the driver installs exactly
once, on the first ``run``, as a user following the docs would. Calling it again
is not a way out and never was: ``instrument()`` is not idempotent, and after a
second call ONE query engine call posts TWO traces, one under each agent name
(measured, not assumed). The documented way to name a second agent is
``decimalai.providers.agent_run(...)`` — the same run scope the raw-provider and
Pydantic AI rails use, documented for LlamaIndex at
``decimalai-docs/sdk/python/frameworks/llamaindex.mdx`` under "Several agents in
one process". Every phase wraps its run in that scope with its own ctx's agent
name, which is exactly what a service handling two tenants' traffic would do,
and it is what C6 (``second_agent``) and C9 (per-lane names) grade.

**Streaming is the interesting path.** ``query(..., streaming=True)`` returns
while the answer is still arriving, so the adapter defers the flush and swaps a
pass-through tee into ``response.response_gen``. ``run`` therefore issues both
queries — plain, then streamed, consuming the stream the way an application
does — and the contract grades the streamed trace like any other: it must carry
the completion text (C4) and token counts (C3), which for a streamed call can
only arrive from LlamaIndex's ``LLMCompletionEndEvent`` *after* the caller has
drained the generator. Whether the *application* also received every token is
not observable on the wire; see the README note this driver's report adds.

**There is no skills rail.** The docs capability table records LlamaIndex's
skills-rail column as "—", and ``instrument()`` takes ``agent_name`` and nothing
else, so C8 is declared N/A with that reason.

NO ASSERTIONS BELOW THIS LINE. That is the driver contract.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator

from . import (
    STUB_MODEL_NAME,
    Capabilities,
    Ctx,
    Driver,
    DriverError,
    fanout_threads,
    stub_script,
    tool_result,
    user_message,
)

#: The one process-wide span handler, installed on first use. LlamaIndex's
#: dispatcher takes handlers additively and never gives them back, so a second
#: install would double-trace every query from then on.
_HANDLER: Any = None
_HANDLER_LOCK = threading.Lock()


# ── the stub model ───────────────────────────────────────────────────────────


def _stub_llm(ctx: Ctx, *, fail: bool = False) -> Any:
    """Map the shared stub script onto llama-index-core's LLM interface.

    A query engine makes exactly one synthesis call, so the script is taken
    without its tool turn; the retrieved document plays the tool's part (see
    ``_index``). Token counts ride on ``additional_kwargs``, which is where
    LlamaIndex normalises usage for several real providers and where the
    adapter's ``_extract_llm_result`` looks for it.
    """
    from llama_index.core.base.llms.types import (
        CompletionResponse,
        CompletionResponseGen,
        LLMMetadata,
    )
    from llama_index.core.llms import CustomLLM
    from llama_index.core.llms.callbacks import llm_completion_callback

    turn = stub_script(ctx, use_tool=False)[0]
    usage = {
        "prompt_tokens": turn.input_tokens,
        "completion_tokens": turn.output_tokens,
    }

    class StubLLM(CustomLLM):
        fail: bool = False

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=8192,
                num_output=64,
                model_name=STUB_MODEL_NAME,
                is_chat_model=False,
            )

        @llm_completion_callback()
        def complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponse:
            if self.fail:
                raise DriverError("conformance: the model failed on purpose")
            return CompletionResponse(text=turn.content, additional_kwargs=dict(usage))

        @llm_completion_callback()
        def stream_complete(
            self, prompt: str, formatted: bool = False, **kwargs: Any
        ) -> CompletionResponseGen:
            if self.fail:
                raise DriverError("conformance: the model failed on purpose")

            def _gen() -> Iterator[CompletionResponse]:
                # Two chunks, LlamaIndex's convention: `text` accumulates,
                # `delta` is the increment. Split so a tracer that reads only
                # the first chunk cannot pass C4 by accident.
                text = turn.content
                cut = max(1, len(text) // 2)
                acc = ""
                for piece in (text[:cut], text[cut:]):
                    acc += piece
                    yield CompletionResponse(
                        text=acc, delta=piece, additional_kwargs=dict(usage)
                    )

            return _gen()

    return StubLLM(fail=fail)


# ── the documented snippet ───────────────────────────────────────────────────


def _instrument_once(ctx: Ctx) -> Any:
    """``decimalai.init(..., llamaindex=True)``, once per process."""
    global _HANDLER
    with _HANDLER_LOCK:
        if _HANDLER is None:
            from decimalai.llamaindex import instrument

            _HANDLER = instrument(agent_name=ctx.agent_name)
    return _HANDLER


def _run_scope(ctx: Ctx) -> Any:
    """The documented per-run scope: ``with agent_run("..."):``.

    One process, one installed handler, one name per run — the shape the docs
    give for serving several agents (or several concurrent requests) from one
    LlamaIndex process.
    """
    from decimalai.providers import agent_run

    return agent_run(ctx.agent_name)


def _index(ctx: Ctx) -> Any:
    """A one-document ``VectorStoreIndex``. Embeddings mocked — no key, no network.

    The document carries ``tool_result(ctx, ...)``, so the retrieval step plays
    the part the tool call plays in the other drivers: the sentinel the run is
    supposed to fetch is in the retrieved text, not in the model's head.
    """
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.embeddings import MockEmbedding

    docs = [Document(text=tool_result(ctx, ctx.prompt_sentinel))]
    return VectorStoreIndex.from_documents(docs, embed_model=MockEmbedding(embed_dim=8))


def _query(ctx: Ctx, index: Any, *, streaming: bool = False, fail: bool = False) -> Any:
    engine = index.as_query_engine(
        llm=_stub_llm(ctx, fail=fail), similarity_top_k=1, streaming=streaming
    )
    return engine.query(user_message(ctx))


def _plain_query(ctx: Ctx) -> Any:
    """One documented query-engine call, under its own run scope — one trace."""
    _instrument_once(ctx)
    with _run_scope(ctx):
        return _query(ctx, _index(ctx))


def run(ctx: Ctx) -> Any:
    """The documented snippet, in both the plain and the streamed form.

    Two traces per call. The streamed half is not decoration: the adapter holds
    the whole tree open until the stream is delivered, so it is the only place
    the deferred flush, the pass-through tee and the post-hoc token stamping are
    exercised at all.
    """
    _instrument_once(ctx)
    with _run_scope(ctx):
        index = _index(ctx)

        plain = _query(ctx, index)

        streamed = _query(ctx, index, streaming=True)
        # Drain it the way an app does. If the tracer consumed the generator
        # instead of teeing it, this comes back empty while the trace still
        # looks complete — which is exactly why the note above says the wire
        # cannot see that case.
        delivered = "".join(chunk for chunk in streamed.response_gen)
    return plain, delivered


def run_error(ctx: Ctx) -> Any:
    """The same query engine, with the synthesis model raising."""
    _instrument_once(ctx)
    with _run_scope(ctx):
        return _query(ctx, _index(ctx), fail=True)


def run_degenerate(ctx: Ctx) -> Any:
    """Nothing to trace, then nothing to declare — LlamaIndex's two model-less runs.

    Building an index is index-time plumbing, not an agent run: the adapter
    classifies that tree as setup and ships nothing. A bare ``retrieve()`` IS a
    run, and it is the case the adapter's manifest logic is written around — no
    model was observed, so it must reuse the manifest it already has rather than
    declare a model-less one, which the diff engine would read as the model
    having been deleted.
    """
    _instrument_once(ctx)
    with _run_scope(ctx):
        index = _index(ctx)  # setup tree — expected to reach the wire as nothing
        retriever = index.as_retriever(similarity_top_k=1)
        return retriever.retrieve(user_message(ctx))


DRIVER = Driver(
    name="llamaindex",
    covers=frozenset({"llamaindex"}),
    requires=("llama_index.core",),
    entrypoint="decimalai.llamaindex.instrument()",
    run=run,
    # One plain query per lane, so the lane count and the trace count line up
    # and C9's verdict is about isolation rather than about `run` emitting two.
    run_concurrent=fanout_threads(_plain_query),
    run_error=run_error,
    run_degenerate=run_degenerate,
    capabilities=Capabilities(
        # Gates C5 only. A RAG query engine has no tool calls; the multi-step
        # structure C5 grades comes from query → retrieve → synthesize → llm,
        # and the retrieved document carries the sentinel a tool would.
        has_tools=True,
        has_skills_rail=False,
        supports_concurrency=True,
        supports_error_path=True,
        supports_degenerate=True,
        reasons={
            "has_skills_rail": (
                "the adapter has no skills rail — decimalai.llamaindex.instrument() "
                "takes agent_name and nothing else, and the docs capability table "
                "records LlamaIndex's skills-rail column as '—'. Skills reach a "
                "LlamaIndex agent only by hand, via "
                "SkillRouter.build_prompt_fragment()"
            ),
        },
    ),
)
