"""LlamaIndex integration for DecimalAI.

Provides a ``DecimalSpanHandler`` that plugs into LlamaIndex's
instrumentation dispatcher (v0.12.0+) to capture query engine,
retriever, LLM, and embedding spans as DecimalAI traces.

Usage::

    import decimalai
    decimalai.init(api_key="...")

    from decimalai.llamaindex import instrument
    instrument(agent_name="my-rag-agent")

    # Then use LlamaIndex as normal — traces are auto-captured
    from llama_index.core import VectorStoreIndex
    index = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine().query("What is the revenue?")

Alternative — if you prefer the OTEL path (requires more packages)::

    pip install openinference-instrumentation-llama-index
    decimalai.init(api_key="...", otel=True)
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    LlamaIndexInstrumentor().instrument()

Requires: ``pip install "decimalai[llamaindex]"``
"""

from __future__ import annotations

import inspect
import logging
import re
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger("decimalai.llamaindex")

# Span types that mean "this tree did agent work". A tree containing none of
# them is index-time plumbing, not a run — see ``_is_setup_tree``.
_RUN_SPAN_TYPES = frozenset({"query", "retrieve", "synthesize", "llm"})


# ── LlamaIndex span handler ──────────────────────────────────────


class DecimalSpanHandler:
    """LlamaIndex SpanHandler that converts spans into DecimalAI traces.

    Buffers spans by their root span ID, then assembles and sends a
    complete RunTrace when the root span exits.

    Integrates with LlamaIndex's instrumentation dispatcher, which drives
    handlers exclusively through ``span_enter()`` / ``span_exit()`` /
    ``span_drop()``. Those delegate to the buffering logic:
    - ``new_span()`` → start buffering
    - ``prepare_to_exit_span()`` → if root, flush → RunTrace
    - ``prepare_to_drop_span()`` → clean up on error
    """

    def __init__(self, agent_name: Optional[str] = None):
        import threading

        from .schema.manifest import ManifestTracker

        self.agent_name = agent_name or "llamaindex-agent"
        # span_id → span data dict
        self._spans: Dict[str, Dict[str, Any]] = {}
        # root_span_id → list of child span_ids (including root)
        self._trees: Dict[str, List[str]] = {}
        # span_id → parent_span_id
        self._parents: Dict[str, Optional[str]] = {}
        # Manifest auto-detection state: version the model config
        # so manifest diff / compat gating engages for LlamaIndex (RAG) agents.
        self._manifest_tracker = ManifestTracker()
        self._manifest_id: Optional[str] = None
        self._manifest_lock = threading.Lock()
        # The model config seen on ANY tree so far. Sticky: index construction
        # and bare retrieval have no model, and letting the manifest drop back
        # to model-less between two query trees would churn the manifest
        # version on every such tree without any real config change.
        self._seen_model: Optional[Dict[str, Any]] = None
        # True once this process holds a manifest_id the backend acknowledged
        # (registered or adopted). Distinct from `_manifest_id`, which is also
        # set to the client-side id when registration FAILED — that case must
        # keep retrying.
        self._manifest_confirmed = False
        self._adoption_attempted = False
        # Roots whose flush is waiting on a streamed response to be delivered.
        self._deferred_roots: set[str] = set()

    def class_name(self) -> str:
        """Return the class name for LlamaIndex's handler registry."""
        return "DecimalSpanHandler"

    # ── Dispatcher-facing interface ──────────────────────────────
    #
    # The dispatcher never calls new_span/prepare_to_* directly — it only
    # invokes span_enter/span_exit/span_drop, and it wraps each call in
    # ``except BaseException: pass``, so a missing method is swallowed
    # silently. Without these three delegates the handler receives nothing.

    def span_enter(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Dispatcher notice that a span started — delegates to ``new_span``."""
        self.new_span(
            id_=id_, bound_args=bound_args, instance=instance,
            parent_span_id=parent_id, tags=tags,
        )

    def span_exit(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Dispatcher notice that a span completed — delegates to
        ``prepare_to_exit_span``."""
        self.prepare_to_exit_span(
            id_=id_, bound_args=bound_args, instance=instance, result=result,
        )

    def span_drop(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> None:
        """Dispatcher notice that a span errored — delegates to
        ``prepare_to_drop_span``."""
        self.prepare_to_drop_span(
            id_=id_, bound_args=bound_args, instance=instance, err=err,
        )

    @property
    def open_spans(self) -> Dict[str, Dict[str, Any]]:
        """Spans buffered but not yet flushed, keyed by span id.

        ``Dispatcher.shutdown()`` iterates ``handler.open_spans`` with no
        exception guard (then span-drops each and calls ``close()``), so
        the attribute is part of the handler contract.
        """
        return self._spans

    def close(self) -> None:
        """Dispatcher shutdown hook — ship any tree still awaiting a stream.

        Trees normally flush as their roots exit; a tree whose root returned a
        live stream flushes from the pass-through wrapper instead (see
        ``_defer_for_stream``). If the process is shutting down with a stream
        still undelivered, ship what we have rather than lose the trace.
        """
        for root_id in list(self._deferred_roots):
            self._deferred_roots.discard(root_id)
            try:
                self._flush_tree(root_id)
            except Exception:  # pragma: no cover - shutdown is best-effort
                logger.debug("Failed to flush deferred tree %s on close", root_id,
                             exc_info=True)

    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """Called when a new span starts.

        Args:
            id_: The span ID assigned by LlamaIndex.
            bound_args: The bound arguments to the function being traced.
            instance: The object instance (e.g., QueryEngine, LLM).
            parent_span_id: Parent span ID, or None if this is root.
            tags: Optional tags.

        Returns:
            The span ID (passthrough).
        """
        span_data: Dict[str, Any] = {
            "id": id_,
            "parent_span_id": parent_span_id,
            "started_at": datetime.now(timezone.utc),
            "ended_at": None,
            # Classify the span type from the instance
            "span_type": _classify_span(instance),
            "name": _get_span_name(instance, bound_args, span_id=id_),
            "status": "running",
            "input_preview": _safe_preview(bound_args),
            "output_preview": None,
            # LLM-specific fields
            "model_name": None,
            "provider": None,
            "input_tokens": None,
            "output_tokens": None,
            "temperature": None,
            "is_llm_call": False,
        }

        # Extract LLM metadata only for true chat/completion spans. Key off the
        # already-computed span_type so embeddings (OpenAIEmbedding, etc.) — which
        # _classify_span resolves to "embed" — are NOT treated as LLM calls. A
        # broader name heuristic here would misroute them into LlmCallRecord and
        # hunt for completion_tokens they never emit.
        if instance is not None and span_data["span_type"] == "llm":
            span_data["is_llm_call"] = True
            span_data["model_name"] = getattr(instance, "model", None) or getattr(instance, "model_name", None)
            if span_data["model_name"] is None:
                # Not every wrapper has a .model attr (MockLLM, some vendors),
                # but every LlamaIndex LLM exposes LLMMetadata.model_name.
                # metadata is a computed property, so guard it.
                try:
                    span_data["model_name"] = instance.metadata.model_name
                except Exception:
                    pass
            span_data["temperature"] = getattr(instance, "temperature", None)
            span_data["provider"] = _detect_provider(instance)

        self._spans[id_] = span_data
        self._parents[id_] = parent_span_id

        # Track tree
        root_id = self._find_root(id_)
        self._trees.setdefault(root_id, []).append(id_)

        return id_

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a span completes successfully.

        If this is the root span of a tree, assembles and sends the
        full RunTrace to the DecimalAI backend.

        Args:
            id_: The span ID.
            bound_args: The bound arguments.
            instance: The object instance.
            result: The return value of the traced function.
        """
        span_data = self._spans.get(id_)
        if not span_data:
            return

        span_data["ended_at"] = datetime.now(timezone.utc)
        span_data["status"] = "success"
        span_data["output_preview"] = _safe_preview(result)

        # Extract LLM response metadata
        if span_data["is_llm_call"] and result is not None:
            self._extract_llm_result(span_data, result)

        # If this is the root span, flush the entire tree
        parent = self._parents.get(id_)
        if parent is None or parent not in self._spans:
            if self._defer_for_stream(id_, span_data, result):
                return
            self._flush_tree(id_)

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Optional[Any] = None,
        err: Optional[Exception] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a span is dropped (error occurred).

        Args:
            id_: The span ID.
            bound_args: The bound arguments.
            instance: The object instance.
            err: The exception that caused the drop.
        """
        span_data = self._spans.get(id_)
        if not span_data:
            return

        # Only a span that is still RUNNING becomes an error. `shutdown()`
        # span-drops everything the handler is holding, which includes the
        # completed spans of a tree that is merely waiting on its stream —
        # re-stamping those would turn a successful run into a failed one.
        if span_data.get("status") == "running":
            span_data["ended_at"] = datetime.now(timezone.utc)
            span_data["status"] = "error"
            span_data["output_preview"] = str(err)[:500] if err else None

        # If root, flush anyway (we want to capture failed traces too)
        parent = self._parents.get(id_)
        if parent is None or parent not in self._spans:
            # Ship it now; a stream still in flight no longer owns the flush.
            self._deferred_roots.discard(id_)
            self._flush_tree(id_)

    # ── Streamed responses ───────────────────────────────────────

    def record_llm_result(self, span_id: Optional[str], response: Any) -> None:
        """Stamp an LLM response onto its span, from the dispatcher's own event.

        LlamaIndex fires ``LLMChatEndEvent`` / ``LLMCompletionEndEvent``
        carrying the ``span_id`` the call belongs to. For a STREAMED call this
        is the only place the finished response appears at all: the traced
        function returned a generator, so ``prepare_to_exit_span`` saw no
        usage, and the event does not fire until the caller has consumed the
        stream. Fills only fields the span is still missing.
        """
        if not span_id or response is None:
            return
        span_data = self._spans.get(span_id)
        if not span_data or not span_data.get("is_llm_call"):
            return
        try:
            self._extract_llm_result(span_data, response)
            if span_data.get("ended_at") is not None:
                # The span already exited — it handed back a generator, so the
                # call really ran until now, not until the generator existed.
                # (Non-streamed calls fire this event BEFORE span_exit, so
                # ended_at is still None there and nothing is overwritten.)
                span_data["ended_at"] = datetime.now(timezone.utc)
        except Exception:  # pragma: no cover - never break the caller's run
            logger.debug("Failed to record LLM result for span %s", span_id, exc_info=True)

    def _defer_for_stream(self, root_id: str, root_data: Dict[str, Any], result: Any) -> bool:
        """Hold a tree open until its streamed response has been delivered.

        ``query_engine.query(..., streaming=True)`` returns while the answer is
        still arriving, so flushing at root exit ships a trace with no output
        and no token counts — the LLM's own end event has not fired yet.

        Swap a PASS-THROUGH wrapper into the response's generator: the app
        receives every token unchanged and in order, while the wrapper
        accumulates the preview and flushes when the stream ends — exhausted,
        errored, or abandoned (see :class:`_StreamTee`), so no trace is held
        hostage by a caller who walks away.

        Returns True when the flush was deferred.
        """
        from . import _config

        if not _config._is_enabled():
            return False  # tracing off — do not touch the app's objects at all

        gen = _live_stream_gen(result)
        if gen is None:
            return False

        def _done(chunks: List[str]) -> None:
            self._finish_stream(root_id, root_data, chunks)

        tee_cls = _AsyncStreamTee if inspect.isasyncgen(gen) else _StreamTee
        try:
            result.response_gen = tee_cls(gen, _done)
        except Exception:  # pragma: no cover - read-only/exotic response object
            logger.debug("Could not wrap streamed response; flushing now", exc_info=True)
            return False
        self._deferred_roots.add(root_id)
        return True

    def _finish_stream(
        self, root_id: str, root_data: Dict[str, Any], chunks: List[str]
    ) -> None:
        """Close out a streamed root and flush its tree (exactly once)."""
        if root_id not in self._deferred_roots:
            return  # already flushed by close(), or already finished
        self._deferred_roots.discard(root_id)
        try:
            # The run really ended when the last token landed, not when the
            # generator was handed over — latency should say so.
            root_data["ended_at"] = datetime.now(timezone.utc)
            if chunks:
                root_data["output_preview"] = _truncate("".join(chunks))
            self._flush_tree(root_id)
        except Exception:  # pragma: no cover - may run during interpreter GC
            logger.debug("Failed to flush streamed tree %s", root_id, exc_info=True)

    # ── Internal helpers ─────────────────────────────────────────

    def _find_root(self, span_id: str) -> str:
        """Walk up the parent chain to find the root span ID."""
        current = span_id
        visited = set()
        while current in self._parents and self._parents[current] is not None:
            if current in visited:
                break
            visited.add(current)
            current = self._parents[current]
        return current

    def _ancestors(self, span_id: str) -> List[str]:
        """Parent chain of ``span_id``, nearest first.

        Bounded by a visited set — a malformed parent link that cycles would
        otherwise spin the caller's thread inside the dispatcher.
        """
        chain: List[str] = []
        current = self._parents.get(span_id)
        seen = {span_id}
        while current is not None and current not in seen:
            seen.add(current)
            chain.append(current)
            current = self._parents.get(current)
        return chain

    def _outermost_ancestor_in(self, span_id: str, ids: set[str]) -> str:
        """Highest ancestor of ``span_id`` that is in ``ids``, else ``span_id``."""
        outermost = span_id
        for ancestor in self._ancestors(span_id):
            if ancestor in ids:
                outermost = ancestor
        return outermost

    def _nearest_ancestor_span(
        self, span_id: str, span_uuids: Dict[str, UUID]
    ) -> Optional[UUID]:
        """TraceSpan id of the nearest ancestor that was emitted as a span."""
        for ancestor in self._ancestors(span_id):
            if ancestor in span_uuids:
                return span_uuids[ancestor]
        return None

    def _extract_llm_result(self, span_data: Dict[str, Any], result: Any) -> None:
        """Extract token counts and model info from an LLM response.

        Every source is tried and each field is filled only if still missing,
        rather than the first matching source winning outright. Two reasons:
        a provider can put the model on ``raw`` and the counts in
        ``additional_kwargs`` (OpenAI does both), and the same span is offered
        a response twice — once from the LLM end event, once from the traced
        function's return value — so whichever arrives second must not
        overwrite real numbers with ``None``.
        """
        if result is None:
            return

        def _fill(field: str, value: Any) -> None:
            if value is not None and span_data.get(field) is None:
                span_data[field] = value

        # ChatResponse / CompletionResponse — the provider's own object under
        # `.raw` (openai.ChatCompletion, anthropic.Message, …).
        raw = getattr(result, "raw", None)
        usage = getattr(raw, "usage", None)
        if usage is not None:
            _fill("input_tokens", getattr(usage, "prompt_tokens", None)
                  or getattr(usage, "input_tokens", None))
            _fill("output_tokens", getattr(usage, "completion_tokens", None)
                  or getattr(usage, "output_tokens", None))
        elif isinstance(raw, dict):
            usage = raw.get("usage") or {}
            _fill("input_tokens", usage.get("prompt_tokens") or usage.get("input_tokens"))
            _fill("output_tokens", usage.get("completion_tokens") or usage.get("output_tokens"))
        _fill("model_name", getattr(raw, "model", None))

        # LlamaIndex normalizes usage onto additional_kwargs for several
        # providers — either flat, or nested under a "usage" key.
        ak = getattr(result, "additional_kwargs", None)
        if isinstance(ak, dict) and ak:
            nested = ak.get("usage")
            source = nested if isinstance(nested, dict) else ak
            _fill("input_tokens", source.get("prompt_tokens") or source.get("input_tokens"))
            _fill("output_tokens", source.get("completion_tokens") or source.get("output_tokens"))

        # Dict-style responses (raw HTTP payloads).
        if isinstance(result, dict):
            usage = result.get("usage") or {}
            _fill("input_tokens", usage.get("prompt_tokens") or usage.get("input_tokens"))
            _fill("output_tokens", usage.get("completion_tokens") or usage.get("output_tokens"))
            _fill("model_name", result.get("model"))

    def _flush_tree(self, root_id: str) -> None:
        """Assemble spans into a RunTrace and send to the backend."""
        from . import _config
        from .schema.common import SpanType, Status
        from .schema.trace import LlmCallRecord, RunTrace, TraceSpan

        if not _config._is_enabled():
            self._cleanup_tree(root_id)
            return

        try:
            client = _config._get_client()
        except Exception:
            logger.debug("SDK not initialized, skipping LlamaIndex trace flush")
            self._cleanup_tree(root_id)
            return

        span_ids = self._trees.get(root_id, [root_id])
        spans_data = [self._spans[sid] for sid in span_ids if sid in self._spans]

        if not spans_data:
            self._cleanup_tree(root_id)
            return

        if _is_setup_tree(spans_data):
            logger.debug(
                "Skipping LlamaIndex setup tree %s (%d spans, no query/retrieval/model)",
                root_id, len(spans_data),
            )
            self._cleanup_tree(root_id)
            return

        # A span the backend can store has to have ended. `dispatcher.shutdown()`
        # drops the ROOT first (open_spans is insertion-ordered), which flushes
        # the tree while its children are still running — every one of them
        # carries ended_at=None and ingest rejects the WHOLE trace with
        # "spans[i]: 'ended_at' is required". Close them out here: they did end,
        # abnormally, at flush time.
        now = datetime.now(timezone.utc)
        for sd in spans_data:
            if sd.get("ended_at") is None:
                sd["ended_at"] = now
                if sd.get("status") == "running":
                    sd["status"] = "error"
                    sd["output_preview"] = sd.get("output_preview") or (
                        "span never completed (process shut down mid-run)"
                    )

        root_data = self._spans.get(root_id, {})

        llm_ids = {sd["id"] for sd in spans_data if sd.get("is_llm_call")}
        # Mint every TraceSpan id up front: a child has to reference its
        # parent's id, and the dispatcher hands spans over in enter order, not
        # parent-resolved order. LLM spans become LlmCallRecords instead, so
        # they get no TraceSpan id and a child of one re-parents to the nearest
        # ancestor that does (never a dangling reference).
        span_uuids: Dict[str, UUID] = {
            sd["id"]: uuid4() for sd in spans_data if sd["id"] not in llm_ids
        }

        # LlamaIndex instruments both the public wrapper (``OpenAI.predict``)
        # and the inner call it delegates to (``OpenAI.chat``), so ONE real
        # request enters the tree as two spans — and only the inner one sees
        # the raw response, hence the token counts. Group each nested LLM span
        # with its outermost LLM ancestor and emit one record per group.
        llm_groups: Dict[str, List[Dict[str, Any]]] = {}
        for sd in spans_data:
            if sd.get("is_llm_call"):
                outer_id = self._outermost_ancestor_in(sd["id"], llm_ids)
                group = llm_groups.setdefault(outer_id, [])
                # Wrapper first — _merge_llm_spans fills each field from the
                # first member that has it, starting at group[0].
                if sd["id"] == outer_id:
                    group.insert(0, sd)
                else:
                    group.append(sd)

        llm_calls: List[LlmCallRecord] = []
        trace_spans: List[TraceSpan] = []
        # Manifest auto-detection: accumulate the first LLM call's model config.
        seen_model: Optional[Dict[str, Any]] = None

        for outer_id, group in llm_groups.items():
            sd = _merge_llm_spans(group)
            if seen_model is None and sd.get("model_name"):
                seen_model = {
                    "provider": sd.get("provider"),
                    "model": sd.get("model_name"),
                    "temperature": sd.get("temperature"),
                }
            latency_ms = None
            if sd.get("started_at") and sd.get("ended_at"):
                latency_ms = int((sd["ended_at"] - sd["started_at"]).total_seconds() * 1000)

            llm_calls.append(LlmCallRecord(
                span_id=self._nearest_ancestor_span(outer_id, span_uuids),
                # Ingest rejects the WHOLE trace over one nameless call
                # ("llm_calls[i]: 'model_name' is required"), and a wrapper that
                # exposes neither `.model` nor `LLMMetadata.model_name` is a
                # real shape. Record the call under a placeholder rather than
                # lose the run; the manifest still sees no model (it reads
                # span_data, which stays None) so nothing declares "unknown".
                model_name=sd.get("model_name") or "unknown",
                provider=sd.get("provider"),
                input_tokens=sd.get("input_tokens"),
                output_tokens=sd.get("output_tokens"),
                temperature=sd.get("temperature"),
                latency_ms=latency_ms,
                status=Status.SUCCESS if sd["status"] == "success" else Status.ERROR,
                started_at=sd.get("started_at"),
                ended_at=sd.get("ended_at"),
            ))

        for sd in spans_data:
            if sd["id"] in llm_ids:
                continue
            span_type = {
                "query": SpanType.AGENT,
                "retrieve": SpanType.RETRIEVAL,
                "synthesize": SpanType.OTHER,
                "embed": SpanType.OTHER,
                "llm": SpanType.LLM,
            }.get(sd.get("span_type", "other"), SpanType.OTHER)

            trace_spans.append(TraceSpan(
                id=span_uuids[sd["id"]],
                parent_span_id=self._nearest_ancestor_span(sd["id"], span_uuids),
                span_type=span_type,
                name=sd.get("name", "unknown"),
                status=Status.SUCCESS if sd["status"] == "success" else Status.ERROR,
                started_at=sd.get("started_at"),
                ended_at=sd.get("ended_at"),
                input_preview=sd.get("input_preview"),
                output_preview=sd.get("output_preview"),
            ))

        # Determine overall trace status
        has_error = any(sd["status"] == "error" for sd in spans_data)

        # Register/version the manifest before building the trace so the trace
        # carries the resulting manifest_id.
        self._maybe_register_manifest(seen_model)

        trace = RunTrace(
            id=uuid4(),
            agent_name=self.agent_name,
            status=Status.ERROR if has_error else Status.SUCCESS,
            source_type="production",
            started_at=root_data.get("started_at"),
            ended_at=root_data.get("ended_at"),
            user_input_preview=root_data.get("input_preview"),
            final_output_preview=root_data.get("output_preview"),
            spans=trace_spans,
            llm_calls=llm_calls,
            manifest_id=self._manifest_id,
        )

        # Send via background sender
        _config._sender.submit(client.ingest_trace, trace)
        logger.debug(
            "LlamaIndex trace flushed: %s (%d spans, %d LLM calls)",
            str(trace.id)[:8], len(trace_spans), len(llm_calls),
        )

        self._cleanup_tree(root_id)

    def _maybe_register_manifest(self, seen_model: Optional[Dict[str, Any]]) -> None:
        """Give this run a manifest_id — without inventing a model change.

        LlamaIndex span handlers surface model/retriever spans (not a single
        agent config object), so the manifest is model-centric — enough for
        manifest diff/compat gating to engage where model drift matters.

        A bare `retrieve()` legitimately uses no model, so there is nothing to
        auto-detect — yet ingest requires a manifest_id, so skipping
        registration lost every such trace to a 400. The trap is the obvious
        repair: registering a model-LESS manifest and later a model-FUL one
        does not read as "we learned the model", it reads as
        ``model_runtime: provider '' → 'openai'``, which the diff engine
        classifies breaking/major and answers with "replay every historical
        trace". Absent and empty are the same thing to that engine — it
        substitutes ``{}`` for a missing surface — so no payload shape can
        express "nothing to declare". The only representation that means it is
        *not re-declaring*:

          1. the model sticks for the life of the handler, so a model-less tree
             between two queries reuses the manifest it already has;
          2. with nothing observed yet, adopt the manifest the agent ALREADY
             has on the backend rather than minting a model-less one — this is
             what stopped every fresh process reverting the agent to its
             model-less v1 and back again;
          3. only a brand-new agent, whose first traced run genuinely observed
             no model, registers the minimal manifest — the honest floor, and
             the backend's own completeness_warnings already names it.

        Thread-safe; only calls the backend when the manifest hash changes.
        """
        from . import _config

        if not _config._is_enabled():
            return

        from .schema.manifest import extract_from_config

        with self._manifest_lock:
            if seen_model and seen_model.get("model"):
                self._seen_model = seen_model

            if self._seen_model is None:
                # Nothing to declare — reuse, never re-declare.
                if self._manifest_confirmed:
                    return
                if self._adopt_existing_manifest():
                    return

            models = {"default": self._seen_model} if self._seen_model else None
            snapshot = extract_from_config(agent_name=self.agent_name, models=models)
            if not self._manifest_tracker.check_and_update(snapshot):
                return  # Same hash — already registered.
            try:
                client = _config._get_client()
                result = client.register_manifest(snapshot)
                self._manifest_id = result.get("manifest_id", snapshot.id)
                self._manifest_confirmed = True
                logger.info(
                    "Registered LlamaIndex manifest %s (hash=%s)",
                    self._manifest_id, snapshot.manifest_hash[:12],
                )
            except Exception as exc:
                logger.warning(
                    "Failed to register LlamaIndex manifest, continuing", exc_info=True
                )
                # Surface the real cause on export_status(), where the trace
                # POST is about to fail with a confusingly different error
                # ("manifest_id is required") against a strict backend.
                try:
                    _config._sender.record_manifest_error(exc)
                except Exception:
                    pass
                # check_and_update already committed this hash; without the
                # reset every later tree short-circuits on it and registration
                # is never retried, so one blip poisons the whole process.
                self._manifest_tracker.reset()
                self._manifest_id = snapshot.id

    def _adopt_existing_manifest(self) -> bool:
        """Ride the agent's current manifest instead of declaring a new one.

        Called only when this process has observed no model at all. Registering
        a model-less manifest here would hash-match the agent's OWN earlier
        model-less version, which the backend treats as a revert: it reactivates
        that version and supersedes the live one — then the first query in the
        same process flips it straight back. Every fresh process did that
        round trip, and each flip counts as a detected change.

        One attempt per handler, best-effort: on any failure we fall through to
        registering, which is still better than losing the trace.
        """
        if self._adoption_attempted:
            return False
        self._adoption_attempted = True

        from . import _config

        try:
            client = _config._get_client()
            listing = client.list_manifests(agent_name=self.agent_name, limit=20)
            active = next(
                (m for m in (listing.get("manifests") or []) if m.get("status") == "active"),
                None,
            )
            manifest_id = (active or {}).get("id")
        except Exception:
            logger.debug(
                "Could not look up an existing manifest for %s; registering one",
                self.agent_name, exc_info=True,
            )
            return False

        if not manifest_id:
            return False

        self._manifest_id = manifest_id
        self._manifest_confirmed = True
        logger.info(
            "Reusing existing manifest %s for %s (this run declared no model)",
            manifest_id, self.agent_name,
        )
        return True

    def _cleanup_tree(self, root_id: str) -> None:
        """Remove all spans belonging to a tree."""
        span_ids = self._trees.pop(root_id, [])
        for sid in span_ids:
            self._spans.pop(sid, None)
            self._parents.pop(sid, None)


# ── Helper functions ─────────────────────────────────────────────


def _merge_llm_spans(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse one instrumented LLM call — wrapper plus inner call — into one.

    ``group[0]`` is the outermost span; the rest are the inner calls it
    delegates to, which is where the token counts land. Fill each field from
    the first member that has it, so the merged record is the union of both
    halves.

    The wall clock spans the whole group rather than just the outermost span:
    a STREAMED call's inner span is stamped with the moment the last token
    arrived (from the LLM end event), which is well after the wrapper returned
    its generator, so taking the wrapper's clock alone reported a 3ms call for
    a multi-second stream.
    """
    merged = dict(group[0])
    for field in ("model_name", "provider", "input_tokens", "output_tokens", "temperature"):
        if merged.get(field) is None:
            for sd in group[1:]:
                if sd.get(field) is not None:
                    merged[field] = sd[field]
                    break
    starts = [sd["started_at"] for sd in group if sd.get("started_at")]
    ends = [sd["ended_at"] for sd in group if sd.get("ended_at")]
    if starts:
        merged["started_at"] = min(starts)
    if ends:
        merged["ended_at"] = max(ends)
    if any(sd.get("status") == "error" for sd in group):
        merged["status"] = "error"
    return merged


# LlamaIndex's own base classes are the ground truth for "what kind of thing
# is this span". A class NAME cannot answer it: "ollama", "cohere", "mistral",
# "groq", "bedrock" and "vertex" contain none of the substrings the old
# heuristic looked for, so every one of those LLM calls was filed as a generic
# span, never reached llm_calls, and left the agent's manifest permanently
# model-less.
#
# (module, attribute, span type). Order matters: embeddings are resolved before
# LLMs so an embedding model can never land in llm_calls, and retriever /
# synthesizer come before query engine because a query engine composes them.
# Modules a given llama-index-core doesn't ship are skipped, not raised.
_SPAN_BASE_SPECS = (
    ("llama_index.core.base.embeddings.base", "BaseEmbedding", "embed"),
    ("llama_index.core.base.llms.base", "BaseLLM", "llm"),
    ("llama_index.core.base.base_retriever", "BaseRetriever", "retrieve"),
    ("llama_index.core.response_synthesizers.base", "BaseSynthesizer", "synthesize"),
    ("llama_index.core.base.base_query_engine", "BaseQueryEngine", "query"),
    ("llama_index.core.chat_engine.types", "BaseChatEngine", "query"),
    ("llama_index.core.agent.workflow.base_agent", "BaseWorkflowAgent", "query"),
)

_span_bases: Optional[List[Tuple[type, str]]] = None


def _span_base_classes() -> List[Tuple[type, str]]:
    """(base class, span type) pairs, resolved from llama-index-core once."""
    global _span_bases
    if _span_bases is None:
        import importlib

        resolved: List[Tuple[type, str]] = []
        for module_path, attr, span_type in _SPAN_BASE_SPECS:
            try:
                base = getattr(importlib.import_module(module_path), attr)
            except Exception:
                continue
            if isinstance(base, type):
                resolved.append((base, span_type))
        _span_bases = resolved
    return _span_bases


def _classify_span(instance: Optional[Any]) -> str:
    """Classify a span type from the LlamaIndex instance.

    Structural (base-class ancestry) first, name heuristic only as a fallback
    for objects that are not LlamaIndex types at all — duck-typed wrappers,
    test doubles, and integrations that predate the base classes.
    """
    if instance is None:
        return "other"

    for base, span_type in _span_base_classes():
        try:
            if isinstance(instance, base):
                return span_type
        except TypeError:  # pragma: no cover - exotic metaclass
            continue

    return _classify_span_by_name(instance)


def _classify_span_by_name(instance: Any) -> str:
    """Last-resort classification from the class name."""
    cls_name = type(instance).__name__.lower()

    if "query" in cls_name or "queryengine" in cls_name:
        return "query"
    if "retriev" in cls_name:
        return "retrieve"
    if "synth" in cls_name or "responsesynthesizer" in cls_name:
        return "synthesize"
    if "embed" in cls_name:
        return "embed"
    # Embeddings are matched above, so the broad provider substrings here can't
    # swallow OpenAIEmbedding / GoogleGenAIEmbedding. The current Google LLM class
    # is `GoogleGenAI` (module llama_index.llms.google_genai) — it contains neither
    # "llm" nor "gemini", so without "google"/"genai" its synthesis calls would
    # classify as "other" and never land in llm_calls.
    if ("llm" in cls_name or "openai" in cls_name or "anthropic" in cls_name
            or "gemini" in cls_name or "google" in cls_name or "genai" in cls_name):
        return "llm"
    if "agent" in cls_name:
        return "query"

    return "other"


def _is_setup_tree(spans_data: List[Dict[str, Any]]) -> bool:
    """True for a tree that is index-time plumbing rather than an agent run.

    ``VectorStoreIndex.from_documents`` drives the node parser and the
    embedding model directly, so every index build lands TWO parentless trees —
    ``SentenceSplitter.__call__`` and ``<Embedding>.get_text_embedding_batch``.
    Shipping those as ``source_type="production"`` inflated the agent's run
    count and, worse, handed them to the compatibility engine as replay
    episodes (a real report scored "keep 2" entirely off index-build traces).

    Deliberately conservative: a run is a tree that queries, retrieves,
    synthesizes or calls a model, and ONE such span anywhere in the tree keeps
    it — so `retriever.retrieve()` and `llm.complete()`, which are runs with no
    query engine above them, still ship.
    """
    return not any(
        sd.get("is_llm_call") or sd.get("span_type") in _RUN_SPAN_TYPES
        for sd in spans_data
    )


# The dispatcher builds every span id as f"{Class}.{method}-{uuid4()}"
# (llama_index_instrumentation/dispatcher.py), so the id carries the METHOD the
# class name alone leaves out — without it a 12-span RAG tree reads as
# "RetrieverQueryEngine, RetrieverQueryEngine, VectorIndexRetriever,
# VectorIndexRetriever, …".
_SPAN_ID_UUID_SUFFIX = re.compile(
    r"-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _get_span_name(
    instance: Optional[Any], bound_args: Any, span_id: Optional[str] = None
) -> str:
    """Get a human-readable name for a span."""
    if span_id:
        name, replaced = _SPAN_ID_UUID_SUFFIX.subn("", span_id)
        if replaced and name:
            return name

    if instance is not None:
        return type(instance).__name__

    if bound_args is not None and hasattr(bound_args, "__name__"):
        return bound_args.__name__

    return "LlamaIndexOperation"


# Integration packages are named for their provider — llama_index.llms.ollama,
# llama_index.llms.bedrock_converse, … — so the module path names any provider
# the list below has never heard of. Checked AFTER the explicit list, never
# before: the list's answers are already in registered manifests, and swapping
# "google" for "google_genai" under a live agent would itself diff as a
# breaking provider change.
_PROVIDER_MODULE_ROOTS = (
    "llama_index.llms.",
    "llama_index.multi_modal_llms.",
    "llama_index.embeddings.",
)


def _detect_provider(instance: Any) -> Optional[str]:
    """Detect the LLM provider from a LlamaIndex LLM instance."""
    cls_name = type(instance).__name__.lower()
    module_name = type(instance).__module__ or ""

    if "openai" in cls_name or "openai" in module_name:
        return "openai"
    if "anthropic" in cls_name or "anthropic" in module_name:
        return "anthropic"
    if "gemini" in cls_name or "google" in cls_name or "genai" in cls_name or "google" in module_name:
        return "google"
    if "cohere" in cls_name or "cohere" in module_name:
        return "cohere"
    if "mistral" in cls_name or "mistral" in module_name:
        return "mistral"
    if "huggingface" in cls_name or "hf" in cls_name:
        return "huggingface"
    if "ollama" in cls_name or "ollama" in module_name:
        return "ollama"

    for root in _PROVIDER_MODULE_ROOTS:
        if module_name.startswith(root):
            provider = module_name[len(root):].split(".", 1)[0]
            if provider:
                return provider

    return None


_PREVIEW_MAX = 300


def _truncate(text: str, max_len: int = _PREVIEW_MAX) -> Optional[str]:
    if len(text) > max_len:
        text = text[:max_len] + "…"
    return text if text else None


def _is_exhaustible(obj: Any) -> bool:
    """True for one-shot iterators — reading them once destroys them."""
    if isinstance(obj, (str, bytes, bytearray)):
        return False
    if inspect.isgenerator(obj) or inspect.isasyncgen(obj):
        return True
    return hasattr(obj, "__next__") or hasattr(obj, "__anext__")


def _is_streaming_response(obj: Any) -> bool:
    """True for LlamaIndex's streaming response wrappers.

    Uses ``dir()`` rather than ``getattr`` on purpose: reading some of these
    attributes is itself the destructive act.
    """
    try:
        names = dir(obj)
    except Exception:  # pragma: no cover - hostile __dir__
        return False
    return "response_gen" in names or "chat_stream" in names


def _live_stream_gen(obj: Any) -> Optional[Any]:
    """The still-unconsumed generator on a streaming response, if any.

    Only reads ``__dict__`` — that both avoids triggering a property getter and
    guarantees the attribute is one we can swap a pass-through wrapper into
    (``StreamingAgentChatResponse.response_gen`` is a read-only property, so it
    is correctly skipped).
    """
    state = getattr(obj, "__dict__", None)
    if not isinstance(state, dict):
        return None
    if state.get("response_txt"):
        return None  # already materialized by its owner
    gen = state.get("response_gen")
    if gen is None or not _is_exhaustible(gen):
        return None
    return gen


def _collect_stream_chunk(chunks: List[str], chunk: Any) -> None:
    """Accumulate one streamed chunk's text, bounded to the preview length."""
    if sum(len(c) for c in chunks) > _PREVIEW_MAX:
        return
    try:
        if isinstance(chunk, str):
            text = chunk
        else:
            # ChatResponse/CompletionResponse chunks carry the incremental
            # token on `.delta`; `.text`/`.message` hold the running total.
            text = getattr(chunk, "delta", None)
            if text is None:
                return
        chunks.append(str(text))
    except Exception:  # pragma: no cover
        pass


class _StreamTee:
    """Pass-through wrapper around a streamed response's generator.

    Yields every chunk unchanged and in order — the app must not be able to
    tell it is being traced — while accumulating the text preview and calling
    ``on_done`` exactly once when the stream ends: exhausted, errored, or
    abandoned.

    Deliberately an ITERATOR CLASS rather than a generator function. A
    generator that is never started does not run its ``finally`` when closed,
    so a stream the caller never touched would silently never flush its trace;
    ``__del__`` fires either way.
    """

    __slots__ = ("_gen", "_on_done", "_chunks", "_done")

    def __init__(self, gen: Any, on_done: Any) -> None:
        self._gen = gen
        self._on_done = on_done
        self._chunks: List[str] = []
        self._done = False

    def __iter__(self) -> "_StreamTee":
        return self

    def __next__(self) -> Any:
        try:
            chunk = next(self._gen)
        except BaseException:
            self._finish()
            raise
        _collect_stream_chunk(self._chunks, chunk)
        return chunk

    def close(self) -> None:
        closer = getattr(self._gen, "close", None)
        if closer is not None:
            closer()
        self._finish()

    def _finish(self) -> None:
        if self._done:
            return
        self._done = True
        try:
            self._on_done(self._chunks)
        except Exception:  # pragma: no cover - may run during interpreter GC
            pass

    def __del__(self) -> None:
        try:
            self._finish()
        except Exception:  # pragma: no cover - interpreter teardown
            pass


class _AsyncStreamTee(_StreamTee):
    """Async twin of :class:`_StreamTee`."""

    __slots__ = ()

    def __aiter__(self) -> "_AsyncStreamTee":
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._gen.__anext__()
        except BaseException:
            self._finish()
            raise
        _collect_stream_chunk(self._chunks, chunk)
        return chunk

    def close(self) -> None:
        # aclose() is a coroutine — calling the sync close() here would leave
        # it un-awaited. Just finish the trace.
        self._finish()

    async def aclose(self) -> None:
        closer = getattr(self._gen, "aclose", None)
        if closer is not None:
            await closer()
        self._finish()


def _safe_preview(obj: Any, max_len: int = _PREVIEW_MAX, *, _depth: int = 0) -> Optional[str]:
    """Convert an object to a string preview, truncating if needed.

    A tracer must never consume what it is only observing. ``str()`` on a
    ``StreamingResponse`` runs ``__str__``, which drains ``response_gen`` into
    ``response_txt`` — the app then receives an EMPTY stream. Streams and
    one-shot iterators are therefore recognised BEFORE any stringification,
    and only text the owner has already materialized is previewed.
    """
    if obj is None:
        return None

    try:
        # Never touch a stream. (StreamingResponse / AsyncStreamingResponse /
        # StreamingAgentChatResponse, and bare ChatResponseGen generators.)
        if _is_exhaustible(obj):
            return None
        if _is_streaming_response(obj):
            state = getattr(obj, "__dict__", None) or {}
            for attr in ("response_txt", "response"):
                value = state.get(attr)
                if value:
                    return _truncate(str(value), max_len)
            return None

        # Handle LlamaIndex QueryBundle
        if hasattr(obj, "query_str"):
            text = str(obj.query_str)
        # Handle LlamaIndex Response
        elif hasattr(obj, "response"):
            text = str(obj.response)
        # Handle ChatMessage
        elif hasattr(obj, "content"):
            text = str(obj.content)
        # Handle BoundArguments — preview the first argument. Recurses rather
        # than str()-ing it: a synthesizer is called WITH a streaming response.
        elif hasattr(obj, "arguments"):
            args = obj.arguments
            if not args or _depth:
                return None
            return _safe_preview(next(iter(args.values())), max_len, _depth=_depth + 1)
        else:
            text = str(obj)

        return _truncate(text, max_len)
    except Exception:
        return None


# ── Public API ───────────────────────────────────────────────────


def instrument(agent_name: Optional[str] = None) -> "DecimalSpanHandler":
    """Install DecimalAI as a span handler for LlamaIndex.

    Registers a ``DecimalSpanHandler`` with LlamaIndex's root
    instrumentation dispatcher. All subsequent LlamaIndex operations
    (queries, retrieval, LLM calls, synthesis) will be automatically
    captured and sent to the DecimalAI backend.

    Requires ``llama-index-core>=0.12.0`` to be installed. (Earlier
    releases either lack the instrumentation dispatcher entirely or call
    span handlers with an incompatible pre-0.10.30 signature.)

    Args:
        agent_name: Name for the agent in DecimalAI. Defaults to
            ``"llamaindex-agent"``.

    Returns:
        The ``DecimalSpanHandler`` instance.

    Raises:
        ImportError: If ``llama-index-core`` is not installed.

    Example::

        import decimalai
        decimalai.init(api_key="...")

        from decimalai.llamaindex import instrument
        instrument(agent_name="my-rag-agent")

        # All LlamaIndex operations are now traced
        from llama_index.core import VectorStoreIndex
        index = VectorStoreIndex.from_documents(docs)
        response = index.as_query_engine().query("What is X?")
    """
    try:
        from llama_index.core.instrumentation import get_dispatcher
    except ImportError:
        raise ImportError(
            "This integration requires llama-index-core>=0.12.0 — either "
            "LlamaIndex is not installed, or the installed llama-index-core "
            "predates its instrumentation dispatcher. Install or upgrade "
            "with: pip install -U \"decimalai[llamaindex]\" "
            "\"llama-index-core>=0.12.0\""
        )

    handler = DecimalSpanHandler(agent_name=agent_name)
    dispatcher = get_dispatcher()
    dispatcher.add_span_handler(handler)
    _install_llm_event_handler(dispatcher, handler)

    logger.info(
        "DecimalAI LlamaIndex span handler installed (agent_name=%s)",
        handler.agent_name,
    )
    return handler


def _install_llm_event_handler(dispatcher: Any, handler: DecimalSpanHandler) -> None:
    """Also listen for LlamaIndex's LLM *end* events.

    Token usage is not on the traced function's return value for a STREAMED
    call: ``llm.stream_chat`` returns a generator, and LlamaIndex fires
    ``LLMChatEndEvent`` with the finished response only once the caller has
    consumed the stream — after the span exited with nothing to read. The event
    carries the ``span_id`` it belongs to, so the usage can still be stamped on
    the right span.

    Best-effort: a llama-index-core without these events keeps the
    return-value extraction and simply reports no tokens for streamed calls.
    """
    try:
        from llama_index.core.instrumentation.event_handlers import BaseEventHandler
        from llama_index.core.instrumentation.events.llm import (
            LLMChatEndEvent,
            LLMCompletionEndEvent,
        )
    except Exception:
        logger.debug(
            "LlamaIndex LLM end events unavailable; streamed calls will report "
            "no token counts", exc_info=True,
        )
        return

    end_events = (LLMChatEndEvent, LLMCompletionEndEvent)

    class DecimalLLMEventHandler(BaseEventHandler):  # type: ignore[misc, valid-type]
        @classmethod
        def class_name(cls) -> str:
            return "DecimalLLMEventHandler"

        def handle(self, event: Any, **kwargs: Any) -> None:
            if isinstance(event, end_events):
                handler.record_llm_result(
                    getattr(event, "span_id", None), getattr(event, "response", None)
                )

    try:
        dispatcher.add_event_handler(DecimalLLMEventHandler())
    except Exception:  # pragma: no cover - handler list is model-validated
        logger.debug("Could not register the DecimalAI LLM event handler", exc_info=True)


# ── Deprecated: install() ────────────────────────────────────────────────────
#
# Renamed to `instrument()` 2026-08-11. "install" was doing double duty across
# this SDK: here it turned on TRACING for a framework, while
# `SkillRouter.install()` added a SKILL to a workspace. Two unrelated actions
# under one word, in one package — and the skill sense is the one users arrive
# with, because it is what every extension marketplace means by install.
#
# Behaviour is unchanged and this alias is not going away soon; it warns so the
# docs and the code agree on one name.
def install(*args, **kwargs):  # pragma: no cover - thin deprecation shim
    warnings.warn(
        "decimalai.llamaindex.install() is deprecated; use "
        "decimalai.llamaindex.instrument() instead. It turns on tracing for llamaindex "
        "and has never had anything to do with installing a skill.",
        DeprecationWarning,
        stacklevel=2,
    )
    return instrument(*args, **kwargs)
