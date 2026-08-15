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

import logging
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

logger = logging.getLogger("decimalai.llamaindex")


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
        """Dispatcher shutdown hook — nothing to release; trees flush as
        their roots exit (or are dropped by shutdown itself)."""

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
            "name": _get_span_name(instance, bound_args),
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

        span_data["ended_at"] = datetime.now(timezone.utc)
        span_data["status"] = "error"
        span_data["output_preview"] = str(err)[:500] if err else None

        # If root, flush anyway (we want to capture failed traces too)
        parent = self._parents.get(id_)
        if parent is None or parent not in self._spans:
            self._flush_tree(id_)

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
        """Extract token counts and model info from an LLM response."""
        # Handle ChatResponse / CompletionResponse
        if hasattr(result, "raw"):
            raw = result.raw
            if hasattr(raw, "usage"):
                usage = raw.usage
                span_data["input_tokens"] = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
                span_data["output_tokens"] = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
            if hasattr(raw, "model"):
                span_data["model_name"] = span_data["model_name"] or raw.model

        # Handle dict-style responses
        elif isinstance(result, dict):
            usage = result.get("usage", {})
            span_data["input_tokens"] = usage.get("prompt_tokens") or usage.get("input_tokens")
            span_data["output_tokens"] = usage.get("completion_tokens") or usage.get("output_tokens")

        # Handle LlamaIndex ChatResponse directly
        elif hasattr(result, "additional_kwargs"):
            ak = result.additional_kwargs or {}
            if "usage" in ak:
                span_data["input_tokens"] = ak["usage"].get("prompt_tokens")
                span_data["output_tokens"] = ak["usage"].get("completion_tokens")

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
                # Wrapper first — _merge_llm_spans reads the call's wall clock
                # off group[0].
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
                model_name=sd.get("model_name"),
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
        """Register a manifest for this run, even when it used no model.

        LlamaIndex span handlers surface model/retriever spans (not a single
        agent config object), so the manifest is model-centric — enough for
        manifest diff/compat gating to engage where model drift matters.

        Index construction and bare retrieval legitimately have no model, so
        there is nothing to auto-detect — but ingest requires a manifest_id, so
        skipping registration there meant every such tree 400'd and the trace
        was lost. Register the model-less manifest instead (agent name only);
        the model, once seen, sticks so later model-less trees reuse the same
        hash rather than flip-flopping the manifest version.

        Thread-safe; only calls the backend when the manifest hash changes.
        """
        from . import _config

        if not _config._is_enabled():
            return

        from .schema.manifest import extract_from_config

        with self._manifest_lock:
            if seen_model and seen_model.get("model"):
                self._seen_model = seen_model
            models = {"default": self._seen_model} if self._seen_model else None
            snapshot = extract_from_config(agent_name=self.agent_name, models=models)
            if not self._manifest_tracker.check_and_update(snapshot):
                return  # Same hash — already registered.
            try:
                client = _config._get_client()
                result = client.register_manifest(snapshot)
                self._manifest_id = result.get("manifest_id", snapshot.id)
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

    def _cleanup_tree(self, root_id: str) -> None:
        """Remove all spans belonging to a tree."""
        span_ids = self._trees.pop(root_id, [])
        for sid in span_ids:
            self._spans.pop(sid, None)
            self._parents.pop(sid, None)


# ── Helper functions ─────────────────────────────────────────────


def _merge_llm_spans(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse one instrumented LLM call — wrapper plus inner call — into one.

    ``group[0]`` is the outermost span (the widest wall clock, so its
    started_at/ended_at win); the rest are the inner calls it delegates to,
    which is where the token counts land. Fill each field from the first member
    that has it, so the merged record is the union of both halves.
    """
    merged = dict(group[0])
    for field in ("model_name", "provider", "input_tokens", "output_tokens", "temperature"):
        if merged.get(field) is None:
            for sd in group[1:]:
                if sd.get(field) is not None:
                    merged[field] = sd[field]
                    break
    if any(sd.get("status") == "error" for sd in group):
        merged["status"] = "error"
    return merged


def _classify_span(instance: Optional[Any]) -> str:
    """Classify a span type from the LlamaIndex instance."""
    if instance is None:
        return "other"

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


def _get_span_name(instance: Optional[Any], bound_args: Any) -> str:
    """Get a human-readable name for a span."""
    if instance is not None:
        return type(instance).__name__

    if bound_args is not None and hasattr(bound_args, "__name__"):
        return bound_args.__name__

    return "LlamaIndexOperation"


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
    if "ollama" in cls_name:
        return "ollama"

    return None


def _safe_preview(obj: Any, max_len: int = 300) -> Optional[str]:
    """Convert an object to a string preview, truncating if needed."""
    if obj is None:
        return None

    try:
        # Handle LlamaIndex QueryBundle
        if hasattr(obj, "query_str"):
            text = str(obj.query_str)
        # Handle LlamaIndex Response
        elif hasattr(obj, "response"):
            text = str(obj.response)
        # Handle ChatMessage
        elif hasattr(obj, "content"):
            text = str(obj.content)
        # Handle BoundArguments — extract the first argument
        elif hasattr(obj, "arguments"):
            args = obj.arguments
            if args:
                first_val = next(iter(args.values()))
                text = str(first_val)
            else:
                return None
        else:
            text = str(obj)

        if len(text) > max_len:
            text = text[:max_len] + "…"
        return text if text else None
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

    logger.info(
        "DecimalAI LlamaIndex span handler installed (agent_name=%s)",
        handler.agent_name,
    )
    return handler


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
