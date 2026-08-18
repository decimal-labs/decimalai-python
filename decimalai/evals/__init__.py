"""DecimalAI Evaluation Framework.

Provides the ``@eval`` decorator for defining custom evaluation functions
that run client-side before trace upload. Scores are attached to the trace
and sent to the platform alongside the trace data.

Usage::

    from decimalai.evals import eval, TraceData, EvalResult

    @eval(name="has_citation")
    def check_citation(trace: TraceData) -> bool:
        return "[source:" in trace.output

    # Register with install()
    from decimalai.langchain import install
    install(agent_name="my-bot", evals=[check_citation])
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import random
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
    get_type_hints,
)

logger = logging.getLogger("decimalai.evals")


def _capture_source_location(fn: Callable) -> Optional[str]:
    """Return "path:lineno" for fn, or None if unavailable.

    Path is made relative to cwd when possible so paths in the UI link
    cleanly to repo-relative source.

    If the process changes cwd between decoration and trace flush
    (test runners, background workers, async code in different working
    dirs), the relative path captured here becomes stale relative to
    the cwd at upload time and any deep-link from UI → source breaks.
    That is accepted: the absolute path and the cwd-at-decoration that
    used to be captured for recovery describe the developer's machine and
    are no longer collected.
    """
    try:
        import os
        src_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
        if not src_file:
            return None
        _, src_line = inspect.getsourcelines(fn)
        try:
            rel = os.path.relpath(src_file, os.getcwd())
        except ValueError:
            rel = src_file
        return f"{rel}:{src_line}"
    except (OSError, TypeError):
        return None


def _capture_source_location_extra(fn: Callable) -> Optional[Dict[str, Any]]:
    """Return {'lineno': int} for fn, or None if unavailable.

    Stored alongside the legacy `source_location` string. This used to also
    carry ``abs_path`` (the absolute path of the file on the developer's
    machine) and ``cwd_at_decoration`` (the process working directory), so
    the UI could recover a deep-link when cwd shifted between decoration and
    upload. Both were filesystem layout from the user's machine and are no
    longer sent; only the non-identifying line number remains.
    """
    try:
        src_file = inspect.getsourcefile(fn) or inspect.getfile(fn)
        if not src_file:
            return None
        _, src_line = inspect.getsourcelines(fn)
        return {"lineno": src_line}
    except (OSError, TypeError):
        return None


# ── Data Types ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolCallView:
    """Read-only view of a single tool call."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    result: Optional[str] = None


@dataclass(frozen=True)
class LlmCallView:
    """Read-only view of a single LLM call."""

    model: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tool_calls: List[ToolCallView] = field(default_factory=list)


@dataclass(frozen=True)
class TraceData:
    """Read-only view of a trace, passed to eval functions.

    This is a lightweight, framework-agnostic representation of a trace.
    Eval functions receive this instead of the raw RunTrace model.
    """

    id: str
    input: Union[str, Dict[str, Any]]
    output: Union[str, Dict[str, Any]]
    status: str  # "success" | "error"
    tool_calls: List[ToolCallView] = field(default_factory=list)
    llm_calls: List[LlmCallView] = field(default_factory=list)
    latency_ms: Optional[int] = None
    total_tokens: Optional[int] = None
    agent_name: str = ""
    manifest_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Structured raw data — when available, preserves full dicts
    raw_input: Optional[Any] = None
    raw_output: Optional[Any] = None
    # Active skills on this trace (populated from span attributes)
    active_skills: List[str] = field(default_factory=list)
    # Context for RAG evaluators (retrieved documents, source text, etc.)
    context: Optional[str] = None


@dataclass
class EvalResult:
    """Result of a single eval check.

    All return types from ``@eval`` functions are coerced into this.
    """

    score: float  # 0.0 to 1.0
    passed: bool
    reason: str = ""
    metadata: Optional[Dict[str, Any]] = None


# ── Scorer type — pure scoring callable ─────────────────────────
#
# A `Scorer` is just the *scoring function shape* a `DecimalEval` wraps —
# the pure callable that takes a `TraceData` and returns a score. The
# `DecimalEval` class (below) layers run-config (name, sampling_rate,
# version, source_location) ON TOP of a scorer. Splitting these two
# concerns lets callers compose scorers across multiple Evals without
# carrying decorator metadata around. It mirrors the `Scorer` / `Eval`
# split other eval libraries (e.g. Braintrust) settled on, so the mental
# model ports.
#
# A `@eval`-decorated function IS a valid Scorer (the `DecimalEval` is
# callable as `(trace) → score|result`), so callers can mix raw scorers
# and `DecimalEval` wrappers wherever a Scorer is accepted.
#
# Usage::
#
#     from decimalai.evals import Scorer
#
#     def exact_match(trace: TraceData) -> bool:
#         return trace.output == trace.metadata.get("expected", "")
#
#     # exact_match satisfies the Scorer protocol — no @eval needed.
Scorer = Callable[
    ["TraceData"],
    Optional[Union[bool, float, "EvalResult", Dict[str, "EvalResult"]]],
]
"""Type alias for the pure scoring-callable shape that backs every eval.

A Scorer takes a TraceData and returns a score, expressed as one of:
- `bool` — pass/fail (coerced to 1.0/0.0).
- `float` — score in [0,1] (clipped).
- `EvalResult` — structured score with reason + metadata.
- `dict[str, EvalResult]` — multi-named score group.
- `None` — skip (e.g., sampling miss).

`DecimalEval` instances satisfy this protocol via their `__call__`."""


# ── The @eval Decorator ──────────────────────────────────────────


class DecimalEval:
    """Wrapper around a user-defined eval function.

    Created by the ``@eval`` decorator. Validates the function signature
    and return type at decoration time, and handles return-type coercion
    and sampling at eval time.
    """

    def __init__(
        self,
        fn: Callable,
        name: Optional[str] = None,
        category: str = "quality",
        sampling_rate: float = 1.0,
        builtin: bool = False,
        version: str = "1",
    ):
        # Validate sampling_rate
        if not 0.0 <= sampling_rate <= 1.0:
            raise ValueError(
                f"sampling_rate must be between 0.0 and 1.0, got {sampling_rate}"
            )

        self.fn = fn
        self.name = name or fn.__name__
        self.category = category
        self.sampling_rate = sampling_rate
        self.builtin = builtin
        self.version = version
        self.is_async = asyncio.iscoroutinefunction(fn)
        self.source_location = _capture_source_location(fn)
        # `source_location_extra` carries only the line number; the abs path
        # and decoration-time cwd it used to hold identified the developer's
        # machine and are no longer collected.
        self.source_location_extra = _capture_source_location_extra(fn)
        self.description = (inspect.getdoc(fn) or "").strip() or None

        # Validate name
        if not self.name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                f"Eval name must be a valid identifier, got '{self.name}'"
            )

        # Validate signature — must accept exactly 1 positional parameter
        sig = inspect.signature(fn)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            and p.default is p.empty
        ]
        if len(params) != 1:
            raise TypeError(
                f"@eval function '{fn.__name__}' must accept exactly "
                f"1 positional argument (TraceData), got {len(params)}"
            )

        # Warn if type hint is missing or wrong
        try:
            hints = get_type_hints(fn)
            param_name = params[0].name
            if param_name in hints and hints[param_name] is not TraceData:
                logger.warning(
                    "Eval '%s': parameter '%s' should be typed as TraceData, "
                    "got %s",
                    self.name,
                    param_name,
                    hints[param_name],
                )
        except Exception:
            pass  # Don't fail on type hint resolution issues

    def __call__(self, trace: TraceData) -> Optional[Union[EvalResult, Dict[str, EvalResult]]]:
        """Run the eval function and coerce the return value."""
        # Check sampling
        if self.sampling_rate < 1.0 and random.random() > self.sampling_rate:
            return None  # Skipped by sampling

        try:
            if self.is_async:
                # Run async function — handle both inside and outside event loops
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    # Inside an existing event loop — create a future and run
                    # in a new thread to avoid blocking the main loop
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(asyncio.run, self.fn(trace))
                        result = future.result(timeout=30)
                else:
                    result = asyncio.run(self.fn(trace))
            else:
                result = self.fn(trace)
        except Exception as e:
            logger.warning("Eval '%s' raised %s: %s", self.name, type(e).__name__, e)
            return None

        return self._coerce_result(result)

    async def async_batch(
        self,
        traces: Sequence[TraceData],
        concurrency: int = 10,
    ) -> List[Optional[Union[EvalResult, Dict[str, EvalResult]]]]:
        """Run this eval over many traces concurrently, returning a same-length list.

        Closes the LLM-judge slow-path: a 30s-per-call judge over 1000 traces
        takes 8+ hours serially via `__call__` in a loop; `async_batch(traces,
        concurrency=10)` cuts that to ~50 min (10× parallelism). Output index
        matches input index — sampling-skips appear as None at their position.

        - Async eval fns are awaited directly under an `asyncio.Semaphore`.
        - Sync eval fns are dispatched via `asyncio.to_thread` so they don't
          block the event loop while parallel judges wait on the network.
        - Exceptions in a single eval are caught and yield None at that index
          (matching `__call__`'s swallow-and-log behavior).
        """
        if not traces:
            return []
        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(trace: TraceData):
            # Sampling: same gate as __call__.
            if self.sampling_rate < 1.0 and random.random() > self.sampling_rate:
                return None
            async with sem:
                try:
                    if self.is_async:
                        result = await self.fn(trace)
                    else:
                        result = await asyncio.to_thread(self.fn, trace)
                except Exception as e:
                    logger.warning(
                        "Eval '%s' raised %s: %s",
                        self.name,
                        type(e).__name__,
                        e,
                    )
                    return None
            return self._coerce_result(result)

        return await asyncio.gather(*(_one(t) for t in traces))

    def _coerce_result(
        self, result: Any
    ) -> Optional[Union[EvalResult, Dict[str, EvalResult]]]:
        """Coerce the raw return value into EvalResult(s)."""
        if result is None:
            return None

        if isinstance(result, EvalResult):
            result.score = max(0.0, min(1.0, result.score))
            return result

        if isinstance(result, bool):
            return EvalResult(
                score=1.0 if result else 0.0,
                passed=result,
            )

        if isinstance(result, (int, float)):
            score = max(0.0, min(1.0, float(result)))
            return EvalResult(score=score, passed=score >= 0.5)

        if isinstance(result, dict):
            # Dict return → multiple named scores
            multi: Dict[str, EvalResult] = {}
            for key, val in result.items():
                if isinstance(val, bool):
                    multi[key] = EvalResult(
                        score=1.0 if val else 0.0, passed=val
                    )
                elif isinstance(val, (int, float)):
                    score = max(0.0, min(1.0, float(val)))
                    multi[key] = EvalResult(score=score, passed=score >= 0.5)
                elif isinstance(val, EvalResult):
                    val.score = max(0.0, min(1.0, val.score))
                    multi[key] = val
                else:
                    logger.warning(
                        "Eval '%s': dict value for key '%s' has unsupported "
                        "type %s, skipping",
                        self.name,
                        key,
                        type(val).__name__,
                    )
            return multi if multi else None

        logger.warning(
            "Eval '%s' returned unsupported type %s, expected "
            "bool/float/dict/EvalResult/None",
            self.name,
            type(result).__name__,
        )
        return None

    def to_score_dicts(
        self, trace: TraceData
    ) -> List[Dict[str, Any]]:
        """Run the eval and return a list of score dicts for the API."""
        coerced = self(trace)
        if coerced is None:
            return []

        scores: List[Dict[str, Any]] = []

        if isinstance(coerced, EvalResult):
            scores.append({
                "name": self.name,
                "score": coerced.score,
                "passed": coerced.passed,
                "reason": coerced.reason or "",
                "category": self.category,
                "source": "builtin" if self.builtin else "custom",
                "eval_version": self.version,
                "metadata": coerced.metadata,
            })
        elif isinstance(coerced, dict):
            for key, eval_result in coerced.items():
                scores.append({
                    "name": key,
                    "score": eval_result.score,
                    "passed": eval_result.passed,
                    "reason": eval_result.reason or "",
                    "category": self.category,
                    "source": "builtin" if self.builtin else "custom",
                    "eval_version": self.version,
                    "metadata": eval_result.metadata,
                })

        return scores

    def to_registration_dict(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Build the payload entry for POST /api/v1/evaluators/register.

        Returns the metadata the platform needs to surface this @eval as a
        first-class evaluator in the UI before any scores have been pushed.
        """
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "source_location": self.source_location,
            "source_location_extra": self.source_location_extra,
            "agent_name": agent_name,
        }

    def __repr__(self) -> str:
        return (
            f"DecimalEval(name={self.name!r}, category={self.category!r}, "
            f"version={self.version!r}, sampling_rate={self.sampling_rate})"
        )


def eval(
    fn: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    category: str = "quality",
    sampling_rate: float = 1.0,
    version: str = "1",
) -> Union[DecimalEval, Callable[..., DecimalEval]]:
    """Decorator to define a DecimalAI evaluation function.

    Can be used with or without arguments::

        @eval
        def check_output(trace: TraceData) -> bool: ...

        @eval(name="has_citation", category="llm_judge", sampling_rate=0.2)
        async def check_citation(trace: TraceData) -> EvalResult: ...

    Args:
        fn: The function to decorate (when used without parentheses).
        name: Score name. Defaults to the function name.
        category: Score category ("quality", "llm_judge", etc.).
        sampling_rate: Fraction of traces to evaluate (0.0–1.0).

    Returns:
        A ``DecimalEval`` wrapper around the function.

    Raises:
        TypeError: If the function signature is invalid.
        ValueError: If the name or sampling_rate is invalid.
    """
    if fn is not None:
        # Used without parentheses: @eval
        return DecimalEval(fn, name=name, category=category, sampling_rate=sampling_rate, version=version)

    # Used with parentheses: @eval(name="...", ...)
    def decorator(f: Callable) -> DecimalEval:
        return DecimalEval(f, name=name, category=category, sampling_rate=sampling_rate, version=version)

    return decorator


# ── Runner ───────────────────────────────────────────────────────


def run_evals(
    trace_data: TraceData,
    evals: Sequence[DecimalEval],
) -> List[Dict[str, Any]]:
    """Run a list of eval functions against a trace and collect scores.

    Args:
        trace_data: The trace to evaluate.
        evals: List of ``DecimalEval`` objects (from ``@eval`` decorator).

    Returns:
        List of score dicts ready for the API payload.
    """
    all_scores: List[Dict[str, Any]] = []

    for ev in evals:
        try:
            scores = ev.to_score_dicts(trace_data)
            all_scores.extend(scores)
        except Exception as e:
            logger.warning(
                "Eval '%s' failed: %s: %s", ev.name, type(e).__name__, e
            )

    return all_scores


def trace_to_trace_data(trace: Any) -> TraceData:
    """Convert a RunTrace (or dict) to a TraceData view for evals.

    Accepts either a RunTrace Pydantic model or a plain dict.
    """
    if isinstance(trace, dict):
        d = trace
    elif hasattr(trace, "model_dump"):
        d = trace.model_dump(mode="json")
    else:
        d = trace.__dict__

    # Flatten tool_calls from LLM calls
    tool_calls: List[ToolCallView] = []
    for llm_call in d.get("llm_calls", []):
        for tc in (llm_call.get("tool_calls_json") or llm_call.get("tool_calls") or []):
            func = tc.get("function", {})
            tool_calls.append(
                ToolCallView(
                    name=func.get("name") or tc.get("name", ""),
                    args=func.get("arguments", {}) if isinstance(func.get("arguments"), dict) else {},
                    result=tc.get("result"),
                )
            )

    # Build LlmCallView list
    llm_calls: List[LlmCallView] = []
    for lc in d.get("llm_calls", []):
        input_tok = lc.get("input_tokens") or lc.get("prompt_tokens")
        output_tok = lc.get("output_tokens") or lc.get("completion_tokens")
        total_tok = lc.get("total_tokens")
        if not total_tok and input_tok and output_tok:
            total_tok = input_tok + output_tok
        llm_calls.append(
            LlmCallView(
                model=lc.get("model_name") or lc.get("model"),
                prompt_tokens=input_tok,
                completion_tokens=output_tok,
                total_tokens=total_tok,
                tool_calls=[
                    ToolCallView(
                        name=(tc.get("function", {}).get("name") or tc.get("name", "")),
                        args=tc.get("function", {}).get("arguments", {}) if isinstance(tc.get("function", {}).get("arguments"), dict) else {},
                        result=tc.get("result"),
                    )
                    for tc in (lc.get("tool_calls_json") or lc.get("tool_calls") or [])
                ],
            )
        )

    # Compute total tokens
    total_tokens = sum(
        (lc.get("total_tokens") or (lc.get("input_tokens") or 0) + (lc.get("output_tokens") or 0))
        for lc in d.get("llm_calls", [])
    ) or None

    # Compute latency: trace-level timestamps first, then LLM call latency
    latency_ms = None
    started = d.get("started_at")
    ended = d.get("ended_at")
    if started and ended:
        from datetime import datetime

        try:
            if isinstance(started, str):
                started = datetime.fromisoformat(started)
            if isinstance(ended, str):
                ended = datetime.fromisoformat(ended)
            latency_ms = int((ended - started).total_seconds() * 1000)
        except (ValueError, TypeError):
            pass

    # Fallback: max latency from LLM calls
    if not latency_ms:
        call_latencies = [lc.get("latency_ms") for lc in d.get("llm_calls", []) if lc.get("latency_ms")]
        if call_latencies:
            latency_ms = max(call_latencies)

    return TraceData(
        id=str(d.get("id", "")),
        input=d.get("user_input_preview") or d.get("user_input") or "",
        output=d.get("final_output_preview") or d.get("final_output") or "",
        status=d.get("status", "success"),
        tool_calls=tool_calls,
        llm_calls=llm_calls,
        latency_ms=latency_ms,
        total_tokens=total_tokens,
        agent_name=d.get("agent_name", ""),
        manifest_id=str(d["manifest_id"]) if d.get("manifest_id") else None,
        metadata=d.get("metadata") or {},
        # Declared on TraceData and documented for custom evaluators to read,
        # but never assigned here — so every evaluator that read it got [] and
        # silently graded as though no skill had ever been active. Faithful to
        # the payload and nothing more: entries may be {"name": …, "hash": …}
        # or bare strings, and only `active_skills` is mirrored. Names the model
        # PULLED live in `skills_loaded_by_agent`, which TraceData has no view
        # of; folding them in here would make a field called "active" mean two
        # different things across SDK versions and destroy the property that an
        # entry in it is an explicit assertion rather than an inference.
        active_skills=[
            e.get("name") if isinstance(e, dict) else e
            for e in (d.get("active_skills") or [])
            if (e.get("name") if isinstance(e, dict) else e)
        ],
    )


def batch_eval(
    trace_ids: List[str],
    evals: Sequence[DecimalEval],
    client: Optional[Any] = None,
    max_workers: int = 4,
) -> Dict[str, Any]:
    """Run evals across multiple traces in batch.

    Fetches traces from the backend, converts to TraceData, runs evals,
    and pushes scores back. Ideal for offline eval passes.

    Args:
        trace_ids: List of trace IDs to evaluate.
        evals: List of DecimalEval objects to run.
        client: DecimalAIClient instance. If None, uses global client.
        max_workers: Max parallel workers for eval execution.

    Returns:
        Summary dict with per-eval pass/fail counts and overall stats.

    Example::

        from decimalai.evals import batch_eval, eval, TraceData

        @eval(name="is_polite")
        def check_polite(trace: TraceData) -> bool:
            return "please" in trace.output.lower()

        results = decimalai.batch_eval(
            trace_ids=["abc", "def", "ghi"],
            evals=[check_polite],
        )
        print(results["summary"])  # {"is_polite": {"passed": 2, "failed": 1}}
    """
    import concurrent.futures

    if client is None:
        from .. import _config
        client = _config._get_client()

    # Fetch traces
    traces_data: List[TraceData] = []
    for tid in trace_ids:
        try:
            resp = client._request_with_retry("GET", f"/api/v1/traces/{tid}")
            resp.raise_for_status()
            trace_dict = resp.json()
            traces_data.append(trace_to_trace_data(trace_dict))
        except Exception as e:
            logger.warning("Failed to fetch trace %s: %s", tid, e)

    # Run evals in parallel
    all_results: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, int]] = {}  # eval_name -> {passed, failed}

    def _eval_one(td: TraceData) -> List[Dict[str, Any]]:
        return run_evals(td, evals)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_eval_one, td): td for td in traces_data}
        for future in concurrent.futures.as_completed(futures):
            td = futures[future]
            try:
                scores = future.result(timeout=60)
                for s in scores:
                    ename = s.get("name", "unknown")
                    if ename not in summary:
                        summary[ename] = {"passed": 0, "failed": 0, "total": 0}
                    summary[ename]["total"] += 1
                    if s.get("passed"):
                        summary[ename]["passed"] += 1
                    else:
                        summary[ename]["failed"] += 1

                # Push scores back to the trace
                if scores:
                    try:
                        client.push_eval_scores(
                            trace_id=td.id,
                            source="batch_eval",
                            scores=scores,
                        )
                    except Exception as e:
                        logger.warning("Failed to push scores for trace %s: %s", td.id, e)

                all_results.extend(scores)
            except Exception as e:
                logger.warning("Eval failed for trace %s: %s", td.id, e)

    return {
        "traces_evaluated": len(traces_data),
        "traces_requested": len(trace_ids),
        "total_scores": len(all_results),
        "summary": summary,
    }


# ── Re-exports for convenience ──────────────────────────────────
# Users can import from decimalai.evals directly:
#   from decimalai.evals import Relevance, json_valid, contains

# Deterministic (always available)
# LLM-powered (requires litellm)
from .builtin import (  # noqa: E402, F401
    BUILTIN_EVALS,
    LLM_JUDGE_BUILTINS,
)
from .llm_evaluators import (  # noqa: E402, F401
    Conciseness,
    Factuality,
    Faithfulness,
    LlmEval,
    Relevance,
    Toxicity,
)
from .prebuilt import (  # noqa: E402, F401
    contains,
    json_valid,
    length_check,
    not_contains,
    regex_match,
)

