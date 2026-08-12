"""Helpers for pushing eval scores from external platforms to DecimalAI.

Provides convenience functions that auto-map results from DeepEval,
LangSmith, and other eval frameworks into ``push_eval_scores()`` calls.

Moved from ``decimalai/eval.py`` to consolidate the eval subsystem.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .._client import DecimalAPIError

if TYPE_CHECKING:
    from .._client import DecimalAIClient

logger = logging.getLogger("decimalai.evals.adapters")


def push_deepeval_results(
    client: "DecimalAIClient",
    test_results: Any,
    trace_id_field: str = "input",
) -> List[Dict[str, Any]]:
    """Auto-map DeepEval test results → push_eval_scores calls.

    Args:
        client: DecimalAI client instance.
        test_results: Result from ``deepeval.evaluate(test_cases, metrics)``.
            Expected to have a ``.test_results`` attribute containing a list
            of test case results, each with ``.metrics_data``.
        trace_id_field: Field name on the test case used to look up the trace ID.
            Typically the test case input contains a trace_id.

    Returns:
        List of API responses from push_eval_scores calls.

    Example::

        from deepeval import evaluate
        from decimalai.evals.adapters import push_deepeval_results

        results = evaluate(test_cases, metrics)
        client = DecimalClient(api_key="...")
        push_deepeval_results(client, results)
    """
    responses: List[Dict[str, Any]] = []

    # Handle both old-style list and new EvaluationResult object
    results_list = getattr(test_results, "test_results", test_results)
    if not isinstance(results_list, (list, tuple)):
        logger.warning("Cannot parse DeepEval results: expected list or EvaluationResult")
        return responses

    for test_result in results_list:
        # Extract trace_id from test case
        trace_id = _extract_trace_id(test_result, trace_id_field)
        if not trace_id:
            logger.warning("Skipping DeepEval result: no trace_id found")
            continue

        # Map metrics to scores
        scores = _map_deepeval_metrics(test_result)
        if not scores:
            continue

        try:
            # "deepeval"/"langsmith"/"braintrust"/"ragas" are RESERVED by the
            # backend for HMAC-signed webhook callbacks (anti-spoofing) and 422
            # on the authenticated push path. Use a distinct, honest provenance
            # label and carry the friendly display name in metadata.source_label.
            resp = client.push_eval_scores(
                trace_id=trace_id,
                source="deepeval-import",
                scores=scores,
                metadata={"source_label": "DeepEval"},
            )
            responses.append(resp)
        except DecimalAPIError as e:
            # A 404 is a per-result bad/cross-tenant trace_id — skip it and keep
            # processing the batch. Anything else (e.g. a 422 validation error or
            # auth failure) would silently drop EVERY score, so surface it
            # instead of swallowing it as the old blanket except did.
            if e.status_code == 404:
                logger.warning("Skipping DeepEval scores for %s: trace not found", trace_id[:8])
                continue
            raise

    logger.info("Pushed DeepEval scores for %d traces", len(responses))
    return responses


def push_langsmith_scores(
    client: "DecimalAIClient",
    trace_id: str,
    run_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Push LangSmith evaluation scores to a DecimalAI trace.

    Args:
        client: DecimalAI client instance.
        trace_id: DecimalAI trace ID to attach scores to.
        run_scores: List of score dicts from LangSmith evaluation.
            Each should have "key" (metric name) and "score" (float).

    Returns:
        API response from push_eval_scores.

    Example::

        from decimalai.evals.adapters import push_langsmith_scores

        scores = [
            {"key": "correctness", "score": 0.9},
            {"key": "helpfulness", "score": 0.85, "comment": "Good response"},
        ]
        push_langsmith_scores(client, trace_id="abc123", run_scores=scores)
    """
    mapped_scores = []
    for s in run_scores:
        mapped_scores.append({
            "name": s.get("key", s.get("name", "unknown")),
            "score": float(s.get("score", 0.0)),
            "reason": s.get("comment") or s.get("reason"),
            "passed": float(s.get("score", 0.0)) >= 0.5,
        })

    # "langsmith" is reserved by the backend for HMAC webhooks (see
    # push_deepeval_results); use a distinct provenance label + display name.
    return client.push_eval_scores(
        trace_id=trace_id,
        source="langsmith-import",
        scores=mapped_scores,
        metadata={"source_label": "LangSmith"},
    )


def push_custom_scores(
    client: "DecimalAIClient",
    trace_id: str,
    source: str,
    scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Push custom evaluation scores to a DecimalAI trace.

    This is a thin wrapper around ``client.push_eval_scores()`` that
    normalizes score formats.

    Args:
        client: DecimalAI client instance.
        trace_id: Trace ID to attach scores to.
        source: Name of the eval source (e.g., "braintrust", "ragas", "custom").
        scores: List of score dicts with at least "name" and "score".

    Returns:
        API response.
    """
    normalized: List[Dict[str, Any]] = []
    for s in scores:
        score_val = float(s.get("score", 0.0))
        normalized.append({
            "name": s.get("name", "unknown"),
            "score": score_val,
            "passed": s.get("passed", score_val >= 0.5),
            "reason": s.get("reason"),
        })

    return client.push_eval_scores(
        trace_id=trace_id,
        source=source,
        scores=normalized,
    )


# ── Internal Helpers ──────────────────────────────────────────────


def _extract_trace_id(test_result: Any, field: str = "input") -> Optional[str]:
    """Extract trace_id from a DeepEval test result."""
    # Try direct attribute
    trace_id = getattr(test_result, "trace_id", None)
    if trace_id:
        return str(trace_id)

    # Try additional_metadata dict
    metadata = getattr(test_result, "additional_metadata", {}) or {}
    if isinstance(metadata, dict):
        trace_id = metadata.get("trace_id") or metadata.get("decimal_trace_id")
        if trace_id:
            return str(trace_id)

    return None


def _map_deepeval_metrics(test_result: Any) -> List[Dict[str, Any]]:
    """Map DeepEval metric data to DecimalAI score format."""
    scores: List[Dict[str, Any]] = []

    # Try metrics_data attribute (DeepEval 1.x)
    metrics_data = getattr(test_result, "metrics_data", None) or []

    for metric in metrics_data:
        name = getattr(metric, "name", None) or getattr(metric, "metric", None)
        if not name:
            continue

        score_val = getattr(metric, "score", None)
        if score_val is None:
            continue

        scores.append({
            "name": str(name).lower().replace(" ", "_"),
            "score": float(score_val),
            "passed": bool(getattr(metric, "success", score_val >= 0.5)),
            "reason": str(getattr(metric, "reason", "") or ""),
        })

    return scores
