"""DeepEval/LangSmith adapters must use NON-reserved source labels.

The backend reserves {deepeval, langsmith, braintrust, ragas} for HMAC-signed
webhook callbacks (anti-spoofing) and 422s on the authenticated push path, so the
SDK adapters send a distinct provenance label ("-import") and carry the friendly
display name in metadata.source_label. They also must not silently swallow a real
(non-404) push error.
"""

import types
from unittest.mock import MagicMock

import httpx
import pytest

from decimalai._client import DecimalAPIError
from decimalai.evals.adapters import push_deepeval_results, push_langsmith_scores

_RESERVED = {"deepeval", "langsmith", "braintrust", "ragas"}


def _deepeval_result(trace_id="t1"):
    metric = types.SimpleNamespace(
        name="Answer Relevancy", score=0.9, success=True, reason="ok"
    )
    return types.SimpleNamespace(
        additional_metadata={"trace_id": trace_id}, metrics_data=[metric]
    )


def _api_error(status_code):
    resp = httpx.Response(
        status_code,
        json={"detail": "boom"},
        request=httpx.Request("POST", "http://localhost:8000/x"),
    )
    return DecimalAPIError(resp)


def test_push_deepeval_uses_non_reserved_source():
    client = MagicMock()
    client.push_eval_scores.return_value = {"status": "ok"}
    push_deepeval_results(client, [_deepeval_result()])
    kwargs = client.push_eval_scores.call_args.kwargs
    assert kwargs["source"] not in _RESERVED
    assert kwargs["source"] == "deepeval-import"
    assert kwargs.get("metadata", {}).get("source_label") == "DeepEval"


def test_push_langsmith_uses_non_reserved_source():
    client = MagicMock()
    client.push_eval_scores.return_value = {"status": "ok"}
    push_langsmith_scores(client, "t1", [{"key": "correctness", "score": 0.9}])
    kwargs = client.push_eval_scores.call_args.kwargs
    assert kwargs["source"] not in _RESERVED
    assert kwargs["source"] == "langsmith-import"
    assert kwargs.get("metadata", {}).get("source_label") == "LangSmith"


def test_push_deepeval_does_not_swallow_non_404_errors():
    """A 422 (validation/auth) would drop EVERY score — it must surface, not warn."""
    client = MagicMock()
    client.push_eval_scores.side_effect = _api_error(422)
    with pytest.raises(DecimalAPIError):
        push_deepeval_results(client, [_deepeval_result()])


def test_push_deepeval_skips_404_and_continues():
    """A 404 is a per-result bad/cross-tenant trace_id — skip it, don't abort."""
    client = MagicMock()
    client.push_eval_scores.side_effect = _api_error(404)
    out = push_deepeval_results(client, [_deepeval_result()])
    assert out == []  # nothing pushed, but no exception raised
