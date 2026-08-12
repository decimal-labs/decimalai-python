"""Lock in: decimalai.eval(source_label=...) actually reaches the backend.

Deep-audit finding (sdk-core): eval() accepted and documented
``source_label``, built ``metadata = {"source_label": ...}``, but then
called push_eval_scores WITHOUT passing metadata — so the documented
param was a silent no-op and the backend never saw the custom label.

The fix threads metadata through push_eval_scores (which forwards it to
the backend's ``metadata`` field; the backend reads ``source_label`` as a
display-name override).
"""

from unittest.mock import MagicMock, patch

import decimalai


def test_eval_forwards_source_label_as_metadata():
    mock_client = MagicMock()
    mock_client.push_eval_scores.return_value = {"ok": True}

    with patch("decimalai._config._get_client", return_value=mock_client):
        decimalai.eval(
            trace_id="abc123",
            name="factual_accuracy",
            score=0.75,
            source="my-pipeline",
            source_label="My RAG Eval",
        )

    assert mock_client.push_eval_scores.called
    _args, kwargs = mock_client.push_eval_scores.call_args
    assert kwargs.get("metadata") == {"source_label": "My RAG Eval"}, (
        "eval() must forward source_label to the backend via metadata — "
        "it was previously dropped, making the documented param a no-op."
    )


def test_eval_without_source_label_sends_no_metadata():
    """When no source_label is given, metadata must be None (not an empty
    dict that the backend would have to special-case)."""
    mock_client = MagicMock()
    mock_client.push_eval_scores.return_value = {"ok": True}

    with patch("decimalai._config._get_client", return_value=mock_client):
        decimalai.eval(trace_id="abc123", name="coherence", score=0.9)

    _args, kwargs = mock_client.push_eval_scores.call_args
    assert kwargs.get("metadata") is None


def test_push_eval_scores_includes_metadata_in_payload():
    """The client method itself must put metadata into the request body so
    the backend's ``metadata`` field receives it."""
    from decimalai._client import DecimalAIClient

    client = DecimalAIClient.__new__(DecimalAIClient)
    captured = {}

    def fake_request(method, path, json=None, **kw):
        captured["json"] = json
        resp = MagicMock()
        resp.json.return_value = {"ok": True}
        return resp

    client._request_with_retry = fake_request  # type: ignore[attr-defined]

    client.push_eval_scores(
        trace_id="abc123",
        source="custom",
        scores=[{"name": "x", "score": 1.0, "passed": True}],
        metadata={"source_label": "Lbl"},
    )

    assert captured["json"].get("metadata") == {"source_label": "Lbl"}
