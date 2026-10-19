"""annotate_trace must match the backend CreateAnnotationRequest.

The endpoint uses extra="forbid", so the SDK must send ONLY allowed fields (never
text/annotation_type/span_id, which it does not take either), and reject an empty
annotation locally instead of letting the backend 422.
"""

from unittest.mock import MagicMock, patch

import pytest

from decimalai._client import DecimalAIClient

_FORBIDDEN = {"text", "annotation_type", "span_id"}
_ALLOWED = {
    "label", "rating", "correctness", "error_categories", "corrected_output",
    "tags", "notes", "score", "flagged_for_review", "add_to_dataset",
}


def _client():
    return DecimalAIClient(api_key="dai_sk_test", base_url="http://localhost:8000")


def _ok_resp():
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"id": "a1"}
    return resp


def test_notes_only_sends_allowed_fields():
    c = _client()
    with patch.object(c, "_request_with_retry", return_value=_ok_resp()) as m:
        c.annotate_trace("t1", notes="looks good")
    body = m.call_args.kwargs["json"]
    assert body == {"notes": "looks good"}
    assert not (_FORBIDDEN & body.keys())


@pytest.mark.parametrize("kwargs", [{"text": "a note"}, {"annotation_type": "note"},
                                    {"span_id": "s1"}])
def test_the_fields_the_endpoint_forbids_are_not_arguments(kwargs):
    c = _client()
    with patch.object(c, "_request_with_retry") as m:
        with pytest.raises(TypeError):
            c.annotate_trace("t1", notes="x", **kwargs)
    m.assert_not_called()


def test_empty_annotation_raises_locally():
    c = _client()
    with patch.object(c, "_request_with_retry") as m:
        with pytest.raises(ValueError):
            c.annotate_trace("t1")
        # A side-flag alone is not substantive content.
        with pytest.raises(ValueError):
            c.annotate_trace("t1", flagged_for_review=True)
    m.assert_not_called()


def test_rich_annotation_fields_pass_through():
    c = _client()
    with patch.object(c, "_request_with_retry", return_value=_ok_resp()) as m:
        c.annotate_trace("t1", label="thumbs_up", rating=5, tags=["good"], score=0.9)
    body = m.call_args.kwargs["json"]
    assert body == {"label": "thumbs_up", "rating": 5, "tags": ["good"], "score": 0.9}
    assert body.keys() <= _ALLOWED
