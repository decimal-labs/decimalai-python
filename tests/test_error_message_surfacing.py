"""Surface the server's error message instead of a bare status line.

``httpx.Response.raise_for_status()`` raises with a generic
``"Client error '400 Bad Request' for url ..."`` and discards the JSON body
the server returned — so a user who omits a required field sees the status
code, not the reason. ``_raise_for_status`` / ``DecimalAPIError`` fix that by
folding the server's ``detail`` / ``message`` and ``request_id`` into the
exception text, while staying a subclass of ``httpx.HTTPStatusError`` for
backward compatibility.
"""

import httpx
import pytest

from decimalai._client import DecimalAPIError, _raise_for_status


def _resp(status, *, json_body=None, text=None, headers=None, reason=None):
    req = httpx.Request("POST", "http://localhost:9999/api/v1/traces")
    kwargs = {"request": req, "headers": headers or {}}
    if json_body is not None:
        kwargs["json"] = json_body
    elif text is not None:
        kwargs["text"] = text
    return httpx.Response(status, **kwargs)


def test_surfaces_server_detail_and_request_id():
    """A FastAPI-style {detail, request_id} body lands in the message."""
    resp = _resp(
        400,
        json_body={
            "detail": "manifest_id is required",
            "request_id": "abc-123",
            "code": "HTTP_400",
        },
    )
    with pytest.raises(DecimalAPIError) as exc:
        _raise_for_status(resp)

    msg = str(exc.value)
    assert "manifest_id is required" in msg
    assert "abc-123" in msg
    assert "400" in msg
    # structured access too
    assert exc.value.status_code == 400
    assert exc.value.server_detail == "manifest_id is required"
    assert exc.value.request_id == "abc-123"


def test_falls_back_to_message_then_error_key():
    """When there's no `detail`, `message` (then `error`) is used."""
    resp = _resp(403, json_body={"message": "forbidden: not your org"})
    with pytest.raises(DecimalAPIError) as exc:
        _raise_for_status(resp)
    assert "forbidden: not your org" in str(exc.value)


def test_validation_list_detail_is_serialized():
    """FastAPI 422 detail (a list) is serialized, not dropped."""
    resp = _resp(
        422,
        json_body={"detail": [{"loc": ["body", "x"], "msg": "field required"}]},
    )
    with pytest.raises(DecimalAPIError) as exc:
        _raise_for_status(resp)
    assert "field required" in str(exc.value)


def test_non_json_body_falls_back_to_text():
    """A plain-text/HTML error body still surfaces something useful."""
    resp = _resp(500, text="Internal Server Error")
    with pytest.raises(DecimalAPIError) as exc:
        _raise_for_status(resp)
    msg = str(exc.value)
    assert "500" in msg
    assert "Internal Server Error" in msg


def test_request_id_from_header_when_absent_in_body():
    """request_id is read from the x-request-id header if not in the body."""
    resp = _resp(
        400,
        json_body={"detail": "bad"},
        headers={"x-request-id": "hdr-req-999"},
    )
    with pytest.raises(DecimalAPIError) as exc:
        _raise_for_status(resp)
    assert exc.value.request_id == "hdr-req-999"
    assert "hdr-req-999" in str(exc.value)


def test_is_backward_compatible_httpstatuserror():
    """Existing `except httpx.HTTPStatusError` handlers keep catching it."""
    resp = _resp(400, json_body={"detail": "nope"})
    with pytest.raises(httpx.HTTPStatusError):
        _raise_for_status(resp)
    # and the standard .response/.request attributes are populated
    try:
        _raise_for_status(resp)
    except httpx.HTTPStatusError as e:
        assert e.response.status_code == 400
        assert e.request is not None


def test_code_is_not_mistaken_for_request_id():
    """Server envelope {detail, code, request_id}: `code` lands in server_code,
    not request_id. Regression for the code→request_id confusion bug — when
    there is no request_id, the machine-readable `code` must NOT be folded into
    request_id (it is an error code, not a trace).
    """
    resp = _resp(
        400,
        json_body={
            "detail": "manifest_id is required",
            "code": "validation_error",
        },
    )
    with pytest.raises(DecimalAPIError) as exc:
        _raise_for_status(resp)

    assert exc.value.server_code == "validation_error"
    assert exc.value.request_id is None
    msg = str(exc.value)
    assert "code=validation_error" in msg
    assert "request_id=validation_error" not in msg


def test_full_envelope_keeps_code_and_request_id_distinct():
    """All three envelope fields are surfaced into their own attributes."""
    resp = _resp(
        409,
        json_body={
            "detail": "duplicate manifest",
            "code": "conflict",
            "request_id": "req-xyz-7",
        },
    )
    with pytest.raises(DecimalAPIError) as exc:
        _raise_for_status(resp)

    assert exc.value.server_detail == "duplicate manifest"
    assert exc.value.server_code == "conflict"
    assert exc.value.request_id == "req-xyz-7"
    msg = str(exc.value)
    assert "code=conflict" in msg
    assert "request_id=req-xyz-7" in msg


def test_success_passes_through():
    """2xx responses do not raise."""
    _raise_for_status(_resp(200, json_body={"ok": True}))
    _raise_for_status(_resp(204))
