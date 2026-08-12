"""A lone UTF-16 surrogate must not crash the upload encoder.

A lone surrogate (e.g. ``"\\ud800"``) is a valid Python ``str`` but cannot be
UTF-8 encoded, so httpx's ``json=`` path (``ensure_ascii=False``) raises
``UnicodeEncodeError`` *client-side, before any request is sent* — crashing
``ingest_trace`` / ``ingest_traces_batch`` / ``register_manifest`` in the
caller's process. ``_scrub_surrogates`` replaces such code points with the
UTF-8 replacement char so the request body builds. These tests exercise the real upload encoder
(``httpx._content.encode_json``); they are RED before the fix (raise) and GREEN
after (encode succeeds), and assert clean payloads round-trip byte-identically.
"""

import httpx._content as _content
import pytest

from decimalai._client import _scrub_surrogates
from decimalai.schema.manifest import ManifestSnapshot
from decimalai.schema.trace import RunTrace

SURROGATE = "\ud800"  # lone high surrogate — not UTF-8 encodable


def _encode(payload):
    """Mirror httpx's json= upload path (ensure_ascii=False); return body bytes.

    ``encode_json`` returns ``(headers, ByteStream)`` and raises while building
    the stream's bytes for an un-encodable payload, so materializing it here is
    what reproduces the client-side crash.
    """
    _headers, stream = _content.encode_json(payload)
    return b"".join(stream)


def test_raw_dump_crashes_encoder_without_scrub():
    """Guard: confirms the bug is real on the actual upload encoder."""
    trace = RunTrace(agent_name=SURROGATE)
    with pytest.raises(UnicodeEncodeError):
        _encode(trace.model_dump(mode="json"))


@pytest.mark.parametrize(
    "trace",
    [
        RunTrace(agent_name=SURROGATE),
        RunTrace(session_id="s", active_skills=[{"name": SURROGATE}]),
    ],
    ids=["agent_name", "active_skills"],
)
def test_scrubbed_trace_encodes(trace):
    """RED before fix (raises), GREEN after: scrubbed trace builds a request body."""
    body = _encode(_scrub_surrogates(trace.model_dump(mode="json")))
    assert isinstance(body, (bytes, bytearray))
    text = body.decode("utf-8")  # well-formed UTF-8 — would raise if not
    assert SURROGATE not in text  # lone surrogate replaced, not on the wire


def test_scrubbed_manifest_encodes():
    """ManifestSnapshot.agent_name surrogate -> scrubbed -> encodes."""
    manifest = ManifestSnapshot(agent_name=SURROGATE, manifest_hash="abc")
    body = _encode(_scrub_surrogates(manifest.model_dump(mode="json")))
    body.decode("utf-8")
    assert SURROGATE not in body.decode("utf-8")


def test_clean_trace_roundtrips_byte_identical():
    """No-op for clean payloads: scrub must not alter a surrogate-free trace."""
    trace = RunTrace(agent_name="weather-bot", active_skills=[{"name": "code-review"}])
    dumped = trace.model_dump(mode="json")
    assert _encode(_scrub_surrogates(dumped)) == _encode(dumped)


def test_scrub_is_pure_does_not_mutate_input():
    """Scrub returns a new structure; the caller's serialized dict is untouched."""
    dumped = RunTrace(agent_name=SURROGATE).model_dump(mode="json")
    before = dumped["agent_name"]
    _scrub_surrogates(dumped)
    assert dumped["agent_name"] == before == SURROGATE
