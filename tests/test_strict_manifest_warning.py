"""Warn at the source when the backend requires manifest_id on ingest.

`decimalai.init(verify=True)` caches the backend's strict-mode flag
(`require_manifest_on_ingest`) onto the global config. The init() docstring
promises the SDK will then "warn about manifest_id-required misconfigurations
at the source". These tests pin that promise: when the cached flag is True and
a trace is ingested without a manifest_id, the client logs a one-time, actionable
warning *before* the POST — rather than letting the caller hit a bare backend 400.
"""

import logging
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Fresh global config + a reset of the warn-once latch before each test."""
    import decimalai._client as client_mod
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    # reset the warn-once latch so each test sees a clean slate
    client_mod._STRICT_MANIFEST_WARNED = False
    yield
    client_mod._STRICT_MANIFEST_WARNED = False


def _make_client():
    from decimalai._client import DecimalAIClient

    c = DecimalAIClient(api_key="dai_sk_test", base_url="http://localhost:8000")
    # stub the transport so we never make a real call; return a benign 200
    c._http = MagicMock()
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok", "id": "trace-1"}
    c._http.request.return_value = resp
    return c


def _set_strict(value):
    import decimalai._config as cfg

    cfg._config.backend_require_manifest_on_ingest = value


def test_warns_when_strict_and_no_manifest_id(caplog):
    """Strict backend + raw trace lacking manifest_id → one actionable warning."""
    _set_strict(True)
    c = _make_client()
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        c.ingest_raw_trace({"agent_name": "x", "input": {}, "output": {}})
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("manifest_id" in m for m in msgs), msgs
    # the POST still happened — we warn, we don't block
    assert c._http.request.called


def test_warns_only_once(caplog):
    """The warning is rate-limited to once per process (no log spam on N traces)."""
    _set_strict(True)
    c = _make_client()
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        for _ in range(5):
            c.ingest_raw_trace({"agent_name": "x", "input": {}, "output": {}})
    warns = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "manifest_id" in r.getMessage()
    ]
    assert len(warns) == 1, f"expected exactly one warning, got {len(warns)}"


def test_no_warning_when_manifest_id_present(caplog):
    """A trace that carries a manifest_id must not trip the warning."""
    _set_strict(True)
    c = _make_client()
    with caplog.at_level(logging.WARNING, logger="decimalai"):
        c.ingest_raw_trace(
            {"agent_name": "x", "input": {}, "output": {}, "manifest_id": "m-1"}
        )
    warns = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "manifest_id" in r.getMessage()
    ]
    assert warns == []


def test_no_warning_when_backend_not_strict(caplog):
    """Default/non-strict backend (flag False or None) → never warn."""
    for flag in (False, None):
        import decimalai._client as client_mod
        client_mod._STRICT_MANIFEST_WARNED = False
        _set_strict(flag)
        c = _make_client()
        with caplog.at_level(logging.WARNING, logger="decimalai"):
            c.ingest_raw_trace({"agent_name": "x", "input": {}, "output": {}})
        warns = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and "manifest_id" in r.getMessage()
        ]
        assert warns == [], f"flag={flag!r} should not warn"
        caplog.clear()
