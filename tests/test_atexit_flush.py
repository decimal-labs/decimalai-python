"""Tests for the atexit flush handler.

Without atexit, a script that ingests <50 traces and exits silently
drops them — the buffer is in-memory and never flushed. The handler
registered by init() guarantees a final flush on interpreter shutdown.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_init_registers_atexit_handler_once(monkeypatch):
    """Repeated init() calls don't stack multiple atexit registrations."""
    import decimalai

    # Reset module-level registration flag so this test is order-independent.
    monkeypatch.setattr(decimalai, "_atexit_registered", False)

    registered_funcs = []
    fake_register = MagicMock(side_effect=lambda f: registered_funcs.append(f))

    with patch("decimalai.atexit.register", fake_register):
        decimalai.init(api_key="dai_sk_test_atexit_1", base_url="http://localhost:8000")
        decimalai.init(api_key="dai_sk_test_atexit_2", base_url="http://localhost:8000")
        decimalai.init(api_key="dai_sk_test_atexit_3", base_url="http://localhost:8000")

    assert fake_register.call_count == 1, (
        "atexit.register should be called exactly once across multiple init() calls"
    )
    assert registered_funcs[0].__name__ == "_atexit_flush"


def test_disabled_init_does_not_register(monkeypatch):
    """init(enabled=False) creates no client, so atexit isn't useful."""
    import decimalai

    monkeypatch.setattr(decimalai, "_atexit_registered", False)
    fake_register = MagicMock()

    with patch("decimalai.atexit.register", fake_register):
        decimalai.init(api_key="dai_sk_x", base_url="http://localhost:8000", enabled=False)

    fake_register.assert_not_called()


def test_atexit_flush_calls_client_flush():
    """The registered handler delegates to client.flush()."""
    from decimalai import _atexit_flush
    from decimalai import _config as _cfg

    fake_client = MagicMock()
    original = _cfg._client
    _cfg._client = fake_client
    try:
        _atexit_flush()
        fake_client.flush.assert_called_once()
    finally:
        _cfg._client = original


def test_atexit_flush_swallows_exceptions():
    """A flush error during shutdown must not raise — the interpreter
    is already exiting and we don't want to mask the original exit reason."""
    from decimalai import _atexit_flush
    from decimalai import _config as _cfg

    fake_client = MagicMock()
    fake_client.flush.side_effect = RuntimeError("network gone")
    original = _cfg._client
    _cfg._client = fake_client
    try:
        # Must not raise
        _atexit_flush()
    finally:
        _cfg._client = original


def test_atexit_flush_no_client_is_noop():
    """If no client was ever initialized, the handler is a no-op."""
    from decimalai import _atexit_flush
    from decimalai import _config as _cfg

    original = _cfg._client
    _cfg._client = None
    try:
        _atexit_flush()  # must not raise
    finally:
        _cfg._client = original
