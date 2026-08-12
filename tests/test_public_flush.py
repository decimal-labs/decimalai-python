"""Tests for the public `decimalai.flush()` helper.

`decimalai.flush()` used to raise AttributeError — only the private
`_atexit_flush` and the underlying
`_client.DecimalAIClient.flush()` existed. Now users in shutdown-fragile
environments (CI SIGKILL, async event loops, daemonized workers, Jupyter
kernel-restart) can trigger a synchronous flush explicitly.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_public_flush_in_module_namespace():
    """decimalai.flush must exist as a callable at module level."""
    import decimalai
    assert hasattr(decimalai, "flush"), "decimalai.flush is missing from the module"
    assert callable(decimalai.flush)
    assert "flush" in decimalai.__all__, "flush should be exported via __all__"


def test_public_flush_delegates_to_client_flush(monkeypatch):
    """When a client exists, decimalai.flush() calls client.flush()."""
    import decimalai
    from decimalai import _config

    fake_client = MagicMock()
    monkeypatch.setattr(_config, "_client", fake_client, raising=False)
    decimalai.flush()
    fake_client.flush.assert_called_once_with()


def test_public_flush_is_noop_when_uninitialized(monkeypatch):
    """When init() hasn't been called, flush() must not raise."""
    import decimalai
    from decimalai import _config

    monkeypatch.setattr(_config, "_client", None, raising=False)
    # No exception, no return value to inspect.
    decimalai.flush()


def test_public_flush_swallows_client_errors(monkeypatch, caplog):
    """A failing client.flush() must NOT propagate — log and continue."""
    import decimalai
    from decimalai import _config

    failing_client = MagicMock()
    failing_client.flush.side_effect = RuntimeError("network down")
    monkeypatch.setattr(_config, "_client", failing_client, raising=False)
    # Should not raise.
    decimalai.flush()
    failing_client.flush.assert_called_once()
