"""Lock in: SDK send failures are visible via WARNING log + `last_send_error()`.

The SDK used to swallow 4xx/5xx responses from the background-sender
path at DEBUG level — `flush()` returned normally, no error surfaced,
traces silently disappeared.

Now:
1. WARNING log fires immediately on send failure (visible at default INFO level)
2. `decimalai.flush()` now also drains the BackgroundSender queue
3. `decimalai.last_send_error()` returns the most recent failure for
   programmatic introspection
"""

import logging
from concurrent.futures import Future

from unittest.mock import MagicMock, patch


def test_last_send_error_returns_none_on_clean_state():
    """No flushes yet — last_send_error should be None."""
    import decimalai
    from decimalai import _config

    # Reset the global sender state so the previous test's state can't leak.
    _config._sender._last_send_error = None

    err = decimalai.last_send_error()
    assert err is None


def test_last_send_error_captures_failed_future():
    """If a background-sent future raises, last_send_error captures it."""
    import decimalai
    from decimalai import _config

    sender = _config._sender
    sender._last_send_error = None

    # Manufacture a failed future and place it in _pending.
    failed = Future()
    failed.set_exception(RuntimeError("simulated 400"))
    sender._pending = [failed]

    sender.flush(timeout=0.5)

    err = decimalai.last_send_error()
    assert err is not None
    assert isinstance(err, RuntimeError)
    assert "simulated 400" in str(err)


def test_send_failure_logs_at_warning_level(caplog):
    """Failed send produces a WARNING-level log (it used to be DEBUG).

    The message no longer says "during flush", because the log is now
    emitted per-failure from the sender thread rather than only at flush
    boundaries. It still says "Background send failed" so log greppers
    keep working.
    """
    import decimalai
    from decimalai import _config

    sender = _config._sender
    sender._last_send_error = None

    failed = Future()
    failed.set_exception(ValueError("simulated backend reject"))
    sender._pending = [failed]

    with caplog.at_level(logging.WARNING, logger="decimalai"):
        sender.flush(timeout=0.5)

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "Expected at least one WARNING-level log from sender.flush"
    assert any("Background send failed" in r.getMessage() for r in warnings)


def test_decimalai_flush_drains_background_sender():
    """`decimalai.flush()` must also drain the BackgroundSender queue.

    It used to drain only `_client._trace_buffer` — the per-trace
    `auto_send=True` path went through `_sender.submit()` which was
    NOT awaited. Verified by patching sender.flush and asserting it
    gets called.
    """
    import decimalai
    from decimalai import _config

    with patch.object(_config._sender, "flush") as mock_flush:
        decimalai.flush()
        # Even with no prior init, flush() should attempt to drain
        # the sender. The fact that mock_flush was hit at all means
        # the new code path runs.
        assert mock_flush.called, (
            "decimalai.flush() must call _sender.flush() so the per-trace "
            "auto_send=True path doesn't escape the user's flush."
        )
