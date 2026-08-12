"""Lock in: a failed manifest registration does NOT poison the tracker.

Deep-audit finding (sdk-core): `_maybe_register_manifest` calls
`check_and_update`, which immediately commits `_last_hash` and returns
True BEFORE the register HTTP call. If that POST fails, the code falls
to the synthetic-id branch but (pre-fix) never rolled back the tracker.
So the next trace with the same manifest early-returned and registration
was never re-attempted even after the backend recovered — a single
first-trace hiccup silently broke ingestion for the whole process run.

The fix resets the tracker on the synthetic-id fallback so a later trace
re-attempts registration once the backend is healthy again.

These tests do NOT run a backend — they mock the client.
"""

from unittest.mock import MagicMock, patch

import decimalai.generic as generic
from decimalai.generic import TraceContext


def _fresh_ctx():
    return TraceContext(agent_name="retry-after-failure", session_id=None, auto_send=False)


def test_failed_registration_does_not_poison_tracker():
    """After a transient registration failure, a later trace with the
    same manifest must RE-ATTEMPT registration (backend recovered).

    Pre-fix: the tracker kept the committed hash, so the second call
    early-returned via check_and_update and register_manifest was never
    called again.
    """
    # Reset shared module state so the test is order-independent.
    generic._manifest_tracker.reset()

    mock_client = MagicMock()
    # First attempt(s) fail with a transient (non-auth) error; the retry
    # budget is exhausted, so we fall to the synthetic-id branch. The
    # second _maybe_register_manifest() call (backend recovered) succeeds.
    mock_client.register_manifest.side_effect = [
        RuntimeError("503 service unavailable"),
        RuntimeError("503 service unavailable"),
        RuntimeError("503 service unavailable"),
        {"manifest_id": "recovered-mid"},
    ]

    with patch("decimalai._config._is_enabled", return_value=True), \
         patch("decimalai._config._get_client", return_value=mock_client), \
         patch("decimalai._config._sender") as mock_sender:
        mock_sender.record_manifest_error = MagicMock()

        # First trace: registration fails, falls back to synthetic id.
        _fresh_ctx()._maybe_register_manifest()
        calls_after_first = mock_client.register_manifest.call_count
        assert calls_after_first >= 1

        # Second trace with the SAME manifest, backend now healthy.
        _fresh_ctx()._maybe_register_manifest()

        # The fix: registration WAS re-attempted. Pre-fix the tracker
        # still held the hash and check_and_update short-circuited,
        # leaving call_count unchanged.
        assert mock_client.register_manifest.call_count > calls_after_first, (
            "A failed registration poisoned the tracker: the recovered "
            "backend was never retried for the same manifest."
        )

    generic._manifest_tracker.reset()


def test_successful_registration_still_dedups():
    """Guard against over-correcting: a SUCCESSFUL registration must
    still dedup — the same manifest is not re-registered on every trace.
    """
    generic._manifest_tracker.reset()

    mock_client = MagicMock()
    mock_client.register_manifest.return_value = {"manifest_id": "ok-mid"}

    with patch("decimalai._config._is_enabled", return_value=True), \
         patch("decimalai._config._get_client", return_value=mock_client):
        _fresh_ctx()._maybe_register_manifest()
        _fresh_ctx()._maybe_register_manifest()

        assert mock_client.register_manifest.call_count == 1, (
            "Same manifest registered more than once — the tracker stopped "
            "deduplicating successful registrations."
        )

    generic._manifest_tracker.reset()
