"""Tests for SDK send-failure handling — retry, backoff, buffer preservation.

Two failure families, deliberately in one file because the buffer rules only
make sense side by side:

* **429 rate limit** — retried with backoff, buffer preserved (original).
* **Transient 5xx** — 502/503/504 from a proxy or load balancer, and 500 on the
  idempotent trace POSTs. Retried on the SAME ladder, buffer preserved. Before
  this, every non-429 status short-circuited the retry loop and ``flush()``
  cleared the buffer, so ONE 503 destroyed up to a full 50-trace batch.

A 4xx stays terminal in both directions: no retry, buffer cleared.
"""

import time
from unittest.mock import patch, MagicMock, PropertyMock

import httpx
import pytest

from decimalai._client import (
    DecimalAIClient,
    DecimalAPIError,
    DecimalQuotaExceededError,
    DecimalRateLimitError,
    _AUTO_FLUSH_THRESHOLD,
    _MAX_RETRIES,
)
from decimalai.schema.trace import RunTrace


def _make_client() -> DecimalAIClient:
    return DecimalAIClient(
        api_key="dai_sk_test",
        base_url="http://localhost:9999",
    )


def _mock_429_response(retry_after: str = "1") -> httpx.Response:
    """Create a mock 429 response with Retry-After header."""
    request = httpx.Request("POST", "http://localhost:9999/api/v1/traces")
    response = httpx.Response(
        429,
        headers={"Retry-After": retry_after, "Content-Type": "application/json"},
        json={"detail": "Rate limit exceeded", "plan": "default", "limit": 120},
        request=request,
    )
    return response


def _mock_200_response() -> httpx.Response:
    """Create a mock 200 response."""
    request = httpx.Request("POST", "http://localhost:9999/api/v1/traces")
    response = httpx.Response(
        200,
        json={"status": "ok", "trace_id": "abc123"},
        request=request,
    )
    return response


def _mock_500_response() -> httpx.Response:
    """Create a mock 500 response."""
    request = httpx.Request("POST", "http://localhost:9999/api/v1/traces")
    response = httpx.Response(
        500,
        text="Internal Server Error",
        request=request,
    )
    return response


def _mock_5xx_response(status: int = 503) -> httpx.Response:
    """Create a mock transient server error, shaped like a real one.

    503s in production come from the Google Frontend in front of Cloud Run while
    a revision has no healthy instance — an HTML body, no JSON, no Retry-After.
    """
    request = httpx.Request("POST", "http://localhost:9999/api/v1/traces")
    return httpx.Response(
        status,
        headers={"Content-Type": "text/html; charset=UTF-8"},
        text="<html><title>503 Service Unavailable</title></html>",
        request=request,
    )


def _mock_400_response() -> httpx.Response:
    """Create a mock 400 — the server has judged the payload itself."""
    request = httpx.Request("POST", "http://localhost:9999/api/v1/traces")
    return httpx.Response(
        400,
        headers={"Content-Type": "application/json"},
        json={"detail": "manifest_id is required", "code": "validation_error"},
        request=request,
    )


class TestRetryOn429:
    """Test retry-with-backoff when server returns 429."""

    @patch("time.sleep")
    def test_retry_on_429_then_success(self, mock_sleep):
        """Client retries on 429 and succeeds on subsequent attempt."""
        client = _make_client()

        # First call: 429, second call: 200
        with patch.object(client._http, "request") as mock_req:
            mock_req.side_effect = [_mock_429_response(), _mock_200_response()]
            trace = RunTrace(agent_name="test")
            result = client.ingest_trace(trace)

        assert result["status"] == "ok"
        assert mock_req.call_count == 2
        mock_sleep.assert_called_once()  # Waited between retries

    @patch("time.sleep")
    def test_retry_respects_retry_after_header(self, mock_sleep):
        """Client uses the Retry-After header value to determine wait time."""
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.side_effect = [
                _mock_429_response(retry_after="5"),
                _mock_200_response(),
            ]
            trace = RunTrace(agent_name="test")
            client.ingest_trace(trace)

        # Should wait at least 5 seconds (Retry-After value)
        actual_delay = mock_sleep.call_args[0][0]
        assert actual_delay >= 5.0

    @patch("time.sleep")
    def test_max_retries_then_raises(self, mock_sleep):
        """Client raises DecimalRateLimitError after exhausting retries."""
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            # All responses are 429
            mock_req.return_value = _mock_429_response()
            trace = RunTrace(agent_name="test")

            with pytest.raises(DecimalRateLimitError) as exc_info:
                client.ingest_trace(trace)

        assert "retries" in str(exc_info.value).lower()
        assert mock_req.call_count == _MAX_RETRIES + 1  # initial + retries

    @patch("time.sleep")
    def test_4xx_errors_not_retried(self, mock_sleep):
        """A 4xx raises immediately without retry — the payload is the problem.

        Retrying it burns wall-clock to reach the same verdict. (This test used
        to assert the same of a 500, which is how a whole class of transient
        server errors ended up with zero retries; see
        TestRetryOnTransientServerErrors.)
        """
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_400_response()
            trace = RunTrace(agent_name="test")

            with pytest.raises(httpx.HTTPStatusError):
                client.ingest_trace(trace)

        assert mock_req.call_count == 1  # No retry
        mock_sleep.assert_not_called()


class TestRetryOnTransientServerErrors:
    """502/503/504 — and 500 on the idempotent trace POSTs — are retried.

    The bug this class exists for: the retry loop returned or raised for every
    non-429 status, so a 503 from the load balancer got zero retries and the
    buffered batch was then thrown away by flush(). Measured on 2026-08-28,
    208 Google-Frontend 503s lined up with 210 trace-export failures in the
    same hour (minute-level r = 0.926).
    """

    @pytest.mark.parametrize("status", [502, 503, 504])
    @patch("time.sleep")
    def test_transient_status_is_retried_then_succeeds(self, mock_sleep, status):
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.side_effect = [_mock_5xx_response(status), _mock_200_response()]
            result = client.ingest_trace(RunTrace(agent_name="test"))

        assert result["status"] == "ok"
        assert mock_req.call_count == 2
        mock_sleep.assert_called_once()

    @patch("time.sleep")
    def test_500_is_retried_on_the_idempotent_trace_post(self, mock_sleep):
        """A trace carries a client-generated id, so replaying it cannot double-write."""
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.side_effect = [_mock_500_response(), _mock_200_response()]
            result = client.ingest_trace(RunTrace(agent_name="test"))

        assert result["status"] == "ok"
        assert mock_req.call_count == 2

    @patch("time.sleep")
    def test_500_is_not_retried_on_a_route_that_did_not_opt_in(self, mock_sleep):
        """500 means the application ran and failed part-way. Only call sites that
        are safe to replay pass idempotent=True; the rest must fail fast rather
        than risk a double write."""
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_500_response()
            with pytest.raises(httpx.HTTPStatusError):
                client._request_with_retry("POST", "/api/v1/evaluators", json={})

        assert mock_req.call_count == 1
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_503_uses_the_same_backoff_ladder_as_429(self, mock_sleep):
        """One ladder — 1s, 2s, 4s — so there is a single scheme to reason about."""
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            with pytest.raises(httpx.HTTPStatusError):
                client.ingest_trace(RunTrace(agent_name="test"))

        assert mock_req.call_count == _MAX_RETRIES + 1
        assert [c[0][0] for c in mock_sleep.call_args_list] == [1.0, 2.0, 4.0]

    @patch("time.sleep")
    def test_503_honours_retry_after_when_the_proxy_sends_one(self, mock_sleep):
        client = _make_client()
        resp = _mock_5xx_response(503)
        resp.headers["Retry-After"] = "7"

        with patch.object(client._http, "request") as mock_req:
            mock_req.side_effect = [resp, _mock_200_response()]
            client.ingest_trace(RunTrace(agent_name="test"))

        assert mock_sleep.call_args[0][0] >= 7.0

    @patch("time.sleep")
    def test_exhausted_retries_raise_the_enriched_api_error(self, mock_sleep):
        """flush() decides preserve-vs-clear off `status_code`, so the error that
        escapes the retry loop must carry the server's real status."""
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            with pytest.raises(DecimalAPIError) as ei:
                client.ingest_trace(RunTrace(agent_name="test"))

        assert ei.value.status_code == 503


class TestFlushBufferPreservation:
    """Test that flush() preserves the buffer on rate limit errors."""

    @patch("time.sleep")
    def test_flush_preserves_buffer_on_rate_limit(self, mock_sleep):
        """Buffer is NOT cleared when flush encounters a rate limit."""
        client = _make_client()
        client._trace_buffer = [
            RunTrace(agent_name="a"),
            RunTrace(agent_name="b"),
        ]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_429_response()
            client.flush()

        # Buffer should still contain the traces
        assert len(client._trace_buffer) == 2

    def test_flush_clears_buffer_on_success(self):
        """Buffer is cleared when flush succeeds."""
        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_200_response()
            client.flush()

        assert len(client._trace_buffer) == 0

    @patch("time.sleep")
    def test_flush_preserves_buffer_when_503_retries_are_exhausted(self, mock_sleep):
        """The bug: a 503 that outlasted the retries used to CLEAR the buffer,
        so up to a full 50-trace batch was destroyed by a fault that was over in
        seconds."""
        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a"), RunTrace(agent_name="b")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            client.flush()  # must not raise

        assert len(client._trace_buffer) == 2, "a 503 must never destroy traces"

    @patch("time.sleep")
    def test_preserved_traces_are_sent_by_the_next_flush(self, mock_sleep):
        """Preserving is only worth anything if the retry actually lands."""
        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            client.flush()
        assert len(client._trace_buffer) == 1

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_200_response()
            client.flush()
        assert len(client._trace_buffer) == 0

    @patch("time.sleep")
    def test_flush_preserves_buffer_on_a_transport_error(self, mock_sleep):
        """A request that never reached the server says nothing about the payload."""
        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.side_effect = httpx.ConnectError("connection refused")
            client.flush()

        assert len(client._trace_buffer) == 1

    @patch("time.sleep")
    def test_flush_clears_buffer_on_4xx(self, mock_sleep):
        """The server has judged these bytes and will judge them the same way
        forever — holding them would wedge every later trace behind them."""
        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_400_response()
            client.flush()

        assert len(client._trace_buffer) == 0
        assert mock_req.call_count == 1, "a 4xx must not be retried either"

    @patch("time.sleep")
    def test_flush_clears_buffer_on_a_serialization_failure(self, mock_sleep):
        """A payload that cannot even be built is permanently malformed."""
        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]

        with patch.object(client, "ingest_traces_batch") as mock_batch:
            mock_batch.side_effect = TypeError("not JSON serializable")
            client.flush()

        assert len(client._trace_buffer) == 0

    @patch("time.sleep")
    def test_a_permanent_drop_is_recorded_for_export_status(self, mock_sleep):
        """A silently dropped batch is the same bug in a different place — the
        loss has to reach the surface a health check reads."""
        import decimalai
        from decimalai import _config

        _config._sender._last_send_error = None
        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_400_response()
            client.flush()

        err = decimalai.last_send_error()
        assert isinstance(err, DecimalAPIError)
        assert err.status_code == 400
        assert "manifest_id is required" in str(err)
        assert decimalai.export_status().last_error is not None

    @patch("time.sleep")
    def test_a_preserved_buffer_is_capped_at_the_auto_flush_threshold(self, mock_sleep):
        """An unbounded buffer is its own bug: a backend that stays down would
        otherwise grow the buffer by one trace per flush forever."""
        client = _make_client()
        client._trace_buffer = [
            RunTrace(agent_name=f"t{i}") for i in range(_AUTO_FLUSH_THRESHOLD + 5)
        ]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            client.flush()

        assert len(client._trace_buffer) == _AUTO_FLUSH_THRESHOLD
        # The NEWEST are kept: during an outage the traces someone goes looking
        # for are the ones next to the symptom.
        assert client._trace_buffer[0].agent_name == "t5"
        assert client._trace_buffer[-1].agent_name == f"t{_AUTO_FLUSH_THRESHOLD + 4}"

    @patch("time.sleep")
    def test_dropping_from_a_full_buffer_says_so(self, mock_sleep, caplog):
        """A dropped trace never comes back, so it must not be silent."""
        import logging

        client = _make_client()
        client._trace_buffer = [
            RunTrace(agent_name=f"t{i}") for i in range(_AUTO_FLUSH_THRESHOLD + 3)
        ]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            with caplog.at_level(logging.WARNING, logger="decimalai"):
                client.flush()

        assert any(
            "DROPPED" in r.getMessage() and "3 trace(s)" in r.getMessage()
            for r in caplog.records
        ), caplog.text

    @patch("time.sleep")
    def test_the_buffer_stays_bounded_for_the_whole_outage(self, mock_sleep, caplog):
        """The cooldown is the only window in which buffer_trace can push past
        the cap — so the cap has to hold there too, or a busy agent grows the
        buffer without limit for as long as the backend is down."""
        import logging

        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            with caplog.at_level(logging.WARNING, logger="decimalai"):
                for i in range(_AUTO_FLUSH_THRESHOLD * 4):
                    client.buffer_trace(RunTrace(agent_name=f"t{i}"))

        assert len(client._trace_buffer) == _AUTO_FLUSH_THRESHOLD
        assert client._dropped_while_buffer_full == _AUTO_FLUSH_THRESHOLD * 3
        # One retry ladder for the whole outage, not one per trace.
        assert mock_req.call_count == _MAX_RETRIES + 1
        # And one warning line, not 150.
        assert len([r for r in caplog.records if "DROPPED" in r.getMessage()]) <= 1

    @patch("time.sleep")
    def test_recovery_reports_what_the_cap_destroyed(self, mock_sleep, caplog):
        """The recovery log is the last chance to say how much never made it."""
        import logging

        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]
        client._dropped_while_buffer_full = 12

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_200_response()
            with caplog.at_level(logging.WARNING, logger="decimalai"):
                client.flush()

        assert any("12 trace(s) were dropped" in r.getMessage() for r in caplog.records)
        assert client._dropped_while_buffer_full == 0

    @patch("time.sleep")
    def test_auto_flush_backs_off_after_a_preserved_failure(self, mock_sleep):
        """A preserved buffer sits AT the auto-flush threshold, so without a
        cooldown every subsequent trace would drag the caller's thread through
        another full retry ladder — a backend outage would stall the agent."""
        client = _make_client()
        client._trace_buffer = [
            RunTrace(agent_name=f"t{i}") for i in range(_AUTO_FLUSH_THRESHOLD)
        ]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            client.flush()
            calls_after_first_flush = mock_req.call_count

            client.buffer_trace(RunTrace(agent_name="next"))
            assert mock_req.call_count == calls_after_first_flush, (
                "buffer_trace must not re-flush while the cooldown is live"
            )

            # An explicit flush() is the caller asking — it always tries.
            client.flush()
            assert mock_req.call_count > calls_after_first_flush

        # Cooldown expired → the auto-flush path is live again.
        client._flush_cooldown_until = 0.0
        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_200_response()
            client.buffer_trace(RunTrace(agent_name="later"))
            assert mock_req.call_count == 1
        assert len(client._trace_buffer) == 0


class TestLastChanceFlush:
    """close() and interpreter exit are the LAST flush — say so.

    "Preserving N traces for the next flush" is the right message while there
    is a next flush. At shutdown there isn't one, and printing it anyway would
    trade a silent loss for a reassuring one.
    """

    @patch("time.sleep")
    def test_close_reports_and_drops_what_it_could_not_deliver(self, mock_sleep, caplog):
        import logging

        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a"), RunTrace(agent_name="b")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_5xx_response(503)
            with caplog.at_level(logging.WARNING, logger="decimalai"):
                client.close()

        assert len(client._trace_buffer) == 0
        assert any(
            "never reached the platform" in r.getMessage() for r in caplog.records
        ), caplog.text

    def test_close_is_silent_when_everything_was_delivered(self, caplog):
        import logging

        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_200_response()
            with caplog.at_level(logging.WARNING, logger="decimalai"):
                client.close()

        assert not [
            r for r in caplog.records if "never reached the platform" in r.getMessage()
        ]

    def test_atexit_handler_reports_undelivered_traces(self):
        """The atexit path is where a plain script's traces actually die."""
        from decimalai import _atexit_flush
        from decimalai import _config as _cfg

        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]
        original = _cfg._client
        _cfg._client = client
        try:
            with patch.object(client, "flush"):  # flush "fails", buffer untouched
                _atexit_flush()
        finally:
            _cfg._client = original

        assert len(client._trace_buffer) == 0


class TestQuotaExceededIsTerminal:
    """A plan quota 429 must fail fast — retrying it cannot succeed.

    A quota and a rate limit share the 429 status code but nothing else: a rate limit
    clears in seconds, a quota does not clear until the billing period rolls over.
    """

    @staticmethod
    def _mock_quota_429(dimension: str = "storage_bytes") -> httpx.Response:
        request = httpx.Request("POST", "http://localhost:9999/api/v1/traces")
        return httpx.Response(
            429,
            headers={
                "X-Quota-Exceeded": dimension,
                "X-RateLimit-Plan": "free",
                "Content-Type": "application/json",
            },
            json={"detail": {
                "error": "limit_exceeded",
                "feature": dimension,
                "plan": "free",
                "resets_in_seconds": 1_900_000,
            }},
            request=request,
        )

    @patch("time.sleep")
    def test_raises_immediately_without_a_single_retry(self, mock_sleep):
        client = _make_client()
        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = self._mock_quota_429()
            with pytest.raises(DecimalQuotaExceededError):
                client.ingest_trace(RunTrace(agent_name="test"))
        assert mock_req.call_count == 1, "a quota 429 must not be retried"
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    def test_carries_the_dimension_and_reset_for_the_caller(self, mock_sleep):
        client = _make_client()
        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = self._mock_quota_429("traces")
            with pytest.raises(DecimalQuotaExceededError) as ei:
                client.ingest_trace(RunTrace(agent_name="test"))
        assert ei.value.dimension == "traces"
        assert ei.value.resets_in_seconds == 1_900_000
        assert ei.value.plan == "free"
        assert "traces" in str(ei.value)

    @patch("time.sleep")
    def test_is_not_caught_by_except_rate_limit(self, mock_sleep):
        """Code that catches a rate limit in order to sleep and retry must NOT
        swallow a quota — that would reintroduce the drop-the-payload path."""
        assert not issubclass(DecimalQuotaExceededError, DecimalRateLimitError)

    @patch("time.sleep")
    def test_a_malformed_body_still_raises_quota_not_rate_limit(self, mock_sleep):
        """The header is the signal; a body we cannot parse must not downgrade it back
        into a retryable rate limit."""
        client = _make_client()
        request = httpx.Request("POST", "http://localhost:9999/api/v1/traces")
        broken = httpx.Response(
            429,
            headers={"X-Quota-Exceeded": "traces", "Content-Type": "application/json"},
            content=b"not json at all",
            request=request,
        )
        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = broken
            with pytest.raises(DecimalQuotaExceededError) as ei:
                client.ingest_trace(RunTrace(agent_name="test"))
        assert ei.value.dimension == "traces"
        assert mock_req.call_count == 1

    @patch("time.sleep")
    def test_a_plain_rate_limit_429_still_retries(self, mock_sleep):
        """Guard the other direction: no X-Quota-Exceeded header means the ordinary
        retry path is untouched."""
        client = _make_client()
        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_429_response()
            with pytest.raises(DecimalRateLimitError):
                client.ingest_trace(RunTrace(agent_name="test"))
        assert mock_req.call_count == _MAX_RETRIES + 1
