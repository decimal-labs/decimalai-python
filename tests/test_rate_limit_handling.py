"""Tests for SDK rate limit handling — retry, backoff, buffer preservation."""

import time
from unittest.mock import patch, MagicMock, PropertyMock

import httpx
import pytest

from decimalai._client import (
    DecimalAIClient,
    DecimalQuotaExceededError,
    DecimalRateLimitError,
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
    def test_non_429_errors_not_retried(self, mock_sleep):
        """Non-429 errors (e.g. 500) raise immediately without retry."""
        client = _make_client()

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_500_response()
            trace = RunTrace(agent_name="test")

            with pytest.raises(httpx.HTTPStatusError):
                client.ingest_trace(trace)

        assert mock_req.call_count == 1  # No retry
        mock_sleep.assert_not_called()


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

    def test_flush_clears_buffer_on_other_errors(self):
        """Buffer is cleared on non-429 errors (data may be invalid)."""
        client = _make_client()
        client._trace_buffer = [RunTrace(agent_name="a")]

        with patch.object(client._http, "request") as mock_req:
            mock_req.return_value = _mock_500_response()
            client.flush()

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
