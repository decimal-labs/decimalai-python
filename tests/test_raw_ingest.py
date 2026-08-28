"""Tests for the raw/direct ingestion interface."""

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Reset global SDK state before each test."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    yield


class TestIngestRaw:
    """Tests for decimalai.ingest_raw()."""

    def test_ingest_raw_sends_payload(self):
        """ingest_raw should POST the dict directly to /api/v1/traces."""
        import decimalai
        import decimalai._config as cfg

        payload = {
            "agent_name": "test-agent",
            "status": "success",
            "started_at": "2025-01-01T00:00:00Z",
            "ended_at": "2025-01-01T00:00:01Z",
            "llm_calls": [
                {"model_name": "gpt-4o", "input_tokens": 10, "output_tokens": 20}
            ],
        }

        cfg._client.ingest_raw_trace.return_value = {"id": "trace-123"}
        result = decimalai.ingest_raw(payload)

        cfg._client.ingest_raw_trace.assert_called_once_with(payload)
        assert result == {"id": "trace-123"}

    def test_ingest_raw_without_init_raises(self):
        """ingest_raw without init should raise DecimalConfigError."""
        import decimalai._config as cfg
        from decimalai._config import DecimalConfigError

        cfg._config = None
        cfg._client = None

        import decimalai

        with pytest.raises(DecimalConfigError):
            decimalai.ingest_raw({"agent_name": "test"})

    def test_ingest_raw_minimal_payload(self):
        """ingest_raw should accept a minimal dict."""
        import decimalai
        import decimalai._config as cfg

        cfg._client.ingest_raw_trace.return_value = {"id": "trace-456"}

        payload = {"agent_name": "minimal-agent"}
        result = decimalai.ingest_raw(payload)

        cfg._client.ingest_raw_trace.assert_called_once_with(payload)
        assert result["id"] == "trace-456"


class TestIngestRawBatch:
    """Tests for decimalai.ingest_raw_batch()."""

    def test_ingest_raw_batch_sends_payloads(self):
        """ingest_raw_batch should POST a list of dicts."""
        import decimalai
        import decimalai._config as cfg

        payloads = [
            {"agent_name": "agent-a", "status": "success"},
            {"agent_name": "agent-b", "status": "error"},
        ]

        cfg._client.ingest_raw_traces_batch.return_value = {
            "ingested": 2
        }

        result = decimalai.ingest_raw_batch(payloads)

        cfg._client.ingest_raw_traces_batch.assert_called_once_with(payloads)
        assert result["ingested"] == 2

    def test_ingest_raw_batch_empty_list(self):
        """ingest_raw_batch should accept an empty list."""
        import decimalai
        import decimalai._config as cfg

        cfg._client.ingest_raw_traces_batch.return_value = {"ingested": 0}

        result = decimalai.ingest_raw_batch([])
        cfg._client.ingest_raw_traces_batch.assert_called_once_with([])


class TestClientRawMethods:
    """Tests for DecimalAIClient.ingest_raw_trace() and ingest_raw_traces_batch()."""

    def test_client_ingest_raw_trace(self):
        """Client.ingest_raw_trace should POST to /api/v1/traces."""
        from decimalai._client import DecimalAIClient
        from unittest.mock import patch

        client = DecimalAIClient(
            api_key="dai_sk_test", base_url="http://localhost:8000"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "trace-789"}

        with patch.object(client, "_request_with_retry", return_value=mock_response) as mock_req:
            result = client.ingest_raw_trace({"agent_name": "raw-agent"})

            # idempotent=True: a raw trace carries a client-generated id, so a
            # retry after a 5xx cannot double-write — that is what buys it the
            # 500 retry the other POST routes deliberately do not get.
            mock_req.assert_called_once_with(
                "POST", "/api/v1/traces", json={"agent_name": "raw-agent"},
                idempotent=True,
            )
            assert result == {"id": "trace-789"}

    def test_client_ingest_raw_traces_batch(self):
        """Client.ingest_raw_traces_batch should POST to /api/v1/traces/batch."""
        from decimalai._client import DecimalAIClient
        from unittest.mock import patch

        client = DecimalAIClient(
            api_key="dai_sk_test", base_url="http://localhost:8000"
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ingested": 2}

        payloads = [{"agent_name": "a"}, {"agent_name": "b"}]

        with patch.object(client, "_request_with_retry", return_value=mock_response) as mock_req:
            result = client.ingest_raw_traces_batch(payloads)

            mock_req.assert_called_once_with(
                "POST", "/api/v1/traces/batch", json=payloads, idempotent=True
            )
            assert result == {"ingested": 2}

    def test_client_ingest_raw_trace_scrubs_surrogates(self):
        """Raw ingest must scrub lone surrogates before the request is
        built. Raw payloads come from custom pipelines / non-Python sources —
        exactly where un-encodable text is most likely — so without scrubbing
        httpx's JSON encoder raises UnicodeEncodeError client-side."""
        from decimalai._client import DecimalAIClient
        from unittest.mock import patch

        client = DecimalAIClient(
            api_key="dai_sk_test", base_url="http://localhost:8000"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "t"}

        with patch.object(client, "_request_with_retry", return_value=mock_response) as mock_req:
            client.ingest_raw_trace({"agent_name": "ok\ud800bad"})

        sent = mock_req.call_args.kwargs["json"]
        assert "\ud800" not in sent["agent_name"]

    def test_client_ingest_raw_traces_batch_scrubs_surrogates(self):
        """Batch raw ingest scrubs every payload in the list, not just the first."""
        from decimalai._client import DecimalAIClient
        from unittest.mock import patch

        client = DecimalAIClient(
            api_key="dai_sk_test", base_url="http://localhost:8000"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ingested": 1}

        with patch.object(client, "_request_with_retry", return_value=mock_response) as mock_req:
            client.ingest_raw_traces_batch([{"agent_name": "x\ud800y"}])

        sent = mock_req.call_args.kwargs["json"]
        assert "\ud800" not in sent[0]["agent_name"]
