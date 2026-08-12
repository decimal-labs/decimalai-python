"""Tests for the DecimalAI HTTP client.

Uses httpx mocking to test client behavior without a real server.
"""

import json
from unittest.mock import patch, MagicMock
from uuid import uuid4

import pytest
import httpx

from decimalai._client import DecimalAIClient
from decimalai.schema.trace import RunTrace, TraceSpan, LlmCallRecord
from decimalai.schema.common import SpanType, Status


class TestDecimalAIClient:
    """Test the HTTP client in isolation."""

    def _make_client(self):
        return DecimalAIClient(
            api_key="dai_sk_test",
            project="test-project",
            base_url="http://localhost:8000",
        )

    def test_client_initialization(self):
        client = self._make_client()
        assert client.api_key == "dai_sk_test"
        assert client.project == "test-project"
        assert client.base_url == "http://localhost:8000"

    def test_config_headers(self):
        client = self._make_client()
        headers = dict(client._http.headers)
        assert headers["authorization"] == "Bearer dai_sk_test"

    def test_project_is_not_sent_as_a_header(self):
        """`project` must not reach the wire — the platform never read it.

        The SDK used to send `X-Decimal-Project`, which the backend ignores
        entirely (a trace's project_id comes only from a project-scoped API
        key), so passing project= silently grouped nothing."""
        client = self._make_client()
        assert client.project == "test-project"  # still recorded locally
        assert "x-decimal-project" not in dict(client._http.headers)

    def test_buffer_under_threshold(self):
        client = self._make_client()

        # Buffer a few traces — well under the 50 threshold, no flush
        trace1 = RunTrace(agent_name="a")
        trace2 = RunTrace(agent_name="b")
        with patch.object(client, "ingest_traces_batch") as mock_batch:
            client.buffer_trace(trace1)
            client.buffer_trace(trace2)
            assert len(client._trace_buffer) == 2
            mock_batch.assert_not_called()

    def test_buffer_auto_flush_at_threshold(self):
        client = self._make_client()

        with patch.object(client, "ingest_traces_batch") as mock_batch:
            mock_batch.return_value = {"status": "ok"}
            # Buffer exactly 50 traces to hit the threshold
            for i in range(50):
                client.buffer_trace(RunTrace(agent_name=f"agent-{i}"))
            # Should have auto-flushed at 50
            mock_batch.assert_called_once()
            assert len(client._trace_buffer) == 0

    def test_flush_empty_buffer_is_noop(self):
        client = self._make_client()
        with patch.object(client, "ingest_traces_batch") as mock_batch:
            client.flush()
            mock_batch.assert_not_called()

    def test_context_manager_flushes(self):
        client = self._make_client()
        client._trace_buffer.append(RunTrace(agent_name="test"))
        with patch.object(client, "ingest_traces_batch") as mock_batch:
            mock_batch.return_value = {"status": "ok"}
            client.close()
            mock_batch.assert_called_once()

    def test_trace_serialization_for_api(self):
        """Verify that trace.model_dump(mode='json') produces valid API payloads."""
        trace = RunTrace(
            project="test",
            agent_name="agent",
            spans=[
                TraceSpan(name="root", span_type=SpanType.AGENT),
                TraceSpan(name="llm", span_type=SpanType.LLM),
            ],
            llm_calls=[
                LlmCallRecord(
                    model_name="gpt-4o",
                    rendered_input=[{"role": "user", "content": "hi"}],
                    output={"content": "hello"},
                )
            ],
        )
        payload = trace.model_dump(mode="json")

        # All UUIDs should be strings
        assert isinstance(payload["id"], str)
        assert all(isinstance(s["id"], str) for s in payload["spans"])
        assert all(isinstance(c["id"], str) for c in payload["llm_calls"])

        # Should be JSON-serializable
        json_str = json.dumps(payload)
        assert len(json_str) > 0

        # Roundtrip should work
        restored = RunTrace.model_validate(json.loads(json_str))
        assert restored.agent_name == "agent"
        assert len(restored.spans) == 2


class TestClientAuth:
    """Test auth-related client behavior."""

    def test_verify_auth_sends_correct_headers(self):
        client = DecimalAIClient(
            api_key="dai_sk_my_key",
            project="my-project",
            base_url="http://localhost:8000",
        )
        # The HTTP client stores the headers
        assert "Authorization" in client._http.headers
        assert client._http.headers["Authorization"] == "Bearer dai_sk_my_key"
        client.close()


class TestImpactReport:
    """impact_report() hits the aggregate endpoint with the right params.

    Before this method existed, API consumers guessed wrong paths
    for "the impact report" (the data only existed composed client-side in
    the dashboard). The aggregate endpoint is GET
    /api/v1/agents/{name}/impact-report — locked here.
    """

    def _make_client(self):
        return DecimalAIClient(
            api_key="dai_sk_test",
            project="test-project",
            base_url="http://localhost:8000",
        )

    def test_impact_report_default_transition(self):
        client = self._make_client()
        fake = MagicMock()
        fake.json.return_value = {"status": "ok", "affected_trace_count": 6}
        with patch.object(client._http, "get", return_value=fake) as mock_get:
            result = client.impact_report("my-agent")
        mock_get.assert_called_once_with(
            "/api/v1/agents/my-agent/impact-report", params={}
        )
        assert result["affected_trace_count"] == 6
        client.close()

    def test_impact_report_pinned_manifests(self):
        client = self._make_client()
        fake = MagicMock()
        fake.json.return_value = {"status": "ok"}
        with patch.object(client._http, "get", return_value=fake) as mock_get:
            client.impact_report(
                "my-agent",
                manifest_id="m2",
                baseline_manifest_id="m1",
            )
        mock_get.assert_called_once_with(
            "/api/v1/agents/my-agent/impact-report",
            params={"manifest_id": "m2", "baseline_manifest_id": "m1"},
        )
        client.close()


class TestRepair:
    """The repair surface wraps the platform /repair endpoints."""

    def _make_client(self):
        return DecimalAIClient(
            api_key="dai_sk_test", project="test-project", base_url="http://localhost:8000"
        )

    def test_repair_preview_posts_correct_payload(self):
        client = self._make_client()
        fake = MagicMock()
        fake.json.return_value = {"rules": [], "previews": [], "total_eligible": 3}
        with patch.object(client._http, "post", return_value=fake) as mock_post:
            result = client.repair_preview("m1", "m2", sample_size=7)
        mock_post.assert_called_once_with(
            "/api/v1/repair/preview",
            json={"old_manifest_id": "m1", "new_manifest_id": "m2", "sample_size": 7},
        )
        assert result["total_eligible"] == 3
        client.close()

    def test_repair_apply_all_uses_apply_route(self):
        client = self._make_client()
        fake = MagicMock()
        fake.json.return_value = {"batch_id": "b1", "status": "completed"}
        with patch.object(client._http, "post", return_value=fake) as mock_post:
            client.repair_apply("m1", "m2")
        mock_post.assert_called_once_with(
            "/api/v1/repair/apply",
            json={"old_manifest_id": "m1", "new_manifest_id": "m2"},
        )
        client.close()

    def test_repair_apply_subset_uses_selective_route(self):
        client = self._make_client()
        fake = MagicMock()
        fake.json.return_value = {"batch_id": "b1", "status": "completed", "rules_applied": 2}
        with patch.object(client._http, "post", return_value=fake) as mock_post:
            client.repair_apply("m1", "m2", approved_rule_indices=[0, 2])
        mock_post.assert_called_once_with(
            "/api/v1/repair/apply-selective",
            json={
                "old_manifest_id": "m1",
                "new_manifest_id": "m2",
                "approved_rule_indices": [0, 2],
            },
        )
        client.close()

    def test_get_repair_batch(self):
        client = self._make_client()
        fake = MagicMock()
        fake.json.return_value = {"id": "b1", "status": "completed"}
        with patch.object(client._http, "get", return_value=fake) as mock_get:
            client.get_repair_batch("b1")
        mock_get.assert_called_once_with("/api/v1/repair/b1")
        client.close()
