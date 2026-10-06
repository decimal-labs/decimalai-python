"""Acceptance checks must survive platform admission without replaying models."""
import json
from unittest.mock import patch

import httpx
import pytest

from decimalai._client import (
    DecimalAIClient, DecimalAPIError, DecimalQuotaExceededError, DecimalRateLimitError,
    _MAX_RETRIES, _MAX_RETRY_WAIT,
)
from decimalai.agent_checks import _confirm, configuration_snapshot, error_diagnosis

BASE_URL = "http://127.0.0.1:8001"


def client_for(handler):
    client = DecimalAIClient(api_key="fixture", base_url=BASE_URL)
    client._http.close()
    client._http = httpx.Client(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    return client


def test_eight_uncached_skill_reads_recover_from_free_tier_burst():
    # The published starter has eight bodies. Auth + prompt + list exhaust a
    # burst of ten at its eighth body; one Retry-After wait replenishes a token.
    state = {"tokens": 10, "sleeps": [], "requests": []}
    def handler(request):
        state["requests"].append(str(request.url))
        if state["tokens"] <= 0:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"detail": "Rate limit exceeded"})
        state["tokens"] -= 1
        path = request.url.path
        if path.endswith("/prompt"):
            return httpx.Response(200, json={"system_prompt": "Draft replies", "content_hash": "prompt-v1"})
        if path.endswith("/skills"):
            return httpx.Response(200, json={"skills": [{"skill_name": f"skill-{i}"} for i in range(8)]})
        if path.endswith("/body"):
            name = path.split("/")[-2]
            return httpx.Response(200, json={"body": f"# {name}\nVerified instruction.", "version": 1, "content_hash": name})
        return httpx.Response(200, json={})
    def sleep(delay):
        state["sleeps"].append(delay)
        state["tokens"] += delay
    client = client_for(handler)
    try:
        with patch("decimalai._client.time.sleep", side_effect=sleep):
            client.verify_auth()
            snapshot = configuration_snapshot(client, "support")
        assert len(snapshot["skills"]) == 8
        assert snapshot["skills"][-1]["body"] == "# skill-7\nVerified instruction."
        assert state["sleeps"] == [1.0]
        assert len(state["requests"]) == 12
        assert state["requests"][-1] == state["requests"][-2]
        assert "agent_name=support" in state["requests"][-1]
    finally:
        client.close()


def test_confirmation_recovers_same_immutable_result_and_polls_without_agent_calls():
    attempts = []
    def handler(request):
        body = json.loads(request.content) if request.content else None
        attempts.append((request.method, request.url.path, body))
        if len(attempts) in (1, 3):
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, json={"status": "pending_trace" if request.method == "PUT" else "passed"})
    client = client_for(handler)
    receipt = {"agent_name": "support", "check_id": "same-id", "dashboard_url": "http://localhost/setup",
               "remote_result": {"suite": "support-draft/v1", "cases": []}}
    try:
        with patch("decimalai._client.time.sleep"):
            _confirm(client, receipt, wait_seconds=5)
        assert receipt["passed"] is True
        assert attempts[0] == attempts[1]
        assert attempts[2] == attempts[3]
        assert [a[0] for a in attempts] == ["PUT", "PUT", "GET", "GET"]
        assert all("/setup/checks/same-id" in a[1] for a in attempts)
    finally:
        client.close()


@pytest.mark.parametrize("headers, error", [
    ({"Retry-After": "3600"}, DecimalRateLimitError),
    ({"Retry-After": "1", "X-Quota-Exceeded": "traces"}, DecimalQuotaExceededError),
])
def test_long_wait_and_plan_quota_fail_without_retry_or_sleep(headers, error):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(429, headers=headers, json={"detail": {"plan": "free"}})
    client = client_for(handler)
    try:
        with patch("decimalai._client.time.sleep") as sleep, pytest.raises(error):
            client.verify_auth()
        assert len(requests) == 1
        sleep.assert_not_called()
    finally:
        client.close()


def test_persistent_throttling_has_finite_attempt_and_wait_budgets():
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "12"})
    client = client_for(handler)
    try:
        with patch("decimalai._client.time.sleep") as sleep, pytest.raises(DecimalRateLimitError):
            client.verify_auth()
        assert len(requests) <= _MAX_RETRIES + 1
        assert sum(c.args[0] for c in sleep.call_args_list) <= _MAX_RETRY_WAIT
        assert [c.args[0] for c in sleep.call_args_list] == [12.0, 12.0]
    finally:
        client.close()


@pytest.mark.parametrize("exc, prefix", [
    (DecimalRateLimitError(1), "decimalai_rate_limit:"),
    (DecimalQuotaExceededError(dimension="traces"), "decimalai_quota:"),
    (RuntimeError("429 RESOURCE_EXHAUSTED"), "provider_quota:"),
    (RuntimeError("429 too many requests"), "provider_rate_limit:"),
])
def test_typed_platform_errors_and_provider_failures_have_distinct_next_steps(exc, prefix):
    assert error_diagnosis(exc, base_url=BASE_URL).startswith(prefix)


@pytest.mark.parametrize("url, headers, prefix", [
    (BASE_URL + "/api/v1/skills/example/body", {}, "decimalai_rate_limit:"),
    (BASE_URL + "/api/v1/traces", {"X-Quota-Exceeded": "traces"}, "decimalai_quota:"),
    ("https://api.openai.com/v1/responses", {}, "provider_rate_limit:"),
])
def test_http_error_origin_separates_backend_admission_from_provider(url, headers, prefix):
    response = httpx.Response(429, headers=headers, request=httpx.Request("GET", url),
                              json={"detail": "fixture-secret-must-not-escape"})
    exc = httpx.HTTPStatusError("429 fixture-secret-must-not-escape", request=response.request, response=response)
    diagnosis = error_diagnosis(exc, base_url=BASE_URL)
    assert diagnosis.startswith(prefix)
    assert "fixture-secret" not in diagnosis


def test_typed_backend_error_stays_backend_even_with_quota_text():
    response = httpx.Response(503, request=httpx.Request("GET", BASE_URL), json={"detail": "quota backend unavailable"})
    assert error_diagnosis(DecimalAPIError(response)).startswith("decimalai_backend:")
