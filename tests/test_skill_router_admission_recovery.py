"""A platform burst must not silently discard routed skill instructions."""
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter, SkillRouterError, _ADMISSION_MAX_WAIT_S, _HotPathBreakers


@pytest.fixture(autouse=True)
def _fresh_breaker(monkeypatch):
    """Each test drives one request against its own breaker.

    Since 2026-10-14 a 429 or 5xx on a hot path (`/route`, `/menu`, `/body`)
    counts toward the breaker of the key that got it, as it should. These tests
    send several in a row with one key on purpose, so sharing its breaker opened
    it by the third test and every later one was short-circuited instead of
    exercising admission recovery.
    """
    monkeypatch.setattr(sr, "_hot_path_breakers", _HotPathBreakers())


def response(status, headers=None, body=None):
    return httpx.Response(status, headers=headers, json=body or {"detail": "Rate limit exceeded"})


@pytest.mark.parametrize("method,path", [
    ("GET", "/api/v1/skills/refund-policy/body"),
    ("POST", "/api/v1/skills/route"),
])
def test_explicit_admission_recovers_exact_request_and_workspace_headers(method, path):
    router = SkillRouter(api_key="fixture", base_url="http://127.0.0.1:8001")
    # A custom caller's workspace header must not change on a retry. Routing
    # contains nested request data; neither the payload nor its scope is rebuilt.
    headers = {**router._headers(), "X-Workspace-Id": "workspace-a"}
    payload = {"query": "two billing entries", "context": {"agent_name": "support", "skills": ["policy"]}}
    params = {"agent_name": "support", "max_chars": 8000, "unused": None}
    expected = deepcopy(payload)
    with patch.object(router, "_headers", return_value=headers) as get_headers, \
            patch("decimalai.skill_router.httpx.request", side_effect=[
                response(429, {"Retry-After": "1"}), response(200, body={"body": "# useful policy"}),
            ]) as request, patch("decimalai.skill_router.time.sleep") as sleep:
        result = router._request(method, path, json=payload, params=params)
    assert result == {"body": "# useful policy"}
    assert request.call_count == 2
    assert request.call_args_list[0].args == request.call_args_list[1].args
    first, second = [c.kwargs for c in request.call_args_list]
    assert first["json"] == second["json"] == expected
    assert first["params"] == second["params"] == {"agent_name": "support", "max_chars": 8000}
    # A route also says how long it will wait (X-Decimal-Route-Budget), the same on the retry.
    budget = {sr._ROUTE_BUDGET_HEADER: f"{sr._hot_path_read_budget():g}"} if path == sr._ROUTE_PATH else {}
    assert first["headers"] == second["headers"] == {**headers, **budget}
    get_headers.assert_called_once()
    sleep.assert_called_once_with(1.0)
    assert payload == expected


@pytest.mark.parametrize("headers", [
    {}, {"Retry-After": "30"}, {"Retry-After": "invalid"}, {"Retry-After": "nan"},
    {"Retry-After": "inf"}, {"Retry-After": "0"},
    {"Retry-After": "1", "X-Quota-Exceeded": "llm_calls"},
])
def test_missing_invalid_large_wait_or_quota_stays_fail_fast(headers):
    router = SkillRouter(api_key="fixture")
    with patch("decimalai.skill_router.httpx.request", return_value=response(429, headers)) as request, \
            patch("decimalai.skill_router.time.sleep") as sleep, pytest.raises(SkillRouterError) as caught:
        router._request("POST", "/api/v1/skills/route", json={"query": "q"})
    assert caught.value.status_code == 429
    assert request.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("retry_after,expected_requests,expected_sleeps", [
    ("1", 3, [1.0, 1.0]), ("3", 2, [3.0]),
])
def test_persistent_admission_obeys_attempt_and_wait_caps(retry_after, expected_requests, expected_sleeps):
    router = SkillRouter(api_key="fixture")
    with patch("decimalai.skill_router.httpx.request", return_value=response(429, {"Retry-After": retry_after})) as request, \
            patch("decimalai.skill_router.time.sleep") as sleep, pytest.raises(SkillRouterError):
        router._request("GET", "/api/v1/skills/policy/body")
    assert request.call_count == expected_requests
    assert [c.args[0] for c in sleep.call_args_list] == expected_sleeps
    assert sum(expected_sleeps) <= _ADMISSION_MAX_WAIT_S


def test_elapsed_network_time_prevents_another_admission_wait():
    router = SkillRouter(api_key="fixture")
    # The request itself takes 4 s, so 4 s + a 2 s Retry-After is past the 5 s
    # cap. A clock the request advances, not a fixed list of readings: the
    # breakers read the monotonic clock too.
    clock = {"now": 0.0}

    def slow_429(*args, **kwargs):
        clock["now"] += 4.0
        return response(429, {"Retry-After": "2"})

    with patch("decimalai.skill_router.httpx.request", side_effect=slow_429) as request, \
            patch("decimalai.skill_router.time.monotonic", side_effect=lambda: clock["now"]), \
            patch("decimalai.skill_router.time.sleep") as sleep, pytest.raises(SkillRouterError):
        router._request("GET", "/api/v1/skills/policy/body")
    assert request.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("method,path,status", [
    ("POST", "/api/v1/skills/policy/publish", 429),
    ("PUT", "/api/v1/skills/policy", 429),
    ("POST", "/api/v1/skills/route", 500),
    ("GET", "/api/v1/skills/policy/body", 503),
])
def test_other_writes_and_server_failures_are_not_replayed(method, path, status):
    router = SkillRouter(api_key="fixture")
    with patch("decimalai.skill_router.httpx.request", return_value=response(status, {"Retry-After": "1"})) as request, \
            patch("decimalai.skill_router.time.sleep") as sleep, pytest.raises(SkillRouterError):
        router._request(method, path, json={"unmodified": {"nested": ["value"]}})
    assert request.call_count == 1
    sleep.assert_not_called()


def test_smart_route_returns_recovered_skills_instead_of_empty_degraded_menu():
    router = SkillRouter(api_key="fixture")
    routed = {"skills": [{"name": "billing-policy"}], "prompt_fragment": "# Billing policy", "routing_id": "r1"}
    with patch("decimalai.skill_router.httpx.request", side_effect=[
        response(429, {"Retry-After": "1"}), response(200, body=routed),
    ]), patch("decimalai.skill_router.time.sleep"):
        assert router.smart_route("two billing entries", agent_name="support") == routed


def test_retry_keeps_initial_nested_inputs_and_auth_scope_when_caller_context_changes():
    router = SkillRouter(api_key="fixture")
    headers = {"Authorization": "Bearer original", "X-Workspace-Id": "workspace-a"}
    payload = {"query": "q", "context": {"workspace": "workspace-a", "skills": ["billing"]}}
    params = {"agent_name": "support", "labels": ["original"]}
    attempts = []
    def request(*args, **kwargs):
        attempts.append(deepcopy(kwargs))
        return response(429, {"Retry-After": "1"}) if len(attempts) == 1 else response(200, body={"skills": []})
    def sleep(_delay):
        headers.update({"Authorization": "Bearer changed", "X-Workspace-Id": "workspace-b"})
        payload["context"]["skills"].append("changed")
        params["labels"].append("changed")
    with patch.object(router, "_headers", return_value=headers), \
            patch("decimalai.skill_router.httpx.request", side_effect=request), \
            patch("decimalai.skill_router.time.sleep", side_effect=sleep):
        router._request("POST", "/api/v1/skills/route", json=payload, params=params)
    assert attempts[0]["json"] == attempts[1]["json"] == {
        "query": "q", "context": {"workspace": "workspace-a", "skills": ["billing"]},
    }
    assert attempts[0]["headers"] == attempts[1]["headers"] == {
        "Authorization": "Bearer original", "X-Workspace-Id": "workspace-a",
        sr._ROUTE_BUDGET_HEADER: f"{sr._hot_path_read_budget():g}",
    }
    assert attempts[0]["params"] == attempts[1]["params"] == {"agent_name": "support", "labels": ["original"]}


# ── An evaluation run has time to wait; a customer's turn does not ──

BODY = "/api/v1/skills/refund-policy/body"


def _evaluation_block():
    from decimalai import _evaluation

    return _evaluation.block(_evaluation.Trial(_evaluation.Override.make("support")))


def _limited(times, retry_after="1"):
    return [response(429, {"Retry-After": retry_after}) for _ in range(times)] + [
        response(200, body={"body": "# useful policy"})]


def test_an_evaluation_run_waits_out_a_rate_limit_a_turn_would_give_up_on():
    router = SkillRouter(api_key="fixture", base_url="http://127.0.0.1:8001")
    with _evaluation_block(), patch("decimalai.skill_router.httpx.request", side_effect=_limited(6)) as request, \
            patch("decimalai.skill_router.time.sleep") as sleep:
        assert router._request("GET", BODY) == {"body": "# useful policy"}
    assert request.call_count == 7
    assert sleep.call_count == 6


def test_outside_an_evaluation_a_turn_still_gives_up_after_its_retries():
    router = SkillRouter(api_key="fixture", base_url="http://127.0.0.1:8001")
    with patch("decimalai.skill_router.httpx.request", side_effect=_limited(6)) as request, \
            patch("decimalai.skill_router.time.sleep"), pytest.raises(SkillRouterError):
        router._request("GET", BODY)
    assert request.call_count == sr._ADMISSION_MAX_RETRIES + 1


def test_even_an_evaluation_run_stops_waiting_at_its_cap():
    router = SkillRouter(api_key="fixture", base_url="http://127.0.0.1:8001")
    wait = int(sr._EVALUATION_ADMISSION_MAX_WAIT_S // 3)
    with _evaluation_block(), patch("decimalai.skill_router.httpx.request", side_effect=_limited(9, str(wait))) as request, \
            patch("decimalai.skill_router.time.sleep"), pytest.raises(SkillRouterError):
        router._request("GET", BODY)
    assert request.call_count == 4


def test_a_call_that_asked_not_to_wait_does_not_wait_in_an_evaluation():
    router = SkillRouter(api_key="fixture", base_url="http://127.0.0.1:8001")
    with _evaluation_block(), patch("decimalai.skill_router.httpx.request", side_effect=_limited(1)) as request, \
            patch("decimalai.skill_router.time.sleep") as sleep, pytest.raises(SkillRouterError):
        router._request("GET", BODY, admission_wait_s=0.0)
    assert request.call_count == 1
    sleep.assert_not_called()


def test_a_write_is_still_never_replayed_in_an_evaluation():
    router = SkillRouter(api_key="fixture", base_url="http://127.0.0.1:8001")
    with _evaluation_block(), patch("decimalai.skill_router.httpx.request", side_effect=_limited(1)) as request, \
            patch("decimalai.skill_router.time.sleep"), pytest.raises(SkillRouterError):
        router._request("POST", "/api/v1/skills", json={"name": "x"})
    assert request.call_count == 1
