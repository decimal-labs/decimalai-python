"""`decimalai init` survives a Cloud Run admission abort on its verify call.

Measured on production 2026-09-03: a single-instance backend at its concurrency limit answers
HTTP 429 ("no available instance") in ~0 ms with no Retry-After to 84-92% of requests. `init`
made one `/api/v1/auth/verify` call and exited on any non-2xx, so the quickstart failed outright
for a customer who ran it while the platform was busy. One more try a moment later is a different
admission decision; a 401 is a real answer and stays one.
"""

from unittest.mock import MagicMock, patch

import httpx
from click.testing import CliRunner

from decimalai.cli.main import _ADMISSION_RETRY_DELAYS_S, cli


def _resp(status: int, body: dict | None = None, headers: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.headers = httpx.Headers(headers or {})
    r.json.return_value = body or {}
    if status >= 400:
        r.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status}", request=MagicMock(), response=MagicMock(status_code=status),
        )
    else:
        r.raise_for_status.return_value = None
    return r


def _init(client):
    with patch("decimalai._client.DecimalAIClient", return_value=client), \
         patch("time.sleep") as sleep:
        result = CliRunner().invoke(cli, [
            "init", "--api-key", "dai_sk_x", "--base-url", "http://localhost:9",
            "--no-test-trace",
        ])
    return result, sleep


def test_one_admission_abort_is_retried_and_init_proceeds():
    client = MagicMock()
    client._http.get.side_effect = [_resp(429), _resp(200, {"workspace_id": "ws-1", "scope": "workspace"})]
    result, sleep = _init(client)
    assert "Connected to workspace: ws-1" in result.output
    assert client._http.get.call_count == 2
    sleep.assert_called_once_with(_ADMISSION_RETRY_DELAYS_S[0])


def test_a_short_retry_after_is_honoured_over_the_schedule():
    client = MagicMock()
    client._http.get.side_effect = [_resp(503, headers={"Retry-After": "2"}), _resp(200, {"scope": "org"})]
    result, sleep = _init(client)
    assert "Connected" in result.output
    sleep.assert_called_once_with(2.0)


def test_an_abort_that_outlasts_the_schedule_still_fails_with_the_status():
    client = MagicMock()
    client._http.get.return_value = _resp(429)
    result, sleep = _init(client)
    assert result.exit_code == 1
    assert "HTTP 429" in result.output
    assert client._http.get.call_count == len(_ADMISSION_RETRY_DELAYS_S) + 1


def test_a_rejected_key_is_not_retried():
    client = MagicMock()
    client._http.get.return_value = _resp(401)
    result, sleep = _init(client)
    assert result.exit_code == 1
    assert "Invalid API key" in result.output
    assert client._http.get.call_count == 1
    sleep.assert_not_called()
