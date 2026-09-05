"""`decimalai init` survives a Cloud Run admission abort on any of its calls.

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


# ── the scaffold path: `decimalai init <agent>` ──────────────────────────────
#
# The scaffold makes two bare GETs of its own — /api/v1/agents and
# /api/v1/agents/{name}/skills — before it writes agent.py. Each one used to go
# straight to raise_for_status, so one admission abort printed "Server returned
# HTTP 429" and wrote nothing: the fleet's scaffold canary filed exactly that as
# `decimalai_init_wrote_no_agent_file` (14 reds through 2026-09-04).

AGENT = "refund-bot"
_OK_AGENTS = {"agents": [{"agent_name": AGENT}]}
_OK_SKILLS = {"agent_name": AGENT, "skills": []}
_PROMPT = {
    "agent_name": AGENT, "system_prompt": "x", "version_number": 1, "content_hash": "h",
    "provenance": "ui", "version_mode": "latest", "pinned_version_number": None,
}


def _scaffold_client(agents_responses, skills_responses):
    """The two GETs the scaffold path makes, each answered from its own queue —
    a flat side_effect list cannot say WHICH call was aborted."""
    agents_it, skills_it = iter(agents_responses), iter(skills_responses)

    def get(url, **kwargs):
        if url.endswith("/api/v1/agents"):
            return next(agents_it)
        if "/skills" in url:
            return next(skills_it)
        raise AssertionError(f"unexpected GET {url}")

    client = MagicMock()
    client._http.get.side_effect = get
    client.get_agent_prompt.return_value = _PROMPT
    return client


def _scaffold(client, tmp_path):
    out = tmp_path / "agent.py"
    with patch("decimalai._client.DecimalAIClient", return_value=client), \
         patch("time.sleep") as sleep:
        result = CliRunner().invoke(cli, [
            "init", AGENT, "--api-key", "dai_sk_x", "--base-url", "http://localhost:9",
            "--out", str(out),
        ])
    return result, sleep, out


def test_one_abort_on_the_agents_list_still_writes_the_file(tmp_path):
    client = _scaffold_client([_resp(429), _resp(200, _OK_AGENTS)], [_resp(200, _OK_SKILLS)])
    result, sleep, out = _scaffold(client, tmp_path)
    assert result.exit_code == 0, result.output
    assert out.exists()
    sleep.assert_called_once_with(_ADMISSION_RETRY_DELAYS_S[0])


def test_one_abort_on_the_skills_read_still_writes_the_file(tmp_path):
    client = _scaffold_client([_resp(200, _OK_AGENTS)], [_resp(429), _resp(200, _OK_SKILLS)])
    result, sleep, out = _scaffold(client, tmp_path)
    assert result.exit_code == 0, result.output
    assert out.exists()
    sleep.assert_called_once_with(_ADMISSION_RETRY_DELAYS_S[0])


def test_an_abort_that_outlasts_the_schedule_on_the_scaffold_path_still_names_the_status(tmp_path):
    client = _scaffold_client([_resp(429)] * (len(_ADMISSION_RETRY_DELAYS_S) + 1), [])
    result, sleep, out = _scaffold(client, tmp_path)
    assert result.exit_code == 1
    assert "HTTP 429" in result.output
    assert not out.exists()
    assert client._http.get.call_count == len(_ADMISSION_RETRY_DELAYS_S) + 1
