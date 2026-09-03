"""A refused manifest registration must not become the agent's registration.

Found on prod 2026-09-03. The backend was in an OOM-restart loop and Cloud Run
aborted ~85% of requests at admission ("no available instance"), so
`POST /api/v1/manifests` was refused for many first traces. The langchain
adapter's `_register_snapshot` then wrote the snapshot's LOCAL id into
`_manifest_ids` and the hash into `_manifest_hashes` — the exact pair its own
dedupe check reads as "already registered" — so every later trace from that
agent shipped an id the platform never stored and ingest answered
`manifest_id '…' does not exist` for the life of the process. 9% of the fleet's
trace POSTs were that 400.

`generic`, `adk`, `otel` and `llamaindex` roll the tracker back on failure and
the Claude Agent SDK adapter leaves the hash unset; langchain did neither, and
also made no retry at all where `generic` makes three.

No backend: the client is a mock, and `time.sleep` is stubbed so the ladder
costs nothing here.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from decimalai.schema.manifest import ManifestTracker, extract_from_config


@pytest.fixture(autouse=True)
def sdk_state(monkeypatch):
    import decimalai._config as cfg
    import decimalai.langchain as lc_mod
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True)
    cfg._client = MagicMock()
    cfg._client.list_manifests.return_value = {"manifests": []}
    sender = MagicMock()
    monkeypatch.setattr(cfg, "_sender", sender)
    monkeypatch.setattr(lc_mod, "_manifest_id", None)
    monkeypatch.setattr(lc_mod, "_manifest_tracker", ManifestTracker())
    monkeypatch.setattr(lc_mod, "_manifest_ids", {})
    monkeypatch.setattr(lc_mod, "_manifest_hashes", {})
    monkeypatch.setattr(lc_mod, "_unregistered_agents", set(), raising=False)
    monkeypatch.setattr(lc_mod, "_manifest_adoption_probed", set(), raising=False)
    monkeypatch.setattr(lc_mod, "_explicit_manifest_config", None)
    monkeypatch.setattr("time.sleep", lambda _s: None)  # the ladder costs nothing here
    yield SimpleNamespace(cfg=cfg, lc=lc_mod, sender=sender)
    cfg._config = None
    cfg._client = None


def _snapshot(agent="support-bot"):
    return extract_from_config(
        agent_name=agent, models={"default": {"provider": "google", "model": "gemini-3.6-flash"}}
    )


def _refused(n):
    return [RuntimeError("503 Service Unavailable: no available instance")] * n


def test_a_refused_registration_is_retried_on_the_next_trace(sdk_state):
    """THE bug. Three refusals exhaust the ladder for trace 1; trace 2 must try again."""
    cfg, lc = sdk_state.cfg, sdk_state.lc
    snap = _snapshot()
    cfg._client.register_manifest.side_effect = _refused(3) + [{"manifest_id": "mf-real"}]

    first = lc._register_snapshot("support-bot", snap)
    assert first == str(snap.id), "trace 1 ships with the snapshot's local id, as before"
    assert cfg._client.register_manifest.call_count == 3, "the ladder is three attempts, like generic's"
    assert "support-bot" in lc._unregistered_agents
    assert "support-bot" not in lc._manifest_hashes, (
        "recording the hash is what made the refusal permanent — the dedupe check reads it"
    )
    sdk_state.sender.record_manifest_error.assert_called_once()

    second = lc._register_snapshot("support-bot", snap)
    assert second == "mf-real", (
        "the platform recovered and the next trace must carry the REGISTERED id; pre-fix this "
        "returned the cached local id and the platform answered 'does not exist' forever"
    )
    assert cfg._client.register_manifest.call_count == 4
    assert lc._manifest_ids["support-bot"] == "mf-real"
    assert "support-bot" not in lc._unregistered_agents


def test_a_transient_refusal_recovers_within_the_ladder(sdk_state):
    cfg, lc = sdk_state.cfg, sdk_state.lc
    snap = _snapshot()
    cfg._client.register_manifest.side_effect = _refused(1) + [{"manifest_id": "mf-2"}]
    assert lc._register_snapshot("support-bot", snap) == "mf-2"
    assert cfg._client.register_manifest.call_count == 2
    sdk_state.sender.record_manifest_error.assert_not_called()


def test_a_successful_registration_still_dedups(sdk_state):
    """Guard against over-correcting: the same shape is registered once per process."""
    cfg, lc = sdk_state.cfg, sdk_state.lc
    snap = _snapshot()
    cfg._client.register_manifest.return_value = {"manifest_id": "mf-1"}
    assert lc._register_snapshot("support-bot", snap) == "mf-1"
    assert lc._register_snapshot("support-bot", snap) == "mf-1"
    assert cfg._client.register_manifest.call_count == 1


def test_an_auth_failure_is_not_retried(sdk_state):
    """401/403 will not come back to life; burning the ladder only delays the trace."""
    cfg, lc = sdk_state.cfg, sdk_state.lc

    class _Forbidden(Exception):
        response = SimpleNamespace(status_code=403)

    cfg._client.register_manifest.side_effect = _Forbidden("403 Forbidden")
    snap = _snapshot()
    assert lc._register_snapshot("support-bot", snap) == str(snap.id)
    assert cfg._client.register_manifest.call_count == 1
    assert "support-bot" in lc._unregistered_agents, "still not a registration — the next trace retries"


def test_an_empty_run_does_not_keep_a_local_fallback_as_its_manifest(sdk_state):
    """`_resolve_manifest_for_empty_run` keeps whatever the agent already has. A local
    fallback id is not something it has; the run must go on to adopt or register."""
    cfg, lc = sdk_state.cfg, sdk_state.lc
    cfg._client.register_manifest.side_effect = _refused(3)
    lc._register_snapshot("support-bot", _snapshot())
    assert "support-bot" in lc._unregistered_agents

    cfg._client.list_manifests.return_value = {
        "manifests": [{"id": "mf-adopted", "status": "active"}]
    }
    lc.CallbackHandler(auto_send=False)._resolve_manifest_for_empty_run("support-bot")
    assert lc._manifest_ids["support-bot"] == "mf-adopted", (
        "an empty run rode the refused registration's local id instead of adopting the "
        "platform's active manifest"
    )
