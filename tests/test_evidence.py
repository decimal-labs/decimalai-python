import json
from uuid import uuid4
import httpx
from decimalai.evidence import EvidenceSpool, EvidenceClient, target_id
from decimalai._evidence_contract import digest


def receipt(event, status="accepted"):
    return {
        "event_id": event["event_id"],
        "seq": event["seq"],
        "event_hash": digest(event),
        "receipt_id": str(uuid4()),
        "status": status,
    }


def test_spool_restart_lost_ack_and_compaction_preserve_identity(tmp_path):
    spool = EvidenceSpool(tmp_path, "target")
    first = spool.append("runtime.heartbeat", {"observer_failures": 0})
    recovered = EvidenceSpool(tmp_path, "target", path=spool.path)
    assert recovered.pending() == [first]
    assert not recovered.acknowledge(
        {"durable": False, "items": [receipt(first)]}, [first]
    )
    assert recovered.pending() == [first]
    assert recovered.acknowledge(
        {"durable": True, "items": [receipt(first, "duplicate")]}, [first]
    )
    recovered.compact()
    second = recovered.append("runtime.heartbeat", {"observer_failures": 0})
    assert second["seq"] == 2 and second["event_id"] != first["event_id"]


def test_wrong_hash_or_unsent_ack_cannot_delete_pending(tmp_path):
    spool = EvidenceSpool(tmp_path, "target")
    event = spool.append("runtime.heartbeat", {})
    wrong = receipt(event)
    wrong["event_hash"] = "0" * 64
    assert not spool.acknowledge({"durable": True, "items": [wrong]}, [event])
    assert spool.pending() == [event]
    assert not spool.acknowledge({"durable": True, "items": []}, [event])
    assert not spool.acknowledge(
        {"durable": True, "items": [receipt(event), receipt(event)]}, [event]
    )


def test_rejected_event_quarantined_not_discarded(tmp_path):
    spool = EvidenceSpool(tmp_path, "target")
    event = spool.append("runtime.heartbeat", {})
    reject = {
        "event_id": event["event_id"],
        "seq": event["seq"],
        "status": "rejected",
        "reason": "identity_conflict",
    }
    assert spool.acknowledge({"durable": True, "items": [reject]}, [event])
    assert spool.health()["quarantined"] == 1
    with spool.db() as db:
        assert json.loads(db.execute("SELECT body FROM events").fetchone()[0]) == event


def test_seal_running_process_and_sensitive_metadata_rejected(tmp_path):
    spool = EvidenceSpool(tmp_path, "target")
    first = spool.append(
        "quality.page_failed", {"lane": "live", "owner_digest": "a" * 64}
    )
    assert (
        spool.append("runtime.heartbeat", {"reason": "Bearer sensitive-token"}) is None
    )
    sealed = spool.seal()
    assert sealed["payload"]["observer_failures"] == 1
    assert sealed["payload"]["ranges"][0]["last"] == first["seq"]
    assert first["boot_id"] == sealed["boot_id"]


def test_partition_isolation_and_upload_capability_fallback(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(404)

    client = EvidenceClient(
        "fake-key",
        "https://example.test",
        tmp_path,
        background=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.upload_once() is False
        assert calls == ["/api/v1/evidence/capabilities"]
        assert client.spool.pending()
        assert target_id("https://example.test", "different-key") != client.target
        try:
            EvidenceSpool(tmp_path, "different", path=client.spool.path)
            assert False, "foreign partition was accepted"
        except ValueError:
            pass
    finally:
        client.close()


def test_upload_commit_then_lost_response_reuses_event_ids(tmp_path):
    sent = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "enabled": True,
                    "schema_version": 1,
                    "acknowledgement": "per_event_after_commit",
                },
            )
        body = json.loads(request.content)
        sent.append(body["events"])
        if len(sent) == 1:
            raise httpx.ReadTimeout("lost ack")
        return httpx.Response(
            200,
            json={
                "batch_id": body["batch_id"],
                "durable": True,
                "items": [receipt(e, "duplicate") for e in body["events"]],
            },
        )

    client = EvidenceClient(
        "fake-key",
        "https://example.test",
        tmp_path,
        background=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        assert not client.upload_once()
        client.next_try = 0
        assert client.upload_once()
        assert sent[0] == sent[1]
        assert client.spool.pending() == []
    finally:
        client.close()


def test_uploader_round_robins_old_and_new_shards(tmp_path):
    sent_streams = []

    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "enabled": True,
                    "schema_version": 1,
                    "acknowledgement": "per_event_after_commit",
                },
            )
        body = json.loads(request.content)
        sent_streams.append(body["stream_id"])
        return httpx.Response(
            200,
            json={
                "batch_id": body["batch_id"],
                "durable": True,
                "items": [receipt(e) for e in body["events"]],
            },
        )

    client = EvidenceClient(
        "fake-key",
        "https://example.test",
        tmp_path,
        background=False,
        transport=httpx.MockTransport(handler),
    )
    other = EvidenceSpool(tmp_path, client.target)
    foreign = EvidenceSpool(tmp_path, "foreign")
    foreign.append("runtime.heartbeat", {})
    for _ in range(200):
        client.spool.append("runtime.heartbeat", {})
        other.append("runtime.heartbeat", {})
    try:
        for _ in range(3):
            client.next_try = 0
            assert client.upload_once()
        assert set(sent_streams) == {client.spool.stream_id, other.stream_id}
        assert foreign.pending()
    finally:
        client.close()


def test_trace_observer_emits_explicit_adapter_witness_only(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from decimalai import evidence

    client = EvidenceClient(
        "fake-key",
        "https://example.test",
        tmp_path,
        background=False,
        transport=httpx.MockTransport(lambda _: httpx.Response(404)),
    )
    monkeypatch.setattr(evidence, "_bindings", {client.target: client})
    witness = {
        "skill_id": str(uuid4()),
        "version_id": str(uuid4()),
        "hash": "b" * 64,
        "body": "private skill body",
    }
    trace = SimpleNamespace(
        id=str(uuid4()),
        parent_trace_id=str(uuid4()),
        skills_delivered_versions=[witness],
        llm_calls=[SimpleNamespace(id=str(uuid4()))],
        session_metadata={"framework": "adk", "prompt": "private prompt"},
    )
    try:
        evidence.observe_trace(
            SimpleNamespace(api_key="fake-key", base_url="https://example.test"), trace
        )
        pending = client.spool.pending()
        invocation = next(e for e in pending if e["kind"] == "adapter.invocation")
        assert invocation["payload"]["witnesses_complete"] is True
        assert invocation["payload"]["framework"] == "adk"
        assert any(e["kind"] == "window.sealed" for e in pending)
        assert "private" not in json.dumps(pending)
    finally:
        client.close()


def test_closed_shard_only_pruned_after_all_hash_checked_acks(tmp_path):
    spool = EvidenceSpool(tmp_path, "target")
    spool.append("runtime.heartbeat", {})
    spool.close_stream()
    assert spool.append("runtime.heartbeat", {}) is None
    assert not spool.prune_closed()
    events = spool.pending()
    assert spool.acknowledge(
        {"durable": True, "items": [receipt(e) for e in events]}, events
    )
    spool.compact()
    assert spool.prune_closed() and not spool.path.exists()
    rejected = EvidenceSpool(tmp_path, "target")
    event = rejected.append("runtime.heartbeat", {})
    rejected.close_stream()
    reject = {
        "event_id": event["event_id"],
        "seq": event["seq"],
        "status": "rejected",
        "reason": "identity_conflict",
    }
    rejected.acknowledge({"durable": True, "items": [reject]}, [event])
    assert not rejected.prune_closed()
    assert rejected.path.exists() and rejected.health()["quarantined"] == 1


def test_revoked_auth_preserves_events_and_backs_off(tmp_path):
    def handler(request):
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "enabled": True,
                    "schema_version": 1,
                    "acknowledgement": "per_event_after_commit",
                },
            )
        return httpx.Response(403)

    client = EvidenceClient(
        "fake-key",
        "https://example.test",
        tmp_path,
        background=False,
        transport=httpx.MockTransport(handler),
    )
    try:
        before = client.spool.pending()
        import time

        assert not client.upload_once()
        assert client.last_status == "authorization_pending"
        assert client.next_try - time.monotonic() >= 3599
        assert client.spool.pending() == before
    finally:
        client.close()
