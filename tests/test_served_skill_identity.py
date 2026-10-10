"""The served version survives cache replay, tool loads and concurrent runs."""

import threading
from unittest.mock import patch

import pytest

from decimalai import skill_router as sr
from decimalai.generic import start_trace
from decimalai.skill_router import SkillRouter, consume_last_delivered_versions

NAME = "python-docstring-conventions"
SKILL = "3b6b49b2-f2e0-4c19-9a96-459f125ae054"
VERSION = "ca093995-2f31-4c5f-aea7-1ccf0b68cf5e"
NEXT_VERSION = "968e481b-2265-4b44-b72c-5cdb8293ce6e"
HASH = "a" * 64
BODY = "Document each supplied Python function with its complete contract."
ROUTE = {"prompt_fragment": NAME, "routing_id": "rt_" + "1" * 24,
         "skills": [{"name": NAME}]}


@pytest.fixture(autouse=True)
def fresh_context():
    sr._body_budget_ctx.set(None)
    consume_last_delivered_versions()
    yield
    sr._body_budget_ctx.set(None)
    consume_last_delivered_versions()


def router():
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=True)


def record(**overrides):
    return {"body": BODY, "version": 4, "skill_id": SKILL, "version_id": VERSION,
            "content_hash": HASH, **overrides}


def witness(**overrides):
    return {"name": NAME, "hash": HASH, "skill_id": SKILL, "version_id": VERSION, **overrides}


@pytest.mark.parametrize("mode", ["prompt", "tool"])
def test_actual_served_pair_is_copied_into_owned_rail_and_native_trace(mode):
    r = router()
    with patch.object(r, "smart_route", return_value=ROUTE), \
            patch.object(r, "get_skill_body_record", return_value=record()), \
            patch("decimalai._config._get_config") as config:
        config.return_value.project = "proof"
        with start_trace(agent_name="native", auto_send=False) as ctx:
            if mode == "prompt":
                text = r.build_prompt_parts("document this function", scope="own")[0]
            else:
                text = r.load_skill(NAME, scope="own")
            assert BODY in text
            expected = witness(**({"routing_id": ROUTE["routing_id"]} if mode == "prompt" else {}))
            trace = ctx.build_trace()
    assert trace.skills_delivered_versions == [expected]
    returned = r.consume_delivered_versions(scope="own")
    assert returned == [expected]
    returned[0]["version_id"] = "consumer mutation"
    assert trace.skills_delivered_versions == [expected]
    assert r.consume_delivered_versions(scope="own") == []


def test_cached_prompt_replays_its_exact_pair_after_newer_fetch_and_mutation():
    r = router()
    with patch.object(r, "smart_route", return_value=ROUTE), \
            patch.object(r, "get_skill_body_record", return_value=record()) as get:
        first = r.build_prompt_parts("old", scope="old")
        consume_last_delivered_versions()[0]["version_id"] = "mutated"
        get.return_value = record(body="New instructions", version_id=NEXT_VERSION, content_hash="b" * 64)
        r.build_prompt_parts("new", scope="new")
        assert consume_last_delivered_versions()[0]["version_id"] == NEXT_VERSION
        assert r.build_prompt_parts("old", scope="old") == first
        assert get.call_count == 2
    expected = witness(routing_id=ROUTE["routing_id"])
    assert consume_last_delivered_versions() == [expected]
    assert r.consume_delivered_versions(scope="old") == [expected]


@pytest.mark.parametrize("identity", [
    {}, {"skill_id": SKILL}, {"version_id": VERSION},
    {"skill_id": SKILL, "version_id": "invalid"},
    {"skill_id": SKILL.upper(), "version_id": VERSION},
])
def test_old_or_malformed_response_never_invents_or_borrows_uuid_pair(identity):
    r = router()
    with patch.object(r, "get_skill_body_record", return_value=record()):
        r.get_skill_body(NAME)
    old = {"body": BODY, "content_hash": HASH, **identity}
    with patch.object(r, "smart_route", return_value=ROUTE), \
            patch.object(r, "get_skill_body_record", return_value=old):
        assert BODY in r.build_prompt_parts("old server", scope="legacy")[0]
    assert consume_last_delivered_versions() == [{"name": NAME, "hash": HASH, "routing_id": ROUTE["routing_id"]}]


@pytest.mark.parametrize("mode", ["prompt", "tool"])
def test_partial_body_carries_neither_served_identity_nor_scoped_witness(mode):
    r = router()
    with patch.object(r, "smart_route", return_value=ROUTE), \
            patch.object(r, "get_skill_body_record", return_value=record(truncated=True)):
        text = r.build_prompt_parts("partial", scope="partial")[0] if mode == "prompt" else r.load_skill(NAME, scope="partial")
    assert "## Skill:" not in text
    assert consume_last_delivered_versions() == []
    assert r.consume_delivered_versions(scope="partial") == []


def test_concurrent_same_name_fetches_keep_each_response_pair():
    r = router()
    barrier = threading.Barrier(2)
    real = r.get_skill_body
    seen, errors = {}, []

    def fetch(*args, **kwargs):
        version = VERSION if threading.current_thread().name == "first" else NEXT_VERSION
        return record(body=version, version_id=version)

    def get(*args, **kwargs):
        body = real(*args, **kwargs)
        barrier.wait(timeout=10)
        return body

    def build(scope):
        try:
            text = r.build_prompt_parts("query", scope=scope)[0]
            seen[scope] = (text, consume_last_delivered_versions())
        except BaseException as exc:
            errors.append(exc)

    with patch.object(r, "smart_route", return_value=ROUTE), \
            patch.object(r, "get_skill_body_record", side_effect=fetch), \
            patch.object(r, "get_skill_body", side_effect=get):
        threads = [threading.Thread(target=build, args=(name,), name=name) for name in ("first", "second")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(15)
    assert not errors and all(not thread.is_alive() for thread in threads)
    for scope, version in (("first", VERSION), ("second", NEXT_VERSION)):
        text, versions = seen[scope]
        assert version in text
        expected = witness(version_id=version, routing_id=ROUTE["routing_id"])
        assert versions == r.consume_delivered_versions(scope=scope) == [expected]


def test_scope_release_and_atomic_owner_check_prevent_foreign_version_handoff():
    r = router()
    with patch.object(r, "get_skill_body_record", return_value=record()):
        r.load_skill(NAME, scope="own")
    assert r.drain_unscoped_rails_for(["other"], include_versions=True) == (None, [], [], [], [])
    assert r.drain_unscoped_rails_for(["own"], include_versions=True)[-1] == [witness()]
    r.consume_loaded_names(scope="own")
    sr._release_scoped_routing_rail(r, "own")
    assert "own" not in r._scoped_routing_rails
    with patch.object(r, "get_skill_body_record", return_value=record()):
        r.load_skill(NAME)
    # Historical assemble-then-invoke names still work. Unknown ownership
    # cannot certify the exact immutable UUIDs of a subsequent adapter run.
    assert r.drain_unscoped_rails_for(["other"], include_versions=True)[-1] == []


def test_version_mirror_is_bounded_without_erasing_another_live_scope(caplog):
    r = router()
    r._MAX_DELIVERED_VERSIONS = 2
    for i in range(3):
        r._record_delivery_versions([witness(hash=str(i))], scope=f"scope-{i}")
    assert len(r._delivered_versions_rail) == 2
    assert r.consume_delivered_versions(scope="scope-0") == [witness(hash="0")]
    assert "unscoped witness evicted" in caplog.text


def test_releasing_identical_witness_preserves_other_scope_and_completed_runs_do_not_accumulate(caplog):
    r = router()
    r._record_delivery_versions([witness()], scope="first")
    r._record_delivery_versions([witness()], scope="second")
    sr._release_scoped_routing_rail(r, "first")
    assert r.consume_delivered_versions(scope="second") == [witness()]
    assert r._delivered_versions_rail == []
    for i in range(r._MAX_DELIVERED_VERSIONS + 2):
        r._record_delivery_versions([witness(hash=str(i))], scope=f"scope-{i}")
        sr._release_scoped_routing_rail(r, f"scope-{i}")
    assert r._delivered_versions_rail == []
    assert r._scoped_routing_rails == {}
    assert "skill version rail overflow" not in caplog.text
