"""An injected body carries its own immutable version, never the latest fetch.

The actual GET/body method is mocked; the hash handoff and prompt assembly are
real. Prompt delivery is distinct from a model's load_skill activation.
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter, consume_last_delivered_versions

BODY = "Return only complete Google-style Python docstrings."
HASH = "a" * 64
OTHER_HASH = "b" * 64
NAME = "python-docstring-conventions"
ROUTING_ID = "rt_" + "1" * 24


@pytest.fixture(autouse=True)
def _fresh_rails():
    sr._last_delivered_versions_ctx.set(None)
    sr._body_budget_ctx.set(None)
    yield
    sr._last_delivered_versions_ctx.set(None)
    sr._body_budget_ctx.set(None)


def _router(**kw):
    return SkillRouter(
        api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=True, **kw,
    )


def _route(names=(NAME,)):
    return {
        "prompt_fragment": "Available skills: " + ", ".join(names),
        "routing_id": ROUTING_ID,
        "skills": [{"name": name} for name in names],
        "stable_menu": "Available skills: " + ", ".join(names),
        "stable_menu_skills": list(names),
        "routing_hint": "Use " + ", ".join(names) + ".",
    }


def _record(body=BODY, digest=HASH):
    rec = {"body": body, "version": 3}
    if digest is not None:
        rec["content_hash"] = digest
    return rec


def test_the_exact_body_response_is_witnessed_without_claiming_an_activation():
    router = _router()
    with patch.object(router, "smart_route", return_value=_route()), \
         patch.object(router, "get_skill_body_record", return_value=_record()):
        prefix, _, _ = router.build_prompt_parts("document this function", scope="run-1")

    assert BODY in prefix
    assert consume_last_delivered_versions() == [{"name": NAME, "hash": HASH, "routing_id": ROUTING_ID}]
    assert consume_last_delivered_versions() == []
    assert router.consume_loaded_names(scope="run-1") == []
    assert router.consume_loaded_hashes(scope="run-1") == {}


def test_cache_replays_the_delivered_version_after_a_newer_fetch_and_consumer_mutation():
    router = _router()
    with patch.object(router, "smart_route", return_value=_route()), \
         patch.object(router, "get_skill_body_record", return_value=_record()):
        first = router.build_prompt_parts("document this function", scope="run-1")
        witnessed = consume_last_delivered_versions()
        witnessed[0]["hash"] = "consumer mutation"
        # A later real routing/fetch updates both latest-seen body and routing
        # state. The old cached prompt must still witness its own decision.
        newer_route = {**_route(), "routing_id": "rt_" + "2" * 24}
        with patch.object(router, "get_skill_body_record", return_value=_record("new body", OTHER_HASH)), \
             patch.object(router, "smart_route", return_value=newer_route):
            router.build_prompt_parts("new query", scope="run-2")
            assert consume_last_delivered_versions() == [
                {"name": NAME, "hash": OTHER_HASH, "routing_id": newer_route["routing_id"]},
            ]
        second = router.build_prompt_parts("document this function", scope="run-1")

    assert first == second
    assert BODY in second[0] and "new body" not in second[0]
    assert consume_last_delivered_versions() == [{"name": NAME, "hash": HASH, "routing_id": ROUTING_ID}]


@pytest.mark.parametrize("body,digest", [(BODY, None), ("", HASH), (None, HASH)])
def test_no_body_or_no_response_hash_never_borrows_a_previous_version(body, digest):
    router = _router()
    with patch.object(router, "get_skill_body_record", return_value=_record()):
        router.get_skill_body(NAME)
    sr._last_delivered_versions_ctx.set([{"name": NAME, "hash": HASH}])
    with patch.object(router, "smart_route", return_value=_route()), \
         patch.object(router, "get_skill_body_record", return_value=_record(body, digest)):
        router.build_prompt_parts("a new query", scope="run-2")
    assert consume_last_delivered_versions() == []


def test_an_override_without_a_hash_cannot_reuse_the_thread_local_from_a_prior_fetch():
    router = _router()
    with patch.object(router, "get_skill_body_record", return_value=_record()):
        router.get_skill_body(NAME)
    with patch.object(router, "smart_route", return_value=_route()), \
         patch.object(router, "get_skill_body", return_value=BODY):
        router.build_prompt_parts("document this function", scope="run-1")
    assert consume_last_delivered_versions() == []


def test_only_the_bodies_that_survive_the_prompt_budget_have_witnesses():
    router = _router(inject_body_top_k=2, max_loaded_bodies=2, body_token_budget=1)
    with patch.object(router, "smart_route", return_value=_route((NAME, "second"))), \
         patch.object(router, "get_skill_body_record", side_effect=[_record(), _record("second body", OTHER_HASH)]):
        prefix, _, _ = router.build_prompt_parts("document this function", scope="run-1")
    assert BODY in prefix and "## Skill: second" not in prefix
    assert consume_last_delivered_versions() == [{"name": NAME, "hash": HASH, "routing_id": ROUTING_ID}]


def test_menu_only_prompt_clears_a_previous_unconsumed_witness():
    router = _router()
    with patch.object(router, "smart_route", return_value=_route()), \
         patch.object(router, "get_skill_body_record", return_value=_record()):
        router.build_prompt_parts("with body", scope="run-1")
        router.build_prompt_parts("menu only", scope="run-2", inject_body=False)
    assert consume_last_delivered_versions() == []


def test_concurrent_prompt_builds_of_one_name_do_not_exchange_body_versions():
    router = _router()
    real_get_body = router.get_skill_body
    gate = threading.Barrier(2)
    seen = {}
    errors = []

    def fetch(name, *args, **kw):
        digest = HASH if threading.current_thread().name == "first" else OTHER_HASH
        return _record(body=digest, digest=digest)

    def synchronized_get_body(*args, **kw):
        body = real_get_body(*args, **kw)
        # Both persistent last-seen writes have happened before either caller
        # reads the per-fetch handoff. Re-reading shared state would fail here.
        gate.wait(timeout=10)
        return body

    def run(scope):
        try:
            prefix, _, _ = router.build_prompt_parts("document this function", scope=scope)
            seen[scope] = (prefix, consume_last_delivered_versions())
        except BaseException as exc:
            errors.append(exc)

    with patch.object(router, "smart_route", return_value=_route()), \
         patch.object(router, "get_skill_body_record", side_effect=fetch), \
         patch.object(router, "get_skill_body", side_effect=synchronized_get_body):
        first = threading.Thread(target=run, args=("run-1",), name="first")
        second = threading.Thread(target=run, args=("run-2",), name="second")
        first.start()
        second.start()
        first.join(20)
        second.join(20)

    assert not first.is_alive() and not second.is_alive()
    assert not errors
    for scope, digest in (("run-1", HASH), ("run-2", OTHER_HASH)):
        prefix, versions = seen[scope]
        assert digest in prefix
        assert versions == [{"name": NAME, "hash": digest, "routing_id": ROUTING_ID}]
