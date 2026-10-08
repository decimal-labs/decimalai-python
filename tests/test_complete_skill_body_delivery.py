"""A full version enters model context only when its complete body fits.

Exercise the public GET wrapper, tool load and prompt/cache paths together;
partial responses must not acquire delivery or immutable-version witnesses.
"""

from unittest.mock import patch

import pytest

from decimalai.skill_router import (
    SkillRouter,
    SkillRouterError,
    consume_last_delivered_names,
    consume_last_delivered_versions,
)

NAME = "python-docstring-conventions"
HASH = "a" * 64
ROUTE = {
    "prompt_fragment": "Available skill: " + NAME,
    "routing_id": "rt_" + "1" * 24,
    "skills": [{"name": NAME}],
    "stable_menu": "Available skill: " + NAME,
    "stable_menu_skills": [NAME],
    "routing_hint": "Use " + NAME + ".",
}


def _router(**kwargs):
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", **kwargs)


def _record(body, **kwargs):
    return {"body": body, "version": 3, "content_hash": HASH, **kwargs}


@pytest.fixture(autouse=True)
def _clear_delivery_rails():
    consume_last_delivered_names()
    consume_last_delivered_versions()
    yield
    consume_last_delivered_names()
    consume_last_delivered_versions()


def test_default_body_cap_tracks_the_existing_total_budget():
    assert _router().per_body_char_limit == 24_000
    assert _router(body_token_budget=1000).per_body_char_limit == 4000
    assert _router(per_body_char_limit=500).per_body_char_limit == 500


@pytest.mark.parametrize("mode", ["prompt", "tool"])
@pytest.mark.parametrize("body,flags", [
    ("partial body", {"truncated": True}),
    ("partial body", {"content_hash_matches_body": False}),
    ("partial body", {"total_chars": 1000}),
    ("partial\n\n[... truncated by the per-body limit]", {}),
    ("partial\n\n[... truncated 20 of 100 chars — request without max_chars for the full body]", {}),
])
def test_partial_server_body_is_never_loaded_or_witnessed(mode, body, flags):
    router = _router(inject_body=True)
    # A previous complete fetch must not lend its hash to a later partial one.
    with patch.object(router, "get_skill_body_record", return_value=_record("complete")):
        assert router.get_skill_body(NAME) == "complete"
    with patch.object(router, "smart_route", return_value=ROUTE), \
            patch.object(router, "get_skill_body_record", return_value=_record(body, **flags)):
        if mode == "prompt":
            out, _, _ = router.build_prompt_parts("document this function", scope="partial")
        else:
            out = router.load_skill(NAME, scope="partial")
            assert "partial body" in out
    assert "## Skill:" not in out
    assert consume_last_delivered_names() == []
    assert consume_last_delivered_versions() == []
    assert router.consume_loaded_names(scope="partial") == []
    assert router.consume_loaded_hashes(scope="partial") == {}
    assert router.loaded_skill_hash(NAME) is None


def test_direct_get_can_return_requested_partial_text_without_a_full_version_hash():
    router = _router()
    with patch.object(router, "_request", return_value=_record("partial", truncated=True)):
        assert router.get_skill_body(NAME, max_chars=256) == "partial"
    assert router.loaded_skill_hash(NAME) is None


def test_explicit_complete_record_can_end_with_a_literal_marker_example():
    body = "Literal example:\n\n[... truncated by the per-body limit]"
    router = _router(inject_body=True)
    with patch.object(router, "smart_route", return_value=ROUTE), \
            patch.object(router, "get_skill_body_record", return_value=_record(
                body, truncated=False, content_hash_matches_body=True, total_chars=len(body),
            )):
        prefix, _, _ = router.build_prompt_parts("explain this marker", scope="literal")
    assert body in prefix
    assert consume_last_delivered_names() == [NAME]
    assert consume_last_delivered_versions() == [{"name": NAME, "hash": HASH, "routing_id": ROUTE["routing_id"]}]


@pytest.mark.parametrize("mode", ["prompt", "tool"])
@pytest.mark.parametrize("kwargs", [
    {"per_body_char_limit": 300},
    {"per_body_char_limit": 2000, "body_token_budget": 50},
])
def test_old_server_oversized_first_body_is_omitted_without_partial_context(mode, kwargs):
    router = _router(inject_body=True, **kwargs)
    body = "x" * 1000
    with patch.object(router, "smart_route", return_value=ROUTE), \
            patch.object(router, "get_skill_body_record", return_value=_record(body)):
        if mode == "prompt":
            out, _, _ = router.build_prompt_parts("document this function", scope="oversize")
        else:
            out = router.load_skill(NAME, scope="oversize")
            assert "budget exhausted" in out
    assert body[:300] not in out
    assert "## Skill:" not in out
    assert consume_last_delivered_names() == []
    assert consume_last_delivered_versions() == []
    assert router.consume_loaded_names(scope="oversize") == []
    assert router.consume_loaded_hashes(scope="oversize") == {}


def test_partial_cache_and_failed_fetch_do_not_lend_witnesses_to_the_next_body():
    router = _router(inject_body=True)
    with patch.object(router, "smart_route", return_value=ROUTE), \
            patch.object(router, "_request", return_value=_record("partial", truncated=True)) as request:
        first = router.build_prompt_parts("partial request", scope="partial")
        assert consume_last_delivered_versions() == []
        assert router.build_prompt_parts("partial request", scope="partial") == first
        assert request.call_count == 1
        assert consume_last_delivered_versions() == []
        request.side_effect = SkillRouterError("temporary read failure")
        router.build_prompt_parts("failed request", scope="failed")
        assert consume_last_delivered_versions() == []
        request.side_effect = None
        request.return_value = _record("new complete body", truncated=False, content_hash_matches_body=True)
        prefix, _, _ = router.build_prompt_parts("complete request", scope="complete")
    assert "new complete body" in prefix
    assert consume_last_delivered_names() == [NAME]
    assert consume_last_delivered_versions() == [{"name": NAME, "hash": HASH, "routing_id": ROUTE["routing_id"]}]
