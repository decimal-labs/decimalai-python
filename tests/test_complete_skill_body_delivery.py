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
    from decimalai import skill_router as sr

    consume_last_delivered_names()
    consume_last_delivered_versions()
    # `build_prompt_parts` and `load_skill` file a turn budget on the
    # module-level ContextVar, and pytest runs every test in one context. Left
    # in place, the next file's unscoped `load_skill` charged its loads to this
    # file's budget object, and a later test there was told "'alpha' is already
    # loaded" for a body it never received:
    # test_langchain_run_scoped_skill_rails failed 2 when run after this file.
    budget_token = sr._body_budget_ctx.set(None)
    yield
    sr._body_budget_ctx.reset(budget_token)
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


# ── a row that cannot go in costs that row, not the turn ─────────────────────────────────────────
#
# Measured on the fleet, 2026-11-01: 5 of 15 CrewAI steps that got no body had ranked first a skill of
# 44,000-45,000 characters, with rows of 6,500-11,000 characters right behind it.

def _ranked(*names):
    return {"prompt_fragment": "MENU", "routing_id": "rt_rows", "strategy": "smart_routing",
            "query_ranked": True, "skills": [{"name": n} for n in names]}


def _records(by_name):
    """A `get_skill_body_record` that answers by name and says which names were asked for."""
    asked = []

    def answer(name, *_args, **_kwargs):
        asked.append(name)
        return by_name[name]
    return answer, asked


def _prompt(router, route, by_name, scope):
    answer, asked = _records(by_name)
    with patch.object(router, "smart_route", return_value=route), \
            patch.object(router, "get_skill_body_record", side_effect=answer):
        prefix, _, _ = router.build_prompt_parts("review this function", scope=scope)
    return prefix, asked


@pytest.mark.parametrize("too_large", [
    _record("partial", truncated=True),               # what a server does with max_chars
    _record("y" * 30_000),                            # an older server that ignores max_chars
])
def test_a_top_row_over_the_limit_gives_way_to_the_next(too_large):
    router = _router(inject_body=True)
    prefix, asked = _prompt(router, _ranked("big", "fits", "third"),
                            {"big": too_large, "fits": _record("the next row"),
                             "third": _record("never fetched")}, "next")
    assert "## Skill: fits" in prefix and "## Skill: big" not in prefix
    assert consume_last_delivered_names() == ["fits"]
    assert asked == ["big", "fits"], "the row after one that fit must not be fetched"


def test_a_top_row_over_the_token_budget_gives_way_to_the_next():
    router = _router(inject_body=True, per_body_char_limit=2000, body_token_budget=50)
    prefix, _ = _prompt(router, _ranked("big", "fits"),
                        {"big": _record("x" * 1000), "fits": _record("short")}, "budget")
    assert "## Skill: fits" in prefix
    assert consume_last_delivered_names() == ["fits"]


def test_a_top_row_that_fits_makes_no_extra_request():
    """This is on the hot path of every turn, so the spare rows must cost nothing normally."""
    router = _router(inject_body=True)
    _, asked = _prompt(router, _ranked("first", "second", "third"),
                       {"first": _record("fits"), "second": _record("b"), "third": _record("c")},
                       "fits")
    assert asked == ["first"]
    assert consume_last_delivered_names() == ["first"]


def test_the_walk_past_a_refused_row_is_bounded():
    """Three refusals end it, so a menu of oversized skills can't turn into a dozen fetches."""
    router = _router(inject_body=True)
    big = _record("partial", truncated=True)
    prefix, asked = _prompt(router, _ranked("a", "b", "c", "d", "e"),
                            {n: big for n in "abcde"}, "bounded")
    assert asked == ["a", "b", "c"]
    assert "## Skill:" not in prefix
    assert consume_last_delivered_names() == []


def test_a_later_body_over_the_budget_still_ends_the_turn():
    """Spare rows are for a turn with no body yet. Once one is in, the budget means stop."""
    router = _router(inject_body=True, inject_body_top_k=3, per_body_char_limit=400,
                     body_token_budget=150)
    _, asked = _prompt(router, _ranked("s1", "s2", "s3"),
                       {n: _record("b" * 400) for n in ("s1", "s2", "s3")}, "later")
    assert asked == ["s1", "s2"]
    assert consume_last_delivered_names() == ["s1"]


def test_an_unranked_result_still_injects_nothing():
    """The spare rows must not widen what counts as a body worth injecting: a menu's top row is the
    workspace's best-rated skill, not one about this turn."""
    router = _router(inject_body=True)
    route = {**_ranked("a", "b", "c"), "strategy": "full_menu", "query_ranked": False}
    prefix, asked = _prompt(router, route, {n: _record("text") for n in "abc"}, "unranked")
    assert asked == [] and "## Skill:" not in prefix
    assert consume_last_delivered_names() == []

