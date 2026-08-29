"""Tests for SkillRouter.load_skill + the per-turn body-load budget.

All network mocked (patch ``router._request`` / ``router.get_skill_body`` —
same idiom as tests/test_skill_router.py). Covers:

  - load_skill happy path (server-trim param, prefixed body block, telemetry)
  - max_loaded_bodies count budget + dedup (already-loaded name is free)
  - body_token_budget (first body always allowed, second large body refused)
  - not-found and empty-name messages
  - client-side per_body_char_limit trim (defense against old backends)
  - fresh-fragment budget reset vs cached-fragment NO reset
  - _BodyLoadBudget deadline refusal
  - inject path guardrail (per-body trim, count cap, token budget,
    server-trim requested via max_chars)
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from decimalai import skill_router as sr
from decimalai.skill_router import (
    LOAD_SKILL_TOOL_NAME,
    SkillRouter,
    _BodyLoadBudget,
    estimate_tokens,
    load_skill_tool_spec,
)

TRUNCATION_MARKER = "[... truncated by the per-body limit]"


@pytest.fixture(autouse=True)
def _fresh_body_budget():
    """The body budget lives in a module-level ContextVar shared by every
    test in this thread — clear it around each test so one test's
    exhausted budget can't leak into the next."""
    sr._body_budget_ctx.set(None)
    sr._last_offered_names_ctx.set(None)
    sr._last_delivered_names_ctx.set(None)
    yield
    sr._body_budget_ctx.set(None)
    sr._last_offered_names_ctx.set(None)
    sr._last_delivered_names_ctx.set(None)


def _router(**kw) -> SkillRouter:
    return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", **kw)


# ── Module-level helpers ─────────────────────────────────────────


class TestEstimateTokens:
    def test_chars_over_four_rounded_up(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("a") == 1
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("abcde") == 2
        assert estimate_tokens("x" * 400) == 100


class TestLoadSkillToolSpec:
    def test_shape(self):
        spec = load_skill_tool_spec()
        assert spec["name"] == LOAD_SKILL_TOOL_NAME == "load_skill"
        assert spec["parameters"]["type"] == "object"
        assert spec["parameters"]["required"] == ["name"]
        assert "name" in spec["parameters"]["properties"]
        assert isinstance(spec["description"], str) and spec["description"]


# ── load_skill ───────────────────────────────────────────────────


class TestLoadSkill:
    def test_happy_path_returns_prefixed_body_and_server_trims(self):
        router = _router()
        mock_response = {"body": "Do X, then Y.", "version": 3}
        with patch.object(router, "_request", return_value=mock_response) as req, \
                patch("decimalai.generic.log_skill_loaded") as logged:
            out = router.load_skill("code-review")

        assert out == "## Skill: code-review\n\nDo X, then Y."
        req.assert_called_once()
        method, path = req.call_args[0]
        assert method == "GET"
        assert path == "/api/v1/skills/code-review/body"
        # Server-side trim requested: max_chars carried as a query param.
        params = req.call_args[1]["params"]
        assert params["max_chars"] == router.per_body_char_limit
        # Best-effort activation telemetry. Asserted on the NAME rather than on
        # the whole call: `log_skill_loaded` also takes an optional `hash=` (the
        # body's content_hash, so the activation resolves to a skill VERSION),
        # and pinning the exact signature here made adding it look like a
        # behaviour change when the reported load is identical.
        logged.assert_called_once()
        assert logged.call_args.kwargs["name"] == "code-review"

    def test_agent_name_kwarg_forwarded_as_param(self):
        router = _router()
        with patch.object(router, "_request", return_value={"body": "B", "version": 1}) as req:
            router.load_skill("code-review", agent_name="shopper")
        assert req.call_args[1]["params"]["agent_name"] == "shopper"

    def test_telemetry_failure_is_non_fatal(self):
        """No active trace → log_skill_loaded raises; load_skill still
        returns the body (the record is best-effort)."""
        router = _router()
        with patch.object(router, "get_skill_body", return_value="B"), \
                patch("decimalai.generic.log_skill_loaded", side_effect=RuntimeError("no trace")):
            out = router.load_skill("code-review")
        assert out == "## Skill: code-review\n\nB"

    def test_max_loaded_bodies_refuses_third_and_dedups_reload(self):
        router = _router(max_loaded_bodies=2)
        with patch.object(
            router, "get_skill_body", side_effect=lambda n, **kw: f"BODY of {n}"
        ):
            assert router.load_skill("a") == "## Skill: a\n\nBODY of a"
            assert router.load_skill("b") == "## Skill: b\n\nBODY of b"

            refusal = router.load_skill("c")
            assert "budget exhausted" in refusal
            assert "2 bodies" in refusal
            # The refusal tells the model what IS loaded so it can proceed.
            assert "a" in refusal and "b" in refusal

            # Dedup: an ALREADY loaded name still costs no budget, but it does
            # NOT get the body back a second time. Returning it again is what let
            # the openai-agents scaffold loop turns 3..10 on one skill and die
            # with MaxTurnsExceeded without ever answering — the refusal path
            # above only fires on DISTINCT names, so it never saw the repeat.
            again = router.load_skill("a")
            assert "already loaded" in again
            assert "BODY of a" not in again
            assert "answer the user" in again.lower()
            # Still free: the repeat is not reported as budget exhaustion, so it
            # did not consume one of the two body slots a second time.
            assert "budget exhausted" not in again

    def test_token_budget_refuses_second_large_body(self):
        router = _router(body_token_budget=50)
        big = "x" * 400  # ~100 tokens — alone over the 50-token budget
        with patch.object(router, "get_skill_body", return_value=big):
            # First body is ALWAYS allowed, even if it alone exceeds the budget
            # (would_exceed only kicks in once something is loaded).
            first = router.load_skill("a")
            assert first.startswith("## Skill: a\n\n")

            refusal = router.load_skill("b")
            assert "load_skill budget exhausted" in refusal
            assert "50-token body budget" in refusal
            assert "'b'" in refusal

    def test_not_found_returns_no_skill_named_message(self):
        router = _router()
        with patch.object(router, "get_skill_body", return_value=None):
            out = router.load_skill("ghost-skill")
        assert "no skill named" in out
        assert "ghost-skill" in out
        # A not-found must not consume budget slots.
        with patch.object(router, "get_skill_body", return_value="B"):
            assert router.load_skill("real").startswith("## Skill: real")

    def test_empty_name_is_an_error_message(self):
        router = _router()
        out = router.load_skill("   ")
        assert "load_skill error" in out
        assert "name is required" in out

    def test_client_side_trim_appends_marker(self):
        router = _router(per_body_char_limit=300)
        with patch.object(router, "get_skill_body", return_value="z" * 1000):
            out = router.load_skill("big-skill")
        prefix = "## Skill: big-skill\n\n"
        assert out.startswith(prefix)
        assert out[len(prefix):] == "z" * 300 + "\n\n" + TRUNCATION_MARKER


# ── Budget lifecycle: fresh fragment resets, cache hit does not ──


class TestBudgetReset:
    def test_fresh_fragment_resets_budget(self):
        router = _router(max_loaded_bodies=1)
        route = {
            "prompt_fragment": "MENU", "routing_id": "rt_1",
            "skills": [{"name": "a"}],
        }
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(router, "get_skill_body", side_effect=lambda n, **kw: f"BODY {n}"):
            router.build_prompt_fragment(query="q", bypass_cache=True)
            assert router.load_skill("a").startswith("## Skill: a")
            assert "budget exhausted" in router.load_skill("b")

            # A FRESH fragment marks a new turn — the budget resets and
            # loads are allowed again.
            router.build_prompt_fragment(query="q", bypass_cache=True)
            assert router.load_skill("b").startswith("## Skill: b")

    def test_cached_fragment_does_not_reset_budget(self):
        router = _router(max_loaded_bodies=1)
        route = {
            "prompt_fragment": "MENU", "routing_id": "rt_2",
            "skills": [{"name": "a"}],
        }
        with patch.object(router, "smart_route", return_value=route) as mock_route, \
                patch.object(router, "get_skill_body", side_effect=lambda n, **kw: f"BODY {n}"):
            router.build_prompt_fragment(query="q2")  # fresh — resets
            assert router.load_skill("a").startswith("## Skill: a")

            # Same key again → cache hit (returns before the reset path).
            fragment, routing_id = router.build_prompt_fragment(query="q2")
            assert (fragment, routing_id) == ("MENU", "rt_2")
            assert mock_route.call_count == 1  # proves it was a cache hit

            # Budget still exhausted — the cap guards exactly this window
            # (multi-LLM-call loops within one turn).
            assert "budget exhausted" in router.load_skill("b")


# ── _BodyLoadBudget deadline ─────────────────────────────────────


class TestBodyLoadBudgetDeadline:
    def test_deadline_refusal_after_first_load(self):
        budget = _BodyLoadBudget(max_bodies=5, token_budget=1000, deadline_s=0.0)
        assert budget.check("first") is None  # no first load yet → no deadline
        budget.record("first", 10)
        # Backdate the first load so the deadline has deterministically passed.
        budget._first_load_at = time.monotonic() - 1.0

        refusal = budget.check("another")
        assert refusal is not None
        assert "deadline" in refusal
        assert "budget exhausted" in refusal

        # A repeat is still free of BUDGET — it is not the deadline refusal —
        # but it now returns the "already loaded, answer now" message rather
        # than None, so the caller hands that to the model instead of the body.
        repeat = budget.check("first")
        assert repeat is not None
        assert "already loaded" in repeat
        assert "budget exhausted" not in repeat


# ── inject path guardrail ────────────────────────────────────────


class TestInjectBodyGuardrail:
    def test_bodies_trimmed_counted_and_token_budgeted(self):
        # per-body trim 400 chars → each trimmed body ≈110 tokens (incl.
        # marker); budget 150 fits only the first body.
        router = _router(
            inject_body=True, inject_body_top_k=3,
            per_body_char_limit=400, body_token_budget=150,
        )
        route = {
            "prompt_fragment": "MENU", "routing_id": "rt_9",
            "skills": [{"name": "s1"}, {"name": "s2"}, {"name": "s3"}],
        }
        big = "b" * 2000
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(router, "get_skill_body", return_value=big) as gsb:
            fragment, routing_id = router.build_prompt_fragment(
                query="q", bypass_cache=True,
            )

        assert routing_id == "rt_9"
        assert fragment.startswith("MENU")
        # Injected body is trimmed to the per-body limit with the marker.
        assert TRUNCATION_MARKER in fragment
        assert "b" * 401 not in fragment
        assert "b" * 400 in fragment
        # Token budget: s1 (~110 tok) fits; s2 would blow 150 → dropped,
        # loop stops (s3 never fetched).
        assert "## Skill: s1" in fragment
        assert "## Skill: s2" not in fragment
        assert "## Skill: s3" not in fragment
        assert gsb.call_count == 2  # s1 injected, s2 fetched-then-dropped
        # Server-side trim requested on every fetch.
        for call in gsb.call_args_list:
            assert call.kwargs.get("max_chars") == 400

    def test_inject_count_capped_by_max_loaded_bodies(self):
        # count cap = min(inject_body_top_k, max_loaded_bodies) = 2.
        router = _router(
            inject_body=True, inject_body_top_k=5, max_loaded_bodies=2,
        )
        route = {
            "prompt_fragment": "MENU", "routing_id": "rt_10",
            "skills": [{"name": f"s{i}"} for i in range(1, 5)],
        }
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(
                    router, "get_skill_body",
                    side_effect=lambda n, **kw: f"BODY_{n}",
                ) as gsb:
            fragment, _ = router.build_prompt_fragment(query="q", bypass_cache=True)

        assert "## Skill: s1" in fragment and "## Skill: s2" in fragment
        assert "## Skill: s3" not in fragment and "## Skill: s4" not in fragment
        assert gsb.call_count == 2
