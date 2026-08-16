"""Tests for SkillRouter."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from decimalai.skill_router import SkillRouter
from decimalai._registry_resolve import RESOLVE_LIMIT
from decimalai.evals import TraceData


# ── SkillRouter Tests ────────────────────────────────────────


class TestSkillRouter:
    """Tests for the SkillRouter SDK client."""

    def test_init_defaults(self):
        router = SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")
        assert router.api_key == "dai_sk_test"
        assert router.base_url == "http://localhost:8000"
        assert router.strategy == "auto"
        # The menu cache is keyed by (category, project_id, agent) — starts empty.
        # Starts empty — asserted through the cache's API rather than by
        # comparing it to `{}`. It is no longer a plain dict: its key now
        # includes the run scope, so it needs the bounded TTL+LRU container or
        # it would grow one entry per run forever in a long-lived server.
        assert router._menu_cache.get(("any", None, None, "")) is None

    def test_headers(self):
        router = SkillRouter(api_key="dai_sk_abc")
        headers = router._headers()
        assert headers["Authorization"] == "Bearer dai_sk_abc"
        assert headers["Content-Type"] == "application/json"

    def test_get_menu_caches_result(self):
        router = SkillRouter(api_key="test")
        mock_response = {
            "skills": [{"name": "code-review"}],
            "prompt_fragment": "## Available Skills\n| code-review |",
            "strategy": "full_menu",
        }

        with patch.object(router, "_request", return_value=mock_response) as mock_req:
            # First call hits the API
            result1 = router.get_menu()
            assert result1 == mock_response
            mock_req.assert_called_once()

            # Second call uses cache
            result2 = router.get_menu()
            assert result2 == mock_response
            mock_req.assert_called_once()  # Still once

            # Force refresh hits API again
            result3 = router.get_menu(force_refresh=True)
            assert mock_req.call_count == 2

    def test_get_menu_cache_is_keyed_by_args(self):
        """The menu cache is keyed by (category, project_id, agent), so a
        second get_menu() with different args does NOT return the first's menu."""
        router = SkillRouter(api_key="test")

        def echo(method, path, params=None):
            cat = (params or {}).get("category", "none")
            return {"skills": [{"name": cat}], "prompt_fragment": "", "strategy": "full_menu"}

        with patch.object(router, "_request", side_effect=echo) as mock_req:
            a1 = router.get_menu(category="a")
            b1 = router.get_menu(category="b")
            assert a1["skills"][0]["name"] == "a"
            assert b1["skills"][0]["name"] == "b"  # not the cached 'a' menu
            assert mock_req.call_count == 2

            # Repeating an already-seen arg-tuple is served from cache (no new call).
            a2 = router.get_menu(category="a")
            assert a2 == a1
            assert mock_req.call_count == 2

    def test_get_menu_prompt_uses_smart_route_with_query(self):
        router = SkillRouter(api_key="test", strategy="auto")

        with patch.object(router, "smart_route", return_value={"prompt_fragment": "routed"}) as mock_route:
            result = router.get_menu_prompt(query="review this code")
            assert result == "routed"
            mock_route.assert_called_once()

    def test_get_menu_prompt_full_menu_without_query(self):
        router = SkillRouter(api_key="test")

        with patch.object(router, "get_menu", return_value={"prompt_fragment": "full"}) as mock_menu:
            result = router.get_menu_prompt()
            assert result == "full"
            mock_menu.assert_called_once()

    def test_sync_skills(self):
        router = SkillRouter(api_key="test")
        skills = [{"name": "a", "description": "A", "body_markdown": "# A"}]

        with patch.object(router, "_request", return_value={"status": "ok", "created": 1}) as mock_req:
            result = router.sync_skills(skills, author="test-user")
            # SDK defaults: local_wins (repo IS the source of truth in CI)
            # + summary response mode (no body bytes needed in the response).
            mock_req.assert_called_once_with(
                "POST",
                "/api/v1/skills/sync",
                json={
                    "skills": skills,
                    "conflict_policy": "local_wins",
                    "response_mode": "summary",
                    "author": "test-user",
                },
            )
            assert result["created"] == 1


# ── TraceData active_skills field ────────────────────────────


class TestTraceDataActiveSkills:
    """Tests for the active_skills field on TraceData."""

    def test_default_empty_list(self):
        trace = TraceData(
            id="t1", input="x", output="y", status="success",
        )
        assert trace.active_skills == []

    def test_populated_skills(self):
        trace = TraceData(
            id="t1", input="x", output="y", status="success",
            active_skills=["code-review", "data-analysis"],
        )
        assert "code-review" in trace.active_skills
        assert "data-analysis" in trace.active_skills
        assert len(trace.active_skills) == 2


# ── CRUD Method Tests ────────────────────────────────────────


class TestSkillRouterCRUD:
    """Tests for the CRUD methods added to SkillRouter."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_list_skills(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"skills": [{"name": "a"}]}) as mock:
            result = router.list_skills()
            mock.assert_called_once_with("GET", "/api/v1/skills", params={})
            assert result == [{"name": "a"}]

    def test_list_skills_with_filters(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"skills": []}) as mock:
            router.list_skills(category="dev", stability="stable")
            mock.assert_called_once_with(
                "GET", "/api/v1/skills",
                params={"category": "dev", "stability": "stable"},
            )

    def test_get_skill(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"name": "cr"}) as mock:
            result = router.get_skill("code-review")
            mock.assert_called_once_with("GET", "/api/v1/skills/code-review")
            assert result["name"] == "cr"

    def test_create_skill(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"skill_id": "x"}) as mock:
            result = router.create_skill(
                "test-skill", "A test", "# Body",
                category="dev", trigger_phrases=["test it"],
            )
            call_args = mock.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/v1/skills"
            payload = call_args[1]["json"]
            assert payload["name"] == "test-skill"
            assert payload["category"] == "dev"
            assert payload["trigger_phrases"] == ["test it"]
            assert result["skill_id"] == "x"

    def test_update_skill(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"status": "ok"}) as mock:
            router.update_skill(
                "skill-id-1", description="New desc", is_active=False,
            )
            call_args = mock.call_args
            assert call_args[0][0] == "PUT"
            assert call_args[0][1] == "/api/v1/skills/skill-id-1"
            payload = call_args[1]["json"]
            assert payload["description"] == "New desc"
            assert payload["is_active"] is False

    def test_delete_skill(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"deleted": True}) as mock:
            result = router.delete_skill("skill-id-1")
            mock.assert_called_once_with("DELETE", "/api/v1/skills/skill-id-1")
            assert result["deleted"] is True


class TestSkillRouterVersions:
    """Tests for the version methods."""

    def _router(self):
        return SkillRouter(api_key="test")

    def test_list_versions(self):
        router = self._router()
        mock_versions = [{"version_number": 2}, {"version_number": 1}]
        with patch.object(router, "_request", return_value={"versions": mock_versions}) as mock:
            result = router.list_versions("code-review")
            mock.assert_called_once_with("GET", "/api/v1/skills/code-review/versions")
            assert len(result) == 2

    def test_get_version(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"body_markdown": "# V1"}) as mock:
            result = router.get_version("code-review", 1)
            mock.assert_called_once_with("GET", "/api/v1/skills/code-review/versions/1")
            assert result["body_markdown"] == "# V1"


class TestSkillRouterAnalytics:
    """Tests for the analytics methods."""

    def _router(self):
        return SkillRouter(api_key="test")

    def test_get_metrics(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"skills": [{"name": "a"}]}) as mock:
            result = router.get_metrics("my-agent", skill_name="cr")
            call_args = mock.call_args
            params = call_args[1]["params"]
            assert params["agent_name"] == "my-agent"
            assert params["skill_name"] == "cr"
            assert params["window_days"] == 30
            assert result == [{"name": "a"}]

    def test_compare_versions(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"verdict": "improved"}) as mock:
            result = router.compare_versions("cr", "hash1", "hash2")
            call_args = mock.call_args
            params = call_args[1]["params"]
            assert params["skill_name"] == "cr"
            assert params["baseline_hash"] == "hash1"
            assert params["candidate_hash"] == "hash2"
            assert result["verdict"] == "improved"

    def test_get_leaderboard(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"leaderboard": [{"rank": 1}]}) as mock:
            result = router.get_leaderboard("my-agent", window_days=7)
            call_args = mock.call_args
            params = call_args[1]["params"]
            assert params["agent_name"] == "my-agent"
            assert params["window_days"] == 7
            assert result == [{"rank": 1}]

    def test_reembed(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"reembedded": 5}) as mock:
            result = router.reembed(target_model="text-embedding-005")
            call_args = mock.call_args
            assert call_args[0][0] == "POST"
            assert call_args[0][1] == "/api/v1/skills/reembed"
            assert call_args[1]["json"]["target_model"] == "text-embedding-005"
            assert result["reembedded"] == 5

    # ── preview (read-only ephemeral snapshot, no fork) ───────────

    def test_preview_returns_snapshot_without_forking(self):
        """preview hits the public registry endpoints and returns body+metadata
        without making any install/fork calls."""
        router = self._router()
        search_resp = {"items": [{"id": "sk_pub123", "name": "pdf"}]}
        detail_resp = {
            "id": "sk_pub123",
            "name": "pdf",
            "body_markdown": "# PDF skill\n\nUse this to ...",
            "description": "Work with PDFs",
            "category": "tools",
            "tags": ["pdf", "documents"],
            "skill_badge": "verified",
            "author_display_name": "DecimalAI",
            "source_type": "platform",
            "source_url": None,
            "install_count": 42,
            "effectiveness": {"skill_score": 0.91},
            "attachments": [{"name": "scripts/extract.py"}],
            "latest_version_number": 3,
        }

        with patch.object(router, "_request", side_effect=[search_resp, detail_resp]) as mock_req:
            snap = router.preview("pdf")

        # Two calls — search + detail — and NOTHING ELSE (no install POST)
        assert mock_req.call_count == 2
        assert mock_req.call_args_list[0][0] == ("GET", "/api/v1/registry/skills")
        assert mock_req.call_args_list[0][1]["params"] == {"q": "pdf", "limit": RESOLVE_LIMIT}
        assert mock_req.call_args_list[1][0] == ("GET", "/api/v1/registry/skills/sk_pub123")

        # Shape contract — all advertised keys present
        for key in (
            "name", "skill_id", "body_markdown", "description",
            "category", "tags", "skill_badge",
            "author_display_name", "source_type", "source_url",
            "install_count", "effectiveness", "attachments",
            "latest_version_number",
        ):
            assert key in snap, f"preview snapshot missing key: {key}"

        assert snap["body_markdown"].startswith("# PDF skill")
        assert snap["install_count"] == 42
        assert snap["tags"] == ["pdf", "documents"]
        assert snap["attachments"][0]["name"] == "scripts/extract.py"

    def test_preview_returns_none_when_not_found(self):
        """Empty registry search returns None — no exception, no detail call."""
        router = self._router()
        with patch.object(router, "_request", return_value={"items": []}) as mock_req:
            snap = router.preview("does-not-exist")
        assert snap is None
        assert mock_req.call_count == 1

    def test_preview_handles_search_failure_gracefully(self):
        """Network/auth failures return None rather than propagating."""
        from decimalai.skill_router import SkillRouterError
        router = self._router()
        with patch.object(router, "_request", side_effect=SkillRouterError("boom")):
            snap = router.preview("anything")
        assert snap is None

    def test_preview_does_not_call_install_endpoint(self):
        """Critical contract: preview must NEVER POST to /install — that would fork.
        Guardrail against accidental coupling to install_skill()."""
        router = self._router()
        with patch.object(
            router, "_request",
            side_effect=[
                {"items": [{"id": "sk_pub123", "name": "pdf"}]},
                {"id": "sk_pub123", "name": "pdf", "body_markdown": "# x"},
            ],
        ) as mock_req:
            router.preview("pdf")
        for call in mock_req.call_args_list:
            method, url = call[0][0], call[0][1]
            assert method == "GET", f"preview issued a non-GET: {method} {url}"
            assert url.startswith("/api/v1/registry/skills"), f"unexpected URL: {url}"
            assert "/install" not in url, f"preview must never POST /install — saw {url}"

    def test_pull_skill_alias_warns_and_delegates(self):
        """`pull_skill` is a deprecated alias for `preview`."""
        import warnings as _w
        router = self._router()
        with patch.object(router, "preview", return_value={"name": "pdf"}) as mock_prev:
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                out = router.pull_skill("pdf")
        mock_prev.assert_called_once_with("pdf")
        assert out == {"name": "pdf"}
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
            "pull_skill() should emit DeprecationWarning"


class TestSkillRouterInjectBody:
    """build_prompt_fragment can deliver the routed skill's BODY (the knowledge K), not just
    the menu row — so a benchmarked skill's value actually reaches the agent at runtime."""

    def _router(self, **kw):
        return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", **kw)

    def test_inject_body_appends_routed_skill_body(self):
        router = self._router(inject_body=True)
        route = {"prompt_fragment": "MENU", "routing_id": "rt_1",
                 "skills": [{"name": "code-review"}, {"name": "other"}]}
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(router, "get_skill_body", return_value="BODY_K") as gsb:
            fragment, routing_id = router.build_prompt_fragment(query="review my PR")
        assert "MENU" in fragment            # the menu still there
        assert "## Skill: code-review" in fragment
        assert "BODY_K" in fragment          # the actual K is now in the prompt
        assert routing_id == "rt_1"
        # top-1 only by default; body guardrail asks the server to trim.
        gsb.assert_called_once_with(
            "code-review", max_chars=router.per_body_char_limit,
        )

    def test_no_inject_by_default(self):
        router = self._router()  # inject_body defaults False — unchanged menu-only behavior
        route = {"prompt_fragment": "MENU", "routing_id": "rt_1", "skills": [{"name": "code-review"}]}
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(router, "get_skill_body", return_value="BODY_K") as gsb:
            fragment, _ = router.build_prompt_fragment(query="review my PR")
        assert fragment == "MENU"
        gsb.assert_not_called()

    def test_no_inject_in_full_menu_mode(self):
        # query=None → full menu (not relevance-ranked) → never inject an arbitrary skill's body.
        router = self._router(inject_body=True)
        menu = {"prompt_fragment": "MENU", "routing_id": "rt_1", "skills": [{"name": "a"}]}
        with patch.object(router, "get_menu", return_value=menu), \
                patch.object(router, "get_skill_body", return_value="BODY_K") as gsb:
            fragment, _ = router.build_prompt_fragment(query=None)
        assert fragment == "MENU"
        gsb.assert_not_called()

    def test_inject_top_k(self):
        router = self._router(inject_body=True, inject_body_top_k=2)
        route = {"prompt_fragment": "MENU", "routing_id": "rt_1",
                 "skills": [{"name": "s1"}, {"name": "s2"}, {"name": "s3"}]}
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(router, "get_skill_body", side_effect=lambda n, **kw: f"BODY_{n}") as gsb:
            fragment, _ = router.build_prompt_fragment(query="q")
        assert "BODY_s1" in fragment and "BODY_s2" in fragment
        assert "BODY_s3" not in fragment
        assert gsb.call_count == 2


class TestDeliveredNamesRail:
    """Activation ladder: body injection emits 'delivered' — the names
    whose full body actually reached the prompt. Menu-only mode emits none
    (a bare menu row is offered, never delivered)."""

    def _router(self, **kw):
        return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000", **kw)

    @pytest.fixture(autouse=True)
    def _fresh_router_ctx(self):
        from decimalai import skill_router as sr
        sr._last_offered_names_ctx.set(None)
        sr._last_delivered_names_ctx.set(None)
        sr._body_budget_ctx.set(None)
        yield
        sr._last_offered_names_ctx.set(None)
        sr._last_delivered_names_ctx.set(None)
        sr._body_budget_ctx.set(None)

    def test_inject_body_emits_delivered_names(self):
        from decimalai.skill_router import consume_last_delivered_names
        router = self._router(inject_body=True, inject_body_top_k=2)
        route = {"prompt_fragment": "MENU", "routing_id": "rt_1",
                 "skills": [{"name": "s1"}, {"name": "s2"}, {"name": "s3"}]}
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(router, "get_skill_body", side_effect=lambda n, **kw: f"BODY_{n}"):
            router.build_prompt_fragment(query="q")
        assert consume_last_delivered_names() == ["s1", "s2"]
        # Consume drains — a second read must be empty (no leak into the
        # next trace).
        assert consume_last_delivered_names() == []

    def test_menu_only_mode_emits_no_delivered(self):
        from decimalai.skill_router import (
            consume_last_delivered_names,
            consume_last_offered_names,
        )
        router = self._router()  # inject_body defaults False
        route = {"prompt_fragment": "MENU", "routing_id": "rt_1",
                 "skills": [{"name": "s1"}]}
        with patch.object(router, "smart_route", return_value=route):
            router.build_prompt_fragment(query="q")
        assert consume_last_offered_names() == ["s1"]
        assert consume_last_delivered_names() == []

    def test_missing_body_is_not_delivered(self):
        from decimalai.skill_router import consume_last_delivered_names
        router = self._router(inject_body=True, inject_body_top_k=2)
        route = {"prompt_fragment": "MENU", "routing_id": "rt_1",
                 "skills": [{"name": "gone"}, {"name": "here"}]}
        bodies = {"here": "BODY_here"}
        with patch.object(router, "smart_route", return_value=route), \
                patch.object(router, "get_skill_body", side_effect=lambda n, **kw: bodies.get(n)):
            fragment, _ = router.build_prompt_fragment(query="q")
        assert "BODY_here" in fragment
        assert consume_last_delivered_names() == ["here"]

    def test_cache_hit_reemits_offered_and_delivered(self):
        from decimalai.skill_router import (
            consume_last_delivered_names,
            consume_last_offered_names,
        )
        router = self._router(inject_body=True)
        route = {"prompt_fragment": "MENU", "routing_id": "rt_1",
                 "skills": [{"name": "s1"}]}
        with patch.object(router, "smart_route", return_value=route) as mock_route, \
                patch.object(router, "get_skill_body", return_value="B"):
            router.build_prompt_fragment(query="q")
            # First turn drains both rails (adapter behavior).
            assert consume_last_offered_names() == ["s1"]
            assert consume_last_delivered_names() == ["s1"]

            # Second turn hits the 30s cache — both rails re-populate so
            # the second trace doesn't lose the names.
            router.build_prompt_fragment(query="q")
            assert mock_route.call_count == 1  # proves it was a cache hit
            assert consume_last_offered_names() == ["s1"]
            assert consume_last_delivered_names() == ["s1"]

    def test_unconsumed_delivered_names_do_not_leak_across_calls(self):
        # Activation-ladder regression: a body-injecting call whose rails are
        # never drained (anthropic adapter / generic quickstart) must not
        # leak its delivered names onto a later no-body call — a
        # langchain/openai_agents drain after call 2 would otherwise stamp
        # call 1's skill as 'delivered' on a trace that never saw it.
        from decimalai.skill_router import (
            consume_last_delivered_names,
            consume_last_offered_names,
        )
        router = self._router(inject_body=True)
        route1 = {"prompt_fragment": "MENU", "routing_id": "rt_1",
                  "skills": [{"name": "sA"}]}
        route2 = {"prompt_fragment": "MENU2", "routing_id": "rt_2",
                  "skills": []}
        with patch.object(router, "smart_route", side_effect=[route1, route2]), \
                patch.object(router, "get_skill_body", side_effect=lambda n, **kw: "B" if n == "sA" else None):
            router.build_prompt_fragment(query="q1")  # delivers sA, never consumed
            router.build_prompt_fragment(query="q2")  # different query → cache miss, no skills
        assert consume_last_offered_names() == []
        assert consume_last_delivered_names() == []

    def test_cache_hit_with_empty_rails_clears_stale_names(self):
        # Same leak via the cache-hit path: a cached no-body result must
        # CLEAR stale unconsumed names, not leave them in place.
        from decimalai.skill_router import consume_last_delivered_names
        router = self._router(inject_body=True)
        route_body = {"prompt_fragment": "MENU", "routing_id": "rt_1",
                      "skills": [{"name": "sA"}]}
        route_none = {"prompt_fragment": "MENU2", "routing_id": "rt_2",
                      "skills": []}
        with patch.object(router, "smart_route", side_effect=[route_none, route_body]), \
                patch.object(router, "get_skill_body", return_value="B"):
            router.build_prompt_fragment(query="empty")   # caches empty rails
            router.build_prompt_fragment(query="body")    # delivers sA, unconsumed
            router.build_prompt_fragment(query="empty")   # cache HIT with empty rails
        assert consume_last_delivered_names() == []



class TestRequestErrorDetail:
    """_request surfaces the server's structured error body from the safety gate.

    A publish-gate refusal returns HTTP 400 with a FastAPI ``detail`` dict carrying
    ``reason``/``message``/``findings`` — SDK users need it for remediation, not just
    the status code. These patch ``httpx.request`` so the real _request parsing runs.
    """

    def _router(self):
        return SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")

    def _resp(self, status, body, *, as_json=True):
        m = MagicMock()
        m.status_code = status
        if as_json:
            m.json.return_value = body
        else:
            m.json.side_effect = json.JSONDecodeError("no json", "", 0)
            m.text = body
        return m

    def test_structured_gate_detail_exposed(self):
        from decimalai.skill_router import SkillRouterError
        router = self._router()
        detail = {
            "reason": "safety_blocked",
            "message": "Cannot publish — the safety scan found 1 critical issue.",
            "findings": [{"check": "live_secret", "line": 4, "severity": "critical"}],
        }
        with patch("httpx.request", return_value=self._resp(400, {"detail": detail})):
            with pytest.raises(SkillRouterError) as ei:
                router._request("POST", "/api/v1/skills/x/publish")
        err = ei.value
        assert err.status_code == 400
        assert err.detail == detail
        assert err.detail["reason"] == "safety_blocked"
        assert err.detail["findings"][0]["check"] == "live_secret"
        # the reason/message are folded into the exception text for humans
        assert "safety_blocked" in str(err)

    def test_string_detail_exposed(self):
        from decimalai.skill_router import SkillRouterError
        router = self._router()
        with patch("httpx.request", return_value=self._resp(409, {"detail": "name taken"})):
            with pytest.raises(SkillRouterError) as ei:
                router._request("POST", "/api/v1/skills/x/publish")
        assert ei.value.detail == "name taken"
        assert "name taken" in str(ei.value)

    def test_non_json_error_leaves_detail_none(self):
        from decimalai.skill_router import SkillRouterError
        router = self._router()
        with patch("httpx.request", return_value=self._resp(500, "<html>502</html>", as_json=False)):
            with pytest.raises(SkillRouterError) as ei:
                router._request("GET", "/api/v1/skills")
        assert ei.value.status_code == 500
        assert ei.value.detail is None


# ── Exact-name resolution (the semantic-search substitution bug) ──────────
#
# `q=` is a SEMANTIC search over the whole registry — it always ranks something.
# Resolving a name by taking items[0] meant a typo or a retired name silently
# fetched an unrelated skill and reported success:
#   `decimalai skills pull totally-bogus-skill-xyz123` → "✓ Pulled agent-skill-format"
# For a SKILL.md — instructions the agent loads and follows — that is
# install-time typosquatting done by our own client, so every resolver must
# match the name exactly or fail.

class TestExactNameResolution:
    """fork / use / preview must never resolve to a skill the caller didn't name."""

    def _router(self):
        return SkillRouter(api_key="dai_sk_test", base_url="https://api.test")

    # Search hit for a name that does NOT exist — semantic neighbours only.
    NEIGHBOURS = {"items": [
        {"id": "sk_a", "name": "agent-skill-format"},
        {"id": "sk_b", "name": "minimax-pdf"},
    ]}

    def test_fork_raises_instead_of_forking_a_neighbour(self):
        router = self._router()
        with patch.object(router, "_request", return_value=self.NEIGHBOURS) as mock_req:
            with pytest.raises(ValueError) as ei:
                router.fork("totally-bogus-skill-xyz123")
        # Never reached the POST /fork — search only.
        assert mock_req.call_count == 1
        assert "totally-bogus-skill-xyz123" in str(ei.value)
        # The near-misses are named so the caller sees what it *would* have taken.
        assert "agent-skill-format" in str(ei.value)

    def test_use_raises_instead_of_recording_a_neighbour(self):
        router = self._router()
        with patch.object(router, "_request", return_value=self.NEIGHBOURS) as mock_req:
            with pytest.raises(ValueError):
                router.use("pdf-procesing")   # one-character typo of a real skill
        assert mock_req.call_count == 1       # no POST /use — telemetry stays clean

    def test_preview_returns_none_instead_of_a_neighbour_body(self):
        router = self._router()
        with patch.object(router, "_request", return_value=self.NEIGHBOURS) as mock_req:
            assert router.preview("pdf-procesing") is None
        assert mock_req.call_count == 1       # no detail fetch

    def test_exact_match_still_resolves_when_not_ranked_first(self):
        """The literal name can rank below a longer 'more relevant' one."""
        router = self._router()
        search = {"items": [
            {"id": "sk_long", "name": "pdf-processing-pro"},
            {"id": "sk_exact", "name": "pdf"},
        ]}
        detail = {"id": "sk_exact", "name": "pdf", "body_markdown": "# PDF"}
        with patch.object(router, "_request", side_effect=[search, detail]):
            snap = router.preview("pdf")
        assert snap is not None and snap["name"] == "pdf"

    def test_case_insensitive_match_accepted_when_unambiguous(self):
        """Registry names aren't all lowercase slugs (`Excel / XLSX`)."""
        router = self._router()
        search = {"items": [{"id": "sk_x", "name": "StatsPAI_skill"}]}
        detail = {"id": "sk_x", "name": "StatsPAI_skill", "body_markdown": "# s"}
        with patch.object(router, "_request", side_effect=[search, detail]):
            snap = router.preview("statspai_skill")
        assert snap is not None and snap["name"] == "StatsPAI_skill"

    def test_ambiguous_case_fold_is_not_a_match(self):
        """Two names differing only in case — guessing is the bug, so refuse."""
        router = self._router()
        search = {"items": [
            {"id": "sk_1", "name": "Report"},
            {"id": "sk_2", "name": "report"},
        ]}
        # 'REPORT' folds onto both; neither is exact.
        with patch.object(router, "_request", return_value=search):
            assert router.preview("REPORT") is None

    def test_empty_search_still_raises(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"items": []}):
            with pytest.raises(ValueError):
                router.fork("nothing-at-all")
