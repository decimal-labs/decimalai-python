"""End-to-end user-journey tests for the SDK SkillRouter workflows.

Full lifecycle: sync → menu → route → CRUD → analytics
Version management: create → update → version → pin → compare
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock, call

import pytest

from decimalai.skill_router import SkillRouter


# ═════════════════════════════════════════════════════
# Full SDK lifecycle
# ═════════════════════════════════════════════════════


class TestCUJ_SDK_S1_FullLifecycle:
    """Simulates the complete SDK skill workflow:
    1. Sync local SKILL.md files to platform
    2. Fetch menu for prompt injection
    3. Smart route a user query
    4. Create a new skill via API
    5. Get skill details and body
    6. Update metadata
    7. Delete skill
    8. Verify analytics
    """

    def test_full_sdk_skill_lifecycle(self):
        router = SkillRouter(api_key="dai_sk_test", base_url="http://localhost:8000")

        with patch.object(router, "_request") as mock_req:
            # ── Step 1: Sync skills ──
            mock_req.return_value = {"status": "ok", "created": 3, "updated": 0, "unchanged": 0}
            result = router.sync_skills([
                {"name": "code-review", "description": "Review code", "body_markdown": "# CR"},
                {"name": "test-gen", "description": "Generate tests", "body_markdown": "# TG"},
                {"name": "doc-writer", "description": "Write docs", "body_markdown": "# DW"},
            ], author="sdk-discovery")
            assert result["created"] == 3
            assert mock_req.call_args[0] == ("POST", "/api/v1/skills/sync")
            print("✓ Step 1: Synced 3 skills")

            # ── Step 2: Get menu for prompt injection ──
            mock_req.return_value = {
                "strategy": "full_menu",
                "skill_count": 3,
                "skills": [
                    {"name": "code-review", "description": "Review code"},
                    {"name": "test-gen", "description": "Generate tests"},
                    {"name": "doc-writer", "description": "Write docs"},
                ],
                "prompt_fragment": "## Available Skills\n| code-review | test-gen | doc-writer |",
            }
            menu = router.get_menu(force_refresh=True)
            assert menu["skill_count"] == 3
            prompt = router.get_menu_prompt()
            assert "code-review" in prompt
            print(f"✓ Step 2: Menu fetched, {menu['skill_count']} skills")

            # ── Step 3: Smart route ──
            mock_req.return_value = {
                "strategy": "smart_routing",
                "skills": [{"name": "code-review", "score": 0.95}],
                "prompt_fragment": "## Selected Skills\n- code-review",
            }
            route_result = router.smart_route("review this Python module for security issues")
            assert route_result["strategy"] == "smart_routing"
            print("✓ Step 3: Smart routed → code-review")

            # ── Step 4: Create a new skill ──
            mock_req.return_value = {"skill_id": "new-id-1", "name": "perf-analyzer", "version": 1}
            created = router.create_skill(
                "perf-analyzer", "Analyze performance bottlenecks",
                "# Performance Analyzer\nProfile and optimize hot paths.",
                category="performance",
                trigger_phrases=["slow", "optimize", "performance"],
            )
            assert created["name"] == "perf-analyzer"
            print(f"✓ Step 4: Created perf-analyzer via SDK")

            # ── Step 5: Get skill details + body ──
            mock_req.return_value = {
                "id": "new-id-1", "name": "perf-analyzer",
                "description": "Analyze performance bottlenecks",
                "category": "performance", "is_active": True,
                "latest_version": {"version_number": 1},
            }
            detail = router.get_skill("perf-analyzer")
            assert detail["category"] == "performance"
            print("✓ Step 5: Got skill detail")

            # ── Step 6: List all skills ──
            mock_req.return_value = {
                "skills": [
                    {"name": "code-review"},
                    {"name": "test-gen"},
                    {"name": "doc-writer"},
                    {"name": "perf-analyzer"},
                ],
            }
            all_skills = router.list_skills()
            assert len(all_skills) == 4
            print(f"✓ Step 6: Listed {len(all_skills)} skills")

            # ── Step 7: Update metadata ──
            mock_req.return_value = {"status": "ok", "skill_id": "new-id-1"}
            router.update_skill(
                "new-id-1",
                description="Deep performance analysis with profiling",
                trigger_phrases=["slow", "optimize", "performance", "profile", "bottleneck"],
            )
            print("✓ Step 7: Updated metadata")

            # ── Step 8: Delete skill ──
            mock_req.return_value = {"status": "ok", "deleted": True}
            router.delete_skill("new-id-1")
            print("✓ Step 8: Deleted perf-analyzer")

            # ── Step 9: Analytics ──
            mock_req.return_value = {
                "agent_name": "my-agent",
                "skills": [
                    {"name": "code-review", "activation_count": 42, "pass_rate": 0.92},
                    {"name": "test-gen", "activation_count": 15, "pass_rate": 0.85},
                ],
            }
            metrics = router.get_metrics("my-agent")
            assert len(metrics) >= 2
            print("✓ Step 9: Got per-skill metrics")

            mock_req.return_value = {
                "leaderboard": [
                    {"name": "code-review", "effectiveness_score": 0.91, "trend": "improving"},
                    {"name": "test-gen", "effectiveness_score": 0.78, "trend": "stable"},
                ],
            }
            lb = router.get_leaderboard("my-agent")
            assert lb[0]["name"] == "code-review"
            print(f"✓ Step 9b: Leaderboard #1 = {lb[0]['name']}")

        print("\n" + "=" * 60)
        print("🎉 PASSED: Full SDK Skill Lifecycle")
        print("=" * 60)


# ═════════════════════════════════════════════════════
# Version management
# ═════════════════════════════════════════════════════


class TestCUJ_SDK_S3_VersionManagement:
    """SDK workflow: create skill → update body → list versions →
    get specific version → compare versions → re-embed.
    """

    def test_version_management_workflow(self):
        router = SkillRouter(api_key="test")

        with patch.object(router, "_request") as mock_req:
            # ── Step 1: Create skill ──
            mock_req.return_value = {"skill_id": "sk-1", "name": "code-review", "version": 1}
            router.create_skill("code-review", "Review code", "# V1 Instructions")
            print("✓ Step 1: Created skill v1")

            # ── Step 2: Update body → new version ──
            mock_req.return_value = {"status": "ok", "skill_id": "sk-1", "latest_version_id": "v2"}
            router.update_skill(
                "sk-1",
                body_markdown="# V2 Instructions\nWith security checks",
                change_summary="Added security review steps",
            )
            print("✓ Step 2: Updated body → version 2")

            # ── Step 3: List versions ──
            mock_req.return_value = {
                "skill_name": "code-review",
                "versions": [
                    {"version_number": 2, "content_hash": "v2hash", "change_summary": "Added security review steps"},
                    {"version_number": 1, "content_hash": "v1hash", "change_summary": None},
                ],
            }
            versions = router.list_versions("code-review")
            assert len(versions) == 2
            assert versions[0]["version_number"] == 2
            print(f"✓ Step 3: Listed {len(versions)} versions")

            # ── Step 4: Get specific version body ──
            mock_req.return_value = {
                "skill_name": "code-review",
                "version_number": 1,
                "body_markdown": "# V1 Instructions",
                "content_hash": "v1hash",
            }
            v1 = router.get_version("code-review", 1)
            assert v1["body_markdown"] == "# V1 Instructions"
            print("✓ Step 4: Got v1 body")

            # ── Step 5: Compare versions ──
            mock_req.return_value = {
                "skill_name": "code-review",
                "verdict": "improved",
                "baseline": {"trace_count": 50, "pass_rate": 0.82},
                "candidate": {"trace_count": 30, "pass_rate": 0.91},
                "delta_pass_rate": 0.09,
            }
            comparison = router.compare_versions("code-review", "v1hash", "v2hash")
            assert comparison["verdict"] == "improved"
            assert comparison["delta_pass_rate"] == 0.09
            print(f"✓ Step 5: Version comparison — verdict={comparison['verdict']}, delta={comparison['delta_pass_rate']:.0%}")

            # ── Step 6: Re-embed after model change ──
            mock_req.return_value = {"reembedded": 10, "target_model": "text-embedding-005"}
            result = router.reembed(target_model="text-embedding-005")
            assert result["reembedded"] == 10
            print(f"✓ Step 6: Re-embedded {result['reembedded']} skills with new model")

        print("\n" + "=" * 60)
        print("🎉 PASSED: Version Management")
        print("=" * 60)
