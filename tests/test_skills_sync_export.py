"""Tests for Skills Sync & Export features.

Covers:
- SkillRouter lifecycle: export, status and update_skills as wrappers over the package
  manager (decimalai.skillpkg), fork and use, and search
- discovery and the sync payload: folders you wrote are synced, folders the package
  manager installed never are
- sync_skills in instrument() for openai_agents and langchain
"""

from __future__ import annotations

import hashlib
import os
import shutil
from unittest.mock import patch

import pytest

from decimalai.skill_router import SkillRouter, SkillRouterError
from decimalai.skills import (
    discover_skills,
    sync_to_platform,
    _read_skill_body,
    _hash_content,
)


# ── Fixtures ──────────────────────────────────────────────────

from tests.fakes.fake_registry import FakeRegistry  # noqa: E402
from tests.test_router_disk_methods import engine_on, no_router_http  # noqa: E402


@pytest.fixture
def registry(monkeypatch):
    """A registry the engine talks to, and a router whose own requests fail."""
    reg = FakeRegistry(keys={"test": "org-test"})
    engine_on(reg, monkeypatch)
    no_router_http(monkeypatch)
    return reg


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """A clean project root directory.

    Also redirects HOME (and Windows ``USERPROFILE``) to the same tmp dir, so
    nothing a test runs can read or write the developer's real home.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return str(tmp_path)


@pytest.fixture
def skill_on_disk(tmp_path):
    """Create a SKILL.md on disk and return the project root."""
    skill_dir = tmp_path / ".agents" / "skills" / "code-review"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: code-review\n"
        "description: Reviews code for bugs and security\n---\n\n"
        "# Code Review\nCheck for SQL injection."
    )
    return str(tmp_path)


# ── SkillRouter Lifecycle Methods ────────────────────────────


class TestSkillRouterExport:
    """SkillRouter.export() is `decimalai skills add`."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_export_adds_and_writes_into_the_one_root(self, project_root, registry):
        registry.add_skill("pdf")
        result = self._router().export("pdf", project_root=project_root)
        assert result["skills_written"] == 1
        assert os.path.exists(os.path.join(project_root, ".agents/skills/pdf/SKILL.md"))
        assert not os.path.exists(os.path.join(project_root, ".claude"))

    def test_export_raises_when_the_registry_is_unreachable(self, project_root, registry):
        """A failed export raises: a zero-count summary could pass for "nothing to do"."""
        registry.add_skill("pdf")
        registry.fault("resolve", "connect")
        with pytest.raises(SkillRouterError) as ei:
            self._router().export("pdf", project_root=project_root)
        assert "pdf" in str(ei.value)
        assert "could not reach the registry" in str(ei.value)
        assert not os.path.exists(os.path.join(project_root, ".agents"))


class TestSkillRouterForkAndUse:
    """SkillRouter.fork() and use() are registry calls: neither writes disk."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_fork_is_db_only_no_disk_write(self):
        """fork() forks via the API and must NOT write SKILL.md to disk."""
        router = self._router()
        search_resp = {"items": [{"id": "cat-1", "name": "pdf"}]}
        fork_resp = {"status": "installed", "skill": {"id": "fork-1", "name": "pdf"}}

        def mock_request(method, path, **kwargs):
            if "registry/skills" in path and method == "GET":
                return search_resp
            if path.endswith("/fork"):
                return fork_resp
            raise AssertionError(f"fork() made an unexpected call: {method} {path}")

        with patch.object(router, "export") as mock_export:
            with patch.object(router, "_request", side_effect=mock_request):
                out = router.fork("pdf")

        mock_export.assert_not_called()
        assert out["skill"]["name"] == "pdf"

    def test_use_posts_to_use_endpoint_not_fork(self):
        """use() resolves the registry skill then POSTs /use (a linked pointer, no fork)."""
        router = self._router()
        search_resp = {"items": [{"id": "pub-1", "name": "pdf"}]}
        use_resp = {"status": "using", "uses": [{"scope": "workspace"}], "shadowed_by_owned": False}
        seen = []

        def mock_request(method, path, **kwargs):
            seen.append((method, path))
            if "registry/skills" in path and method == "GET":
                return search_resp
            if path.endswith("/use"):
                return use_resp
            raise AssertionError(f"unexpected call: {method} {path}")

        with patch.object(router, "_request", side_effect=mock_request):
            out = router.use("pdf", scope="workspace", mode="latest")
        assert out["status"] == "using"
        posts = [p for (m, p) in seen if m == "POST"]
        assert posts and posts[0].endswith("/use")
        assert all(not p.endswith("/fork") and not p.endswith("/install") for p in posts)

    def test_use_validates_scope_and_mode(self):
        import pytest as _pytest
        router = self._router()
        with _pytest.raises(ValueError):
            router.use("pdf", scope="bogus")
        with _pytest.raises(ValueError):
            router.use("pdf", mode="bogus")


class TestSkillRouterStatus:
    """SkillRouter.status() is `decimalai skills status` for the project it is given."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def _installed(self, project_root, registry):
        registry.add_skill("code-review")
        router = self._router()
        router.export("code-review", project_root=project_root)
        return router

    def test_status_returns_correct_structure(self, project_root, registry):
        result = self._installed(project_root, registry).status(project_root=project_root)
        for key in ("synced", "modified_locally", "missing_locally", "untracked",
                    "total_on_disk", "total_on_platform", "problems", "result"):
            assert key in result, key
        assert result["synced"] == ["code-review"]
        assert result["modified_locally"] == []

    def test_status_detects_modified_locally(self, project_root, registry):
        router = self._installed(project_root, registry)
        path = os.path.join(project_root, ".agents/skills/code-review/SKILL.md")
        with open(path, "a", encoding="utf-8") as f:
            f.write("\nA local edit.\n")
        result = router.status(project_root=project_root)
        assert result["modified_locally"] == ["code-review"]
        assert result["synced"] == []

    def test_status_detects_untracked(self, project_root, registry):
        router = self._installed(project_root, registry)
        for folder in (".agents/skills/local-only", ".claude/skills/also-mine"):
            d = os.path.join(project_root, folder)
            os.makedirs(d)
            name = os.path.basename(folder)
            with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
                f.write(f"---\nname: {name}\ndescription: Mine\n---\n# Mine\nDo my thing.\n")
        result = router.status(project_root=project_root)
        assert {"local-only", "also-mine"} <= set(result["untracked"])
        assert "code-review" not in result["untracked"]

    def test_status_detects_missing_locally(self, project_root, registry):
        router = self._installed(project_root, registry)
        shutil.rmtree(os.path.join(project_root, ".agents/skills/code-review"))
        result = router.status(project_root=project_root)
        assert result["missing_locally"] == ["code-review"]


class TestSkillRouterUpdateSkills:
    """SkillRouter.update_skills() is `decimalai skills update`."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_update_skips_when_synced(self, project_root, registry):
        registry.add_skill("code-review")
        router = self._router()
        router.export("code-review", project_root=project_root)
        result = router.update_skills(project_root=project_root)
        assert result["updated"] == 0
        assert result["skipped"] > 0

    def test_update_moves_a_latest_entry_to_the_new_version(self, project_root, registry):
        skill = registry.add_skill("code-review")
        router = self._router()
        router.export("code-review", project_root=project_root)
        registry.add_version(skill, "# Code review\n\nVersion two.\n")
        result = router.update_skills(project_root=project_root)
        assert result["updated_skills"] == ["code-review"]
        with open(os.path.join(project_root, ".agents/skills/code-review/SKILL.md"),
                  encoding="utf-8") as f:
            assert "Version two." in f.read()


class TestSkillRouterSearch:
    """Tests for SkillRouter.search()."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_search_sends_correct_params(self):
        router = self._router()
        mock_resp = {"items": [{"name": "pdf", "effectiveness": {"avg_pass_rate": 0.9}}]}

        with patch.object(router, "_request", return_value=mock_resp) as mock_req:
            results = router.search("pdf conversion")
            call_args = mock_req.call_args
            assert call_args[0] == ("GET", "/api/v1/registry/skills")
            params = call_args[1]["params"]
            assert params["q"] == "pdf conversion"
            assert params["sort"] == "effectiveness"
            assert params["limit"] == 20
            assert len(results) == 1

    def test_search_with_filters(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"items": []}) as mock_req:
            router.search(
                "code review",
                category="dev-tools",
                tags=["python", "security"],
                badge="verified",
                sort="installs",
                limit=5,
            )
            params = mock_req.call_args[1]["params"]
            assert params["category"] == "dev-tools"
            assert params["tags"] == "python,security"
            assert params["badge"] == "verified"
            assert params["sort"] == "installs"
            assert params["limit"] == 5

    def test_search_limits_to_100(self):
        router = self._router()
        with patch.object(router, "_request", return_value={"items": []}) as mock_req:
            router.search("test", limit=500)
            assert mock_req.call_args[1]["params"]["limit"] == 100

    def test_search_returns_empty_on_error(self):
        router = self._router()
        with patch.object(router, "_request", side_effect=Exception("Network")):
            results = router.search("test")
            assert results == []

# ── Sync on Install ──────────────────────────────────────────


class TestSyncOnInstrument:
    """Tests that sync_skills runs during instrument().

    Verifies that the instrument() code path includes sync_skills logic.
    We verify structurally by checking the source code contains the
    sync call, since actually calling instrument() requires full SDK init.
    (The behaviour is pinned in tests/test_router_disk_methods.py.)
    """

    def test_openai_agents_instrument_has_sync_code(self):
        """Verify openai_agents.instrument() contains sync_skills logic."""
        import inspect
        from decimalai import openai_agents
        source = inspect.getsource(openai_agents.instrument)
        assert "sync_skills" in source, "instrument() must call sync_skills"
        assert "_sync_skills_background" in source, "sync runs in background"
        assert "_sender.submit" in source, "sync uses background thread"

    def test_langchain_instrument_has_sync_code(self):
        """Verify langchain.instrument() contains sync_skills logic."""
        import inspect
        from decimalai import langchain
        source = inspect.getsource(langchain.instrument)
        assert "sync_skills" in source, "instrument() must call sync_skills"
        assert "_sync_skills_background" in source, "sync runs in background"
        assert "_sender.submit" in source, "sync uses background thread"

    def test_sync_is_non_blocking(self):
        """Verify sync_skills is wrapped in try/except (non-fatal)."""
        import inspect
        from decimalai import openai_agents
        source = inspect.getsource(openai_agents.instrument)
        # The sync block should be wrapped in exception handling
        assert "except Exception" in source, "sync must be non-fatal"

    @pytest.mark.parametrize("module", ["openai_agents", "langchain"])
    def test_sync_up_skips_managed_folders(self, module):
        import importlib
        import inspect

        source = inspect.getsource(importlib.import_module(f"decimalai.{module}").instrument)
        assert "_sync_up_items" in source


# ── Discovery Paths ──────────────────────────────────────────


class TestDiscoveryPaths:
    """Tests for expanded discovery path coverage."""

    def test_discovers_from_claude_skills_dir(self, tmp_path):
        d = tmp_path / ".claude" / "skills" / "test-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: test-skill\ndescription: Test\n---\n\n# Test\nDo the test thing.\n"
        )
        skills = discover_skills([str(tmp_path / ".claude" / "skills")], include_global=False)
        assert len(skills) == 1
        assert skills[0]["name"] == "test-skill"

    def test_discovers_from_agents_skills_dir(self, tmp_path):
        d = tmp_path / ".agents" / "skills" / "my-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: My skill\n---\n\n# My Skill\nDo my thing.\n"
        )
        skills = discover_skills([str(tmp_path / ".agents" / "skills")], include_global=False)
        assert len(skills) == 1

    def test_discovers_from_windsurf_dir(self, tmp_path):
        d = tmp_path / ".windsurf" / "skills" / "wind-skill"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text(
            "---\nname: wind-skill\ndescription: Windsurf\n---\n\n# Wind\nSurf the wind.\n"
        )
        skills = discover_skills([str(tmp_path / ".windsurf" / "skills")], include_global=False)
        assert len(skills) == 1

    def test_cross_agent_deduplication(self, tmp_path):
        """Same skill in .claude/skills and .agents/skills deduplicated."""
        content = "---\nname: shared\ndescription: Shared\n---\n\n# Shared\nUsed by both.\n"
        for d in [".claude/skills/shared", ".agents/skills/shared"]:
            p = tmp_path / d
            p.mkdir(parents=True)
            (p / "SKILL.md").write_text(content)

        skills = discover_skills(
            [str(tmp_path / ".claude" / "skills"), str(tmp_path / ".agents" / "skills")],
            include_global=False,
        )
        assert len(skills) == 1  # Deduplicated

    def test_hash_content_is_deterministic(self):
        body = "# Some Content\nLine 1\nLine 2"
        h1 = _hash_content(body)
        h2 = _hash_content(body)
        assert h1 == h2
        assert h1.startswith("sha256:")
        expected = "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()
        assert h1 == expected


# ── Read Skill Body ──────────────────────────────────────────


class TestReadSkillBody:
    """Tests for _read_skill_body (the sync payload helper)."""

    def test_reads_body_from_skill_dir(self, skill_on_disk):
        body = _read_skill_body(
            os.path.join(skill_on_disk, ".agents/skills/code-review")
        )
        assert body is not None
        assert "SQL injection" in body

    def test_returns_none_for_missing_dir(self):
        body = _read_skill_body("/nonexistent/path")
        assert body is None

    def test_returns_none_for_empty_body(self, tmp_path):
        d = tmp_path / ".agents" / "skills" / "empty"
        d.mkdir(parents=True)
        (d / "SKILL.md").write_text("---\nname: empty\ndescription: Empty\n---\n\n")
        body = _read_skill_body(str(d))
        assert body is None


# ── End-to-End Round-Trip ────────────────────────────────────


class _FakeSyncRouter:
    """Stands in for SkillRouter in sync_to_platform and keeps what it was sent."""

    sent: list = []

    def __init__(self, **kw):
        pass

    def sync_skills(self, skills, **kwargs):
        type(self).sent.extend(skills)
        return {"created": len(skills), "updated": 0, "unchanged": 0}


class TestRoundTrip:
    """Export -> discover -> sync, for an installed folder and for one you wrote."""

    def _export(self, project_root, registry):
        registry.add_skill("code-review", body="# Code Review\n\nCheck for SQL injection.\n")
        SkillRouter(api_key="test", base_url="http://localhost:8000").export(
            "code-review", project_root=project_root)
        return os.path.join(project_root, ".agents", "skills")

    def test_export_then_discover(self, project_root, registry):
        """The folder export writes is discovered as managed, with a readable body."""
        root = self._export(project_root, registry)
        skills = discover_skills([root], include_global=False)
        assert len(skills) == 1
        assert skills[0]["name"] == "code-review"
        assert skills[0]["hash"].startswith("sha256:")
        assert skills[0]["managed"] is True
        assert "SQL injection" in _read_skill_body(skills[0]["source_path"])

    def test_an_exported_folder_is_never_synced(self, project_root, registry, monkeypatch):
        """It is a registry skill: uploading it would copy it into your org."""
        root = self._export(project_root, registry)
        monkeypatch.setattr(_FakeSyncRouter, "sent", [])
        monkeypatch.setattr("decimalai.skill_router.SkillRouter", _FakeSyncRouter)
        monkeypatch.chdir(project_root)
        sync_to_platform("dai_sk_test", search_paths=[root])
        assert _FakeSyncRouter.sent == []

    def test_version_tracking_via_hash(self, project_root):
        """Editing a skill changes its content hash (auto-versioning)."""
        skill_dir = os.path.join(project_root, ".claude/skills/code-review")
        os.makedirs(skill_dir)
        skill_md_path = os.path.join(skill_dir, "SKILL.md")
        with open(skill_md_path, "w") as f:
            f.write("---\nname: code-review\ndescription: Reviews code\n---\n\n# V1\nCheck for bugs.\n")
        s1 = discover_skills([os.path.join(project_root, ".claude/skills")], include_global=False)

        with open(skill_md_path, "w") as f:
            f.write("---\nname: code-review\ndescription: Reviews code\n---\n\n# V2\nCheck for bugs AND security.\n")
        s2 = discover_skills([os.path.join(project_root, ".claude/skills")], include_global=False)

        assert s1[0]["hash"] != s2[0]["hash"], "Hash should change when body changes"

    def test_legacy_ondisk_skill_type_syncs_as_new_taxonomy(self, project_root, monkeypatch):
        """A legacy on-disk `skill-type: convention` (pre-split spelling) is
        discovered/synced as the new (preference, public) pair — older SKILL.md
        files keep working end-to-end."""
        skill_dir = os.path.join(project_root, ".claude/skills/legacy-style")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(
                "---\nname: legacy-style\ndescription: Legacy house style\n"
                "skill-type: convention\n---\n\n"
                "# Legacy Style\nFollow the old house format.\n"
            )

        discovered = discover_skills(
            [os.path.join(project_root, ".claude/skills")], include_global=False,
        )
        assert discovered[0]["skill_type"] == "preference"
        assert discovered[0]["skill_scope"] == "public"

        captured = {}

        class FakeRouter:
            def __init__(self, **kw):
                pass

            def sync_skills(self, skills, **kwargs):
                captured["skills"] = skills
                return {"created": 1, "updated": 0, "unchanged": 0}

        monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)
        monkeypatch.chdir(project_root)  # the sync stamps this project's install identity
        sync_to_platform(
            "dai_sk_test",
            search_paths=[os.path.join(project_root, ".claude/skills")],
        )
        (payload,) = captured["skills"]
        assert payload["skill_type"] == "preference"
        assert payload["skill_scope"] == "public"

    def test_disable_model_invocation_imports_as_user_invocation(self, project_root):
        """A Claude-Code-authored `disable-model-invocation: true` (no explicit
        `invocation` key) imports as invocation=user."""
        skill_dir = os.path.join(project_root, ".claude/skills/on-demand")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write(
                "---\nname: on-demand\ndescription: Only when asked\n"
                "disable-model-invocation: true\n---\n\n"
                "# On Demand\nRun only when explicitly invoked.\n"
            )

        discovered = discover_skills(
            [os.path.join(project_root, ".claude/skills")], include_global=False,
        )
        assert discovered[0]["invocation"] == "user"

    def test_an_unlabeled_skill_gains_no_labels(self, project_root):
        """Absent labels stay absent through discovery: no defaults stamped."""
        skill_dir = os.path.join(project_root, ".claude/skills/plain-skill")
        os.makedirs(skill_dir)
        with open(os.path.join(skill_dir, "SKILL.md"), "w") as f:
            f.write("---\nname: plain-skill\ndescription: No labels\n---\n\n# Plain\nNothing fancy here.\n")

        discovered = discover_skills(
            [os.path.join(project_root, ".claude/skills")], include_global=False,
        )
        assert discovered[0]["skill_type"] is None
        assert discovered[0]["skill_scope"] is None
        assert discovered[0]["invocation"] is None
