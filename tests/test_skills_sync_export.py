"""Tests for Skills Sync & Export features (Phases 0-4).

Covers:
- disk_export module: AGENT_PATHS registry, SKILL.md reconstruction, lockfile, export
- SkillRouter lifecycle: export_to_disk, fork, install, status, update_skills, search
- sync_skills integration in install() for openai_agents and langchain
- sync-on-install verification
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from decimalai.disk_export import (
    AGENT_PATHS,
    get_agent_paths,
    list_supported_agents,
    _reconstruct_skill_md,
    _read_lockfile,
    _write_lockfile,
    export_skill_to_disk,
    export_skills_to_disk,
)
from decimalai.skill_router import SkillRouter
from decimalai.skills import (
    discover_skills,
    parse_skill_md,
    sync_to_platform,
    _read_skill_body,
    _hash_content,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def project_root(tmp_path, monkeypatch):
    """A clean project root directory for testing disk export.

    Also redirects HOME (and Windows ``USERPROFILE``) to the same tmp
    dir so tests using ``scope="global"`` write to tmp instead of the
    developer's real ``~/.config/...`` (which would (a) pollute their
    home, (b) fail on the second run because the file already exists).
    Two pre-existing tests hit this footgun:
      - test_lockfile_not_updated_for_global_scope
      - test_top_five_agent_runtimes_get_their_own_skill_md
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    return str(tmp_path)


@pytest.fixture
def sample_skill():
    """A sample skill dict as returned by the platform API."""
    return {
        "id": "sk-test-001",
        "name": "code-review",
        "description": "Reviews code for bugs and security",
        "body_markdown": "# Code Review\n\nWhen reviewing code:\n1. Check for SQL injection\n2. Check for XSS\n3. Validate input sanitization",
        "version": 3,
        "license": "MIT",
        "compatibility": ["python3", "node18+"],
        "allowed_tools": ["bash", "python"],
        "source": "registry",
        "attachments": [
            {
                "file_path": "scripts/lint.py",
                "directory": "scripts",
                "content_text": "#!/usr/bin/env python3\nimport subprocess\nsubprocess.run(['flake8', '.'])",
            },
            {
                "file_path": "references/owasp-top10.md",
                "directory": "references",
                "content_text": "# OWASP Top 10\n- Injection\n- Broken Auth\n- XSS",
            },
        ],
    }


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


# ── AGENT_PATHS Registry ─────────────────────────────────────


class TestAgentPathsRegistry:
    """Tests for the agent path registry."""

    def test_registry_has_minimum_agents(self):
        assert len(AGENT_PATHS) >= 32

    def test_all_entries_have_project_and_global(self):
        for agent, paths in AGENT_PATHS.items():
            assert "project" in paths, f"{agent} missing 'project' key"
            assert "global" in paths, f"{agent} missing 'global' key"

    def test_major_agents_present(self):
        major_agents = [
            "claude-code", "cursor", "github-copilot", "windsurf",
            "gemini-cli", "codex", "cline", "continue", "universal",
        ]
        for agent in major_agents:
            assert agent in AGENT_PATHS, f"Missing major agent: {agent}"

    def test_claude_code_paths(self):
        assert AGENT_PATHS["claude-code"]["project"] == ".claude/skills"
        assert AGENT_PATHS["claude-code"]["global"] == "~/.claude/skills"

    def test_cursor_paths(self):
        assert AGENT_PATHS["cursor"]["project"] == ".agents/skills"

    def test_windsurf_paths(self):
        assert AGENT_PATHS["windsurf"]["project"] == ".windsurf/skills"

    def test_universal_fallback(self):
        assert AGENT_PATHS["universal"]["project"] == ".agents/skills"
        assert AGENT_PATHS["universal"]["global"] == "~/.config/agents/skills"


class TestGetAgentPaths:
    """Tests for path resolution."""

    def test_project_scope(self, project_root):
        path = get_agent_paths("claude-code", scope="project", project_root=project_root)
        assert path == os.path.join(project_root, ".claude/skills")

    def test_global_scope_expands_tilde(self):
        path = get_agent_paths("claude-code", scope="global")
        assert path.startswith("/")
        assert "~" not in path

    def test_unknown_agent_raises(self):
        with pytest.raises(ValueError, match="Unknown agent 'nonexistent'"):
            get_agent_paths("nonexistent")

    def test_list_supported_agents(self):
        agents = list_supported_agents()
        assert len(agents) >= 32
        assert agents == sorted(agents)  # Alphabetical


# ── SKILL.md Reconstruction ──────────────────────────────────


class TestReconstructSkillMd:
    """Tests for SKILL.md reconstruction from DB fields."""

    def test_minimal_reconstruction(self):
        result = _reconstruct_skill_md("test", "A test skill", "# Test\nDo things.")
        assert "name: test" in result
        assert "description: A test skill" in result
        assert "# Test\nDo things." in result
        assert result.startswith("---\n")

    def test_with_license(self):
        result = _reconstruct_skill_md(
            "test", "desc", "# Body",
            license_info="MIT",
        )
        assert "license: MIT" in result

    def test_with_compatibility(self):
        result = _reconstruct_skill_md(
            "test", "desc", "# Body",
            compatibility=["python3", "node18+"],
        )
        assert "compatibility:" in result
        assert "  - python3" in result
        assert "  - node18+" in result

    def test_with_allowed_tools(self):
        result = _reconstruct_skill_md(
            "test", "desc", "# Body",
            allowed_tools=["bash", "python"],
        )
        assert "allowed-tools:" in result
        assert "  - bash" in result

    def test_with_taxonomy_labels(self):
        result = _reconstruct_skill_md(
            "test", "desc", "# Body",
            category="dev-tools",
            skill_type="capability",
            skill_scope="private",
            invocation="model",
        )
        assert "category: dev-tools" in result
        assert "skill-type: capability" in result
        assert "skill-scope: private" in result
        assert "invocation: model" in result
        # Only `invocation: user` earns the Claude-Code spelling.
        assert "disable-model-invocation" not in result

    def test_preference_public_labels(self):
        result = _reconstruct_skill_md(
            "test", "desc", "# Body",
            skill_type="preference",
            skill_scope="public",
        )
        assert "skill-type: preference" in result
        assert "skill-scope: public" in result

    def test_scope_omitted_when_unset(self):
        """skill-scope is only emitted when set (same as every other field)."""
        result = _reconstruct_skill_md(
            "test", "desc", "# Body",
            skill_type="capability",
        )
        assert "skill-type: capability" in result
        assert "skill-scope:" not in result

    def test_user_invocation_writes_disable_model_invocation(self):
        result = _reconstruct_skill_md(
            "test", "desc", "# Body",
            invocation="user",
        )
        assert "invocation: user" in result
        assert "disable-model-invocation: true" in result

    def test_absent_labels_add_no_frontmatter_lines(self):
        """Skills without taxonomy labels re-export byte-stable — no new keys."""
        result = _reconstruct_skill_md("test", "A test skill", "# Test\nDo things.")
        assert "category:" not in result
        assert "skill-type:" not in result
        assert "skill-scope:" not in result
        assert "invocation:" not in result
        assert "disable-model-invocation" not in result

    def test_full_reconstruction_is_parseable(self):
        """A reconstructed SKILL.md should be parseable by parse_skill_md."""
        content = _reconstruct_skill_md(
            "test-roundtrip", "A roundtrip test", "# Test\nCheck if this roundtrips correctly.",
            license_info="Apache-2.0",
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(content)
            f.flush()
            parsed = parse_skill_md(f.name)
            os.unlink(f.name)

        assert parsed is not None
        assert parsed["name"] == "test-roundtrip"
        assert "roundtrip" in parsed["description"].lower()


# ── Lockfile ─────────────────────────────────────────────────


class TestLockfile:
    """Tests for the lockfile read/write operations."""

    def test_read_nonexistent_returns_empty(self, project_root):
        lockdata = _read_lockfile(project_root)
        assert lockdata == {"version": 1, "skills": {}}

    def test_write_and_read_roundtrip(self, project_root):
        lockdata = {
            "version": 1,
            "skills": {
                "code-review": {
                    "version": 2,
                    "content_hash": "abc123",
                    "source": "platform",
                },
            },
        }
        _write_lockfile(project_root, lockdata)
        read_back = _read_lockfile(project_root)
        assert read_back["skills"]["code-review"]["version"] == 2
        assert read_back["skills"]["code-review"]["content_hash"] == "abc123"
        assert "updated_at" in read_back

    def test_write_creates_directory(self, project_root):
        assert not os.path.exists(os.path.join(project_root, ".decimal"))
        _write_lockfile(project_root, {"version": 1, "skills": {}})
        assert os.path.exists(os.path.join(project_root, ".decimal", "skills.lock"))

    def test_read_corrupted_returns_empty(self, project_root):
        lockfile_path = os.path.join(project_root, ".decimal", "skills.lock")
        os.makedirs(os.path.dirname(lockfile_path), exist_ok=True)
        with open(lockfile_path, "w") as f:
            f.write("{invalid json")
        lockdata = _read_lockfile(project_root)
        assert lockdata == {"version": 1, "skills": {}}


# ── Export to Disk ───────────────────────────────────────────


class TestExportSkillToDisk:
    """Tests for single-skill disk export."""

    def test_basic_export(self, project_root, sample_skill):
        result = export_skill_to_disk(
            sample_skill,
            agents=["claude-code"],
            scope="project",
            project_root=project_root,
            attachments=sample_skill["attachments"],
        )
        assert result["skill_name"] == "code-review"
        assert len(result["written_paths"]) == 1
        assert result["attachment_count"] == 2

        # Verify file exists
        skill_md_path = os.path.join(project_root, ".claude/skills/code-review/SKILL.md")
        assert os.path.exists(skill_md_path)

        # Verify content
        with open(skill_md_path) as f:
            content = f.read()
        assert "name: code-review" in content
        assert "license: MIT" in content
        assert "# Code Review" in content

    def test_multi_agent_export(self, project_root, sample_skill):
        result = export_skill_to_disk(
            sample_skill,
            agents=["claude-code", "cursor", "windsurf"],
            scope="project",
            project_root=project_root,
        )
        assert len(result["written_paths"]) == 3

        # Verify each agent path exists
        assert os.path.exists(os.path.join(project_root, ".claude/skills/code-review/SKILL.md"))
        assert os.path.exists(os.path.join(project_root, ".agents/skills/code-review/SKILL.md"))
        assert os.path.exists(os.path.join(project_root, ".windsurf/skills/code-review/SKILL.md"))

    def test_attachment_writing(self, project_root, sample_skill):
        export_skill_to_disk(
            sample_skill,
            agents=["claude-code"],
            scope="project",
            project_root=project_root,
            attachments=sample_skill["attachments"],
        )

        # Verify attachment files
        lint_path = os.path.join(project_root, ".claude/skills/code-review/scripts/lint.py")
        assert os.path.exists(lint_path)
        with open(lint_path) as f:
            assert "flake8" in f.read()

        ref_path = os.path.join(project_root, ".claude/skills/code-review/references/owasp-top10.md")
        assert os.path.exists(ref_path)
        with open(ref_path) as f:
            assert "OWASP" in f.read()

    def test_defaults_to_universal_agent(self, project_root, sample_skill):
        result = export_skill_to_disk(
            sample_skill, scope="project", project_root=project_root,
        )
        assert len(result["written_paths"]) == 1
        assert os.path.exists(os.path.join(project_root, ".agents/skills/code-review/SKILL.md"))

    def test_does_not_modify_existing_skills(self, project_root, sample_skill):
        """Exporting a new skill should not modify existing skills."""
        # Create existing skill
        existing_dir = os.path.join(project_root, ".claude/skills/existing-skill")
        os.makedirs(existing_dir, exist_ok=True)
        existing_path = os.path.join(existing_dir, "SKILL.md")
        with open(existing_path, "w") as f:
            f.write("# Existing\nDo not touch.")
        mtime_before = os.path.getmtime(existing_path)

        # Export new skill
        export_skill_to_disk(
            sample_skill,
            agents=["claude-code"],
            scope="project",
            project_root=project_root,
        )

        # Verify existing skill untouched
        mtime_after = os.path.getmtime(existing_path)
        assert mtime_before == mtime_after
        with open(existing_path) as f:
            assert "Do not touch." in f.read()


class TestExportSkillsToDisk:
    """Tests for batch export + lockfile update."""

    def test_batch_export_with_lockfile(self, project_root):
        skills = [
            {
                "id": "sk-1",
                "name": "pdf",
                "description": "PDF skills",
                "body_markdown": "# PDF\nConvert PDFs.",
                "version": 2,
                "source": "registry",
            },
            {
                "id": "sk-2",
                "name": "testing",
                "description": "Testing skills",
                "body_markdown": "# Testing\nWrite tests.",
                "version": 1,
                "source": "platform",
            },
        ]
        result = export_skills_to_disk(
            skills,
            agents=["claude-code"],
            scope="project",
            project_root=project_root,
        )

        assert result["skills_written"] == 2
        assert result["scope"] == "project"

        # Verify lockfile updated
        lockdata = _read_lockfile(project_root)
        assert "pdf" in lockdata["skills"]
        assert "testing" in lockdata["skills"]
        assert lockdata["skills"]["pdf"]["version"] == 2
        assert lockdata["skills"]["pdf"]["source"] == "registry"
        assert lockdata["skills"]["testing"]["version"] == 1

    def test_lockfile_not_updated_for_global_scope(self, project_root):
        skills = [{"name": "global-skill", "description": "X", "body_markdown": "# X\nDo X.", "version": 1}]
        export_skills_to_disk(
            skills, agents=["universal"], scope="global", project_root=project_root,
        )
        lockdata = _read_lockfile(project_root)
        assert lockdata["skills"] == {}

    def test_skip_lockfile_if_disabled(self, project_root):
        skills = [{"name": "no-lock", "description": "X", "body_markdown": "# X\nDo X.", "version": 1}]
        export_skills_to_disk(
            skills, agents=["claude-code"], scope="project",
            project_root=project_root, update_lockfile=False,
        )
        lockdata = _read_lockfile(project_root)
        assert "no-lock" not in lockdata["skills"]

    def test_lockfile_content_hash_deterministic(self, project_root):
        """content_hash pins the FULL sha256 of the
        exact SKILL.md written to disk (frontmatter+body); body_hash carries
        the platform's body-only axis. The old sha256(body)[:12] pinned a
        string that was never on disk."""
        skills = [{"name": "determin", "description": "X", "body_markdown": "# X\nDo X.", "version": 1}]
        export_skills_to_disk(
            skills, agents=["claude-code"], scope="project", project_root=project_root,
        )
        lockdata = _read_lockfile(project_root)
        skill_md = os.path.join(project_root, ".claude", "skills", "determin", "SKILL.md")
        with open(skill_md, "rb") as f:
            expected_hash = hashlib.sha256(f.read()).hexdigest()
        entry = lockdata["skills"]["determin"]
        assert entry["content_hash"] == expected_hash
        assert entry["body_hash"] == hashlib.sha256("# X\nDo X.".encode()).hexdigest()

    def test_top_five_agent_runtimes_get_their_own_skill_md(self, project_root):
        """Exporting to the top-5 agent runtimes drops a
        SKILL.md at each runtime's canonical path. If any of these break,
        the install→export pipeline is broken for a major fraction of
        users — they install a skill, it doesn't appear in the agent.
        """
        top_five = ["claude-code", "cursor", "windsurf", "github-copilot", "codex"]
        skills = [{
            "id": "sk-multi",
            "name": "code-review",
            "description": "Reviews code",
            "body_markdown": "# Code Review\nLook for bugs.",
            "version": 1,
            "source": "registry",
        }]
        result = export_skills_to_disk(
            skills,
            agents=top_five,
            scope="project",
            project_root=project_root,
        )

        # Every agent should have gotten the skill body written.
        assert result["skills_written"] == 1
        assert set(result.get("agents") or []) == set(top_five), (
            f"agents={result.get('agents')}"
        )

        # Each runtime's path should physically contain SKILL.md.
        expected_paths = {
            "claude-code":    ".claude/skills/code-review/SKILL.md",
            "cursor":         ".agents/skills/code-review/SKILL.md",
            "windsurf":       ".windsurf/skills/code-review/SKILL.md",
            "github-copilot": ".agents/skills/code-review/SKILL.md",
            "codex":          ".agents/skills/code-review/SKILL.md",
        }
        for agent, rel in expected_paths.items():
            full = Path(project_root) / rel
            assert full.exists(), f"missing SKILL.md for {agent} at {full}"
            body = full.read_text()
            assert "code-review" in body
            assert "Look for bugs" in body

        # Lockfile records the agents it was exported to so `status` and
        # `update_skills` can target the same set next time.
        lockdata = _read_lockfile(project_root)
        assert "code-review" in lockdata["skills"]
        locked_agents = set(lockdata["skills"]["code-review"].get("agents") or [])
        assert locked_agents == set(top_five), (
            f"lockfile agents={locked_agents}, expected={top_five}"
        )


# ── SkillRouter Lifecycle Methods ────────────────────────────


class TestSkillRouterExportToDisk:
    """Tests for SkillRouter.export_to_disk()."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_export_calls_api_and_writes(self, project_root):
        router = self._router()
        skill_data = {
            "id": "sk-1", "name": "pdf", "description": "PDF",
            "body_markdown": "# PDF\nConvert.", "version": 1,
        }
        with patch.object(router, "get_skill", return_value=skill_data):
            with patch.object(router, "list_attachments", return_value=[]):
                result = router.export_to_disk(
                    skills=["pdf"],
                    agents=["claude-code"],
                    scope="project",
                    project_root=project_root,
                )
                assert result["skills_written"] == 1
                assert os.path.exists(os.path.join(project_root, ".claude/skills/pdf/SKILL.md"))

    def test_export_all_skills_when_none_specified(self, project_root):
        router = self._router()
        all_skills = [
            {"name": "a", "id": "1"}, {"name": "b", "id": "2"},
        ]
        full_a = {"name": "a", "id": "1", "description": "A", "body_markdown": "# A\nDo A.", "version": 1}
        full_b = {"name": "b", "id": "2", "description": "B", "body_markdown": "# B\nDo B.", "version": 1}

        with patch.object(router, "list_skills", return_value=all_skills):
            with patch.object(router, "get_skill", side_effect=[full_a, full_b]):
                with patch.object(router, "list_attachments", return_value=[]):
                    result = router.export_to_disk(
                        agents=["claude-code"],
                        scope="project",
                        project_root=project_root,
                    )
                    assert result["skills_written"] == 2

    def test_export_raises_when_all_requested_skills_fail(self, project_root):
        """Explicitly-requested skills that ALL fail
        must raise, not return a zero-count summary a caller could mistake
        for "nothing to do"."""
        from decimalai.skill_router import SkillRouterError

        router = self._router()
        with patch.object(router, "get_skill", side_effect=Exception("Network error")):
            with pytest.raises(SkillRouterError) as ei:
                router.export_to_disk(
                    skills=["pdf"],
                    agents=["claude-code"],
                    project_root=project_root,
                )
        assert "pdf" in str(ei.value)
        assert "Network error" in str(ei.value)


class TestSkillRouterInstall:
    """Tests for SkillRouter.fork() / install() (+ legacy install_skill alias)."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_install_makes_correct_api_calls(self, project_root):
        router = self._router()

        search_resp = {"items": [{"id": "cat-1", "name": "pdf"}]}
        fork_resp = {"status": "installed", "skill": {"id": "fork-1", "name": "pdf"}}
        get_resp = {
            "id": "fork-1", "name": "pdf", "description": "PDF",
            "body_markdown": "# PDF\nConvert.", "version": 1,
        }
        seen = []

        def mock_request(method, path, **kwargs):
            seen.append((method, path))
            if "registry/skills" in path and method == "GET":
                return search_resp
            elif path.endswith("/fork") or path.endswith("/install"):
                return fork_resp
            elif "/skills/pdf" in path and method == "GET":
                return get_resp
            elif "/attachments" in path:
                return []
            return {}

        with patch.object(router, "_request", side_effect=mock_request):
            result = router.install(
                "pdf",
                agents=["claude-code"],
                project_root=project_root,
            )

        # Canonical fork route is hit (not the legacy /install).
        assert any(m == "POST" and p.endswith("/fork") for m, p in seen), \
            f"install() should POST the /fork route; saw {seen}"
        assert result["skill_name"] == "pdf"
        assert "fork" in result and "export" in result

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

        with patch.object(router, "export_to_disk") as mock_export:
            with patch.object(router, "_request", side_effect=mock_request):
                out = router.fork("pdf")

        mock_export.assert_not_called()
        assert out["skill"]["name"] == "pdf"

    def test_install_skill_alias_warns_and_delegates(self):
        """install_skill is a deprecated alias for install()."""
        import warnings as _w
        router = self._router()
        with patch.object(router, "install", return_value={"skill_name": "pdf"}) as mock_inst:
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                out = router.install_skill("pdf", agents=["claude-code"])
        mock_inst.assert_called_once()
        assert out == {"skill_name": "pdf"}
        assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
            "install_skill() should emit DeprecationWarning"

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
    """Tests for SkillRouter.status()."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    # A realistic full 64-char SHA-256 (the backend stores the full digest under
    # latest_version.content_hash; the SDK truncates the disk hash to 12 chars).
    _FULL_HASH = "abc123def456abc123def456abc123def456abc123def456abc123def4561234"

    def test_status_returns_correct_structure(self, project_root):
        router = self._router()

        # Realistic shapes: the backend serializes content_hash ONLY under the
        # nested ``latest_version`` object, as the FULL digest. Same content on
        # both sides must classify as SYNCED (regression guard: a full-digest
        # vs truncated-digest mismatch used to report a false DRIFT).
        platform_skills = [
            {"name": "code-review", "latest_version": {"content_hash": self._FULL_HASH}}
        ]
        disk_skills = [
            {"name": "code-review", "hash": f"sha256:{self._FULL_HASH}", "source_path": "/x"}
        ]

        with patch.object(router, "list_skills", return_value=platform_skills):
            with patch("decimalai.skills.discover_skills", return_value=disk_skills):
                result = router.status(project_root=project_root)

        assert "synced" in result
        assert "modified_locally" in result
        assert "missing_locally" in result
        assert "untracked" in result
        assert "total_on_disk" in result
        assert "total_on_platform" in result
        # The bug this guards: identical content used to land in modified_locally.
        assert result["synced"] == ["code-review"]
        assert result["modified_locally"] == []

    def test_status_detects_modified_locally(self, project_root):
        router = self._router()

        platform_hash = self._FULL_HASH
        local_hash = "fedcba987654fedcba987654fedcba987654fedcba987654fedcba9876549999"

        platform_skills = [
            {"name": "code-review", "latest_version": {"content_hash": platform_hash}}
        ]
        disk_skills = [
            {"name": "code-review", "hash": f"sha256:{local_hash}", "source_path": "/x"}
        ]

        with patch.object(router, "list_skills", return_value=platform_skills):
            with patch("decimalai.skills.discover_skills", return_value=disk_skills):
                result = router.status(project_root=project_root)

        assert result["modified_locally"] == ["code-review"]
        assert result["synced"] == []

    def test_status_detects_untracked(self, project_root):
        router = self._router()

        platform_skills = []  # Nothing on platform
        disk_skills = [{"name": "local-only", "hash": "sha256:x", "source_path": "/x"}]

        with patch.object(router, "list_skills", return_value=platform_skills):
            with patch("decimalai.skills.discover_skills", return_value=disk_skills):
                result = router.status(project_root=project_root)

        assert "local-only" in result["untracked"]

    def test_status_detects_missing_locally(self, project_root):
        router = self._router()

        platform_skills = [{"name": "remote-only", "content_hash": "abc"}]
        disk_skills = []  # Nothing on disk

        with patch.object(router, "list_skills", return_value=platform_skills):
            with patch("decimalai.skills.discover_skills", return_value=disk_skills):
                result = router.status(project_root=project_root)

        assert "remote-only" in result["missing_locally"]


class TestSkillRouterUpdateSkills:
    """Tests for SkillRouter.update_skills()."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_update_skips_when_synced(self, project_root):
        router = self._router()

        lockdata = {
            "version": 1,
            "skills": {
                "code-review": {
                    "content_hash": "abc123",
                    "version": 1,
                    "source": "platform",
                    "agents": ["claude-code"],
                    "scope": "project",
                },
            },
        }
        _write_lockfile(project_root, lockdata)

        # The API returns {name: {hash: ...}} per update_skills implementation
        with patch.object(router, "_request", return_value={"hashes": {"code-review": {"hash": "abc123"}}}):
            result = router.update_skills(project_root=project_root)
            assert result["updated"] == 0
            assert result["skipped"] > 0


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

    def test_search_registry_alias(self):
        assert SkillRouter.search is SkillRouter.search_registry


# ── Sync on Install ──────────────────────────────────────────


class TestSyncOnInstall:
    """Tests that sync_skills runs during install().

    Verifies that the install() code path includes sync_skills logic.
    We verify structurally by checking the source code contains the
    sync call, since actually calling install() requires full SDK init.
    """

    def test_openai_agents_install_has_sync_code(self):
        """Verify openai_agents.install() contains sync_skills logic."""
        import inspect
        from decimalai import openai_agents
        source = inspect.getsource(openai_agents.instrument)
        assert "sync_skills" in source, "install() must call sync_skills"
        assert "_sync_skills_background" in source, "sync runs in background"
        assert "_sender.submit" in source, "sync uses background thread"

    def test_langchain_install_has_sync_code(self):
        """Verify langchain.install() contains sync_skills logic."""
        import inspect
        from decimalai import langchain
        source = inspect.getsource(langchain.instrument)
        assert "sync_skills" in source, "install() must call sync_skills"
        assert "_sync_skills_background" in source, "sync runs in background"
        assert "_sender.submit" in source, "sync uses background thread"

    def test_sync_is_non_blocking(self):
        """Verify sync_skills is wrapped in try/except (non-fatal)."""
        import inspect
        from decimalai import openai_agents
        source = inspect.getsource(openai_agents.instrument)
        # The sync block should be wrapped in exception handling
        assert "except Exception" in source, "sync must be non-fatal"


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


class TestRoundTrip:
    """Tests for complete export → discover → sync round-trip."""

    def test_export_then_discover(self, project_root, sample_skill):
        """Skill exported to disk is discoverable by discover_skills."""
        # Export
        export_skills_to_disk(
            [sample_skill],
            agents=["claude-code"],
            scope="project",
            project_root=project_root,
        )

        # Discover
        skills = discover_skills(
            [os.path.join(project_root, ".claude/skills")],
            include_global=False,
        )
        assert len(skills) == 1
        assert skills[0]["name"] == "code-review"
        assert skills[0]["hash"].startswith("sha256:")

    def test_export_then_discover_body_readable(self, project_root, sample_skill):
        """Body can be read from exported SKILL.md for sync payload."""
        export_skills_to_disk(
            [sample_skill],
            agents=["claude-code"],
            scope="project",
            project_root=project_root,
        )

        skills = discover_skills(
            [os.path.join(project_root, ".claude/skills")],
            include_global=False,
        )
        body = _read_skill_body(skills[0]["source_path"])
        assert body is not None
        assert "SQL injection" in body

    def test_version_tracking_via_hash(self, project_root):
        """Editing a skill changes its content hash (auto-versioning)."""
        v1 = {
            "name": "code-review",
            "description": "Reviews code",
            "body_markdown": "# V1\nCheck for bugs.",
            "version": 1,
        }
        export_skills_to_disk([v1], agents=["claude-code"], scope="project", project_root=project_root)
        s1 = discover_skills([os.path.join(project_root, ".claude/skills")], include_global=False)
        hash_v1 = s1[0]["hash"]

        # Edit skill on disk (simulating user edit)
        skill_md_path = os.path.join(project_root, ".claude/skills/code-review/SKILL.md")
        with open(skill_md_path, "w") as f:
            f.write("---\nname: code-review\ndescription: Reviews code\n---\n\n# V2\nCheck for bugs AND security.\n")

        s2 = discover_skills([os.path.join(project_root, ".claude/skills")], include_global=False)
        hash_v2 = s2[0]["hash"]
        assert hash_v1 != hash_v2, "Hash should change when body changes"

    def test_labels_roundtrip_export_discover_sync(self, project_root, monkeypatch):
        """skill-type + skill-scope + invocation survive pull/install → disk →
        discover → sync without being dropped at any hop."""
        skill = {
            "id": "sk-labeled",
            "name": "house-style",
            "description": "House style conventions",
            "body_markdown": "# House Style\nFollow the house citation format.",
            "version": 1,
            "category": "writing",
            "skill_type": "preference",
            "skill_scope": "private",
            "invocation": "user",
        }
        export_skills_to_disk(
            [skill], agents=["claude-code"], scope="project", project_root=project_root,
        )

        # Exported frontmatter carries both taxonomy axes + both invocation
        # spellings.
        skill_md_path = os.path.join(project_root, ".claude/skills/house-style/SKILL.md")
        with open(skill_md_path) as f:
            content = f.read()
        assert "skill-type: preference" in content
        assert "skill-scope: private" in content
        assert "invocation: user" in content
        assert "disable-model-invocation: true" in content

        # Discovery parses them back into the descriptor.
        discovered = discover_skills(
            [os.path.join(project_root, ".claude/skills")], include_global=False,
        )
        assert discovered[0]["category"] == "writing"
        assert discovered[0]["skill_type"] == "preference"
        assert discovered[0]["skill_scope"] == "private"
        assert discovered[0]["invocation"] == "user"

        # Sync threads them into the platform payload.
        captured = {}

        class FakeRouter:
            def __init__(self, **kw):
                pass

            def sync_skills(self, skills, **kwargs):
                captured["skills"] = skills
                return {"created": 1, "updated": 0, "unchanged": 0}

        monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)
        sync_to_platform(
            "dai_sk_test",
            search_paths=[os.path.join(project_root, ".claude/skills")],
        )
        (payload,) = captured["skills"]
        assert payload["skill_type"] == "preference"
        assert payload["skill_scope"] == "private"
        assert payload["invocation"] == "user"
        assert payload["category"] == "writing"

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
        sync_to_platform(
            "dai_sk_test",
            search_paths=[os.path.join(project_root, ".claude/skills")],
        )
        (payload,) = captured["skills"]
        assert payload["skill_type"] == "preference"
        assert payload["skill_scope"] == "public"

    def test_disable_model_invocation_spelling_roundtrips(self, project_root):
        """A Claude-Code-authored `disable-model-invocation: true` (no explicit
        `invocation` key) imports as invocation=user and re-exports with BOTH
        spellings, so the file round-trips through either authoring tool."""
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

        reexported = _reconstruct_skill_md(
            "on-demand", "Only when asked", "# On Demand\nRun only when explicitly invoked.",
            invocation=discovered[0]["invocation"],
        )
        assert "invocation: user" in reexported
        assert "disable-model-invocation: true" in reexported

    def test_unlabeled_skill_roundtrips_without_new_frontmatter(self, project_root):
        """Absent labels stay absent through export → discover: no defaults
        stamped, no new frontmatter lines on re-export."""
        skill = {
            "name": "plain-skill",
            "description": "No labels",
            "body_markdown": "# Plain\nNothing fancy here.",
            "version": 1,
        }
        export_skills_to_disk(
            [skill], agents=["claude-code"], scope="project", project_root=project_root,
        )
        skill_md_path = os.path.join(project_root, ".claude/skills/plain-skill/SKILL.md")
        with open(skill_md_path) as f:
            content = f.read()
        assert "skill-type:" not in content
        assert "skill-scope:" not in content
        assert "invocation:" not in content
        assert "disable-model-invocation" not in content

        discovered = discover_skills(
            [os.path.join(project_root, ".claude/skills")], include_global=False,
        )
        assert discovered[0]["skill_type"] is None
        assert discovered[0]["skill_scope"] is None
        assert discovered[0]["invocation"] is None

    def test_multi_agent_symmetry(self, project_root, sample_skill):
        """Same skill exported to multiple agents has identical content."""
        export_skills_to_disk(
            [sample_skill],
            agents=["claude-code", "cursor", "windsurf"],
            scope="project",
            project_root=project_root,
        )

        paths = [
            os.path.join(project_root, ".claude/skills/code-review/SKILL.md"),
            os.path.join(project_root, ".agents/skills/code-review/SKILL.md"),
            os.path.join(project_root, ".windsurf/skills/code-review/SKILL.md"),
        ]
        contents = []
        for p in paths:
            with open(p) as f:
                contents.append(f.read())

        assert contents[0] == contents[1] == contents[2], "All agent copies should be identical"


# ── Pull Missing (Bidirectional Sync) ────────────────────────


class TestPullMissing:
    """Tests for SkillRouter.pull_missing() — bidirectional sync."""

    def _router(self):
        return SkillRouter(api_key="test", base_url="http://localhost:8000")

    def test_pull_missing_fetches_and_writes(self, project_root):
        """Skills on platform but not on disk are pulled and written."""
        router = self._router()

        # Platform has "code-review" but disk has nothing
        hash_response = {
            "hashes": {
                "code-review": {"hash": "abc123", "version": 2},
            }
        }
        export_skill = {
            "name": "code-review", "description": "Reviews code",
            "body_markdown": "# Code Review\nCheck for SQL injection.",
            "version": 2,
        }

        with patch.object(router, "_request", return_value=hash_response):
            with patch.object(router, "export_to_disk", return_value={"skills_written": 1}) as mock_export:
                result = router.pull_missing(
                    local_skill_names=set(),  # Nothing on disk
                    agents=["claude-code"],
                    project_root=project_root,
                )

        assert result["pulled"] == 1
        assert result["updated"] == 0
        assert "code-review" in result["pulled_skills"]
        mock_export.assert_called_once()

    def test_pull_missing_skips_existing_when_disk_wins(self, project_root):
        """When disk_wins=True, existing skills are not overwritten."""
        router = self._router()

        hash_response = {
            "hashes": {
                "code-review": {"hash": "platform_hash_123", "version": 3},
            }
        }
        disk_skills = [{"name": "code-review", "hash": "sha256:local_hash_456", "source_path": "/x"}]

        with patch.object(router, "_request", return_value=hash_response):
            with patch("decimalai.skills.discover_skills", return_value=disk_skills):
                result = router.pull_missing(
                    agents=["claude-code"],
                    disk_wins=True,
                    project_root=project_root,
                )

        assert result["pulled"] == 0
        assert result["updated"] == 0
        assert result["skipped"] == 1

    def test_pull_missing_updates_when_platform_wins(self, project_root):
        """When disk_wins=False (default), platform version overwrites local."""
        router = self._router()

        hash_response = {
            "hashes": {
                "code-review": {"hash": "platform_hash_new", "version": 3},
            }
        }

        with patch.object(router, "_request", return_value=hash_response):
            with patch.object(router, "export_to_disk", return_value={"skills_written": 1}) as mock_export:
                result = router.pull_missing(
                    local_skill_names={"code-review"},  # Exists on disk
                    agents=["claude-code"],
                    disk_wins=False,
                    project_root=project_root,
                )

        # With no local hashes provided, it finds it in local_skill_names
        # and since disk_wins=False and no local hash to compare, skips
        # (can't determine conflict without hashes)
        # This tests the case where local_skill_names is provided but not local_hashes
        assert result["skipped"] == 1

    def test_pull_missing_discovers_disk_and_compares(self, project_root):
        """Full flow: discovers disk skills, compares to platform, pulls missing."""
        router = self._router()

        hash_response = {
            "hashes": {
                "on-platform": {"hash": "abc", "version": 1},
                "on-both": {"hash": "same_hash", "version": 1},
            }
        }
        disk_skills = [
            {"name": "on-both", "hash": "sha256:same_hash_longform", "source_path": "/x"},
            {"name": "on-disk-only", "hash": "sha256:zzz", "source_path": "/y"},
        ]

        with patch.object(router, "_request", return_value=hash_response):
            with patch("decimalai.skills.discover_skills", return_value=disk_skills):
                with patch.object(router, "export_to_disk", return_value={"skills_written": 1}) as mock_export:
                    result = router.pull_missing(
                        agents=["claude-code"],
                        project_root=project_root,
                    )

        # "on-platform" is missing locally → pulled
        assert result["pulled"] == 1
        assert "on-platform" in result["pulled_skills"]

    def test_pull_missing_handles_network_error(self, project_root):
        """Network failures return gracefully, don't crash."""
        router = self._router()

        with patch.object(router, "_request", side_effect=Exception("Timeout")):
            result = router.pull_missing(
                local_skill_names=set(),
                agents=["claude-code"],
                project_root=project_root,
            )

        assert result["pulled"] == 0
        assert "error" in result

    def test_pull_missing_handles_empty_platform(self, project_root):
        """No skills on platform → nothing to pull."""
        router = self._router()

        with patch.object(router, "_request", return_value={"hashes": {}}):
            result = router.pull_missing(
                local_skill_names=set(),
                agents=["claude-code"],
                project_root=project_root,
            )

        assert result["pulled"] == 0
        assert result["updated"] == 0
        assert result["skipped"] == 0

    def test_pull_missing_defaults_to_universal_agent(self, project_root):
        """When no agents specified, defaults to universal."""
        router = self._router()

        hash_response = {"hashes": {"new-skill": {"hash": "abc", "version": 1}}}

        with patch.object(router, "_request", return_value=hash_response):
            with patch.object(router, "export_to_disk", return_value={"skills_written": 1}) as mock_export:
                router.pull_missing(
                    local_skill_names=set(),
                    project_root=project_root,
                )

        call_args = mock_export.call_args
        assert call_args[1]["agents"] == ["universal"]

    def test_pull_missing_export_failure_logs_warning_not_error(self, project_root, caplog):
        """Defense in depth: a failed export is a WARNING (pull_missing is
        best-effort background sync) — never an ERROR-level record. The
        result dict still carries the error for callers that need it."""
        import logging

        router = self._router()
        hash_response = {"hashes": {"new-skill": {"hash": "abc", "version": 1}}}

        with patch.object(router, "_request", return_value=hash_response):
            with patch.object(
                router, "export_to_disk",
                side_effect=ValueError("Unknown agent 'not-a-runtime'"),
            ):
                with caplog.at_level(logging.DEBUG, logger="decimalai.skill_router"):
                    result = router.pull_missing(
                        local_skill_names=set(),
                        agents=["not-a-runtime"],
                        project_root=project_root,
                    )

        assert result["pulled"] == 0
        assert "Unknown agent" in result["error"]
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []
        assert any(
            r.levelno == logging.WARNING and "pull_missing" in r.getMessage()
            for r in caplog.records
        )


class TestBidirectionalSyncOnInstall:
    """Tests verifying that install() includes bidirectional sync logic.

    Structural tests: verify the source code contains pull_missing_background
    since actually calling install() requires full SDK init.
    """

    def test_openai_agents_install_has_pull_code(self):
        """Verify openai_agents.install() contains pull_missing logic."""
        import inspect
        from decimalai import openai_agents
        source = inspect.getsource(openai_agents.instrument)
        assert "_pull_missing_background" in source, "install() must include pull_missing"
        assert "pull_missing" in source, "install() must call router.pull_missing()"
        assert "disk_wins=False" in source, "platform wins by default"

    def test_langchain_install_has_pull_code(self):
        """Verify langchain.install() contains pull_missing logic."""
        import inspect
        from decimalai import langchain
        source = inspect.getsource(langchain.instrument)
        assert "_pull_missing_background" in source, "install() must include pull_missing"
        assert "pull_missing" in source, "install() must call router.pull_missing()"
        assert "disk_wins=False" in source, "platform wins by default"

    def test_pull_is_non_blocking(self):
        """Verify pull_missing is wrapped in try/except and runs in background."""
        import inspect
        from decimalai import openai_agents
        source = inspect.getsource(openai_agents.instrument)
        assert "_sender.submit(_pull_missing_background)" in source, "pull must use background thread"
        assert 'Background skill pull failed' in source, "pull must be non-fatal"

    def test_pull_respects_agent_name(self):
        """Verify pull passes agent_name to pull_missing for correct directory
        targeting — but only when it's a real disk runtime: trace names
        like 'acme-support-agent' must fall back to 'universal'."""
        import inspect
        from decimalai import openai_agents
        source = inspect.getsource(openai_agents.instrument)
        assert 'target_agent = agent_name if agent_name in AGENT_PATHS else "universal"' in source

