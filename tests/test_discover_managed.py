"""Discovery reads the folders `decimalai skills add` writes, and still reads hand-authored ones.

A managed folder is ``.agents/skills/<install-name>/SKILL.md`` in format 1: the frontmatter
``name`` is the install name, and ``metadata`` carries the registry's stamp
(``decimalai-skill-id``, ``-version-id``, ``-body-sha256``, ``-name`` and the taxonomy). The
fixtures below are rendered by ``skillevaluation.skillfmt``, the same code the registry
renders with, so these tests read the exact bytes an install writes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from skillevaluation import skillfmt

from decimalai import skills as sk
from decimalai.skillpkg.migration import V1_AGENT_DIRS
from decimalai.skills import (
    DEFAULT_SKILL_PATHS,
    discover_skills,
    parse_skill_md,
    sync_to_platform,
)

SKILL_ID = "6f1c0000-0000-0000-0000-000000000001"
VERSION_ID = "9b2e0000-0000-0000-0000-000000000003"
BODY = (
    "# Acme refund policy\n\n"
    "Refund window is thirty calendar days from delivery.\n"
    "Partial refunds require a supervisor approval code.\n"
)


def _render(install_name="acme-refund-policy", *, skill_id=SKILL_ID, version_id=VERSION_ID,
            registry_name="acme/refund-policy", description="Apply Acme refund rules.",
            **extra):
    fields = skillfmt.RenderFields(
        skill_id=skill_id, version_id=version_id, version=3, registry_name=registry_name,
        body=BODY, description=description, **extra,
    )
    return skillfmt.render_skill_md(fields, install_name=install_name, variant="foreign")


def _install(root: Path, install_name="acme-refund-policy", **kw) -> Path:
    d = root / install_name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(_render(install_name, **kw), encoding="utf-8")
    return d


def _hand_authored(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: hand-authored {name}\n---\n"
        f"# {name}\nA skill somebody wrote by hand, long enough to read.\n",
        encoding="utf-8",
    )
    return d


def _lock(project: Path, version=2) -> None:
    (project / ".decimal").mkdir(parents=True, exist_ok=True)
    (project / ".decimal" / "skills.lock").write_text(
        json.dumps({"version": version, "root": ".agents/skills", "skills": {}}) + "\n",
        encoding="utf-8",
    )


# ── the stamp ────────────────────────────────────────────────


class TestStamp:
    def test_stamp_fields_and_the_managed_flag_are_set(self, tmp_path):
        d = _install(tmp_path)
        got = parse_skill_md(str(d / "SKILL.md"))
        assert got["name"] == "acme-refund-policy"
        assert got["managed"] is True
        assert got["registry_name"] == "acme/refund-policy"
        assert got["skill_id"] == SKILL_ID
        assert got["version_id"] == VERSION_ID
        assert got["body_sha256"] == skillfmt.body_sha256(BODY)
        assert got["description"] == "Apply Acme refund rules."

    def test_a_hand_authored_folder_is_not_managed(self, tmp_path):
        got = parse_skill_md(str(_hand_authored(tmp_path, "mine") / "SKILL.md"))
        assert got["managed"] is False
        assert got["registry_name"] is None
        assert got["skill_id"] is None
        assert got["version_id"] is None
        assert got["body_sha256"] is None

    def test_taxonomy_is_read_from_metadata(self, tmp_path):
        d = _install(tmp_path, category="support", skill_type="capability",
                     skill_scope="public", invocation_mode="user")
        text = (d / "SKILL.md").read_text(encoding="utf-8")
        assert "\ncategory:" not in text and "decimalai-category" in text  # the format's place
        got = parse_skill_md(str(d / "SKILL.md"))
        assert got["category"] == "support"
        assert got["skill_type"] == "capability"
        assert got["skill_scope"] == "public"
        assert got["invocation"] == "user"

    def test_a_model_invoked_install_reads_as_model(self, tmp_path):
        got = parse_skill_md(str(_install(tmp_path) / "SKILL.md"))
        assert got["invocation"] == "model"

    def test_a_quoted_description_with_escapes_round_trips(self, tmp_path):
        """The renderer quotes every description, escaping quotes, backslashes and the
        characters YAML 1.1 would read as line breaks. The reader decodes them back."""
        tricky = 'Say "refund" \\ not credit then stop'
        got = parse_skill_md(str(_install(tmp_path, description=tricky) / "SKILL.md"))
        assert got["description"] == tricky

    def test_the_body_and_its_hash_are_the_body_that_was_written(self, tmp_path):
        got = parse_skill_md(str(_install(tmp_path) / "SKILL.md"))
        assert got["body"] == BODY.strip()
        assert got["hash"] == sk._hash_content(BODY.strip())


# ── where discovery looks ────────────────────────────────────


class TestSearchPaths:
    def test_default_paths_cover_every_agent_project_dir(self):
        """Every project folder a 0.14 SDK wrote skills to is one discovery reads."""
        project_dirs = set(V1_AGENT_DIRS.values())
        assert project_dirs <= set(DEFAULT_SKILL_PATHS), sorted(
            project_dirs - set(DEFAULT_SKILL_PATHS)
        )

    def test_default_paths_resolve_from_the_project_root(self, tmp_path, monkeypatch):
        """Run from a subdirectory, discovery still reads the project's skill dirs."""
        (tmp_path / ".git").mkdir()
        _hand_authored(tmp_path / ".claude" / "skills", "root-skill")
        deep = tmp_path / "src" / "app"
        deep.mkdir(parents=True)
        monkeypatch.chdir(deep)
        assert {s["name"] for s in discover_skills()} == {"root-skill"}

    def test_a_subdirectorys_own_skill_dirs_are_still_read(self, tmp_path, monkeypatch):
        """Resolving from the project root must not lose what the working directory has
        always contributed (a package with its own .claude/skills inside a monorepo)."""
        (tmp_path / ".git").mkdir()
        _hand_authored(tmp_path / ".claude" / "skills", "root-skill")
        pkg = tmp_path / "packages" / "billing"
        _hand_authored(pkg / ".claude" / "skills", "package-skill")
        monkeypatch.chdir(pkg)
        assert {s["name"] for s in discover_skills()} == {"root-skill", "package-skill"}

    def test_the_working_directorys_skill_wins_a_name_shared_with_the_root(
        self, tmp_path, monkeypatch
    ):
        """The nearer folder wins: a package's own `deploy` is the one its scripts load,
        and before discovery read the root, the package's was the only one read."""
        (tmp_path / ".git").mkdir()
        _hand_authored(tmp_path / ".claude" / "skills", "deploy")
        pkg = tmp_path / "packages" / "api"
        mine = _hand_authored(pkg / ".claude" / "skills", "deploy")
        monkeypatch.chdir(pkg)
        (got,) = discover_skills()
        assert os.path.realpath(got["source_path"]) == os.path.realpath(mine)

    def test_at_the_project_root_the_default_paths_stay_relative(self, tmp_path, monkeypatch):
        """A descriptor's source_path reaches the sync payload; run from the root, it is
        relative as it always was, so no home directory travels with it."""
        (tmp_path / ".git").mkdir()
        _hand_authored(tmp_path / ".claude" / "skills", "root-skill")
        monkeypatch.chdir(tmp_path)
        (got,) = discover_skills()
        assert got["source_path"] == os.path.join(".claude", "skills", "root-skill")

    def test_an_explicit_project_root_is_read_instead_of_the_cwd(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        _hand_authored(project / ".agents" / "skills", "in-project")
        elsewhere = tmp_path / "elsewhere"
        _hand_authored(elsewhere / ".agents" / "skills", "in-cwd")
        monkeypatch.chdir(elsewhere)
        assert {s["name"] for s in discover_skills(project_root=str(project))} == {
            "in-project"
        }

    def test_the_managed_root_is_included_with_explicit_search_paths(
        self, tmp_path, monkeypatch
    ):
        """An installed project's .agents/skills is read even when the caller names other
        directories, so prompt-presence detection knows the skills the agent loads."""
        project = tmp_path / "project"
        _lock(project)
        _install(project / ".agents" / "skills")
        other = tmp_path / "authored"
        _hand_authored(other, "mine")
        monkeypatch.chdir(project)
        got = {s["name"]: s for s in discover_skills([str(other)])}
        assert set(got) == {"mine", "acme-refund-policy"}
        assert got["acme-refund-policy"]["managed"] is True

    def test_the_managed_root_is_found_from_a_subdirectory(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        _lock(project)
        _install(project / ".agents" / "skills")
        deep = project / "src"
        deep.mkdir()
        other = tmp_path / "authored"
        other.mkdir()
        monkeypatch.chdir(deep)
        assert {s["name"] for s in discover_skills([str(other)])} == {"acme-refund-policy"}

    def test_the_managed_root_is_not_listed_twice(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        _lock(project)
        _install(project / ".agents" / "skills")
        monkeypatch.chdir(project)
        got = discover_skills([".agents/skills"])
        assert [s["name"] for s in got] == ["acme-refund-policy"]

    @pytest.mark.parametrize("lock", ["v1", "none", "corrupt"])
    def test_no_version_2_lock_means_no_managed_root(self, tmp_path, monkeypatch, lock):
        """Without a version 2 lock nothing is installed, so an explicit search stays
        exactly what the caller asked for."""
        project = tmp_path / "project"
        _install(project / ".agents" / "skills")
        if lock == "v1":
            _lock(project, version=1)
        elif lock == "corrupt":
            (project / ".decimal").mkdir()
            (project / ".decimal" / "skills.lock").write_text("{not json", encoding="utf-8")
        other = tmp_path / "authored"
        _hand_authored(other, "mine")
        monkeypatch.chdir(project)
        assert {s["name"] for s in discover_skills([str(other)])} == {"mine"}

    def test_a_symlinked_managed_root_is_not_followed(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        _lock(project)
        real = tmp_path / "real-skills"
        _install(real)
        (project / ".agents").mkdir()
        (project / ".agents" / "skills").symlink_to(real, target_is_directory=True)
        other = tmp_path / "authored"
        _hand_authored(other, "mine")
        monkeypatch.chdir(project)
        assert {s["name"] for s in discover_skills([str(other)])} == {"mine"}


class TestRepositoryBoundary:
    """The project search stops at the repository (the first directory with a ``.git``
    entry), so a folder of repositories with a ``.decimal/`` of its own above them is
    never read as the project of a repository inside it."""

    @staticmethod
    def _outer(tmp_path: Path, lock_version: int) -> Path:
        outer = tmp_path / "outer"
        _lock(outer, version=lock_version)
        _hand_authored(outer / ".agents" / "skills", "x")
        repo = outer / "repo"
        repo.mkdir()
        (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/repo\n")  # a worktree
        pkg = repo / "pkg"
        pkg.mkdir()
        return pkg

    def test_default_paths_do_not_read_a_stray_project_above_the_repo(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(self._outer(tmp_path, lock_version=1))
        assert "x" not in {s["name"] for s in discover_skills()}

    def test_the_managed_root_is_not_taken_from_above_the_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(self._outer(tmp_path, lock_version=2))
        other = tmp_path / "authored"
        _hand_authored(other, "mine")
        assert {s["name"] for s in discover_skills([str(other)])} == {"mine"}

    def test_the_repos_own_skills_are_still_read(self, tmp_path, monkeypatch):
        pkg = self._outer(tmp_path, lock_version=1)
        _hand_authored(pkg.parent / ".claude" / "skills", "in-repo")
        monkeypatch.chdir(pkg)
        assert {s["name"] for s in discover_skills()} == {"in-repo"}


class TestHomeIsNotAProject:
    """The project search never goes above home, so from a folder under home that is not a
    repository it can end AT home: a ``~/.decimal/`` (``decimalai skills`` run there once),
    or a ``~/.git`` (dotfiles kept in git). Home's skill directories are the personal ones
    (``~/.claude/skills``), which discovery reads only with ``include_global=True``, so
    they are not read as that folder's project skills, and nothing in them is synced."""

    @staticmethod
    def _home(tmp_path: Path, monkeypatch, marker: str = ".decimal") -> Path:
        home = tmp_path / "home"
        home.mkdir()
        if marker == ".decimal":
            (home / ".decimal").mkdir()
        elif marker == "skills.toml":
            (home / ".decimal").mkdir()
            (home / ".decimal" / "skills.toml").write_text("", encoding="utf-8")
        elif marker == "skills.lock":
            _lock(home)
        elif marker == ".git":
            (home / ".git").mkdir()
        _hand_authored(home / ".claude" / "skills", "personal")
        monkeypatch.setenv("HOME", str(home))
        return home

    @staticmethod
    def _folder(home: Path, monkeypatch) -> Path:
        folder = home / "scratch" / "notes"
        folder.mkdir(parents=True)
        monkeypatch.chdir(folder)
        return folder

    @pytest.mark.parametrize("marker", [".decimal", "skills.toml", "skills.lock", ".git"])
    def test_personal_skills_are_not_read_from_a_folder_under_home(
        self, tmp_path, monkeypatch, marker
    ):
        home = self._home(tmp_path, monkeypatch, marker)
        folder = self._folder(home, monkeypatch)
        assert discover_skills() == []
        _hand_authored(folder / ".claude" / "skills", "local")
        assert {s["name"] for s in discover_skills()} == {"local"}  # its own are read

    def test_sync_to_platform_uploads_no_personal_skill(self, tmp_path, monkeypatch):
        self._folder(self._home(tmp_path, monkeypatch), monkeypatch)
        sent = []

        class FakeRouter:
            def __init__(self, **kw):
                pass

            def sync_skills(self, skills, **kwargs):
                sent.extend(s["name"] for s in skills)
                return {"created": len(skills), "updated": 0, "unchanged": 0}

        monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)
        sync_to_platform("dai_sk_test")
        assert sent == []

    def test_a_version_2_lock_at_home_is_not_used_from_below_it(self, tmp_path, monkeypatch):
        home = self._home(tmp_path, monkeypatch, "skills.lock")
        _install(home / ".agents" / "skills")
        self._folder(home, monkeypatch)
        other = tmp_path / "authored"
        _hand_authored(other, "mine")
        assert {s["name"] for s in discover_skills([str(other)])} == {"mine"}

    def test_run_from_home_itself_the_personal_skills_are_read_as_before(
        self, tmp_path, monkeypatch
    ):
        """At home the default paths are relative, so they are home's, as they always were;
        and home's managed root counts, since home is the working directory."""
        home = self._home(tmp_path, monkeypatch, "skills.lock")
        _install(home / ".agents" / "skills")
        monkeypatch.chdir(home)
        assert {s["name"] for s in discover_skills()} == {"personal", "acme-refund-policy"}
        other = tmp_path / "authored"
        _hand_authored(other, "mine")
        assert {s["name"] for s in discover_skills([str(other)])} == {
            "mine", "acme-refund-policy"
        }


# ── deduplication ────────────────────────────────────────────


class TestDedup:
    def test_one_registry_skill_in_two_dirs_is_one_descriptor(self, tmp_path):
        """Deduplicated by skill_id first: the same install under two folder names is one
        skill, and the first directory searched wins."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        _install(a, "acme-refund-policy")
        _install(b, "acme-refund-policy-3f9a1c")
        got = discover_skills([str(a), str(b)])
        assert [s["name"] for s in got] == ["acme-refund-policy"]

    def test_two_registry_skills_sharing_a_name_dedupe_by_name(self, tmp_path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _install(a, "pdf", skill_id="id-one", registry_name="one/pdf")
        _install(b, "pdf", skill_id="id-two", registry_name="two/pdf")
        got = discover_skills([str(a), str(b)])
        assert [(s["name"], s["skill_id"]) for s in got] == [("pdf", "id-one")]

    def test_a_hand_authored_skill_and_an_install_with_its_name_dedupe_by_name(
        self, tmp_path
    ):
        a = tmp_path / "a"
        b = tmp_path / "b"
        _hand_authored(a, "acme-refund-policy")
        _install(b)
        got = discover_skills([str(a), str(b)])
        assert [(s["name"], s["managed"]) for s in got] == [("acme-refund-policy", False)]


# ── sync ─────────────────────────────────────────────────────


def test_sync_to_platform_never_sends_a_managed_folder(tmp_path, monkeypatch):
    root = tmp_path / "skills"
    _install(root)
    _hand_authored(root, "mine")
    sent = {}

    class FakeRouter:
        def __init__(self, **kw):
            pass

        def sync_skills(self, skills, **kwargs):
            sent["names"] = sorted(s["name"] for s in skills)
            return {"created": 1, "updated": 0, "unchanged": 0}

    monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)
    monkeypatch.chdir(tmp_path)  # the sync stamps the working directory's install identity
    sync_to_platform("dai_sk_test", search_paths=[str(root)])
    assert sent["names"] == ["mine"]


# ── the reader without skillevaluation ───────────────────────


def test_the_reader_imports_and_reads_without_skillevaluation(tmp_path):
    """`decimalai` and `decimalai.skills` import nothing outside the standard library at
    import time. With skillevaluation (or PyYAML) missing, the reader uses the simple reader
    and says so once, and discovery (which finds the project through
    `decimalai.skillpkg.project`) still works."""
    d = _hand_authored(tmp_path, "mine")
    project = tmp_path / "project"
    _lock(project)
    _install(project / ".agents" / "skills")
    (project / "src").mkdir()
    code = textwrap.dedent(f"""
        import logging, os, sys
        sys.modules["skillevaluation"] = None
        sys.modules["yaml"] = None
        logging.basicConfig(level=logging.WARNING, stream=sys.stdout, format="%(message)s")
        import decimalai
        from decimalai.skills import discover_skills, parse_skill_md
        got = parse_skill_md({str(d / "SKILL.md")!r})
        parse_skill_md({str(d / "SKILL.md")!r})
        print("NAME", got["name"], got["managed"])
        os.chdir({str(project / "src")!r})
        print("FOUND", sorted(s["name"] for s in discover_skills()))
        print("EXPLICIT", "mine" in {{s["name"] for s in discover_skills([{str(tmp_path)!r}])}})
    """)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(Path(sk.__file__).resolve().parents[1])},
    )
    assert proc.returncode == 0, proc.stderr
    assert "NAME mine False" in proc.stdout
    assert "FOUND ['acme-refund-policy']" in proc.stdout, proc.stdout
    assert "EXPLICIT True" in proc.stdout, proc.stdout
    assert proc.stdout.count("skillevaluation") == 1, proc.stdout


def test_without_skillfmt_the_simple_reader_still_reads_a_managed_folder(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sk, "_skillfmt", lambda: None)
    got = parse_skill_md(str(_install(tmp_path) / "SKILL.md"))
    assert got["name"] == "acme-refund-policy"
    assert got["managed"] is True
    assert got["skill_id"] == SKILL_ID
    assert got["registry_name"] == "acme/refund-policy"


def test_the_fallback_name_grammar_is_skillfmts():
    assert sk._NAME_RE.pattern == skillfmt.NAME_RE.pattern
    assert sk._NAME_MAX == skillfmt.NAME_MAX
