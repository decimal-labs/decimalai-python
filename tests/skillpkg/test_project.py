"""Where a project's managed skills live, and ``skills_dir()``.

Every documented loader snippet starts from ``decimalai.skills_dir()``, so its
contract is pinned here: the root is absolute and symlink-resolved (Strands,
deepagents and the Claude plugin need that), it can be given relative to a working
directory (openai-agents lists nothing outside its cwd), and it refuses rather than
guesses: no lock or no root is "not installed", and a symlink anywhere between the
project and the root is refused, because the engine never writes through one.

The core dependencies the engine reads its files with are pinned here too.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import project as P

REPO_ROOT = Path(__file__).resolve().parents[2]
POSIX = os.name == "posix"


def _installed(directory: Path, lock: bytes = L.dumps(L.empty())) -> Path:
    """A project with a lock and an (empty) root: what `skills install` leaves."""
    (directory / ".decimal").mkdir(parents=True, exist_ok=True)
    (directory / ".decimal" / "skills.lock").write_bytes(lock)
    (directory / ".agents" / "skills").mkdir(parents=True, exist_ok=True)
    return directory


V1_LOCK = b'{"version": 1, "skills": {"acme/refund-policy": {"version": 1}}}\n'


@pytest.fixture(autouse=True)
def _home_above_tmp(tmp_path, monkeypatch):
    """Bound every project search just above the test's own directory, whatever sits
    higher up. (Home itself is never the project of a directory below it.)"""
    monkeypatch.setenv("HOME", str(tmp_path.parent))


# ── the layout ───────────────────────────────────────────────────────────────


class TestLayout:
    def test_the_paths_are_pinned(self, tmp_path):
        project = P.Project.at(tmp_path)
        d = project.directory
        assert d == tmp_path.resolve()
        assert project.state_dir == d / ".decimal"
        assert project.manifest_path == d / ".decimal" / "skills.toml"
        assert project.lock_path == d / ".decimal" / "skills.lock"
        assert project.npx_lock_path == d / ".decimal" / "skills.lock.json"
        assert project.journal_path == d / ".decimal" / "skills.lock.journal"
        assert project.process_lock_path == d / ".decimal" / ".skills.lck"
        assert project.staging_dir == d / ".decimal" / "staging"
        assert project.backup_dir == d / ".decimal" / "backup"
        assert project.migrated_dir == d / ".decimal" / "migrated-v1"
        assert project.agents_dir == d / ".agents"
        assert project.skills_root == d / ".agents" / "skills"
        assert project.plugin_json_path == d / ".agents" / ".claude-plugin" / "plugin.json"

    def test_the_root_and_the_plugin_name_are_fixed(self):
        assert P.ROOT == ".agents/skills"
        assert P.PLUGIN_NAME == "decimalai-skills"
        # A skill may never take the plugin's name as its own install name.
        from skillevaluation import skillfmt

        assert P.PLUGIN_NAME in skillfmt.RESERVED

    def test_gitignore_lines_cover_every_per_checkout_file(self):
        # What `.decimal/.gitignore` must list: everything under .decimal/ except the
        # manifest and the lock, which are committed.
        assert P.GITIGNORE_LINES == (
            "install.json", "staging/", "backup/", "migrated-v1/",
            "skills.lock.journal", ".skills.lck",
        )
        assert "skills.lock" not in P.GITIGNORE_LINES
        assert "skills.toml" not in P.GITIGNORE_LINES

    def test_skill_dir_is_one_folder_under_the_root(self, tmp_path):
        project = P.Project.at(tmp_path)
        assert project.skill_dir("acme-refund-policy") == project.skills_root / "acme-refund-policy"

    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "../x", "a\\b", "/abs", "x\x00y"])
    def test_skill_dir_refuses_anything_but_one_component(self, tmp_path, bad):
        with pytest.raises(E.UsageError):
            P.Project.at(tmp_path).skill_dir(bad)

    def test_rel_is_project_relative_and_posix(self, tmp_path):
        project = P.Project.at(tmp_path)
        assert project.rel(project.skill_dir("pdf") / "SKILL.md") == ".agents/skills/pdf/SKILL.md"
        assert project.rel(project.lock_path) == ".decimal/skills.lock"


# ── finding the project ──────────────────────────────────────────────────────


class TestFind:
    @pytest.mark.parametrize("marker", ["skills.toml", "skills.lock"])
    def test_the_nearest_ancestor_with_a_manifest_or_lock_wins(self, tmp_path, marker):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".decimal").mkdir()
        (tmp_path / ".decimal" / marker).write_text("")
        deep = tmp_path / "src" / "pkg"
        deep.mkdir(parents=True)
        assert P.Project.find(deep).directory == tmp_path.resolve()

    @pytest.mark.parametrize("git", ["file", "dir"])
    def test_the_search_stops_at_the_repository(self, tmp_path, git):
        # A folder of repositories (or of git worktrees) with its own stale .decimal/
        # above them: a run inside one repository never adopts it.
        outer = tmp_path / "outer"
        _installed(outer)
        repo = outer / "repo"
        pkg = repo / "pkg"
        pkg.mkdir(parents=True)
        if git == "file":  # a git worktree
            (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/repo\n")
        else:
            (repo / ".git").mkdir()
        assert P.Project.find(pkg).directory == repo.resolve()

    def test_a_manifest_between_the_start_and_the_repository_root_wins(self, tmp_path):
        (tmp_path / ".git").mkdir()
        sub = tmp_path / "services" / "agent"
        (sub / ".decimal").mkdir(parents=True)
        (sub / ".decimal" / "skills.toml").write_text("")
        start = sub / "src"
        start.mkdir()
        assert P.Project.find(start).directory == sub.resolve()

    def test_a_repository_at_the_start_stops_the_search_there(self, tmp_path):
        _installed(tmp_path)
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        assert P.Project.find(repo).directory == repo.resolve()

    def test_without_a_repository_the_start_is_the_project(self, tmp_path):
        start = tmp_path / "a" / "b"
        start.mkdir(parents=True)
        assert P.Project.find(start).directory == start.resolve()

    def test_the_search_never_goes_above_home(self, tmp_path, monkeypatch):
        _installed(tmp_path / "outer")
        home = tmp_path / "outer" / "home"
        start = home / "scratch" / "agent"
        start.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        assert P.Project.find(start).directory == start.resolve()

    @pytest.mark.parametrize("home_holds", ["install.json", "skills.toml", "skills.lock", ".git"])
    def test_home_is_never_the_project_of_a_directory_below_it(self, tmp_path, monkeypatch,
                                                              home_holds):
        # Running the SDK in ~ leaves ~/.decimal/install.json; a dotfiles repository makes
        # ~ a repository. Neither may turn ~ into the project of a folder under it: the
        # project's readers would then treat ~/.claude/skills as project skills.
        home = tmp_path / "home"
        if home_holds == ".git":
            (home / ".git").mkdir(parents=True)
        else:
            (home / ".decimal").mkdir(parents=True)
            (home / ".decimal" / home_holds).write_bytes(L.dumps(L.empty()))
            (home / ".agents" / "skills").mkdir(parents=True)
        (home / ".claude" / "skills" / "my-private-notes").mkdir(parents=True)
        work = home / "notebooks" / "experiment"
        work.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        assert P.Project.find(work).directory == work.resolve()
        with pytest.raises(E.SkillsNotInstalledError):
            P.skills_dir(work)

    def test_a_repository_under_home_is_still_found(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        (home / ".decimal").mkdir(parents=True)
        (home / ".decimal" / "install.json").write_text("{}")
        repo = home / "code" / "agent"
        (repo / ".git").mkdir(parents=True)
        (repo / "src").mkdir()
        monkeypatch.setenv("HOME", str(home))
        assert P.Project.find(repo / "src").directory == repo.resolve()

    @pytest.mark.parametrize("holds", [None, "install.json", "skills.toml"])
    def test_run_from_home_itself_home_is_the_project(self, tmp_path, monkeypatch, holds):
        home = tmp_path / "home"
        home.mkdir()
        if holds:
            (home / ".decimal").mkdir()
            (home / ".decimal" / holds).write_text("")
        monkeypatch.setenv("HOME", str(home))
        assert P.Project.find(home).directory == home.resolve()

    def test_outside_home_the_repository_is_the_only_bound(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path / "somewhere" / "else"))
        _installed(tmp_path / "app")
        start = tmp_path / "app" / "src"
        start.mkdir()
        assert P.Project.find(start).directory == (tmp_path / "app").resolve()

    def test_without_a_marker_it_is_the_repository_root(self, tmp_path):
        (tmp_path / ".git").mkdir()
        deep = tmp_path / "a" / "b"
        deep.mkdir(parents=True)
        assert P.Project.find(deep).directory == tmp_path.resolve()

    def test_a_decimal_dir_without_a_marker_is_the_install_root(self, tmp_path):
        # `.decimal/install.json` alone (an older SDK) still marks the project.
        (tmp_path / ".decimal").mkdir()
        (tmp_path / ".decimal" / "install.json").write_text("{}")
        deep = tmp_path / "x"
        deep.mkdir()
        assert P.Project.find(deep).directory == tmp_path.resolve()

    def test_the_default_start_is_the_cwd(self, tmp_path, monkeypatch):
        _installed(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        assert P.Project.find().directory == tmp_path.resolve()
        assert P.find_project().directory == tmp_path.resolve()

    def test_at_takes_the_directory_as_given(self, tmp_path):
        _installed(tmp_path)
        sub = tmp_path / "sub"
        sub.mkdir()
        assert P.Project.at(sub).directory == sub.resolve()

    @pytest.mark.skipif(not POSIX, reason="symlinks")
    def test_the_directory_is_resolved(self, tmp_path, monkeypatch):
        real = tmp_path / "real"
        real.mkdir()
        (tmp_path / "link").symlink_to(real, target_is_directory=True)
        monkeypatch.chdir(tmp_path)
        assert P.Project.at("link").directory == real.resolve()
        assert P.Project.at("link").skills_root == real.resolve() / ".agents" / "skills"


# ── skills_dir() ─────────────────────────────────────────────────────────────


class TestSkillsDir:
    def test_absolute_and_resolved(self, tmp_path):
        _installed(tmp_path)
        root = P.skills_dir(tmp_path)
        assert root.is_absolute()
        assert root == (tmp_path / ".agents" / "skills").resolve()
        assert root == Path(os.path.realpath(root))

    def test_found_from_a_subdirectory_and_from_the_cwd(self, tmp_path, monkeypatch):
        _installed(tmp_path)
        sub = tmp_path / "app" / "agents"
        sub.mkdir(parents=True)
        expected = (tmp_path / ".agents" / "skills").resolve()
        assert P.skills_dir(sub) == expected
        monkeypatch.chdir(sub)
        assert P.skills_dir() == expected

    @pytest.mark.skipif(not POSIX, reason="symlinks")
    def test_a_symlink_above_the_project_is_resolved_not_refused(self, tmp_path):
        # macOS makes /tmp a link of exactly this kind: a link above the project.
        real = tmp_path / "real"
        _installed(real)
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        assert P.skills_dir(link) == (real / ".agents" / "skills").resolve()

    def test_no_lock_is_not_installed(self, tmp_path):
        (tmp_path / ".agents" / "skills").mkdir(parents=True)
        (tmp_path / ".decimal").mkdir()
        (tmp_path / ".decimal" / "skills.toml").write_text('root = ".agents/skills"\n')
        with pytest.raises(E.SkillsNotInstalledError) as exc:
            P.skills_dir(tmp_path)
        assert exc.value.exit_code == 3
        assert "decimalai skills install" in str(exc.value)

    @pytest.mark.parametrize("raw", [V1_LOCK, b"{}\n", b'{"skills": {}}\n'])
    def test_a_version_1_lock_is_refused_with_the_migrate_fix(self, tmp_path, raw):
        _installed(tmp_path, raw)
        with pytest.raises(E.LegacyLockError) as exc:
            P.skills_dir(tmp_path)
        assert exc.value.exit_code == 3
        assert "decimalai skills migrate" in exc.value.fix

    def test_the_npx_lock_is_refused_with_the_migrate_fix(self, tmp_path):
        _installed(tmp_path)
        (tmp_path / ".decimal" / "skills.lock.json").write_text('{"skills": {}}\n')
        with pytest.raises(E.LegacyLockError, match="migrate"):
            P.skills_dir(tmp_path)

    @pytest.mark.parametrize("raw", [b"", b"{", b"[]", b'{"version": 3, "skills": {}}\n',
                                     b'{"version": "2", "skills": {}}\n'])
    def test_an_unreadable_or_unknown_lock_is_refused(self, tmp_path, raw):
        _installed(tmp_path, raw)
        with pytest.raises(E.SkillsLockError) as exc:
            P.skills_dir(tmp_path)
        assert exc.value.exit_code == 3

    def test_a_version_2_lock_is_not_read_further(self, tmp_path):
        # A stale lock_hash or merged entries are for `install` to report; the loaders
        # can still read the root, so skills_dir() still names it.
        import json

        data = json.loads(L.dumps(L.empty()))
        data["lock_hash"] = "0" * 64
        data["skills"]["acme/refund-policy"] = {"version": 3, "content_hash": "x"}
        _installed(tmp_path, json.dumps(data).encode())
        assert P.skills_dir(tmp_path) == (tmp_path / ".agents" / "skills").resolve()

    def test_the_stale_umbrella_case(self, tmp_path):
        # A directory holding many repositories and worktrees, with a version 1 lock and
        # a root full of leftovers. From inside a worktree, that root is never returned.
        umbrella = tmp_path / "repos"
        _installed(umbrella, V1_LOCK)
        (umbrella / ".agents" / "skills" / "leftover").mkdir()
        worktree = umbrella / ".worktrees" / "task" / "sdk"
        (worktree / "pkg").mkdir(parents=True)
        (worktree / ".git").write_text("gitdir: ../../../sdk/.git/worktrees/sdk\n")
        with pytest.raises(E.SkillsNotInstalledError):
            P.skills_dir(worktree / "pkg")
        assert P.Project.find(worktree / "pkg").directory == worktree.resolve()

    def test_no_root_is_not_installed(self, tmp_path):
        (tmp_path / ".decimal").mkdir()
        (tmp_path / ".decimal" / "skills.lock").write_bytes(L.dumps(L.empty()))
        with pytest.raises(E.SkillsNotInstalledError, match=r"\.agents/skills"):
            P.skills_dir(tmp_path)

    def test_an_empty_directory_is_not_installed(self, tmp_path):
        with pytest.raises(E.SkillsNotInstalledError):
            P.skills_dir(tmp_path)

    def test_a_file_where_the_root_belongs_is_refused(self, tmp_path):
        (tmp_path / ".decimal").mkdir()
        (tmp_path / ".decimal" / "skills.lock").write_bytes(L.dumps(L.empty()))
        (tmp_path / ".agents").mkdir()
        (tmp_path / ".agents" / "skills").write_text("not a directory")
        with pytest.raises(E.IllegalEntryError):
            P.skills_dir(tmp_path)

    @pytest.mark.skipif(not POSIX, reason="symlinks")
    @pytest.mark.parametrize("component", [".agents", ".agents/skills"])
    def test_a_symlinked_component_is_refused(self, tmp_path, component):
        project = tmp_path / "project"
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / "skills").mkdir(parents=True)
        (project / ".decimal").mkdir(parents=True)
        (project / ".decimal" / "skills.lock").write_bytes(L.dumps(L.empty()))
        if component == ".agents":
            (project / ".agents").symlink_to(elsewhere, target_is_directory=True)
        else:
            (project / ".agents").mkdir()
            (project / ".agents" / "skills").symlink_to(
                elsewhere / "skills", target_is_directory=True)
        with pytest.raises(E.IllegalEntryError) as exc:
            P.skills_dir(project)
        assert exc.value.exit_code == 5
        assert component in str(exc.value)

    def test_relative_to_a_directory_above_the_root(self, tmp_path):
        _installed(tmp_path)
        assert P.skills_dir(tmp_path, relative_to=tmp_path) == Path(".agents/skills")
        assert P.skills_dir(tmp_path, relative_to=tmp_path / ".agents") == Path("skills")
        rel = P.skills_dir(tmp_path, relative_to=tmp_path)
        assert not rel.is_absolute()

    def test_relative_to_the_cwd(self, tmp_path, monkeypatch):
        _installed(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert P.skills_dir(relative_to=Path.cwd()) == Path(".agents/skills")
        assert (Path.cwd() / P.skills_dir(relative_to=Path.cwd())).is_dir()

    @pytest.mark.skipif(not POSIX, reason="symlinks")
    def test_relative_to_compares_resolved_paths(self, tmp_path):
        # The cwd is often reached through a link (/tmp on macOS); the answer must not
        # depend on which spelling of the same directory the caller passes.
        real = tmp_path / "real"
        _installed(real)
        link = tmp_path / "link"
        link.symlink_to(real, target_is_directory=True)
        assert P.skills_dir(real, relative_to=link) == Path(".agents/skills")

    def test_relative_to_a_directory_beside_the_root_raises(self, tmp_path):
        _installed(tmp_path / "project")
        (tmp_path / "other").mkdir()
        with pytest.raises(E.UsageError) as exc:
            P.skills_dir(tmp_path / "project", relative_to=tmp_path / "other")
        assert isinstance(exc.value, ValueError)
        assert "not under" in exc.value.message


# ── symlink refusal and atomic writes ────────────────────────────────────────


class TestRefuseSymlinks:
    @pytest.mark.skipif(not POSIX, reason="symlinks")
    def test_every_component_below_the_project_is_checked(self, tmp_path):
        project = P.Project.at(tmp_path / "p")
        project.directory.mkdir()
        (tmp_path / "elsewhere").mkdir()
        (project.directory / ".decimal").symlink_to(tmp_path / "elsewhere",
                                                    target_is_directory=True)
        with pytest.raises(E.IllegalEntryError, match=r"\.decimal"):
            project.refuse_symlinks(project.lock_path)

    @pytest.mark.skipif(not POSIX, reason="symlinks")
    def test_the_last_component_is_checked_too(self, tmp_path):
        project = P.Project.at(tmp_path)
        project.state_dir.mkdir()
        (tmp_path / "target").write_text("{}")
        project.lock_path.symlink_to(tmp_path / "target")
        with pytest.raises(E.IllegalEntryError, match=r"skills\.lock"):
            project.refuse_symlinks(project.lock_path)

    def test_missing_components_are_fine(self, tmp_path):
        P.Project.at(tmp_path).refuse_symlinks(tmp_path / ".agents" / "skills" / "x" / "SKILL.md")

    def test_a_path_outside_the_project_is_a_usage_error(self, tmp_path):
        project = P.Project.at(tmp_path / "p")
        with pytest.raises(E.UsageError):
            project.refuse_symlinks(tmp_path / "elsewhere")


class TestWriteFile:
    def test_writes_exact_bytes_and_mode(self, tmp_path):
        project = P.Project.at(tmp_path)
        target = project.state_dir / "x" / "f.txt"
        project.write_file(target, b"a\r\nb\n", mode=0o644)
        assert target.read_bytes() == b"a\r\nb\n"
        if POSIX:
            assert target.stat().st_mode & 0o777 == 0o644

    def test_a_failed_replace_leaves_the_old_file_and_no_temp(self, tmp_path, monkeypatch):
        project = P.Project.at(tmp_path)
        target = project.lock_path
        project.write_file(target, b"old\n")

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(P.os, "replace", boom)
        with pytest.raises(OSError, match="disk full"):
            project.write_file(target, b"new\n")
        assert target.read_bytes() == b"old\n"
        assert sorted(p.name for p in project.state_dir.iterdir()) == ["skills.lock"]

    @pytest.mark.skipif(not POSIX, reason="symlinks")
    def test_never_writes_through_a_symlink(self, tmp_path):
        project = P.Project.at(tmp_path / "p")
        project.directory.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (project.directory / ".decimal").symlink_to(outside, target_is_directory=True)
        with pytest.raises(E.IllegalEntryError):
            project.write_file(project.lock_path, b"{}\n")
        assert list(outside.iterdir()) == []


# ── packaging ────────────────────────────────────────────────────────────────


def _core_dependencies() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: the conditional core dependency
        import tomli as tomllib
    from packaging.requirements import Requirement

    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        raw = tomllib.load(fh)["project"]["dependencies"]
    reqs = [Requirement(r) for r in raw]
    names = [r.name.lower() for r in reqs]
    assert len(names) == len(set(names)), f"a core dependency is listed twice: {names}"
    return {r.name.lower(): r for r in reqs}


class TestCoreDependencies:
    def test_the_engine_dependencies_are_pinned(self):
        deps = _core_dependencies()
        # The format module (install names, hashes, lock_hash, the validator).
        assert str(deps["skillevaluation"].specifier) == ">=0.8.0"
        assert deps["skillevaluation"].marker is None
        # SKILL.md frontmatter is YAML.
        assert str(deps["pyyaml"].specifier) == ">=6.0"
        assert deps["pyyaml"].marker is None
        # The manifest is TOML; Python 3.11 added tomllib.
        tomli = deps["tomli"]
        assert str(tomli.specifier) == ">=2"
        assert tomli.marker is not None
        assert tomli.marker.evaluate({"python_version": "3.10"})
        assert not tomli.marker.evaluate({"python_version": "3.11"})
        assert not tomli.marker.evaluate({"python_version": "3.13"})

    def test_this_environment_satisfies_them(self):
        from importlib import metadata

        from packaging.markers import default_environment

        env = default_environment()
        for name, req in _core_dependencies().items():
            if name not in ("skillevaluation", "pyyaml", "tomli"):
                continue
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            have = metadata.version(req.name)
            assert req.specifier.contains(have, prereleases=True), (name, have)

    def test_the_manifest_reads_with_tomli_where_there_is_no_tomllib(self, tmp_path):
        # Python 3.10 has no tomllib. Hide it, and stand tomli in for it (here a thin
        # wrapper of tomllib that records its use): the manifest must load through it.
        (tmp_path / ".decimal").mkdir()
        (tmp_path / ".decimal" / "skills.toml").write_text(
            'root = ".agents/skills"\n\n[skills]\n"acme/refund-policy" = "3"\n')
        code = textwrap.dedent(f"""
            import json, sys, types
            try:
                import tomllib as real
            except ModuleNotFoundError:  # 3.10 itself: wrap the real tomli
                import tomli as real
            used = []
            stub = types.ModuleType("tomli")
            def loads(s, **kw):
                used.append(1)
                return real.loads(s, **kw)
            stub.loads = loads
            stub.TOMLDecodeError = real.TOMLDecodeError
            sys.modules["tomllib"] = None
            sys.modules["tomli"] = stub
            from decimalai.skillpkg import manifest, project
            m = manifest.load(project.Project.at({str(tmp_path)!r}))
            print(json.dumps({{"specs": {{n: e.spec for n, e in m.entries.items()}},
                              "used": bool(used)}}))
        """)
        env = {k: v for k, v in os.environ.items() if not k.startswith("DECIMAL")}
        env.update({"DECIMAL_AUTOINIT": "false", "DECIMALAI_SKIP_VERIFY": "1",
                    "PYTHONDONTWRITEBYTECODE": "1"})
        child = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                               capture_output=True, text=True, timeout=120)
        assert child.returncode == 0, child.stderr
        import json

        assert json.loads(child.stdout.strip().splitlines()[-1]) == {
            "specs": {"acme/refund-policy": "3"}, "used": True,
        }
