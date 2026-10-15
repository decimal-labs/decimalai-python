"""``decimalai skills migrate``: a version 1 lock, or npx's lock, becomes a manifest and a version 2 lock.

The conversion runs once, when asked, and it is the one step that resets pins and moves
folders, so this file pins what it promises:

- a fleet-shaped version 1 lock gives ``latest`` pins, and its hashes (the empty body, in
  every entry) and version numbers are ignored: the folders hold the registry's bytes;
- an npx lock gives exact pins at the versions it recorded;
- every folder a legacy lock declared inside ``.agents/skills`` moves to
  ``.decimal/migrated-v1/<id>/``, byte for byte; folders outside the root stay exactly where
  they are and are listed; nothing is deleted;
- a version 1 ``skill_id`` that its name now resolves past is skipped and reported; entries
  that do not resolve are reported, and change nothing without ``yes=True`` (exit 7);
- an existing ``.decimal/skills.toml`` wins;
- a second run is a no-op, and a run that crashes at any step leaves the old state
  recoverable and is finished by the next run.

The registry is ``tests/fakes/fake_registry.py``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Optional

import pytest

from decimalai import skillpkg
from decimalai.skillpkg import errors as E
from decimalai.skillpkg import fs
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import manifest as M
from decimalai.skillpkg import migration
from decimalai.skillpkg.client import RegistryClient
from decimalai.skillpkg.project import Project
from tests.fakes.fake_registry import BASE_URL, OWNER_KEY, OWNER_ORG, FakeRegistry

FIXTURES = Path(__file__).parent / "fixtures"
POSIX = os.name == "posix"
needs_posix = pytest.mark.skipif(not POSIX, reason="POSIX symlinks")
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

FLEET = ("incident-triage", "release-notes", "vendor-onboarding", "weekly-digest")


# ── fixtures and helpers ─────────────────────────────────────────────────────


@pytest.fixture
def registry() -> FakeRegistry:
    return FakeRegistry()


@pytest.fixture
def client(registry):
    with RegistryClient(api_key=OWNER_KEY, base_url=BASE_URL,
                        transport=registry.transport()) as c:
        yield c


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _old_folder(directory: Path, rel: str, text: str = "stale copy") -> Path:
    """A folder as a legacy install left it: a SKILL.md and one attachment."""
    folder = directory.joinpath(*rel.split("/"))
    (folder / "references").mkdir(parents=True, exist_ok=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {folder.name}\ndescription: {text}\n---\n\n{text}\n", encoding="utf-8")
    (folder / "references" / "notes.md").write_text(f"{text}: notes\n", encoding="utf-8")
    return folder


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _seed_fleet(registry: FakeRegistry, data=None, *, versions: int = 3) -> dict:
    data = data if data is not None else _fixture("fleet_v1_lock.json")
    return {name: registry.add_skill(name, skill_id=entry["skill_id"], versions=versions)
            for name, entry in data["skills"].items() if entry.get("skill_id")}


def _seed_npx(registry: FakeRegistry) -> dict:
    return {
        "contract-review-6f1c2a": registry.add_skill(
            "acme/contract-review", url_slug="contract-review-6f1c2a", versions=3,
            files=[{"path": "references/clauses.md", "content": "# Clauses\n\nRead them all.\n"}]),
        "pdf-tools": registry.add_skill("pdf-tools", versions=4),
    }


def _fleet_files(directory: Path, data=None) -> Project:
    """A project as a fleet machine has it: a version 1 lock and its folders in the root."""
    data = data if data is not None else _fixture("fleet_v1_lock.json")
    project = Project.at(directory)
    (directory / ".git").mkdir(parents=True, exist_ok=True)
    _write_json(project.lock_path, data)
    for name in data["skills"]:
        _old_folder(directory, f".agents/skills/{name}")
    return project


def _npx_files(directory: Path, data=None) -> Project:
    """A project after ``npx decimalai add``: its lock, and the files it lists."""
    data = data if data is not None else _fixture("npx_lock.json")
    project = Project.at(directory)
    (directory / ".git").mkdir(parents=True, exist_ok=True)
    _write_json(project.npx_lock_path, data)
    for entry in data.values():
        for rel in entry["files"]:
            path = directory.joinpath(*rel.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"npx wrote {rel}\n", encoding="utf-8")
    return project


def _tree(path: Path) -> dict:
    """Every file under ``path``: relative path -> (bytes, executable)."""
    out = {}
    if not path.exists():
        return out
    for p in sorted(path.rglob("*")):
        if p.is_file() and not p.is_symlink():
            out[p.relative_to(path).as_posix()] = (
                p.read_bytes(), bool(p.stat().st_mode & stat.S_IXUSR) if POSIX else None)
    return out


def _files(directory: Path) -> dict:
    """Every file of the project, by project-relative path (the old state, for a crash test)."""
    return {rel: data for rel, (data, _) in _tree(directory).items()
            if not rel.startswith(".git/")}


def _backups(project: Project) -> list:
    root = project.migrated_dir
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _served(registry: FakeRegistry, skill, number: int, install_name: str) -> dict:
    """The bytes the registry serves for one version, by path."""
    bundle = registry.render(skill, skill.version(number), install_name=install_name)
    return {f.path: f.content.encode("utf-8") for f in bundle.files}


def _rows(result) -> dict:
    return {c.name: c for c in result.changes if isinstance(c, migration.MigratedSkill)}


def _backed_up_locks(result) -> dict:
    return {c.path: c.moved_to for c in result.changes if isinstance(c, migration.LockBackup)}


def _texts(problems) -> list:
    return [str(p) for p in problems]


# ── a fleet-shaped version 1 lock ────────────────────────────────────────────


class TestFleetLock:
    def test_gives_latest_pins_keyed_by_registry_name(self, tmp_path, registry, client):
        project = _fleet_files(tmp_path)
        _seed_fleet(registry)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        manifest = M.load(project)
        assert {n: e.spec for n, e in manifest.entries.items()} == {n: "latest" for n in FLEET}
        lock = L.load(project)
        assert sorted(lock.skills) == sorted(FLEET)
        for name, entry in lock.skills.items():
            assert (entry["pin"], entry["version"], entry["name"]) == ("latest", 3, name)
        assert not project.npx_lock_path.exists()
        assert {r.outcome for r in _rows(result).values()} == {"installed"}

    def test_its_hashes_and_version_numbers_are_ignored(self, tmp_path, registry, client):
        data = _fixture("fleet_v1_lock.json")
        # A version the resolver path never meant, and hashes of another body.
        data["skills"]["release-notes"].update(
            version=7, content_hash="ab" * 32, body_hash="cd" * 32)
        project = _fleet_files(tmp_path, data)
        skills = _seed_fleet(registry, data)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        lock = L.load(project)
        for name in FLEET:
            served = _served(registry, skills[name], 3, name)
            on_disk = {rel: b for rel, (b, _) in _tree(project.skill_dir(name)).items()}
            assert on_disk == served, name
            entry = lock.skills[name]
            assert entry["version"] == 3
            assert entry["body_sha256"] not in (EMPTY_SHA256, "cd" * 32)
            assert entry["files"]["SKILL.md"] == hashlib.sha256(served["SKILL.md"]).hexdigest()

    def test_empty_body_pins_are_flagged(self, tmp_path, registry, client):
        data = _fixture("fleet_v1_lock.json")
        data["skills"]["weekly-digest"]["content_hash"] = "12" * 32  # a real-looking pin
        _fleet_files(tmp_path, data)
        _seed_fleet(registry, data)

        result = migration.migrate(directory=tmp_path, client=client)

        rows = _rows(result)
        assert {n for n, r in rows.items() if r.empty_body_pin} == set(FLEET) - {"weekly-digest"}
        assert rows["release-notes"].claim == "version 1"
        assert any("empty body" in t for t in _texts(result.problems))
        assert result.ok


# ── an npx lock ──────────────────────────────────────────────────────────────


class TestNpxLock:
    def test_gives_exact_pins(self, tmp_path, registry, client):
        project = _npx_files(tmp_path)
        skills = _seed_npx(registry)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        manifest = M.load(project)
        assert {n: e.spec for n, e in manifest.entries.items()} == {
            "acme/contract-review": "2", "pdf-tools": "3"}
        lock = L.load(project)
        assert {k: (e["pin"], e["version"]) for k, e in lock.skills.items()} == {
            "acme-contract-review": ("exact", 2), "pdf-tools": ("exact", 3)}
        served = _served(registry, skills["contract-review-6f1c2a"], 2, "acme-contract-review")
        assert {r: b for r, (b, _) in _tree(project.skill_dir("acme-contract-review")).items()} \
            == served
        assert not project.npx_lock_path.exists()
        (backup,) = _backups(project)
        assert (backup / "skills.lock.json").read_bytes() == \
            (json.dumps(_fixture("npx_lock.json"), indent=2) + "\n").encode()
        assert _backed_up_locks(result) == {
            ".decimal/skills.lock.json": project.rel(backup / "skills.lock.json")}

    def test_its_folders_outside_the_root_stay_and_are_listed(self, tmp_path, registry, client):
        project = _npx_files(tmp_path)
        _seed_npx(registry)
        before = _tree(tmp_path / ".claude")

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        assert _tree(tmp_path / ".claude") == before
        rows = _rows(result)
        assert rows["pdf-tools"].old_folders == (
            migration.OldFolder(".claude/skills/pdf-tools", None),)
        assert rows["contract-review-6f1c2a"].old_folders == (
            migration.OldFolder(".claude/skills/contract-review-6f1c2a", None),)
        assert project.skill_dir("pdf-tools").is_dir()

    def test_its_folders_inside_the_root_move_to_the_backup(self, tmp_path, registry, client):
        data = {"pdf-tools": {"version": 3, "sha256": "ab" * 32,
                              "files": [".agents/skills/pdf-tools/SKILL.md"]}}
        project = _npx_files(tmp_path, data)
        _seed_npx(registry)
        old = (tmp_path / ".agents/skills/pdf-tools/SKILL.md").read_bytes()

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        (backup,) = _backups(project)
        assert (backup / "skills" / "pdf-tools" / "SKILL.md").read_bytes() == old
        assert project.skill_dir("pdf-tools").joinpath("SKILL.md").read_bytes() != old
        assert _rows(result)["pdf-tools"].old_folders == (migration.OldFolder(
            ".agents/skills/pdf-tools", project.rel(backup / "skills" / "pdf-tools")),)

    def test_an_unusable_version_is_unresolvable(self, tmp_path, registry, client):
        data = _fixture("npx_lock.json")
        data["pdf-tools"]["version"] = "three"
        _npx_files(tmp_path, data)
        _seed_npx(registry)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert any("pdf-tools" in t and "three" in t for t in _texts(result.problems))


# ── old folders: moved inside the root, left outside it, never deleted ───────


class TestOldFolders:
    def test_in_root_folders_move_to_the_backup_byte_for_byte(self, tmp_path, registry, client):
        project = _fleet_files(tmp_path)
        _seed_fleet(registry)
        own = _old_folder(tmp_path, ".agents/skills/my-own-skill", "hand-authored")
        old = {name: _tree(project.skill_dir(name)) for name in FLEET}
        own_before = _tree(own)
        lock_bytes = project.lock_path.read_bytes()

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        (backup,) = _backups(project)
        assert backup.parent == project.migrated_dir
        for name in FLEET:
            assert _tree(backup / "skills" / name) == old[name], name
            assert _rows(result)[name].old_folders == (migration.OldFolder(
                f".agents/skills/{name}", project.rel(backup / "skills" / name)),)
        assert (backup / "skills.lock").read_bytes() == lock_bytes
        assert _tree(own) == own_before
        assert sorted(p.name for p in project.skills_root.iterdir()) == sorted(
            FLEET + ("my-own-skill",))
        assert _backed_up_locks(result) == {
            ".decimal/skills.lock": project.rel(backup / "skills.lock")}

    def test_the_backup_is_ignored_by_git(self, tmp_path, registry, client):
        project = _fleet_files(tmp_path)
        _seed_fleet(registry)

        migration.migrate(directory=tmp_path, client=client)

        assert (project.migrated_dir / ".gitignore").read_bytes() == b"*\n"

    def test_folders_outside_the_root_stay_and_are_listed(self, tmp_path, registry, client):
        data = _fixture("fleet_v1_lock.json")
        data["skills"]["incident-triage"]["agents"] = ["claude-code", "universal"]
        project = _fleet_files(tmp_path, data)
        _seed_fleet(registry, data)
        outside = _old_folder(tmp_path, ".claude/skills/incident-triage", "claude copy")
        before = _tree(outside)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        assert _tree(outside) == before
        (backup,) = _backups(project)
        assert _rows(result)["incident-triage"].old_folders == (
            migration.OldFolder(".agents/skills/incident-triage",
                                project.rel(backup / "skills" / "incident-triage")),
            migration.OldFolder(".claude/skills/incident-triage", None),
        )
        assert any(".claude/skills/incident-triage" in t for t in _texts(result.problems))

    def test_a_namespaced_key_finds_its_folder_under_the_url_slug(self, tmp_path, registry, client):
        # Older releases named the folder after the url_slug when the name had a slash.
        data = {"version": 1, "skills": {"acme/contract-review": {
            "version": 1, "content_hash": "e3b0c44298fc", "source": "platform",
            "skill_id": "c0ffee00-0000-4000-8000-0000000000aa", "agents": ["universal"],
            "scope": "project"}}}
        project = Project.at(tmp_path)
        _write_json(project.lock_path, data)
        _old_folder(tmp_path, ".agents/skills/acme-contract-review")
        registry.add_skill("acme/contract-review", skill_id="c0ffee00-0000-4000-8000-0000000000aa",
                           url_slug="acme-contract-review", versions=2)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        # The old folder moved away, so the new one takes the base name, not the hashed one.
        assert sorted(L.load(project).skills) == ["acme-contract-review"]
        (backup,) = _backups(project)
        assert (backup / "skills" / "acme-contract-review" / "SKILL.md").is_file()

    def test_a_hand_authored_folder_keeps_its_name(self, tmp_path, registry, client):
        project = _npx_files(tmp_path)
        _seed_npx(registry)
        own = _old_folder(tmp_path, ".agents/skills/pdf-tools", "mine")
        before = _tree(own)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        assert _tree(own) == before
        (name,) = [k for k, e in L.load(project).skills.items() if e["name"] == "pdf-tools"]
        assert name.startswith("pdf-tools-") and name != "pdf-tools"


# ── resolution ───────────────────────────────────────────────────────────────


class TestResolution:
    def test_a_skill_id_mismatch_is_skipped_and_reported(self, tmp_path, registry, client):
        data = _fixture("fleet_v1_lock.json")
        lost = "c0ffee00-0000-4000-8000-000000000099"
        data["skills"]["vendor-onboarding"]["skill_id"] = lost
        project = _fleet_files(tmp_path, data)
        _seed_fleet(registry, {"skills": {k: v for k, v in data["skills"].items()
                                          if k != "vendor-onboarding"}})
        other = registry.add_skill("vendor-onboarding", versions=2)  # same name, another skill
        before = _tree(project.skill_dir("vendor-onboarding"))

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        row = _rows(result)["vendor-onboarding"]
        assert row.outcome == "skipped"
        assert lost in row.note and other.skill_id in row.note
        assert any(lost in t for t in _texts(result.problems))
        assert "vendor-onboarding" not in M.load(project)
        assert all(e["skill_id"] != other.skill_id for e in L.load(project).skills.values())
        # A skill the registry no longer has may still be loaded from its folder: it stays.
        assert _tree(project.skill_dir("vendor-onboarding")) == before
        assert row.old_folders == (migration.OldFolder(".agents/skills/vendor-onboarding"),)
        (backup,) = _backups(project)
        assert not (backup / "skills" / "vendor-onboarding").exists()
        assert any(".agents/skills/vendor-onboarding" in t and "left in place" in t
                   for t in _texts(result.problems))

    def test_an_empty_skill_id_resolves_by_name(self, tmp_path, registry, client):
        data = _fixture("fleet_v1_lock.json")
        data["skills"]["weekly-digest"]["skill_id"] = ""
        project = _fleet_files(tmp_path, data)
        _seed_fleet(registry, data)
        digest = registry.add_skill("weekly-digest", versions=2)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        assert L.load(project).skills["weekly-digest"]["skill_id"] == digest.skill_id

    def test_unresolvable_entries_exit_7_and_change_nothing(self, tmp_path, registry, client):
        data = _fixture("fleet_v1_lock.json")
        data["skills"]["retired-skill"] = dict(
            data["skills"]["weekly-digest"], skill_id="c0ffee00-0000-4000-8000-000000000098")
        project = _fleet_files(tmp_path, data)
        _seed_fleet(registry, {"skills": {k: v for k, v in data["skills"].items()
                                          if k != "retired-skill"}})
        before = _files(tmp_path)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert result.changes == ()
        (problem,) = [p for p in result.problems if isinstance(p, E.SkillpkgError)]
        assert isinstance(problem, E.SkillNotFoundError) and problem.skill == "retired-skill"
        assert "--yes" in problem.fix
        assert _files(tmp_path) == before
        assert registry.calls("bundle") == []
        assert project.lock_path.read_bytes() == before[".decimal/skills.lock"]

    def test_yes_migrates_the_rest_and_reports_them(self, tmp_path, registry, client):
        data = _fixture("fleet_v1_lock.json")
        data["skills"]["retired-skill"] = dict(
            data["skills"]["weekly-digest"], skill_id="c0ffee00-0000-4000-8000-000000000098")
        project = _fleet_files(tmp_path, data)
        _seed_fleet(registry, {"skills": {k: v for k, v in data["skills"].items()
                                          if k != "retired-skill"}})
        before = _tree(project.skill_dir("retired-skill"))

        result = migration.migrate(directory=tmp_path, client=client, yes=True)

        assert result.ok
        row = _rows(result)["retired-skill"]
        assert row.outcome == "unresolvable"
        assert any(isinstance(p, E.SkillNotFoundError) for p in result.problems)
        assert sorted(L.load(project).skills) == sorted(FLEET)
        assert "retired-skill" not in M.load(project)
        assert _tree(project.skill_dir("retired-skill")) == before
        assert row.old_folders == (migration.OldFolder(".agents/skills/retired-skill"),)
        (backup,) = _backups(project)
        assert not (backup / "skills" / "retired-skill").exists()
        assert sorted(p.name for p in backup.joinpath("skills").iterdir()) == sorted(FLEET)

    def _both_locks(self, tmp_path, registry, *, npx_files):
        pdf = registry.add_skill("pdf-tools", versions=4)
        project = Project.at(tmp_path)
        _write_json(project.lock_path, {"version": 1, "skills": {"pdf-tools": {
            "version": 1, "content_hash": "e3b0c44298fc", "source": "platform",
            "skill_id": pdf.skill_id, "agents": ["universal"], "scope": "project"}}})
        folder = _old_folder(tmp_path, ".agents/skills/pdf-tools", "v1 copy")
        _write_json(project.npx_lock_path, {"pdf-tools": {
            "version": 3, "sha256": "ab" * 32, "files": npx_files}})
        return project, pdf, folder

    def test_a_skill_in_both_locks_installs_at_the_npx_pin(self, tmp_path, registry, client):
        project, pdf, folder = self._both_locks(
            tmp_path, registry, npx_files=[".claude/skills/pdf-tools/SKILL.md"])
        _old_folder(tmp_path, ".claude/skills/pdf-tools", "npx copy")
        before = _tree(folder)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        rows = [c for c in result.changes if isinstance(c, migration.MigratedSkill)]
        (v1,) = [r for r in rows if r.source == "v1"]
        (npx,) = [r for r in rows if r.source == "npx"]
        assert (npx.outcome, npx.version, npx.spec) == ("installed", 3, "3")
        assert v1.outcome == "duplicate" and "npx" in v1.note
        assert {n: e.spec for n, e in M.load(project).entries.items()} == {"pdf-tools": "3"}
        ((name, entry),) = L.load(project).skills.items()
        assert (entry["pin"], entry["version"], entry["skill_id"]) == ("exact", 3, pdf.skill_id)
        # The duplicate's skill is being installed, so its old folder moves to the backup
        # like any migrating entry's: two copies of one skill must never load side by side.
        (backup,) = _backups(project)
        assert _tree(backup / "skills" / "pdf-tools") == before
        assert v1.old_folders == (migration.OldFolder(
            ".agents/skills/pdf-tools", project.rel(backup / "skills" / "pdf-tools")),)
        assert name == "pdf-tools"
        assert sorted(p.name for p in project.skills_root.iterdir()) == ["pdf-tools"]

    def test_a_folder_that_a_migrating_entry_declares_moves(self, tmp_path, registry, client):
        project, _, folder = self._both_locks(
            tmp_path, registry, npx_files=[".agents/skills/pdf-tools/SKILL.md"])
        before = _tree(folder)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        (backup,) = _backups(project)
        assert _tree(backup / "skills" / "pdf-tools") == before
        assert sorted(L.load(project).skills) == ["pdf-tools"]
        moved = migration.OldFolder(".agents/skills/pdf-tools",
                                    project.rel(backup / "skills" / "pdf-tools"))
        rows = [c for c in result.changes if isinstance(c, migration.MigratedSkill)]
        assert [r.old_folders for r in rows] == [(moved,), (moved,)]

    def test_every_request_says_migrate_and_none_reaches_body(self, tmp_path, registry, client):
        _fleet_files(tmp_path)
        _npx_files(tmp_path)
        _seed_fleet(registry)
        _seed_npx(registry)

        assert migration.migrate(directory=tmp_path, client=client).ok

        assert registry.calls("resolve") and registry.calls("bundle")
        assert {r.json["purpose"] for r in registry.calls("resolve")} == {"migrate"}
        assert {r.query["purpose"] for r in registry.calls("bundle")} == {"migrate"}
        assert registry.calls("body") == []


# ── safety: the trust gate's evaluation, without the question ────────────────


def _rated(registry: FakeRegistry, name: str, *, band: str = "clean", withheld: bool = False,
           relationship: Optional[str] = None) -> None:
    """Make every resolve answer for ``name`` carry this safety (and relationship)."""
    def tamper(data: dict) -> None:
        for item in data["items"]:
            if item.get("ok") and item.get("registry_name") == name:
                item["safety"] = {"band": band, "withheld": withheld}
                if relationship is not None:
                    item["relationship"] = relationship
    registry.tamper_resolve = tamper


class TestSafety:
    def test_a_blocked_answer_is_refused_and_changes_nothing(self, tmp_path, registry, client):
        _fleet_files(tmp_path)
        _seed_fleet(registry)
        _rated(registry, "weekly-digest", band="blocked")
        before = _files(tmp_path)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.exit_code == E.ExitCode.UNAVAILABLE
        (problem,) = [p for p in result.problems if isinstance(p, E.SkillpkgError)]
        assert isinstance(problem, E.SkillBlockedError) and problem.skill == "weekly-digest"
        assert "blocked" in problem.message and "--yes" in problem.fix
        assert _files(tmp_path) == before
        assert registry.calls("bundle") == []

    def test_with_yes_a_blocked_skill_is_left_out_and_its_folder_stays(
            self, tmp_path, registry, client):
        project = _fleet_files(tmp_path)
        _seed_fleet(registry)
        _rated(registry, "weekly-digest", band="blocked")
        before = _tree(project.skill_dir("weekly-digest"))

        result = migration.migrate(directory=tmp_path, client=client, yes=True)

        assert result.ok, result.problems
        row = _rows(result)["weekly-digest"]
        assert row.outcome == "unresolvable" and "blocked" in row.note
        assert row.old_folders == (migration.OldFolder(".agents/skills/weekly-digest"),)
        assert _tree(project.skill_dir("weekly-digest")) == before
        assert "weekly-digest" not in M.load(project)
        assert sorted(L.load(project).skills) == sorted(set(FLEET) - {"weekly-digest"})
        assert any(isinstance(p, E.SkillBlockedError) for p in result.problems)

    @pytest.mark.parametrize("relationship", ["public", None])
    def test_a_withheld_skill_is_refused_unless_owned_or_used(
            self, tmp_path, registry, client, relationship):
        _fleet_files(tmp_path)
        _seed_fleet(registry)
        _rated(registry, "release-notes", withheld=True, relationship=relationship)
        before = _files(tmp_path)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.exit_code == E.ExitCode.UNAVAILABLE
        (problem,) = [p for p in result.problems if isinstance(p, E.SkillpkgError)]
        assert isinstance(problem, E.SkillWithheldError) and problem.skill == "release-notes"
        assert _files(tmp_path) == before

    @pytest.mark.parametrize("relationship", ["owned", "used"])
    def test_a_withheld_skill_the_org_owns_or_uses_installs_with_a_warning(
            self, tmp_path, registry, client, relationship):
        project = _fleet_files(tmp_path)
        _seed_fleet(registry)
        _rated(registry, "release-notes", withheld=True, relationship=relationship)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        assert "release-notes" in L.load(project).skills
        (warning,) = [t for t in _texts(result.problems) if "safety review" in t]
        assert warning.startswith("release-notes:") and "withheld" in warning

    @pytest.mark.parametrize("band", ["caution", "unreviewed"])
    @pytest.mark.parametrize("relationship", ["public", "used", "owned"])
    def test_caution_and_unreviewed_install_with_a_warning_naming_the_band(
            self, tmp_path, registry, client, band, relationship):
        project = _fleet_files(tmp_path)
        _seed_fleet(registry)
        _rated(registry, "incident-triage", band=band, relationship=relationship)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        assert "incident-triage" in L.load(project).skills
        (warning,) = [t for t in _texts(result.problems) if "safety review" in t]
        assert warning.startswith("incident-triage:") and band in warning

    def test_a_clean_skill_gets_no_safety_warning(self, tmp_path, registry, client):
        _fleet_files(tmp_path)
        _seed_fleet(registry)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok and not [t for t in _texts(result.problems) if "safety review" in t]

    def test_a_blocked_skill_in_both_locks_is_refused_for_both(self, tmp_path, registry, client):
        project = Project.at(tmp_path)
        pdf = registry.add_skill("pdf-tools", versions=4)
        _write_json(project.lock_path, {"version": 1, "skills": {"pdf-tools": {
            "version": 1, "skill_id": pdf.skill_id, "agents": ["universal"]}}})
        folder = _old_folder(tmp_path, ".agents/skills/pdf-tools")
        _write_json(project.npx_lock_path, {"pdf-tools": {
            "version": 3, "sha256": "ab" * 32, "files": [".claude/skills/pdf-tools/SKILL.md"]}})
        _rated(registry, "pdf-tools", band="blocked")
        before = _tree(folder)

        result = migration.migrate(directory=tmp_path, client=client, yes=True)

        assert result.ok, result.problems
        rows = [c for c in result.changes if isinstance(c, migration.MigratedSkill)]
        assert [r.outcome for r in rows] == ["unresolvable", "unresolvable"]
        assert _tree(folder) == before  # nothing of that skill installs, so nothing moves
        assert L.load(project).skills == {}

    @pytest.mark.parametrize("band", ["clean", "caution", "unreviewed", "blocked", "brand-new"])
    @pytest.mark.parametrize("withheld", [False, True])
    @pytest.mark.parametrize("relationship", ["owned", "used", "public", None])
    def test_the_verdict_is_the_trust_gates(self, band, withheld, relationship):
        # The trust gate of `add` and `install` decides the same refusals; where it would
        # ask (and refuse without a terminal), the migration warns instead.
        from types import SimpleNamespace

        from decimalai.skillpkg import ops
        from decimalai.skillpkg.client import RenderKey, Resolved, Safety

        resolved = Resolved(
            ref="acme/x", skill_id="sid", registry_name="acme/x", install_name="acme-x",
            version_id="vid", version=3, render=RenderKey(1, "0" * 64, "foreign"),
            body_sha256="0" * 64, description_sha256="0" * 64, files=(),
            relationship=relationship, safety=Safety(band, withheld))
        run = SimpleNamespace(interactive=False, confirm=None)

        refusal, warning = migration._trust(resolved)

        try:
            ops._gate(run, resolved, yes=True)
            gate_refusal = None
        except E.SkillpkgError as exc:
            gate_refusal = type(exc)
        assert {"blocked": E.SkillBlockedError, "withheld": E.SkillWithheldError,
                None: None}[refusal] is gate_refusal
        try:
            ops._gate(run, resolved, yes=False)
            gate_asks = False
        except E.ConfirmationRequiredError:
            gate_asks = True
        except E.SkillpkgError:
            gate_asks = False
        if gate_asks:
            assert warning is not None
        if warning is not None:
            assert refusal is None and (band in warning or "withheld" in warning)
        assert (warning is None) == (refusal is not None or (band == "clean" and not withheld))


# ── the manifest ─────────────────────────────────────────────────────────────


class TestManifest:
    def test_an_existing_manifest_wins(self, tmp_path, registry, client):
        project = _fleet_files(tmp_path)
        _seed_fleet(registry)
        text = (
            "# Our skills. Keep this comment.\n"
            'root = ".agents/skills"\n'
            "\n"
            "[skills]\n"
            '"incident-triage" = "2"   # pinned on purpose\n'
        )
        project.manifest_path.write_text(text, encoding="utf-8")

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        after = project.manifest_path.read_text(encoding="utf-8")
        assert after.startswith(text)
        manifest = M.load(project)
        assert manifest["incident-triage"].spec == "2"
        assert {n: e.spec for n, e in manifest.entries.items() if n != "incident-triage"} == {
            n: "latest" for n in FLEET if n != "incident-triage"}
        entry = L.load(project).skills["incident-triage"]
        assert (entry["version"], entry["pin"]) == (2, "exact")

    def test_a_skill_the_manifest_lists_by_id_keeps_its_line(self, tmp_path, registry, client):
        project = _fleet_files(tmp_path)
        skills = _seed_fleet(registry)
        sid = skills["incident-triage"].skill_id
        text = f'[skills]\n"triage" = {{ version = "1", id = "{sid}" }}\n'
        project.manifest_path.write_text(text, encoding="utf-8")

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        manifest = M.load(project)
        assert "incident-triage" not in manifest
        assert manifest["triage"] == M.ManifestEntry("triage", "1", sid)
        assert project.manifest_path.read_text(encoding="utf-8").startswith(text)
        assert L.load(project).skills["incident-triage"]["version"] == 1

    def test_the_id_is_recorded_when_the_name_alone_finds_another_skill(
            self, tmp_path, registry, client):
        # The recorded skill is another organization's public one; the caller's own
        # organization has a skill of the same name, which a lookup by name prefers.
        data = _fixture("fleet_v1_lock.json")
        project = _fleet_files(tmp_path, data)
        recorded = data["skills"]["release-notes"]["skill_id"]
        registry.add_skill("release-notes", skill_id=recorded, url_slug="release-notes-6f00aa",
                           versions=3)
        _seed_fleet(registry, {"skills": {k: v for k, v in data["skills"].items()
                                          if k != "release-notes"}})
        registry.add_skill("release-notes", org=OWNER_ORG, public=False,
                           url_slug="release-notes-ours")

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        manifest = M.load(project)
        assert manifest["release-notes"] == M.ManifestEntry("release-notes", "latest", recorded)
        assert manifest["incident-triage"].id is None
        assert L.load(project).skills["release-notes"]["skill_id"] == recorded


# ── a second run, and nothing to do ──────────────────────────────────────────


class TestIdempotence:
    def test_a_second_run_is_a_no_op(self, tmp_path, registry, client):
        _fleet_files(tmp_path)
        _npx_files(tmp_path)
        _seed_fleet(registry)
        _seed_npx(registry)
        assert migration.migrate(directory=tmp_path, client=client).ok
        before = _files(tmp_path)
        stamps = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")}
        calls = len(registry.requests)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok and result.changes == () and result.problems == ()
        assert _files(tmp_path) == before
        assert {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*")} == stamps
        assert len(registry.requests) == calls

    def test_a_project_with_no_legacy_lock_is_left_alone(self, tmp_path, client, registry):
        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok and result.changes == ()
        assert list(tmp_path.iterdir()) == []
        assert registry.requests == []

    def test_skills_already_in_the_lock_are_not_fetched_again(self, tmp_path, registry, client):
        project = _npx_files(tmp_path, {"pdf-tools": _fixture("npx_lock.json")["pdf-tools"]})
        _seed_npx(registry)
        assert migration.migrate(directory=tmp_path, client=client).ok
        first = copy.deepcopy(dict(L.load(project).skills["pdf-tools"]))
        _npx_files(tmp_path)  # npx ran again: both skills, pdf-tools at the same version
        bundles = len(registry.calls("bundle"))

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.ok, result.problems
        rows = _rows(result)
        assert rows["pdf-tools"].outcome == "already_installed"
        assert rows["contract-review-6f1c2a"].outcome == "installed"
        assert len(registry.calls("bundle")) == bundles + 1
        lock = L.load(project)
        assert dict(lock.skills["pdf-tools"]) == first
        assert sorted(lock.skills) == ["acme-contract-review", "pdf-tools"]
        assert len(_backups(project)) == 2


# ── a crash at any step ──────────────────────────────────────────────────────


class _Crash(Exception):
    pass


def _crash_at(mp, module, step: str, nth: int) -> None:
    target = migration if module == "migration" else fs
    original = target._checkpoint
    seen = {"n": 0}

    def checkpoint(name: str) -> None:
        original(name)
        if name == step:
            seen["n"] += 1
            if seen["n"] == nth:
                raise _Crash(f"{module}:{step}#{nth}")

    mp.setattr(target, "_checkpoint", checkpoint)


def _mixed(directory: Path) -> Project:
    """A fleet lock with its root folders, an npx lock with folders outside the root, and a
    hand-authored folder in the root."""
    project = _fleet_files(directory)
    _npx_files(directory)
    _old_folder(directory, ".agents/skills/my-own-skill", "hand-authored")
    return project


def _final(project: Project) -> dict:
    lock = L.load(project)  # also proves no legacy lock is left
    return {
        "manifest": project.manifest_path.read_text(encoding="utf-8"),
        "lock": {k: {f: v for f, v in e.items() if f != "installed_at"}
                 for k, e in lock.skills.items()},
        "root": _tree(project.skills_root),
        "outside": _tree(project.directory / ".claude"),
        "plugin": project.plugin_json_path.read_bytes(),
        "staging": project.staging_dir.exists(),
        "journal": project.journal_path.exists(),
    }


def _backup_name(rel: str) -> str:
    """Where the migration keeps a file of the old state, inside its backup folder."""
    if rel == ".decimal/skills.lock":
        return "skills.lock"
    if rel == ".decimal/skills.lock.json":
        return "skills.lock.json"
    assert rel.startswith(".agents/skills/"), rel
    return "skills/" + rel[len(".agents/skills/"):]


def _assert_recoverable(project: Project, before: dict) -> None:
    """Every file of the old state is where it was, or in a backup folder."""
    backups = _backups(project)
    for rel, data in before.items():
        here = project.directory.joinpath(*rel.split("/"))
        if here.is_file() and here.read_bytes() == data:
            continue
        name = _backup_name(rel)
        assert any((b / name).is_file() and (b / name).read_bytes() == data for b in backups), (
            f"{rel} is neither in place nor in a backup")


CRASHES = [
    ("migration", "staged", 1),
    ("migration", "backed_up", 1),
    ("migration", "originals_removed", 1),
    ("migration", "folder_moved", 1),
    ("migration", "folder_moved", 3),
    ("migration", "manifest_written", 1),
    ("migration", "skill_installed", 1),
    ("migration", "skill_installed", 4),
    ("fs", "journal", 2),
    ("fs", "new_moved", 3),
    ("migration", "finished", 1),
]


class TestCrash:
    @pytest.mark.parametrize("module,step,nth", CRASHES, ids=[f"{m}-{s}-{n}" for m, s, n in CRASHES])
    def test_a_crash_leaves_the_old_state_recoverable_and_a_rerun_finishes(
            self, tmp_path, registry, client, module, step, nth):
        _seed_fleet(registry)
        _seed_npx(registry)
        twin = _mixed(tmp_path / "twin")
        assert migration.migrate(directory=twin.directory, client=client).ok
        expected = _final(twin)

        project = _mixed(tmp_path / "crashed")
        before = _files(project.directory)
        with pytest.MonkeyPatch.context() as mp:
            _crash_at(mp, module, step, nth)
            with pytest.raises(_Crash):
                migration.migrate(directory=project.directory, client=client)

        _assert_recoverable(project, before)
        result = migration.migrate(directory=project.directory, client=client)
        assert result.ok, result.problems
        assert _final(project) == expected
        _assert_recoverable(project, before)
        assert not project.process_lock_path.exists()
        assert migration.migrate(directory=project.directory, client=client).changes == ()

    def test_the_legacy_locks_are_copied_before_the_originals_go(
            self, tmp_path, registry, client, monkeypatch):
        # The disk fills up while the backup is written: the originals must still be there.
        _seed_fleet(registry)
        _seed_npx(registry)
        project = _mixed(tmp_path)
        before = _files(tmp_path)
        write_file = Project.write_file

        def failing(self, path, data, **kwargs):
            if Path(path).parent.parent == project.migrated_dir and Path(path).name != ".gitignore":
                raise OSError(28, "No space left on device")
            return write_file(self, path, data, **kwargs)

        monkeypatch.setattr(Project, "write_file", failing)
        with pytest.raises(OSError):
            migration.migrate(directory=tmp_path, client=client)
        monkeypatch.setattr(Project, "write_file", write_file)

        assert project.lock_path.read_bytes() == before[".decimal/skills.lock"]
        assert project.npx_lock_path.read_bytes() == before[".decimal/skills.lock.json"]
        _assert_recoverable(project, before)
        assert migration.migrate(directory=tmp_path, client=client).ok


# ── refusals ─────────────────────────────────────────────────────────────────


class TestRefusals:
    @needs_posix
    def test_a_symlinked_folder_in_the_root_refuses_with_exit_5(self, tmp_path, registry, client):
        project = _fleet_files(tmp_path)
        _seed_fleet(registry)
        target = project.skill_dir("weekly-digest")
        elsewhere = tmp_path / "elsewhere"
        target.rename(elsewhere)
        target.symlink_to(elsewhere, target_is_directory=True)
        before = _files(tmp_path)

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        assert isinstance(result.problems[0], E.IllegalEntryError)
        assert _files(tmp_path) == before
        assert os.path.islink(target)

    def test_a_directory_that_is_not_one_is_a_usage_error(self, tmp_path, client):
        (tmp_path / "file").write_text("x", encoding="utf-8")

        result = migration.migrate(directory=tmp_path / "file", client=client)

        assert result.exit_code == E.ExitCode.USAGE

    def test_a_corrupt_npx_lock_exits_3(self, tmp_path, client):
        project = Project.at(tmp_path)
        project.state_dir.mkdir()
        project.npx_lock_path.write_text("{not json", encoding="utf-8")

        result = migration.migrate(directory=tmp_path, client=client)

        assert result.exit_code == E.ExitCode.LOCK
        assert isinstance(result.problems[0], E.SkillsLockError)
        assert project.npx_lock_path.read_text(encoding="utf-8") == "{not json"

    def test_unsafe_npx_paths_are_never_touched(self, tmp_path, registry, client):
        outside = tmp_path / "outside"
        _old_folder(outside, "pdf-tools", "not the project's")
        before = _tree(outside)
        data = {"pdf-tools": {"version": 3, "sha256": "ab" * 32, "files": [
            "../outside/pdf-tools/SKILL.md", "/etc/pdf-tools/SKILL.md"]}}
        project = Project.at(tmp_path / "project")
        _write_json(project.npx_lock_path, data)
        _seed_npx(registry)

        result = migration.migrate(directory=project.directory, client=client)

        assert result.ok, result.problems
        assert _tree(outside) == before
        assert _rows(result)["pdf-tools"].old_folders == ()
        assert any("../outside/pdf-tools/SKILL.md" in t for t in _texts(result.problems))


# ── the module and its table ─────────────────────────────────────────────────


class TestModule:
    def test_migrate_is_the_package_export(self):
        assert skillpkg.migrate is migration.migrate

    def test_the_agent_folders_are_the_ones_the_old_writer_used(self):
        from decimalai.disk_export import AGENT_PATHS

        assert migration.V1_AGENT_DIRS == {a: p["project"] for a, p in AGENT_PATHS.items()}

    def test_the_table_names_each_skill_its_claim_version_and_old_folder(
            self, tmp_path, registry, client):
        data = _fixture("fleet_v1_lock.json")
        data["skills"]["incident-triage"]["agents"] = ["claude-code", "universal"]
        project = _fleet_files(tmp_path, data)
        _seed_fleet(registry, data)
        _old_folder(tmp_path, ".claude/skills/incident-triage")

        result = migration.migrate(directory=tmp_path, client=client)
        table = migration.format_table(result.changes)

        lines = table.splitlines()
        (backup,) = _backups(project)
        row = next(line for line in lines if line.startswith("incident-triage "))
        assert "v1 version 1 (empty body)" in row
        assert "incident-triage@3" in row
        assert project.rel(backup / "skills" / "incident-triage") in row
        assert ".claude/skills/incident-triage (left in place)" in row
        assert project.rel(backup / "skills.lock") in table
