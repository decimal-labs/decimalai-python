"""Folders on disk: their states, staging, the journaled swap, recovery, the process lock.

The engine's safety rests on four properties, each tested here:

* ``scan_root`` tells a folder the engine wrote (CLEAN) from one someone edited
  (MODIFIED), and never mistakes a symlink or a stray file for a skill (ILLEGAL);
* staging checks every byte against the declared hashes before anything under the
  root changes;
* a crash between any two steps of a swap, followed by recovery, leaves the old
  folder and entry or the new ones, never a folder that looks like a local edit;
* two runs never interleave: ``ProcessLock`` is exclusive, and a dead holder's lock
  is taken over.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest
from skillevaluation import skillfmt

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import fs
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import project as P

REPO_ROOT = Path(__file__).resolve().parents[2]
POSIX = os.name == "posix"
needs_posix = pytest.mark.skipif(not POSIX, reason="POSIX modes and symlinks")


def _id(n: int) -> str:
    return str(uuid.UUID(int=n))


def _bundle(name="acme-refund-policy", *, sid=1, vid=101, version=3,
            body="# Refunds\n\nApply the policy.\n", files=None) -> skillfmt.Bundle:
    if files is None:
        files = [
            skillfmt.Attachment("references/fees.md", "| fee |\n|---|\n"),
            skillfmt.Attachment("scripts/compute.py", "print(1)\n", executable=True),
        ]
    fields = skillfmt.RenderFields(
        skill_id=_id(sid), version_id=_id(vid), version=version, registry_name=f"acme/{name}",
        body=body, description="Apply Acme refund rules.")
    return skillfmt.render_bundle(fields, files, install_name=name, variant="foreign")


def _entry(b: skillfmt.Bundle, *, version=3) -> dict:
    return L.make_entry(
        b.install_name, name=f"acme/{b.install_name}", skill_id=b.skill_id,
        version_id=b.version_id, version=version,
        render={"format": b.format, "hash": b.render_hash, "variant": b.variant},
        body_sha256=b.body_sha256, description_sha256=b.description_sha256,
        files={f.path: f.sha256 for f in b.files},
        executable=[f.path for f in b.files if f.mode == "0755"],
        unavailable=[{"path": u.path, "reason": u.reason} for u in b.unavailable],
        source={"base_url": "https://api.example.test", "org": None},
        pin="latest", invocation=b.invocation, installed_at="2026-10-20T10:00:00Z")


def _contents(b: skillfmt.Bundle) -> dict:
    return {f.path: f.content.encode("utf-8") for f in b.files}


def _v1(*extra: skillfmt.Attachment):
    return _bundle(files=[
        skillfmt.Attachment("references/fees.md", "| fee |\n|---|\n"),
        skillfmt.Attachment("scripts/compute.py", "print(1)\n", executable=True),
        *extra,
    ])


def _v2(*extra: skillfmt.Attachment):
    # A new version: SKILL.md changes, references/fees.md is gone, a template appears.
    return _bundle(vid=102, version=4, body="# Refunds v4\n\nApply the new policy.\n", files=[
        skillfmt.Attachment("scripts/compute.py", "print(2)\n", executable=True),
        skillfmt.Attachment("templates/reply.md", "Dear customer,\n"),
        *extra,
    ])


def _write_folder(project: P.Project, b: skillfmt.Bundle) -> Path:
    folder = project.skill_dir(b.install_name)
    for f in b.files:
        path = folder / f.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f.content.encode("utf-8"))
        path.chmod(int(f.mode, 8))
    return folder


def _installed(tmp_path, *bundles) -> tuple:
    """A project in which ``bundles`` are installed, CLEAN, and locked."""
    project = P.Project.at(tmp_path)
    lock = L.empty()
    for b in bundles:
        _write_folder(project, b)
        lock = lock.with_entry(b.install_name, _entry(b))
    project.skills_root.mkdir(parents=True, exist_ok=True)
    L.save(project, lock)
    L.write_plugin_json(project, lock.lock_hash)
    return project, lock


def _tree(path: Path) -> dict:
    """Every file under ``path``: relative path -> (bytes, executable)."""
    out = {}
    for p in sorted(path.rglob("*")):
        if p.is_file():
            out[p.relative_to(path).as_posix()] = (
                p.read_bytes(), bool(p.stat().st_mode & stat.S_IXUSR) if POSIX else None)
    return out


# ── folder states ────────────────────────────────────────────────────────────


class TestScanRoot:
    def test_clean(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        states = fs.scan_root(project, lock.skills)
        assert list(states) == ["acme-refund-policy"]
        assert states["acme-refund-policy"].state is fs.State.CLEAN
        assert states["acme-refund-policy"].files == ()

    def test_modified_by_an_edit(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skill_dir("acme-refund-policy") / "references" / "fees.md").write_text("x\n")
        st = fs.scan_root(project, lock.skills)["acme-refund-policy"]
        assert st.state is fs.State.MODIFIED
        assert st.edited == ("references/fees.md",)
        assert st.files == ("references/fees.md",)

    def test_modified_by_a_missing_file(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skill_dir("acme-refund-policy") / "scripts" / "compute.py").unlink()
        st = fs.scan_root(project, lock.skills)["acme-refund-policy"]
        assert st.state is fs.State.MODIFIED and st.missing == ("scripts/compute.py",)

    def test_modified_by_an_extra_file(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skill_dir("acme-refund-policy") / "references" / "notes.md").write_text("mine\n")
        st = fs.scan_root(project, lock.skills)["acme-refund-policy"]
        assert st.state is fs.State.MODIFIED and st.extra == ("references/notes.md",)

    @needs_posix
    def test_modified_by_an_exec_bit(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skill_dir("acme-refund-policy") / "references" / "fees.md").chmod(0o755)
        (project.skill_dir("acme-refund-policy") / "scripts" / "compute.py").chmod(0o644)
        st = fs.scan_root(project, lock.skills)["acme-refund-policy"]
        assert st.state is fs.State.MODIFIED
        assert st.mode == ("references/fees.md", "scripts/compute.py")
        assert st.edited == ()

    def test_an_empty_directory_does_not_count(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skill_dir("acme-refund-policy") / "assets").mkdir()
        assert fs.scan_root(project, lock.skills)["acme-refund-policy"].state is fs.State.CLEAN

    def test_an_unavailable_file_may_be_absent_or_present(self, tmp_path):
        b = _bundle(files=[skillfmt.Attachment("assets/logo.png", None)])
        assert [u.path for u in b.unavailable] == ["assets/logo.png"]
        project, lock = _installed(tmp_path, b)
        assert fs.scan_root(project, lock.skills)["acme-refund-policy"].state is fs.State.CLEAN
        logo = project.skill_dir("acme-refund-policy") / "assets" / "logo.png"
        logo.parent.mkdir()
        logo.write_bytes(b"\x89PNG")
        assert fs.scan_root(project, lock.skills)["acme-refund-policy"].state is fs.State.CLEAN

    def test_missing(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        import shutil

        shutil.rmtree(project.skill_dir("acme-refund-policy"))
        assert fs.scan_root(project, lock.skills)["acme-refund-policy"].state is fs.State.MISSING

    def test_no_root_means_every_entry_is_missing(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        import shutil

        shutil.rmtree(project.agents_dir)
        assert {n: s.state for n, s in fs.scan_root(project, lock.skills).items()} == {
            "acme-refund-policy": fs.State.MISSING}

    def test_unmanaged(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        mine = project.skills_root / "my-own-skill"
        mine.mkdir()
        (mine / "SKILL.md").write_text("---\nname: my-own-skill\ndescription: d\n---\nbody\n")
        bare = project.skills_root / "Not A Skill Yet"
        bare.mkdir()
        states = fs.scan_root(project, lock.skills)
        assert list(states) == sorted(states)
        assert states["my-own-skill"].state is fs.State.UNMANAGED
        assert states["Not A Skill Yet"].state is fs.State.UNMANAGED
        assert not states["my-own-skill"].managed

    def test_unmanaged_stamped(self, tmp_path):
        # An orphan of an older install, or a copied managed folder.
        project, lock = _installed(tmp_path)
        b = _v1()
        _write_folder(project, b)
        st = fs.scan_root(project, lock.skills)["acme-refund-policy"]
        assert st.state is fs.State.UNMANAGED_STAMPED
        assert st.skill_id == b.skill_id

    def test_an_unparseable_skill_md_is_plain_unmanaged(self, tmp_path):
        project, lock = _installed(tmp_path)
        bad = project.skills_root / "bad"
        bad.mkdir()
        (bad / "SKILL.md").write_bytes(b"---\nname: [unclosed\n---\n\xff")
        assert fs.scan_root(project, lock.skills)["bad"].state is fs.State.UNMANAGED

    def test_a_file_in_the_root_is_illegal(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skills_root / "README.md").write_text("hi\n")
        st = fs.scan_root(project, lock.skills)["README.md"]
        assert st.state is fs.State.ILLEGAL and "not a directory" in st.reason

    def test_a_locked_name_that_is_a_file_is_illegal(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        import shutil

        shutil.rmtree(project.skill_dir("acme-refund-policy"))
        project.skill_dir("acme-refund-policy").write_text("x")
        assert fs.scan_root(project, lock.skills)["acme-refund-policy"].state is fs.State.ILLEGAL

    @needs_posix
    @pytest.mark.parametrize("where", ["folder", "file", "subdir", "unmanaged-file"])
    def test_a_symlink_anywhere_is_illegal(self, tmp_path, where):
        project, lock = _installed(tmp_path / "p", _v1())
        outside = tmp_path / "outside"
        (outside / "d").mkdir(parents=True)
        (outside / "f.md").write_text("x\n")
        folder = project.skill_dir("acme-refund-policy")
        name = "acme-refund-policy"
        if where == "folder":
            import shutil

            shutil.rmtree(folder)
            folder.symlink_to(outside / "d", target_is_directory=True)
        elif where == "file":
            target = folder / "references" / "fees.md"
            target.unlink()
            target.symlink_to(outside / "f.md")
        elif where == "subdir":
            (folder / "assets").symlink_to(outside / "d", target_is_directory=True)
        else:
            mine = project.skills_root / "mine"
            mine.mkdir()
            (mine / "SKILL.md").symlink_to(outside / "f.md")
            name = "mine"
        st = fs.scan_root(project, lock.skills)[name]
        assert st.state is fs.State.ILLEGAL
        assert "symlink" in st.reason

    @needs_posix
    @pytest.mark.parametrize("component", [".agents", ".agents/skills"])
    def test_a_symlinked_root_is_refused(self, tmp_path, component):
        project, lock = _installed(tmp_path / "p", _v1())
        import shutil

        moved = tmp_path / "moved"
        shutil.move(str(project.directory / component), str(moved))
        (project.directory / component).symlink_to(moved, target_is_directory=True)
        with pytest.raises(E.IllegalEntryError):
            fs.scan_root(project, lock.skills)

    def test_the_root_as_a_file_is_refused(self, tmp_path):
        project = P.Project.at(tmp_path)
        project.agents_dir.mkdir()
        project.skills_root.write_text("x")
        with pytest.raises(E.IllegalEntryError):
            fs.scan_root(project, {})

    def test_the_oracle_project_is_clean(self, tmp_path):
        from tests.native_loaders import oracle

        oracle.write(tmp_path)
        project = P.Project.at(tmp_path)
        lock = L.load(project)
        states = fs.scan_root(project, lock.skills)
        assert {s.state for s in states.values()} == {fs.State.CLEAN}
        assert set(states) == set(lock.skills)


# ── staging ──────────────────────────────────────────────────────────────────


class TestStage:
    def test_writes_exact_bytes_into_staging_only(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        root_before = _tree(project.skills_root)
        b = _v2()
        txid = fs.new_txid()
        staged = fs.stage(project, txid, b.install_name, _contents(b), _entry(b, version=4))
        assert staged == project.staging_dir / txid / b.install_name
        assert {k: v[0] for k, v in _tree(staged).items()} == _contents(b)
        assert _tree(project.skills_root) == root_before

    @needs_posix
    def test_modes_ignore_the_umask(self, tmp_path):
        project = P.Project.at(tmp_path)
        b = _v1()
        old = os.umask(0o077)
        try:
            staged = fs.stage(project, fs.new_txid(), b.install_name, _contents(b), _entry(b))
        finally:
            os.umask(old)
        assert (staged / "scripts" / "compute.py").stat().st_mode & 0o777 == 0o755
        assert (staged / "SKILL.md").stat().st_mode & 0o777 == 0o644
        assert (staged / "references" / "fees.md").stat().st_mode & 0o777 == 0o644

    def test_a_hash_mismatch_is_an_integrity_error_and_writes_nothing(self, tmp_path):
        project = P.Project.at(tmp_path)
        b = _v1()
        contents = _contents(b)
        contents["references/fees.md"] = b"tampered\n"
        txid = fs.new_txid()
        with pytest.raises(E.IntegrityError) as exc:
            fs.stage(project, txid, b.install_name, contents, _entry(b))
        assert exc.value.exit_code == 6
        assert "references/fees.md" in str(exc.value)
        assert not (project.staging_dir / txid / b.install_name).exists()

    @pytest.mark.parametrize("change", ["extra", "missing"])
    def test_the_file_set_must_equal_the_entry(self, tmp_path, change):
        project = P.Project.at(tmp_path)
        b = _v1()
        contents = _contents(b)
        if change == "extra":
            contents["references/more.md"] = b"x\n"
        else:
            del contents["references/fees.md"]
        with pytest.raises(E.BundleContractError):
            fs.stage(project, fs.new_txid(), b.install_name, contents, _entry(b))

    @pytest.mark.parametrize("path", ["../escape.md", "/abs.md", "references/../../x.md",
                                      "references/SKILL.md", "notes.md", "references\\x.md"])
    def test_an_unsafe_path_is_refused(self, tmp_path, path):
        project = P.Project.at(tmp_path)
        b = _v1()
        contents = _contents(b)
        contents[path] = b"x\n"
        entry = _entry(b)
        entry["files"] = dict(entry["files"], **{path: hashlib.sha256(b"x\n").hexdigest()})
        with pytest.raises((E.BundleContractError, E.SkillsLockError)):
            fs.stage(project, fs.new_txid(), b.install_name, contents, entry)
        assert not (tmp_path / "escape.md").exists()
        assert not Path("/abs.md").exists()

    def test_a_skill_md_that_breaks_the_contract_is_refused(self, tmp_path):
        project = P.Project.at(tmp_path)
        b = _v1()
        contents = _contents(b)
        broken = contents["SKILL.md"].replace(b"name: acme-refund-policy", b"name: other")
        contents["SKILL.md"] = broken
        entry = _entry(b)
        entry["files"]["SKILL.md"] = hashlib.sha256(broken).hexdigest()
        with pytest.raises(E.BundleContractError) as exc:
            fs.stage(project, fs.new_txid(), b.install_name, contents, entry)
        assert any(p["rule"] == "F3" for p in exc.value.details["problems"])

    def test_restaging_replaces_a_stale_staged_folder(self, tmp_path):
        project = P.Project.at(tmp_path)
        b = _v1()
        txid = fs.new_txid()
        stale = project.staging_dir / txid / b.install_name / "references" / "old.md"
        stale.parent.mkdir(parents=True)
        stale.write_text("stale\n")
        staged = fs.stage(project, txid, b.install_name, _contents(b), _entry(b))
        assert not (staged / "references" / "old.md").exists()

    @needs_posix
    def test_a_symlinked_staging_dir_is_refused(self, tmp_path):
        project = P.Project.at(tmp_path / "p")
        project.state_dir.mkdir(parents=True)
        (tmp_path / "elsewhere").mkdir()
        project.staging_dir.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
        b = _v1()
        with pytest.raises(E.IllegalEntryError):
            fs.stage(project, fs.new_txid(), b.install_name, _contents(b), _entry(b))
        assert list((tmp_path / "elsewhere").iterdir()) == []

    def test_txids_are_unique_single_components(self):
        ids = {fs.new_txid() for _ in range(50)}
        assert len(ids) == 50
        for txid in ids:
            assert "/" not in txid and "\\" not in txid and not txid.startswith(".")


# ── the swap ─────────────────────────────────────────────────────────────────


def _staged_v2(project):
    b = _v2()
    txid = fs.new_txid()
    fs.stage(project, txid, b.install_name, _contents(b), _entry(b, version=4))
    return txid, b.install_name, _entry(b, version=4)


class TestSwap:
    def test_a_whole_folder_swap_removes_stale_files(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        txid, name, new = _staged_v2(project)
        new_lock = fs.swap(project, txid, name, new, lock)
        folder = project.skill_dir(name)
        assert not (folder / "references").exists(), "a stale file survived the swap"
        assert {k: v[0] for k, v in _tree(folder).items()} == _contents(_v2())
        assert new_lock.skills[name] == new
        assert L.load(project).skills[name] == new
        assert fs.scan_root(project, new_lock.skills)[name].state is fs.State.CLEAN
        assert not project.journal_path.exists()

    def test_fetch_new(self, tmp_path):
        project, lock = _installed(tmp_path)
        txid, name, new = _staged_v2(project)
        new_lock = fs.swap(project, txid, name, new, lock)
        assert fs.scan_root(project, new_lock.skills)[name].state is fs.State.CLEAN

    def test_delete(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        txid = fs.new_txid()
        new_lock = fs.swap(project, txid, "acme-refund-policy", None, lock)
        assert not project.skill_dir("acme-refund-policy").exists()
        assert "acme-refund-policy" not in new_lock.skills
        assert L.load(project).skills == {}

    def test_the_steps_are_the_checkpoints(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(fs, "_checkpoint", seen.append)
        project, lock = _installed(tmp_path, _v1())
        txid, name, new = _staged_v2(project)
        fs.swap(project, txid, name, new, lock)
        assert tuple(seen) == fs.CHECKPOINTS

    def test_a_pending_journal_refuses_a_new_swap(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        txid, name, new = _staged_v2(project)
        project.journal_path.write_text("{}")
        with pytest.raises(E.JournalConflictError):
            fs.swap(project, txid, name, new, lock)
        assert fs.scan_root(project, lock.skills)[name].state is fs.State.CLEAN

    def test_an_unstaged_folder_refuses_before_touching_anything(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        with pytest.raises(E.UsageError):
            fs.swap(project, fs.new_txid(), "acme-refund-policy", _entry(_v2(), version=4), lock)
        assert not project.journal_path.exists()
        assert fs.scan_root(project, lock.skills)["acme-refund-policy"].state is fs.State.CLEAN

    @needs_posix
    def test_a_symlinked_target_is_refused(self, tmp_path):
        project, lock = _installed(tmp_path / "p", _v1())
        txid, name, new = _staged_v2(project)
        import shutil

        shutil.move(str(project.skill_dir(name)), str(tmp_path / "real"))
        project.skill_dir(name).symlink_to(tmp_path / "real", target_is_directory=True)
        with pytest.raises(E.IllegalEntryError):
            fs.swap(project, txid, name, new, lock)
        assert not project.journal_path.exists()
        assert (tmp_path / "real" / "SKILL.md").exists()

    def test_cross_device_falls_back_to_copy_and_verify(self, tmp_path, monkeypatch):
        monkeypatch.setattr(fs, "_same_device", lambda a, b: False)
        renames = []
        real_rename = os.rename

        def spy(src, dst):
            renames.append((Path(src).parent, Path(dst).parent))
            return real_rename(src, dst)

        monkeypatch.setattr(fs.os, "rename", spy)
        project, lock = _installed(tmp_path, _v1())
        txid, name, new = _staged_v2(project)
        new_lock = fs.swap(project, txid, name, new, lock)
        assert fs.scan_root(project, new_lock.skills)[name].state is fs.State.CLEAN
        # Every rename stayed inside one directory: the copy was made beside its target.
        assert renames and all(a == b for a, b in renames)
        assert not any(p.name.endswith(".partial") for p in project.skills_root.iterdir())


class TestBackupAndCleanup:
    def test_backup_copies_the_folder(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skill_dir("acme-refund-policy") / "references" / "fees.md").write_text("mine\n")
        txid = fs.new_txid()
        backup = fs.backup_folder(project, txid, "acme-refund-policy")
        assert backup == project.backup_dir / txid / "acme-refund-policy"
        assert _tree(backup) == _tree(project.skill_dir("acme-refund-policy"))

    def test_discard_staging(self, tmp_path):
        project = P.Project.at(tmp_path)
        b = _v1()
        t1, t2 = fs.new_txid(), fs.new_txid()
        fs.stage(project, t1, b.install_name, _contents(b), _entry(b))
        fs.stage(project, t2, b.install_name, _contents(b), _entry(b))
        fs.discard_staging(project, t1)
        assert sorted(p.name for p in project.staging_dir.iterdir()) == [t2]
        fs.discard_staging(project)
        assert not project.staging_dir.exists()
        fs.discard_staging(project)  # nothing there: fine


# ── crash injection and recovery ─────────────────────────────────────────────


class Crash(BaseException):
    """Stands in for the process dying: not an Exception, so nothing catches it."""


def _scenario(tmp_path, kind):
    """A project, a staged change and the entries before and after it."""
    if kind == "replace":
        project, lock = _installed(tmp_path, _v1())
        txid, name, new = _staged_v2(project)
    elif kind == "fetch_new":
        project, lock = _installed(tmp_path)
        txid, name, new = _staged_v2(project)
    elif kind == "refetch_missing":
        project, lock = _installed(tmp_path, _v1())
        import shutil

        shutil.rmtree(project.skill_dir("acme-refund-policy"))
        b = _v1()
        txid = fs.new_txid()
        name, new = b.install_name, lock.skills[b.install_name]
        fs.stage(project, txid, name, _contents(b), new)
    elif kind == "delete":
        project, lock = _installed(tmp_path, _v1())
        txid, name, new = fs.new_txid(), "acme-refund-policy", None
    elif kind == "carry":
        # The user placed an unavailable (binary) file by hand; the new version still
        # lists it as unavailable, so the swap carries it into the new folder.
        project, lock = _installed(tmp_path, _v1(_logo()))
        _place_logo(project)
        b = _v2(_logo())
        txid, name, new = fs.new_txid(), b.install_name, _entry(b, version=4)
        fs.stage(project, txid, name, _contents(b), new)
    else:
        raise AssertionError(kind)
    old = lock.skills.get(name)
    return project, lock, txid, name, old, new, _folder(project, name, old)


def _folder(project, name, entry):
    """The folder's state against ``entry``: a State, or None when there is no folder."""
    path = project.skill_dir(name)
    if not os.path.lexists(path):
        return None
    if entry is None:
        return fs.State.UNMANAGED
    return fs.check_folder(path, entry).state


def _consistent(project, name, entry, before) -> bool:
    """The lock holds ``entry`` and the folder is as that entry's side of the swap left it:
    the new folder (CLEAN, or gone for a removal), or exactly the state before the swap."""
    lock = L.load(project, missing_ok=True)
    if lock.skills.get(name) != entry:
        return False
    state = _folder(project, name, entry)
    return state in ({fs.State.CLEAN} if entry is not None else {None}) or state == before


def _assert_settled(project, name, old, new, txid, before):
    lock = L.load(project, missing_ok=True)  # strict: the hash recomputes
    entry = lock.skills.get(name)
    assert entry in (old, new), "the lock holds neither the old nor the new entry"
    # The folder is the new one (CLEAN, or gone for a removal), or exactly what was there
    # before the swap began: never anything in between.
    allowed = set()
    if entry == new:
        allowed.add(fs.State.CLEAN if new is not None else None)
    if entry == old:
        allowed.add(before)
    state = _folder(project, name, entry)
    assert state in allowed, f"after recovery the folder is {state}; allowed {allowed}"
    assert state not in (fs.State.MODIFIED, fs.State.ILLEGAL)
    assert not project.journal_path.exists()
    assert not (project.staging_dir / txid).exists()
    assert project.plugin_json_path.read_bytes() == L.plugin_json_bytes(lock.lock_hash)
    assert fs.recover(project) is None  # settling twice is a no-op
    return entry


SCENARIOS = ["replace", "fetch_new", "refetch_missing", "delete", "carry"]


class TestCrashInjection:
    @pytest.mark.parametrize("kind", SCENARIOS)
    @pytest.mark.parametrize("step", fs.CHECKPOINTS)
    def test_a_crash_after_any_step_recovers_to_old_or_new(self, tmp_path, monkeypatch,
                                                         kind, step):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, kind)

        def crash_at(label):
            if label == step:
                raise Crash(label)

        monkeypatch.setattr(fs, "_checkpoint", crash_at)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        # Until recovery runs, the journal says so: a crash never passes for a local edit.
        assert project.journal_path.exists() or _consistent(project, name, new, before) \
            or _consistent(project, name, old, before)
        recovery = fs.recover(project)
        assert recovery is not None and recovery.name == name
        entry = _assert_settled(project, name, old, new, txid, before)
        # The step order decides which: before the new folder is in place, the old one
        # stands; once it is (or, for a removal, once the old one is gone), the new one.
        if old != new:
            assert recovery.outcome == ("new" if entry == new else "old")
            past_the_new_folder = fs.CHECKPOINTS.index(step) >= fs.CHECKPOINTS.index(
                "new_moved") or (new is None and step == "old_moved")
            assert (entry == new) == past_the_new_folder, (step, entry)
        if kind == "carry":
            # Old folder or new, the file the user placed by hand is still there.
            logo = project.skill_dir(name) / "assets" / "logo.png"
            assert logo.read_bytes() == LOGO, step

    @pytest.mark.parametrize("kind", SCENARIOS)
    @pytest.mark.parametrize("which", ["journal", "lock"])
    def test_a_crash_inside_an_atomic_write_recovers(self, tmp_path, monkeypatch, kind, which):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, kind)
        target = project.journal_path if which == "journal" else project.lock_path
        real_replace = os.replace

        def crash_replace(src, dst):
            if Path(dst) == target:
                raise Crash(f"died writing {target.name}")
            return real_replace(src, dst)

        monkeypatch.setattr(P.os, "replace", crash_replace)
        if which == "lock" and old == new:
            # Re-fetching a MISSING folder leaves the lock's bytes as they were, so the
            # lock is not rewritten and there is no write to die in.
            fs.swap(project, txid, name, new, lock)
            monkeypatch.undo()
            fs.discard_staging(project, txid)
            assert _assert_settled(project, name, old, new, txid, before) == new
            return
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        if which == "journal":
            # No journal was written, so nothing had moved: there is nothing to settle,
            # and the next run discards the staged copy.
            assert fs.recover(project) is None
            fs.discard_staging(project)
            assert _assert_settled(project, name, old, new, txid, before) == old
        else:
            # The new folder was in place; only the lock write died.
            assert fs.recover(project).outcome == "new"
            assert _assert_settled(project, name, old, new, txid, before) == new

    def test_a_crash_while_removing_the_journal(self, tmp_path, monkeypatch):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "replace")
        real_unlink = os.unlink

        def crash_unlink(path, *a, **kw):
            if Path(path) == project.journal_path:
                raise Crash("died removing the journal")
            return real_unlink(path, *a, **kw)

        monkeypatch.setattr(fs.os, "unlink", crash_unlink)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        assert fs.recover(project).outcome == "new"
        assert _assert_settled(project, name, old, new, txid, before) == new

    @pytest.mark.parametrize("move", ["old", "new"])
    def test_a_crash_inside_a_cross_device_move_recovers(self, tmp_path, monkeypatch, move):
        # Across devices a move is copy, rename, then remove the source; die before the
        # removal, so the folder exists in both places.
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "replace")
        monkeypatch.setattr(fs, "_same_device", lambda a, b: False)
        source = project.skill_dir(name) if move == "old" else project.staging_dir / txid / name
        aside = source.with_name(f".{source.name}.partial")  # renamed aside, then deleted
        real_rmtree = fs.shutil.rmtree

        def crash_rmtree(path, *a, **kw):
            if Path(path) == aside:
                raise Crash(f"died removing {path}")
            return real_rmtree(path, *a, **kw)

        monkeypatch.setattr(fs.shutil, "rmtree", crash_rmtree)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        assert fs.recover(project).outcome == move
        assert _assert_settled(project, name, old, new, txid, before) == (
            old if move == "old" else new)

    @pytest.mark.parametrize("step", fs.CHECKPOINTS)
    def test_a_killed_process_recovers(self, tmp_path, step):
        """The real thing: the child dies with os._exit, so no cleanup code runs."""
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "replace")
        new_path = tmp_path / "new_entry.json"
        new_path.write_text(json.dumps(new))
        code = textwrap.dedent("""
            import json, os, sys
            from decimalai.skillpkg import fs, lock as L, project as P
            directory, step, txid, name, new_path = sys.argv[1:]
            def die(label):
                if label == step:
                    os._exit(41)
            fs._checkpoint = die
            project = P.Project.at(directory)
            fs.swap(project, txid, name, json.loads(open(new_path).read()), L.load(project))
            os._exit(0)
        """)
        child = subprocess.run(
            [sys.executable, "-c", code, str(project.directory), step, txid, name, str(new_path)],
            cwd=REPO_ROOT, env=_child_env(), capture_output=True, text=True, timeout=120)
        assert child.returncode == 41, child.stderr
        assert project.journal_path.exists() or _consistent(project, name, new, before)
        fs.recover(project)
        _assert_settled(project, name, old, new, txid, before)


def _damage_lock(project, damage):
    """What a non-frozen install may find and repair: a merged entry, or a stale hash."""
    data = json.loads(project.lock_path.read_bytes())
    if damage == "foreign":
        # What SDK 0.13.9 merges into whatever lock it finds.
        data["skills"]["legacy-skill"] = {"source": "registry", "hash": "sha256:abc",
                                          "path": ".agents/skills/legacy-skill"}
    else:  # a hand edit that leaves lock_hash stale
        data["skills"]["acme-refund-policy"]["pin"] = "exact"
    project.lock_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


class TestRecoveryWithALockThatNeededRepair:
    @pytest.mark.parametrize("damage", ["foreign", "hash"])
    @pytest.mark.parametrize("step", fs.CHECKPOINTS)
    def test_recovery_settles_and_keeps_the_damage_for_the_next_command(
            self, tmp_path, monkeypatch, damage, step):
        project, _ = _installed(tmp_path, _v1())
        damaged = _damage_lock(project, damage)
        lock = L.load(project, repair=True)        # a non-frozen install, then a crash
        txid, name, new = _staged_v2(project)

        def crash_at(label):
            if label == step:
                raise Crash(label)

        monkeypatch.setattr(fs, "_checkpoint", crash_at)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        recovery = fs.recover(project)              # must not refuse the damaged lock
        assert recovery is not None
        assert not project.journal_path.exists()
        repaired = L.load(project, repair=True)
        entry = repaired.skills[name]
        assert fs.check_folder(project.skill_dir(name), entry).state is fs.State.CLEAN
        assert (entry == new) == (recovery.outcome == "new")
        if step == "lock_written":
            # The crashed run had already written its repaired lock.
            assert L.load(project).hash_ok
            return
        # Before its lock write, the damage is still there for `install` to report.
        if damage == "foreign":
            with pytest.raises(E.ForeignLockEntriesError):
                L.load(project)
            assert "legacy-skill" in json.loads(project.lock_path.read_bytes())["skills"]
        else:
            with pytest.raises(E.LockHashMismatch):
                L.load(project)
            assert json.loads(project.lock_path.read_bytes())["lock_hash"] == \
                damaged["lock_hash"], "recovery blessed a hand edit by recomputing lock_hash"

    def test_a_crash_with_no_lock_at_all(self, tmp_path, monkeypatch):
        project, lock = _installed(tmp_path)
        project.lock_path.unlink()
        txid, name, new = _staged_v2(project)

        def crash_at(label):
            if label == "new_moved":
                raise Crash(label)

        monkeypatch.setattr(fs, "_checkpoint", crash_at)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, L.empty())
        monkeypatch.undo()
        assert fs.recover(project).outcome == "new"
        assert L.load(project).skills[name] == new


class TestOnlyCopies:
    def _orphaned_old(self, tmp_path, monkeypatch):
        """A crash after the old folder moved to .old, then the journal deleted by hand."""
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "replace")

        def crash_at(label):
            if label == "old_moved":
                raise Crash(label)

        monkeypatch.setattr(fs, "_checkpoint", crash_at)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        project.journal_path.unlink()
        return project, txid, name

    @pytest.mark.parametrize("txid_given", [False, True])
    def test_discard_staging_never_deletes_the_only_copy_of_a_locked_folder(
            self, tmp_path, monkeypatch, txid_given):
        project, txid, name = self._orphaned_old(tmp_path, monkeypatch)
        old_copy = project.staging_dir / txid / ".old" / name
        assert not project.skill_dir(name).exists() and old_copy.is_dir()
        assert fs.recover(project) is None
        with pytest.raises(E.JournalConflictError) as exc:
            fs.discard_staging(project, txid if txid_given else None)
        assert old_copy.is_dir() and (old_copy / "SKILL.md").is_file()
        assert project.rel(old_copy) in exc.value.message
        assert project.rel(old_copy) in exc.value.fix and ".agents/skills" in exc.value.fix

    def test_a_copy_the_root_also_holds_is_discarded(self, tmp_path, monkeypatch):
        project, txid, name = self._orphaned_old(tmp_path, monkeypatch)
        old_copy = project.staging_dir / txid / ".old" / name
        import shutil

        shutil.copytree(old_copy, project.skill_dir(name))
        fs.discard_staging(project)
        assert not project.staging_dir.exists()

    def test_a_copy_of_something_the_lock_no_longer_lists_is_discarded(self, tmp_path):
        # A removal leaves the removed folder in .old; the lock has no entry for it.
        project, lock = _installed(tmp_path, _v1())
        txid = fs.new_txid()
        fs.swap(project, txid, "acme-refund-policy", None, lock)
        assert (project.staging_dir / txid / ".old" / "acme-refund-policy").is_dir()
        fs.discard_staging(project, txid)
        assert not project.staging_dir.exists()


class TestCrossDeviceRemoval:
    """Across devices a move is copy, rename into place, then remove the source. The
    source is renamed aside in one step before it is deleted, so a crash never leaves a
    half-deleted folder where a skill belongs."""

    def test_a_crash_while_deleting_the_old_folder_keeps_every_copy_needed(
            self, tmp_path, monkeypatch):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "carry")
        monkeypatch.setattr(fs, "_same_device", lambda a, b: False)
        target = project.skill_dir(name)
        aside = project.skills_root / f".{name}.partial"
        real_rmtree = fs.shutil.rmtree
        seen = []

        def partial_rmtree(path, *a, **kw):
            seen.append(Path(path))
            if Path(path) in (target, aside):
                # delete one file, then die, as a kill -9 inside rmtree would
                (Path(path) / "assets" / "logo.png").unlink()
                raise Crash("died while removing the old folder")
            return real_rmtree(path, *a, **kw)

        monkeypatch.setattr(fs.shutil, "rmtree", partial_rmtree)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        assert target not in seen, "the old folder was deleted in place, not renamed aside"
        assert not target.exists()
        old_copy = project.staging_dir / txid / ".old" / name
        assert (old_copy / "assets" / "logo.png").read_bytes() == LOGO
        recovery = fs.recover(project)
        assert recovery.outcome == "old"
        assert (target / "assets" / "logo.png").read_bytes() == LOGO
        assert _assert_settled(project, name, old, new, txid, before) == old

    def test_recovery_prefers_a_complete_old_copy(self, tmp_path, monkeypatch):
        # Both copies exist (the crash came after the copy, before the source was moved
        # aside), and the one in the root has since lost a hand-placed file. It still
        # reads as the old version, but the copy in .old is the complete one.
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "carry")
        monkeypatch.setattr(fs, "_same_device", lambda a, b: False)
        target = project.skill_dir(name)
        real_rename = os.rename

        def crash_rename(src, dst):
            if Path(src) == target:
                raise Crash("died before moving the old folder aside")
            return real_rename(src, dst)

        monkeypatch.setattr(fs.os, "rename", crash_rename)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        (target / "assets" / "logo.png").unlink()
        assert fs.check_folder(target, old).state is fs.State.CLEAN
        assert fs.recover(project).outcome == "old"
        assert (target / "assets" / "logo.png").read_bytes() == LOGO
        assert _assert_settled(project, name, old, new, txid, before) == old

    def test_identical_copies_keep_the_folder_in_the_root(self, tmp_path, monkeypatch):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "carry")
        monkeypatch.setattr(fs, "_same_device", lambda a, b: False)
        target = project.skill_dir(name)
        inode = target.stat().st_ino
        real_rename = os.rename

        def crash_rename(src, dst):
            if Path(src) == target:
                raise Crash("died before moving the old folder aside")
            return real_rename(src, dst)

        monkeypatch.setattr(fs.os, "rename", crash_rename)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        assert fs.recover(project).outcome == "old"
        assert target.stat().st_ino == inode


class TestRecovery:
    def test_no_journal_is_nothing_to_do(self, tmp_path):
        project, _ = _installed(tmp_path, _v1())
        assert fs.recover(project) is None

    def test_a_folder_matching_neither_is_a_conflict_and_nothing_is_touched(
            self, tmp_path, monkeypatch):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "replace")

        def crash_at(label):
            if label == "old_moved":
                raise Crash(label)

        monkeypatch.setattr(fs, "_checkpoint", crash_at)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        # Someone puts their own folder where the skill was.
        mine = project.skill_dir(name)
        mine.mkdir()
        (mine / "SKILL.md").write_text("mine\n")
        journal = project.journal_path.read_bytes()
        lock_bytes = project.lock_path.read_bytes()
        staging = _tree(project.staging_dir)
        with pytest.raises(E.JournalConflictError) as exc:
            fs.recover(project)
        assert exc.value.exit_code == 1
        assert exc.value.skill == name
        assert project.journal_path.read_bytes() == journal
        assert project.lock_path.read_bytes() == lock_bytes
        assert _tree(project.staging_dir) == staging
        assert (mine / "SKILL.md").read_text() == "mine\n"

    def test_an_edit_after_the_crash_is_a_conflict(self, tmp_path, monkeypatch):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "fetch_new")

        def crash_at(label):
            if label == "new_moved":
                raise Crash(label)

        monkeypatch.setattr(fs, "_checkpoint", crash_at)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        (project.skill_dir(name) / "SKILL.md").write_text("edited\n")
        with pytest.raises(E.JournalConflictError):
            fs.recover(project)

    @pytest.mark.parametrize("raw", [b"", b"{", b"[]", b'{"txid": "t"}',
                                     b'{"txid": "../x", "name": "a", "old_entry": null, '
                                     b'"new_entry": null, "old_present": false}'])
    def test_an_unreadable_journal_is_a_conflict(self, tmp_path, raw):
        project, _ = _installed(tmp_path, _v1())
        project.journal_path.write_bytes(raw)
        with pytest.raises(E.JournalConflictError):
            fs.recover(project)
        assert project.journal_path.read_bytes() == raw

    def test_read_journal(self, tmp_path, monkeypatch):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "replace")
        assert fs.read_journal(project) is None

        def crash_at(label):
            if label == "journal":
                raise Crash(label)

        monkeypatch.setattr(fs, "_checkpoint", crash_at)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        journal = fs.read_journal(project)
        assert journal == fs.Journal(txid=txid, name=name, old_entry=old, new_entry=new,
                                     old_present=True)


# ── OS clutter, hand-placed files, crash debris ─────────────────────────────

LOGO = b"\x89PNG\r\n\x1a\n hand-placed"


def _logo(*, served: bool = False) -> skillfmt.Attachment:
    """assets/logo.png: unavailable (a binary the registry does not store), or served."""
    return skillfmt.Attachment("assets/logo.png", "served logo\n" if served else None)


def _place_logo(project, data: bytes = LOGO, mode: int = 0o640) -> Path:
    logo = project.skill_dir("acme-refund-policy") / "assets" / "logo.png"
    logo.parent.mkdir(exist_ok=True)
    logo.write_bytes(data)
    logo.chmod(mode)
    return logo


def _clutter(folder: Path) -> list:
    """Drop OS and tool clutter into ``folder``; returns what was made."""
    made = []
    for rel in (".DS_Store", "Thumbs.db", "desktop.ini", "references/.DS_Store",
                "scripts/__pycache__/compute.cpython-312.pyc", "scripts/compute.pyc",
                "references/deep/__pycache__/x/y.txt"):
        path = folder / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"clutter")
        made.append(rel)
    return made


class TestClutter:
    def test_the_list_is_one_constant(self):
        assert fs.CLUTTER == (".DS_Store", "Thumbs.db", "desktop.ini", "__pycache__/", "*.pyc")

    def test_clutter_inside_a_skill_folder_is_ignored(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        _clutter(project.skill_dir("acme-refund-policy"))
        st = fs.scan_root(project, lock.skills)["acme-refund-policy"]
        assert st.state is fs.State.CLEAN, st
        assert fs.check_folder(project.skill_dir("acme-refund-policy"),
                               lock.skills["acme-refund-policy"]).state is fs.State.CLEAN

    def test_clutter_in_the_root_is_not_reported(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        for rel in (".DS_Store", "Thumbs.db", "desktop.ini", "stale.pyc", "__pycache__/x.pyc"):
            path = project.skills_root / rel
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"clutter")
        assert list(fs.scan_root(project, lock.skills)) == ["acme-refund-policy"]

    def test_clutter_inside_an_unmanaged_folder_is_ignored_too(self, tmp_path):
        project, lock = _installed(tmp_path)
        mine = project.skills_root / "mine"
        mine.mkdir()
        (mine / "SKILL.md").write_text("---\nname: mine\ndescription: d\n---\nbody\n")
        _clutter(mine)
        assert fs.scan_root(project, lock.skills)["mine"].state is fs.State.UNMANAGED

    @pytest.mark.parametrize("rel", ["notes.DS_Store", "DS_Store", "references/x.pyc.txt",
                                     "references/pycache/x.md", "scripts/__pycache__.md",
                                     "references/Thumbs.db.bak"])
    def test_only_the_exact_names_count(self, tmp_path, rel):
        project, lock = _installed(tmp_path, _v1())
        path = project.skill_dir("acme-refund-policy") / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mine")
        st = fs.scan_root(project, lock.skills)["acme-refund-policy"]
        assert st.state is fs.State.MODIFIED and st.extra == (rel,)

    def test_a_directory_named_like_a_clutter_file_is_not_clutter(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skill_dir("acme-refund-policy") / ".DS_Store").mkdir()
        (project.skill_dir("acme-refund-policy") / ".DS_Store" / "x.md").write_bytes(b"x")
        st = fs.scan_root(project, lock.skills)["acme-refund-policy"]
        assert st.state is fs.State.MODIFIED and st.extra == (".DS_Store/x.md",)

    @needs_posix
    def test_a_symlink_named_like_clutter_is_still_illegal(self, tmp_path):
        project, lock = _installed(tmp_path / "p", _v1())
        (tmp_path / "f").write_text("x")
        (project.skill_dir("acme-refund-policy") / ".DS_Store").symlink_to(tmp_path / "f")
        (project.skills_root / "Thumbs.db").symlink_to(tmp_path / "f")
        states = fs.scan_root(project, lock.skills)
        assert states["acme-refund-policy"].state is fs.State.ILLEGAL
        assert states["Thumbs.db"].state is fs.State.ILLEGAL

    def test_a_swap_drops_clutter_with_the_old_folder(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        _clutter(project.skill_dir("acme-refund-policy"))
        txid, name, new = _staged_v2(project)
        new_lock = fs.swap(project, txid, name, new, lock)
        assert {k: v[0] for k, v in _tree(project.skill_dir(name)).items()} == _contents(_v2())
        assert fs.scan_root(project, new_lock.skills)[name].state is fs.State.CLEAN

    def test_a_served_clutter_path_is_refused(self, tmp_path):
        # It could never be checked after it was written: the comparison ignores it.
        project = P.Project.at(tmp_path)
        b = _bundle(files=[skillfmt.Attachment("scripts/__pycache__/x.cpython-312.pyc", "x\n")])
        with pytest.raises(E.BundleContractError, match="__pycache__"):
            fs.stage(project, fs.new_txid(), b.install_name, _contents(b), _entry(b))


class TestHandPlacedUnavailableFiles:
    def test_carried_into_the_new_folder_with_its_mode(self, tmp_path):
        project, lock = _installed(tmp_path, _v1(_logo()))
        _place_logo(project, mode=0o640)
        b = _v2(_logo())
        txid, warned = fs.new_txid(), []
        fs.stage(project, txid, b.install_name, _contents(b), _entry(b, version=4))
        new_lock = fs.swap(project, txid, b.install_name, _entry(b, version=4), lock,
                           on_warning=warned.append)
        logo = project.skill_dir(b.install_name) / "assets" / "logo.png"
        assert logo.read_bytes() == LOGO
        if POSIX:
            assert logo.stat().st_mode & 0o777 == 0o640
        assert fs.scan_root(project, new_lock.skills)[b.install_name].state is fs.State.CLEAN
        assert warned == []
        assert not project.backup_dir.exists()

    def test_served_bytes_win_and_the_hand_placed_copy_is_backed_up(self, tmp_path, caplog):
        project, lock = _installed(tmp_path, _v1(_logo()))
        _place_logo(project)
        b = _v2(_logo(served=True))
        entry = _entry(b, version=4)
        assert "assets/logo.png" in entry["files"]
        txid = fs.new_txid()
        fs.stage(project, txid, b.install_name, _contents(b), entry)
        with caplog.at_level("WARNING", logger="decimalai.skillpkg"):
            new_lock = fs.swap(project, txid, b.install_name, entry, lock)
        folder = project.skill_dir(b.install_name)
        assert (folder / "assets" / "logo.png").read_bytes() == b"served logo\n"
        assert fs.scan_root(project, new_lock.skills)[b.install_name].state is fs.State.CLEAN
        backup = project.backup_dir / txid / b.install_name / "assets" / "logo.png"
        assert backup.read_bytes() == LOGO
        messages = [r.getMessage() for r in caplog.records if r.name == "decimalai.skillpkg"]
        assert len(messages) == 1
        assert "assets/logo.png" in messages[0] and project.rel(backup) in messages[0]

    def test_the_warning_goes_to_the_callback_when_one_is_given(self, tmp_path, caplog):
        project, lock = _installed(tmp_path, _v1(_logo()))
        _place_logo(project)
        b = _v2(_logo(served=True))
        txid, warned = fs.new_txid(), []
        fs.stage(project, txid, b.install_name, _contents(b), _entry(b, version=4))
        with caplog.at_level("WARNING", logger="decimalai.skillpkg"):
            fs.swap(project, txid, b.install_name, _entry(b, version=4), lock,
                    on_warning=warned.append)
        assert len(warned) == 1 and "assets/logo.png" in warned[0]
        assert not [r for r in caplog.records if r.name == "decimalai.skillpkg"]

    def test_a_hand_placed_copy_equal_to_the_served_bytes_needs_no_backup(self, tmp_path):
        project, lock = _installed(tmp_path, _v1(_logo()))
        _place_logo(project, data=b"served logo\n", mode=0o644)
        b = _v2(_logo(served=True))
        txid, warned = fs.new_txid(), []
        fs.stage(project, txid, b.install_name, _contents(b), _entry(b, version=4))
        fs.swap(project, txid, b.install_name, _entry(b, version=4), lock,
                on_warning=warned.append)
        assert warned == [] and not project.backup_dir.exists()

    @needs_posix
    def test_a_symlink_inside_the_old_folder_is_refused_before_anything_moves(self, tmp_path):
        project, lock = _installed(tmp_path / "p", _v1(_logo()))
        (tmp_path / "secret").write_text("not yours")
        logo = project.skill_dir("acme-refund-policy") / "assets" / "logo.png"
        logo.parent.mkdir()
        logo.symlink_to(tmp_path / "secret")
        b = _v2(_logo())
        txid = fs.new_txid()
        fs.stage(project, txid, b.install_name, _contents(b), _entry(b, version=4))
        with pytest.raises(E.IllegalEntryError):
            fs.swap(project, txid, b.install_name, _entry(b, version=4), lock)
        assert not project.journal_path.exists()
        assert os.path.islink(logo)
        assert not (project.staging_dir / txid / b.install_name / "assets").exists()

    def test_nothing_to_carry_when_the_file_was_never_placed(self, tmp_path):
        project, lock = _installed(tmp_path, _v1(_logo()))
        b = _v2(_logo())
        txid = fs.new_txid()
        fs.stage(project, txid, b.install_name, _contents(b), _entry(b, version=4))
        fs.swap(project, txid, b.install_name, _entry(b, version=4), lock)
        assert not (project.skill_dir(b.install_name) / "assets" / "logo.png").exists()

    def test_a_file_the_new_entry_no_longer_lists_is_backed_up_and_named(self, tmp_path):
        # v2 no longer has the logo at all: it is not carried into the folder, but the
        # hand-placed copy is saved and the warning says where.
        project, lock = _installed(tmp_path, _v1(_logo()))
        _place_logo(project)
        txid, name, new = _staged_v2(project)
        warned = []
        fs.swap(project, txid, name, new, lock, on_warning=warned.append)
        assert not (project.skill_dir(name) / "assets" / "logo.png").exists()
        backup = project.backup_dir / txid / name / "assets" / "logo.png"
        assert backup.read_bytes() == LOGO
        assert len(warned) == 1
        assert "assets/logo.png" in warned[0] and project.rel(backup) in warned[0]

    def test_a_users_own_extra_file_is_not_treated_as_hand_placed(self, tmp_path):
        # Only paths the old entry listed as unavailable are hand-placed files; any other
        # extra file makes the folder MODIFIED, which the caller backs up as a whole.
        project, lock = _installed(tmp_path, _v1())
        (project.skill_dir("acme-refund-policy") / "references" / "mine.md").write_text("x\n")
        txid, name, new = _staged_v2(project)
        warned = []
        fs.swap(project, txid, name, new, lock, on_warning=warned.append)
        assert warned == [] and not project.backup_dir.exists()


class TestCrashDebris:
    def _debris(self, project, name="acme-refund-policy") -> Path:
        partial = project.skills_root / f".{name}.partial"
        (partial / "references").mkdir(parents=True)
        (partial / "SKILL.md").write_text("half a copy")
        return partial

    def test_a_partial_copy_is_not_reported(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        self._debris(project)
        self._debris(project, "another-skill")
        assert list(fs.scan_root(project, lock.skills)) == ["acme-refund-policy"]

    @pytest.mark.parametrize("name", [".Acme.partial", ".acme--x.partial", ".partial",
                                      "acme.partial", ".acme.partial.old", "..partial"])
    def test_only_the_exact_pattern_is_debris(self, tmp_path, name):
        project, lock = _installed(tmp_path)
        (project.skills_root / name).mkdir()
        assert fs.scan_root(project, lock.skills)[name].state is fs.State.UNMANAGED
        fs.discard_staging(project)
        assert fs.recover(project) is None
        assert (project.skills_root / name).is_dir()

    def test_a_file_matching_the_pattern_is_still_illegal(self, tmp_path):
        project, lock = _installed(tmp_path)
        (project.skills_root / ".acme.partial").write_text("x")
        assert fs.scan_root(project, lock.skills)[".acme.partial"].state is fs.State.ILLEGAL
        fs.discard_staging(project)
        assert (project.skills_root / ".acme.partial").exists()

    @needs_posix
    def test_a_symlink_matching_the_pattern_is_still_illegal_and_kept(self, tmp_path):
        project, lock = _installed(tmp_path / "p")
        (tmp_path / "d").mkdir()
        (project.skills_root / ".acme.partial").symlink_to(tmp_path / "d",
                                                           target_is_directory=True)
        assert fs.scan_root(project, lock.skills)[".acme.partial"].state is fs.State.ILLEGAL
        fs.discard_staging(project)
        assert os.path.islink(project.skills_root / ".acme.partial")
        assert (tmp_path / "d").is_dir()

    @pytest.mark.parametrize("txid", [None, "given"])
    def test_discard_staging_removes_it(self, tmp_path, txid):
        project, _ = _installed(tmp_path, _v1())
        partial = self._debris(project)
        fs.discard_staging(project, fs.new_txid() if txid else None)
        assert not partial.exists()
        assert project.skill_dir("acme-refund-policy").is_dir()

    def test_recover_removes_it_even_with_no_journal(self, tmp_path):
        project, _ = _installed(tmp_path, _v1())
        partial = self._debris(project)
        assert fs.recover(project) is None
        assert not partial.exists()

    def test_a_crash_inside_a_cross_device_copy_leaves_debris_that_recovery_removes(
            self, tmp_path, monkeypatch):
        project, lock, txid, name, old, new, before = _scenario(tmp_path, "replace")
        monkeypatch.setattr(fs, "_same_device", lambda a, b: False)
        real_copytree = fs.shutil.copytree

        def crash_copytree(src, dst, *a, **kw):
            if Path(dst).parent == project.skills_root:
                Path(dst).mkdir()
                (Path(dst) / "SKILL.md").write_text("half")
                raise Crash("died mid-copy")
            return real_copytree(src, dst, *a, **kw)

        monkeypatch.setattr(fs.shutil, "copytree", crash_copytree)
        with pytest.raises(Crash):
            fs.swap(project, txid, name, new, lock)
        monkeypatch.undo()
        partial = project.skills_root / f".{name}.partial"
        assert partial.is_dir()
        assert f".{name}.partial" not in fs.scan_root(project, L.load(project).skills)
        assert fs.recover(project).outcome == "old"
        assert not partial.exists()
        assert _assert_settled(project, name, old, new, txid, before) == old


def _case_sensitive(directory: Path) -> bool:
    probe = directory / "case-probe"
    probe.mkdir()
    try:
        return not (directory / "CASE-PROBE").exists()
    finally:
        probe.rmdir()


class TestCaseInsensitiveNames:
    """On a macOS or Windows checkout, folders whose names differ only by case are one
    folder. Such names are ILLEGAL: never MISSING, UNMANAGED or CLEAN, so the caller
    refuses to fetch, replace or name anything there."""

    def test_a_locked_name_present_only_under_another_case(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        import shutil

        shutil.move(str(project.skill_dir("acme-refund-policy")),
                    str(project.skills_root / "Acme-Refund-Policy-tmp"))
        os.rename(project.skills_root / "Acme-Refund-Policy-tmp",
                  project.skills_root / "Acme-Refund-Policy")
        states = fs.scan_root(project, lock.skills)
        assert states["acme-refund-policy"].state is fs.State.ILLEGAL
        assert "Acme-Refund-Policy" in states["acme-refund-policy"].reason
        assert states["Acme-Refund-Policy"].state is fs.State.ILLEGAL
        assert "case" in states["Acme-Refund-Policy"].reason

    def _two_cases(self, project, monkeypatch, first, second):
        """Folders ``first`` and ``second`` side by side: real on a case-sensitive
        filesystem, and simulated in the root listing on a case-insensitive one."""
        (project.skills_root / first).mkdir(exist_ok=True)
        if _case_sensitive(project.skills_root):
            (project.skills_root / second).mkdir()
            return
        real = fs._root_children

        def listing(root):
            return real(root) + [(second, stat.S_IFDIR | 0o755)]

        monkeypatch.setattr(fs, "_root_children", listing)

    def test_a_locked_folder_beside_a_case_twin(self, tmp_path, monkeypatch):
        project, lock = _installed(tmp_path, _v1())
        self._two_cases(project, monkeypatch, "acme-refund-policy", "Acme-Refund-Policy")
        states = fs.scan_root(project, lock.skills)
        assert states["acme-refund-policy"].state is fs.State.ILLEGAL
        assert states["Acme-Refund-Policy"].state is fs.State.ILLEGAL
        assert "Acme-Refund-Policy" in states["acme-refund-policy"].reason

    def test_two_unmanaged_case_twins(self, tmp_path, monkeypatch):
        project, lock = _installed(tmp_path)
        self._two_cases(project, monkeypatch, "notes", "Notes")
        states = fs.scan_root(project, lock.skills)
        assert states["notes"].state is fs.State.ILLEGAL
        assert states["Notes"].state is fs.State.ILLEGAL

    def test_distinct_names_are_untouched(self, tmp_path):
        project, lock = _installed(tmp_path, _v1())
        (project.skills_root / "Acme-Refund-Policies").mkdir()
        states = fs.scan_root(project, lock.skills)
        assert states["acme-refund-policy"].state is fs.State.CLEAN
        assert states["Acme-Refund-Policies"].state is fs.State.UNMANAGED


# ── the process lock ─────────────────────────────────────────────────────────


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _plant(project, content: str, *, age: float = 0.0) -> None:
    project.state_dir.mkdir(parents=True, exist_ok=True)
    project.process_lock_path.write_text(content)
    if age:
        then = time.time() - age
        os.utime(project.process_lock_path, (then, then))


class TestProcessLock:
    def test_acquire_writes_the_pid_and_release_removes_it(self, tmp_path):
        project = P.Project.at(tmp_path)
        lock = fs.ProcessLock(project)
        with lock:
            assert lock.held
            assert project.process_lock_path.read_text().split()[0] == str(os.getpid())
        assert not lock.held
        assert not project.process_lock_path.exists()

    def test_a_second_holder_waits_then_fails_busy(self, tmp_path):
        project = P.Project.at(tmp_path)
        with fs.ProcessLock(project):
            start = time.monotonic()
            with pytest.raises(E.SkillsBusyError) as exc:
                fs.ProcessLock(project, timeout=0.3).acquire()
            assert time.monotonic() - start >= 0.3
            assert exc.value.exit_code == 1
            assert str(os.getpid()) in exc.value.message

    def test_a_live_holder_is_never_taken_over(self, tmp_path):
        project = P.Project.at(tmp_path)
        _plant(project, f"{os.getppid()} sometoken\n")
        with pytest.raises(E.SkillsBusyError):
            fs.ProcessLock(project, timeout=0.2).acquire()
        assert project.process_lock_path.read_text() == f"{os.getppid()} sometoken\n"

    def test_a_dead_holder_is_taken_over(self, tmp_path):
        project = P.Project.at(tmp_path)
        _plant(project, f"{_dead_pid()} oldtoken\n")
        with fs.ProcessLock(project, timeout=1.0) as lock:
            assert lock.held
            assert project.process_lock_path.read_text().split()[0] == str(os.getpid())

    def test_an_old_lock_is_taken_over_even_with_a_live_pid(self, tmp_path):
        project = P.Project.at(tmp_path)
        _plant(project, f"{os.getppid()} oldtoken\n", age=11 * 60)
        with fs.ProcessLock(project, timeout=1.0) as lock:
            assert lock.held

    def test_our_own_pid_left_by_a_dead_predecessor_is_stale(self, tmp_path):
        # A container restarts with the same pid; the lock on the volume is not ours.
        project = P.Project.at(tmp_path)
        _plant(project, f"{os.getpid()} notours\n")
        with fs.ProcessLock(project, timeout=1.0) as lock:
            assert lock.held

    def test_unreadable_content_is_stale_only_by_age(self, tmp_path):
        project = P.Project.at(tmp_path)
        _plant(project, "")
        with pytest.raises(E.SkillsBusyError):
            fs.ProcessLock(project, timeout=0.2).acquire()
        _plant(project, "garbage", age=11 * 60)
        with fs.ProcessLock(project, timeout=1.0) as lock:
            assert lock.held

    def test_release_leaves_a_lock_that_is_not_ours(self, tmp_path):
        project = P.Project.at(tmp_path)
        lock = fs.ProcessLock(project).acquire()
        project.process_lock_path.write_text(f"{os.getppid()} someone-else\n")
        lock.release()
        assert project.process_lock_path.read_text() == f"{os.getppid()} someone-else\n"

    def test_refresh_after_a_takeover_lets_go_and_says_so(self, tmp_path):
        project = P.Project.at(tmp_path)
        a = fs.ProcessLock(project, stale_after=600).acquire()
        # A's run stalls (a laptop asleep, SIGSTOP, a hung request) past stale_after.
        then = time.time() - 700
        os.utime(project.process_lock_path, (then, then))
        b = fs.ProcessLock(project, timeout=1.0).acquire()
        assert b.held
        b_lock = project.process_lock_path.read_text()
        b_mtime = project.process_lock_path.stat().st_mtime_ns
        with pytest.raises(E.SkillsBusyError):
            a.refresh()
        assert not a.held
        assert project.process_lock_path.read_text() == b_lock
        assert project.process_lock_path.stat().st_mtime_ns == b_mtime, "refreshed B's lock"
        a.release()  # a no-op now: B's lock stays
        assert project.process_lock_path.read_text() == b_lock
        b.release()

    def test_refresh_after_the_lock_vanished(self, tmp_path):
        project = P.Project.at(tmp_path)
        a = fs.ProcessLock(project).acquire()
        project.process_lock_path.unlink()
        with pytest.raises(E.SkillsBusyError):
            a.refresh()
        assert not a.held

    def test_refresh_keeps_a_long_run_from_looking_stale(self, tmp_path):
        project = P.Project.at(tmp_path)
        lock = fs.ProcessLock(project).acquire()
        then = time.time() - 11 * 60
        os.utime(project.process_lock_path, (then, then))
        lock.refresh()
        assert time.time() - project.process_lock_path.stat().st_mtime < 60
        lock.release()

    @needs_posix
    def test_a_symlinked_lock_path_is_refused(self, tmp_path):
        project = P.Project.at(tmp_path)
        project.state_dir.mkdir()
        (tmp_path / "target").write_text("")
        project.process_lock_path.symlink_to(tmp_path / "target")
        with pytest.raises(E.IllegalEntryError):
            fs.ProcessLock(project, timeout=0.2).acquire()

    @pytest.mark.parametrize("stale_first", [False, True])
    def test_exclusive_across_processes(self, tmp_path, stale_first):
        project = P.Project.at(tmp_path)
        if stale_first:
            _plant(project, f"{_dead_pid()} dead\n")
        log = tmp_path / "log.txt"
        code = textwrap.dedent("""
            import os, sys, time
            from decimalai.skillpkg import fs, project as P
            project = P.Project.at(sys.argv[1])
            for _ in range(3):
                with fs.ProcessLock(project, timeout=60, poll=0.005):
                    with open(sys.argv[2], "a") as fh:
                        fh.write(f"+{os.getpid()}\\n")
                    time.sleep(0.02)
                    with open(sys.argv[2], "a") as fh:
                        fh.write(f"-{os.getpid()}\\n")
        """)
        children = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path), str(log)],
                                     cwd=REPO_ROOT, env=_child_env(),
                                     stderr=subprocess.PIPE, text=True)
                    for _ in range(4)]
        for child in children:
            _, err = child.communicate(timeout=120)
            assert child.returncode == 0, err
        lines = log.read_text().split()
        assert len(lines) == 4 * 3 * 2
        for enter, leave in zip(lines[::2], lines[1::2]):
            assert enter[0] == "+" and leave == "-" + enter[1:], lines
        assert not project.process_lock_path.exists()


def _child_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("DECIMAL")}
    env.update({"DECIMAL_AUTOINIT": "false", "DECIMALAI_SKIP_VERIFY": "1",
                "DECIMAL_API_KEY": "", "DECIMAL_AUTO_TRACE": "",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO_ROOT)})
    return env
