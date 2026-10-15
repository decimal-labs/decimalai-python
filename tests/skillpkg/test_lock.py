"""``.decimal/skills.lock`` (version 2) and ``.agents/.claude-plugin/plugin.json``.

The lock records exactly what the engine wrote, so it is read strictly and never
reset: a corrupt file, a version 1 lock, npx's lock, entries an older decimalai
merged in, and a hand edit that breaks ``lock_hash`` are each a named error. It is
written atomically, sorted and newline-terminated, and only when its content changes,
so a commit diff shows exactly what an install changed.
"""
from __future__ import annotations

import copy
import json
import os
import uuid
from pathlib import Path

import pytest
from skillevaluation import skillfmt

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import project as P

INSTALLED_AT = "2026-10-20T10:00:00Z"
V0139_LOCK = (Path(__file__).resolve().parents[1]
              / "native_loaders" / "fixtures" / "v0139_project" / "decimal" / "skills.lock")


def _id(n: int) -> str:
    return str(uuid.UUID(int=n))


def _bundle(install_name="acme-refund-policy", *, registry="acme/refund-policy", sid=1,
            vid=101, version=3, body="# Refunds\n\nApply the policy.\n",
            description="Apply Acme refund rules.", files=None) -> skillfmt.Bundle:
    if files is None:
        files = [
            skillfmt.Attachment("references/fees.md", "| fee |\n|---|\n"),
            skillfmt.Attachment("scripts/compute.py", "print(1)\n", executable=True),
        ]
    fields = skillfmt.RenderFields(
        skill_id=_id(sid), version_id=_id(vid), version=version, registry_name=registry,
        body=body, description=description)
    return skillfmt.render_bundle(fields, files, install_name=install_name, variant="foreign")


def _entry(bundle: skillfmt.Bundle, *, registry="acme/refund-policy", version=3,
           pin="latest", installed_at=INSTALLED_AT) -> dict:
    return L.make_entry(
        bundle.install_name,
        name=registry,
        skill_id=bundle.skill_id,
        version_id=bundle.version_id,
        version=version,
        render={"format": bundle.format, "hash": bundle.render_hash, "variant": bundle.variant},
        body_sha256=bundle.body_sha256,
        description_sha256=bundle.description_sha256,
        files={f.path: f.sha256 for f in bundle.files},
        executable=[f.path for f in bundle.files if f.mode == "0755"],
        unavailable=[{"path": u.path, "reason": u.reason} for u in bundle.unavailable],
        source={"base_url": "https://api.example.test", "org": None},
        pin=pin,
        invocation=bundle.invocation,
        installed_at=installed_at,
    )


def _two_skill_lock() -> L.Lock:
    a = _entry(_bundle())
    b = _entry(_bundle("pdf-tools", registry="pdf-tools", sid=2, vid=202, version=1),
               registry="pdf-tools", version=1, pin="exact")
    return L.empty().with_entry("acme-refund-policy", a).with_entry("pdf-tools", b)


def _project(tmp_path) -> P.Project:
    project = P.Project.at(tmp_path)
    project.state_dir.mkdir(parents=True, exist_ok=True)
    return project


def _write_raw(project: P.Project, data) -> None:
    raw = data if isinstance(data, bytes) else (json.dumps(data, indent=2) + "\n").encode()
    project.lock_path.write_bytes(raw)


# ── entries ──────────────────────────────────────────────────────────────────


class TestMakeEntry:
    def test_the_documented_shape(self):
        entry = _entry(_bundle())
        assert set(entry) == {
            "name", "skill_id", "version_id", "version", "render", "body_sha256",
            "description_sha256", "path", "files", "executable", "unavailable", "source",
            "pin", "invocation", "installed_at",
        }
        assert entry["path"] == ".agents/skills/acme-refund-policy"
        assert entry["executable"] == ["scripts/compute.py"]
        assert set(entry["files"]) == {"SKILL.md", "references/fees.md", "scripts/compute.py"}

    def test_installed_at_defaults_to_now_in_utc(self):
        entry = _entry(_bundle(), installed_at=None)
        assert entry["installed_at"].endswith("Z") and len(entry["installed_at"]) == 20

    def test_lists_are_sorted(self):
        b = _bundle(files=[
            skillfmt.Attachment("scripts/b.py", "1\n", executable=True),
            skillfmt.Attachment("scripts/a.py", "2\n", executable=True),
        ])
        assert _entry(b)["executable"] == ["scripts/a.py", "scripts/b.py"]

    def test_a_bad_entry_is_refused(self):
        with pytest.raises(E.SkillsLockError):
            L.make_entry("Not A Name", **{k: v for k, v in _entry(_bundle()).items()
                                          if k not in ("path",)})


class TestInstalledAt:
    def test_kept_when_version_render_and_files_are_unchanged(self):
        old = _entry(_bundle(), installed_at="2026-01-01T00:00:00Z")
        new = _entry(_bundle(), installed_at="2026-10-20T10:00:00Z", pin="exact")
        assert L.keep_installed_at(new, old)["installed_at"] == "2026-01-01T00:00:00Z"
        assert L.keep_installed_at(new, old)["pin"] == "exact"

    @pytest.mark.parametrize("change", ["version_id", "render", "files"])
    def test_renewed_when_any_of_them_changes(self, change):
        old = _entry(_bundle(), installed_at="2026-01-01T00:00:00Z")
        new = copy.deepcopy(_entry(_bundle(), installed_at="2026-10-20T10:00:00Z"))
        if change == "version_id":
            new["version_id"] = _id(999)
        elif change == "render":
            new["render"]["hash"] = "0" * 64
        else:
            new["files"]["references/fees.md"] = "1" * 64
        assert L.keep_installed_at(new, old)["installed_at"] == "2026-10-20T10:00:00Z"

    def test_without_an_old_entry_the_new_one_stands(self):
        new = _entry(_bundle())
        assert L.keep_installed_at(new, None) == new

    def test_neither_argument_is_mutated(self):
        old = _entry(_bundle(), installed_at="2026-01-01T00:00:00Z")
        new = _entry(_bundle())
        before = copy.deepcopy((old, new))
        L.keep_installed_at(new, old)
        assert (old, new) == before


# ── the Lock value ───────────────────────────────────────────────────────────


class TestLockValue:
    def test_lock_hash_is_skillfmts(self):
        lock = _two_skill_lock()
        assert lock.lock_hash == skillfmt.lock_hash(".agents/skills", dict(lock.skills))
        assert L.empty().lock_hash == skillfmt.lock_hash(".agents/skills", {})

    def test_installed_at_is_not_part_of_the_hash(self):
        lock = _two_skill_lock()
        entry = dict(lock.skills["pdf-tools"], installed_at="1999-01-01T00:00:00Z")
        assert lock.with_entry("pdf-tools", entry).lock_hash == lock.lock_hash

    def test_any_other_field_is(self):
        lock = _two_skill_lock()
        entry = dict(lock.skills["pdf-tools"], pin="latest")
        assert lock.with_entry("pdf-tools", entry).lock_hash != lock.lock_hash

    def test_with_entry_returns_a_new_lock(self):
        lock = _two_skill_lock()
        smaller = lock.with_entry("pdf-tools", None)
        assert set(lock.skills) == {"acme-refund-policy", "pdf-tools"}
        assert set(smaller.skills) == {"acme-refund-policy"}
        assert smaller.with_entry("never-there", None).skills == smaller.skills

    def test_by_skill_id(self):
        lock = _two_skill_lock()
        assert lock.by_skill_id(_id(2))[0] == "pdf-tools"
        assert lock.by_skill_id(_id(77)) is None

    def test_plugin_version(self):
        lock = _two_skill_lock()
        assert lock.plugin_version == "lock-" + lock.lock_hash[:12]


# ── writing ──────────────────────────────────────────────────────────────────


class TestSave:
    def test_sorted_indented_newline_terminated(self, tmp_path):
        project = _project(tmp_path)
        lock = _two_skill_lock()
        assert L.save(project, lock) is True
        raw = project.lock_path.read_bytes()
        data = json.loads(raw)
        assert raw == (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
        assert raw.endswith(b"}\n") and not raw.endswith(b"\n\n")
        assert data == {"version": 2, "root": ".agents/skills",
                        "lock_hash": lock.lock_hash, "skills": json.loads(json.dumps(lock.skills))}

    def test_round_trip(self, tmp_path):
        project = _project(tmp_path)
        lock = _two_skill_lock()
        L.save(project, lock)
        back = L.load(project)
        assert dict(back.skills) == dict(lock.skills)
        assert back.lock_hash == lock.lock_hash
        assert L.dumps(back) == project.lock_path.read_bytes()

    def test_rewritten_only_when_the_content_changes(self, tmp_path):
        project = _project(tmp_path)
        lock = _two_skill_lock()
        L.save(project, lock)
        before = project.lock_path.stat()
        assert L.save(project, L.load(project)) is False
        after = project.lock_path.stat()
        assert (before.st_ino, before.st_mtime_ns) == (after.st_ino, after.st_mtime_ns)
        assert L.save(project, lock.with_entry("pdf-tools", None)) is True

    def test_atomic(self, tmp_path, monkeypatch):
        project = _project(tmp_path)
        L.save(project, _two_skill_lock())
        before = project.lock_path.read_bytes()

        def crash(src, dst):
            raise OSError("crash between write and rename")

        monkeypatch.setattr(P.os, "replace", crash)
        with pytest.raises(OSError):
            L.save(project, L.empty())
        assert project.lock_path.read_bytes() == before
        assert sorted(p.name for p in project.state_dir.iterdir()) == ["skills.lock"]

    def test_creates_the_state_dir(self, tmp_path):
        project = P.Project.at(tmp_path)
        L.save(project, L.empty())
        assert L.load(project).skills == {}

    @pytest.mark.skipif(os.name != "posix", reason="symlinks")
    def test_refuses_a_symlinked_state_dir(self, tmp_path):
        project = P.Project.at(tmp_path / "p")
        project.directory.mkdir()
        (tmp_path / "elsewhere").mkdir()
        project.state_dir.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
        with pytest.raises(E.IllegalEntryError):
            L.save(project, L.empty())
        with pytest.raises(E.IllegalEntryError):
            L.load(project)
        assert list((tmp_path / "elsewhere").iterdir()) == []


# ── reading ──────────────────────────────────────────────────────────────────


class TestLoadRefusals:
    def test_missing(self, tmp_path):
        project = _project(tmp_path)
        with pytest.raises(E.SkillsNotInstalledError):
            L.load(project)
        assert L.load(project, missing_ok=True).skills == {}

    @pytest.mark.parametrize("raw", [b"", b"{", b'{"version": 2, "root"', b"\xff\xfe", b"[]",
                                     b'"lock"', b"null"])
    def test_corrupt_is_an_error_never_a_reset(self, tmp_path, raw):
        project = _project(tmp_path)
        project.lock_path.write_bytes(raw)
        for kwargs in ({}, {"missing_ok": True}, {"repair": True}):
            with pytest.raises(E.SkillsLockError) as exc:
                L.load(project, **kwargs)
            assert type(exc.value) is E.SkillsLockError
            assert ".decimal/skills.lock" in str(exc.value)
            assert exc.value.exit_code == 3
        assert project.lock_path.read_bytes() == raw

    @pytest.mark.parametrize("data", [
        {"version": 1, "skills": {}},
        {"skills": {"acme/refund-policy": {"version": 3}}},
        {},
    ])
    def test_version_1_or_none_is_legacy(self, tmp_path, data):
        project = _project(tmp_path)
        _write_raw(project, data)
        with pytest.raises(E.LegacyLockError) as exc:
            L.load(project, repair=True)
        assert "decimalai skills migrate" in str(exc.value)

    def test_the_real_0139_lock_is_legacy(self, tmp_path):
        project = _project(tmp_path)
        project.lock_path.write_bytes(V0139_LOCK.read_bytes())
        with pytest.raises(E.LegacyLockError):
            L.load(project)

    @pytest.mark.parametrize("with_v2", [False, True])
    def test_the_npx_lock_is_legacy(self, tmp_path, with_v2):
        project = _project(tmp_path)
        if with_v2:
            L.save(project, _two_skill_lock())
        project.npx_lock_path.write_text('{"skills": {"acme-refund-policy": {"version": 3}}}\n')
        for kwargs in ({}, {"missing_ok": True}):
            with pytest.raises(E.LegacyLockError) as exc:
                L.load(project, **kwargs)
            assert "skills.lock.json" in exc.value.message

    @pytest.mark.parametrize("version", [3, "2", 2.0, True])
    def test_an_unknown_version_is_refused(self, tmp_path, version):
        project = _project(tmp_path)
        _write_raw(project, {"version": version, "root": ".agents/skills", "skills": {}})
        with pytest.raises(E.SkillsLockError) as exc:
            L.load(project)
        assert type(exc.value) is E.SkillsLockError
        if version == 3:
            assert "pip install -U decimalai" in exc.value.fix

    def test_another_root_is_refused(self, tmp_path):
        project = _project(tmp_path)
        _write_raw(project, {"version": 2, "root": ".claude/skills",
                             "lock_hash": skillfmt.lock_hash(".claude/skills", {}), "skills": {}})
        with pytest.raises(E.SkillsLockError, match=r"\.claude/skills"):
            L.load(project)

    def test_skills_must_be_a_map(self, tmp_path):
        project = _project(tmp_path)
        _write_raw(project, {"version": 2, "root": ".agents/skills", "skills": []})
        with pytest.raises(E.SkillsLockError):
            L.load(project)


class TestForeignEntries:
    """What SDK 0.13.9 does to a version 2 lock: it reads any JSON, merges its own
    version 1 entries into ``skills`` and writes the file back."""

    def _merged(self, project) -> list:
        L.save(project, _two_skill_lock())
        lockdata = json.loads(project.lock_path.read_text())       # 0.13.9's _read_lockfile
        v1 = json.loads(V0139_LOCK.read_text())["skills"]
        lockdata["skills"].update(v1)                               # export_skills_to_disk
        lockdata["updated_at"] = "2026-10-21T00:00:00+00:00"        # _write_lockfile
        project.lock_path.write_text(json.dumps(lockdata, indent=2))
        return sorted(v1)

    def test_detected_and_named(self, tmp_path):
        project = _project(tmp_path)
        names = self._merged(project)
        with pytest.raises(E.ForeignLockEntriesError) as exc:
            L.load(project)
        assert exc.value.exit_code == 3
        assert exc.value.details["entries"] == names
        for name in names:
            assert name in exc.value.message

    def test_repair_drops_them_and_keeps_the_rest(self, tmp_path):
        project = _project(tmp_path)
        names = self._merged(project)
        lock = L.load(project, repair=True)
        assert lock.dropped == tuple(names)
        assert set(lock.skills) == {"acme-refund-policy", "pdf-tools"}
        # The version 2 entries were not touched, so the stored hash recomputes.
        assert lock.hash_ok
        assert lock.lock_hash == _two_skill_lock().lock_hash

    @pytest.mark.parametrize("missing", ["version_id", "files", "render"])
    def test_an_entry_lacking_any_marker_is_foreign(self, tmp_path, missing):
        project = _project(tmp_path)
        lock = _two_skill_lock()
        entry = dict(lock.skills["pdf-tools"])
        del entry[missing]
        skills = dict(lock.skills, **{"pdf-tools": entry})
        _write_raw(project, {"version": 2, "root": ".agents/skills",
                             "lock_hash": skillfmt.lock_hash(".agents/skills", skills),
                             "skills": skills})
        with pytest.raises(E.ForeignLockEntriesError) as exc:
            L.load(project)
        assert exc.value.details["entries"] == ["pdf-tools"]


class TestLockHash:
    def test_a_hand_edit_is_a_mismatch(self, tmp_path):
        project = _project(tmp_path)
        L.save(project, _two_skill_lock())
        data = json.loads(project.lock_path.read_text())
        data["skills"]["pdf-tools"]["pin"] = "latest"
        _write_raw(project, data)
        with pytest.raises(E.LockHashMismatch) as exc:
            L.load(project)
        assert exc.value.exit_code == 6
        repaired = L.load(project, repair=True)
        assert not repaired.hash_ok
        assert repaired.stored_hash == data["lock_hash"]
        assert repaired.skills["pdf-tools"]["pin"] == "latest"

    def test_a_missing_hash_is_a_mismatch(self, tmp_path):
        project = _project(tmp_path)
        L.save(project, _two_skill_lock())
        data = json.loads(project.lock_path.read_text())
        del data["lock_hash"]
        _write_raw(project, data)
        with pytest.raises(E.LockHashMismatch):
            L.load(project)

    def test_saving_a_repaired_lock_recomputes_it(self, tmp_path):
        project = _project(tmp_path)
        L.save(project, _two_skill_lock())
        data = json.loads(project.lock_path.read_text())
        data["lock_hash"] = "0" * 64
        _write_raw(project, data)
        L.save(project, L.load(project, repair=True))
        assert L.load(project).hash_ok


class TestEntryValidation:
    """A hand-edited lock must not steer a write or a read outside a skill folder."""

    def _write(self, project, mutate):
        lock = _two_skill_lock()
        skills = copy.deepcopy(dict(lock.skills))
        name = mutate(skills) or "acme-refund-policy"
        hashable = all(isinstance(e, dict) for e in skills.values())
        _write_raw(project, {"version": 2, "root": ".agents/skills",
                             "lock_hash": skillfmt.lock_hash(".agents/skills", skills)
                             if hashable else "0" * 64,
                             "skills": skills})
        return name

    @pytest.mark.parametrize("label, mutate", [
        ("key not an install name", lambda s: s.__setitem__("Acme/Refund", s.pop("pdf-tools"))),
        ("key a path", lambda s: s.__setitem__("../x", s.pop("pdf-tools"))),
        ("path elsewhere", lambda s: s["pdf-tools"].__setitem__("path", ".agents/skills/x")),
        ("files not a map", lambda s: s["pdf-tools"].__setitem__("files", ["SKILL.md"])),
        ("no SKILL.md", lambda s: s["pdf-tools"]["files"].pop("SKILL.md")),
        ("unsafe file", lambda s: s["pdf-tools"]["files"].__setitem__("../../etc/x", "0" * 64)),
        ("absolute file", lambda s: s["pdf-tools"]["files"].__setitem__("/etc/x", "0" * 64)),
        ("dotdot file", lambda s: s["pdf-tools"]["files"].__setitem__(
            "references/../../x", "0" * 64)),
        ("nested SKILL.md", lambda s: s["pdf-tools"]["files"].__setitem__(
            "references/SKILL.md", "0" * 64)),
        ("bad file hash", lambda s: s["pdf-tools"]["files"].__setitem__("SKILL.md", "abc")),
        ("executable not a file", lambda s: s["pdf-tools"].__setitem__(
            "executable", ["scripts/none.py"])),
        ("unsafe unavailable", lambda s: s["pdf-tools"].__setitem__(
            "unavailable", [{"path": "../x", "reason": "binary"}])),
        ("unavailable not a list", lambda s: s["pdf-tools"].__setitem__("unavailable", {})),
        ("format 2", lambda s: s["pdf-tools"]["render"].__setitem__("format", 2)),
        ("bad variant", lambda s: s["pdf-tools"]["render"].__setitem__("variant", "mine")),
        ("short render hash", lambda s: s["pdf-tools"]["render"].__setitem__("hash", "ab")),
        ("version 0", lambda s: s["pdf-tools"].__setitem__("version", 0)),
        ("version string", lambda s: s["pdf-tools"].__setitem__("version", "3")),
        ("version bool", lambda s: s["pdf-tools"].__setitem__("version", True)),
        ("bad pin", lambda s: s["pdf-tools"].__setitem__("pin", "3")),
        ("invocation any", lambda s: s["pdf-tools"].__setitem__("invocation", "any")),
        ("empty skill_id", lambda s: s["pdf-tools"].__setitem__("skill_id", "")),
        ("empty name", lambda s: s["pdf-tools"].__setitem__("name", "")),
        ("bad body hash", lambda s: s["pdf-tools"].__setitem__("body_sha256", "x")),
        ("source not a map", lambda s: s["pdf-tools"].__setitem__("source", "api")),
        ("entry not a map", lambda s: s.__setitem__("pdf-tools", "1.0")),
        ("one skill twice", lambda s: s["pdf-tools"].__setitem__(
            "skill_id", s["acme-refund-policy"]["skill_id"])),
    ])
    def test_refused_by_name(self, tmp_path, label, mutate):
        project = _project(tmp_path)
        self._write(project, mutate)
        with pytest.raises(E.SkillsLockError) as exc:
            L.load(project, repair=True)
        assert type(exc.value) is E.SkillsLockError, label
        assert exc.value.skill, f"{label}: the error names no entry"

    def test_unknown_fields_are_kept_and_hashed(self, tmp_path):
        project = _project(tmp_path)
        self._write(project, lambda s: s["pdf-tools"].__setitem__("future_field", [1]))
        lock = L.load(project)
        assert lock.skills["pdf-tools"]["future_field"] == [1]


# ── agreement with the native-loader oracle ──────────────────────────────────


class TestOracle:
    def test_the_oracle_lock_loads_and_dumps_byte_for_byte(self, tmp_path):
        from tests.native_loaders import oracle

        oracle.write(tmp_path)
        project = P.Project.at(tmp_path)
        lock = L.load(project)
        assert lock.hash_ok
        assert L.dumps(lock) == project.lock_path.read_bytes()
        assert L.plugin_json_bytes(lock.lock_hash) == project.plugin_json_path.read_bytes()


# ── plugin.json ──────────────────────────────────────────────────────────────


class TestPluginJson:
    def test_bytes(self):
        lock = _two_skill_lock()
        raw = L.plugin_json_bytes(lock.lock_hash)
        assert raw.endswith(b"}\n") and b"\n" not in raw[:-1]
        data = json.loads(raw)
        assert list(data) == ["name", "version", "description", "author"]
        assert data["name"] == "decimalai-skills" == P.PLUGIN_NAME
        assert data["version"] == "lock-" + lock.lock_hash[:12]

    def test_written_only_when_the_hash_changes(self, tmp_path):
        project = P.Project.at(tmp_path)
        lock = _two_skill_lock()
        assert L.write_plugin_json(project, lock.lock_hash) is True
        assert project.plugin_json_path.read_bytes() == L.plugin_json_bytes(lock.lock_hash)
        before = project.plugin_json_path.stat()
        assert L.write_plugin_json(project, lock.lock_hash) is False
        assert project.plugin_json_path.stat().st_mtime_ns == before.st_mtime_ns
        assert L.write_plugin_json(project, L.empty().lock_hash) is True

    @pytest.mark.parametrize("content", [
        b'{"name": "my-tools", "version": "1.0.0"}\n',                       # another plugin
        b'{"name": "decimalai-skills", "version": "x", "hooks": {}}\n',      # a key it never writes
        b"not json at all\n",
        b"[1, 2]\n",
    ])
    def test_a_plugin_json_it_did_not_write_is_refused(self, tmp_path, content):
        project = P.Project.at(tmp_path)
        project.plugin_json_path.parent.mkdir(parents=True)
        project.plugin_json_path.write_bytes(content)
        assert L.plugin_json_is_ours(project) is False
        with pytest.raises(E.LocalChangesError) as excinfo:
            L.write_plugin_json(project, L.empty().lock_hash)
        assert excinfo.value.exit_code == E.ExitCode.LOCAL_CHANGES
        assert "--force" in excinfo.value.fix
        assert project.plugin_json_path.read_bytes() == content

    def test_replace_foreign_replaces_it(self, tmp_path):
        project = P.Project.at(tmp_path)
        project.plugin_json_path.parent.mkdir(parents=True)
        project.plugin_json_path.write_bytes(b'{"name": "my-tools"}\n')
        assert L.write_plugin_json(project, L.empty().lock_hash, replace_foreign=True) is True
        assert project.plugin_json_path.read_bytes() == L.plugin_json_bytes(L.empty().lock_hash)

    @pytest.mark.parametrize("content", [
        None,                                                               # absent
        b'{"name": "decimalai-skills", "version": "lock-000000000000"}\n',  # an older lock
        b'{"name":"decimalai-skills","version":"lock-1","description":"d","author":{"name":"x"}}',
    ])
    def test_an_absent_or_engine_written_file_is_ours(self, tmp_path, content):
        project = P.Project.at(tmp_path)
        if content is not None:
            project.plugin_json_path.parent.mkdir(parents=True)
            project.plugin_json_path.write_bytes(content)
        assert L.plugin_json_is_ours(project) is True
        assert L.write_plugin_json(project, L.empty().lock_hash) is True

    @pytest.mark.skipif(os.name != "posix", reason="symlinks")
    def test_refuses_a_symlinked_agents_dir(self, tmp_path):
        project = P.Project.at(tmp_path / "p")
        project.directory.mkdir()
        (tmp_path / "elsewhere").mkdir()
        project.agents_dir.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
        with pytest.raises(E.IllegalEntryError):
            L.write_plugin_json(project, L.empty().lock_hash)
