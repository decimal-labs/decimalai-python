"""``skills install`` (no argument): make the folders and the lock agree with the manifest.

Pinned here:

- a new manifest entry is resolved and added; a lock entry the manifest no longer lists
  is removed when its folder is CLEAN;
- a MISSING folder is fetched at its locked ``version_id`` and render, never resolved;
- ``install`` never moves a pin or a render, and makes no network call when every folder
  is CLEAN; a spec moved by hand is resolved at the new spec;
- a MODIFIED folder it must touch is refused (exit 5) unless ``force=True``, which backs
  it up first; one it does not need to touch is reported and left alone;
- a version 1 or npx lock is exit 3, and so is a missing or unreadable manifest;
- entries an older decimalai merged into the lock are dropped (their folders stay,
  unmanaged) and a stale ``lock_hash`` is recomputed, each with a warning.
"""
from __future__ import annotations

import json
import shutil

import pytest

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import fs
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import ops
from tests.skillpkg.test_ops_add import (
    Env,
    actions,
    error_of,
    notices,
    skill_md,
)


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)


def _installed(env: Env, *names: str, versions: int = 1) -> None:
    for name in names:
        env.skill(name, versions=versions)
        assert env.add(name).ok


class TestReconcile:
    def test_new_manifest_entries_are_resolved_and_added(self, env):
        env.skill("acme/refund-policy")
        env.skill("acme/returns", versions=2)
        env.write_manifest({"acme/refund-policy": "latest", "acme/returns": "1"})
        result = env.install()
        assert result.ok, result.problems
        assert sorted(actions(result)) == [("add", "acme-refund-policy"), ("add", "acme-returns")]
        lock = env.lock()
        assert lock.skills["acme-returns"]["version"] == 1
        assert lock.skills["acme-returns"]["pin"] == "exact"
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN,
                                "acme-returns": fs.State.CLEAN}
        (resolve,) = env.calls("resolve")
        assert resolve.json["purpose"] == "install"
        assert len(resolve.json["items"]) == 2, "new entries are resolved in one batch"

    def test_a_hand_written_slug_is_recorded_under_its_canonical_name(self, env):
        env.skill("acme/refund-policy", url_slug="refunds-by-acme")
        env.write_manifest({"refunds-by-acme": "latest"})
        assert env.install().ok
        assert list(env.manifest()) == ["acme/refund-policy"]
        # A second run sees the entry as installed: nothing more to do.
        calls = len(env.calls())
        assert env.install().ok
        assert len(env.calls()) == calls

    def test_an_entry_removed_from_the_manifest_is_removed(self, env):
        _installed(env, "acme/refund-policy", "acme/returns")
        env.write_manifest({"acme/refund-policy": "latest"})
        result = env.install()
        assert result.ok, result.problems
        assert actions(result) == [("remove", "acme-returns")]
        assert list(env.lock().skills) == ["acme-refund-policy"]
        assert not env.folder("acme-returns").exists()
        assert env.project.plugin_json_path.read_bytes() == \
            L.plugin_json_bytes(env.lock().lock_hash)

    def test_a_modified_folder_it_must_remove_is_refused(self, env):
        _installed(env, "acme/refund-policy", "acme/returns")
        (env.folder("acme-returns") / "SKILL.md").write_text("edited\n")
        env.write_manifest({"acme/refund-policy": "latest"})
        before = env.tree()
        result = env.install()
        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        err = error_of(result)
        assert isinstance(err, E.ModifiedFolderError)
        assert "SKILL.md" in err.message and "--force" in err.fix
        assert env.tree() == before

    def test_force_removes_a_modified_folder_after_backing_it_up(self, env):
        _installed(env, "acme/refund-policy", "acme/returns")
        (env.folder("acme-returns") / "SKILL.md").write_text("edited\n")
        env.write_manifest({"acme/refund-policy": "latest"})
        result = env.install(force=True)
        assert result.ok, result.problems
        change = result.changes[0]
        assert (change.action, change.install_name) == ("remove", "acme-returns")
        assert (env.dir / change.backup / "SKILL.md").read_text() == "edited\n"
        assert not env.folder("acme-returns").exists()

    def test_a_removed_entrys_hand_placed_file_is_backed_up(self, env):
        env.skill("acme/refund-policy", files=({"path": "assets/logo.png", "content": None},))
        assert env.add("acme/refund-policy").ok
        logo = env.folder("acme-refund-policy") / "assets" / "logo.png"
        logo.parent.mkdir()
        logo.write_bytes(b"\x89PNG mine")
        env.write_manifest({})
        result = env.install()
        assert result.ok, result.problems
        assert notices(result, "hand_placed")
        assert [b.read_bytes() for b in env.project.backup_dir.rglob("logo.png")] == \
            [b"\x89PNG mine"]

    def test_a_missing_folder_is_fetched_at_its_locked_version_and_render(self, env):
        skill = env.skill("acme/refund-policy")
        assert env.add("acme/refund-policy").ok
        locked = env.lock().skills["acme-refund-policy"]
        before = env.tree()
        env.registry.add_version(skill)                 # v2 upstream
        skill.description = "A new description."       # and a new render of v1
        shutil.rmtree(env.folder("acme-refund-policy"))
        calls = len(env.calls())

        result = env.install()
        assert result.ok, result.problems
        assert actions(result) == [("fetch", "acme-refund-policy")]
        new = env.calls()[calls:]
        assert [c.route for c in new] == ["bundle"], "a locked folder is never resolved"
        query = new[0].query
        assert query["render_hash"] == locked["render"]["hash"]
        assert query["purpose"] == "install"
        assert new[0].path.endswith(locked["version_id"])
        assert env.tree() == before
        assert env.lock().skills["acme-refund-policy"] == locked

    def test_a_modified_folder_it_need_not_touch_is_reported_and_kept(self, env):
        _installed(env, "acme/refund-policy")
        (env.folder("acme-refund-policy") / "notes.md").write_text("mine\n")
        before = env.tree()
        calls = len(env.calls())
        result = env.install()
        assert result.ok
        (note,) = notices(result, "modified")
        assert "notes.md" in note.message and "--force" in (note.fix or "")
        assert env.tree() == before
        assert len(env.calls()) == calls

    def test_force_restores_a_modified_folder_to_the_locked_bytes(self, env):
        _installed(env, "acme/refund-policy")
        clean = env.tree()
        (env.folder("acme-refund-policy") / "SKILL.md").write_text("edited\n")
        result = env.install(force=True)
        assert result.ok, result.problems
        assert actions(result) == [("restore", "acme-refund-policy")]
        backup = result.changes[0].backup
        assert (env.dir / backup / "SKILL.md").read_text() == "edited\n"
        after = {k: v for k, v in env.tree().items() if not k.startswith(".decimal/backup/")}
        assert after == clean

    def test_an_empty_manifest_writes_an_empty_lock_for_frozen(self, env):
        env.write_manifest({})
        result = env.install()
        assert result.ok, result.problems
        assert dict(env.lock().skills) == {}
        assert env.project.plugin_json_path.exists()
        assert env.install(frozen=True).ok

    def test_unmanaged_folders_are_left_alone(self, env):
        _installed(env, "acme/refund-policy")
        mine = env.folder("my-own-skill")
        mine.mkdir()
        (mine / "SKILL.md").write_text("---\nname: my-own-skill\ndescription: x\n---\nx\n")
        before = env.tree()
        assert env.install().ok
        assert env.tree() == before


class TestNeverMovesAPin:
    def test_latest_stays_at_the_locked_version_with_no_network_call(self, env):
        skill = env.skill("acme/refund-policy")
        assert env.add("acme/refund-policy").ok
        env.registry.add_version(skill)
        skill.description = "Changed upstream."
        before = env.tree()
        calls = len(env.calls())
        result = env.install()
        assert result.ok
        assert actions(result) == []
        assert len(env.calls()) == calls, "everything is CLEAN: no network call"
        assert env.tree() == before

    def test_everything_clean_needs_no_client_at_all(self, env, monkeypatch):
        _installed(env, "acme/refund-policy", "acme/returns")

        def no_client(*args, **kwargs):
            raise AssertionError("a client was created")

        monkeypatch.setattr("decimalai.skillpkg.client.RegistryClient.from_env", no_client)
        result = ops.install(directory=env.dir)
        assert result.ok, result.problems

    def test_a_spec_moved_by_hand_is_resolved_at_the_new_spec(self, env):
        env.skill("acme/refund-policy", versions=3)
        assert env.add("acme/refund-policy").ok
        env.write_manifest({"acme/refund-policy": "1"})
        result = env.install()
        assert result.ok, result.problems
        assert actions(result) == [("update", "acme-refund-policy")]
        entry = env.lock().skills["acme-refund-policy"]
        assert (entry["version"], entry["pin"]) == (1, "exact")
        assert "Step 1:" in skill_md(env, "acme-refund-policy")

    def test_latest_to_its_own_version_changes_only_the_pin(self, env):
        env.skill("acme/refund-policy", versions=2)
        assert env.add("acme/refund-policy").ok
        calls = len(env.calls())
        env.write_manifest({"acme/refund-policy": "2"})
        result = env.install()
        assert result.ok
        assert actions(result) == [("pin", "acme-refund-policy")]
        assert len(env.calls()) == calls
        assert env.lock().skills["acme-refund-policy"]["pin"] == "exact"

    def test_exact_to_latest_keeps_the_version(self, env):
        skill = env.skill("acme/refund-policy", versions=2)
        assert env.add("acme/refund-policy@1").ok
        env.registry.add_version(skill)
        calls = len(env.calls())
        env.write_manifest({"acme/refund-policy": "latest"})
        result = env.install()
        assert result.ok
        assert actions(result) == [("pin", "acme-refund-policy")]
        assert len(env.calls()) == calls
        entry = env.lock().skills["acme-refund-policy"]
        assert (entry["version"], entry["pin"]) == (1, "latest")


class TestLockRepair:
    def test_entries_an_older_decimalai_merged_in_are_dropped(self, env):
        _installed(env, "acme/refund-policy")
        data = json.loads(env.lock_bytes())
        data["skills"]["old-skill"] = {"name": "old-skill", "content_hash": "abc"}
        env.project.lock_path.write_text(json.dumps(data))
        old = env.folder("old-skill")
        old.mkdir()
        (old / "SKILL.md").write_text("---\nname: old-skill\ndescription: x\n---\nx\n")

        result = env.install()
        assert result.ok, result.problems
        (note,) = notices(result, "dropped_entry")
        assert "old-skill" in note.message
        lock = env.lock()
        assert list(lock.skills) == ["acme-refund-policy"] and lock.hash_ok
        assert env.states()["old-skill"] is fs.State.UNMANAGED
        assert (old / "SKILL.md").exists()

    def test_a_stale_lock_hash_is_recomputed_with_a_warning(self, env):
        _installed(env, "acme/refund-policy")
        data = json.loads(env.lock_bytes())
        data["lock_hash"] = "0" * 64
        env.project.lock_path.write_text(json.dumps(data))
        result = env.install()
        assert result.ok
        assert notices(result, "lock_hash")
        assert env.lock().hash_ok


class TestInstallRefusals:
    def test_a_version_1_lock_is_exit_3_naming_migrate(self, env):
        env.write_manifest({"acme/refund-policy": "latest"})
        env.project.lock_path.write_text(json.dumps({"version": 1, "skills": {}}))
        before = env.tree()
        result = env.install()
        assert result.exit_code == E.ExitCode.LOCK
        err = error_of(result)
        assert isinstance(err, E.LegacyLockError) and "migrate" in err.fix
        assert env.tree() == before

    def test_an_npx_lock_is_exit_3(self, env):
        env.write_manifest({"acme/refund-policy": "latest"})
        env.project.npx_lock_path.write_text("{}")
        result = env.install()
        assert isinstance(error_of(result), E.LegacyLockError)

    def test_a_corrupt_lock_is_exit_3_and_never_reset(self, env):
        env.write_manifest({"acme/refund-policy": "latest"})
        env.project.lock_path.write_text("{not json")
        result = env.install()
        assert result.exit_code == E.ExitCode.LOCK
        assert env.project.lock_path.read_text() == "{not json"

    def test_no_manifest_is_exit_3(self, env):
        result = env.install()
        assert result.exit_code == E.ExitCode.LOCK
        assert isinstance(error_of(result), E.SkillsNotInstalledError)
        assert env.tree() == {}
        assert not env.project.state_dir.exists()

    def test_an_unreadable_manifest_is_exit_3(self, env):
        env.project.state_dir.mkdir()
        env.project.manifest_path.write_text('[skills]\n"x" = "^3"\n')
        result = env.install()
        assert result.exit_code == E.ExitCode.LOCK
        assert isinstance(error_of(result), E.ManifestError)

    def test_every_unresolvable_entry_is_reported_and_nothing_is_installed(self, env):
        env.skill("acme/refund-policy")
        env.write_manifest({"acme/refund-policy": "latest", "acme/gone": "latest",
                            "acme/also-gone": "latest"})
        before = env.tree()
        result = env.install()
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert sorted(e.skill for e in result.problems if isinstance(e, E.SkillpkgError)) == \
            ["acme/also-gone", "acme/gone"]
        assert env.tree() == before
        assert env.calls("bundle") == []

    def test_one_skill_listed_twice_is_exit_7(self, env):
        env.skill("acme/refund-policy", url_slug="refunds-by-acme")
        assert env.add("acme/refund-policy").ok
        env.write_manifest({"acme/refund-policy": "latest", "refunds-by-acme": "1"})
        before = env.tree()
        result = env.install()
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        err = error_of(result)
        assert isinstance(err, E.AlreadyInstalledError) and "twice" in err.message
        assert env.tree() == before

    def test_a_manifest_key_the_registry_cannot_read_is_exit_2(self, env):
        env.skill("acme/refund-policy")
        env.write_manifest({"acme/refund-policy": "latest", "acme/bad\x01name": "latest"})
        before = env.tree()
        result = env.install()
        assert result.exit_code == E.ExitCode.USAGE
        err = error_of(result)
        assert isinstance(err, E.UsageError) and "control character" in err.fix
        assert str(err).isprintable()
        assert env.tree() == before

    def test_a_caution_skill_on_a_non_tty_needs_yes(self, env):
        env.skill("acme/refund-policy", band="caution")
        env.write_manifest({"acme/refund-policy": "latest"})
        result = env.install(yes=False)
        assert isinstance(error_of(result), E.ConfirmationRequiredError)
        assert env.install(yes=True).ok

    def test_another_run_holding_the_project_is_exit_1(self, env, monkeypatch):
        _installed(env, "acme/refund-policy")
        monkeypatch.setattr(ops, "LOCK_TIMEOUT", 0.2)
        with fs.ProcessLock(env.project):
            result = env.install()
        assert result.exit_code == E.ExitCode.PROBLEMS
        assert isinstance(error_of(result), E.SkillsBusyError)

    @pytest.mark.parametrize("target", ["SKILL.md", "references"])
    def test_a_symlink_inside_a_locked_folder_is_exit_5_even_with_force(
            self, env, tmp_path, target):
        import os
        if os.name != "posix":
            pytest.skip("symlinks")
        _installed(env, "acme/refund-policy")
        env.write_manifest({})
        folder = env.folder("acme-refund-policy")
        path = folder / target
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        os.symlink(tmp_path, path)
        result = env.install(force=True)
        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        assert isinstance(error_of(result), E.IllegalEntryError)
        assert os.path.islink(path)
