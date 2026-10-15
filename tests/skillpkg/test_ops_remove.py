"""``skills remove``: the folder, the lock entry and the manifest entry, and no network.

Pinned here:

- the name may be the install name or the registry name;
- a CLEAN folder is removed; a MODIFIED one is refused (exit 5) unless ``force=True``,
  which backs it up first; a MISSING one just leaves the lock;
- a name the project does not manage is exit 7, and nothing changes;
- ``plugin.json`` follows the new lock, other entries and the manifest's comments stay;
- no registry call is ever made: server state (Use links, forks) is not touched.
"""
from __future__ import annotations

import os
import shutil

import pytest

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import fs
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import ops
from tests.skillpkg.test_ops_add import (
    POSIX,
    REFUND_FILES,
    Env,
    actions,
    error_of,
    notices,
)


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)


@pytest.fixture
def two(env) -> Env:
    for name in ("acme/refund-policy", "acme/returns"):
        env.skill(name)
        assert env.add(name).ok
    return env


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("remove created a registry client")

    monkeypatch.setattr("decimalai.skillpkg.client.RegistryClient.__init__", refuse)


class TestRemove:
    @pytest.mark.parametrize("name", ["acme-returns", "acme/returns"])
    def test_removes_folder_lock_entry_and_manifest_entry(self, two, name, no_network):
        env = two
        calls = len(env.calls())
        result = env.remove(name)
        assert result.ok, result.problems
        assert actions(result) == [("remove", "acme-returns")]
        assert result.changes[0].skill == "acme/returns"
        assert not env.folder("acme-returns").exists()
        lock = env.lock()
        assert list(lock.skills) == ["acme-refund-policy"]
        assert list(env.manifest()) == ["acme/refund-policy"]
        assert env.project.plugin_json_path.read_bytes() == L.plugin_json_bytes(lock.lock_hash)
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN}
        assert len(env.calls()) == calls
        assert not env.project.staging_dir.exists()

    def test_the_manifest_keeps_its_comments(self, two):
        env = two
        text = env.manifest_text().replace(
            '"acme/refund-policy" = "latest"',
            '# the refund rules\n"acme/refund-policy" = "latest"  # keep')
        env.project.manifest_path.write_text(text)
        assert env.remove("acme-returns").ok
        after = env.manifest_text()
        assert '# the refund rules\n"acme/refund-policy" = "latest"  # keep' in after
        assert "acme/returns" not in after

    def test_a_modified_folder_is_exit_5(self, two):
        env = two
        (env.folder("acme-returns") / "SKILL.md").write_text("edited\n")
        before = env.tree()
        result = env.remove("acme-returns")
        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        err = error_of(result)
        assert isinstance(err, E.ModifiedFolderError) and "--force" in err.fix
        assert env.tree() == before

    def test_force_removes_a_modified_folder_and_keeps_a_backup(self, two):
        env = two
        (env.folder("acme-returns") / "SKILL.md").write_text("edited\n")
        result = env.remove("acme-returns", force=True)
        assert result.ok, result.problems
        backup = result.changes[0].backup
        assert (env.dir / backup / "SKILL.md").read_text() == "edited\n"
        assert any(backup in n.message for n in notices(result, "backup"))
        assert not env.folder("acme-returns").exists()
        assert "acme-returns" not in env.lock().skills

    def test_a_hand_placed_file_at_an_unavailable_path_is_backed_up(self, env):
        env.skill("acme/refund-policy", files=REFUND_FILES + (
            {"path": "assets/logo.png", "content": None},))
        assert env.add("acme/refund-policy").ok
        logo = env.folder("acme-refund-policy") / "assets" / "logo.png"
        logo.parent.mkdir()
        logo.write_bytes(b"\x89PNG mine")
        result = env.remove("acme-refund-policy")
        assert result.ok, result.problems
        (note,) = notices(result, "hand_placed")
        backups = list((env.project.backup_dir).rglob("logo.png"))
        assert [b.read_bytes() for b in backups] == [b"\x89PNG mine"]
        assert env.project.rel(backups[0]) in note.message
        assert not env.folder("acme-refund-policy").exists()

    def test_a_missing_folder_leaves_the_lock(self, two):
        env = two
        shutil.rmtree(env.folder("acme-returns"))
        result = env.remove("acme-returns")
        assert result.ok
        assert "acme-returns" not in env.lock().skills
        assert list(env.manifest()) == ["acme/refund-policy"]

    def test_a_manifest_entry_not_yet_installed_is_removed(self, two):
        env = two
        env.write_manifest({"acme/refund-policy": "latest", "acme/returns": "latest",
                            "acme/later": "3"})
        result = env.remove("acme/later")
        assert result.ok
        assert sorted(env.manifest()) == ["acme/refund-policy", "acme/returns"]
        assert sorted(env.lock().skills) == ["acme-refund-policy", "acme-returns"]

    def test_the_last_skill_leaves_an_empty_lock(self, env):
        env.skill("acme/refund-policy")
        assert env.add("acme/refund-policy").ok
        assert env.remove("acme-refund-policy").ok
        lock = env.lock()
        assert dict(lock.skills) == {}
        assert env.project.plugin_json_path.read_bytes() == L.plugin_json_bytes(lock.lock_hash)
        assert env.project.skills_root.is_dir()

    def test_an_unmanaged_folder_is_not_a_managed_name(self, two):
        env = two
        mine = env.folder("my-own-skill")
        mine.mkdir()
        (mine / "SKILL.md").write_text("---\nname: my-own-skill\ndescription: x\n---\nx\n")
        before = env.tree()
        result = env.remove("my-own-skill")
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert isinstance(error_of(result), E.SkillNotManagedError)
        assert env.tree() == before

    def test_an_unknown_name_is_exit_7(self, two):
        before = two.tree()
        result = two.remove("acme/nope")
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        err = error_of(result)
        assert isinstance(err, E.SkillNotManagedError) and err.skill == "acme/nope"
        assert two.tree() == before

    def test_an_empty_project_is_exit_7_and_creates_nothing(self, env):
        result = env.remove("acme-refund-policy")
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert env.tree() == {}
        assert not env.project.state_dir.exists()

    def test_a_symlinked_folder_is_never_followed(self, two, tmp_path):
        if not POSIX:
            pytest.skip("symlinks")
        env = two
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / "keep.md").write_text("keep\n")
        shutil.rmtree(env.folder("acme-returns"))
        os.symlink(target, env.folder("acme-returns"))
        result = env.remove("acme-returns", force=True)
        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        assert isinstance(error_of(result), E.IllegalEntryError)
        assert (target / "keep.md").read_text() == "keep\n"

    @pytest.mark.skipif(not POSIX or os.geteuid() == 0, reason="permissions, non-root")
    def test_an_unreadable_file_is_exit_5_not_a_traceback(self, two):
        env = two
        fees = env.folder("acme-returns") / "references" / "fees.md"
        fees.chmod(0o000)
        try:
            result = env.remove("acme-returns", force=True)
            assert result.exit_code == E.ExitCode.LOCAL_CHANGES
            err = error_of(result)
            assert isinstance(err, E.IllegalEntryError)
            assert "unreadable" in err.message and "chmod" in err.fix
            assert fees.exists()
        finally:
            fees.chmod(0o644)

    def test_a_lock_an_older_decimalai_merged_into_is_exit_3(self, two):
        import json

        env = two
        data = json.loads(env.lock_bytes())
        data["skills"]["old"] = {"name": "old"}
        env.project.lock_path.write_text(json.dumps(data))
        result = env.remove("acme-returns")
        assert result.exit_code == E.ExitCode.LOCK
        assert "decimalai skills install" in error_of(result).fix

    def test_remove_through_the_package_api(self, two):
        from decimalai import skillpkg

        assert skillpkg.remove is ops.remove
        assert skillpkg.remove("acme-returns", directory=two.dir).ok
