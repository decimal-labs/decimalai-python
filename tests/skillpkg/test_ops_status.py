"""``skills status``: what is installed, and whether it agrees with the manifest and lock.

Read-only. Pinned here:

- one row per skill: its folder state, version, pin and, for MODIFIED, the files;
- manifest and lock disagreement, an interrupted run, a legacy lock (with the migrate
  command), a lock that git ignores, entries an older decimalai merged in;
- ``remote=True`` asks the registry: outdated (v1 -> v2), a description changed upstream
  on the same version, a locked version since revoked; without it, no network call;
- exit 0 always, except ``check=True``, which exits 1 on any drift;
- nothing on disk changes.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import fs
from decimalai.skillpkg import ops
from tests.skillpkg.test_ops_add import (
    POSIX,
    Env,
    notices,
)


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)


def rows(result) -> dict:
    return {r.install_name: r for r in result.changes}


def codes(result) -> list:
    return sorted(n.code for n in notices(result))


@pytest.fixture
def two(env) -> Env:
    for name in ("acme/refund-policy", "acme/returns"):
        env.skill(name)
        assert env.add(name).ok
    return env


class TestLocal:
    def test_a_clean_project(self, two):
        env = two
        before = env.tree()
        calls = len(env.calls())
        result = env.status(check=True)
        assert result.exit_code == E.ExitCode.OK, result.problems
        got = rows(result)
        assert sorted(got) == ["acme-refund-policy", "acme-returns"]
        row = got["acme-returns"]
        assert (row.state, row.skill, row.version, row.pin, row.spec) == \
            ("clean", "acme/returns", 1, "latest", "latest")
        assert codes(result) == []
        assert len(env.calls()) == calls, "status makes no network call without remote"
        assert env.tree() == before

    def test_an_empty_project(self, env):
        result = env.status(check=True)
        assert result.ok
        assert result.changes == ()
        assert env.tree() == {}

    def test_folder_states_are_reported_and_check_fails(self, two):
        env = two
        (env.folder("acme-returns") / "SKILL.md").write_text("edited\n")
        shutil.rmtree(env.folder("acme-refund-policy"))
        mine = env.folder("my-own-skill")
        mine.mkdir()
        (mine / "SKILL.md").write_text("---\nname: my-own-skill\ndescription: x\n---\nx\n")
        before = env.tree()

        result = env.status()
        assert result.exit_code == E.ExitCode.OK
        got = rows(result)
        assert got["acme-returns"].state == "modified"
        assert got["acme-returns"].files == ("SKILL.md",)
        assert got["acme-refund-policy"].state == "missing"
        assert got["my-own-skill"].state == "unmanaged"
        assert {"modified", "missing"} <= set(codes(result))
        assert "unmanaged" not in [n.code for n in notices(result) if n.drift]

        checked = env.status(check=True)
        assert checked.exit_code == E.ExitCode.PROBLEMS
        assert env.tree() == before

    @pytest.mark.skipif(not POSIX or os.geteuid() == 0, reason="permissions, non-root")
    def test_an_unreadable_file_is_reported_not_raised(self, two):
        env = two
        fees_dir = env.folder("acme-returns")
        target = next(p for p in fees_dir.rglob("*") if p.is_file() and p.name != "SKILL.md")
        target.chmod(0o000)
        try:
            result = env.status()
            assert result.exit_code == E.ExitCode.OK
            row = rows(result)["acme-returns"]
            assert row.state == "illegal" and row.reason.startswith("unreadable: ")
            assert env.status(check=True).exit_code == E.ExitCode.PROBLEMS
        finally:
            target.chmod(0o644)

    def test_unmanaged_folders_alone_are_not_drift(self, two):
        mine = two.folder("my-own-skill")
        mine.mkdir()
        (mine / "SKILL.md").write_text("---\nname: my-own-skill\ndescription: x\n---\nx\n")
        assert two.status(check=True).ok

    def test_manifest_and_lock_disagreement(self, two):
        env = two
        env.write_manifest({"acme/refund-policy": "1", "acme/new": "latest"})
        result = env.status(check=True)
        assert result.exit_code == E.ExitCode.PROBLEMS
        assert {"not_installed", "not_in_manifest", "spec_mismatch"} <= set(codes(result))
        got = rows(result)
        assert got[None].skill == "acme/new" and got[None].state == "not_installed"
        assert got["acme-refund-policy"].spec == "1"

    def test_an_interrupted_run(self, two):
        env = two
        entry = dict(env.lock().skills["acme-returns"])
        journal = fs.Journal("20261020T000000Z-0000abcd", "acme-returns", entry, entry, True)
        env.project.journal_path.write_text(json.dumps(journal.to_dict()))
        before = env.tree()
        result = env.status(check=True)
        assert result.exit_code == E.ExitCode.PROBLEMS
        (note,) = notices(result, "interrupted")
        assert "decimalai skills install" in note.fix
        assert rows(result)["acme-returns"].state == "interrupted"
        assert env.tree() == before

    @pytest.mark.parametrize("legacy", ["v1", "npx"])
    def test_a_legacy_lock_names_migrate(self, env, legacy):
        env.write_manifest({"acme/refund-policy": "latest"})
        if legacy == "v1":
            env.project.lock_path.write_text(json.dumps({"version": 1, "skills": {}}))
        else:
            env.project.npx_lock_path.write_text("{}")
        result = env.status(check=True)
        assert result.exit_code == E.ExitCode.PROBLEMS
        (note,) = notices(result, "legacy_lock")
        assert "decimalai skills migrate" in note.fix

    def test_entries_an_older_decimalai_merged_in(self, two):
        env = two
        data = json.loads(env.lock_bytes())
        data["skills"]["old"] = {"name": "old"}
        env.project.lock_path.write_text(json.dumps(data))
        result = env.status(check=True)
        assert result.exit_code == E.ExitCode.PROBLEMS
        assert "foreign_entries" in codes(result)
        assert sorted(rows(result)) == ["acme-refund-policy", "acme-returns"]

    def test_an_unreadable_lock(self, two):
        two.project.lock_path.write_text("{oops")
        result = two.status(check=True)
        assert result.exit_code == E.ExitCode.PROBLEMS
        assert "lock_unreadable" in codes(result)

    @pytest.mark.skipif(shutil.which("git") is None, reason="git")
    def test_a_gitignore_that_ignores_the_lock(self, two):
        env = two
        subprocess.run(["git", "init", "-q", str(env.dir)], check=True)
        (env.dir / ".gitignore").write_text(".decimal/\n")
        result = env.status(check=True)
        assert result.exit_code == E.ExitCode.PROBLEMS
        ignored = [n.path for n in notices(result, "lock_ignored")]
        assert ".decimal/skills.lock" in ignored

    @pytest.mark.skipif(shutil.which("git") is None, reason="git")
    def test_the_engines_own_gitignore_keeps_the_lock_committable(self, two):
        subprocess.run(["git", "init", "-q", str(two.dir)], check=True)
        assert two.status(check=True).ok


class TestRemote:
    def test_outdated_and_description_changed_and_revoked(self, env):
        a = env.skill("acme/a")
        b = env.skill("acme/b")
        c = env.skill("acme/c")
        for name in ("acme/a", "acme/b", "acme/c"):
            assert env.add(name).ok
        env.registry.add_version(a)
        b.description = "Edited upstream."
        c.band = "blocked"
        before = env.tree()

        result = env.status(remote=True)
        assert result.exit_code == E.ExitCode.OK
        got = rows(result)
        assert any("outdated (v1 -> v2)" in f for f in got["acme-a"].remote)
        assert any("description changed upstream" in f for f in got["acme-b"].remote)
        assert any("revoked" in f and "blocked" in f for f in got["acme-c"].remote)
        assert {"outdated", "description_changed", "revoked"} <= set(codes(result))
        assert env.status(remote=True, check=True).exit_code == E.ExitCode.PROBLEMS
        assert env.tree() == before
        assert env.calls("bundle")[3:] == [], "status fetches no bundle"

    def test_an_exact_pin_is_not_outdated(self, env):
        skill = env.skill("acme/a", versions=2)
        assert env.add("acme/a@1").ok
        env.registry.add_version(skill)
        result = env.status(remote=True, check=True)
        assert result.ok, result.problems
        assert rows(result)["acme-a"].remote == ()

    def test_current_is_clean(self, two):
        result = two.status(remote=True, check=True)
        assert result.ok, result.problems

    def test_a_network_failure_is_exit_8(self, two):
        two.registry.fault("resolve", "connect")
        result = two.status(remote=True)
        assert result.exit_code == E.ExitCode.NETWORK


def test_results_are_json_ready(two):
    env = two
    (env.folder("acme-returns") / "SKILL.md").write_text("edited\n")
    for result in (env.status(remote=True), env.update(dry_run=True),
                   env.remove("acme/nope"), env.install()):
        data = json.loads(json.dumps(result.to_dict()))
        assert data["exit_code"] == int(result.exit_code)
        assert len(data["changes"]) == len(result.changes)
        assert len(data["problems"]) == len(result.problems)


def test_status_through_the_package_api(two):
    from decimalai import skillpkg

    assert skillpkg.status is ops.status
    assert skillpkg.status(directory=two.dir).ok
