"""``skills update``: move ``latest`` entries, change a spec, refresh a render.

Pinned here:

- with no argument every ``latest`` entry moves to the current version and render, and
  exact pins do not move;
- ``name@N`` and ``name@latest`` change the spec, then update; ``name`` alone on an exact
  pin refreshes only its render (the same version);
- a render-only change (a description edited upstream) rewrites SKILL.md; files the new
  version dropped disappear and new exec bits apply; folder names never change;
- a MODIFIED folder is refused (exit 5); ``force=True`` backs it up first and says where;
  ``dry_run=True`` changes nothing;
- a failure while staging skill 3 of 5 leaves every folder and the lock untouched.
"""
from __future__ import annotations

import shutil

import pytest
from skillevaluation import skillfmt

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import fs
from decimalai.skillpkg import ops
from tests.skillpkg.test_ops_add import (
    POSIX,
    Env,
    actions,
    error_of,
    frontmatter,
    notices,
    skill_md,
)


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)

FIVE = ("acme/one", "acme/two", "acme/three", "acme/four", "acme/five")


def _installed(env: Env, *refs: str, versions: int = 1) -> list:
    skills = []
    for ref in refs:
        name, _, spec = ref.partition("@")
        skills.append(env.skill(name, versions=versions))
        assert env.add(ref).ok
    return skills


class TestUpdateAll:
    def test_latest_moves_and_exact_stays(self, env):
        moving, pinned = _installed(env, "acme/moving", "acme/pinned@1")
        env.registry.add_version(moving)
        env.registry.add_version(pinned)
        pinned.description = "Edited upstream."
        result = env.update()
        assert result.ok, result.problems
        assert actions(result) == [("update", "acme-moving")]
        assert (result.changes[0].old_version, result.changes[0].version) == (1, 2)
        assert str(result.changes[0]) == "updated acme/moving v1 -> v2"
        lock = env.lock()
        assert lock.skills["acme-moving"]["version"] == 2
        assert lock.skills["acme-pinned"]["version"] == 1
        assert lock.skills["acme-pinned"]["description_sha256"] == \
            skillfmt.description_sha256("Use when the task involves acme/pinned.")
        resolve = env.calls("resolve")[-1].json
        assert resolve["purpose"] == "update"
        assert [i.get("skill_id") for i in resolve["items"]] == [moving.skill_id], \
            "only latest entries are resolved, by skill id"

    def test_nothing_new_upstream_changes_nothing(self, env):
        _installed(env, "acme/one", "acme/two")
        before = env.tree()
        result = env.update()
        assert result.ok
        assert actions(result) == []
        assert env.tree() == before

    def test_a_render_only_change_rewrites_skill_md(self, env):
        (skill,) = _installed(env, "acme/refund-policy")
        old = env.lock().skills["acme-refund-policy"]
        skill.description = "Apply the new refund rules."
        result = env.update()
        assert result.ok, result.problems
        change = result.changes[0]
        assert (change.action, change.old_version, change.version) == ("update", 1, 1)
        assert change.description_changed
        new = env.lock().skills["acme-refund-policy"]
        assert new["version_id"] == old["version_id"]
        assert new["render"]["hash"] != old["render"]["hash"]
        assert new["installed_at"] >= old["installed_at"]
        description = frontmatter(env, "acme-refund-policy")["description"]
        assert description == "Apply the new refund rules."
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN}

    def test_stale_files_disappear_and_new_exec_bits_apply(self, env):
        (skill,) = _installed(env, "acme/refund-policy")
        env.registry.add_version(skill, files=[
            {"path": "references/fees.md", "content": "# Fees\n\nNow 3%.\n", "executable": True},
            {"path": "templates/reply.md", "content": "Dear customer,\n"},
        ])
        assert env.update().ok
        folder = env.folder("acme-refund-policy")
        files = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file())
        assert files == ["SKILL.md", "references/fees.md", "templates/reply.md"]
        if POSIX:
            assert (folder / "references" / "fees.md").stat().st_mode & 0o777 == 0o755
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN}

    def test_folder_names_never_change(self, env):
        (skill,) = _installed(env, "acme/refund-policy")
        skill.name = "acme/refunds-renamed"
        env.registry.add_version(skill)
        result = env.update()
        assert result.ok, result.problems
        lock = env.lock()
        assert list(lock.skills) == ["acme-refund-policy"]
        assert lock.skills["acme-refund-policy"]["name"] == "acme/refunds-renamed"
        assert frontmatter(env, "acme-refund-policy")["name"] == "acme-refund-policy"
        # The manifest follows the registry's canonical name, so install still agrees.
        assert list(env.manifest()) == ["acme/refunds-renamed"]
        calls = len(env.calls())
        assert env.install(frozen=True).ok
        assert len(env.calls()) == calls

    def test_a_rename_alone_keeps_the_locked_name_and_the_bytes(self, env):
        # The registry keeps the name a rendering was made with, so the same render hash
        # still serves the old name: the lock and the folder's stamp keep it too.
        (skill,) = _installed(env, "acme/refund-policy")
        before = env.tree()
        skill.name = "acme/refunds-renamed"
        result = env.update()
        assert result.ok, result.problems
        assert actions(result) == []
        assert env.tree() == before
        assert env.lock().skills["acme-refund-policy"]["name"] == "acme/refund-policy"
        shutil.rmtree(env.folder("acme-refund-policy"))
        assert env.install(frozen=True).ok
        assert env.tree() == before

    def test_new_and_removed_manifest_entries_are_reconciled_too(self, env):
        _installed(env, "acme/one", "acme/two")
        env.skill("acme/three")
        env.write_manifest({"acme/one": "latest", "acme/three": "latest"})
        result = env.update()
        assert result.ok, result.problems
        assert sorted(actions(result)) == [("add", "acme-three"), ("remove", "acme-two")]


class TestUpdateOne:
    def test_name_at_n_changes_the_spec_then_updates(self, env):
        _installed(env, "acme/refund-policy", versions=3)
        result = env.update("acme/refund-policy@1")
        assert result.ok, result.problems
        assert env.manifest()["acme/refund-policy"].spec == "1"
        entry = env.lock().skills["acme-refund-policy"]
        assert (entry["version"], entry["pin"]) == (1, "exact")
        assert "Step 1:" in skill_md(env, "acme-refund-policy")

    def test_name_at_latest_moves_an_exact_pin(self, env):
        _installed(env, "acme/refund-policy@1", versions=3)
        result = env.update("acme-refund-policy@latest")
        assert result.ok, result.problems
        assert env.manifest()["acme/refund-policy"].spec == "latest"
        entry = env.lock().skills["acme-refund-policy"]
        assert (entry["version"], entry["pin"]) == (3, "latest")

    def test_name_alone_on_an_exact_pin_refreshes_only_its_render(self, env):
        (skill,) = _installed(env, "acme/refund-policy@1", versions=2)
        env.registry.add_version(skill)
        skill.description = "Edited after version 1."
        old = env.lock().skills["acme-refund-policy"]
        # An older version renders with its own description: give it a new render by
        # changing a trigger field that applies to every version.
        skill.license = "MIT"
        result = env.update("acme-refund-policy")
        assert result.ok, result.problems
        new = env.lock().skills["acme-refund-policy"]
        assert new["version_id"] == old["version_id"] and new["version"] == 1
        assert new["render"]["hash"] != old["render"]["hash"]
        assert frontmatter(env, "acme-refund-policy")["license"] == "MIT"
        assert env.manifest()["acme/refund-policy"].spec == "1"

    def test_only_the_named_entry_is_touched(self, env):
        one, two = _installed(env, "acme/one", "acme/two")
        env.registry.add_version(one)
        env.registry.add_version(two)
        result = env.update("acme/two")
        assert result.ok
        assert actions(result) == [("update", "acme-two")]
        assert env.lock().skills["acme-one"]["version"] == 1

    def test_an_unknown_name_is_exit_7(self, env):
        _installed(env, "acme/one")
        before = env.tree()
        result = env.update("acme/nope")
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        assert isinstance(error_of(result), E.SkillNotManagedError)
        assert env.tree() == before

    def test_a_malformed_spec_is_a_usage_error(self, env):
        _installed(env, "acme/one")
        assert env.update("acme/one@^2").exit_code == E.ExitCode.USAGE


class TestLocalEdits:
    def test_a_modified_folder_is_exit_5(self, env):
        (skill,) = _installed(env, "acme/refund-policy")
        (env.folder("acme-refund-policy") / "references" / "fees.md").write_text("mine\n")
        env.registry.add_version(skill)
        before = env.tree()
        result = env.update()
        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        err = error_of(result)
        assert isinstance(err, E.ModifiedFolderError)
        assert "references/fees.md" in err.message and "--force" in err.fix
        assert env.tree() == before

    def test_force_prints_the_backup_path(self, env):
        (skill,) = _installed(env, "acme/refund-policy")
        (env.folder("acme-refund-policy") / "references" / "fees.md").write_text("mine\n")
        env.registry.add_version(skill)
        result = env.update(force=True)
        assert result.ok, result.problems
        backup = result.changes[0].backup
        assert backup.startswith(".decimal/backup/")
        assert backup in str(result.changes[0])
        assert (env.dir / backup / "references" / "fees.md").read_text() == "mine\n"
        (note,) = notices(result, "backup")
        assert backup in note.message
        assert env.lock().skills["acme-refund-policy"]["version"] == 2
        assert env.states() == {"acme-refund-policy": fs.State.CLEAN}

    def test_a_modified_folder_with_nothing_to_update_is_kept(self, env):
        _installed(env, "acme/refund-policy")
        (env.folder("acme-refund-policy") / "notes.md").write_text("mine\n")
        before = env.tree()
        result = env.update()
        assert result.ok
        assert env.tree() == before


class TestDryRun:
    def test_dry_run_changes_nothing(self, env):
        one, two = _installed(env, "acme/one", "acme/two")
        env.registry.add_version(one)
        env.registry.add_version(two)
        before = env.tree()
        result = env.update(dry_run=True)
        assert result.ok, result.problems
        assert sorted(actions(result)) == [("update", "acme-one"), ("update", "acme-two")]
        assert all(c.dry_run for c in result.changes)
        assert [c.version for c in result.changes] == [2, 2]
        assert env.tree() == before
        assert env.calls("bundle")[2:] == [], "a dry run fetches no bundle"

    def test_dry_run_reads_without_taking_the_project(self, env, monkeypatch):
        (skill,) = _installed(env, "acme/one")
        env.registry.add_version(skill)
        monkeypatch.setattr(ops, "LOCK_TIMEOUT", 0.2)
        with fs.ProcessLock(env.project):
            result = env.update(dry_run=True)
        assert result.ok, result.problems
        assert actions(result) == [("update", "acme-one")]

    def test_dry_run_reports_what_would_be_refused(self, env):
        (skill,) = _installed(env, "acme/refund-policy")
        (env.folder("acme-refund-policy") / "SKILL.md").write_text("edited\n")
        env.registry.add_version(skill)
        before = env.tree()
        result = env.update(dry_run=True)
        assert result.exit_code == E.ExitCode.LOCAL_CHANGES
        assert env.tree() == before


class TestNothingAppliedIfOneFails:
    def test_a_failure_while_staging_skill_3_of_5_leaves_disk_and_lock_untouched(
            self, env, monkeypatch):
        skills = _installed(env, *FIVE)
        for skill in skills:
            env.registry.add_version(skill)
        before = env.tree()
        staged = []
        real_stage = fs.stage

        def stage(project, txid, name, contents, entry):
            staged.append(name)
            if len(staged) == 3:
                raise E.BundleContractError("SKILL.md breaks the format", skill=name)
            return real_stage(project, txid, name, contents, entry)

        monkeypatch.setattr(fs, "stage", stage)
        result = env.update()
        assert result.exit_code == E.ExitCode.INTEGRITY
        assert len(staged) == 3
        assert env.tree() == before, "no folder, no lock entry, no manifest line changed"
        assert not env.project.staging_dir.exists()
        assert not env.project.journal_path.exists()

    def test_tampered_bytes_for_skill_3_of_5_leave_everything_untouched(self, env):
        skills = _installed(env, *FIVE)
        for skill in skills:
            env.registry.add_version(skill)
        before = env.tree()
        third = skills[2]

        def tamper(data):
            if data["skill_id"] == third.skill_id:
                data["files"][0]["content"] += "injected\n"

        env.registry.tamper = tamper
        result = env.update()
        assert result.exit_code == E.ExitCode.INTEGRITY
        assert error_of(result).skill == "acme/three"
        assert env.tree() == before

    def test_every_refused_entry_is_reported(self, env):
        skills = _installed(env, *FIVE)
        for skill in skills:
            env.registry.add_version(skill)
        skills[1].band = "blocked"
        skills[3].band = "blocked"
        before = env.tree()
        result = env.update()
        assert result.exit_code == E.ExitCode.UNAVAILABLE
        refused = sorted(e.skill for e in result.problems if isinstance(e, E.SkillpkgError))
        assert refused == ["acme/four", "acme/two"]
        assert env.tree() == before


class TestTrustOnUpdate:
    def test_a_new_version_of_a_caution_skill_needs_yes_on_a_non_tty(self, env):
        (skill,) = _installed(env, "acme/refund-policy")
        skill.band = "caution"
        env.registry.add_version(skill)
        result = env.update(yes=False)
        assert isinstance(error_of(result), E.ConfirmationRequiredError)
        assert env.lock().skills["acme-refund-policy"]["version"] == 1
        assert env.update(yes=True).ok

    def test_a_render_refresh_of_the_accepted_version_does_not_ask_again(self, env):
        (skill,) = _installed(env, "acme/refund-policy")
        skill.band = "caution"
        skill.description = "Edited."
        assert env.update(yes=False).ok
