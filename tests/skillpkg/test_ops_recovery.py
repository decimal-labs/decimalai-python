"""Crashes, interrupted runs and concurrent runs, seen through the operations.

Pinned here:

- a crash part way through applying an update leaves every skill completed before it
  locked at its new version, and the rest at their old one; the next command settles
  the interrupted skill, and then the project is consistent;
- a crash while staging changes nothing under the root, and the next run discards the
  staging it left;
- a journal whose folder matches neither side is exit 1, and nothing is touched;
- another run holding the project is exit 1; a long run refreshes its hold between
  skills, and stops if another run took it over;
- warnings from a swap (a hand-placed file carried over or backed up) reach the result.
"""
from __future__ import annotations

import json

import pytest

from decimalai.skillpkg import errors as E
from decimalai.skillpkg import fs
from decimalai.skillpkg import lock as L
from decimalai.skillpkg import ops
from tests.skillpkg.test_ops_add import (
    Env,
    error_of,
    notices,
)


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)

FIVE = ("acme/one", "acme/two", "acme/three", "acme/four", "acme/five")


class Crash(BaseException):
    """A process dying: not an Exception, so nothing in the engine catches it."""


def _five(env: Env) -> list:
    skills = []
    for name in FIVE:
        skills.append(env.skill(name))
        assert env.add(name).ok
    for skill in skills:
        env.registry.add_version(skill)
    return skills


def _crash_at(monkeypatch, step: str, nth: int) -> None:
    seen = []

    def checkpoint(name):
        if name == step:
            seen.append(name)
            if len(seen) == nth:
                raise Crash(f"{step} #{nth}")

    monkeypatch.setattr(fs, "_checkpoint", checkpoint)


def _versions(env: Env) -> dict:
    lock = L.load(env.project, repair=True)
    return {name: entry["version"] for name, entry in lock.skills.items()}


class TestCrashDuringApply:
    @pytest.mark.parametrize("step", ["journal", "old_moved", "new_moved", "lock_written"])
    def test_completed_skills_stay_locked_and_the_next_run_settles(self, env, monkeypatch, step):
        _five(env)
        # Swaps run in install-name order: five, four, one, three, two.
        _crash_at(monkeypatch, step, 3)
        with pytest.raises(Crash):
            env.update()
        monkeypatch.setattr(fs, "_checkpoint", lambda name: None)

        versions = _versions(env)
        assert versions["acme-five"] == 2 and versions["acme-four"] == 2
        assert versions["acme-three"] == 1 and versions["acme-two"] == 1
        # The skill being swapped is old or new, never half; its journal says which.
        assert env.project.journal_path.exists()

        result = env.install()
        assert result.ok, result.problems
        states = env.states()
        assert set(states.values()) == {fs.State.CLEAN}, states
        assert not env.project.journal_path.exists()
        assert not env.project.staging_dir.exists()
        settled = _versions(env)
        assert settled["acme-one"] in (1, 2)
        lock = env.lock()
        assert env.project.plugin_json_path.read_bytes() == L.plugin_json_bytes(lock.lock_hash)

        # And a second update completes the job.
        assert env.update().ok
        assert set(_versions(env).values()) == {2}

    def test_the_recovery_is_reported(self, env, monkeypatch):
        _five(env)
        _crash_at(monkeypatch, "old_moved", 1)
        with pytest.raises(Crash):
            env.update()
        monkeypatch.setattr(fs, "_checkpoint", lambda name: None)
        result = env.install()
        assert result.ok
        (note,) = notices(result, "recovered")
        assert "acme-five" in note.message

    def test_the_manifest_is_written_after_the_folders(self, env, monkeypatch):
        skill = env.skill("acme/one", versions=2)
        assert env.add("acme/one").ok
        manifest = env.manifest_text()
        _crash_at(monkeypatch, "lock_written", 1)
        with pytest.raises(Crash):
            env.update("acme/one@1")
        assert env.manifest_text() == manifest
        monkeypatch.setattr(fs, "_checkpoint", lambda name: None)
        assert env.update("acme/one@1").ok
        assert env.manifest()["acme/one"].spec == "1"
        assert skill.skill_id == env.lock().skills["acme-one"]["skill_id"]


class TestCrashBeforeApply:
    def test_a_crash_while_staging_changes_nothing_and_is_cleaned_up(self, env, monkeypatch):
        _five(env)
        before = env.tree()
        real_stage = fs.stage
        count = []

        def stage(*args, **kwargs):
            count.append(1)
            if len(count) == 3:
                raise Crash("staging")
            return real_stage(*args, **kwargs)

        monkeypatch.setattr(fs, "stage", stage)
        with pytest.raises(Crash):
            env.update()
        monkeypatch.setattr(fs, "stage", real_stage)
        after = {k: v for k, v in env.tree().items() if not k.startswith(".decimal/staging/")}
        assert after == before
        assert set(_versions(env).values()) == {1}

        calls = len(env.calls())
        assert env.install().ok
        assert len(env.calls()) == calls
        assert not env.project.staging_dir.exists()


class TestJournalConflict:
    def test_a_folder_matching_neither_side_is_exit_1_and_untouched(self, env):
        env.skill("acme/one")
        assert env.add("acme/one").ok
        entry = dict(env.lock().skills["acme-one"])
        other = dict(entry, version_id="00000000-0000-4000-8000-0000000000ff",
                     files=dict(entry["files"], **{"SKILL.md": "f" * 64}))
        # The journal says no folder stood there and `other` was arriving; the folder
        # that stands there is neither.
        journal = fs.Journal("20261020T000000Z-0000abcd", "acme-one", None, other, False)
        env.project.journal_path.write_text(json.dumps(journal.to_dict()))
        before = env.tree()
        for call in (env.install, env.update, lambda: env.remove("acme-one"),
                     lambda: env.add("acme/one")):
            result = call()
            assert result.exit_code == E.ExitCode.PROBLEMS, result.problems
            assert isinstance(error_of(result), E.JournalConflictError)
            assert env.tree() == before


class TestOneRunAtATime:
    @pytest.mark.parametrize("op", ["add", "install", "update", "remove"])
    def test_another_run_holding_the_project_is_exit_1(self, env, monkeypatch, op):
        env.skill("acme/one")
        assert env.add("acme/one").ok
        monkeypatch.setattr(ops, "LOCK_TIMEOUT", 0.2)
        before = env.tree()
        with fs.ProcessLock(env.project):
            result = {"add": lambda: env.add("acme/one@1"), "install": env.install,
                      "update": env.update, "remove": lambda: env.remove("acme-one")}[op]()
        assert result.exit_code == E.ExitCode.PROBLEMS
        assert isinstance(error_of(result), E.SkillsBusyError)
        assert env.tree() == before

    def test_the_hold_is_refreshed_between_skills(self, env, monkeypatch):
        _five(env)
        refreshed = []
        real = fs.ProcessLock.refresh

        def refresh(self):
            refreshed.append(1)
            return real(self)

        monkeypatch.setattr(fs.ProcessLock, "refresh", refresh)
        assert env.update().ok
        assert len(refreshed) >= 5

    def test_a_run_whose_hold_was_taken_over_stops(self, env, monkeypatch):
        _five(env)
        calls = []

        def taken_over(self):
            calls.append(1)
            if len(calls) == 3:
                raise E.SkillsBusyError("another run took over")

        monkeypatch.setattr(fs.ProcessLock, "refresh", taken_over)
        result = env.update()
        assert result.exit_code == E.ExitCode.PROBLEMS
        assert isinstance(error_of(result), E.SkillsBusyError)
        assert list(_versions(env).values()).count(2) < 5, "it stopped"
        assert set(env.states().values()) == {fs.State.CLEAN}
        assert not env.project.journal_path.exists()


class TestSwapWarnings:
    def test_a_hand_placed_file_the_new_version_serves_is_backed_up_and_reported(self, env):
        logo = {"path": "assets/logo.png", "content": None}
        skill = env.skill("acme/one", files=({"path": "references/a.md", "content": "a\n"}, logo))
        assert env.add("acme/one").ok
        placed = env.folder("acme-one") / "assets" / "logo.png"
        placed.parent.mkdir()
        placed.write_bytes(b"\x89PNG mine")
        env.registry.add_version(skill, files=[{"path": "references/a.md", "content": "a\n"},
                                               {"path": "assets/logo.png", "content": "svg\n"}])
        result = env.update()
        assert result.ok, result.problems
        (note,) = notices(result, "hand_placed")
        assert "assets/logo.png" in note.message and ".decimal/backup/" in note.message
        assert placed.read_text() == "svg\n"


@pytest.mark.parametrize("op", ["install", "status", "remove", "frozen"])
def test_an_os_error_is_a_result_not_a_traceback(env, monkeypatch, op):
    env.skill("acme/one")
    assert env.add("acme/one").ok

    def denied(*args, **kwargs):
        raise PermissionError(13, "Permission denied", str(env.project.skills_root))

    monkeypatch.setattr(fs, "scan_root", denied)
    result = {"install": env.install, "status": env.status,
              "remove": lambda: env.remove("acme-one"),
              "frozen": lambda: env.install(frozen=True)}[op]()
    assert result.exit_code == E.ExitCode.PROBLEMS
    err = error_of(result)
    assert "Permission denied" in err.message and str(env.project.skills_root) in err.message
    assert not env.project.process_lock_path.exists()


def test_interrupted_staging_from_another_txid_is_discarded(env):
    env.skill("acme/one")
    assert env.add("acme/one").ok
    stale = env.project.staging_dir / "20261020T000000Z-deadbeef" / "acme-one"
    stale.mkdir(parents=True)
    (stale / "SKILL.md").write_text("half written\n")
    assert env.install().ok
    assert not env.project.staging_dir.exists()
