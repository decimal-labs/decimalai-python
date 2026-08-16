"""Instrumenting an agent must not create files in the user's repo.

The defect: `decimalai.init(openai_agents=True, ...)` — the exact one-liner the
docs open with — pulled every platform skill down to `.agents/skills/` in
whatever directory the user ran from. With no disk-loading runtime in the
process nothing ever read those files back, so the user got a tree of SKILL.md
files in their repo, often inside a git checkout, in exchange for nothing.

`disk_sync` did not distinguish reading from writing. Discovering local skills
and uploading them creates nothing and is fine on by default; pulling skills
down creates a directory. This pins that split.
"""

from __future__ import annotations

import pytest

from decimalai.skill_router import should_auto_pull_to_disk


@pytest.fixture(autouse=True)
def _reset_hint_state(monkeypatch):
    """The hint is once-per-process; tests need it armed each time."""
    import decimalai.skill_router as sr

    monkeypatch.setattr(sr, "_disk_mirror_hinted", False)
    for var in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CURSOR_AGENT",
                "DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def in_empty_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _set_authority(monkeypatch, value):
    """Point _get_config() at a stub carrying just the field under test."""
    import decimalai._config as cfg

    class _Stub:
        skill_authority = value

    monkeypatch.setattr(cfg, "_get_config", lambda: _Stub())


def test_declines_to_create_a_directory_nobody_asked_for(in_empty_dir, monkeypatch):
    _set_authority(monkeypatch, "auto")
    allowed, why = should_auto_pull_to_disk("universal")
    assert not allowed
    assert "nothing reads" in why


def test_a_detected_harness_is_not_consent_to_write(in_empty_dir, monkeypatch):
    """Being inside Claude Code does not authorize creating the directory.

    This is the case that made the original fix look like it worked while it
    did not: the developer's own shell exports CLAUDECODE=1, so treating runtime
    detection as permission meant the write still happened everywhere it
    mattered — including under the test suite.
    """
    _set_authority(monkeypatch, "auto")
    monkeypatch.setenv("CLAUDECODE", "1")
    allowed, _ = should_auto_pull_to_disk("universal")
    assert not allowed


def test_a_detected_harness_gets_told_the_feature_exists(in_empty_dir, monkeypatch, caplog):
    _set_authority(monkeypatch, "auto")
    monkeypatch.setenv("CLAUDECODE", "1")
    with caplog.at_level("INFO", logger="decimalai.skill_router"):
        should_auto_pull_to_disk("universal")
        should_auto_pull_to_disk("universal")  # once per process, not per call

    hints = [r for r in caplog.records if "reads skills from disk" in r.message]
    assert len(hints) == 1
    assert "skill_authority='harness'" in hints[0].getMessage()


def test_no_hint_when_no_harness_is_present(in_empty_dir, monkeypatch, caplog):
    """A plain script gets silence, not advice about an editor it isn't using."""
    _set_authority(monkeypatch, "auto")
    with caplog.at_level("INFO", logger="decimalai.skill_router"):
        should_auto_pull_to_disk("universal")
    assert not [r for r in caplog.records if "reads skills from disk" in r.message]


def test_explicit_disk_sync_true_is_honoured(in_empty_dir, monkeypatch):
    """`install(disk_sync=True)` by hand is the user asking. Don't second-guess.

    This is the whole distinction: `disk_sync` is tri-state, and only `None`
    means "we guessed". The guess is what wrote to people's repos; the explicit
    value is a request and must keep working in an empty directory.
    """
    _set_authority(monkeypatch, "auto")
    allowed, why = should_auto_pull_to_disk("universal", explicitly_requested=True)
    assert allowed
    assert "explicitly" in why


def test_explicit_request_is_not_assumed_from_a_derived_value(in_empty_dir, monkeypatch):
    """The derived default also lands on True — it must not read as a request."""
    _set_authority(monkeypatch, "auto")
    assert not should_auto_pull_to_disk("universal", explicitly_requested=False)[0]


def test_explicit_harness_authority_still_mirrors(in_empty_dir, monkeypatch):
    """The opt-in must work on a first run, before any directory exists."""
    _set_authority(monkeypatch, "harness")
    allowed, why = should_auto_pull_to_disk("universal")
    assert allowed
    assert "harness" in why


def test_an_existing_directory_keeps_getting_updated(in_empty_dir, monkeypatch):
    """Presence of the directory is the user's consent; stale is the surprise."""
    _set_authority(monkeypatch, "auto")
    (in_empty_dir / ".agents" / "skills").mkdir(parents=True)
    allowed, why = should_auto_pull_to_disk("universal")
    assert allowed
    assert ".agents/skills" in why


def test_consent_is_read_per_agent_not_globally(in_empty_dir, monkeypatch):
    """A Cursor checkout must not authorize writing Claude Code's directory."""
    _set_authority(monkeypatch, "auto")
    (in_empty_dir / ".agents" / "skills").mkdir(parents=True)
    assert should_auto_pull_to_disk("universal")[0]
    assert not should_auto_pull_to_disk("claude-code")[0]


def test_an_unknown_agent_name_never_writes(in_empty_dir, monkeypatch):
    """An agent with no known path has no directory to consent with."""
    _set_authority(monkeypatch, "auto")
    assert not should_auto_pull_to_disk("not-a-real-runtime")[0]


def test_an_uninitialised_config_does_not_authorize_writing(in_empty_dir, monkeypatch):
    """_get_config() raises before init(); that must fail closed, not open."""
    import decimalai._config as cfg

    def _boom():
        raise cfg.DecimalConfigError("init() not called")

    monkeypatch.setattr(cfg, "_get_config", _boom)
    assert not should_auto_pull_to_disk("universal")[0]
