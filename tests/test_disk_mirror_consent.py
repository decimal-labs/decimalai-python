"""Instrumenting an agent must not create files in the user's repo.

The defect: `decimalai.init(openai_agents=True, ...)` — the exact one-liner the
docs open with — pulled every platform skill down to `.agents/skills/` in
whatever directory the user ran from. With no disk-loading runtime in the
process nothing ever read those files back, so the user got a tree of SKILL.md
files in their repo, often inside a git checkout, in exchange for nothing.

The adapters no longer write disk at all (`tests/test_router_skillpkg_shims.py`
pins that). `should_auto_pull_to_disk` stays for code that still asks the old
question, and its answer changed in one place: a skills directory that already
exists is no longer consent. `decimalai skills` creates `.agents/skills` itself,
so "the folder exists" would read every managed project as a request to mirror
every org skill into it.
"""

from __future__ import annotations

import pytest

from decimalai.skill_router import should_auto_pull_to_disk


@pytest.fixture(autouse=True)
def _no_disk_runtime(monkeypatch):
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
    assert "nobody asked" in why and "decimalai skills add" in why


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


def test_asking_logs_nothing(in_empty_dir, monkeypatch, caplog):
    """It only answers the question. The notice for someone who asked for skills on
    disk comes from instrument(), once (see tests/test_router_skillpkg_shims.py); the
    old hint offered disk mirroring, which no longer exists."""
    import decimalai.skill_router as sr

    # Armed as the old once-per-process hint would be on a first call.
    monkeypatch.setattr(sr, "_disk_mirror_hinted", False, raising=False)
    _set_authority(monkeypatch, "auto")
    monkeypatch.setenv("CLAUDECODE", "1")
    with caplog.at_level("INFO", logger="decimalai.skill_router"):
        should_auto_pull_to_disk("universal")
    assert caplog.records == []


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


@pytest.mark.parametrize("agent,folder", [
    ("universal", ".agents/skills"),
    ("claude-code", ".claude/skills"),
    ("cursor", ".agents/skills"),
])
def test_an_existing_directory_is_not_consent(in_empty_dir, monkeypatch, agent, folder):
    """It used to be: the folder's existence meant "keep it up to date". That turned
    every checkout with `.agents/skills` (which `decimalai skills install` creates)
    into a target for a mirror of every org skill."""
    _set_authority(monkeypatch, "auto")
    (in_empty_dir / folder).mkdir(parents=True)
    assert not should_auto_pull_to_disk(agent)[0]


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


# ── the duplicate-injection warning describes what the SDK does now ───


def test_the_duplicate_warning_fires_once_inside_a_disk_runtime(monkeypatch, caplog):
    import decimalai.skill_router as sr

    monkeypatch.setattr(sr, "_disk_runtime_warned", False)
    monkeypatch.setenv("CLAUDECODE", "1")
    with caplog.at_level("WARNING", logger="decimalai.skill_router"):
        sr._warn_if_disk_runtime_detected("langchain")
        sr._warn_if_disk_runtime_detected("langchain")

    (record,) = [r for r in caplog.records if "decimalai.langchain" in r.getMessage()]
    message = record.getMessage()
    assert record.levelname == "WARNING"
    assert "claude-code" in message
    # 0.15: nothing is mirrored; registry skills reach disk only through the commands.
    assert "no longer writes skills to disk" in message
    assert "`decimalai skills add`" in message and "`decimalai skills install`" in message
    assert "mirroring" not in message and "disk_sync" not in message
    assert "DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING=1" in message


def test_the_duplicate_warning_needs_a_disk_runtime(monkeypatch, caplog):
    """The trigger is unchanged: no disk-loading runtime, no warning."""
    import decimalai.skill_router as sr

    monkeypatch.setattr(sr, "_disk_runtime_warned", False)
    with caplog.at_level("WARNING", logger="decimalai.skill_router"):
        sr._warn_if_disk_runtime_detected("langchain")
    assert [r for r in caplog.records if "decimalai.langchain" in r.getMessage()] == []
    assert sr._disk_runtime_warned is False
