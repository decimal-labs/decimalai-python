"""Regression: auto/programmatic sync must not blind-clobber newer remote edits.

- A background/programmatic sync now defaults to ``newer_wins`` (was the SDK-level
  ``local_wins``, which always overwrote the dashboard).
- ``local_updated_at`` is derived git-aware so a fresh checkout (every file's
  mtime reset to "now") doesn't make the local copy always "win" — without a
  timestamp, the backend's ``newer_wins`` falls back to local-wins.
"""

from __future__ import annotations

from decimalai.skills import (
    _local_updated_at_iso,
    _with_local_timestamps,
    sync_to_platform,
)


def test_local_updated_at_iso_returns_iso_or_none(tmp_path):
    assert _local_updated_at_iso(str(tmp_path / "missing.md")) is None
    f = tmp_path / "SKILL.md"
    f.write_text("hello")
    ts = _local_updated_at_iso(str(f))
    assert ts and "T" in ts  # ISO-8601 timestamp


def test_with_local_timestamps_adds_only_when_source_path(tmp_path):
    sdir = tmp_path / "s"
    sdir.mkdir()
    (sdir / "SKILL.md").write_text("body")
    out = _with_local_timestamps([
        {"name": "a", "source_path": str(sdir)},
        {"name": "b"},  # no source_path → left untouched
    ])
    assert out[0].get("local_updated_at")
    assert "local_updated_at" not in out[1]


def test_sync_to_platform_defaults_newer_wins_and_sends_timestamp(tmp_path, monkeypatch):
    sdir = tmp_path / "myskill"
    sdir.mkdir()
    (sdir / "SKILL.md").write_text(
        "---\nname: myskill\ndescription: test skill\n---\n\n"
        "## When to use\n\nUse this when testing the sync clobber fix. " * 4
    )

    captured = {}

    class FakeRouter:
        def __init__(self, **kw):
            pass

        def sync_skills(self, skills, **kwargs):
            captured["skills"] = skills
            captured["kwargs"] = kwargs
            return {"created": 1, "updated": 0, "unchanged": 0}

    monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)

    sync_to_platform("dai_sk_test", search_paths=[str(tmp_path)])

    assert captured["kwargs"]["conflict_policy"] == "newer_wins"  # the fix
    assert captured["skills"], "expected the discovered skill to be synced"
    assert all("local_updated_at" in s for s in captured["skills"])


def test_sync_to_platform_allows_local_wins_override(tmp_path, monkeypatch):
    sdir = tmp_path / "ci-skill"
    sdir.mkdir()
    (sdir / "SKILL.md").write_text(
        "---\nname: ci-skill\ndescription: ci\n---\n\n"
        "## When to use\n\nRepo is the source of truth in CI. " * 4
    )
    captured = {}

    class FakeRouter:
        def __init__(self, **kw):
            pass

        def sync_skills(self, skills, **kwargs):
            captured["kwargs"] = kwargs
            return {"created": 1}

    monkeypatch.setattr("decimalai.skill_router.SkillRouter", FakeRouter)
    sync_to_platform("dai_sk_test", search_paths=[str(tmp_path)], conflict_policy="local_wins")
    assert captured["kwargs"]["conflict_policy"] == "local_wins"
