"""The router's loader warns, once, inside a runtime that loads SKILL.md files itself.

Claude Code and Cursor read skills from disk on their own. A skill that the router
delivers and that is also installed reaches the prompt twice, so enabling the loader
there logs one warning that says how to keep a single source. The SDK never writes
skills to disk on its own (`tests/test_router_disk_methods.py` pins that), so the
warning points at the commands that do.
"""

from __future__ import annotations

import pytest

import decimalai.skill_router as sr


@pytest.fixture(autouse=True)
def _no_disk_runtime(monkeypatch):
    for var in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CURSOR_AGENT",
                "DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sr, "_disk_runtime_warned", False)


def test_the_duplicate_warning_fires_once_inside_a_disk_runtime(monkeypatch, caplog):
    monkeypatch.setenv("CLAUDECODE", "1")
    with caplog.at_level("WARNING", logger="decimalai.skill_router"):
        sr._warn_if_disk_runtime_detected("langchain")
        sr._warn_if_disk_runtime_detected("langchain")

    (record,) = [r for r in caplog.records if "decimalai.langchain" in r.getMessage()]
    message = record.getMessage()
    assert record.levelname == "WARNING"
    assert "claude-code" in message
    assert "never writes skills to disk" in message
    assert "`decimalai skills add`" in message and "`decimalai skills install`" in message
    assert "mirroring" not in message and "disk_sync" not in message
    assert "DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING=1" in message


def test_the_duplicate_warning_needs_a_disk_runtime(caplog):
    with caplog.at_level("WARNING", logger="decimalai.skill_router"):
        sr._warn_if_disk_runtime_detected("langchain")
    assert [r for r in caplog.records if "decimalai.langchain" in r.getMessage()] == []
    assert sr._disk_runtime_warned is False
