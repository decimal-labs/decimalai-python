"""Repeat `decimalai.langchain.instrument()` calls and the skill loader.

`decimalai.init(langchain=True)` calls instrument() under the hood, so a
user's later explicit `instrument(enable_skill_loader=True)` hits the
`_installed` early return. That used to be a fully silent no-op: the skill
loader was never installed and nothing was logged above DEBUG. Now the
loader — an independent, idempotent monkey-patch — is still installed on a
repeat call, and any other configuration a repeat call would drop is
surfaced at WARNING.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def already_instrumented(monkeypatch):
    """Simulate a process where instrument() has already run (e.g. via
    decimalai.init(langchain=True)), with the loader install stubbed so no
    global monkey-patch leaks out of the test."""
    import decimalai.langchain as lc_mod
    import decimalai.skill_router as sr_mod

    monkeypatch.setattr(lc_mod, "_installed", True)
    monkeypatch.setattr(lc_mod, "_install_agent_name", "joke-bot")
    monkeypatch.setattr(lc_mod, "_skill_loader_installed", False)

    calls: dict = {"install": 0, "disk_warn": []}
    monkeypatch.setattr(
        lc_mod, "_install_skill_loader",
        lambda: calls.__setitem__("install", calls["install"] + 1),
    )
    monkeypatch.setattr(
        sr_mod, "_warn_if_disk_runtime_detected",
        lambda runtime: calls["disk_warn"].append(runtime),
    )
    return calls


class TestRepeatInstrumentSkillLoader:

    def test_repeat_call_installs_skill_loader(self, already_instrumented):
        from decimalai.langchain import instrument

        instrument(enable_skill_loader=True)

        assert already_instrumented["install"] == 1
        assert already_instrumented["disk_warn"] == ["langchain"]

    def test_repeat_call_without_flag_does_not_install(self, already_instrumented):
        from decimalai.langchain import instrument

        instrument()

        assert already_instrumented["install"] == 0

    def test_repeat_call_skips_already_installed_loader(
        self, already_instrumented, monkeypatch
    ):
        import decimalai.langchain as lc_mod
        monkeypatch.setattr(lc_mod, "_skill_loader_installed", True)

        lc_mod.instrument(enable_skill_loader=True)

        assert already_instrumented["install"] == 0
        assert already_instrumented["disk_warn"] == []


class TestRepeatInstrumentWarnsOnDroppedConfig:

    def test_ignored_config_logged_at_warning(self, already_instrumented, caplog):
        from decimalai.langchain import instrument

        with caplog.at_level(logging.WARNING, logger="decimalai.langchain"):
            instrument(agent_name="other-bot", prompts={"system": "You are terse."})

        warning = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning) == 1
        assert "agent_name" in warning[0].getMessage()
        assert "prompts" in warning[0].getMessage()

    def test_no_new_config_stays_quiet(self, already_instrumented, caplog):
        """A repeat call that changes nothing (e.g. init() run twice with the
        same agent_name) must not cry wolf."""
        from decimalai.langchain import instrument

        with caplog.at_level(logging.WARNING, logger="decimalai.langchain"):
            instrument(agent_name="joke-bot")

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
