"""Repeat `decimalai.langchain.instrument()` calls.

`decimalai.init(langchain=True)` and `agent_config()` both call instrument() under the hood, so a
user's later explicit `instrument(...)` hits the `_installed` early return. That used to be a
fully silent no-op; now any configuration a repeat call would drop is surfaced at WARNING.

(Through 0.24 a repeat call also installed the skill loader the first one had left off; the loader
is gone since 0.25, and so are those tests: an agent gets its skills from `agent_config()`.)
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def already_instrumented(monkeypatch):
    """A process where instrument() has already run (e.g. via decimalai.init(langchain=True))."""
    import decimalai.langchain as lc_mod

    monkeypatch.setattr(lc_mod, "_installed", True)
    monkeypatch.setattr(lc_mod, "_install_agent_name", "joke-bot")


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

    def test_the_skill_loaders_arguments_are_gone(self, already_instrumented):
        from decimalai.langchain import instrument

        for removed in ("enable_skill_loader", "enable_load_skill_tool", "skill_body_top_k",
                        "priority_skills", "runtime_policy"):
            with pytest.raises(TypeError, match=removed):
                instrument(**{removed: True})
