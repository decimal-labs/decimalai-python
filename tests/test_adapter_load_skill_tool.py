"""Adapter registration tests for the native load_skill tool.

  - openai_agents, pydantic_ai and anthropic patch nothing: ``agent_config()``
    hands the tool to the developer, and tests/test_openai_agents_load_agent.py,
    tests/test_pydantic_ai_load_agent.py and tests/test_anthropic_load_agent.py
    drive it through the real frameworks.
  - langchain: instrument(enable_load_skill_tool=True) is accepted but
    DORMANT — logs a warning, does not raise.
"""

from __future__ import annotations

import logging
import os



# Silence the one-shot disk-runtime warning (Claude Code / Cursor detection).
os.environ.setdefault("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")


# ── langchain: dormant param ─────────────────────────────────────


class TestDormantAdapters:
    def test_langchain_enable_load_skill_tool_warns_and_stays_dormant(
        self, caplog, monkeypatch,
    ):
        import decimalai.langchain as lc

        # instrument() early-returns when already installed — force a fresh
        # pass so the dormancy warning path actually runs. disk_sync=False
        # keeps the install away from disk discovery / background sync.
        # The global handler this publishes is torn down by the autouse
        # _clear_langchain_global_handler fixture in tests/conftest.py.
        monkeypatch.setattr(lc, "_installed", False)

        with caplog.at_level(logging.WARNING, logger="decimalai.langchain"):
            lc.instrument(enable_load_skill_tool=True, disk_sync=False)  # must not raise

        assert any(
            "enable_load_skill_tool is not supported" in rec.getMessage()
            for rec in caplog.records
        )
