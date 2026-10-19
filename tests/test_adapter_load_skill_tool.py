"""Adapter registration tests for the native load_skill tool.

Drives the adapters with *synthetic* framework modules — same pattern as
tests/test_pydantic_ai_skill_loader.py — so no pydantic-ai behavior is
exercised, only our adapter surface:

  - pydantic_ai: patched Agent.__init__ registers load_skill via
    ``self.tool_plain(fn, name=..., description=..., prepare=...)``
    (config-gated) and the skills function via ``self.instructions(fn)``, and
    _skills_system_prompt appends LOAD_SKILL_PROMPT_HINT only once the tool is
    active.
  - openai_agents patches nothing since 0.18: ``load_agent()`` hands the tool
    to the developer, and tests/test_openai_agents_load_agent.py drives it
    through the real Agents SDK.
  - anthropic + langchain: install(enable_load_skill_tool=True) is
    accepted but DORMANT — logs a warning, does not raise.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest

from decimalai.skill_router import LOAD_SKILL_PROMPT_HINT

# Silence the one-shot disk-runtime warning (Claude Code / Cursor detection).
os.environ.setdefault("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")


def _fresh_config(monkeypatch, *, load_skill_tool_env: str | None = None):
    """Install a fresh DecimalConfig built AFTER adjusting the env, so the
    ``load_skill_tool`` default_factory re-reads DECIMALAI_LOAD_SKILL_TOOL."""
    import decimalai._config as config_mod

    if load_skill_tool_env is not None:
        monkeypatch.setenv("DECIMALAI_LOAD_SKILL_TOOL", load_skill_tool_env)
    else:
        monkeypatch.delenv("DECIMALAI_LOAD_SKILL_TOOL", raising=False)
    monkeypatch.setattr(
        config_mod, "_config", config_mod.DecimalConfig(api_key="dai_sk_test"),
    )


# ── pydantic_ai ──────────────────────────────────────────────────


def _make_fake_pydantic_ai():
    """Fake ``pydantic_ai`` module — Agent records instructions(fn) and
    tool_plain(fn, ...) registrations, and has the ``iter`` the run scope
    patches (the only surface the adapter touches)."""
    mod = types.ModuleType("pydantic_ai")

    class Agent:
        def __init__(self, model=None, system_prompt=""):
            self.model = model
            self.base_system_prompt = system_prompt
            self.registered_instructions = []
            self.registered_tools = []
            self.registered_tool_kwargs = []

        def instructions(self, fn):
            # The adapter registers an instructions function, not a system
            # prompt, since 2026-10-14 (see tests/test_pydantic_ai_phase1.py).
            self.registered_instructions.append(fn)
            return fn

        def tool_plain(self, fn, **kwargs):
            # Real signature: name=, description=, prepare=, … — the adapter
            # registers under `decimalai_load_skill` with a `prepare` that
            # picks the exposed name per run.
            self.registered_tools.append(fn)
            self.registered_tool_kwargs.append(kwargs)
            return fn

        @asynccontextmanager
        async def iter(self, prompt):
            # Present because the real one is: `instrument()` brackets every run
            # by patching `Agent.iter`, which every supported pydantic-ai has.
            yield prompt

    mod.Agent = Agent
    return mod, Agent


@pytest.fixture
def fake_pydantic_ai(monkeypatch):
    """Synthetic pydantic_ai + full adapter-state reset (mirrors
    tests/test_pydantic_ai_skill_loader.py, plus the new tool flag)."""
    import decimalai.pydantic_ai as pa

    mod, Agent = _make_fake_pydantic_ai()
    original_init = Agent.__init__
    original_iter = Agent.iter

    monkeypatch.setitem(sys.modules, "pydantic_ai", mod)
    monkeypatch.setattr(pa, "_skill_loader_installed", False)
    # `install()` also installs the run scope and name capture on this class;
    # reset them with the loader so each test patches its own fake.
    monkeypatch.setattr(pa, "_run_scope_installed", False)
    monkeypatch.setattr(pa, "_name_capture_installed", False)
    monkeypatch.setattr(pa, "_default_agent_name", None)
    monkeypatch.setattr(pa, "_skill_router_singleton", None)
    monkeypatch.setattr(pa, "_load_skill_tool_active", False)

    yield mod, Agent, pa

    Agent.__init__ = original_init
    Agent.iter = original_iter


class TestPydanticAiLoadSkillTool:
    def test_tool_registered_via_tool_plain(self, fake_pydantic_ai):
        _, Agent, pa = fake_pydantic_ai

        pa.install(enable_skill_loader=True)
        agent = Agent("openai:gpt-4o", system_prompt="BASE")

        assert len(agent.registered_tools) == 1
        assert agent.registered_tools[0].__name__ == "load_skill"
        # Keyed under the namespaced name so a caller's own `load_skill` can
        # still register; `prepare` shows it to the model as `load_skill`
        # unless that name is taken (tests/test_pydantic_ai_phase1.py).
        assert agent.registered_tool_kwargs[0]["name"] == pa._NAMESPACED_TOOL_NAME
        assert agent.registered_tool_kwargs[0]["prepare"] is pa._prepare_load_skill
        # The module flag now gates the prompt hint.
        assert pa._load_skill_tool_active is True
        # The skills instructions function is still registered alongside the tool.
        assert pa._skills_system_prompt in agent.registered_instructions
        # ...and the base prompt survived the patched __init__.
        assert agent.base_system_prompt == "BASE"

    def test_registered_tool_calls_router_load_skill(self, fake_pydantic_ai):
        _, Agent, pa = fake_pydantic_ai

        pa.install(enable_skill_loader=True)
        agent = Agent("openai:gpt-4o")
        tool_fn = agent.registered_tools[0]

        fake_router = MagicMock()
        fake_router.load_skill.return_value = "## Skill: x\n\nbody"
        pa._skill_router_singleton = fake_router

        out = tool_fn("x")

        assert out == "## Skill: x\n\nbody"
        # scope=None because this call runs outside any agent run — there is no
        # trace to attribute the load to, and the adapter must say so rather
        # than invent a run key. agent_name is the name the run is ROUTED
        # under (`_run_agent_name`); this stand-in Agent has none, so None. It
        # used to be left off entirely, so the platform resolved the body for
        # no agent even when routing had named one (see
        # tests/test_pydantic_ai_phase1.py::TestLoadSkillNamesTheRoutedAgent).
        fake_router.load_skill.assert_called_once_with("x", scope=None, agent_name=None)

    def test_registered_tool_names_the_bound_agent(self, fake_pydantic_ai):
        _, Agent, pa = fake_pydantic_ai

        pa.install(agent_name="support", enable_skill_loader=True)
        agent = Agent("openai:gpt-4o")
        tool_fn = agent.registered_tools[0]
        fake_router = MagicMock()
        fake_router.load_skill.return_value = "## Skill: x\n\nbody"
        pa._skill_router_singleton = fake_router

        tool_fn("x")

        fake_router.load_skill.assert_called_once_with("x", scope=None, agent_name="support")

    def test_tool_returns_error_string_when_router_unavailable(self, fake_pydantic_ai):
        """The tool must always return a string the model can act on."""
        _, Agent, pa = fake_pydantic_ai

        pa.install(enable_skill_loader=True)
        agent = Agent("openai:gpt-4o")
        tool_fn = agent.registered_tools[0]

        fake_router = MagicMock()
        fake_router.load_skill.side_effect = RuntimeError("backend down")
        pa._skill_router_singleton = fake_router

        out = tool_fn("x")
        assert isinstance(out, str)
        assert "load_skill error" in out

    def test_kill_switch_env_skips_tool_plain(self, fake_pydantic_ai, monkeypatch):
        _, Agent, pa = fake_pydantic_ai
        _fresh_config(monkeypatch, load_skill_tool_env="0")

        pa.install(enable_skill_loader=True)
        agent = Agent("openai:gpt-4o")

        assert agent.registered_tools == []
        assert pa._load_skill_tool_active is False
        # Only the tool is gated — the skill loader itself still installs.
        assert pa._skills_system_prompt in agent.registered_instructions

    def test_prompt_hint_appended_only_when_tool_active(self, fake_pydantic_ai):
        _, _, pa = fake_pydantic_ai

        fake_router = MagicMock()
        fake_router.build_prompt_fragment.return_value = (
            "## Available Skills\n| a | b |", "rt_7",
        )
        pa._skill_router_singleton = fake_router

        class _Ctx:
            agent = None

        # Flag off (no agent got the tool) → no hint.
        out = asyncio.run(pa._skills_system_prompt(_Ctx()))
        assert out == "## Available Skills\n| a | b |"
        assert LOAD_SKILL_PROMPT_HINT not in out

        # Flag on → hint appended after the fragment.
        pa._load_skill_tool_active = True
        out2 = asyncio.run(pa._skills_system_prompt(_Ctx()))
        assert out2.startswith("## Available Skills")
        assert out2.endswith(LOAD_SKILL_PROMPT_HINT)


# ── anthropic + langchain: dormant param ─────────────────────────


class TestDormantAdapters:
    def test_anthropic_enable_load_skill_tool_warns_and_stays_dormant(self, caplog):
        import decimalai.anthropic as an

        with caplog.at_level(logging.WARNING, logger="decimalai.anthropic"):
            an.install(enable_load_skill_tool=True)  # must not raise

        assert any(
            "enable_load_skill_tool is not supported" in rec.getMessage()
            for rec in caplog.records
        )

    def test_langchain_enable_load_skill_tool_warns_and_stays_dormant(
        self, caplog, monkeypatch,
    ):
        import decimalai.langchain as lc

        # install() early-returns when already installed — force a fresh
        # pass so the dormancy warning path actually runs. disk_sync=False
        # keeps the install away from disk discovery / background sync.
        # The global handler this publishes is torn down by the autouse
        # _clear_langchain_global_handler fixture in tests/conftest.py.
        monkeypatch.setattr(lc, "_installed", False)

        with caplog.at_level(logging.WARNING, logger="decimalai.langchain"):
            lc.install(enable_load_skill_tool=True, disk_sync=False)  # must not raise

        assert any(
            "enable_load_skill_tool is not supported" in rec.getMessage()
            for rec in caplog.records
        )
