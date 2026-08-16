"""Adapter registration tests for the native load_skill tool.

Drives the adapters with *synthetic* framework modules — same pattern as
tests/test_pydantic_ai_skill_loader.py — so no pydantic-ai / openai-agents
behavior is exercised, only our adapter surface:

  - pydantic_ai: patched Agent.__init__ registers load_skill via
    ``self.tool_plain(fn)`` (config-gated), and _skills_system_prompt
    appends LOAD_SKILL_PROMPT_HINT only once the tool is active.
  - openai_agents: patched Agent.__init__ appends the function_tool to
    ``agent.tools`` (once per agent, never duplicated), and the wrapped
    instructions callable appends LOAD_SKILL_PROMPT_HINT only when the
    agent actually has the tool.
  - anthropic + langchain: install(enable_load_skill_tool=True) is
    accepted but DORMANT — logs a warning, does not raise.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import types
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
    """Fake ``pydantic_ai`` module — Agent records system_prompt(fn) and
    tool_plain(fn) registrations (the only surface the adapter touches)."""
    mod = types.ModuleType("pydantic_ai")

    class Agent:
        def __init__(self, model=None, system_prompt=""):
            self.model = model
            self.base_system_prompt = system_prompt
            self.registered_system_prompts = []
            self.registered_tools = []

        def system_prompt(self, fn):
            self.registered_system_prompts.append(fn)
            return fn

        def tool_plain(self, fn):
            self.registered_tools.append(fn)
            return fn

    mod.Agent = Agent
    return mod, Agent


@pytest.fixture
def fake_pydantic_ai(monkeypatch):
    """Synthetic pydantic_ai + full adapter-state reset (mirrors
    tests/test_pydantic_ai_skill_loader.py, plus the new tool flag)."""
    import decimalai.pydantic_ai as pa

    mod, Agent = _make_fake_pydantic_ai()
    original_init = Agent.__init__

    monkeypatch.setitem(sys.modules, "pydantic_ai", mod)
    monkeypatch.setattr(pa, "_skill_loader_installed", False)
    monkeypatch.setattr(pa, "_skill_router_singleton", None)
    monkeypatch.setattr(pa, "_load_skill_tool_active", False)

    yield mod, Agent, pa

    Agent.__init__ = original_init


class TestPydanticAiLoadSkillTool:
    def test_tool_registered_via_tool_plain(self, fake_pydantic_ai):
        _, Agent, pa = fake_pydantic_ai

        pa.install(enable_skill_loader=True)
        agent = Agent("openai:gpt-4o", system_prompt="BASE")

        assert len(agent.registered_tools) == 1
        assert agent.registered_tools[0].__name__ == "load_skill"
        # The module flag now gates the prompt hint.
        assert pa._load_skill_tool_active is True
        # The skills system prompt is still registered alongside the tool.
        assert pa._skills_system_prompt in agent.registered_system_prompts
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
        # than invent a run key.
        fake_router.load_skill.assert_called_once_with("x", scope=None)

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
        assert pa._skills_system_prompt in agent.registered_system_prompts

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


# ── openai_agents ────────────────────────────────────────────────


def _make_fake_agents_module():
    """Fake ``agents`` module: Agent stores tools/instructions kwargs;
    function_tool(fn) yields an object with .name and .fn (the two
    attributes the adapter relies on)."""
    mod = types.ModuleType("agents")

    class Agent:
        def __init__(self, *args, **kwargs):
            self.name = kwargs.get("name")
            self.tools = list(kwargs.get("tools") or [])
            self.instructions = kwargs.get("instructions")

    def function_tool(fn):
        return types.SimpleNamespace(name=fn.__name__, fn=fn)

    mod.Agent = Agent
    mod.function_tool = function_tool
    return mod, Agent


@pytest.fixture
def fake_agents(monkeypatch):
    import decimalai.openai_agents as oa
    from decimalai import skill_router as sr

    mod, Agent = _make_fake_agents_module()
    monkeypatch.setitem(sys.modules, "agents", mod)
    monkeypatch.setattr(oa, "_skill_loader_installed", False)
    monkeypatch.setattr(oa, "_skill_router_singleton", None)
    # Don't let another test's offered/delivered-names contextvars leak
    # into instructions_fn's consume_last_*_names().
    sr._last_offered_names_ctx.set(None)
    sr._last_delivered_names_ctx.set(None)
    oa._skills_offered_ctx.set(None)
    oa._skills_delivered_ctx.set(None)

    yield mod, Agent, oa


def _load_skill_tools(agent):
    return [t for t in agent.tools if getattr(t, "name", None) == "load_skill"]


class TestOpenAIAgentsLoadSkillTool:
    def test_tool_appended_to_new_agent(self, fake_agents):
        _, Agent, oa = fake_agents

        oa._install_skill_loader()
        agent = Agent(instructions="base")

        assert len(_load_skill_tools(agent)) == 1
        tool = _load_skill_tools(agent)[0]
        assert callable(tool.fn)
        # String instructions got wrapped into the skill-aware callable.
        assert callable(agent.instructions)

    def test_idempotent_each_new_agent_gets_exactly_one(self, fake_agents):
        _, Agent, oa = fake_agents

        oa._install_skill_loader()
        oa._install_skill_loader()  # second install is a no-op

        a1 = Agent(instructions="one")
        a2 = Agent(instructions="two")

        assert len(_load_skill_tools(a1)) == 1
        assert len(_load_skill_tools(a2)) == 1
        # Each agent owns its own tools list — no cross-agent sharing.
        assert a1.tools is not a2.tools

    def test_pre_existing_load_skill_tool_not_duplicated(self, fake_agents):
        _, Agent, oa = fake_agents

        oa._install_skill_loader()
        mine = types.SimpleNamespace(name="load_skill", fn=lambda name: "mine")
        agent = Agent(instructions="base", tools=[mine])

        assert len(_load_skill_tools(agent)) == 1
        assert agent.tools[0] is mine  # the user's tool wins

    def test_tool_fn_calls_router_load_skill(self, fake_agents):
        _, Agent, oa = fake_agents

        oa._install_skill_loader()
        agent = Agent(instructions="base")
        tool = _load_skill_tools(agent)[0]

        fake_router = MagicMock()
        fake_router.load_skill.return_value = "## Skill: x\n\nbody"
        oa._skill_router_singleton = fake_router

        assert tool.fn("x") == "## Skill: x\n\nbody"
        fake_router.load_skill.assert_called_once_with("x")

    def test_instructions_append_hint_when_agent_has_tool(self, fake_agents):
        _, Agent, oa = fake_agents

        oa._install_skill_loader()
        agent = Agent(instructions="base")

        fake_router = MagicMock()
        fake_router.build_prompt_fragment.return_value = (
            "## Recommended Skills\n| a | b |", "rt_1",
        )
        oa._skill_router_singleton = fake_router

        out = agent.instructions(types.SimpleNamespace(), agent)

        assert out.startswith("## Recommended Skills")
        assert LOAD_SKILL_PROMPT_HINT in out
        assert out.endswith("base")  # base instructions preserved after the fragment
        # Hint sits between the fragment and the base prompt.
        assert out.index("## Recommended Skills") < out.index(LOAD_SKILL_PROMPT_HINT)

    def test_instructions_no_hint_when_tool_gated_off(self, fake_agents, monkeypatch):
        _, Agent, oa = fake_agents
        _fresh_config(monkeypatch, load_skill_tool_env="0")

        oa._install_skill_loader()
        agent = Agent(instructions="base")

        # Gate honored: no tool appended...
        assert _load_skill_tools(agent) == []

        fake_router = MagicMock()
        fake_router.build_prompt_fragment.return_value = (
            "## Recommended Skills\n| a | b |", "rt_1",
        )
        oa._skill_router_singleton = fake_router

        out = agent.instructions(types.SimpleNamespace(), agent)

        # ...so the fragment is delivered WITHOUT the load_skill hint.
        assert "## Recommended Skills" in out
        assert LOAD_SKILL_PROMPT_HINT not in out
        assert out.endswith("base")


class TestOpenAIAgentsDeliveredRail:
    """Activation ladder: the instructions callable drains the Router's
    delivered-names rail into the adapter's per-trace contextvar, exactly
    mirroring the offered rail."""

    def test_instructions_fn_consumes_delivered_rail(self, fake_agents):
        _, Agent, oa = fake_agents
        from decimalai import skill_router as sr

        oa._install_skill_loader()
        agent = Agent(instructions="base")

        fake_router = MagicMock()

        def fake_build(query=None, agent_name=None):
            # What the real build_prompt_fragment does on a body inject.
            sr._last_offered_names_ctx.set(["s1"])
            sr._last_delivered_names_ctx.set(["s1"])
            return ("## Recommended Skills\n| s1 |\n\n## Skill: s1\n\nBODY", "rt_1")

        fake_router.build_prompt_fragment.side_effect = fake_build
        oa._skill_router_singleton = fake_router

        agent.instructions(types.SimpleNamespace(), agent)

        assert oa._consume_skills_offered() == ["s1"]
        assert oa._consume_skills_delivered() == ["s1"]
        # Router rails were drained by the adapter — nothing leaks forward.
        assert sr.consume_last_offered_names() == []
        assert sr.consume_last_delivered_names() == []

    def test_menu_only_leaves_delivered_empty(self, fake_agents):
        _, Agent, oa = fake_agents
        from decimalai import skill_router as sr

        oa._install_skill_loader()
        agent = Agent(instructions="base")

        fake_router = MagicMock()

        def fake_build(query=None, agent_name=None):
            sr._last_offered_names_ctx.set(["s1"])  # menu row only — no body
            return ("## Recommended Skills\n| s1 |", "rt_1")

        fake_router.build_prompt_fragment.side_effect = fake_build
        oa._skill_router_singleton = fake_router

        agent.instructions(types.SimpleNamespace(), agent)

        assert oa._consume_skills_offered() == ["s1"]
        assert oa._consume_skills_delivered() == []


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
