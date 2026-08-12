"""Tests for the Pydantic AI skill-loader adapter (decimalai.pydantic_ai).

Drives the adapter with a *synthetic* ``pydantic_ai.Agent`` — no
pydantic-ai install required — so these run in PR CI alongside the other
framework introspection/handler tests.

Adapter contract under test (decimalai/pydantic_ai.py):
  - ``install(enable_skill_loader=True)`` monkey-patches
    ``pydantic_ai.Agent.__init__`` so every newly constructed Agent gets
    the DecimalAI skills system-prompt function registered via
    ``self.system_prompt(...)`` (see _install_skill_loader / patched_init).
  - The registered function ``_skills_system_prompt`` calls
    ``SkillRouter.build_prompt_fragment()`` and returns the resulting
    prompt fragment (the actual skill knowledge K injected per turn),
    stashing the ``routing_id`` for downstream trace stamping.

We assert the *observable* effects: the function is registered on the
real (synthetic) Agent instance, the base system prompt is preserved,
and invoking the registered function returns the router's fragment and
sets the routing_id — not merely that a patch ran.
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from unittest.mock import MagicMock

import pytest

# The adapter logs a one-shot disk-runtime warning when enabled inside a
# disk-loading runtime (Claude Code / Cursor). Silence it for the suite.
os.environ.setdefault("DECIMALAI_SUPPRESS_DISK_RUNTIME_WARNING", "1")


# ── Synthetic pydantic_ai.Agent ──────────────────────────────────
# Matches only the surface the adapter touches: __init__ and a
# .system_prompt(fn) registration method (which the real pydantic_ai
# Agent exposes as both a decorator and a function call).


def _make_fake_pydantic_ai():
    """Build a fake ``pydantic_ai`` module with a minimal Agent."""
    mod = types.ModuleType("pydantic_ai")

    class Agent:
        def __init__(self, model=None, system_prompt=""):
            self.model = model
            self.base_system_prompt = system_prompt
            # Functions registered via @agent.system_prompt / agent.system_prompt(fn)
            self.registered_system_prompts = []

        def system_prompt(self, fn):
            # Real pydantic_ai supports decorator AND function-call form;
            # both append the fn and return it.
            self.registered_system_prompts.append(fn)
            return fn

    mod.Agent = Agent
    return mod, Agent


@pytest.fixture
def fake_pydantic_ai(monkeypatch):
    """Install a synthetic pydantic_ai module and reset adapter state.

    Restores ``Agent.__init__`` and the install-guard flag so each test
    starts from a clean, un-patched Agent.
    """
    import decimalai.pydantic_ai as pa

    mod, Agent = _make_fake_pydantic_ai()
    original_init = Agent.__init__

    monkeypatch.setitem(sys.modules, "pydantic_ai", mod)
    # Force a fresh install in every test (the adapter is install-once).
    monkeypatch.setattr(pa, "_skill_loader_installed", False)
    # Don't leak a router singleton between tests.
    monkeypatch.setattr(pa, "_skill_router_singleton", None)

    yield mod, Agent, pa

    # Undo the monkeypatch of Agent.__init__ so the synthetic class is clean.
    Agent.__init__ = original_init


# ── Install / patch contract ─────────────────────────────────────


class TestSkillLoaderInstall:
    def test_install_registers_skills_prompt_on_new_agent(self, fake_pydantic_ai):
        """After install(enable_skill_loader=True), constructing an Agent
        registers the DecimalAI skills system-prompt function."""
        _, Agent, pa = fake_pydantic_ai

        pa.install(enable_skill_loader=True)

        agent = Agent("openai:gpt-4o", system_prompt="You are helpful")

        # The real observable effect: our function is now registered.
        assert pa._skills_system_prompt in agent.registered_system_prompts
        # ...and exactly once (no double-registration on a single construct).
        assert agent.registered_system_prompts.count(pa._skills_system_prompt) == 1

    def test_base_system_prompt_preserved(self, fake_pydantic_ai):
        """The patched __init__ still runs the original __init__ — the
        user's base system_prompt survives."""
        _, Agent, pa = fake_pydantic_ai

        pa.install(enable_skill_loader=True)
        agent = Agent("openai:gpt-4o", system_prompt="BASE PROMPT")

        assert agent.base_system_prompt == "BASE PROMPT"
        assert agent.model == "openai:gpt-4o"

    def test_no_loader_when_disabled(self, fake_pydantic_ai):
        """install() with the default (enable_skill_loader=False) must NOT
        patch Agent — constructing one registers nothing."""
        _, Agent, pa = fake_pydantic_ai

        pa.install()  # enable_skill_loader defaults False

        agent = Agent("openai:gpt-4o")
        assert agent.registered_system_prompts == []
        assert pa._skill_loader_installed is False

    def test_install_is_idempotent(self, fake_pydantic_ai):
        """Calling install twice must not double-patch (each new Agent gets
        exactly one registration)."""
        _, Agent, pa = fake_pydantic_ai

        pa.install(enable_skill_loader=True)
        pa.install(enable_skill_loader=True)

        agent = Agent("openai:gpt-4o")
        assert agent.registered_system_prompts.count(pa._skills_system_prompt) == 1


# ── Registered-function behavior ─────────────────────────────────


class TestSkillsSystemPrompt:
    def test_returns_router_fragment_and_sets_routing_id(self, fake_pydantic_ai):
        """The registered function returns the SkillRouter's prompt
        fragment (the injected skill K) and stashes the routing_id.

        routing_id is read inside the SAME async context — _set_routing_id
        writes a ContextVar, which doesn't propagate out of asyncio.run's
        copied context."""
        _, _, pa = fake_pydantic_ai

        fake_router = MagicMock()
        fake_router.build_prompt_fragment.return_value = (
            "## Available Skills\n| code-review | Review a PR |",
            "rt_42",
        )
        pa._skill_router_singleton = fake_router

        class _FakeAgentObj:
            name = "shopper"

        class _Ctx:
            agent = _FakeAgentObj()

        async def _run():
            fragment = await pa._skills_system_prompt(_Ctx())
            return fragment, pa.get_current_routing_id()

        fragment, routing_id = asyncio.run(_run())

        assert fragment == "## Available Skills\n| code-review | Review a PR |"
        assert routing_id == "rt_42"
        # Full-menu mode (query=None) + agent_name threaded from ctx.agent.name.
        fake_router.build_prompt_fragment.assert_called_once_with(
            query=None, agent_name="shopper"
        )

    def test_returns_empty_when_router_unavailable(self, fake_pydantic_ai):
        """If no SkillRouter can be built, the function degrades to '' —
        it must never break agent construction."""
        _, _, pa = fake_pydantic_ai
        pa._skill_router_singleton = None

        # Force _get_skill_router() to fail (no config / import error path).
        import decimalai.pydantic_ai as pa_mod

        def _boom():
            return None

        pa_mod._get_skill_router = _boom  # type: ignore[assignment]

        class _Ctx:
            agent = None

        out = asyncio.run(pa._skills_system_prompt(_Ctx()))
        assert out == ""

    def test_router_exception_is_non_fatal(self, fake_pydantic_ai):
        """A router that raises mid-call yields '' rather than propagating —
        the skill loader is best-effort."""
        _, _, pa = fake_pydantic_ai

        fake_router = MagicMock()
        fake_router.build_prompt_fragment.side_effect = RuntimeError("backend down")
        pa._skill_router_singleton = fake_router

        class _Ctx:
            agent = None

        out = asyncio.run(pa._skills_system_prompt(_Ctx()))
        assert out == ""
