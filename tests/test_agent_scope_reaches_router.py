"""The installed agent name must reach the Router on the auto-inject paths.

Why this file exists. ``resolve_agent_skills`` on the platform builds an
agent's offered set from two sources: org-OWNED skills whose ``offer_scope``
is ``workspace`` (ambient — every agent sees them), and ``SkillSubscription``
rows, where an agent-scope row matches on ``agent_name``.

A skill *pulled from the registry* belongs to another org, so it is never in
the ambient set. It reaches one agent only through an agent-scope row — which
the resolver can only match when the router is told which agent it is serving.

``langchain`` and ``anthropic`` both auto-inject skills without sending the
name, so every registry skill attached to a single agent was silently never
offered: no error, no empty-menu warning, just a prompt with none of the
skills the user picked. ``openai_agents`` and ``pydantic_ai`` already send it.

These tests pin the argument at the call boundary rather than asserting on a
rendered fragment, because the bug was never in what the Router returned — it
was in what the adapter asked for.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest


class _SpyRouter:
    """Records what the adapter asked the Router for."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def build_prompt_fragment(
        self, query: Optional[str] = None, **kwargs: Any
    ) -> Tuple[str, Optional[str]]:
        self.calls.append({"query": query, **kwargs})
        return "", None

    # `skill_system` stamps rails off the router after the call; the anthropic
    # path tolerates these being absent, but a spy that answers them keeps the
    # test exercising the real code path instead of an exception handler.
    def consume_routing_id(self) -> None:
        return None

    def consume_offered_names(self) -> List[str]:
        return []

    def consume_delivered_names(self) -> List[str]:
        return []

    def consume_loaded_names(self) -> List[str]:
        return []


class TestLangChainSendsAgentName:
    def _inject(self, monkeypatch, agent_name: Optional[str]):
        from langchain_core.messages import HumanMessage

        import decimalai.langchain as lc

        spy = _SpyRouter()
        monkeypatch.setattr(lc, "_get_skill_router", lambda: spy)
        monkeypatch.setattr(lc, "_install_agent_name", agent_name)
        lc._inject_skills_into_input([HumanMessage(content="refund my order")])
        return spy

    def test_installed_agent_name_reaches_the_router(self, monkeypatch):
        spy = self._inject(monkeypatch, "refund-bot")

        assert spy.calls, "the adapter never consulted the Router"
        assert spy.calls[0]["agent_name"] == "refund-bot", (
            "agent_name did not reach build_prompt_fragment — agent-scope "
            "skills (every registry skill pulled onto one agent) resolve to "
            "nothing without it"
        )

    def test_query_is_still_extracted(self, monkeypatch):
        """Scoping must not cost the semantic-routing signal."""
        spy = self._inject(monkeypatch, "refund-bot")
        assert spy.calls[0]["query"] == "refund my order"

    def test_no_installed_name_sends_none_not_a_placeholder(self, monkeypatch):
        """Un-named installs keep the old org-wide behaviour.

        A placeholder here would be worse than None: it would match no
        agent-scope row and silently scope the menu to an agent that does
        not exist, rather than falling back to the ambient set.
        """
        spy = self._inject(monkeypatch, None)
        assert spy.calls[0]["agent_name"] is None

    def test_router_failure_still_returns_input_unchanged(self, monkeypatch):
        """Fail-open is the property that lets this run in a request path."""
        from langchain_core.messages import HumanMessage

        import decimalai.langchain as lc

        class _Boom:
            def build_prompt_fragment(self, *a: Any, **k: Any):
                raise RuntimeError("router down")

        monkeypatch.setattr(lc, "_get_skill_router", lambda: _Boom())
        monkeypatch.setattr(lc, "_install_agent_name", "refund-bot")

        original = [HumanMessage(content="hi")]
        assert lc._inject_skills_into_input(original) is original


class TestAnthropicSendsAgentName:
    def _inject(self, monkeypatch, agent_name: Optional[str]) -> _SpyRouter:
        import decimalai.anthropic as an

        spy = _SpyRouter()
        monkeypatch.setattr(an, "_get_skill_router", lambda: spy)
        monkeypatch.setattr(an, "_install_agent_name", agent_name)

        kwargs: Dict[str, Any] = {
            "system": "You are a refunds agent.",
            "messages": [{"role": "user", "content": "refund my order"}],
        }
        an._inject_skills_into_create_kwargs(kwargs)
        return spy

    def test_installed_agent_name_reaches_the_router(self, monkeypatch):
        spy = self._inject(monkeypatch, "refund-bot")

        assert spy.calls, "the adapter never consulted the Router"
        assert spy.calls[0]["agent_name"] == "refund-bot"

    def test_no_installed_name_sends_none(self, monkeypatch):
        spy = self._inject(monkeypatch, None)
        assert spy.calls[0]["agent_name"] is None

    def test_instrument_records_the_agent_name(self, monkeypatch):
        """`instrument(agent_name=...)` is the only way to set it on this
        adapter — it has no `init()` wiring, unlike langchain."""
        import decimalai.anthropic as an

        monkeypatch.setattr(an, "_install_agent_name", None)
        an.instrument(agent_name="refund-bot")
        assert an._install_agent_name == "refund-bot"

    def test_instrument_without_agent_name_does_not_clear_it(self, monkeypatch):
        """A later bare `instrument()` must not silently unscope the menu."""
        import decimalai.anthropic as an

        monkeypatch.setattr(an, "_install_agent_name", "refund-bot")
        an.instrument()
        assert an._install_agent_name == "refund-bot"


def test_openai_agents_still_passes_agent_name_at_call_time():
    """Guards the adapter this fix was modelled on against regression."""
    import inspect

    import decimalai.openai_agents as oa

    src = inspect.getsource(oa)
    assert "agent_name=getattr(agent, \"name\", None)" in src, (
        "openai_agents stopped passing agent_name — the reference "
        "implementation for per-agent skill scoping"
    )
