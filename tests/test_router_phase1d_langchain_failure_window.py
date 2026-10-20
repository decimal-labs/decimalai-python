"""Phase 1d, on real LangChain graphs: one failed route does not strip skills
from the NEXT conversation, and does not make one turn retry on every model call.

LangChain routes without ``scope=`` (its injection runs inside arbitrary model
calls), and its ambient run id is the id of the runnable around each model
call: different for every model call of one turn, as the first test pins. So
the router cannot key a failure by the turn. It holds an unscoped failure for
a short window instead (see ``test_router_phase1d_failure_window.py``).

Driven end to end: ``create_agent`` with a tool loop, the real skill loader
patch, a real ``SkillRouter`` whose transport is a stubbed ``httpx.request``,
and a fake clock where a model call takes 1 s and a timed-out route 5 s.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain")

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from decimalai import skill_router as sr
from decimalai.skill_router import SkillRouter, _ambient_scope
from tests.test_langchain_phase1 import loader, sdk  # noqa: F401  (fixtures)


@pytest.fixture(autouse=True)
def _no_menu_fallback(monkeypatch):
    """These tests pin how long a FAILED route is held. Since 2026-10-20 a slow
    route is first answered by the agent's menu (test_route_menu_fallback.py),
    which this fake platform does not serve; off here, so the rule is tested alone."""
    monkeypatch.setattr(sr, "_MENU_FALLBACK_REASONS", frozenset())


QUERY = "I want a refund for my order"
ROUTE = {
    "strategy": "smart_routing",
    "skills": [{"name": "refund-policy", "description": "Refund rules"}],
    "prompt_fragment": "## Recommended Skills\n| refund-policy | Refund rules |",
    "routing_id": "rt_" + "e" * 24,
}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return 1_700_000_000.0 + self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def __getattr__(self, name):
        import time

        return getattr(time, name)


@tool
def lookup_order(order: str) -> str:
    """Look up an order by id."""
    return "order 42: delivered, opened"


class _ToolLoopModel(BaseChatModel):
    """Calls ``lookup_order`` twice, then answers: three model calls per turn.

    Each call costs 1 s of fake time and records whether the routed skill was
    in its prompt.
    """

    clock: Any = None
    seen: List[bool] = []

    @property
    def _llm_type(self) -> str:
        return "tool-loop-stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.clock.now += 1.0
        self.seen.append(any("refund-policy" in str(m.content) for m in messages))
        done = sum(1 for m in messages if m.type == "tool")
        if done < 2:
            msg = AIMessage(content="", tool_calls=[
                {"name": "lookup_order", "args": {"order": "42"}, "id": f"call-{done}"},
            ])
        else:
            msg = AIMessage(content="Refunds on opened boxes carry a fee.")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools, **kwargs):
        return self


class _Transport:
    """/route: the first call times out (5 s) unless ``fail_first`` is off; every
    later one answers. Records the ambient run id each call was made under."""

    def __init__(self, clock: _Clock) -> None:
        self.clock = clock
        self.fail_first = True
        self.routes: List[Dict[str, Any]] = []

    def __call__(self, method, url, **kw):
        assert url.endswith("/api/v1/skills/route"), url
        self.routes.append({"scope": _ambient_scope()})
        if self.fail_first and len(self.routes) == 1:
            self.clock.now += 5.0
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json=ROUTE)


@pytest.fixture
def world(monkeypatch, loader):  # noqa: F811
    import decimalai.langchain as lc

    clock = _Clock()
    monkeypatch.setattr(sr, "time", clock)
    monkeypatch.setattr(sr, "_hot_path_breakers", sr._HotPathBreakers())
    transport = _Transport(clock)
    monkeypatch.setattr(httpx, "request", transport)
    monkeypatch.setattr(lc, "_skill_router_singleton", SkillRouter(
        api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=False,
    ))
    model = _ToolLoopModel(clock=clock, seen=[])
    graph = create_agent(model, tools=[lookup_order], system_prompt="You help.", name="support")
    return clock, transport, model, graph


def _conversation(graph) -> None:
    graph.invoke({"messages": [{"role": "user", "content": QUERY}]})


class TestLangChainFailedRoute:
    def test_langchains_ambient_run_id_names_each_model_call_not_the_turn(
        self, world, monkeypatch,
    ):
        """Why the failure cannot be keyed by LangChain's run: every model call
        of one turn arrives under a different ambient id. (A router whose cache
        expires between model calls, so that every call routes.)"""
        import decimalai.langchain as lc

        clock, transport, model, graph = world
        transport.fail_first = False
        monkeypatch.setattr(lc, "_skill_router_singleton", SkillRouter(
            api_key="dai_sk_test", base_url="http://localhost:8000", inject_body=False,
            fragment_cache_ttl=0.5,
        ))
        _conversation(graph)
        scopes = [r["scope"] for r in transport.routes]
        assert len(scopes) == 3 and None not in scopes
        assert len(set(scopes)) == 3, scopes

    def test_the_turn_pays_the_timeout_once(self, world):
        clock, transport, model, graph = world
        _conversation(graph)
        assert len(model.seen) == 3
        assert len(transport.routes) == 1, (
            f"{len(transport.routes)} routes in one turn: every model call retried the dead endpoint"
        )
        assert model.seen == [False, False, False]

    def test_the_next_conversation_gets_its_skills(self, world):
        """The defect: the second conversation, same suggested prompt, was served
        the first one's failure for the rest of the 30 s TTL."""
        clock, transport, model, graph = world
        _conversation(graph)
        clock.now += 3.0  # another user, a few seconds after the first turn ended
        _conversation(graph)

        assert len(transport.routes) == 2
        assert model.seen[3:] == [True, True, True], model.seen
