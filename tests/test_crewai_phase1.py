"""CrewAI Phase 1: say when two skill catalogs are live, and never stall a loop silently.

Two gaps from the 2026-10-13 framework integration audit, both pinned here on
crewai 1.15.27, the only CrewAI the adapter supports:

1. **Two catalogs.** CrewAI has its own Agent Skills (``Agent(skills=...)`` /
   ``Crew(skills=...)``). They render into CrewAI's system prompt — an
   ``<available_skills>`` catalog plus a native ``load_skill`` tool for
   metadata-level skills, an always-on ``<skills>`` block for activated ones.
   Phase 1 warned when the DecimalAI loader's routed menu landed beside them.
   Since native mode (Phase 3, D7) the router stays out of such an agent's calls
   altogether and says so once per agent at INFO; its loads are recorded as
   activations (``tests/native/test_crewai_native.py``).

2. **The routing call and the event loop.** CrewAI hooks are synchronous by
   CrewAI's design (``crewai/hooks/dispatch.py``: hooks "may be invoked from
   async seams, so they must not block on heavy I/O"), and the routing call is
   an HTTP request. Measured below: the default executor (``AgentExecutor``, a
   Flow) runs the hook on a worker thread for every kickoff entry point, sync or
   async, so the caller's loop keeps running while skills are routed. The
   deprecated ``CrewAgentExecutor``'s native-async path runs it ON the loop; the
   adapter cannot await there, so it keeps delivering and says once that the
   loop is blocked.

Layer 1 (``TestHookOnARunningLoop``) needs no crewai and runs on every suite
run. Everything else drives real CrewAI classes — ``Agent``, ``Crew``, ``Task``,
CrewAI's skill loader and executors — with a ``BaseLLM`` subclass that captures
the request instead of sending it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import threading
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

# CrewAI 1.15.27 reads these three (crewai_core/telemetry.py
# `_is_telemetry_disabled`, crewai/events/listeners/tracing/utils.py); it does
# not read CREWAI_TELEMETRY_OPT_OUT. Set before any crewai module loads.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


def _real_crewai_installed() -> bool:
    try:
        return importlib.util.find_spec("crewai.hooks") is not None
    except (ImportError, ValueError):
        return False


HAS_REAL_CREWAI = _real_crewai_installed()

needs_crewai = pytest.mark.skipif(
    not HAS_REAL_CREWAI,
    reason="crewai is not an SDK dependency (it pins openai<3); run in .venv-openai2",
)

SENTINEL = "SENTINEL-SKILLBODY-CREWAI-P1-7a3c"
QUESTION = "What fee applies to an opened box return?"
CREW_SYSTEM = "You are Support. You are terse.\nYour personal goal is: Resolve the ticket"
PREFIX = (
    "== DecimalAI skills ==\n"
    "| refund-policy | how refunds work |\n\n"
    "## Skill: refund-policy\n\n"
    f"Opened boxes carry a 23.5% restocking fee. {SENTINEL}\n"
)
TAIL = "Most relevant for this request: refund-policy."
ROUTING_ID = "rt_" + "d" * 24
NATIVE_NAME = "returns-window"

TWO_CATALOGS = "two skill catalogs"
LOOP_BLOCKED = "running asyncio event loop"


# ── routers ─────────────────────────────────────────────────


class _Router:
    """A constant PREFIX carrying a body, and a record of every call: the thread
    it ran on and whether that thread was running an event loop."""

    offered = ["refund-policy"]

    def __init__(self, delay_s: float = 0.0, ticks: Optional[List[int]] = None) -> None:
        self.delay_s = delay_s
        self.ticks = ticks
        self.calls: List[Dict[str, Any]] = []

    def build_prompt_parts(self, query=None, *, agent_name=None, scope=None, **kw):
        import decimalai.skill_router as sr

        try:
            asyncio.get_running_loop()
            on_loop = True
        except RuntimeError:
            on_loop = False
        before = list(self.ticks) if self.ticks is not None else None
        if self.delay_s:
            time.sleep(self.delay_s)  # the stand-in for a slow /skills/route
        after = list(self.ticks) if self.ticks is not None else None
        self.calls.append({
            "thread": threading.current_thread().name,
            "on_loop": on_loop,
            "ticks_during": None if before is None else len(after) - len(before),
            "query": query,
        })
        sr._last_offered_names_ctx.set(list(self.offered))
        sr._last_delivered_names_ctx.set(["refund-policy"])
        return PREFIX, TAIL, ROUTING_ID


class _OverlappingRouter(_Router):
    """DecimalAI's menu also offers the skill the agent loads natively."""

    offered = ["refund-policy", NATIVE_NAME]


class _EmptyRouter(_Router):
    def build_prompt_parts(self, query=None, **kw):
        import decimalai.skill_router as sr

        self.calls.append({"query": query})
        sr._last_offered_names_ctx.set(None)
        sr._last_delivered_names_ctx.set(None)
        return "", "", None


# ── fixtures ────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Fresh SDK config, fresh adapter globals (the two one-shot warnings
    included), an empty OTel skill rail."""
    import decimalai._config as cfg
    import decimalai.crewai as dc
    from decimalai._config import DecimalConfig
    from decimalai.otel import _active_agent_name, _reset_skill_rails

    saved_cfg = (cfg._config, cfg._client)
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    saved = {
        name: getattr(dc, name)
        for name in (
            "_tracing_installed", "_skill_loader_installed", "_install_agent_name",
            "_skill_router_singleton", "_native_skills_warned", "_loop_block_warned",
            "_router_off_logged",
        )
        if hasattr(dc, name)
    }
    dc._router_off_logged = set()
    dc._skill_loader_installed = False
    dc._install_agent_name = None
    dc._skill_router_singleton = None
    dc._native_skills_warned = False
    dc._loop_block_warned = False
    token = _active_agent_name.set(None)
    _reset_skill_rails()
    yield
    _reset_skill_rails()
    _active_agent_name.reset(token)
    for name, value in saved.items():
        setattr(dc, name, value)
    cfg._config, cfg._client = saved_cfg


@pytest.fixture
def use_router(monkeypatch):
    def _use(router):
        import decimalai.crewai as dc

        monkeypatch.setattr(dc, "_skill_router_singleton", router)
        return router

    return _use


ROUTER_OFF = "does not inject"


def _infos(caplog, needle: str) -> List[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == "decimalai.crewai" and r.levelno == logging.INFO
        and needle in r.getMessage()
    ]


def _warnings(caplog, needle: str) -> List[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == "decimalai.crewai" and r.levelno >= logging.WARNING
        and needle in r.getMessage()
    ]


def _context(messages=None, *, agent=None) -> SimpleNamespace:
    """The members of ``crewai.hooks.LLMCallHookContext`` the adapter reads;
    ``test_crewai_skill_delivery.py`` pins this shape to the real class."""
    return SimpleNamespace(
        executor="executor",
        messages=messages if messages is not None else [
            {"role": "system", "content": CREW_SYSTEM},
            {"role": "user", "content": QUESTION},
        ],
        agent=agent if agent is not None else SimpleNamespace(role="Support"),
        task=SimpleNamespace(description=QUESTION),
        crew=None,
        llm=None,
        iterations=0,
    )


# ── layer 1: no crewai needed ───────────────────────────────


class TestHookOnARunningLoop:
    def test_a_hook_run_on_the_loop_warns_once_and_still_delivers(self, use_router, caplog):
        """CrewAI hooks cannot await. When CrewAI runs this one on the thread
        that is running the event loop, the routing request blocks the loop
        until it returns — the adapter says so, once, and still delivers:
        dropping skills there would swap a visible stall for silent
        non-delivery."""
        from decimalai.crewai import _skill_hook

        router = use_router(_Router())
        first, second = _context(), _context()

        async def _turns() -> None:
            _skill_hook(first)
            _skill_hook(second)

        with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
            asyncio.run(_turns())

        assert [c["on_loop"] for c in router.calls] == [True, True]
        assert len(_warnings(caplog, LOOP_BLOCKED)) == 1, caplog.text
        for ctx in (first, second):
            assert any(SENTINEL in str(m["content"]) for m in ctx.messages), (
                "the warning cost the delivery"
            )

    def test_off_the_loop_there_is_nothing_to_warn_about(self, use_router, caplog):
        from decimalai.crewai import _skill_hook

        use_router(_Router())
        with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
            _skill_hook(_context())
        assert _warnings(caplog, LOOP_BLOCKED) == []

    def test_an_agent_without_native_skills_is_not_warned_about(self, use_router, caplog):
        from decimalai.crewai import _skill_hook

        use_router(_Router())
        with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
            _skill_hook(_context(agent=SimpleNamespace(role="Support", skills=None)))
        assert _warnings(caplog, TWO_CATALOGS) == []


# ── layer 2: real CrewAI ────────────────────────────────────


def _llm(seen: List[List[Dict[str, Any]]]):
    from crewai.llms.base_llm import BaseLLM

    class _CapturingLLM(BaseLLM):
        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None, **kw):
            seen.append([dict(m) for m in messages])
            return "Thought: I know it\nFinal Answer: 23.5%"

        async def acall(self, messages, tools=None, callbacks=None, available_functions=None,
                        from_task=None, from_agent=None, response_model=None, **kw):
            return self.call(messages)

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return True

        def get_context_window_size(self) -> int:
            return 8192

    return _CapturingLLM(model="capturing-stub")


def _skills_dir(root: Path) -> Path:
    """A CrewAI skill search path holding one SKILL.md."""
    skill = root / "skills" / NATIVE_NAME
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {NATIVE_NAME}\ndescription: When returns close.\n---\n\n"
        "Returns close 30 days after delivery.\n",
        encoding="utf-8",
    )
    return root / "skills"


INLINE_SKILL = (
    "---\nname: tone-guide\ndescription: How to sound.\n---\n\nBe terse and kind.\n"
)


def _agent(seen, **kw):
    from crewai import Agent

    return Agent(
        role="Support", goal="Resolve the ticket", backstory="You are terse.",
        llm=_llm(seen), verbose=False, **kw,
    )


def _crew(agent, **kw):
    from crewai import Crew, Task

    task = Task(description=QUESTION, expected_output="the fee", agent=agent)
    return Crew(agents=[agent], tasks=[task], verbose=False, **kw)


@needs_crewai
class TestNativeSkillsAreNamed:
    """CrewAI's own skills, configured the three ways CrewAI documents, with the
    DecimalAI loader on as well: the router stays out of that agent's calls."""

    @pytest.fixture(autouse=True)
    def _loader(self):
        from crewai.hooks import unregister_before_llm_call_hook

        import decimalai.crewai as dc

        dc._tracing_installed = True  # no global tracer provider from a unit test
        dc.instrument(agent_name="support", enable_skill_loader=True)
        yield
        unregister_before_llm_call_hook(dc._skill_hook)

    def test_agent_skills_turn_the_router_off_and_say_so_once(self, tmp_path, use_router, caplog):
        router = use_router(_Router())
        seen: List[List[Dict[str, Any]]] = []
        agent = _agent(seen, skills=[_skills_dir(tmp_path)])
        with caplog.at_level(logging.INFO, logger="decimalai.crewai"):
            _crew(agent).kickoff()
            _crew(agent).kickoff()

        assert router.calls == [], "the routing call was made for an agent it cannot serve"
        found = _infos(caplog, ROUTER_OFF)
        assert len(found) == 1, caplog.text
        assert NATIVE_NAME in found[0]
        assert "load_skill" in found[0], "the native load_skill tool goes unnamed"
        assert _warnings(caplog, TWO_CATALOGS) == []
        # One catalog reaches the model: CrewAI's.
        request = "\n".join(str(m["content"]) for m in seen[0])
        assert SENTINEL not in request
        assert "<available_skills>" in request and NATIVE_NAME in request

    def test_crew_level_skills_are_detected_too(self, tmp_path, use_router, caplog):
        """``Crew(skills=...)`` reaches each agent at kickoff
        (``crews/utils.py::setup_agents`` → ``Agent.set_skills``)."""
        router = use_router(_Router())
        seen: List[List[Dict[str, Any]]] = []
        with caplog.at_level(logging.INFO, logger="decimalai.crewai"):
            _crew(_agent(seen), skills=[_skills_dir(tmp_path)]).kickoff()
        found = _infos(caplog, ROUTER_OFF)
        assert len(found) == 1 and NATIVE_NAME in found[0], caplog.text
        assert router.calls == []
        assert not any(SENTINEL in str(m["content"]) for m in seen[0])

    def test_an_inline_activated_skill_is_detected_as_always_on(self, use_router, caplog):
        """An inline SKILL.md string is activated on load: CrewAI renders its
        body in an always-on ``<skills>`` block and registers no tool."""
        use_router(_Router())
        seen: List[List[Dict[str, Any]]] = []
        with caplog.at_level(logging.INFO, logger="decimalai.crewai"):
            _crew(_agent(seen, skills=[INLINE_SKILL])).kickoff()
        found = _infos(caplog, ROUTER_OFF)
        assert len(found) == 1, caplog.text
        assert "tone-guide" in found[0]
        assert "always-on" in found[0]
        assert "load_skill" not in found[0], "no native tool exists to name"
        assert not any(SENTINEL in str(m["content"]) for m in seen[0])

    def test_an_agent_without_skills_is_still_routed_beside_one_with(
        self, tmp_path, use_router, caplog,
    ):
        """The router is turned off per agent, not for the process."""
        router = use_router(_OverlappingRouter())
        native_seen: List[List[Dict[str, Any]]] = []
        plain_seen: List[List[Dict[str, Any]]] = []
        with caplog.at_level(logging.INFO, logger="decimalai.crewai"):
            _crew(_agent(native_seen, skills=[_skills_dir(tmp_path)])).kickoff()
            _crew(_agent(plain_seen)).kickoff()
        assert len(router.calls) == 1
        assert not any(SENTINEL in str(m["content"]) for m in native_seen[0])
        assert any(SENTINEL in str(m["content"]) for m in plain_seen[0])

    def test_no_native_skills_no_warning(self, use_router, caplog):
        use_router(_Router())
        seen: List[List[Dict[str, Any]]] = []
        with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
            _crew(_agent(seen)).kickoff()
        assert _warnings(caplog, TWO_CATALOGS) == []
        assert seen and any(SENTINEL in str(m["content"]) for m in seen[0])

    def test_nothing_routed_means_one_catalog_and_no_warning(
        self, tmp_path, use_router, caplog,
    ):
        """The claim is "two catalogs are in this call". When DecimalAI routed
        nothing, CrewAI's is the only one."""
        use_router(_EmptyRouter())
        seen: List[List[Dict[str, Any]]] = []
        with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
            _crew(_agent(seen, skills=[_skills_dir(tmp_path)])).kickoff()
        assert _warnings(caplog, TWO_CATALOGS) == []


@needs_crewai
class TestTheRoutingCallAndTheEventLoop:
    """Where CrewAI runs this hook, measured: a router that sleeps, and a
    ticker coroutine on the caller's loop that counts while it does."""

    @pytest.fixture(autouse=True)
    def _loader(self):
        from crewai.hooks import unregister_before_llm_call_hook

        import decimalai.crewai as dc

        dc._tracing_installed = True
        dc.instrument(agent_name="support", enable_skill_loader=True)
        yield
        unregister_before_llm_call_hook(dc._skill_hook)

    @staticmethod
    def _run_with_ticker(start, ticks: List[int]) -> None:
        async def _main() -> None:
            done = asyncio.Event()

            async def _ticker() -> None:
                while not done.is_set():
                    ticks.append(1)
                    await asyncio.sleep(0.01)

            ticker = asyncio.create_task(_ticker())
            try:
                await start()
            finally:
                done.set()
                await ticker

        asyncio.run(_main())

    @pytest.mark.parametrize("entry", [
        "Crew.kickoff_async", "Crew.akickoff", "Agent.kickoff_async", "Agent.akickoff",
    ])
    def test_the_default_executor_routes_off_the_loop(self, entry, use_router, caplog):
        ticks: List[int] = []
        router = use_router(_Router(delay_s=0.3, ticks=ticks))
        seen: List[List[Dict[str, Any]]] = []
        agent = _agent(seen)
        crew = _crew(agent)
        start = {
            "Crew.kickoff_async": lambda: crew.kickoff_async(),
            "Crew.akickoff": lambda: crew.akickoff(),
            "Agent.kickoff_async": lambda: agent.kickoff_async(QUESTION),
            "Agent.akickoff": lambda: agent.akickoff(QUESTION),
        }[entry]

        with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
            self._run_with_ticker(start, ticks)

        assert router.calls, "the hook never routed"
        for call in router.calls:
            assert call["on_loop"] is False, f"{entry} ran the routing call on the loop"
            assert call["ticks_during"] >= 3, (
                f"{entry}: the caller's event loop stood still while skills were "
                f"routed ({call['ticks_during']} ticks in 0.3 s)"
            )
        assert seen and any(SENTINEL in str(m["content"]) for m in seen[0])
        assert _warnings(caplog, LOOP_BLOCKED) == []

    def test_the_deprecated_executors_akickoff_blocks_the_loop_and_says_so(
        self, use_router, caplog,
    ):
        """``Agent(executor_class=CrewAgentExecutor)`` + ``Crew.akickoff()``:
        ``CrewAgentExecutor._ainvoke_loop`` awaits ``aget_llm_response``, which
        runs the hooks inline on the loop thread. The ticker's zero is the
        evidence for the warning's claim — if a CrewAI release moves this hook
        off the loop, this assertion is what says the warning went stale."""
        from crewai.agents.crew_agent_executor import CrewAgentExecutor

        ticks: List[int] = []
        router = use_router(_Router(delay_s=0.3, ticks=ticks))
        seen: List[List[Dict[str, Any]]] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            agent = _agent(seen, executor_class=CrewAgentExecutor)
            crew = _crew(agent)
            with caplog.at_level(logging.WARNING, logger="decimalai.crewai"):
                self._run_with_ticker(lambda: crew.akickoff(), ticks)

        assert router.calls and router.calls[0]["on_loop"] is True
        assert router.calls[0]["ticks_during"] == 0
        found = _warnings(caplog, LOOP_BLOCKED)
        assert len(found) == 1, caplog.text
        assert "CrewAgentExecutor" in found[0]
        assert seen and any(SENTINEL in str(m["content"]) for m in seen[0]), (
            "the warning cost the delivery"
        )
