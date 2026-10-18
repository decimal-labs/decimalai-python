"""crewai native mode, child side: ``Agent(skills=decimalai.crewai.skills(),
tools=decimalai.crewai.skill_tools())`` run as a ``Crew`` with a scripted ``BaseLLM``,
and ``decimalai.init(crewai=True)`` pointed at a probe. Runs in a child process
(``_runner.run_scenario``) with the project as its working directory, where the
runner's environment has turned CrewAI's telemetry off and given it its own HOME.
Observes only; the assertions are the parent's.

* :func:`column`: the native conformance column's scenario (see ``column.py``).
  CrewAI has no listing step the model calls: its catalog is in the system prompt.
* :func:`native`: the same two runs, plus what ``test_crewai_native.py`` grades beyond
  the column: the system prompt of every model call, a run whose agent carries an
  ACTIVATED (always-on) skill, a control agent with no skills of its own, and the
  ``decimalai.crewai`` log lines.
* :func:`live`: the live native check's trials (``tests/native/live.py``): the same
  loader with a real OpenAI model, every tool call read off CrewAI's event bus.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from tests.native.column import (
    RESOURCE,
    SCRIPT,
    SCRIPT_ARGS,
    TARGET,
    Run,
    Session,
    update_to_v2,
)

ROUTER_SKILL = {
    "name": "router-only-skill",
    "description": "A skill only DecimalAI's router offers.",
    "body": "# Router only\n\nSENTINEL-ROUTER-BODY-CREWAI-63e2: only an injected body has this.",
}
ROUTER_SENTINEL = "SENTINEL-ROUTER-BODY-CREWAI-63e2"


class _Logs(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append({"level": record.levelname, "logger": record.name,
                           "message": record.getMessage()})


def _watch() -> _Logs:
    handler = _Logs()
    for name in ("decimalai.crewai", "decimalai.native"):
        log = logging.getLogger(name)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
    return handler


def _agent(role: str, skills=None, tools=None):
    from crewai import Agent

    from tests.native_loaders.stubs.crewai_llm import ScriptedLLM

    llm = ScriptedLLM(model="scripted/fake")
    llm.script, llm.requests = [], []
    kwargs: Dict[str, Any] = {}
    if skills is not None:
        kwargs["skills"] = skills
    if tools is not None:
        kwargs["tools"] = tools
    return Agent(role=role, goal="Answer refund questions", backstory="b", llm=llm, **kwargs)


def _native_agent():
    import decimalai.crewai

    return _agent("Support", skills=decimalai.crewai.skills(),
                  tools=decimalai.crewai.skill_tools())


def _kickoff(agent, steps, run: Run, calls: List[Dict[str, Any]]) -> None:
    """One ``Crew.kickoff()`` of ``agent``, the model following ``steps``. Every tool
    result the model is handed goes to ``run.saw``."""
    from crewai import Crew, Task

    llm = agent.llm
    llm.script, llm.requests = list(steps), []
    Crew(agents=[agent], tasks=[Task(description="refund for $100 after 45 days?",
                                     expected_output="an answer", agent=agent)]).kickoff()
    # One task's messages persist across its model calls, so the last request holds
    # every tool result of the run, in order.
    last = llm.requests[-1]["messages"] if llm.requests else []
    for message in last:
        if message.get("role") == "tool":
            run.saw(str(message.get("content")))
    for request in llm.requests:
        calls.append({"agent": agent.role, "system": "\n".join(
            str(m.get("content", "")) for m in request["messages"]
            if m.get("role") in ("system", "developer"))})


def _run1_steps():
    from tests.native_loaders.stubs.crewai_llm import tool_call

    return [[tool_call("load_skill", {"skill_name": TARGET}, "c1")],
            [tool_call("load_skill", {"skill_name": TARGET}, "c2")],
            [tool_call("load_skill_resource", {"skill_name": TARGET, "file_path": RESOURCE}, "c3")],
            [tool_call("run_skill_script", {"skill_name": TARGET, "file_path": SCRIPT,
                                            "args": list(SCRIPT_ARGS)}, "c4")],
            "done"]


def _run2_steps():
    from tests.native_loaders.stubs.crewai_llm import tool_call

    return [[tool_call("load_skill", {"skill_name": TARGET}, "c5")], "done"]


def _two_runs(s: Session, calls: List[Dict[str, Any]]) -> None:
    agent = _native_agent()

    def run1(run: Run) -> None:
        _kickoff(agent, _run1_steps(), run, calls)
        run.loaded_target(2)

    def run2(run: Run) -> None:
        _kickoff(agent, _run2_steps(), run, calls)
        run.loaded_target()

    s.run(run1)
    s.notes["v2_expected"] = update_to_v2()
    s.run(run2)


def column(ctx: Dict[str, Any]) -> Dict[str, Any]:
    s = Session("crewai", distributions=["crewai"], crewai=True)
    _two_runs(s, [])
    return s.result()


def native(ctx: Dict[str, Any]) -> Dict[str, Any]:
    logs = _watch()
    s = Session("crewai", distributions=["crewai"], crewai=True)
    s.probe.skills = [dict(ROUTER_SKILL)]
    calls: List[Dict[str, Any]] = []
    _two_runs(s, calls)

    def always_on(run: Run) -> None:
        # An activated skill: CrewAI renders its body into every call's prompt and
        # announces it with SkillUsedEvent; no tool call delivers it.
        from crewai.skills.loader import activate_skill, discover_skills

        import decimalai.crewai

        (target,) = [k for k in discover_skills(decimalai.crewai.skills()[0]) if k.name == TARGET]
        activated = activate_skill(target)
        run.saw(activated.instructions)
        _kickoff(_agent("Desk", skills=[activated]), ["done"], run, calls)

    def control(run: Run) -> None:
        _kickoff(_agent("Plain"), ["done"], run, calls)

    s.run(always_on)
    s.run(control)
    out = s.result()
    out["calls"] = calls
    out["logs"] = logs.lines
    return out


# ── the live check (``test_native_live.py``): a REAL model, the same loader ──────────────


class _ToolEvents:
    """The tool calls CrewAI's own event bus reports during one kickoff.

    ``ToolUsageStartedEvent`` is emitted for every tool call the model makes, an unknown
    tool name included, before CrewAI runs it; ``ToolUsageFinishedEvent`` carries what the
    model was handed back. Handlers run on the bus's thread pool, so the bus is flushed and
    the events put back in emission order before they are read.
    """

    def __init__(self) -> None:
        import threading

        self.started: List[Any] = []
        self.finished: List[Any] = []
        self._lock = threading.Lock()

    def _on_started(self, source: Any, event: Any) -> None:
        with self._lock:
            self.started.append(event)

    def _on_finished(self, source: Any, event: Any) -> None:
        with self._lock:
            self.finished.append(event)

    def __enter__(self) -> "_ToolEvents":
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.tool_usage_events import (
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )

        crewai_event_bus.register_handler(ToolUsageStartedEvent, self._on_started)
        crewai_event_bus.register_handler(ToolUsageFinishedEvent, self._on_finished)
        return self

    def __exit__(self, *exc: Any) -> None:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.tool_usage_events import (
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )

        crewai_event_bus.flush()
        crewai_event_bus.off(ToolUsageStartedEvent, self._on_started)
        crewai_event_bus.off(ToolUsageFinishedEvent, self._on_finished)

    @staticmethod
    def _ordered(events: List[Any]) -> List[Any]:
        return sorted(events, key=lambda e: (getattr(e, "emission_sequence", None) or 0,
                                             getattr(e, "timestamp", None)))

    def into(self, run: Run) -> None:
        for event in self._ordered(self.started):
            run.called(event.tool_name, event.tool_args)
        for event in self._ordered(self.finished):
            run.saw(str(event.output))


def _live_trial(run: Run) -> None:
    """One kickoff of a fresh agent: CrewAI's own loader over the installed skills
    (``decimalai.crewai.skills()`` + ``skill_tools()``), a real OpenAI model, the live
    check's instructions as the agent's backstory and its question as the task."""
    from crewai import LLM, Agent, Crew, Task

    import decimalai.crewai
    from tests.native import live

    agent = Agent(role="Customer support agent", goal="Answer the customer's question",
                  backstory=live.INSTRUCTIONS, llm=LLM(model=f"openai/{live.openai_model()}"),
                  skills=decimalai.crewai.skills(), tools=decimalai.crewai.skill_tools())
    task = Task(description=live.QUESTION, expected_output="Your answer to the customer.",
                agent=agent)
    events = _ToolEvents()
    try:
        with events:
            out = Crew(agents=[agent], tasks=[task]).kickoff()
    finally:
        events.into(run)
    run.answered(getattr(out, "raw", None) if out is not None else None)


def live(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The live native check (``tests/native/live.py``): ``live.trials(ctx)`` independent
    trials, each a fresh agent with a real model."""
    from tests.native import live as live_check

    s = Session("crewai", distributions=["crewai", "openai"], crewai=True)
    for _ in range(live_check.trials(ctx)):
        s.run(_live_trial)
    return s.result()
