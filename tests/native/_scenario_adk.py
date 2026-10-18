"""google-adk native mode, child side: ADK's own ``SkillToolset`` over
``decimalai.adk.skills()``, a scripted ``BaseLlm``, ``decimalai.init(adk=True)``
pointed at a probe. Runs in a child process (``_runner.run_scenario``) with the
project as its working directory. Observes only; the assertions are the parent's.

* :func:`column`: the native conformance column's scenario (see ``column.py``).
* :func:`native`: the same two runs, plus what ``test_adk_native.py`` grades beyond
  the column: the system instruction of every model call (is the router out of the
  native agent's requests?), a control agent with no toolset (is the router on at
  all?), and the ``decimalai.adk`` log lines.
"""

from __future__ import annotations

import asyncio
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

#: What the probe's router serves. Only an injected menu or body can carry these.
ROUTER_SKILL = {
    "name": "router-only-skill",
    "description": "A skill only DecimalAI's router offers.",
    "body": "# Router only\n\nSENTINEL-ROUTER-BODY-ADK-41c7: only an injected body has this.",
}
ROUTER_SENTINEL = "SENTINEL-ROUTER-BODY-ADK-41c7"


class _Logs(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: List[Dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append({"level": record.levelname, "logger": record.name,
                           "message": record.getMessage()})


def _watch() -> _Logs:
    handler = _Logs()
    for name in ("decimalai.adk", "decimalai.native"):
        log = logging.getLogger(name)
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)
    return handler


def _agent(name: str, *, toolset: bool):
    from google.adk.agents import LlmAgent
    from google.adk.code_executors.unsafe_local_code_executor import (
        UnsafeLocalCodeExecutor,
    )
    from google.adk.tools.skill_toolset import SkillToolset

    import decimalai.adk
    from tests.native_loaders.stubs.adk_llm import ScriptedLlm

    tools = [SkillToolset(skills=decimalai.adk.skills(),
                          code_executor=UnsafeLocalCodeExecutor())] if toolset else []
    return LlmAgent(name=name, model=ScriptedLlm(model="scripted"),
                    instruction="Answer refund questions.", tools=tools)


def _invoke(agent, steps, run: Run, calls: List[Dict[str, Any]]) -> None:
    """One invocation of ``agent`` on a fresh runner and session, the model following
    ``steps``. Every tool result the model is handed goes to ``run.saw``."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    model = agent.model
    model._steps, model._responses, model._instructions = list(steps), [], []

    async def go() -> None:
        runner = InMemoryRunner(agent=agent, app_name="native")
        session = await runner.session_service.create_session(app_name="native", user_id="u")
        msg = types.Content(role="user", parts=[types.Part(text="refund for $100 after 45 days?")])
        async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=msg):
            pass
        await runner.close()

    asyncio.run(go())
    for i, results in enumerate(model._responses):
        for result in results:
            run.saw(result)
        calls.append({"agent": agent.name, "results": results,
                      "system": model._instructions[i] if i < len(model._instructions) else ""})


def _run1_steps():
    from tests.native_loaders.stubs.adk_llm import answer, call

    return [call("list_skills"),
            call("load_skill", skill_name=TARGET),
            call("load_skill", skill_name=TARGET),
            call("load_skill_resource", skill_name=TARGET, file_path=RESOURCE),
            call("run_skill_script", skill_name=TARGET, file_path=SCRIPT, args=list(SCRIPT_ARGS)),
            answer("done")]


def _run2_steps():
    from tests.native_loaders.stubs.adk_llm import answer, call

    return [call("load_skill", skill_name=TARGET), answer("done")]


def _two_runs(s: Session, calls: List[Dict[str, Any]]) -> None:
    agent = _agent("support", toolset=True)

    def run1(run: Run) -> None:
        _invoke(agent, _run1_steps(), run, calls)
        run.loaded_target(2)

    def run2(run: Run) -> None:
        _invoke(agent, _run2_steps(), run, calls)
        run.loaded_target()

    s.run(run1)
    s.notes["v2_expected"] = update_to_v2()
    s.run(run2)


def column(ctx: Dict[str, Any]) -> Dict[str, Any]:
    s = Session("google-adk", distributions=["google-adk"], adk=True)
    _two_runs(s, [])
    return s.result()


def native(ctx: Dict[str, Any]) -> Dict[str, Any]:
    logs = _watch()
    s = Session("google-adk", distributions=["google-adk"], adk=True)
    s.probe.skills = [dict(ROUTER_SKILL)]
    calls: List[Dict[str, Any]] = []
    _two_runs(s, calls)

    def control(run: Run) -> None:
        from tests.native_loaders.stubs.adk_llm import answer

        _invoke(_agent("plain", toolset=False), [answer("done")], run, calls)

    s.run(control)
    out = s.result()
    out["calls"] = calls
    out["logs"] = logs.lines
    return out
