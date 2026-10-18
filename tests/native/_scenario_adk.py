"""google-adk native mode, child side: ADK's own ``SkillToolset`` over
``decimalai.adk.skills()``, a scripted ``BaseLlm``, ``decimalai.init(adk=True)``
pointed at a probe. Runs in a child process (``_runner.run_scenario``) with the
project as its working directory. Observes only; the assertions are the parent's.

* :func:`column`: the native conformance column's scenario (see ``column.py``).
* :func:`native`: the same two runs, plus what ``test_adk_native.py`` grades beyond
  the column: the system instruction of every model call (is the router out of the
  native agent's requests?), a control agent with no toolset (is the router on at
  all?), and the ``decimalai.adk`` log lines.
* :func:`live`: the live native check's trials (``tests/native/live.py``): the same
  toolset with a real OpenAI model through ``LiteLlm``.
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


# ── the live check (``test_native_live.py``): a REAL model, the same toolset ────────────


def _text_of(content: Any) -> str:
    parts = getattr(content, "parts", None) or []
    return "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", None))


def _live_trial(run: Run) -> None:
    """One invocation of a fresh agent on a fresh runner and session: ADK's own
    ``SkillToolset`` over ``decimalai.adk.skills()`` (as :func:`_agent` builds it) and a
    real OpenAI model through ADK's ``LiteLlm``. Every ``function_call`` event is a tool
    call the model made; every ``function_response`` event is what it was handed back."""
    from google.adk.agents import LlmAgent
    from google.adk.code_executors.unsafe_local_code_executor import (
        UnsafeLocalCodeExecutor,
    )
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.runners import InMemoryRunner
    from google.adk.tools.skill_toolset import SkillToolset
    from google.genai import types

    import decimalai.adk
    from tests.native import live

    agent = LlmAgent(name="support", model=LiteLlm(model=f"openai/{live.openai_model()}"),
                     instruction=live.INSTRUCTIONS,
                     tools=[SkillToolset(skills=decimalai.adk.skills(),
                                         code_executor=UnsafeLocalCodeExecutor())])
    answers: List[str] = []
    errors: List[str] = []

    async def go() -> None:
        runner = InMemoryRunner(agent=agent, app_name="native-live")
        session = await runner.session_service.create_session(app_name="native-live",
                                                               user_id="u")
        msg = types.Content(role="user", parts=[types.Part(text=live.QUESTION)])
        try:
            async for event in runner.run_async(user_id="u", session_id=session.id,
                                                new_message=msg):
                if event.partial:
                    continue
                for call in event.get_function_calls():
                    run.called(call.name, dict(call.args or {}))
                for response in event.get_function_responses():
                    run.saw(response.response)
                if event.error_code:
                    errors.append(f"{event.error_code}: {event.error_message}")
                elif event.is_final_response() and _text_of(event.content):
                    answers.append(_text_of(event.content))
        finally:
            await runner.close()

    asyncio.run(go())
    if errors and not answers:
        # ADK reported the model call's failure as an event, not an exception: surface it,
        # so a provider's refusal is graded as one and not as a run with no answer.
        raise RuntimeError("the model call failed: " + "; ".join(errors))
    run.answered("\n".join(answers) if answers else None)


def live(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The live native check (``tests/native/live.py``): ``live.trials(ctx)`` independent
    trials, each a fresh agent with a real model, set up as :func:`column` is."""
    from tests.native import live as live_check

    s = Session("google-adk", distributions=["google-adk", "litellm"], adk=True)
    for _ in range(live_check.trials(ctx)):
        s.run(_live_trial)
    return s.result()
