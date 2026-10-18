"""Pydantic AI harness native-mode scenarios (child side; no assertions).

The documented native mode: ``decimalai.init(otel=True)``, ``Agent.instrument_all()``
(Pydantic AI's own spans, which cover any model, the scripted one included) and
``decimalai.pydantic_ai.instrument()`` (one trace per agent run), with
``decimalai.pydantic_ai_harness.skills(project)`` in a ``LocalWorkspace(project)`` beside
the harness's ``FileSystem`` and ``Shell``. The model is a ``FunctionModel`` whose steps
use only what the harness told it (the ``Skill directory`` line of a loaded skill).

``column`` is the native column's scenario (``column.py`` describes the shape).
``checks`` builds agents with the skill router's loader installed and a skill to
route, and returns what each model request was shown.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from tests.native import column as col
from tests.native.column import RESOURCE, SCRIPT, SCRIPT_ARGS, TARGET, Run, Session

DISTRIBUTIONS = ["pydantic-ai-slim", "pydantic-ai-harness"]
USER = "refund for $100 after 45 days?"


def _session(*, skill_loader: bool = False) -> Session:
    from pydantic_ai import Agent

    import decimalai.pydantic_ai

    s = Session("pydantic-ai-harness", distributions=DISTRIBUTIONS, otel=True)
    Agent.instrument_all()
    decimalai.pydantic_ai.instrument(enable_skill_loader=skill_loader)
    return s


def _export_spans() -> None:
    """Export the run's spans now. The exporter turns them into the run's trace, and
    ``decimalai.flush()`` drains only the sender (and CrewAI's pipeline): the generic
    OpenTelemetry batch would otherwise wait out its schedule."""
    from opentelemetry import trace

    flush = getattr(trace.get_tracer_provider(), "force_flush", None)
    if callable(flush):
        flush(timeout_millis=5000)


def _returns(messages) -> List[Any]:
    """The tool results (and retry prompts) of the newest request: what the model got."""
    from pydantic_ai.messages import ModelRequest, RetryPromptPart, ToolReturnPart

    for m in reversed(messages):
        if isinstance(m, ModelRequest):
            return [p for p in m.parts if isinstance(p, (ToolReturnPart, RetryPromptPart))]
    return []


class _Script:
    """A FunctionModel following a list of steps, one list across runs."""

    def __init__(self, steps: List[Callable[[Any], Any]]) -> None:
        self.steps = steps
        self.i = 0
        self.run: Optional[Run] = None
        self.dir = ""
        self.instructions: List[str] = []     # what each request was instructed
        self.tools: List[List[str]] = []      # the tools each request offered

    def __call__(self, messages, info):
        from pydantic_ai.messages import ModelResponse, TextPart

        self.instructions.append(info.instructions or "")
        self.tools.append(sorted(t.name for t in info.function_tools))
        for part in _returns(messages):
            content = part.content
            text = content.get("instructions") if isinstance(content, dict) else content
            if self.run is not None:
                self.run.saw(text)
            found = re.search(r"Skill directory: `([^`]+)`", text if isinstance(text, str) else "")
            if found:
                self.dir = found.group(1)
        step = self.steps[self.i] if self.i < len(self.steps) else (lambda: None)
        self.i += 1
        parts = step()
        return ModelResponse(parts=parts or [TextPart("done")])

    # steps
    def load(self):
        from pydantic_ai.messages import ToolCallPart

        if self.run is not None:
            self.run.loaded_target()
        return [ToolCallPart("load_capability", {"id": TARGET}, tool_call_id=f"c{self.i}")]

    def read_resource(self):
        from pydantic_ai.messages import ToolCallPart

        return [ToolCallPart("read_file", {"path": f"{self.dir}/{RESOURCE}"},
                             tool_call_id=f"c{self.i}")]

    def run_script(self):
        from pydantic_ai.messages import ToolCallPart

        cmd = " ".join(("python3", f"{self.dir}/{SCRIPT}", *SCRIPT_ARGS))
        return [ToolCallPart("run_command", {"command": cmd}, tool_call_id=f"c{self.i}")]

    def answer(self):
        return None


def _agent(script: _Script, project: str, *, observed: bool = True):
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai_harness import FileSystem, Shell, Skills

    import decimalai
    import decimalai.pydantic_ai_harness

    skills = (decimalai.pydantic_ai_harness.skills(project) if observed
              else Skills(str(decimalai.skills_dir(relative_to=project))))
    return Agent(FunctionModel(script, model_name="scripted-harness"), name="harness-agent",
                 capabilities=[LocalWorkspace(project), skills, FileSystem(), Shell()])


def column(ctx: Dict[str, Any]) -> Dict[str, Any]:
    import warnings

    s = _session()
    project = str(col.project().root)
    script = _Script([])
    # Run 1: load, load again, read the resource, run the script, answer.
    # Run 2 (after the update): load, answer.
    script.steps = [script.load, script.load, script.read_resource, script.run_script,
                    script.answer, script.load, script.answer]
    agent = _agent(script, project)

    def one_run(run: Run) -> None:
        script.run = run
        with warnings.catch_warnings():
            # The harness warns on every run that it ignores `allowed-tools`.
            warnings.simplefilter("ignore", UserWarning)
            agent.run_sync(USER)
        _export_spans()

    s.run(one_run)
    s.notes["v2"] = col.update_to_v2()
    s.run(one_run)
    s.notes["catalog"] = script.instructions[0][-700:] if script.instructions else ""
    s.notes["received"] = [r["received"] for r in s.runs]
    return s.result()


def checks(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The router's loader installed, with a skill to route: an Agent with the harness's
    Skills (made by the pointer helper), one with a plain ``Skills``, and one without."""
    import logging
    import warnings

    from pydantic_ai import Agent
    from pydantic_ai.models.function import FunctionModel

    from tests.conformance.isolation import CONFORMANCE_SKILLS

    logs: List[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logs.append(f"{record.levelname} {record.name}: {record.getMessage()}")

    logging.getLogger("decimalai").addHandler(_Capture(level=logging.INFO))
    logging.getLogger("decimalai").setLevel(logging.INFO)

    s = _session(skill_loader=True)
    s.probe.skills = [dict(sk) for sk in CONFORMANCE_SKILLS]
    project = str(col.project().root)
    out: Dict[str, Any] = {}

    def one(label: str, build) -> None:
        script = _Script([])
        script.steps = [script.answer]
        agent = build(script)

        def capture(run: Run) -> None:
            script.run = run
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                agent.run_sync(USER)
            _export_spans()

        s.run(capture)
        out[label] = {"instructions": script.instructions[0] if script.instructions else "",
                      "tools": script.tools[0] if script.tools else []}

    one("observed", lambda script: _agent(script, project))
    one("plain_skills", lambda script: _agent(script, project, observed=False))
    one("no_skills", lambda script: Agent(FunctionModel(script, model_name="scripted-harness"),
                                          name="plain-agent"))
    result = s.result()
    out["traces"] = [r["traces"] for r in result["runs"]]
    out["errors"] = [r["error"] for r in result["runs"]]
    out["logs"] = logs[:60]
    return out


# ── the live check (tests/native/test_native_live.py) ────────────────────────


def _record_messages(run: Run, messages: List[Any]) -> None:
    """Every tool call the model made and everything it was handed, from the run's own
    message history."""
    from pydantic_ai.messages import (ModelRequest, ModelResponse, RetryPromptPart,
                                      ToolCallPart, ToolReturnPart)

    instructed = False
    for m in messages:
        if isinstance(m, ModelRequest):
            if not instructed and getattr(m, "instructions", None):
                instructed = True
                run.saw(m.instructions)
            for part in m.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    run.saw(part.model_response_str() if isinstance(part, ToolReturnPart)
                            else part.model_response())
        elif isinstance(m, ModelResponse):
            for part in m.parts:
                if isinstance(part, ToolCallPart):
                    try:
                        args = part.args_as_dict()
                    except Exception:  # noqa: BLE001 - malformed JSON from the model: keep it raw
                        args = part.args
                    run.called(part.tool_name, args)


def live(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Each trial: a fresh harness agent built as the column builds it (the pointer
    helper's ``Skills`` in a ``LocalWorkspace`` beside ``FileSystem`` and ``Shell``) with a
    REAL OpenAI model (``openai:<id>``), asked the live question; every tool call it made,
    what it was given and its answer (see ``tests/native/live.py``)."""
    import warnings

    from pydantic_ai import Agent, capture_run_messages
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai_harness import FileSystem, Shell

    import decimalai.pydantic_ai_harness
    from tests.native import live as lv

    s = _session()
    project = str(col.project().root)

    def trial(run: Run) -> None:
        agent = Agent(f"openai:{lv.openai_model()}", name="harness-agent",
                      instructions=lv.INSTRUCTIONS,
                      capabilities=[LocalWorkspace(project),
                                    decimalai.pydantic_ai_harness.skills(project),
                                    FileSystem(), Shell()])
        with capture_run_messages() as messages:
            try:
                with warnings.catch_warnings():
                    # The harness warns on every run that it ignores `allowed-tools`.
                    warnings.simplefilter("ignore", UserWarning)
                    result = agent.run_sync(lv.QUESTION)
                run.answered(result.output)
            finally:
                _record_messages(run, messages)
                _export_spans()

    for _ in range(lv.trials(ctx)):
        s.run(trial)
    s.notes["model"] = lv.openai_model()
    return s.result()
