"""Strands Agents in native mode: the scenarios (child side; see ``_runner.py``).

Each function runs in a child process whose working directory is the project
``native_project`` wrote, drives a real ``strands.Agent`` with the scripted model the
native-loader suite uses (``stubs/strands_model.py``), with ``decimalai.strands.skills()``
as its plugin and Strands' own ``file_editor`` and ``shell`` for reading a skill's files and
running its script, and returns JSON. Nothing here asserts.

The documented native-mode setup, which every scenario below follows:
``decimalai.init(otel=True, agent_name=...)`` (Strands' own spans carry the run), and
``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental,gen_ai_span_attributes_only``
set before Strands builds its tracer, so tool arguments and results land on the spans as
attributes (Strands' default puts them in span events, which the exporter does not read).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any, Dict, List, Set

from tests.native import column as col

AGENT = "native-strands"
DISTRIBUTIONS = ["strands-agents"]
CONTENT_ON_SPANS = "gen_ai_latest_experimental,gen_ai_span_attributes_only"


def _model_class():
    from tests.native_loaders.stubs.strands_model import ScriptedModel

    class Model(ScriptedModel):
        # Strands names a model call's span by ``model.config["model_id"]``, which every
        # provider model has and the stub does not; without it no span is an LLM call.
        config = {"model_id": "scripted-offline"}

    return Model


def _flush_spans() -> None:
    """Export the run's spans now. ``decimalai.flush()`` drains the sender but not the
    OpenTelemetry batch processor ``init(otel=True)`` installed (see the report), so a
    short-lived process must do this itself or its last run never becomes a trace."""
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


class _Script:
    """The model's moves, built from what the tools answered, as a model would."""

    def __init__(self) -> None:
        self.folder = ""

    def load(self, messages) -> tuple:
        return ("tool", "skills", {"skill_name": col.TARGET})

    def _located(self, messages) -> str:
        from tests.native_loaders.stubs.strands_model import last_tool_result_text

        if not self.folder:
            m = re.search(r"(?m)^Location: (\S+)/SKILL\.md$", last_tool_result_text(messages))
            self.folder = m.group(1) if m else ""
        return self.folder

    def read(self, messages) -> tuple:
        folder = self._located(messages)
        if not folder:
            return ("text", f"could not locate {col.TARGET}")
        return ("tool", "file_editor", {"command": "view", "path": f"{folder}/{col.RESOURCE}"})

    def run(self, messages) -> tuple:
        command = " ".join((sys.executable, f"{self.folder}/{col.SCRIPT}", *col.SCRIPT_ARGS))
        return ("tool", "shell", {"command": command})

    @staticmethod
    def answer(messages) -> tuple:
        return ("text", "done")


def _new_tool_results(model: Any, seen: Set[str]) -> List[str]:
    """Tool results the model was sent that no earlier run already sent it. Strands keeps the
    conversation on the agent, so run 2's requests still carry run 1's results."""
    out: List[str] = []
    for request in model.seen:
        for message in request["messages"]:
            for block in message.get("content", []):
                result = block.get("toolResult")
                if not result or result.get("toolUseId") in seen:
                    continue
                seen.add(result.get("toolUseId"))
                out.append("\n".join(str(c.get("text", c.get("json", "")))
                                     for c in result.get("content", [])))
    return out


def _agent(script_steps, plugin):
    from strands import Agent
    from strands.vended_tools import file_editor, shell

    model = _model_class()(script_steps)
    agent = Agent(model=model, name=AGENT, system_prompt="Answer refund questions.",
                  tools=[file_editor, shell], plugins=[plugin], callback_handler=None)
    return model, agent


def column(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The native column (``column.py``): list, load twice, read, run; update; load again."""
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = CONTENT_ON_SPANS
    s = col.Session("strands-agents", distributions=DISTRIBUTIONS, otel=True,
                    agent_name=AGENT)
    from decimalai.strands import skills

    sc = _Script()
    # One script for both runs: the model takes a step per call, across runs.
    model, agent = _agent([sc.load, sc.load, sc.read, sc.run, sc.answer,
                           sc.load, sc.answer], skills())
    seen: Set[str] = set()

    def run(loads: int):
        def fn(r: col.Run) -> None:
            first = len(model.seen)
            try:
                agent("refund for $100 after 45 days?")
            finally:
                _flush_spans()
            if len(model.seen) > first:
                r.saw(model.seen[first]["system_prompt"] or "")
            for text in _new_tool_results(model, seen):
                r.saw(text)
            r.loaded_target(loads)
        return fn

    s.run(run(2))
    s.notes["v2"] = col.update_to_v2()
    s.run(run(1))
    s.notes["folder"] = sc.folder
    return s.result()


def listing(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """``skills()`` as documented: what it lists, and what the model is offered."""
    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = CONTENT_ON_SPANS
    s = col.Session("strands-agents", distributions=DISTRIBUTIONS, otel=True,
                    agent_name=AGENT)
    from strands import AgentSkills

    from decimalai.strands import skills

    plugin = skills()
    sc = _Script()
    model, agent = _agent([sc.answer], plugin)
    out: Dict[str, Any] = {"is_agent_skills": isinstance(plugin, AgentSkills),
                           "plugin_name": plugin.name,
                           "available": sorted(x.name for x in plugin.get_available_skills(agent))}

    def fn(r: col.Run) -> None:
        try:
            agent("hello")
        finally:
            _flush_spans()
        out["system_prompt"] = model.seen[0]["system_prompt"] if model.seen else None

    s.run(fn)
    result = s.result()
    result["observed"] = out
    return result


def user_trace(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """A load inside the user's own ``decimalai.start_trace()``, with no OTel exporter."""
    s = col.Session("strands-agents", distributions=DISTRIBUTIONS)
    import decimalai

    from decimalai.strands import skills

    sc = _Script()
    model, agent = _agent([sc.load, sc.answer], skills())
    seen: Set[str] = set()

    def fn(r: col.Run) -> None:
        with decimalai.start_trace(agent_name="users-own-trace"):
            agent("refund?")
        for text in _new_tool_results(model, seen):
            r.saw(text)
        r.loaded_target()

    s.run(fn)
    return s.result()


# ── the live check (``test_native_live.py``): a REAL model, the same plugin ─────────────


def _record_messages(r: col.Run, messages: List[Dict[str, Any]]) -> None:
    """Every tool call the model made (its ``toolUse`` blocks) and every tool result it was
    handed, from the agent's own conversation, in order."""
    for message in messages:
        for block in message.get("content") or []:
            use = block.get("toolUse")
            if use:
                r.called(use.get("name"), use.get("input"))
            result = block.get("toolResult")
            if result:
                r.saw("\n".join(str(c.get("text", c.get("json", "")))
                                for c in result.get("content") or []))


def _live_trial(r: col.Run) -> None:
    """One invocation of a fresh agent: Strands' own ``AgentSkills`` plugin over the
    installed skills (``decimalai.strands.skills()``), its ``file_editor`` and ``shell``
    as documented, and a real OpenAI model."""
    from strands import Agent
    from strands.models.openai import OpenAIModel
    from strands.vended_tools import file_editor, shell

    from decimalai.strands import skills
    from tests.native import live

    agent = Agent(model=OpenAIModel(model_id=live.openai_model()), name=AGENT,
                  system_prompt=live.INSTRUCTIONS, tools=[file_editor, shell],
                  plugins=[skills()], callback_handler=None)
    result = None
    try:
        result = agent(live.QUESTION)
    finally:
        _flush_spans()
        r.saw(agent.system_prompt or "")
        _record_messages(r, agent.messages)
    r.answered(str(result).strip() if result is not None else None)


def live(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The live native check (``tests/native/live.py``): ``live.trials(ctx)`` independent
    trials, each a fresh agent with a real model, set up as :func:`column` is."""
    from tests.native import live as live_check

    os.environ["OTEL_SEMCONV_STABILITY_OPT_IN"] = CONTENT_ON_SPANS
    s = col.Session("strands-agents", distributions=DISTRIBUTIONS + ["openai"], otel=True,
                    agent_name=AGENT)
    for _ in range(live_check.trials(ctx)):
        s.run(_live_trial)
    return s.result()
