"""Microsoft Agent Framework in native mode: the scenarios (child side; see ``_runner.py``).

Each function runs in a child process whose working directory is the project
``native_project`` wrote, drives the real ``agent_framework.Agent`` with the scripted chat
client the native-loader suite uses (``stubs/maf_client.py``: the framework's own
function-invocation loop, approval flow and middleware all run), through
``decimalai.agent_framework.skills()``, and returns JSON. Nothing here asserts.

The documented native-mode setup, which every scenario below follows:
``decimalai.init(otel=True, agent_name=...)`` (Agent Framework's own spans carry the run),
and Agent Framework's ``enable_sensitive_telemetry()`` where a scenario needs tool
arguments and results on the trace (Agent Framework leaves them off its spans otherwise).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from tests.native import column as col

AGENT = "native-maf"
DISTRIBUTIONS = ["agent-framework-core"]


def _load(name: str = col.TARGET):
    return ("calls", [("load_skill", {"skill_name": name})])


READ = ("calls", [("read_skill_resource", {"skill_name": col.TARGET,
                                            "resource_name": col.RESOURCE})])
RUN = ("calls", [("run_skill_script", {"skill_name": col.TARGET, "script_name": col.SCRIPT,
                                        "args": list(col.SCRIPT_ARGS)})])


def _flush_spans() -> None:
    """Export the run's spans now. ``decimalai.flush()`` drains the sender but not the
    OpenTelemetry batch processor ``init(otel=True)`` installed (see the report), so a
    short-lived process must do this itself or its last run never becomes a trace."""
    from opentelemetry import trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


def _results_since(client: Any, cursor: int) -> List[str]:
    """Every function result the model was sent in requests ``cursor`` and later, once."""
    from tests.native_loaders.stubs.maf_client import function_results

    seen: Dict[str, str] = {}
    for request in client.requests[cursor:]:
        for call_id, result in function_results(request["messages"]):
            seen.setdefault(call_id, result if isinstance(result, str) else _text(result))
    return list(seen.values())


def _text(result: Any) -> str:
    if isinstance(result, (list, tuple)):
        return "".join(getattr(c, "text", None) or str(c) for c in result)
    return str(result)


def column(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The native column (``column.py``): list, load twice, read, run; update; load again."""
    s = col.Session("agent-framework", distributions=DISTRIBUTIONS, otel=True,
                    agent_name=AGENT)
    from agent_framework import Agent
    from agent_framework.observability import enable_sensitive_telemetry

    from decimalai.agent_framework import skills
    from tests.native_loaders.stubs.maf_client import ScriptedChatClient

    enable_sensitive_telemetry()
    # One script for both runs: the client pops a step per model call.
    client = ScriptedChatClient([_load(), _load(), READ, RUN, ("text", "done"),
                                 _load(), ("text", "done")])
    agent = Agent(client=client, name=AGENT, instructions="Answer refund questions.",
                  context_providers=[skills(auto_approve_scripts=True)])

    def run(loads: int):
        def fn(r: col.Run) -> None:
            cursor = len(client.requests)
            try:
                asyncio.run(agent.run("refund for $100 after 45 days?"))
            finally:
                _flush_spans()
            first = client.requests[cursor] if len(client.requests) > cursor else None
            if first is not None:
                r.saw(str(first["options"].get("instructions", "")))
            for text in _results_since(client, cursor):
                r.saw(text)
            r.loaded_target(loads)
        return fn

    s.run(run(2))
    s.notes["v2"] = col.update_to_v2()
    s.run(run(1))
    s.notes["requests"] = len(client.requests)
    return s.result()


def default_approval(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """``skills()`` as documented, no flags: what the model is offered, which tools run
    unasked, and whether a script waits for approval."""
    s = col.Session("agent-framework", distributions=DISTRIBUTIONS, otel=True,
                    agent_name=AGENT)
    from agent_framework import Agent

    from decimalai.agent_framework import skills
    from tests.native_loaders.stubs.maf_client import ScriptedChatClient

    client = ScriptedChatClient([_load(), READ, RUN, ("text", "done")])
    agent = Agent(client=client, name=AGENT, instructions="Answer refund questions.",
                  context_providers=[skills()])
    out: Dict[str, Any] = {}

    def fn(r: col.Run) -> None:
        try:
            response = asyncio.run(agent.run("refund for $100 after 45 days?"))
        finally:
            _flush_spans()
        out["instructions"] = str(client.requests[0]["options"].get("instructions", ""))
        out["tools_offered"] = sorted(
            getattr(t, "name", str(t)) for t in client.requests[0]["options"].get("tools") or [])
        out["results"] = _results_since(client, 0)
        out["approval_requests"] = [
            getattr(getattr(req, "function_call", None), "name", None)
            for req in (getattr(response, "user_input_requests", None) or [])]
        out["model_calls"] = len(client.requests)

    s.run(fn)
    result = s.result()
    result["observed"] = out
    return result


def user_trace(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """A load inside the user's own ``decimalai.start_trace()``, with no OTel exporter."""
    s = col.Session("agent-framework", distributions=DISTRIBUTIONS)
    import decimalai
    from agent_framework import Agent

    from decimalai.agent_framework import skills
    from tests.native_loaders.stubs.maf_client import ScriptedChatClient

    client = ScriptedChatClient([_load(), ("text", "done")])
    agent = Agent(client=client, name=AGENT, instructions="Answer refund questions.",
                  context_providers=[skills()])

    def fn(r: col.Run) -> None:
        with decimalai.start_trace(agent_name="users-own-trace"):
            asyncio.run(agent.run("refund?"))
        for text in _results_since(client, 0):
            r.saw(text)
        r.loaded_target()

    s.run(fn)
    return s.result()


# ── the live check (tests/native/test_native_live.py) ────────────────────────


def _observer(run: col.Run) -> Any:
    """Chat middleware recording, per model call, what the model was instructed (first
    call) and every function call in its response, so a run that stops early still
    reports what it did."""
    import json

    from agent_framework import chat_middleware

    state = {"instructed": False}

    @chat_middleware
    async def observe(context: Any, call_next: Any) -> None:
        if not state["instructed"]:
            state["instructed"] = True
            run.saw(str((context.options or {}).get("instructions", "")))
        await call_next()
        for message in getattr(context.result, "messages", None) or []:
            for c in message.contents:
                if c.type == "function_call":
                    args = c.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            pass
                    run.called(c.name, args)

    return observe


def live(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Each trial: a fresh ``Agent`` with ``skills()`` as the column builds it and a REAL
    OpenAI model (``OpenAIChatClient``), asked the live question; every tool call it made,
    what it was given and its answer (see ``tests/native/live.py``)."""
    s = col.Session("agent-framework", distributions=DISTRIBUTIONS + ["agent-framework-openai"],
                    otel=True, agent_name=AGENT)
    from agent_framework import Agent
    from agent_framework.observability import enable_sensitive_telemetry
    from agent_framework.openai import OpenAIChatClient

    from decimalai.agent_framework import skills
    from tests.native import live as lv

    enable_sensitive_telemetry()

    def trial(r: col.Run) -> None:
        agent = Agent(client=OpenAIChatClient(lv.openai_model()), name=AGENT,
                      instructions=lv.INSTRUCTIONS,
                      context_providers=[skills(auto_approve_scripts=True)],
                      middleware=[_observer(r)])
        try:
            response = asyncio.run(agent.run(lv.QUESTION))
        finally:
            _flush_spans()
        for message in response.messages:
            for c in message.contents:
                if c.type == "function_result":
                    r.saw(c.result if c.exception is None else f"EXCEPTION: {c.exception}")
        r.answered(response.text)

    for _ in range(lv.trials(ctx)):
        s.run(trial)
    s.notes["model"] = lv.openai_model()
    return s.result()
