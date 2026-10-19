"""OpenAI Agents SDK native mode, run in a child (``_runner.run_scenario``).

A ``SandboxAgent`` with the sandbox's own ``Skills`` capability, built by
``decimalai.openai_agents.sandbox_skills()``, in a local Unix sandbox, driven by the SDK's
``ScriptedModel``; DecimalAI's instrumentation (``decimalai.init(openai_agents=True)``)
sends each run's trace to a probe. Nothing here asserts: the parent grades what comes back.
Since 0.18 that init gives no agent skills of its own, so DecimalAI's skills reach an agent
only through ``load_agent()`` (the ``two_catalogs`` scenario).

The sandbox's ``load_skill`` returns only a status; the model reads the body by opening
``SKILL.md`` in the copy the load made, so a "load" here is ``load_skill`` and then
``cat SKILL.md``, and what the model received is every tool output it was handed.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from tests.native import column as col

DISTRIBUTIONS = ["openai-agents"]
QUESTION = "A customer wants a refund for $100 after 45 days."
FOLDER = f".agents/{col.TARGET}"


class _Logs(logging.Handler):
    """The adapter's own log lines, for the router-off and two-catalog notices."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: List[Dict[str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append({"level": record.levelname, "message": record.getMessage()})


def watch_logs() -> _Logs:
    handler = _Logs()
    logger = logging.getLogger("decimalai.openai_agents")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def manifest():
    """``Manifest()``, plus the one read-only grant this suite's fresh ``HOME`` needs.

    On macOS a sandboxed command may read its interpreter only when it is a system one or
    lives under ``$HOME/.<dir>``; the child's ``HOME`` is empty, so a uv- or pyenv-managed
    Python would lose its standard library (see ``loaders/openai_agents.py``).
    """
    from agents.sandbox import Manifest, SandboxPathGrant

    if sys.platform != "darwin":
        return Manifest()
    return Manifest(extra_path_grants=(
        SandboxPathGrant(path=str(Path(sys.base_prefix).resolve()), read_only=True),))


AGENT = "native-refunds"
INSTRUCTIONS = "Answer refund questions."


def agent(model: Any, *, instructions: Any = INSTRUCTIONS, **kw: Any) -> Any:
    """The documented native-mode agent: the shell, and the pointer helper's ``Skills``."""
    from agents.sandbox import SandboxAgent
    from agents.sandbox.capabilities import Shell

    from decimalai.openai_agents import sandbox_skills

    return SandboxAgent(name=AGENT, instructions=instructions,
                        model=model, capabilities=[Shell(), sandbox_skills()],
                        default_manifest=manifest(), **kw)


class Turns:
    """One scripted model, given each run's turns as the run starts, so run 2 runs the
    same agent object."""

    def __init__(self) -> None:
        from agents.testing import ScriptedModel

        self.model = ScriptedModel([])
        self.start = 0

    def script(self, *calls: Any) -> None:
        from agents.testing import assistant_message, function_call

        turns: List[Any] = [[function_call(name, args, call_id=f"c{i}")]
                            for i, (name, args) in enumerate(calls, 1)]
        turns.append([assistant_message("done")])
        self.start = len(self.model.calls)
        self.model.extend(turns)

    @property
    def calls(self) -> tuple:
        return self.model.calls[self.start:]


def load() -> tuple:
    return ("load_skill", {"skill_name": col.TARGET})


def cat(rel: str) -> tuple:
    return ("exec_command", {"cmd": f"cat {FOLDER}/{rel}"})


def run_script() -> tuple:
    return ("exec_command", {"cmd": " ".join(("python3", f"{FOLDER}/{col.SCRIPT}",
                                              *col.SCRIPT_ARGS))})


def drive(turns: Turns, the_agent: Any, *calls: tuple) -> Dict[str, Any]:
    """One ``Runner.run`` of ``the_agent``; what the model was given and offered."""
    from agents import RunConfig, Runner
    from agents.sandbox import SandboxRunConfig
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    turns.script(*calls)
    config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
    asyncio.run(Runner.run(the_agent, QUESTION, run_config=config, max_turns=len(calls) + 2))
    calls_seen = turns.calls
    outputs: List[str] = []
    if calls_seen:
        for item in calls_seen[-1].input:
            d = item if isinstance(item, dict) else item.model_dump()
            if d.get("type") == "function_call_output":
                outputs.append(str(d.get("output")))
    return {
        "system": [c.system_instructions or "" for c in calls_seen],
        "tools": [sorted(getattr(t, "name", "") for t in (c.tools or [])) for c in calls_seen],
        "outputs": outputs,
        "target_loads": sum(1 for name, args in calls if name == "load_skill"
                            and args.get("skill_name") == col.TARGET),
    }


def _into(run: col.Run, got: Dict[str, Any]) -> None:
    for text in got["system"][:1] + got["outputs"]:
        run.saw(text)
    run.loaded_target(got["target_loads"])


def _route_calls(session: col.Session) -> int:
    return sum(1 for r in session.probe.requests if r.path.startswith("/api/v1/skills/route"))


def column_runs(session: col.Session, first: List[tuple], second: List[tuple],
                **agent_kw: Any) -> Dict[str, Any]:
    turns = Turns()
    the_agent = agent(turns.model, **agent_kw)
    seen: Dict[str, Any] = {}

    def run1(run: col.Run) -> None:
        seen["run1"] = got = drive(turns, the_agent, *first)
        _into(run, got)

    def run2(run: col.Run) -> None:
        seen["run2"] = got = drive(turns, the_agent, *second)
        _into(run, got)

    session.run(run1)
    session.notes["v2"] = col.update_to_v2()
    session.run(run2)
    return seen


def column(ctx: dict) -> dict:
    """The native conformance column's scenario (see ``tests/native/column.py``)."""
    s = col.Session("openai-agents", distributions=DISTRIBUTIONS, openai_agents=True)
    column_runs(s, [load(), cat("SKILL.md"), load(), cat(col.RESOURCE), run_script()],
                [load(), cat("SKILL.md")])
    return s.result()


def native(ctx: dict) -> dict:
    """The column's runs with the router holding skills it would inject, so the router-off
    rule (D7) is visible: what the model was shown, which tools it had, the notices."""
    from tests.conformance.isolation import CONFORMANCE_SKILLS

    logs = watch_logs()
    s = col.Session("openai-agents", distributions=DISTRIBUTIONS, openai_agents=True)
    s.probe.skills = [dict(sk) for sk in CONFORMANCE_SKILLS]
    seen = column_runs(s, [load(), cat("SKILL.md"), load(), cat(col.RESOURCE), run_script()],
                       [load(), cat("SKILL.md")])
    s.notes.update({
        "router_skills": [sk["name"] for sk in CONFORMANCE_SKILLS],
        "route_calls": _route_calls(s),
        "system": [seen[k]["system"][0] if seen[k]["system"] else "" for k in ("run1", "run2")],
        "tools": [seen[k]["tools"][0] if seen[k]["tools"] else [] for k in ("run1", "run2")],
        "outputs": [seen[k]["outputs"] for k in ("run1", "run2")],
        "logs": logs.lines,
    })
    return s.result()


def two_catalogs(ctx: dict) -> dict:
    """The same agent also given DecimalAI's skills with ``load_agent()``: its
    instructions and its ``load_skill`` tool, beside the sandbox's own ``Skills``.
    DecimalAI's skills turn off for it (D7), so one catalog reaches the model."""
    from tests.conformance.isolation import CONFORMANCE_SKILLS

    from decimalai.openai_agents import load_agent

    logs = watch_logs()
    s = col.Session("openai-agents", distributions=DISTRIBUTIONS, openai_agents=True)
    s.probe.skills = [dict(sk) for sk in CONFORMANCE_SKILLS]
    s.probe.register_agent(AGENT, system_prompt=INSTRUCTIONS)
    support = load_agent(AGENT)
    turns = Turns()
    the_agent = agent(turns.model, instructions=support.instructions, tools=[*support.tools])
    seen: Dict[str, Any] = {}

    def once(run: col.Run) -> None:
        seen.setdefault("runs", []).append(drive(turns, the_agent, load(), cat("SKILL.md")))
        _into(run, seen["runs"][-1])

    s.run(once)
    s.run(once)
    s.notes.update({
        "router_skills": [sk["name"] for sk in CONFORMANCE_SKILLS],
        "route_calls": _route_calls(s),
        "system": [r["system"][0] for r in seen["runs"]],
        "tools": [r["tools"][0] for r in seen["runs"]],
        "logs": logs.lines,
    })
    return s.result()


def unknown_name(ctx: dict) -> dict:
    """The model asks for a skill nobody installed, then for the real one: the guarded
    capability answers the first instead of aborting the run (W3)."""
    s = col.Session("openai-agents", distributions=DISTRIBUTIONS, openai_agents=True)
    turns = Turns()
    the_agent = agent(turns.model)
    seen: Dict[str, Any] = {}

    def once(run: col.Run) -> None:
        seen["run"] = got = drive(turns, the_agent,
                                  ("load_skill", {"skill_name": "no-such-skill"}), load(),
                                  cat("SKILL.md"))
        _into(run, got)

    s.run(once)
    s.notes["outputs"] = seen.get("run", {}).get("outputs")
    return s.result()


def symlinked(ctx: dict) -> dict:
    """The installed folders replaced by symlinks to copies elsewhere: the sandbox lists
    none of them, and sandbox_skills() says so (W3)."""
    import shutil

    logs = _Logs()
    logging.getLogger("decimalai.openai_agents").addHandler(logs)
    root = col.skills_dir()
    elsewhere = root.parent.parent.parent / "elsewhere-skills"
    shutil.copytree(root, elsewhere)
    for folder in list(root.iterdir()):
        if folder.is_dir():
            shutil.rmtree(folder)
            folder.symlink_to(elsewhere / folder.name, target_is_directory=True)
    from decimalai.openai_agents import sandbox_skills

    sandbox_skills()
    return {"logs": logs.lines}


# ── the live check (tests/native/test_native_live.py) ────────────────────────


def _call_args(raw: Any) -> Any:
    """A tool call's arguments as the model sent them: parsed JSON where it is JSON."""
    import json

    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def _live_hooks(run: col.Run) -> Any:
    """Run hooks that record, as the run goes, what the model was given and every tool
    call it made (from each model response), so a run that stops early still reports
    what it did."""
    from agents import RunHooks

    class Observe(RunHooks):
        def __init__(self) -> None:
            self.prompted = False

        async def on_llm_start(self, context, agent, system_prompt, input_items) -> None:
            if not self.prompted:
                self.prompted = True
                run.saw(system_prompt)

        async def on_llm_end(self, context, agent, response) -> None:
            for item in response.output:
                d = item if isinstance(item, dict) else item.model_dump()
                if d.get("type") in ("message", "reasoning"):
                    continue
                run.called(d.get("name") or d.get("type"),
                           _call_args(d.get("arguments", d.get("input", d.get("action")))))

        async def on_tool_end(self, context, agent, tool, result) -> None:
            run.saw(str(result))

    return Observe()


def live(ctx: dict) -> dict:
    """Each trial: a fresh ``SandboxAgent`` with ``sandbox_skills()`` and a REAL OpenAI
    model, asked the live question; every tool call it made, what it was given and its
    answer (see ``tests/native/live.py``)."""
    from tests.native import live as lv

    s = col.Session("openai-agents", distributions=DISTRIBUTIONS, openai_agents=True)
    # One event loop for every trial, as a long-lived app has: the SDK's OpenAI provider
    # shares one process-wide httpx client, whose pooled connections belong to the loop
    # that opened them, so an `asyncio.run` per trial failed a later trial with "Event loop
    # is closed" (seen live, trial 2 of 3).
    loop = asyncio.new_event_loop()

    def trial(run: col.Run) -> None:
        from agents import RunConfig, Runner
        from agents.sandbox import SandboxAgent, SandboxRunConfig
        from agents.sandbox.capabilities import Shell
        from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

        from decimalai.openai_agents import sandbox_skills

        the_agent = SandboxAgent(name="live-support", instructions=lv.INSTRUCTIONS,
                                 model=lv.openai_model(),
                                 capabilities=[Shell(), sandbox_skills()],
                                 default_manifest=manifest())
        config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()))
        result = loop.run_until_complete(Runner.run(
            the_agent, lv.QUESTION, run_config=config, hooks=_live_hooks(run), max_turns=12))
        run.answered(result.final_output)

    try:
        for _ in range(lv.trials(ctx)):
            s.run(trial)
    finally:
        loop.close()
    s.notes["model"] = lv.openai_model()
    return s.result()
