"""crewai (openai 2 environment): ``Agent(skills=[...])`` run as a ``Crew``.

Documented snippet::

    Agent(role=..., goal=..., backstory=..., llm=llm, skills=[decimalai.skills_dir()])

A bad folder is NOT raised: it becomes a ``SkillLoadFailedEvent`` on
``crewai_event_bus`` plus a log line, so the check subscribes before building the
agent and calls ``crewai_event_bus.flush()`` before reading (handlers run on a
thread pool; reading early is a race). CrewAI has the strictest metadata parser
of the eight: a value must be a string, so a nested map or an unquoted number
drops the skill.

Two upstream facts are re-proven on every run (``proofs``):

* no native resource reader: the only skill tool CrewAI gives the model is
  ``load_skill``; ``load_resources`` renders a file listing, nothing more;
* no native script runner: ``Agent(allow_code_execution=True)`` is a deprecated
  no-op that yields no tools.

CrewAI writes under ``$HOME`` (the parent gives each child a fresh one) and phones
home unless its telemetry variables are off (the parent sets them).
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Dict, List

from ._observe import Ctx, attempt, executable, new_observation, unique, watch_loggers

NAME = "crewai"
CHANNEL = ("crewai.skills",)


def _subscribe(failed: List[str]) -> None:
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.skill_events import SkillLoadFailedEvent

    crewai_event_bus.on(SkillLoadFailedEvent)(
        lambda src, ev: failed.append(f"SkillLoadFailedEvent {ev.skill_name}: {ev.error}"[:300]))


def discover(ctx: Ctx) -> Dict[str, Any]:
    channel = watch_loggers(*CHANNEL)
    failed: List[str] = []
    out: Dict[str, Any] = {"listed": None, "raised": None, "channel": channel.lines}
    try:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.skills.loader import discover_skills

        _subscribe(failed)
        out["listed"] = sorted(s.name for s in discover_skills(ctx.skills_dir()))
        crewai_event_bus.flush()
    except Exception as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    out["channel"] = channel.lines + failed
    return out


def observe(ctx: Ctx) -> Dict[str, Any]:
    sc = ctx.scenario
    obs = new_observation(NAME, "crewai")
    channel = watch_loggers(*CHANNEL)
    failed: List[str] = []

    from crewai import Agent, Crew, Task
    from crewai.events.event_bus import crewai_event_bus
    from crewai.skills.loader import activate_skill, discover_skills, load_resources

    from ..stubs.crewai_llm import ScriptedLLM, tool_call

    _subscribe(failed)
    root = ctx.skills_dir()
    llm = ScriptedLLM(model="scripted/fake")
    llm.script = [[tool_call("load_skill", {"skill_name": sc.target}, "c1")], "done"]
    llm.requests = []
    agent = attempt(obs, "discover", lambda: Agent(role="Support", goal="Answer refund questions",
                                                   backstory="b", llm=llm, skills=[root]))
    if agent is not None:
        by = {s.name: s for s in (agent.skills or [])}
        obs["listed"] = sorted(by)
        target = by.get(sc.target)
        if target is not None:
            # The body CrewAI delivers is the one its own activation loads.
            active = attempt(obs, "activate", lambda: activate_skill(target))
            obs["body"] = getattr(active, "instructions", None)
            md = target.frontmatter.metadata
            obs["stamp"] = {"via": "loader", "path": None,
                            "metadata": dict(md) if isinstance(md, dict) else md}
            obs["notes"]["allowed_tools_as_parsed"] = target.frontmatter.allowed_tools
        attempt(obs, "run", lambda: Crew(agents=[agent], tasks=[Task(
            description="refund for $100 after 45 days?", expected_output="an answer",
            agent=agent)]).kickoff())
    crewai_event_bus.flush()
    obs["channel"] = channel.lines + failed

    if llm.requests:
        first = llm.requests[0]
        system = "\n".join(str(m.get("content", "")) for m in first["messages"]
                           if m.get("role") == "system")
        obs["offered"] = unique(re.findall(r'<skill name="([^"]+)"', system))
        obs["user_only"]["offered"] = sc.user_only in obs["offered"]
        tools = [t.get("function", {}).get("name") or t.get("name") for t in (first.get("tools") or [])]
        obs["proofs"]["skill_tools"] = sorted(n for n in tools if n and "skill" in n.lower())
        last = llm.requests[-1]["messages"]
        tool_msgs = [m for m in last if m.get("role") == "tool"]
        obs["body_text"] = str(tool_msgs[-1]["content"]) if tool_msgs else None

    listing = attempt(obs, "resources", lambda: load_resources(
        next(x for x in discover_skills(root) if x.name == sc.target)))
    files = getattr(listing, "resource_files", None) if listing is not None else None
    obs["proofs"]["resource_listing"] = (
        sorted(f"{d}/{f}" for d, names in files.items() for f in names) if files else None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = attempt(obs, "proof", lambda: Agent(role="x", goal="y", backstory="z",
                                                    llm=ScriptedLLM(model="scripted/fake"),
                                                    allow_code_execution=True))
    obs["proofs"]["code_execution_tools"] = len(probe.tools or []) if probe is not None else None
    script = root / sc.target / sc.script
    bit = executable(script)
    obs["exec_bit"] = {"path": str(script), "source": bit, "copy": bit}
    return obs
