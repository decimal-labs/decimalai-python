"""agent-framework-core (Microsoft Agent Framework): ``SkillsProvider`` on an ``Agent``.

Documented snippet::

    SkillsProvider.from_paths(str(decimalai.skills_dir()), script_runner=runner)

A bad folder is a log line on ``agent_framework`` (ERROR) and the skill is skipped,
so the channel is that logger. Every skill tool is approval-gated by default; the
check opts the three out (``disable_*_approval``) rather than answering prompts.

MAF ships NO script runner: without one a file script says it "requires a runner".
So N5 here means "MAF listed the script and handed the runner the right full
path", graded with a trivial subprocess runner; and the no-runner fact is
re-proven on every run (``proofs["no_runner"]``), so the day MAF ships one, N5
fails until it is graded through MAF's own.

Caching is on by default, so a long-lived provider keeps serving the old version
after an update; each check here is one cold process.
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from ._observe import (
    Ctx,
    attempt,
    body_of,
    executable,
    new_observation,
    unique,
    watch_loggers,
)

NAME = "agent-framework"
CHANNEL = ("agent_framework",)


def _runner(ran: List[str]):
    def run(skill, script, args=None):
        argv = [str(a) for a in (args.values() if isinstance(args, dict) else (args or []))]
        ran.append(script.full_path)
        p = subprocess.run([sys.executable, script.full_path, *argv], capture_output=True,
                           text=True, timeout=30)
        return p.stdout
    return run


async def _skills(root: str, runner=None):
    from agent_framework import FileSkillsSource, SkillsSourceContext

    source = FileSkillsSource(root, script_runner=runner) if runner else FileSkillsSource(root)
    return await source.get_skills(SkillsSourceContext(agent=None))


def discover(ctx: Ctx) -> Dict[str, Any]:
    channel = watch_loggers(*CHANNEL)
    out: Dict[str, Any] = {"listed": None, "raised": None, "channel": channel.lines}
    try:
        out["listed"] = sorted(s.frontmatter.name for s in asyncio.run(_skills(str(ctx.skills_dir()))))
    except Exception as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


def observe(ctx: Ctx) -> Dict[str, Any]:
    sc = ctx.scenario
    obs = new_observation(NAME, "agent-framework-core")
    channel = watch_loggers(*CHANNEL)
    obs["channel"] = channel.lines

    from agent_framework import Agent, SkillsProvider

    from ..stubs.maf_client import ScriptedChatClient, function_results

    root = str(ctx.skills_dir())
    ran: List[str] = []
    found = attempt(obs, "discover", lambda: asyncio.run(_skills(root, _runner(ran))))
    if found is None:
        return obs
    by = {s.frontmatter.name: s for s in found}
    obs["listed"] = sorted(by)
    target = by.get(sc.target)
    if target is not None:
        md = target.frontmatter.metadata
        obs["stamp"] = {"via": "loader", "path": None,
                        "metadata": dict(md) if isinstance(md, dict) else md}
        obs["notes"]["allowed_tools_as_parsed"] = target.frontmatter.allowed_tools
        obs["notes"]["scripts_listed"] = [s.name for s in getattr(target, "_scripts", [])]

    async def no_runner_proof() -> str:
        bare = {s.frontmatter.name: s for s in await _skills(root)}
        skill = bare.get(sc.target)
        script = await skill.get_script(sc.script) if skill else None
        if script is None:
            return "(the target or its script was not listed)"
        try:
            await script.run(skill, args=list(sc.script_args))
        except ValueError as exc:
            return str(exc)
        return "(ran without a runner)"

    obs["proofs"]["no_runner"] = attempt(obs, "proof", lambda: asyncio.run(no_runner_proof()))

    client = ScriptedChatClient([
        ("calls", [("load_skill", {"skill_name": sc.target})]),
        ("calls", [("read_skill_resource", {"skill_name": sc.target, "resource_name": sc.resource})]),
        ("calls", [("run_skill_script", {"skill_name": sc.target, "script_name": sc.script,
                                         "args": list(sc.script_args)})]),
        ("text", "done"),
    ])
    provider = SkillsProvider.from_paths(root, script_runner=_runner(ran),
                                         disable_load_skill_approval=True,
                                         disable_read_skill_resource_approval=True,
                                         disable_run_skill_script_approval=True)
    agent = Agent(client=client, instructions="Answer refund questions.", context_providers=[provider])
    attempt(obs, "run", lambda: asyncio.run(agent.run("refund for $100 after 45 days?")))
    if client.requests:
        instructions = str(client.requests[0]["options"].get("instructions", ""))
        obs["offered"] = unique(re.findall(r"<name>([^<]+)</name>", instructions))
        obs["user_only"]["offered"] = sc.user_only in obs["offered"]
        results = [str(r) for _, r in function_results(client.requests[-1]["messages"])]
        obs["notes"]["results"] = [r[:400] for r in results]
        obs["body_text"] = results[0] if results else None
        obs["body"] = _body(results[0]) if results else None
        obs["resource_text"] = results[1] if len(results) > 1 else None
        obs["resource_exact"] = True
        obs["script_stdout"] = results[2] if len(results) > 2 else None
    obs["notes"]["runner_called_with"] = ran[-1:]
    script = Path(ran[-1]) if ran else Path(root) / sc.target / sc.script
    bit = executable(script)
    obs["exec_bit"] = {"path": str(script), "source": bit, "copy": bit}
    return obs


def _body(loaded: str):
    """load_skill returns the whole SKILL.md, then MAF's own resource and script blocks."""
    m = re.search(r"\n\n<(available_resources|available_scripts)\b", loaded)
    text = loaded[:m.start()] if m else loaded
    return body_of(text) if text.startswith("---\n") else text
