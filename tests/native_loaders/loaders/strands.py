"""strands-agents (openai 2 environment): the ``AgentSkills`` plugin, strict.

Documented snippet::

    AgentSkills(skills=str(decimalai.skills_dir()), strict=True)

``strict=True`` on the plugin does NOT raise: it logs "failed to load skill" and
skips, per skill. Only ``Skill.from_file(path, strict=True)`` raises, so the check
uses both: the per-folder strict parse (``proofs["strict_parse"]``) and the plugin,
which is the documented path. Strands sits in the openai 2 environment because its
``mcp<2.2`` cap would move ``mcp`` under claude-agent-sdk and openai-agents in
the openai 3 one.

Never an ``Agent()`` without a model (the default is Bedrock, on the network).
``file_editor`` rejects relative paths, so the skills root must be absolute.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict

from ._observe import Ctx, attempt, executable, new_observation, unique, watch_loggers

NAME = "strands-agents"
CHANNEL = ("strands",)


def discover(ctx: Ctx) -> Dict[str, Any]:
    channel = watch_loggers(*CHANNEL)
    out: Dict[str, Any] = {"listed": None, "raised": None, "channel": channel.lines}
    try:
        from strands import Agent, AgentSkills

        from ..stubs.strands_model import ScriptedModel

        plugin = AgentSkills(skills=str(ctx.skills_dir()), strict=True)
        agent = Agent(model=ScriptedModel([]), plugins=[plugin], callback_handler=None)
        out["listed"] = sorted(s.name for s in plugin.get_available_skills(agent))
    except Exception as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


def observe(ctx: Ctx) -> Dict[str, Any]:
    sc = ctx.scenario
    obs = new_observation(NAME, "strands-agents")
    channel = watch_loggers(*CHANNEL)
    obs["channel"] = channel.lines

    from strands import Agent, AgentSkills
    from strands.vended_plugins.skills.skill import Skill
    from strands.vended_tools import file_editor, shell

    from ..stubs.strands_model import ScriptedModel, last_tool_result_text

    root = ctx.skills_dir()
    d = root / sc.target
    strict: Dict[str, str] = {}
    for p in sorted(root.iterdir()):
        try:
            Skill.from_file(p, strict=True)
            strict[p.name] = "ok"
        except Exception as exc:  # noqa: BLE001
            strict[p.name] = f"{type(exc).__name__}: {exc}"[:300]
    obs["proofs"]["strict_parse"] = strict

    located: Dict[str, str] = {}

    def folder(messages) -> str:
        """The skill folder from the skills tool's "Location:" line, as the model reads it."""
        if "dir" not in located:
            m = re.search(r"(?m)^Location: (\S+)/SKILL\.md$", last_tool_result_text(messages))
            located["dir"] = m.group(1) if m else ""
        return located["dir"]

    def view(messages):
        if not folder(messages):
            return ("text", f"could not load {sc.target}")
        return ("tool", "file_editor", {"command": "view", "path": f"{folder(messages)}/{sc.resource}"})

    def run(messages):
        cmd = " ".join((sys.executable, f"{located['dir']}/{sc.script}", *sc.script_args))
        return ("tool", "shell", {"command": cmd})

    steps = [
        lambda m: ("tool", "skills", {"skill_name": sc.target}),
        view,
        run,
        lambda m: ("text", "done"),
    ]
    model = ScriptedModel(steps)
    plugin = AgentSkills(skills=str(root), strict=True)
    agent = attempt(obs, "discover", lambda: Agent(model=model, system_prompt="Answer refund questions.",
                                                   tools=[file_editor, shell], plugins=[plugin],
                                                   callback_handler=None))
    if agent is None:
        return obs
    available = attempt(obs, "discover", lambda: plugin.get_available_skills(agent))
    if available is not None:
        by = {s.name: s for s in available}
        obs["listed"] = sorted(by)
        target = by.get(sc.target)
        if target is not None:
            obs["body"] = target.instructions
            md = target.metadata
            obs["stamp"] = {"via": "loader", "path": None,
                            "metadata": dict(md) if isinstance(md, dict) else md}
            obs["notes"]["allowed_tools_as_parsed"] = target.allowed_tools
    attempt(obs, "run", lambda: agent("refund for $100 after 45 days?"))
    if model.seen:
        system = model.seen[0]["system_prompt"] or ""
        obs["offered"] = unique(re.findall(r"<name>([^<]+)</name>", system))
        obs["user_only"]["offered"] = sc.user_only in obs["offered"]
    seen = model.seen
    obs["body_text"] = last_tool_result_text(seen[1]["messages"]) if len(seen) > 1 else None
    viewed = last_tool_result_text(seen[2]["messages"]) if len(seen) > 2 else None
    obs["notes"]["resource_raw"] = viewed
    obs["resource_text"] = _viewed(viewed)
    obs["resource_exact"] = False
    obs["script_stdout"] = last_tool_result_text(seen[3]["messages"]) if len(seen) > 3 else None
    script = (Path(located["dir"]) if located.get("dir") else d) / sc.script
    bit = executable(script)
    obs["exec_bit"] = {"path": str(script), "source": bit, "copy": bit}
    return obs


_VIEW_HEADER = re.compile(r"\AHere's the result of running `cat -n` on [^\n]*:\n")
_CAT_N = re.compile(r"^ *\d+(?:\t|  )", re.M)


def _viewed(text):
    """file_editor's ``view`` output with its ``cat -n`` decoration removed."""
    if text is None:
        return None
    return _CAT_N.sub("", _VIEW_HEADER.sub("", text))
