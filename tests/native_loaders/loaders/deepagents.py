"""deepagents: ``create_deep_agent(skills=[...])`` over a real local shell backend.

Documented snippet::

    create_deep_agent(model, skills=[str(decimalai.skills_dir())],
                      backend=LocalShellBackend(root_dir=project, virtual_mode=False))

``virtual_mode=False``: with ``True`` the prompt advertises virtual ``/skills/...``
paths that ``execute`` cannot open. The parsed skill list is kept out of the
agent's output, so discovery is the listing the skills middleware renders into
the system prompt (``**<name>**``), which is what the model sees. The stamp and
body come from ``read_file``'s output: the same bytes the model saw.

A namespaced or uppercase ``name`` is listed under THAT name, with a warning on
the ``deepagents`` logger: renamed, not dropped.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

from ._observe import (
    Ctx,
    attempt,
    body_of,
    executable,
    new_observation,
    stamp_of,
    unique,
    watch_loggers,
    without_line_numbers,
)

NAME = "deepagents"
CHANNEL = ("deepagents",)


def discover(ctx: Ctx) -> Dict[str, Any]:
    channel = watch_loggers(*CHANNEL)
    out: Dict[str, Any] = {"listed": None, "raised": None, "channel": channel.lines}
    try:
        from deepagents.backends import FilesystemBackend

        # Private, so it is used only here (the ledger matrix), never in the contract:
        # if upstream renames it, the matrix cell reports the crash by name.
        from deepagents.middleware.skills import _list_skills

        backend = FilesystemBackend(root_dir=ctx.project, virtual_mode=False)
        out["listed"] = sorted(m["name"] for m in _list_skills(backend, str(ctx.skills_dir())))
    except Exception as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    return out


def observe(ctx: Ctx) -> Dict[str, Any]:
    sc = ctx.scenario
    obs = new_observation(NAME, "deepagents")
    channel = watch_loggers(*CHANNEL)
    obs["channel"] = channel.lines

    from deepagents import create_deep_agent
    from deepagents.backends import LocalShellBackend
    from langchain_core.messages import SystemMessage, ToolMessage

    from ..stubs.deepagents_model import ScriptedModel, final, tool_call

    root = ctx.skills_dir()
    paths: Dict[str, str] = {}

    def skill_md(messages) -> str:
        """The SKILL.md path the system prompt's listing gives the target, as a model reads it."""
        if "md" not in paths:
            system = next((m for m in messages if isinstance(m, SystemMessage)), None)
            text = (system.text if isinstance(system.text, str) else system.text()) if system else ""
            m = re.search(r"(?m)^- \*\*" + re.escape(sc.target) + r"\*\*:.*(?:\n  -> (?!Read).*)*"
                          r"\n  -> Read `([^`]+)`", text)
            paths["md"] = m.group(1) if m else ""
        return paths["md"]

    def folder(messages) -> str:
        return skill_md(messages).rsplit("/", 1)[0]

    def step_read(messages):
        if not skill_md(messages):
            return final(f"{sc.target} is not in the skills listing")
        return tool_call("read_file", {"file_path": skill_md(messages), "limit": 1000}, "c1")

    def step_resource(messages):
        if not skill_md(messages):
            return final("done")
        return tool_call("read_file", {"file_path": f"{folder(messages)}/{sc.resource}"}, "c2")

    def step_script(messages):
        if not skill_md(messages):
            return final("done")
        cmd = " ".join((sys.executable, f"{folder(messages)}/{sc.script}", *sc.script_args))
        return tool_call("execute", {"command": cmd}, "c3")

    model = ScriptedModel(steps=[step_read, step_resource, step_script, lambda m: final("done")])
    backend = LocalShellBackend(root_dir=ctx.project, virtual_mode=False,
                                env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
    agent = attempt(obs, "discover", lambda: create_deep_agent(model=model, skills=[str(root)],
                                                               backend=backend))
    if agent is None:
        return obs
    out = attempt(obs, "run", lambda: agent.invoke(
        {"messages": [{"role": "user", "content": "refund for $100 after 45 days?"}]}))
    system = ""
    if model.seen_requests:
        msg = next((m for m in model.seen_requests[0] if isinstance(m, SystemMessage)), None)
        if msg is not None:
            system = msg.text if isinstance(msg.text, str) else msg.text()
    obs["notes"]["allowed_tools_line"] = [ln.strip() for ln in system.splitlines()
                                          if "Allowed tools" in ln][:3]
    if system:
        obs["listed"] = obs["offered"] = unique(re.findall(r"(?m)^- \*\*([^*]+)\*\*:", system))
        obs["user_only"]["offered"] = sc.user_only in obs["offered"]
    results = {m.tool_call_id: str(m.content) for m in (out or {}).get("messages", [])
               if isinstance(m, ToolMessage)}
    skill_text = _viewed(results.get("c1"))
    obs["body_text"] = results.get("c1")
    obs["body"] = body_of(skill_text)
    obs["stamp"] = {"via": "read_file", "path": paths.get("md"), "metadata": stamp_of(skill_text)}
    obs["notes"]["resource_raw"] = results.get("c2")
    obs["resource_text"] = _file_text(_viewed(results.get("c2")))
    obs["resource_exact"] = False
    obs["script_stdout"] = results.get("c3")
    script = Path(paths["md"]).parent / sc.script if paths.get("md") else root / sc.target / sc.script
    bit = executable(script)
    obs["exec_bit"] = {"path": str(script), "source": bit, "copy": bit}
    return obs


_VIEW_HEADER = re.compile(r"\A@@ lines \d+-\d+ of \d+ @@\n")


def _viewed(text):
    """read_file's output with its viewer decoration (a range header, line numbers) removed."""
    if text is None:
        return None
    return without_line_numbers(_VIEW_HEADER.sub("", text))


def _file_text(text):
    # read_file renders a file's lines; the file's own trailing newline is the last line's end.
    if text is None:
        return None
    return text if text.endswith("\n") else text + "\n"
