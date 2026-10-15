"""claude-agent-sdk: the bundled Claude Code CLI loading ``.agents`` as a plugin.

Documented snippet::

    query(prompt, options=ClaudeAgentOptions(
        plugins=[{"type": "local", "path": str(decimalai.skills_dir().parent)}]))

The plugin root IS ``.agents/`` (absolute: a relative ``--plugin-dir`` fails with
no visible error from a subdirectory), so its default skills folder is the same
``.agents/skills`` the other seven loaders read. The CLI is the loader, so the
real binary runs, against the local Messages stub. ``permission_mode="default"``
(the CLI's own default, "auto", makes a classifier side call), no setting sources,
an isolated ``HOME``.

Scripted turns: load the target skill; Read its resource; run its script with the
scoped Bash rule; a Bash outside that rule (must be denied); load the foreign skill
and run ITS script (must be denied: a skill the org did not author pre-approves no
Bash); load the user-only skill (must be refused).

Then the same snippet once more, from a subdirectory of the project (made for the
run and removed after it): ``skills_dir()`` walks up to the project, and the CLI,
started there, must still load the plugin with every skill.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._observe import (
    Ctx,
    attempt,
    executable,
    new_observation,
    read_text,
    split_frontmatter,
    stamp_of,
    unique,
    without_line_numbers,
)

NAME = "claude-agent-sdk"
#: Where the subdirectory run starts, relative to the project.
SUBDIRECTORY = "src"


def _scrub_parent_session() -> None:
    for k in list(os.environ):
        if k.startswith(("CLAUDE", "ANTHROPIC")) and k not in (
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
                "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL", "CLAUDE_CONFIG_DIR"):
            os.environ.pop(k, None)


def _options(ctx: Ctx, stub_url: str, stderr: List[str], max_turns: int, cwd: Path):
    from claude_agent_sdk import ClaudeAgentOptions

    plugin_root = ctx.skills_dir().parent
    return plugin_root, ClaudeAgentOptions(
        cwd=str(cwd),
        plugins=[{"type": "local", "path": str(plugin_root)}],
        setting_sources=[],
        skills="all",
        permission_mode="default",
        max_turns=max_turns,
        stderr=stderr.append,
        env={
            "HOME": str(ctx.home), "CLAUDE_CONFIG_DIR": str(ctx.home / ".claude"),
            "ANTHROPIC_BASE_URL": stub_url, "ANTHROPIC_API_KEY": "dummy",
            "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
            "NO_PROXY": "127.0.0.1,localhost",
            "DISABLE_TELEMETRY": "1", "DISABLE_AUTOUPDATER": "1", "DISABLE_ERROR_REPORTING": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
            "GIT_SSH_COMMAND": "/usr/bin/false", "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0",
        },
    )


def _session(ctx: Ctx, steps: list, max_turns: int, cwd: Optional[Path] = None) -> Dict[str, Any]:
    _scrub_parent_session()
    from claude_agent_sdk import query

    from ..stubs.anthropic_messages import Stub

    stderr: List[str] = []
    messages: List[Any] = []
    with Stub(steps) as stub:
        plugin_root, opts = _options(ctx, stub.url, stderr, max_turns, cwd or ctx.project)

        async def run() -> None:
            async for m in query(prompt="A customer wants a refund for $100 after 45 days.",
                                 options=opts):
                messages.append(m)

        asyncio.run(run())
        requests = stub.main_loop_requests()
    return {"messages": messages, "requests": requests, "stderr": stderr, "plugin_root": plugin_root}


def _init(messages: list) -> dict:
    return next((m.data for m in messages
                 if type(m).__name__ == "SystemMessage" and getattr(m, "subtype", "") == "init"), {})


def _our_plugin(init: dict, plugin_root: Path) -> Optional[dict]:
    for p in init.get("plugins") or []:
        path = p.get("path")
        if path and os.path.realpath(path) == os.path.realpath(plugin_root):
            return p
    return None


def _ours(names: List[str], prefix: Optional[str]) -> List[str]:
    if not prefix:
        return []
    return unique(n.split(":", 1)[1] for n in names if n.startswith(prefix + ":"))


def _listing(requests: List[dict], prefix: Optional[str]) -> Optional[List[str]]:
    """Names in the skill listing the model was sent (the CLI sends a budgeted subset)."""
    from ..stubs.anthropic_messages import text_of

    if not requests or not prefix:
        return None
    text = text_of(requests[0])
    return unique(re.findall(r"(?m)^- " + re.escape(prefix) + r":([^\s:]+):", text))


def _from_subdirectory(ctx: Ctx) -> Dict[str, Any]:
    """The snippet as a user runs it from ``<project>/src``: no turns, just the init."""
    sub = ctx.project / SUBDIRECTORY
    made = not sub.exists()
    sub.mkdir(exist_ok=True)
    here = os.getcwd()
    try:
        os.chdir(sub)
        s = _session(ctx, steps=[], max_turns=1, cwd=sub)
    finally:
        os.chdir(here)
        if made:
            sub.rmdir()
    init = _init(s["messages"])
    return {
        "cwd": str(sub),
        "plugin_root": str(s["plugin_root"]),
        "plugin": _our_plugin(init, s["plugin_root"]),
        "plugins": init.get("plugins"),
        "plugin_errors": init.get("plugin_errors"),
        "skills": [n for n in init.get("skills") or [] if ":" in n],
    }


def discover(ctx: Ctx) -> Dict[str, Any]:
    out: Dict[str, Any] = {"listed": None, "raised": None, "channel": []}
    try:
        s = _session(ctx, steps=[], max_turns=1)
    except Exception as exc:  # noqa: BLE001
        out["raised"] = f"{type(exc).__name__}: {exc}"[:300]
        return out
    init = _init(s["messages"])
    plugin = _our_plugin(init, s["plugin_root"])
    out["listed"] = _ours(init.get("skills") or [], plugin and plugin.get("name"))
    out["channel"] = [str(e) for e in (init.get("plugin_errors") or [])]
    return out


def observe(ctx: Ctx) -> Dict[str, Any]:
    from ..stubs.anthropic_messages import last_base_dir, text_of, tool_use

    sc = ctx.scenario
    obs = new_observation(NAME, "claude-agent-sdk")
    foreign = (ctx.by_variant("foreign") or [None])[0]

    def qualified(body: dict, name: str) -> str:
        """``<plugin>:<name>`` as the model's listing spells it; a skill the listing hides
        (user-only) is addressed with the prefix the listing uses for the others."""
        text = text_of(body)
        m = re.search(r"(?m)^- ([^\s:]+:" + re.escape(name) + r"):", text)
        if m:
            return m.group(1)
        others = "|".join(re.escape(n) for n in ctx.names)
        m = re.search(r"(?m)^- ([^\s:]+):(?:" + others + r"):", text) if others else None
        return f"{m.group(1) if m else ctx.plugin_name}:{name}"

    def base(body: dict) -> str:
        return last_base_dir(body) or str(ctx.skills_dir() / "MISSING-BASE-DIR")

    run_script = " ".join(("python3", "{base}/" + sc.script, *sc.script_args))
    steps = [
        lambda b: [tool_use("Skill", {"skill": qualified(b, sc.target)})],
        lambda b: [tool_use("Read", {"file_path": f"{base(b)}/{sc.resource}"})],
        lambda b: [tool_use("Bash", {"command": run_script.format(base=base(b)),
                                     "description": "run the skill's script"})],
        lambda b: [tool_use("Bash", {"command": "python3 -c 'print(6*7)'",
                                     "description": "a command outside the skill's rule"})],
    ]
    if foreign:
        steps += [
            lambda b: [tool_use("Skill", {"skill": qualified(b, foreign)})],
            lambda b: [tool_use("Bash", {"command": run_script.format(base=base(b)),
                                         "description": "run the foreign skill's script"})],
        ]
    steps.append(lambda b: [tool_use("Skill", {"skill": qualified(b, sc.user_only)})])

    s = attempt(obs, "run", lambda: _session(ctx, steps, max_turns=len(steps) + 2))
    if s is None:
        return obs
    init = _init(s["messages"])
    plugin = _our_plugin(init, s["plugin_root"])
    prefix = plugin.get("name") if plugin else None
    obs["listed"] = _ours(init.get("skills") or [], prefix)
    obs["offered"] = _listing(s["requests"], prefix)
    obs["channel"] = [str(e) for e in (init.get("plugin_errors") or [])]

    # Tool results, in call order, keyed by the tool_use id the stub minted.
    calls: List[dict] = []
    results: Dict[str, dict] = {}
    for m in s["messages"]:
        content = getattr(m, "content", None)
        if not isinstance(content, list):
            continue
        for b in content:
            kind = type(b).__name__
            if kind == "ToolUseBlock":
                calls.append({"id": b.id, "name": b.name, "input": b.input})
            elif kind == "ToolResultBlock":
                c = b.content
                text = c if isinstance(c, str) else "\n".join(
                    x.get("text", "") for x in (c or []) if isinstance(x, dict))
                results[b.tool_use_id] = {"text": text, "is_error": bool(b.is_error)}
    final = next((m for m in s["messages"] if type(m).__name__ == "ResultMessage"), None)
    denials = [d.get("tool_input", {}) for d in (getattr(final, "permission_denials", None) or [])]

    def result_of(i: int) -> Optional[dict]:
        return results.get(calls[i]["id"]) if i < len(calls) else None

    # What the model saw after each call: the request that followed it.
    reqs = s["requests"]

    def seen_after(i: int) -> str:
        if i + 1 >= len(reqs):
            return ""
        msgs = reqs[i + 1].get("messages", [])
        last_assistant = max((k for k, mm in enumerate(msgs) if mm.get("role") == "assistant"),
                             default=-1)
        return text_of(msgs[last_assistant + 1:])

    target_base = None
    m = re.search(r"Base directory for this skill:\s*(\S+)", seen_after(0))
    if m:
        target_base = m.group(1)
    obs["body_text"] = seen_after(0) or None
    if target_base:
        skill_md = Path(target_base) / "SKILL.md"
        file_text = read_text(skill_md)
        obs["stamp"] = {"via": "file", "path": str(skill_md), "metadata": stamp_of(file_text)}
        delivered = obs["body_text"] or ""
        _, file_body = split_frontmatter(file_text or "")
        # The CLI hands the model the file's body after a "Base directory" header.
        obs["body"] = file_body if file_body and file_body.strip() in delivered else None
        script = Path(target_base) / sc.script
        bit = executable(script)
        obs["exec_bit"] = {"path": str(script), "source": bit, "copy": bit}
    read = result_of(1)
    obs["notes"]["resource_raw"] = read["text"] if read else None
    obs["resource_text"] = without_line_numbers(read["text"]) if read else None
    obs["resource_exact"] = False
    scoped = result_of(2)
    obs["script_stdout"] = scoped["text"] if scoped else None

    commands_denied = [d.get("command") for d in denials if d.get("command")]
    skills_denied = [d.get("skill") for d in denials if d.get("skill")]
    foreign_tools = None
    if foreign:
        fdir = Path(s["plugin_root"]) / "skills" / foreign
        fm, _ = split_frontmatter(read_text(fdir / "SKILL.md") or "")
        if fm is not None:
            import yaml

            foreign_tools = (yaml.safe_load(fm) or {}).get("allowed-tools")
    i_user = len(calls) - 1
    user_res = result_of(i_user) if calls and calls[-1]["name"] == "Skill" else None
    obs["user_only"] = {
        "offered": (sc.user_only in obs["offered"]) if obs["offered"] is not None else None,
        "refusal": (user_res or {}).get("text"),
    }
    obs["claude"] = {
        "cli_version": init.get("claude_code_version"),
        "plugin": plugin,
        "plugins": init.get("plugins"),
        "plugin_errors": init.get("plugin_errors"),
        "plugin_root": str(s["plugin_root"]),
        "skills": [n for n in init.get("skills") or [] if ":" in n],
        "calls": [{"name": c["name"], "input": c["input"], "result": results.get(c["id"])}
                  for c in calls],
        "scoped_command": calls[2]["input"].get("command") if len(calls) > 2 else None,
        "outside_command": calls[3]["input"].get("command") if len(calls) > 3 else None,
        "foreign": foreign,
        "foreign_command": calls[5]["input"].get("command") if foreign and len(calls) > 5 else None,
        "foreign_loaded": bool(foreign and len(calls) > 4 and result_of(4)
                               and not result_of(4)["is_error"]),
        "foreign_allowed_tools": foreign_tools,
        "commands_denied": commands_denied,
        "skills_denied": skills_denied,
        "stderr_tail": s["stderr"][-5:],
        "from_subdirectory": attempt(obs, "subdirectory", lambda: _from_subdirectory(ctx)),
    }
    return obs
