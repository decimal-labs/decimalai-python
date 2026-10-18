"""Claude Agent SDK native mode, run in a child (``_runner.run_scenario``).

The bundled Claude Code CLI is the loader, so the real binary runs, pointed at the local
Messages stub (``tests/native_loaders/stubs/anthropic_messages.py``) and loading the
project's skills through ``decimalai.claude_agent_sdk.skills_plugin()``. DecimalAI's
instrumentation (``decimalai.init(claude_agent_sdk=True)``) wraps every ``query()`` and
sends its trace to a probe. Nothing here asserts: the parent grades what comes back.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from tests.native import column as col
from tests.native_loaders.stubs.anthropic_messages import Stub, last_base_dir, text_of, tool_use

#: What the model is asked; the stub's script decides what it does.
PROMPT = "A customer wants a refund for $100 after 45 days."
ARGS = " ".join(col.SCRIPT_ARGS)
DISTRIBUTIONS = ["claude-agent-sdk"]


def _env(stub_url: str) -> Dict[str, str]:
    home = Path(os.environ["HOME"])
    (home / ".claude").mkdir(parents=True, exist_ok=True)
    return {
        "HOME": str(home), "CLAUDE_CONFIG_DIR": str(home / ".claude"),
        "ANTHROPIC_BASE_URL": stub_url, "ANTHROPIC_API_KEY": "dummy",
        "HTTP_PROXY": "http://127.0.0.1:9", "HTTPS_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost",
        "DISABLE_TELEMETRY": "1", "DISABLE_AUTOUPDATER": "1", "DISABLE_ERROR_REPORTING": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
        "GIT_SSH_COMMAND": "/usr/bin/false", "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0",
    }


def options(stub_url: str, stderr: List[str], max_turns: int) -> Any:
    """The documented native-mode options: the pointer helper's plugin, and the
    isolation the native-loader suite uses (no setting sources, no classifier side call)."""
    from claude_agent_sdk import ClaudeAgentOptions

    from decimalai.claude_agent_sdk import skills_plugin

    return ClaudeAgentOptions(
        cwd=str(Path.cwd()), plugins=[skills_plugin()], setting_sources=[], skills="all",
        permission_mode="default", max_turns=max_turns, stderr=stderr.append,
        env=_env(stub_url))


def qualified(body: dict, name: str) -> str:
    """``<plugin>:<name>`` as the model's own listing spells it."""
    import re

    m = re.search(r"(?m)^- ([^\s:]+:" + re.escape(name) + r"):", text_of(body))
    return m.group(1) if m else f"decimalai-skills:{name}"


def load(args: Optional[str] = None) -> Callable[[dict], list]:
    def step(body: dict) -> list:
        inp = {"skill": qualified(body, col.TARGET)}
        if args:
            inp["args"] = args
        return [tool_use("Skill", inp)]
    return step


def _base(body: dict) -> str:
    return last_base_dir(body) or "MISSING-BASE-DIR"


def read_resource(body: dict) -> list:
    return [tool_use("Read", {"file_path": f"{_base(body)}/{col.RESOURCE}"})]


def run_script(body: dict) -> list:
    return [tool_use("Bash", {"command": f"python3 {_base(body)}/{col.SCRIPT} {ARGS}",
                              "description": "run the skill's script"})]


def query(steps: List[Callable[[dict], list]]) -> Dict[str, Any]:
    """One ``query()`` run through the patched ``claude_agent_sdk.query``."""
    import claude_agent_sdk

    stderr: List[str] = []
    messages: List[Any] = []
    with Stub(steps) as stub:
        opts = options(stub.url, stderr, max_turns=len(steps) + 2)

        async def go() -> None:
            async for m in claude_agent_sdk.query(prompt=PROMPT, options=opts):
                messages.append(m)

        asyncio.run(go())
        requests = stub.main_loop_requests()
    init = next((m.data for m in messages if type(m).__name__ == "SystemMessage"
                 and getattr(m, "subtype", "") == "init"), {})
    loads = sum(1 for m in messages if type(m).__name__ == "AssistantMessage"
                for b in getattr(m, "content", None) or []
                if type(b).__name__ == "ToolUseBlock" and b.name == "Skill"
                and str((b.input or {}).get("skill", "")).endswith(":" + col.TARGET))
    return {"init": init, "received": _received(requests), "target_loads": loads,
            "stderr": stderr[-5:]}


def _received(requests: List[dict]) -> List[str]:
    """What each model turn was given since its previous turn: the new user content."""
    out = []
    for body in requests[1:]:
        msgs = body.get("messages", [])
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=-1)
        out.append(text_of(msgs[last + 1:]))
    return out


def _into(run: col.Run, got: Dict[str, Any]) -> None:
    for text in got["received"]:
        run.saw(text)
    run.loaded_target(got["target_loads"])


def column_runs(session: col.Session, first: List[Callable[[dict], list]]) -> None:
    """Run 1 (``first``), the update, then run 2 (load, answer), into ``session``."""
    seen: Dict[str, Any] = {}

    def run1(run: col.Run) -> None:
        seen["run1"] = got = query(first)
        _into(run, got)

    def run2(run: col.Run) -> None:
        seen["run2"] = got = query([load()])
        _into(run, got)

    session.run(run1)
    session.notes["v2"] = col.update_to_v2()
    session.run(run2)
    from decimalai.claude_agent_sdk import skills_plugin

    init = (seen.get("run1") or {}).get("init") or {}
    session.notes.update({
        "cli_version": init.get("claude_code_version"),
        "plugins": init.get("plugins"),
        "plugin_errors": init.get("plugin_errors"),
        "init_skills": init.get("skills"),
        "plugin_path": skills_plugin()["path"],
        "stderr": (seen.get("run1") or {}).get("stderr"),
    })


def column(ctx: dict) -> dict:
    """The native conformance column's scenario (see ``tests/native/column.py``)."""
    s = col.Session("claude-agent-sdk", distributions=DISTRIBUTIONS, claude_agent_sdk=True)
    column_runs(s, [load(), load(), read_resource, run_script])
    return s.result()


def native(ctx: dict) -> dict:
    """This adapter's own checks: as the column, but the first load passes ``args``,
    which the CLI appends to the body it hands the model."""
    s = col.Session("claude-agent-sdk", distributions=DISTRIBUTIONS, claude_agent_sdk=True)
    column_runs(s, [load(ARGS), load(), read_resource, run_script])
    return s.result()
