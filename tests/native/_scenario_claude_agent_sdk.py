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


# ── the live check (``test_native_live.py``): the real CLI and a REAL Claude model ──────

#: The budget Claude model the live check runs, overridable by ``LIVE_LLM_ANTHROPIC_MODEL``.
#: Never Opus: the live lanes are budget tier.
DEFAULT_LIVE_MODEL = "claude-haiku-4-5-20251001"
#: Enough turns for a load, a resource read, a script run and the answer.
LIVE_MAX_TURNS = 8


def live_model() -> str:
    model = os.environ.get("LIVE_LLM_ANTHROPIC_MODEL") or DEFAULT_LIVE_MODEL
    if "opus" in model.lower():
        raise ValueError(f"the live check runs a budget model, not {model!r}")
    return model


def _scrub_host_env() -> None:
    """Drop what a parent Claude Code session or a developer's shell leaves in this child's
    environment (``ANTHROPIC_BASE_URL``, OAuth and host-session variables), which the SDK
    passes on to the CLI: the run must reach the real API with ``ANTHROPIC_API_KEY`` and
    nothing else, never a host's own session."""
    for name in list(os.environ):
        if name.startswith(("CLAUDE", "ANTHROPIC_")) and name != "ANTHROPIC_API_KEY":
            os.environ.pop(name, None)


def _live_env() -> Dict[str, str]:
    """:func:`_env` without the stub: the real API, the caller's key, the same isolation."""
    env = _env("")
    for key in ("ANTHROPIC_BASE_URL", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        env.pop(key)
    env["ANTHROPIC_API_KEY"] = os.environ.get("ANTHROPIC_API_KEY", "")
    return env


def live_options(stderr: List[str], env: Optional[Dict[str, str]] = None) -> Any:
    """The documented native-mode options as :func:`options` builds them, with the live
    check's instructions as the system prompt and a real budget model."""
    from claude_agent_sdk import ClaudeAgentOptions

    from decimalai.claude_agent_sdk import skills_plugin
    from tests.native import live

    return ClaudeAgentOptions(
        cwd=str(Path.cwd()), plugins=[skills_plugin()], setting_sources=[], skills="all",
        permission_mode="default", system_prompt=live.INSTRUCTIONS, model=live_model(),
        max_turns=LIVE_MAX_TURNS, stderr=stderr.append,
        env=env if env is not None else _live_env())


def _record_stream(run: col.Run, messages: List[Any]) -> None:
    """Every tool call the model made (its ``ToolUseBlock``s) and what it was handed back
    (tool results, and the text the CLI adds to the conversation, a skill's body)."""
    for m in messages:
        kind = type(m).__name__
        content = getattr(m, "content", None)
        if kind == "AssistantMessage":
            for block in content or []:
                if type(block).__name__ == "ToolUseBlock":
                    run.called(block.name, block.input)
        elif kind == "UserMessage":
            for block in content if isinstance(content, list) else [content]:
                if isinstance(block, str):
                    run.saw(block)
                elif type(block).__name__ == "ToolResultBlock":
                    run.saw(block.content)
                elif type(block).__name__ == "TextBlock":
                    run.saw(block.text)


def _refusal(messages: List[Any]) -> Optional[str]:
    """The provider's refusal, in the CLI's own words, when the API refused the run.

    The CLI reports a failed API call as an assistant message with ``error`` set (its kind:
    ``billing_error``, ``rate_limit``, ...) and the API's message as text, then a result
    with ``is_error``; the stream itself does not raise for it."""
    for m in messages:
        if type(m).__name__ == "AssistantMessage" and getattr(m, "error", None):
            text = " ".join(getattr(b, "text", "") for b in m.content or []
                            if type(b).__name__ == "TextBlock")
            return f"the API refused the run ({m.error}): {text.strip()}"
    for m in messages:
        if type(m).__name__ == "ResultMessage" and m.is_error and m.api_error_status:
            return (f"the API refused the run (HTTP {m.api_error_status}): "
                    f"{m.result or m.errors}")
    return None


def _live_query(run: col.Run, opts: Any, stderr: List[str]) -> None:
    import claude_agent_sdk

    from tests.native import live

    messages: List[Any] = []

    async def go() -> None:
        async for m in claude_agent_sdk.query(prompt=live.QUESTION, options=opts):
            messages.append(m)

    failure: Optional[BaseException] = None
    try:
        asyncio.run(go())
    except Exception as exc:  # noqa: BLE001 - classified below, then raised
        failure = exc
    _record_stream(run, messages)
    refused = _refusal(messages)
    if refused:
        raise RuntimeError(refused) from failure
    if failure is not None:
        raise RuntimeError(f"{type(failure).__name__}: {failure}; "
                           f"stderr: {' '.join(stderr[-5:])[-800:]}") from failure
    result = next((m for m in reversed(messages) if type(m).__name__ == "ResultMessage"), None)
    if result is None:
        raise RuntimeError(f"the run ended without a result; stderr: {' '.join(stderr[-5:])[-800:]}")
    if result.is_error:
        raise RuntimeError(f"the run ended in error ({result.subtype}): "
                           f"{result.result or result.errors}")
    run.answered(result.result)


def live(ctx: dict) -> dict:
    """The live native check (``tests/native/live.py``): ``live.trials(ctx)`` independent
    ``query()`` runs of the real bundled CLI (each a fresh session) with
    :func:`skills_plugin`, a real budget Claude model and ``ANTHROPIC_API_KEY``."""
    from tests.native import live as live_check

    live_model()
    _scrub_host_env()
    s = col.Session("claude-agent-sdk", distributions=DISTRIBUTIONS, claude_agent_sdk=True)

    def trial(run: col.Run) -> None:
        stderr: List[str] = []
        _live_query(run, live_options(stderr), stderr)

    for _ in range(live_check.trials(ctx)):
        s.run(trial)
    return s.result()
