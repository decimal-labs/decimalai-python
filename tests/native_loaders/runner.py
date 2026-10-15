"""The parent side: start one child per (loader, project), bring its observation home.

The parent imports no framework and runs no loader code. It builds the child's
environment, runs ``_child.py`` with the project as the working directory, and
reads back JSON. A child that crashes, times out or writes nothing is a
``LoaderProcessError`` — the cells for that loader then ERROR, naming it. It is
never a skip: an ungraded loader reported as skipped is the failure this suite
exists to remove.

The environment each child gets (the flakiness register, applied once, here):

* nothing from a parent Claude Code session: every ``CLAUDE*`` and ``ANTHROPIC*``
  key is dropped (agents run the gate from inside Claude Code), and the Claude
  child gets a stub ``ANTHROPIC_BASE_URL`` and a dummy key of its own;
* a fresh ``HOME`` and ``CLAUDE_CONFIG_DIR`` per child (Claude reads ``~/.claude``;
  CrewAI writes under ``$HOME``);
* no network: ``HTTP(S)_PROXY`` point at a closed port, and ``NO_PROXY`` admits
  only the loopback stubs;
* no git over SSH, which ignores the HTTP proxy (an interactive Claude Code once
  cloned a public repository that way): ``GIT_SSH_COMMAND=/usr/bin/false``, no
  global or system git config, no terminal prompt, and Claude Code's official
  marketplace auto-install off;
* telemetry, auto-update, error reporting and tracing off for every framework;
* this interpreter's ``bin/`` first on ``PATH``: on macOS the openai-agents sandbox
  runs commands under ``sandbox-exec``, where only a ``python3`` on ``PATH`` whose
  venv the profile grants can start.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .oracle import PLUGIN_NAME

CHILD = Path(__file__).resolve().parent / "_child.py"

#: How long one loader child may run. Cold, with no bytecode cache, the slowest
#: (deepagents, CrewAI) take 4-8 s; a hung CLI must become a named error, not a hang.
TIMEOUT_ENV = "DECIMAL_NATIVE_LOADER_TIMEOUT"
DEFAULT_TIMEOUT = 120
#: Children run at once. Each is its own process with its own stub port.
JOBS_ENV = "DECIMAL_NATIVE_LOADER_JOBS"
DEFAULT_JOBS = 4

DEAD_PROXY = "http://127.0.0.1:9"

_DROP_PREFIXES = ("CLAUDE", "ANTHROPIC")
_KEEP = {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"}

_FIXED = {
    "HTTP_PROXY": DEAD_PROXY, "HTTPS_PROXY": DEAD_PROXY, "ALL_PROXY": DEAD_PROXY,
    "http_proxy": DEAD_PROXY, "https_proxy": DEAD_PROXY,
    "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
    "GIT_SSH_COMMAND": "/usr/bin/false", "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "DISABLE_TELEMETRY": "1", "DISABLE_AUTOUPDATER": "1", "DISABLE_ERROR_REPORTING": "1",
    "CREWAI_DISABLE_TELEMETRY": "true", "CREWAI_DISABLE_TRACKING": "true",
    "CREWAI_TRACING_ENABLED": "false", "OTEL_SDK_DISABLED": "true",
    "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    "LANGCHAIN_TRACING": "false", "OPENAI_AGENTS_DISABLE_TRACING": "1",
    "PYDANTIC_AI_NO_BANNER": "1", "DO_NOT_TRACK": "1",
}


class LoaderProcessError(RuntimeError):
    """A loader child did not return an observation."""


def child_env(home: Path, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    src = dict(os.environ if base is None else base)
    env = {k: v for k, v in src.items()
           if not (k.startswith(_DROP_PREFIXES) and k not in _KEEP)}
    env.update(_FIXED)
    env["HOME"] = str(home)
    env["CLAUDE_CONFIG_DIR"] = str(home / ".claude")
    env["XDG_CONFIG_HOME"] = str(home / ".config")
    env["XDG_DATA_HOME"] = str(home / ".local" / "share")
    env["XDG_CACHE_HOME"] = str(home / ".cache")
    bindir = str(Path(sys.executable).parent)
    rest = [p for p in env.get("PATH", "/usr/bin:/bin").split(os.pathsep) if p and p != bindir]
    env["PATH"] = os.pathsep.join([bindir] + rest)
    return env


def timeout() -> int:
    try:
        return int(os.environ.get(TIMEOUT_ENV) or DEFAULT_TIMEOUT)
    except ValueError:
        return DEFAULT_TIMEOUT


def jobs() -> int:
    try:
        return max(1, int(os.environ.get(JOBS_ENV) or DEFAULT_JOBS))
    except ValueError:
        return DEFAULT_JOBS


def run_child(loader: str, mode: str, project: Path, expect: Dict[str, dict],
              plugin_name: str, workdir: Optional[Path] = None) -> dict:
    """One loader, one project, one process. Returns the observation dict."""
    project = Path(project).resolve()
    scratch = Path(tempfile.mkdtemp(prefix=f"native-{loader}-", dir=workdir)).resolve()
    home = scratch / "home"
    (home / ".claude").mkdir(parents=True)
    ctx_path, out_path = scratch / "ctx.json", scratch / "out.json"
    ctx_path.write_text(json.dumps({"project": str(project), "home": str(home),
                                    "expect": expect, "plugin_name": plugin_name}))
    try:
        proc = subprocess.run(
            [sys.executable, str(CHILD), loader, mode, str(ctx_path), str(out_path)],
            cwd=str(project), env=child_env(home), capture_output=True, text=True,
            timeout=timeout(),
        )
    except subprocess.TimeoutExpired as exc:
        raise LoaderProcessError(
            f"{loader} ({mode}) did not finish within {timeout()}s ({TIMEOUT_ENV}); "
            f"stderr tail:\n{(exc.stderr or b'')[-2000:]!r}") from None
    if proc.returncode != 0 or not out_path.is_file():
        raise LoaderProcessError(
            f"{loader} ({mode}) child exited {proc.returncode} without an observation; "
            f"stderr tail:\n{proc.stderr[-3000:]}")
    result = json.loads(out_path.read_text())
    result.setdefault("child_stderr_tail", proc.stderr[-1500:])
    if "crashed" in result:
        raise LoaderProcessError(
            f"{loader} ({mode}) crashed outside the stages it observes: {result['crashed']}\n"
            f"{result.get('traceback', '')}")
    shutil.rmtree(scratch, ignore_errors=True)
    return result


Cell = Tuple[str, str, Path, Dict[str, dict], str]   # loader, mode, project, expect, plugin


def run_many(cells: Iterable[Cell], workdir: Optional[Path] = None) -> List[object]:
    """Run cells in parallel; each result is an observation or the LoaderProcessError."""
    cells = list(cells)

    def one(cell: Cell) -> object:
        try:
            return run_child(*cell, workdir=workdir)
        except LoaderProcessError as exc:
            return exc

    with ThreadPoolExecutor(jobs()) as pool:
        return list(pool.map(one, cells))


# ── one project, every loader this environment grades ────────────────────────


def available(loader) -> List[str]:
    """What ``loader`` needs that this interpreter cannot import (empty = runnable)."""
    import importlib.util

    missing = []
    for mod in loader.imports:
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(mod)
    return missing


class Run:
    """Every loader this environment grades, observed once on one project.

    The children start on the first ``observation()`` call, all at once (``jobs()``
    at a time), so the per-commit cost is the slowest loader, not the sum.
    """

    def __init__(self, label: str, project: Path, expect, workdir: Path) -> None:
        self.label = label
        self.project = Path(project).resolve()
        self.expect = expect
        self.workdir = Path(workdir).resolve()
        self._results: Optional[Dict[str, object]] = None

    def graded_here(self) -> List[str]:
        from tests.conformance.environments import loader_graded_elsewhere

        from .loaders import LOADERS

        return [l.name for l in LOADERS
                if loader_graded_elsewhere(l.name) is None and not available(l)]

    def _run(self) -> Dict[str, object]:
        if self._results is None:
            names = self.graded_here()
            cells = [(n, "observe", self.project, self.expect.as_child_expect(), PLUGIN_NAME)
                     for n in names]
            self._results = dict(zip(names, run_many(cells, workdir=self.workdir)))
        return self._results

    def observation(self, name: str) -> dict:
        """The observation for ``name``, or a declared skip naming why there is none."""
        from tests.conformance.environments import loader_graded_elsewhere

        from . import ledger
        from .loaders import LOADERS_BY_NAME

        if loader_graded_elsewhere(name) is not None:
            ledger.skip_declared("elsewhere", f"{name}:*", f"{name} is not graded here")
        missing = available(LOADERS_BY_NAME[name])
        if missing:
            ledger.skip_declared("unavailable", f"{name}:*",
                                 f"{name} cannot run: {', '.join(missing)} not importable")
        result = self._run()[name]
        if isinstance(result, LoaderProcessError):
            raise result
        return result  # type: ignore[return-value]

