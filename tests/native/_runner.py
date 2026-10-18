"""Run one native-mode scenario in a child process and get its JSON back.

Each framework's scenario (``decimalai.init`` against a probe, the real framework, its
own skill loader, a scripted model) runs in a process of its own: CrewAI's event bus,
Strands' plugin registry, ADK's runners, OpenTelemetry's global provider and every
adapter's ``instrument()`` patch are process-global, and these tests ride the default
unit run, so nothing they set may outlive them. The parent imports no framework.

A scenario is a module-level function ``fn(ctx: dict) -> dict`` in a module under
``tests/native``, named as ``"tests.native.<module>:<function>"``. It runs with the
project as its working directory and returns plain JSON. It may raise: the child
reports the exception and the traceback, and :func:`run_scenario` raises
:class:`ScenarioError` with them, so a crash is a failure, never a skip.

Unlike the native-loader suite's children, tracing is ON here: the point is the trace.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from tests.native_loaders.runner import child_env

CHILD = Path(__file__).resolve().parent / "_child.py"
REPO = CHILD.parent.parent.parent

TIMEOUT_ENV = "DECIMAL_NATIVE_TIMEOUT"
DEFAULT_TIMEOUT = 180

#: The native-loader children switch every tracer off; these children must not.
_TRACING_OFF = ("OTEL_SDK_DISABLED", "OPENAI_AGENTS_DISABLE_TRACING")


class ScenarioError(RuntimeError):
    """A scenario child crashed, hung or returned nothing."""


def scenario_env(home: Path) -> Dict[str, str]:
    env = child_env(home)
    for key in _TRACING_OFF:
        env.pop(key, None)
    # The SDK must not find a real key or a real backend in the developer's environment.
    for key in [k for k in env if k.startswith(("DECIMAL_", "DECIMALAI_"))]:
        env.pop(key)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO)] + [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def live_env(home: Path) -> Dict[str, str]:
    """A live scenario's environment: the caller's own, so the provider key and any proxy
    reach the real model, minus DecimalAI's (the child points DecimalAI at a probe), with a
    HOME of its own. Tracing on."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DECIMAL_", "DECIMALAI_"))}
    for key in _TRACING_OFF:
        env.pop(key, None)
    env.update({"HOME": str(home), "CREWAI_DISABLE_TELEMETRY": "true",
                "CREWAI_TRACING_ENABLED": "false", "DO_NOT_TRACK": "1",
                "PYTHONPATH": os.pathsep.join([str(REPO)] + [
                    p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])})
    return env


def run_scenario(target: str, ctx: Dict[str, Any], *, cwd: Path,
                 timeout: Optional[int] = None, live: bool = False) -> Dict[str, Any]:
    """Run ``target`` (``"tests.native.<module>:<function>"``) with ``ctx`` in a child
    whose working directory is ``cwd``; return what it returned. ``live`` gives the child
    the caller's network and provider keys (:func:`live_env`) instead of the hermetic
    environment."""
    cwd = Path(cwd).resolve()
    work = cwd.parent / f".native-child-{os.getpid()}-{abs(hash(target)) % 10**8}"
    home = work / "home"
    home.mkdir(parents=True, exist_ok=True)
    ctx_path, out_path = work / "ctx.json", work / "out.json"
    ctx_path.write_text(json.dumps(ctx, default=str))
    limit = timeout or int(os.environ.get(TIMEOUT_ENV) or DEFAULT_TIMEOUT)
    try:
        proc = subprocess.run(
            [sys.executable, str(CHILD), target, str(ctx_path), str(out_path)],
            cwd=cwd, env=live_env(home) if live else scenario_env(home),
            capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired as exc:
        raise ScenarioError(f"{target} did not finish in {limit}s\n"
                            f"{(exc.stderr or '')[-3000:]}") from None
    if not out_path.exists():
        raise ScenarioError(f"{target} wrote nothing (exit {proc.returncode})\n"
                            f"{proc.stderr[-4000:]}")
    result = json.loads(out_path.read_text())
    if isinstance(result, dict) and result.get("crashed"):
        raise ScenarioError(f"{target} raised {result['crashed']}\n{result.get('traceback', '')}"
                            f"\n--- stderr ---\n{proc.stderr[-2000:]}")
    return result
