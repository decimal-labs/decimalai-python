"""The native conformance column: one scenario per framework, graded by one contract.

Each framework whose own loader reads ``.agents/skills`` has a scenario module,
``tests/native/_scenario_<framework>.py``, with a ``column(ctx) -> dict`` function that runs
in a child process (``_runner.run_scenario``) with the project as its working directory. It
drives the REAL framework, its OWN loader (through DecimalAI's pointer helper) and a
scripted model, with DecimalAI's instrumentation pointed at a probe, through:

* **run 1**: the loader lists the skills (where it has a listing step); the model loads
  :data:`TARGET`, loads it **again**, reads :data:`RESOURCE` and runs :data:`SCRIPT` with
  :data:`SCRIPT_ARGS`, through the framework's own routes; then answers;
* :func:`update_to_v2` (``decimalai skills update`` replacing the target with version 2);
* **run 2**, with the same agent object in the same process: the model loads
  :data:`TARGET` and answers.

It returns :func:`Session.result` — no assertion lives in a scenario; ``column_contract``
grades every framework the same way. This module is the scenario side: it imports no
framework and asserts nothing.

The shape a scenario returns::

    {"framework": str,
     "versions": {distribution: version},
     "runs": [{"traces": [trace payload, ...],     # what the probe received in the run
               "received": [str, ...],              # every text the model was given, in order
               "target_loads": int}, ...],          # how many times the model loaded TARGET
     "na": {item: reason},                          # an item that cannot apply, and why
     "notes": {...}}
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from tests.native_loaders import oracle

SC = oracle.SCENARIO
TARGET = SC.target
RESOURCE = SC.resource
RESOURCE_MARKER = SC.resource_marker
SCRIPT = SC.script
SCRIPT_ARGS = tuple(SC.script_args)
SCRIPT_OUT = SC.script_out

#: How long to let the background sender finish after ``decimalai.flush()``.
SETTLE_SECONDS = 0.75


def project():
    """The project the scenario runs in (its working directory)."""
    from tests.native.conftest import NativeProject

    root = Path.cwd().resolve()
    lock = json.loads((root / ".decimal" / "skills.lock").read_text())
    return NativeProject(root=root, lock=lock)


def update_to_v2() -> Dict[str, str]:
    """Do what ``decimalai skills update`` does to the target; return v2's expected entry."""
    return project().update_to_v2()


def skills_dir() -> Path:
    return project().skills_dir


class Run:
    """What one run gave the model, as the scenario saw it."""

    def __init__(self) -> None:
        self.received: List[str] = []
        self.target_loads = 0
        #: ``(tool name, arguments)`` of every tool call the model made, in order (the live
        #: suite grades whether a real model called the loader).
        self.calls: List[Dict[str, Any]] = []
        #: The model's final answer.
        self.answer: Optional[str] = None

    def saw(self, text: Any) -> None:
        """Record a text the model was given (a tool result, a message)."""
        if text is None:
            return
        self.received.append(text if isinstance(text, str) else json.dumps(text, default=str))

    def loaded_target(self, times: int = 1) -> None:
        self.target_loads += times

    def called(self, name: Any, args: Any = None) -> None:
        """Record one tool call the model made."""
        self.calls.append({"name": str(name), "args": args if isinstance(args, (dict, list, str))
                           else json.dumps(args, default=str)})

    def answered(self, text: Any) -> None:
        self.answer = text if isinstance(text, str) or text is None else str(text)


class Session:
    """A probe, DecimalAI pointed at it, and the traces each run sent.

    ``init`` takes the ``decimalai.init`` flags the framework's documented native-mode
    setup uses (``langchain=True``, ``otel=True``, ...).
    """

    def __init__(self, framework: str, *, distributions: Iterable[str] = (),
                 **init: Any) -> None:
        import decimalai
        from tests.conformance.probe import Probe

        self.framework = framework
        self.distributions = list(distributions)
        from tests.conformance.isolation import CONFORMANCE_SKILLS

        self.probe = Probe().start()
        # Skills of the router's own, so a router that injected into the run would show:
        # a routing request and a routing id (NC7). The native loader serves the agent.
        self.probe.skills = [dict(skill) for skill in CONFORMANCE_SKILLS]
        os.environ["DECIMAL_API_KEY"] = self.probe.api_key
        os.environ["DECIMAL_BASE_URL"] = self.probe.base_url
        decimalai.init(api_key=self.probe.api_key, base_url=self.probe.base_url,
                       verify=False, **init)
        self.runs: List[Dict[str, Any]] = []
        self.na: Dict[str, str] = {}
        self.notes: Dict[str, Any] = {}

    def run(self, fn: Callable[[Run], None]) -> Run:
        """Run ``fn`` as one run of the agent and keep what reached the probe."""
        cursor = self.probe.mark()
        run = Run()
        error: Optional[str] = None
        try:
            fn(run)
        except Exception as exc:  # noqa: BLE001 - a failed run is a result, graded as such
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
        _flush()
        requests = self.probe.since(cursor)
        self.runs.append({
            "traces": _traces(requests),
            "route_calls": sum(1 for r in requests if r.path.split("?")[0].rstrip("/") in _ROUTING),
            "received": run.received,
            "target_loads": run.target_loads,
            "calls": run.calls,
            "answer": run.answer,
            "error": error,
        })
        return run

    def not_applicable(self, item: str, reason: str) -> None:
        self.na[item] = reason

    def result(self) -> Dict[str, Any]:
        self.probe.stop()
        return {"framework": self.framework, "versions": _versions(self.distributions),
                "runs": self.runs, "na": self.na, "notes": self.notes}


#: The router's routing endpoints on the probe: a request to either means it routed the run.
_ROUTING = ("/api/v1/skills/route", "/api/v1/skills/menu")


def _flush() -> None:
    import decimalai

    for _ in range(2):
        try:
            decimalai.flush()
        except Exception:  # noqa: BLE001 - flushing must not mask the run's result
            pass
        time.sleep(SETTLE_SECONDS)


def _traces(requests: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for req in requests:
        if req.method != "POST" or not req.accepted:
            continue
        if req.path.rstrip("/").endswith("/traces"):
            out.append(req.body)
        elif req.path.rstrip("/").endswith("/traces/batch"):
            body = req.body
            items = body.get("traces") if isinstance(body, dict) else body
            out.extend(t for t in (items or []) if isinstance(t, dict))
    return out


def _versions(distributions: Iterable[str]) -> Dict[str, str]:
    import decimalai

    # The module, not the installed metadata: a venv's editable install can carry an older
    # version number than the source on PYTHONPATH that is actually running.
    out = {"decimalai": decimalai.__version__}
    for dist in distributions:
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = "absent"
    return out
