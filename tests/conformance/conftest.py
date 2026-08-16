"""Fixtures and the end-of-run conformance matrix.

One full set of driver phases per session — the runs are the expensive part, the
grading is not, so every contract item reads the SAME capture rather than
re-running the framework thirteen times.

Each driver's phases run in their OWN process (``isolation.py``), because every
adapter installs process-global instrumentation that whichever driver went first
could otherwise impose on the ones after it. Only the capture comes back; the
grading still happens here, in the parent, against ``contract.py``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from .contract import FAIL, NA, PASS, Result
from .drivers import Driver, all_drivers
from .harness import Observation
from .isolation import CONFORMANCE_SKILLS, DriverProcessError, run_drivers

#: Set this to a path and the matrix is also written there as JSON, for a caller
#: that needs the result structurally rather than as terminal text. The release
#: gate sets it, so its report can name which drivers did not run — the one way
#: this tier goes green while grading nothing. Deliberately opt-in: an
#: unconditional write would put a file in somebody's cwd, which C11 exists to
#: forbid.
REPORT_JSON_ENV = "DECIMAL_CONFORMANCE_REPORT_JSON"

#: Skills the probe's router offers when a driver declares a skills rail. Defined
#: in ``isolation`` (the child needs it too) and re-exported here, where it has
#: always lived.
__all__ = ["CONFORMANCE_SKILLS", "observations", "record"]

#: (driver, item, Result) as graded, for the terminal matrix.
_REPORT: List[Tuple[str, Result]] = []
#: driver name -> why it did not run at all.
_UNAVAILABLE: Dict[str, str] = {}
#: driver name -> why its process died. Distinct from _UNAVAILABLE on purpose: a
#: missing dependency is an expected local condition, a dead child is a defect.
_CRASHED: Dict[str, str] = {}


def record(driver_name: str, result: Result) -> None:
    _REPORT.append((driver_name, result))


class _Observations(Dict[str, Optional[Observation]]):
    """The captures, with a crashed driver kept LOUD.

    A driver whose process died has no capture. Returning ``None`` for it would
    make every one of its items skip — thirteen quiet skips reading as "not
    installed" — so the lookup raises instead, and the driver's name is in the
    message.
    """

    def __init__(
        self,
        data: Dict[str, Optional[Observation]],
        failures: Dict[str, str],
    ) -> None:
        super().__init__(data)
        self.failures = failures

    def __getitem__(self, key: str) -> Optional[Observation]:
        if key in self.failures:
            raise DriverProcessError(self.failures[key])
        return super().__getitem__(key)


def _selected_driver_names(session: pytest.Session) -> set:
    """Which drivers this session actually collected items for.

    Under ``-k langchain`` there is no reason to run the other ten frameworks;
    with each driver in its own process there is no longer any reason to pretend
    there is. Order is NOT taken from here — see below.
    """
    names = set()
    for item in session.items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        name = params.get("driver_name")
        if isinstance(name, str):
            names.add(name)
    return names


@pytest.fixture(scope="session")
def observations(request: pytest.FixtureRequest) -> Dict[str, Optional[Observation]]:
    """Run each selected driver's phases once, in its OWN process.

    ``None`` == driver unavailable (its imports are missing). A driver whose
    process crashed is neither ``None`` nor a capture: looking it up raises.
    """
    wanted = _selected_driver_names(request.session)
    out: Dict[str, Optional[Observation]] = {}
    runnable: List[Driver] = []
    # Registry order, always — never selection order, never sorted. Order must
    # not be a knob the caller can turn, or "it passes if you run it this way"
    # comes back in a different shape.
    for driver in all_drivers():
        if driver.name not in wanted:
            continue
        if not driver.available:
            _UNAVAILABLE[driver.name] = (
                f"missing import(s): {', '.join(driver.missing_requirements)}"
            )
            out[driver.name] = None
            continue
        runnable.append(driver)

    observed, failures = run_drivers(runnable)
    out.update(observed)
    for name, why in failures.items():
        _CRASHED[name] = why
    return _Observations(out, failures)


def _cell(result: Result) -> str:
    return {PASS: "PASS", FAIL: "FAIL", NA: "N/A "}[result.status]


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    if not _REPORT and not _UNAVAILABLE and not _CRASHED:
        return
    write = terminalreporter.write_line
    write("")
    write("=" * 78)
    write("CONFORMANCE MATRIX — one contract, applied to every framework")
    write("=" * 78)
    by_driver: Dict[str, List[Result]] = {}
    for name, result in _REPORT:
        by_driver.setdefault(name, []).append(result)
    for name, results in sorted(by_driver.items()):
        passed = sum(r.status == PASS for r in results)
        failed = sum(r.status == FAIL for r in results)
        na = sum(r.status == NA for r in results)
        write("")
        write(f"{name}:  {passed} pass  {failed} FAIL  {na} n/a")
        for r in results:
            message = r.message if len(r.message) <= 240 else r.message[:237] + "..."
            write(f"  {_cell(r)}  {r.item:<4} {r.title:<24} {message}")
    for name, why in sorted(_UNAVAILABLE.items()):
        write("")
        write(f"{name}:  NOT RUN — {why}")
    for name, why in sorted(_CRASHED.items()):
        write("")
        write(f"{name}:  NOT GRADED — the driver process died:")
        for line in why.splitlines():
            write(f"    {line}")
    write("=" * 78)
    _write_report_json(by_driver)


def _write_report_json(by_driver: Dict[str, List[Result]]) -> None:
    """Mirror the matrix to JSON when the caller asked for it. Never fatal.

    A reporting side-effect must not be able to fail a conformance run: the
    matrix on screen is the primary artefact, this is a convenience for the
    release gate.
    """
    dest = os.environ.get(REPORT_JSON_ENV)
    if not dest:
        return
    payload = {
        "drivers": {
            name: {
                "pass": sum(r.status == PASS for r in results),
                "fail": sum(r.status == FAIL for r in results),
                "na": sum(r.status == NA for r in results),
                "items": {r.item: r.status for r in results},
                "failed_items": [r.item for r in results if r.status == FAIL],
            }
            for name, results in sorted(by_driver.items())
        },
        # The load-bearing half: a driver here was never graded, so every item it
        # "passed" is a skip. A report that omits this reads as full coverage.
        # A crashed child belongs here for the same reason it belongs in the
        # matrix — it was not graded — but keeps its own key so a reader cannot
        # mistake a dead process for an uninstalled framework.
        "not_run": dict(sorted(_UNAVAILABLE.items())),
        "crashed": dict(sorted(_CRASHED.items())),
    }
    try:
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    except Exception:  # pragma: no cover - reporting must never break the run
        pass
