"""Fixtures and the end-of-run conformance matrix.

One probe and one full set of driver phases per session — the runs are the
expensive part, the grading is not, so every contract item reads the SAME
capture rather than re-running the framework thirteen times.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from .contract import FAIL, NA, PASS, Result
from .drivers import all_drivers
from .harness import Observation, observe
from .probe import Probe

#: Set this to a path and the matrix is also written there as JSON, for a caller
#: that needs the result structurally rather than as terminal text. The release
#: gate sets it, so its report can name which drivers did not run — the one way
#: this tier goes green while grading nothing. Deliberately opt-in: an
#: unconditional write would put a file in somebody's cwd, which C11 exists to
#: forbid.
REPORT_JSON_ENV = "DECIMAL_CONFORMANCE_REPORT_JSON"

#: Skills the probe's router offers when a driver declares a skills rail.
CONFORMANCE_SKILLS: Tuple[Dict[str, str], ...] = (
    {
        "name": "conformance-skill-alpha",
        "description": "Alpha skill offered by the conformance probe.",
        "body": "# Alpha\n\nAlpha guidance for the conformance run.",
    },
    {
        "name": "conformance-skill-beta",
        "description": "Beta skill offered by the conformance probe.",
        "body": "# Beta\n\nBeta guidance for the conformance run.",
    },
)

#: (driver, item, Result) as graded, for the terminal matrix.
_REPORT: List[Tuple[str, Result]] = []
#: driver name -> why it did not run at all.
_UNAVAILABLE: Dict[str, str] = {}


def record(driver_name: str, result: Result) -> None:
    _REPORT.append((driver_name, result))


@pytest.fixture(scope="session")
def observations() -> Dict[str, Optional[Observation]]:
    """Run every available driver's phases once. None == driver unavailable."""
    out: Dict[str, Optional[Observation]] = {}
    for driver in all_drivers():
        if not driver.available:
            _UNAVAILABLE[driver.name] = (
                f"missing import(s): {', '.join(driver.missing_requirements)}"
            )
            out[driver.name] = None
            continue
        probe = Probe().start()
        try:
            out[driver.name] = observe(driver, probe, skills=CONFORMANCE_SKILLS)
        finally:
            probe.stop()
    return out


def _cell(result: Result) -> str:
    return {PASS: "PASS", FAIL: "FAIL", NA: "N/A "}[result.status]


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ARG001
    if not _REPORT and not _UNAVAILABLE:
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
        "not_run": dict(sorted(_UNAVAILABLE.items())),
    }
    try:
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2))
    except Exception:  # pragma: no cover - reporting must never break the run
        pass
