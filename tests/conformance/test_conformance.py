"""contract items × drivers, plus the guards that keep the suite honest.

The parametrisation is the whole design in one line: every framework is graded
by the SAME fifteen functions. Nothing here knows what a framework is.

The *coverage* guards — "does every framework the product advertises have a
driver at all?" — live next door in ``test_coverage.py``, because they answer a
question about the set rather than about any adapter, and they run with no
framework installed.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from . import contract
from .conftest import record
from .drivers import DRIVER_MODULES, all_drivers
from .harness import Observation
from .probe import BACKEND_VALIDATOR_SHA256

#: Tier A — hermetic. No provider key, no platform backend, runs on every commit.
#: (Tier B is the same contract functions against a real backend; see README.)
pytestmark = pytest.mark.conformance

DRIVERS = all_drivers()
DRIVER_NAMES = [d.name for d in DRIVERS]

# The platform repo is a private sibling checkout that most clones will not have.
# Resolve it relative to this one and let the guard skip when it is absent, rather
# than hardcoding anybody's home directory.
_SIBLING = Path(__file__).resolve().parents[3]
BACKEND_TRACE_SERVICE = (
    _SIBLING / "platform" / "backend" / "app" / "services" / "trace_service.py"
)

# Contract items that are known-red today, as `driver:item`. The suite went live
# against adapters that already had defects, so gating on "everything green"
# would have meant gating on nothing — a red-on-day-one job gets switched off
# within a week. Same shape as scripts/lint_org_scoping_baseline.txt on the
# platform side: a NEW failure fails the build, and an entry here that starts
# PASSING also fails, so the list cleans itself up instead of outliving the
# defect. Shrink it; never add to it to make a build go green.
BASELINE_PATH = Path(__file__).parent / "known_failures.txt"


def _baseline() -> set:
    if not BASELINE_PATH.exists():
        return set()
    out = set()
    for line in BASELINE_PATH.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


BASELINE = _baseline()


# ── the matrix ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("item", contract.ITEM_ORDER)
@pytest.mark.parametrize("driver_name", DRIVER_NAMES)
def test_contract(
    driver_name: str, item: str, observations: Dict[str, Optional[Observation]]
) -> None:
    obs = observations[driver_name]
    if obs is None:
        pytest.skip(f"{driver_name}: driver dependencies are not installed")
    result = contract.grade(item, obs)
    record(driver_name, result)
    if result.status == contract.NA:
        pytest.skip(f"{item} N/A for {driver_name} — {result.message}")

    key = f"{driver_name}:{item}"
    if key in BASELINE:
        assert result.status != contract.PASS, (
            f"{key} is listed in {BASELINE_PATH.name} but now PASSES — delete the "
            f"line so the next regression in this item is caught."
        )
        pytest.xfail(f"{key} known-red: {result.message}")

    assert result.status == contract.PASS, f"{item} {result.title}: {result.message}"


# ── guards ───────────────────────────────────────────────────────────────────


def _backend_validator_fingerprint(source: str) -> str:
    """Hash the backend functions/constants ``probe.py`` mirrors.

    Deliberately narrow: the two validators plus the four allowlist/bound
    constants they read. Hashing the whole 200-line ``ingest_trace`` would churn
    on every unrelated edit and the guard would be turned off within a month.
    """
    tree = ast.parse(source)
    want_fn = {"validate_element_shapes", "_validate_payload"}
    want_const = {
        "_TRACE_STATUS_ALLOWED",
        "_TRACE_SOURCE_TYPE_ALLOWED",
        "_TRACE_TIMESTAMP_MAX_PAST_DAYS",
        "_TRACE_TIMESTAMP_MAX_FUTURE_DAYS",
    }
    chunks: List[tuple] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fn:
            chunks.append((node.name, ast.get_source_segment(source, node)))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in want_const:
                    chunks.append((target.id, ast.get_source_segment(source, node)))
    chunks.sort()
    blob = "\n".join(c[1] for c in chunks)
    return hashlib.sha256(blob.encode()).hexdigest()


def test_backend_validator_has_not_drifted() -> None:
    """The probe must reject exactly what the backend rejects.

    ``probe.validate_trace_payload`` is a hand port. If the backend's rules move
    and the port does not, the probe becomes more permissive than production and
    C2 silently stops catching the "the backend 400s this" defect class — the
    single most valuable thing this suite does. Skips (loudly) when the platform
    repo is not checked out beside this one.
    """
    if not BACKEND_TRACE_SERVICE.exists():
        pytest.skip(
            f"platform repo not on disk at {BACKEND_TRACE_SERVICE} — the port cannot "
            f"be checked here; it is checked wherever both repos are present"
        )
    actual = _backend_validator_fingerprint(BACKEND_TRACE_SERVICE.read_text())
    assert actual == BACKEND_VALIDATOR_SHA256, (
        "the backend's trace validator changed since probe.py was ported from it.\n"
        f"  expected {BACKEND_VALIDATOR_SHA256}\n"
        f"  actual   {actual}\n"
        "Re-read trace_service._validate_payload, update probe.validate_trace_payload "
        "to match, then record the new fingerprint in probe.BACKEND_VALIDATOR_SHA256."
    )


def test_drivers_contain_no_assertions() -> None:
    """A driver that asserts has stolen the contract's job.

    The single property this suite is built to preserve is that adding a
    framework means writing a driver, never writing an assertion. Enforce it
    structurally so it cannot erode one convenient exception at a time.
    """
    offenders: List[str] = []
    here = Path(__file__).parent / "drivers"
    for module in DRIVER_MODULES:
        path = here / f"{module}.py"
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                offenders.append(f"{path.name}:{node.lineno} assert statement")
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                )
                if any(n and n.split(".")[0] in {"pytest", "unittest"} for n in names):
                    offenders.append(f"{path.name}:{node.lineno} imports a test framework")
    assert not offenders, (
        "drivers must contain no assertions — move the check into contract.py so "
        f"every framework gets it: {offenders}"
    )


def test_every_capability_flag_gates_a_real_item() -> None:
    """A capability flag that gates nothing is a flag nobody can turn off safely."""
    from .drivers import CAPABILITY_ITEMS

    unknown = {
        item
        for items in CAPABILITY_ITEMS.values()
        for item in items
        if item not in contract.ITEMS
    }
    assert not unknown, f"CAPABILITY_ITEMS references non-existent contract items: {unknown}"
