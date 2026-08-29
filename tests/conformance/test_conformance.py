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
from .delivery import DELIVERY_MODES
from .drivers import DRIVER_MODULES, all_drivers
from .harness import Observation
from .journey import (
    SCAFFOLD_KEYS,
    JourneyCapture,
    journey_framework,
    journey_na_ledger,
    missing_requirements,
)
from .na_ledger import skip_declared
from .probe import BACKEND_VALIDATOR_SHA256

from decimalai.cli.scaffold import install_command

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
        skip_declared(
            "unavailable", f"{driver_name}:{item}",
            f"{driver_name}: driver dependencies are not installed",
        )
    result = contract.grade(item, obs)
    record(driver_name, result)
    if result.status == contract.NA:
        # Not a bare skip. An N/A is a hole in the matrix that reads like
        # coverage, so it has to be one the ledger already knows about.
        skip_declared(
            "na", f"{driver_name}:{item}",
            f"{item} N/A for {driver_name} — {result.message}",
        )

    key = f"{driver_name}:{item}"
    if key in BASELINE:
        assert result.status != contract.PASS, (
            f"{key} is listed in {BASELINE_PATH.name} but now PASSES — delete the "
            f"line so the next regression in this item is caught."
        )
        pytest.xfail(f"{key} known-red: {result.message}")

    assert result.status == contract.PASS, f"{item} {result.title}: {result.message}"


# ── the delivery matrix ──────────────────────────────────────────────────────

#: Only a framework with a rail has a body channel to vary. The four are
#: langchain, anthropic, openai-agents and pydantic-ai — the same four the SDK's
#: own scaffold ledger calls seam-carrying (``cli/scaffold.py``), cross-checked
#: by ``test_coverage.test_rail_declarations_match_the_scaffold_seam_ledger``.
RAIL_DRIVER_NAMES = [d.name for d in DRIVERS if d.capabilities.has_skills_rail]

#: Delivery cells that are known-red today, as ``driver:mode``. Same contract as
#: ``known_failures.txt`` and the same self-cleaning property: a listed cell that
#: starts PASSING fails the build, so the file shrinks instead of outliving the
#: defect. Recorded once, on the day the axis went live, against an adapter that
#: already had the defect — never as a pressure valve for a new one.
DELIVERY_BASELINE_PATH = Path(__file__).parent / "known_delivery_failures.txt"


def _delivery_baseline() -> set:
    if not DELIVERY_BASELINE_PATH.exists():
        return set()
    out = set()
    for line in DELIVERY_BASELINE_PATH.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


DELIVERY_BASELINE = _delivery_baseline()


@pytest.mark.parametrize("delivery_mode", DELIVERY_MODES)
@pytest.mark.parametrize("driver_name", RAIL_DRIVER_NAMES)
def test_delivery_channel(
    driver_name: str,
    delivery_mode: str,
    delivery_observations: Dict[tuple, Optional[Observation]],
) -> None:
    """Each body channel, on its own, with the other one switched off.

    C14 asks whether a body reached the model AT ALL. That closes "zero
    channels". This closes the half of it that comes next: one channel silently
    contributing nothing while the other covers for it. The defect that started
    all of this was arithmetic — ``inject_skill_body`` False AND no ``load_skill``
    tool — and each half looked defensible on its own.

    Both settings are public SDK surface, so every cell is a configuration a
    user can actually be in, and every cell is also a kill-switch test: the OFF
    channel must really be off.
    """
    obs = delivery_observations[(driver_name, delivery_mode)]
    if obs is None:
        skip_declared(
            "unavailable", f"{driver_name}:{delivery_mode}",
            f"{driver_name}: driver dependencies are not installed",
        )
    result = contract.grade_delivery(obs)
    record(driver_name, result)
    if result.status == contract.NA:
        skip_declared(
            "delivery_na", f"{driver_name}:{delivery_mode}",
            f"{delivery_mode} N/A for {driver_name} — {result.message}",
        )

    key = f"{driver_name}:{delivery_mode}"
    if key in DELIVERY_BASELINE:
        assert result.status != contract.PASS, (
            f"{key} is listed in {DELIVERY_BASELINE_PATH.name} but now PASSES — "
            f"delete the line so the next regression in this channel is caught."
        )
        pytest.xfail(f"{key} known-red: {result.message}")

    assert result.status == contract.PASS, (
        f"{result.item} {result.title}: {result.message}"
    )


# ── the journey ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("driver_name", DRIVER_NAMES)
def test_journey(
    driver_name: str, journey_captures: Dict[str, Optional[JourneyCapture]]
) -> None:
    """The whole path a user walks, hermetically, for every scaffoldable framework.

    An agent exists on the platform with a prompt and skills; ``decimalai init``
    writes a file for it; that file runs; the skill's knowledge is in front of
    the model. Every one of those steps is the real thing — the real CLI over
    real HTTP, the real generated source under ``runpy``, the real provider SDK
    against a local stub — and the only fixtures are the platform on one side
    (``probe.py``) and the model on the other (``journey.JourneyModel``).

    Its own item, deliberately. Every adapter item was green on 2026-08-28 while
    this was broken end to end, so "the adapter delivers a body when a driver
    hands it one" and "the product's own entry point produces something that
    works" must be able to fail separately and read differently.
    """
    framework = journey_framework(driver_name)
    if framework is None:
        # Not a bare skip, and not this suite's judgement either: the reason is
        # read out of the SDK's OWN scaffold ledger, and the ledger entry is
        # cross-checked against it by test_coverage.
        ledger = journey_na_ledger(driver_name)
        skip_declared(
            "journey_na", f"{driver_name}:{contract.JOURNEY_ITEM}",
            f"`decimalai init --framework {sorted(SCAFFOLD_KEYS[driver_name])[0]}` "
            f"writes no file: decimalai/cli/scaffold.py classifies it under "
            f"{ledger}, so there is no journey to walk",
        )
    missing = missing_requirements(framework)
    if missing:
        skip_declared(
            "unavailable", f"{driver_name}:{contract.JOURNEY_ITEM}",
            f"{driver_name}: the file `decimalai init --framework {framework}` "
            f"writes cannot run here — missing {', '.join(missing)}. Install what "
            f"the CLI itself prints: {install_command(framework)}",
        )
    capture = journey_captures[driver_name]
    if capture is None:  # pragma: no cover - the two branches above cover it
        skip_declared(
            "journey_na", f"{driver_name}:{contract.JOURNEY_ITEM}",
            f"{driver_name}: no journey was captured",
        )
    result = contract.grade_journey(capture)
    record(driver_name, result)
    assert result.status == contract.PASS, f"{result.item} {result.title}: {result.message}"


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
