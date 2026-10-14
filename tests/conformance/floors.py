"""The SDK's declared framework floors, so the suite never grades below one.

Every framework extra in ``pyproject.toml`` floors that framework's newest
release, and the SDK carries no code for anything older ("Which versions we
support" in the README). A driver run against an older release therefore grades
an adapter on a framework it no longer claims to support: a red cell there is a
version the SDK dropped on purpose, and a green one is luck.

That is not hypothetical. A test environment can hold a framework it was never
built to carry — one pulled in as somebody else's dependency, or left behind by
an older install — and a driver runs wherever its imports happen to resolve.
openai-agents 0.20.0 sat in an openai 2 environment (openai-agents needs openai 3
from 0.21) and its column was graded there, below the 0.23.1 floor.

So a driver names the distributions whose floors gate it (``Driver.frameworks``),
and one installed below its floor makes the driver unavailable, with the reason.
That is graded like any other missing framework: it skips, naming where the
driver IS graded, and under ``DECIMAL_CONFORMANCE_REQUIRE_ALL=1`` it FAILS in an
environment the driver belongs to — which is how a stale environment is caught
rather than graded.

The floors are read from ``pyproject.toml`` itself, never retyped, so a floor
raised there moves the gate on the same commit.
"""

from __future__ import annotations

import re
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional

#: The checkout's pyproject.toml — the single source for every floor.
PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_lines(text: str) -> List[str]:
    """Every quoted requirement in ``[project.optional-dependencies]``.

    Hand-parsed for the reason ``test_coverage.declared_extras`` gives: Python
    3.10 has no ``tomllib`` and no TOML parser is a test dependency. Comments
    are dropped before quotes are read, so a floor mentioned in prose is not
    mistaken for one declared.
    """
    lines = text.splitlines()
    try:
        start = lines.index("[project.optional-dependencies]") + 1
    except ValueError:
        return []
    out: List[str] = []
    for line in lines[start:]:
        if line.startswith("["):
            break
        out += re.findall(r'"([^"]+)"', line.split("#", 1)[0])
    return out


@lru_cache(maxsize=1)
def declared_floors() -> Dict[str, str]:
    """distribution (canonical name) -> the highest ``>=`` floor any extra declares.

    Empty when pyproject is not on disk, which leaves every driver as available
    as its imports make it — the behaviour before floors were read at all.
    """
    from packaging.requirements import InvalidRequirement, Requirement
    from packaging.version import Version

    try:
        text = PYPROJECT.read_text()
    except OSError:
        return {}
    floors: Dict[str, Version] = {}
    for raw in _requirement_lines(text):
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        for spec in req.specifier:
            if spec.operator in (">=", "==", "~="):
                version = Version(spec.version)
                name = _canonical(req.name)
                if name not in floors or version > floors[name]:
                    floors[name] = version
    return {name: str(version) for name, version in floors.items()}


def below_floor(distribution: str) -> Optional[str]:
    """Why ``distribution`` is too old to grade, or None when it is fine.

    None also when it is not installed at all (the driver's imports report
    that) or when pyproject declares no floor for it.
    """
    from packaging.version import Version

    floor = declared_floors().get(_canonical(distribution))
    if floor is None:
        return None
    try:
        installed = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None
    if Version(installed) < Version(floor):
        return (
            f"{distribution} {installed} is installed, below the SDK's floor "
            f"{floor} (pyproject.toml) — the adapter does not support it, so "
            f"nothing graded against it would mean anything"
        )
    return None
