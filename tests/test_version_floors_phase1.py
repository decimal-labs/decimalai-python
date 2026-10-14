"""The extras must admit the framework versions the suite actually runs.

pyproject's optional-dependencies capped `anthropic<1.0.0` and
`openinference-instrumentation-anthropic<2.0.0` long after the failure that
prompted the caps was fixed upstream: the openai 3 environment runs anthropic
1.13.0 with instrumentor 3.0.3, and the anthropic conformance column passes
there. A cap that excludes the version under test is a claim nobody can run —
`pip install "decimalai[providers-tests]"` would have resolved a DIFFERENT
anthropic from the one every test here exercised.

Upper bounds only. A floor above the installed version means this environment is
stale, which is the floors job's business (ci.yml `floors`, installed at
`--resolution lowest-direct`), not a property of the extras.

Runs in both test environments (it is in tests/conftest.py's openai 2 slice), and
checks only distributions that are installed here — each environment holds a
different half of the frameworks.
"""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path
from typing import Dict, List

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"


def optional_dependencies(text: str) -> Dict[str, List[str]]:
    """``[project.optional-dependencies]`` as {extra: [requirement strings]}.

    Hand-parsed: Python 3.10 (still in the CI matrix) has no ``tomllib``, and no
    TOML parser is a declared test dependency. The section is a flat run of
    ``name = [ "...", ... ]`` lists, which is all this has to read.
    """
    lines = text.splitlines()
    start = lines.index("[project.optional-dependencies]") + 1
    out: Dict[str, List[str]] = {}
    current = None
    for line in lines[start:]:
        if line.startswith("["):
            break
        m = re.match(r"^([A-Za-z0-9._-]+)\s*=\s*\[(.*)$", line)
        if m:
            current = m.group(1)
            out[current] = re.findall(r'"([^"]+)"', m.group(2).split("#", 1)[0])
            if "]" in m.group(2).split("#", 1)[0]:
                current = None
            continue
        if current is None:
            continue
        code = line.split("#", 1)[0]
        out[current] += re.findall(r'"([^"]+)"', code)
        if code.strip().startswith("]"):
            current = None
    return out


def installed_version(name: str):
    try:
        return Version(metadata.version(name))
    except metadata.PackageNotFoundError:
        return None


def cap_violations(extras: Dict[str, List[str]]) -> List[str]:
    """``extra: requirement`` lines whose UPPER bound excludes what is installed."""
    env = default_environment()
    bad: List[str] = []
    for extra, reqs in sorted(extras.items()):
        for raw in reqs:
            req = Requirement(raw)
            if req.name.lower() == "decimalai":
                continue  # a self-reference; its own extra's lines are checked there
            if req.marker is not None and not req.marker.evaluate({**env, "extra": extra}):
                continue
            have = installed_version(req.name)
            if have is None:
                continue
            uppers = [s for s in req.specifier if s.operator in ("<", "<=", "~=", "==")]
            if any(not s.contains(have, prereleases=True) for s in uppers):
                bad.append(f"[{extra}] {raw!r} excludes the installed {req.name} {have}")
    return bad


def test_the_parser_reads_every_extra():
    extras = optional_dependencies(PYPROJECT.read_text())
    # A few that must always be there, with at least one requirement each — a
    # parser that silently reads nothing would make the cap check vacuous.
    for name in ("langchain", "openai", "dev", "conformance-tests", "providers-tests"):
        assert extras.get(name), f"parsed no requirements for [{name}]: {extras.get(name)!r}"
    assert sum(len(v) for v in extras.values()) > 40


def test_no_extra_caps_below_the_installed_version():
    bad = cap_violations(optional_dependencies(PYPROJECT.read_text()))
    assert not bad, (
        "pyproject.toml caps a dependency below the version this environment runs:\n  "
        + "\n  ".join(bad)
        + "\n\nThe suite and the conformance matrix grade the installed version, so "
        "the extra is promising users something nobody tested. Lift the cap if the "
        "installed version passes conformance; if it does not, fix the adapter. A "
        "cap stays only while it excludes a version that is MEASURED to fail."
    )


def test_cap_violations_flags_a_stale_cap(monkeypatch):
    """The check itself bites: the pre-2026-10-14 anthropic caps are flagged
    whenever a newer anthropic is what is installed."""
    import sys

    module = sys.modules[__name__]
    monkeypatch.setattr(
        module, "installed_version",
        lambda name: {"anthropic": Version("1.13.0"),
                      "openinference-instrumentation-anthropic": Version("3.0.3"),
                      "openai": Version("3.30.0")}.get(name),
    )
    stale = {"providers-tests": [
        "openai>=2.26.0",
        "anthropic>=0.40.0,<1.0.0",
        "openinference-instrumentation-anthropic>=0.1.14,<2.0.0",
    ]}
    flagged = cap_violations(stale)
    assert len(flagged) == 2, flagged
    assert any("anthropic 1.13.0" in f for f in flagged)
    assert any("openinference-instrumentation-anthropic 3.0.3" in f for f in flagged)
