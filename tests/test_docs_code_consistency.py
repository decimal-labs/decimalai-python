"""Consistency between what we SHIP and what we SAY about it.

Drifts this file locks down, the first two of which were live in the tree:

  1. An error told users to run `decimal registry install --force`. There is
     no `decimal` binary (pyproject declares exactly one console script,
     `decimalai`) and no `registry` command group. A user following the
     message hits "command not found".
  2. examples/demo_real_agent.py imported `decimalai.langchain.install`,
     which is a deprecation shim that emits DeprecationWarning — while the
     comment above it advertised the "NEW SDK SETUP".
  3. The CI langchain-compat matrix pinned a langchain-core floor
     (0.3.84) that `pip install -e ".[dev]"` can no longer resolve,
     because the pyproject extra floors it at 1.3.0.
  4. Every `decimalai skills <verb>` and `decimalai.skillpkg.<fn>()` the SDK's
     messages and docstrings name must exist, or the advice is a dead end.

These are cheap string/AST assertions on purpose: they cost nothing to run
and they catch the class of bug where the code moves and the prose doesn't.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _console_scripts() -> list[str]:
    """Names declared under [project.scripts] in pyproject.toml."""
    text = PYPROJECT.read_text()
    block = re.search(r"^\[project\.scripts\]\s*$(.*?)(?=^\[|\Z)", text,
                      re.MULTILINE | re.DOTALL)
    assert block, "no [project.scripts] table in pyproject.toml"
    return re.findall(r"^([A-Za-z0-9_.-]+)\s*=", block.group(1), re.MULTILINE)


# ── 1. The SDK has exactly one console script ──────────────────────


def test_the_one_console_script_is_decimalai():
    """Messages tell users to run `decimalai ...`; there is no other binary."""
    assert _console_scripts() == ["decimalai"]


# ── 2. Examples use instrument(), not the deprecated install() shim ──


def _example_scripts() -> list[Path]:
    return sorted((REPO_ROOT / "examples").rglob("*.py"))


def test_examples_do_not_use_the_deprecated_langchain_install_shim():
    """`decimalai.langchain.install` warns DeprecationWarning; an example
    that a user copy-pastes must not hand them a deprecated call."""
    offenders = []
    for path in _example_scripts():
        src = path.read_text()
        if re.search(r"^\s*from\s+decimalai\.langchain\s+import\s+.*\binstall\b",
                     src, re.MULTILINE):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "examples import the deprecated decimalai.langchain.install shim "
        f"(use instrument instead): {offenders}"
    )


def test_demo_real_agent_calls_instrument():
    path = REPO_ROOT / "examples" / "demo_real_agent.py"
    src = path.read_text()
    assert "from decimalai.langchain import instrument" in src
    assert "instrument(agent_name=AGENT_NAME)" in src
    # The confirmation print must name the function the script actually calls.
    assert "via install()" not in src, (
        "demo prints that tracing was installed 'via install()' but calls instrument()"
    )


# ── 3. CI's langchain-core floor == the pyproject floor ──────────────


def _pyproject_langchain_core_floor() -> str:
    text = PYPROJECT.read_text()
    m = re.search(r'"langchain-core>=([0-9][^,"]*)', text)
    assert m, "no langchain-core floor found in pyproject.toml"
    return m.group(1)


def test_ci_langchain_matrix_tests_the_declared_floor():
    ci = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    if not ci.exists():  # pragma: no cover - repo always ships it
        pytest.skip("no ci.yml in this checkout")
    m = re.search(r"^\s*langchain-core:\s*\[(.+?)\]\s*$",
                  ci.read_text(), re.MULTILINE)
    assert m, "no `langchain-core:` matrix in .github/workflows/ci.yml"
    matrix = [v.strip().strip('"\'') for v in m.group(1).split(",")]
    floor = _pyproject_langchain_core_floor()
    assert floor in matrix, (
        f"CI langchain-compat matrix is {matrix}, which does not test the "
        f"pyproject floor {floor!r} — CI is verifying a version the extra "
        "can no longer resolve."
    )
    assert "latest" in matrix, f"CI matrix should also test `latest`; got {matrix}"


# ── 4. The SDK names commands and functions that exist ──────────────

#: The `decimalai skills` verbs of the package manager.
SKILLS_VERBS = {"add", "install", "update", "remove", "status", "verify", "migrate", "pull",
                "list", "sync", "scan", "review", "benchmark", "push"}


def _sdk_sources() -> str:
    return "".join(path.read_text(encoding="utf-8")
                   for path in sorted((REPO_ROOT / "decimalai").rglob("*.py")))


def test_messages_name_real_skills_verbs():
    named = set(re.findall(r"`decimalai skills ([a-z][a-z-]*)", _sdk_sources()))
    assert {"add", "install", "update", "migrate"} <= named, named
    assert named <= SKILLS_VERBS, named - SKILLS_VERBS


def test_messages_name_real_python_functions():
    from decimalai import skillpkg

    for name in set(re.findall(r"decimalai\.skillpkg\.(\w+)\(", _sdk_sources())):
        assert callable(getattr(skillpkg, name, None)), name
