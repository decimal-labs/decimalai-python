"""Consistency between what we SHIP and what we SAY about it.

Three drifts this file locks down, each of which was live in the tree:

  1. `SkillExportFileExistsError` told users to run `decimal registry
     install --force`. There is no `decimal` binary (pyproject declares
     exactly one console script, `decimalai`) and no `registry` command
     group. A user following the message hits "command not found".
  2. examples/demo_real_agent.py imported `decimalai.langchain.install`,
     which is a deprecation shim that emits DeprecationWarning — while the
     comment above it advertised the "NEW SDK SETUP".
  3. The CI langchain-compat matrix pinned a langchain-core floor
     (0.3.84) that `pip install -e ".[dev]"` can no longer resolve,
     because the pyproject extra floors it at 1.3.0.

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


# ── 1. The overwrite error names a command that actually exists ──────


def test_overwrite_error_names_a_real_console_script():
    from decimalai.disk_export import SkillExportFileExistsError

    msg = str(SkillExportFileExistsError("/tmp/x/SKILL.md"))
    scripts = _console_scripts()
    assert scripts == ["decimalai"], (
        f"expected exactly one console script `decimalai`; got {scripts}. "
        "If this changed deliberately, update the user-facing strings too."
    )
    # The message shows an example invocation; it must use the real binary.
    example = re.search(r"\(e\.g\., ([^)]+)\)", msg)
    assert example, f"no `(e.g., ...)` example in the message: {msg!r}"
    argv0 = example.group(1).split()[0]
    assert argv0 in scripts, (
        f"error message tells the user to run {argv0!r}, which is not a "
        f"declared console script ({scripts}); message={msg!r}"
    )
    # `registry` was never a command group — `skills` is.
    assert "registry install" not in msg, (
        f"message names the non-existent `registry` command group: {msg!r}"
    )


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
