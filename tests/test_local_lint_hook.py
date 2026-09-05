"""The repo ships a local lint hook, and it runs the same command CI runs.

`Lint` failed on `main` twice on 2026-09-03 for findings ruff catches in under a
second; the only local hook was the secret gate, so the verdict arrived from hosted
CI minutes later on a commit already pushed. `scripts/install-git-hooks.sh` writes
the hook, and the hook reads its arguments out of `ci.yml` rather than copying them,
so the two cannot drift.

Static, on the installer: `.git/hooks/` is untracked, so a test that read the live
hook would pass on any machine where it was never installed — which is the hole.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "scripts" / "install-git-hooks.sh"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def test_the_installer_exists_and_writes_the_chained_hook():
    src = INSTALLER.read_text(encoding="utf-8")
    assert "pre-commit.pre-guard" in src, (
        "the hook must be written as pre-commit.pre-guard — commit-guard.sh owns "
        "pre-commit here and chains to that name; overwriting it would turn off the "
        "secret gate"
    )
    assert "ruff" in src


def test_the_hook_reads_its_arguments_from_ci_rather_than_copying_them():
    src = INSTALLER.read_text(encoding="utf-8")
    assert "ci.yml" in src and "ruff check" in src, (
        "the hook no longer derives its arguments from ci.yml, so local and CI lint "
        "can drift — which is how a green local commit fails Lint on main"
    )


def test_the_fallback_arguments_match_what_ci_runs_today():
    """The hook falls back to a literal list if ci.yml cannot be parsed. That literal
    is the thing most likely to go stale, so pin it to the real command."""
    ci_cmd = re.search(r"run:\s*ruff check ([^\n]+)", CI.read_text(encoding="utf-8"))
    assert ci_cmd, "ci.yml no longer runs `ruff check` — update the hook and this test"
    fallback = re.search(r'args="(decimalai/[^"]+)"', INSTALLER.read_text(encoding="utf-8"))
    assert fallback, "the hook has no literal fallback argument list"
    assert fallback.group(1).split() == ci_cmd.group(1).strip().split()
