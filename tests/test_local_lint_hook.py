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


def test_the_hook_lints_only_what_is_being_committed():
    """Staged content, never the tree.

    The first version checked all of `decimalai/`, and within hours it blocked one
    session's commit over ANOTHER session's half-finished edit in a file the committer
    had never opened — this worktree is shared by several sessions at once. A pre-commit
    hook that fails on work outside the commit is a hook people learn to bypass.
    """
    src = INSTALLER.read_text(encoding="utf-8")
    assert "git diff --cached --name-only" in src, (
        "the hook no longer selects staged files, so it lints other sessions' work"
    )
    assert "$STAGED" in src, "the staged file list is computed and then not used"


def test_the_hooks_flags_still_match_what_ci_runs():
    """The flags are literal in the hook — deriving them from the YAML at run time split
    `--select` from its value the first time it ran. Literal plus this test is the honest
    trade: drift fails here rather than silently in CI."""
    ci_cmd = re.search(r"run:\s*ruff check ([^\n]+)", CI.read_text(encoding="utf-8"))
    assert ci_cmd, "ci.yml no longer runs `ruff check` — update the hook and this test"
    ci_flags = [t for t in ci_cmd.group(1).split() if not t.startswith("decimalai")]

    hook_cmd = re.search(r'"\$RUFF" check ([^\n]*?) \$STAGED',
                         INSTALLER.read_text(encoding="utf-8"))
    assert hook_cmd, "the hook no longer invokes ruff with a literal flag list"
    assert hook_cmd.group(1).split() == ci_flags, (
        f"hook flags {hook_cmd.group(1).split()} != ci.yml flags {ci_flags}"
    )
