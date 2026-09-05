#!/usr/bin/env bash
# Install the repo's local lint hook. Re-run any time; idempotent.
#
# `Lint` failed on `main` twice on 2026-09-03 for import-order and unused-import findings
# that ruff catches in under a second. The only local hook here was the secret gate, so
# ruff's verdict arrived from hosted CI minutes later, on a commit already pushed.
#
# The hook is written as `pre-commit.pre-guard` because commit-guard.sh owns `pre-commit`
# in this repo and chains to that name — so both gates stay live and both installers stay
# idempotent. If commit-guard is not installed here, this also drops a `pre-commit` that
# execs it, so the lint gate works on its own.
#
# Bypass: `git commit --no-verify` (this hook only).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .git/hooks

cat > .git/hooks/pre-commit.pre-guard <<'HOOK'
#!/usr/bin/env bash
# Lint gate for decimalai-python, chained beneath commit-guard.sh.
#
# WHY: `Lint` failed on `main` twice on 2026-09-03 for import-order and unused-import
# findings ruff catches in under a second. The only local hook was the secret gate, so
# ruff's verdict arrived from hosted CI minutes later, on a commit already pushed.
#
# IT LINTS WHAT IS BEING COMMITTED, NOT THE TREE. This worktree is shared by several
# sessions at once. The first version checked all of `decimalai/` and within hours blocked
# one session's commit over ANOTHER session's half-finished edit, in a file the committer
# had never opened. A pre-commit hook that fails on work outside the commit is a hook
# people learn to bypass. Staged content only.
#
# The flags are literal, and `tests/test_local_lint_hook.py` asserts they still equal the
# ones ci.yml's Lint job passes — parsing them out of the YAML at run time looked tidier
# and split `--select` from its value the first time it ran.
#
# Bypass this hook alone with `git commit --no-verify` (the secret gate above has its own
# explicit, visible escape hatch).
set -uo pipefail
root="$(git rev-parse --show-toplevel)"
cd "$root" || exit 0

RUFF=""
for cand in "$root/.venv/bin/ruff" "$(command -v ruff || true)"; do
  [ -n "$cand" ] && [ -x "$cand" ] && { RUFF="$cand"; break; }
done
if [ -z "$RUFF" ]; then
  echo "[lint] ruff not found (.venv/bin/ruff or PATH) - skipping; CI still runs it." >&2
  exit 0
fi

# Staged, still-present .py files under the package ci.yml lints.
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '^decimalai/.*\.py$' || true)
if [ -z "$STAGED" ]; then
  echo "[lint] no staged files under decimalai/ - nothing for ruff to check."
  exit 0
fi

# shellcheck disable=SC2086
if ! "$RUFF" check --select I,E,W,F --ignore E501,E402,F821,F841 $STAGED; then
  echo "[lint] ruff failed on the files you are committing - the same checks ci.yml runs." >&2
  exit 1
fi
echo "[lint] ruff clean ($(printf '%s\n' "$STAGED" | wc -l | tr -d ' ') staged file(s))"
HOOK
chmod +x .git/hooks/pre-commit.pre-guard

if [ ! -f .git/hooks/pre-commit ]; then
  printf '#!/usr/bin/env bash\nexec "$(git rev-parse --git-dir)/hooks/pre-commit.pre-guard" "$@"\n' \
    > .git/hooks/pre-commit
  chmod +x .git/hooks/pre-commit
  echo "installed .git/hooks/pre-commit -> pre-commit.pre-guard (ruff, the ci.yml Lint command)"
else
  echo "installed .git/hooks/pre-commit.pre-guard (ruff, the ci.yml Lint command)"
  grep -q 'pre-commit.pre-guard' .git/hooks/pre-commit \
    || echo "  NOTE: .git/hooks/pre-commit exists and does not chain to it — check it." >&2
fi
