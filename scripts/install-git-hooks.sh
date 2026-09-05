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
# findings that take a second to catch locally. This repo's local hooks ran only the
# secret gate, so ruff's verdict arrived from hosted CI minutes later, on a commit that
# was already pushed.
#
# It runs the EXACT arguments ci.yml's Lint job uses, read out of that file rather than
# copied, so the two cannot drift: change the CI command and this gate changes with it in
# the same commit. Bypass this hook alone with `git commit --no-verify` (the secret gate
# above has its own explicit, visible escape hatch).
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

args="$(python3 - <<'PY'
import re, sys
try:
    body = open(".github/workflows/ci.yml").read()
except OSError:
    sys.exit(0)
m = re.search(r"run:\s*(ruff check [^\n]+)", body)
print(m.group(1)[len("ruff check "):].strip() if m else "")
PY
)"
[ -z "$args" ] && args="decimalai/ --select I,E,W,F --ignore E501,E402,F821,F841"

# shellcheck disable=SC2086
if ! "$RUFF" check $args; then
  echo "[lint] ruff failed - the same check ci.yml's Lint job runs." >&2
  exit 1
fi
echo "[lint] ruff clean ($args)"
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
