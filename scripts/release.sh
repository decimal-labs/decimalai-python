#!/usr/bin/env bash
# Release decimalai to PyPI.
#
# Usage:  ./scripts/release.sh                      (run from anywhere inside the repo)
#         SKIP_LIVE_LLM_GATE=1 ./scripts/release.sh (skip the live-LLM gate when every
#                                                    provider is quota-blocked — see below)
#
# This script runs every gate a release must pass, then (after a typed
# confirmation) uploads to PyPI with twine and tags the GitHub Release as a
# record of what shipped. See RELEASING.md for why the upload is local.
#
# What it enforces before that irreversible upload:
#   1. the version is new (PyPI is append-only), un-tagged locally, and consistent
#      across pyproject.toml / decimalai.__version__;
#   2. the package builds, passes `twine check`, and the built wheel imports +
#      its CLI runs in a throwaway environment  (cheap — runs before any spend);
#   3. the LIVE-LLM release gate is green — real model calls through a clean-room
#      wheel. This is the gate CI deliberately does NOT run (no keys/backend in
#      CI); running it here is the whole point of a local required release step.
#      Bypass with SKIP_LIVE_LLM_GATE=1 when the gate harness isn't available or
#      every provider is quota-blocked (429/RESOURCE_EXHAUSTED) and the release
#      must still ship — gates 1+2 still run.
#
# Prerequisites: `uv`, `gh` (authed), a reachable DecimalAI backend
# (DECIMAL_BACKEND_URL), a provider key exported in your environment, and — for
# gate 3 only — the live-gate harness at DECIMAL_RELEASE_GATE_DIR. See RELEASING.md.
set -euo pipefail

NAME="decimalai"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Which providers the live gate exercises. Defaults to google because the OpenAI
# test key is quota-blocked; set RELEASE_GATE_PROVIDERS="google openai" once it
# has quota.
PROVIDERS="${RELEASE_GATE_PROVIDERS:-google}"
# The live-gate harness (clean-room wheel + real-model matrix) is maintainer-only
# tooling that lives outside this repo — no default path, point the env var at
# your checkout. Unset means gate 3 cannot run; see the check below.
GATE_DIR="${DECIMAL_RELEASE_GATE_DIR:-}"
BACKEND_URL="${DECIMAL_BACKEND_URL:-http://localhost:8000}"

# --- resolve the package version -------------------------------------------
# pyproject.toml is the source of truth; decimalai/__init__.py hardcodes a
# matching __version__ (the clean-env smoke test below proves they agree).
VERSION="$(grep -m1 '^version = ' pyproject.toml | sed -E 's/.*"([^"]+)".*/\1/')"
if [[ -z "$VERSION" ]]; then
  echo "ERROR: could not read version from pyproject.toml" >&2; exit 1
fi
echo "==> Releasing $NAME $VERSION (live gate providers: $PROVIDERS)"

# --- refuse to clobber an existing release (PyPI is append-only) -----------
HTTP="$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/$NAME/$VERSION/json" || echo 000)"
if [[ "$HTTP" == "200" ]]; then
  echo "ERROR: $NAME $VERSION already exists on PyPI — a version can never be reused." >&2
  echo "       bump 'version' in pyproject.toml AND __version__ in decimalai/__init__.py." >&2
  exit 1
fi

# --- refuse to clobber an existing tag -------------------------------------
if git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null; then
  echo "ERROR: tag v$VERSION already exists locally." >&2; exit 1
fi

# --- the release commit must already be on the remote ----------------------
# gh cuts the Release (and tag) on the remote; releasing an unpushed commit would
# silently tag the wrong revision. Push your intended release commit first.
if ! git branch -r --contains HEAD 2>/dev/null | grep -q .; then
  echo "ERROR: HEAD ($(git rev-parse --short HEAD)) is not on any remote branch." >&2
  echo "       push your release commit first:  git push" >&2
  exit 1
fi

# --- changelog reminder (advisory; the file may not exist yet) -------------
if [[ -f CHANGELOG.md ]] && ! grep -q "$VERSION" CHANGELOG.md; then
  echo "WARNING: no '$VERSION' entry found in CHANGELOG.md" >&2
fi

# --- build + validate (cheap; before we spend any model budget) ------------
rm -rf dist
uv build
uvx twine check dist/*

# --- smoke-test the built wheel in a throwaway environment -----------------
# Asserting against $VERSION (from pyproject) also proves __init__.py agrees.
wheels=(dist/*.whl); WHEEL="${wheels[0]}"
echo "==> Smoke-testing $WHEEL"
uv run --no-project --with "$WHEEL" -- python -c "
import decimalai as m
assert m.__version__ == '$VERSION', f'wheel reports {m.__version__}, expected $VERSION'
print('  import OK | __version__ =', m.__version__)
"
uv run --no-project --with "$WHEEL" -- decimalai --help >/dev/null
echo "  CLI entry point OK"

# --- the LIVE-LLM gate (real models; the T2 check CI cannot run) -----------
# Skippable via SKIP_LIVE_LLM_GATE=1 — the escape hatch for when EVERY provider is
# quota-blocked (429 / RESOURCE_EXHAUSTED) and a release must still ship. The cheap
# structural gates above (build, twine check, clean-env import + CLI, version
# parity) ALWAYS run; only the real-model T2 check is bypassed. Use sparingly: it
# ships without proving the SDK against live providers. See RELEASING.md
# ("Skipping the live gate"). Default is to RUN the gate.
if [[ "${SKIP_LIVE_LLM_GATE:-0}" == "1" ]]; then
  echo "⚠ SKIP_LIVE_LLM_GATE=1 — bypassing the live-LLM release gate."
  echo "  Structural gates passed (build/twine/import/CLI/version); the real-model"
  echo "  T2 check did NOT run. Shipping $NAME $VERSION without live validation"
  echo "  (typically: all providers quota-blocked)."
else
  if [[ -z "$GATE_DIR" || ! -d "$GATE_DIR" ]]; then
    echo "ERROR: live-gate harness not found. Point DECIMAL_RELEASE_GATE_DIR at it," >&2
    echo "       or set SKIP_LIVE_LLM_GATE=1 to release without the live gate." >&2
    exit 1
  fi
  HEALTH="$(curl -s -o /dev/null -w '%{http_code}' "$BACKEND_URL/health" || echo 000)"
  if [[ "$HEALTH" != "200" ]]; then
    echo "ERROR: backend not healthy at $BACKEND_URL/health (got $HEALTH)." >&2
    echo "       start a DecimalAI backend and set DECIMAL_BACKEND_URL to it" >&2
    echo "       (or set SKIP_LIVE_LLM_GATE=1 to release without the live gate)" >&2
    exit 1
  fi
  echo "==> Live-LLM release gate (clean-room wheel, providers: $PROVIDERS)"
  make -C "$GATE_DIR" release-gate-cleanroom ARGS="--providers $PROVIDERS"
fi

# --- confirm, then the one irreversible step -------------------------------
echo
if [[ "${SKIP_LIVE_LLM_GATE:-0}" == "1" ]]; then
  echo "Structural gates green for $NAME $VERSION (live-LLM gate SKIPPED)."
else
  echo "All gates green for $NAME $VERSION."
fi
echo "The next step UPLOADS $NAME $VERSION to PyPI from this machine, then tags the"
echo "GitHub Release for the record. A PyPI version can never be replaced or reused."
read -r -p "Type 'yes' to publish: " ANS
[[ "$ANS" == "yes" ]] || { echo "Aborted — nothing released."; exit 1; }

# ── PUBLISH LOCALLY FIRST, then tag. ────────────────────────────────────────
#
# This used to cut a GitHub Release and let publish.yml upload via OIDC, which made
# every release depend on hosted CI being available. When CI was unavailable the
# release could not happen at all — and a CI job that fails with ZERO steps executed
# and NO logs looks like a broken test, so the failure is easy to misdiagnose and
# expensive to chase.
#
# The gates above already run everything CI would have run, locally, on the real
# supported Python range. CI was re-running the same tests on rented hardware while
# adding a hard third-party dependency to an otherwise-local pipeline.
#
# So: upload from here, and treat CI as a nice-to-have mirror. CI availability must
# never block a release.
echo
echo "==> Uploading to PyPI (local twine)"
TWINE="$(command -v twine || echo "$ROOT/.venv/bin/twine")"
if [[ ! -x "$TWINE" ]]; then
  echo "ERROR: twine not found. pip install twine, or set TWINE=/path/to/twine" >&2
  exit 1
fi
"$TWINE" upload "$ROOT"/dist/"$NAME"-"$VERSION"*  # refuses if the version exists

echo "==> Verifying PyPI has it"
for _ in 1 2 3 4 5 6; do
  CODE="$(curl -s -o /dev/null -w '%{http_code}' "https://pypi.org/pypi/$NAME/$VERSION/json")"
  [[ "$CODE" == "200" ]] && break
  sleep 5
done
if [[ "$CODE" != "200" ]]; then
  echo "ERROR: PyPI does not report $NAME $VERSION after upload (got $CODE)." >&2
  exit 1
fi
echo "    ✔ live: https://pypi.org/project/$NAME/$VERSION/"

# The tag is now a RECORD of what shipped, not the trigger. If CI is healthy it will
# also run publish.yml; that upload is a no-op because the version already exists,
# which is exactly the behaviour we want — never a second source of truth.
echo "==> Tagging GitHub Release v$VERSION (record only; publish already happened)"
if gh release create "v$VERSION" --target "$(git rev-parse HEAD)" \
     --title "v$VERSION" --generate-notes 2>/dev/null; then
  echo "    ✔ tagged"
else
  echo "    ⚠ could not create the GitHub Release (CI unavailable / auth?) —"
  echo "      NOT fatal: $NAME $VERSION is already on PyPI. Tag later with:"
  echo "      gh release create v$VERSION --generate-notes"
fi
