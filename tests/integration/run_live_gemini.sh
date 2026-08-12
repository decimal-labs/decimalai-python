#!/usr/bin/env bash
#
# Run the live-LLM framework matrix against a local DecimalAI backend using a
# real Gemini key. Exercises LangChain, the generic decorator, and the OpenAI
# Agents SDK (driven through Gemini's OpenAI-compatible endpoint) — every path
# makes real Gemini calls and asserts the trace + auto-detected manifest land
# in the backend.
#
# Usage:
#   ./tests/integration/run_live_gemini.sh                 # full gemini matrix
#   ./tests/integration/run_live_gemini.sh -v              # + verbose
#   LIVE_K="gemini and openai_agents" ./tests/integration/run_live_gemini.sh
#
# Requirements:
#   - A DecimalAI backend reachable at $DECIMAL_BACKEND_URL (default
#     http://localhost:8000). Point it at the hosted API or your own
#     deployment; the tests assert the traces they send land there.
#   - GEMINI_API_KEY exported, OR a `GEMINI_API_KEY=...` line in a .env file.
#     The .env is looked up at $DECIMAL_LIVE_ENV_FILE if set, else at the
#     repo root (./.env, which is gitignored).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKEND_URL="${DECIMAL_BACKEND_URL:-http://localhost:8000}"
LIVE_K="${LIVE_K:-gemini or google}"

# 1. Resolve the Gemini key (prefer an already-exported one; else read a .env).
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  ENV_FILE="${DECIMAL_LIVE_ENV_FILE:-$REPO_ROOT/.env}"
  if [[ -f "$ENV_FILE" ]]; then
    GEMINI_API_KEY="$(grep -E '^GEMINI_API_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2-)"
  fi
fi
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "ERROR: GEMINI_API_KEY not set and not found in a .env file." >&2
  echo "  export GEMINI_API_KEY=...  (or point DECIMAL_LIVE_ENV_FILE at a .env that has one)" >&2
  exit 1
fi
export GEMINI_API_KEY
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-$GEMINI_API_KEY}"  # langchain-google-genai reads this
export RUN_LIVE_LLM_TESTS=1

# 2. Backend must be up — the suite skips (not fails) otherwise, so fail loudly here.
if ! curl -sf -m 3 "$BACKEND_URL/health" >/dev/null 2>&1; then
  echo "ERROR: DecimalAI backend not reachable at $BACKEND_URL" >&2
  echo "  set DECIMAL_BACKEND_URL to a running backend (the hosted API, or your own)." >&2
  exit 1
fi

# 3. Run. `-m live_llm` overrides the default 'not live_llm' marker filter in
#    pyproject. Extra args ($@) pass straight through (e.g. -v, a node id).
cd "$REPO_ROOT"
echo "Running live Gemini matrix (-k \"$LIVE_K\") against $BACKEND_URL ..."
exec .venv/bin/python -m pytest tests/integration/ -m live_llm -k "$LIVE_K" "$@"
