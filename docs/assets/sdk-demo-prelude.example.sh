#!/usr/bin/env bash
# Prelude for sdk-demo.tape — evaluated OFF-SCREEN by the tape's hidden first
# line, so no key ever appears in a frame. Usage, from the repo root:
#   cd docs/assets
#   DECIMALAI_TAPE_PRELUDE="source sdk-demo-prelude.example.sh" vhs sdk-demo.tape
# Export DECIMAL_API_KEY first (free key: app.decimal.ai/settings).

export DECIMAL_API_KEY="${DECIMAL_API_KEY:-dai_sk_...}"
# Optional: point at a non-default stack.
# export DECIMAL_BASE_URL=...   # API
# export DECIMAL_APP_URL=...    # frontend used in printed links

# The on-screen regression-check command uses the canonical CI form (no
# manifest id — CI auto-discovers it via flush_manifest_for_ci()). For the
# recording, inject the id of the v2 manifest the demo just seeded:
decimalai() {
  if [ "$1" = "regression-check" ]; then
    local base="${DECIMAL_BASE_URL:-https://api.decimal.ai}" id
    id=$(curl -s -H "Authorization: Bearer $DECIMAL_API_KEY" \
      "$base/api/v1/manifests?agent_name=%5BDemo%5D%20support-agent&limit=5" \
      | python3 -c 'import json,sys; ms=json.load(sys.stdin)["manifests"]; print(next(m["id"] for m in ms if m["version_label"]=="v2"))')
    command decimalai "$@" --candidate-manifest-id "$id"
  else
    command decimalai "$@"
  fi
}
