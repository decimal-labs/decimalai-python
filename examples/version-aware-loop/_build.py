#!/usr/bin/env python3
"""Build version-aware-loop/manifest_change.ipynb.

The .ipynb is GENERATED. Edit this file, then re-run it:

    python examples/version-aware-loop/_build.py

DESIGN RULE FOR THIS NOTEBOOK: no cell may print an outcome it did not
observe. Every count, version label and verdict below is read back out of an
HTTP response. The earlier revision printed fixed strings — "3 traces sent",
"Impact: 3 traces classified as 'repair'" — that rendered identically whether
the call succeeded or returned 401, and whose numbers disagreed with what the
API actually returns. Two helpers now stand between the reader and every
claim: `traces_accepted()` diffs `decimalai.export_status()` and raises on a
failed send, and `show_impact()` GETs the impact report and prints the
response body. If something is broken, the notebook shows a traceback.
"""
import json
import os


def _s(source):
    lines = source.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": _s(s)}


def code(s):
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": _s(s),
    }


META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

cells = [
    md("""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/version-aware-loop/manifest_change.ipynb)

# Version-Aware Loop: Manifest Changes & Impact Reports

**DecimalAI's core differentiator: automatic detection of agent changes and their impact on your training data.**

This notebook dives deeper than the quickstart into:
- How manifests are auto-detected from tools, models, and prompts
- What triggers a new manifest version — and what does *not*
- How the impact report classifies every trace as **keep / repair / replay / drop**
- When to use mechanical repair vs. LLM replay

> **No LLM API key required** — every agent below is a mock function, so nothing
> here spends model credits.
>
> **A DecimalAI API key is required.** Every number this notebook prints is read
> back from the API. If a call fails you get a traceback, not a green checkmark.
"""),

    md("""
## What Is a Manifest?

A **manifest** is a snapshot of your agent's configuration at a point in time:
- **Tools**: names, schemas, descriptions
- **Models**: provider, model name, temperature
- **Prompts**: system prompts, templates
- **Subagents**: names and configurations

DecimalAI hashes this snapshot. When the hash changes, a new manifest version is
created and existing traces are classified against the new version.
"""),

    md("## Setup"),

    # NOTE: `%pip` would be the more correct magic (it installs into the
    # kernel's own environment; `!pip` uses whatever pip is first on PATH).
    # Left as `!pip` deliberately: examples/test_examples.py syntax-checks every
    # notebook by DELETING lines that start with `!` and compiling the rest, and
    # it does not strip `%` at all — so a `%pip` line fails that check. Change
    # both together, not this file alone.
    code("!pip install -q decimalai"),

    code('''
# ── Credentials ────────────────────────────────────────────────
# In Colab: paste your key into the string below.
# Anywhere else: export DECIMAL_API_KEY instead — an environment variable
# that is already set wins over the literal here.
import os

API_KEY = os.environ.get("DECIMAL_API_KEY") or "dai_sk_..."  # ← Replace with your key
BASE_URL = os.environ.get("DECIMAL_BASE_URL", "https://api.decimal.ai").rstrip("/")

if not API_KEY.startswith("dai_") or API_KEY.endswith("..."):
    raise RuntimeError(
        "No DecimalAI API key. Paste one into API_KEY above, or set the "
        "DECIMAL_API_KEY environment variable. Keys live at "
        "https://app.decimal.ai/settings (Settings → API Key)."
    )

os.environ["DECIMAL_API_KEY"] = API_KEY
os.environ["DECIMAL_BASE_URL"] = BASE_URL

import decimalai

# init() probes GET /api/v1/auth/verify and raises DecimalConfigError if the
# server rejects the key or the base URL is unreachable. Nothing below this
# line runs on a bad key.
decimalai.init()
print(f"decimalai {decimalai.__version__} initialised against {BASE_URL}")
'''),

    md("""
## Every Number Below Is Read Back From the API

Two helpers do all the reporting, so no cell can claim an outcome it did not
observe:

- **`traces_accepted(n, label)`** flushes the SDK's send queue and diffs
  `decimalai.export_status()`. If fewer than `n` traces actually reached the
  backend — or a manifest registration failed — it raises, quoting the server's
  own error text.
- **`show_impact()`** GETs `/api/v1/agents/<agent>/impact-report` and prints the
  response body. A non-200 raises. The `keep / repair / replay / drop` counts
  are the API's answer for *your* traces; this notebook has no opinion about
  what they ought to be.

If your key is wrong, the network is down, or the backend rejects a trace, you
get a traceback here instead of a checkmark. That is the point.
"""),

    code('''
# ── Reporting helpers ──────────────────────────────────────────
import time
import uuid

import httpx  # a dependency of decimalai, so it is already installed

# A fresh agent name per run. The impact report always describes the *latest*
# manifest transition for an agent, so a run-scoped name keeps the numbers below
# about the four traces you are sending now instead of leftovers from last time.
AGENT = f"demo-agent-{uuid.uuid4().hex[:8]}"

api = httpx.Client(
    base_url=BASE_URL,
    headers={"Authorization": f"Bearer {API_KEY}"},
    timeout=30.0,
)

# Free-plan keys allow a burst of 10 requests, refilling at one per second.
# Running this notebook top-to-bottom fires roughly twenty (each trace is a
# manifest registration plus an ingest), so the helper pauses between scenarios
# rather than tripping its own rate limit. A 429 that survives the pause is
# still reported as a failure — it is never swallowed.
PACE_SECONDS = 4.0

_seen = {"sent": 0, "failed": 0, "manifest_error": None}


def traces_accepted(expected: int, label: str) -> None:
    """Flush, then verify the backend really accepted `expected` new traces."""
    decimalai.flush()
    st = decimalai.export_status()

    sent = st.sent - _seen["sent"]
    failed = st.failed - _seen["failed"]
    manifest_error = (
        st.last_manifest_error
        if st.last_manifest_error != _seen["manifest_error"]
        else None
    )
    _seen.update(
        sent=st.sent, failed=st.failed, manifest_error=st.last_manifest_error
    )

    if failed or manifest_error or sent != expected:
        raise RuntimeError(
            f"{label}: expected {expected} trace(s) to reach {BASE_URL}, but "
            f"{sent} were accepted and {failed} failed.\\n"
            f"  last ingest error:   {st.last_error}\\n"
            f"  last manifest error: {manifest_error}"
        )

    print(f"✅ {label}: {sent} trace(s) accepted by {BASE_URL}, 0 failed")
    time.sleep(PACE_SECONDS)  # let the rate-limit bucket refill


def _get(path: str, attempts: int = 4) -> httpx.Response:
    """GET with a bounded retry on 429. Any other status is returned as-is."""
    for attempt in range(attempts):
        resp = api.get(path)
        if resp.status_code != 429 or attempt == attempts - 1:
            return resp
        wait = int(resp.headers.get("Retry-After") or 2)
        print(f"   (rate limited on {path} — retrying in {wait}s)")
        time.sleep(wait)
    raise AssertionError("unreachable")


_previous_candidate = {"id": None}


def show_impact() -> dict:
    """Print the agent's latest manifest transition, exactly as the API reports it."""
    path = f"/api/v1/agents/{AGENT}/impact-report"
    resp = _get(path)
    if resp.status_code != 200:
        raise RuntimeError(f"GET {path} → HTTP {resp.status_code}: {resp.text[:400]}")

    rep = resp.json()

    if rep.get("status") != "ok":
        print(f"No impact report yet — status={rep.get('status')!r}")
        print(f"  {rep.get('severity_reason') or rep.get('human_summary') or ''}")
        print(f"  manifests so far: {rep.get('manifest_count')}, "
              f"traces so far: {rep.get('trace_count')}")
        return rep

    if rep["candidate_manifest_id"] == _previous_candidate["id"]:
        print("⚠️  No NEW manifest version was created — the manifest hash did not")
        print("    change, so this is still the previous transition:")
    _previous_candidate["id"] = rep["candidate_manifest_id"]

    print(f"{rep['baseline_version_label']} → {rep['candidate_version_label']}"
          f"  (agent {rep['agent_name']})")
    print()
    print("Surface changes detected by the backend:")
    for ch in rep["surface_changes"]:
        print(f"  · {ch['type']:<32} {ch['surface_name']:<16} "
              f"severity={ch['severity']:<7} affects {ch['affected_count']} trace(s)")
    if not rep["surface_changes"]:
        print("  (none)")

    b = rep["compat_summary"]
    print()
    print(f"Impact report ({rep['affected_trace_count']} trace(s) affected):")
    print(f"  keep {b['keep']}    repair {b['repair']}    "
          f"replay {b['replay']}    drop {b['drop']}")
    print(f"  severity: {rep['severity']} — {rep['severity_reason']}")
    print(f"  summary:  {rep['human_summary']}")
    return rep


print(f"Agent for this run: {AGENT}")
'''),

    md("""
## Scenario 1: Tool Rename

The most common change. You rename a tool for clarity; the functionality stays
the same. Send three traces on the old name, one on the new name, and read the
impact report the backend produces.
"""),

    code('''
# ── v1: Original agent ──
@decimalai.trace(agent_name=AGENT)
def agent_v1(query: str) -> str:
    decimalai.log_tool_call(
        name="get_stock_price", input={"ticker": "AAPL"}, output={"price": 178.52}
    )
    return "AAPL is at $178.52"


for q in ["What is AAPL price?", "Check TSLA stock", "NVDA current price"]:
    agent_v1(q)

traces_accepted(3, "v1 — tool get_stock_price")
'''),

    code('''
# ── v2: Renamed tool ──
@decimalai.trace(agent_name=AGENT)
def agent_v2(query: str) -> str:
    # Same function, better name.
    decimalai.log_tool_call(
        name="lookup_ticker", input={"ticker": "AAPL"}, output={"price": 178.52}
    )
    return "AAPL is at $178.52"


agent_v2("What is AAPL price?")

traces_accepted(1, "v2 — tool lookup_ticker")
print()
_ = show_impact()
'''),

    md("""
### How To Read That Report

1. **The diff is structural.** A rename reaches the backend as two independent
   facts — `tool_removed: get_stock_price` and `tool_added: lookup_ticker`.
   Nothing in a trace says "these two are the same tool under a new name."
2. **The buckets are counted from stored traces, not from the diff.** Each
   surface change carries `affected_count` — how many historical traces touched
   that surface. `keep / repair / replay / drop` is the backend's proposal for
   what to do with them.
3. **Whatever bucket you got is the honest answer for your data.** A rename the
   backend can pair up lands in `repair`: a mechanical string replacement in the
   stored trace, no LLM cost, instant. One it cannot pair lands in `drop`. Read
   the printed counts — this notebook deliberately does not assert them.
"""),

    md("""
## Scenario 2: Tool Added

Adding a new tool should not invalidate existing traces — they simply never used
it. Watch which bucket the backend puts them in.
"""),

    code('''
# ── v3: Added a new tool ──
@decimalai.trace(agent_name=AGENT)
def agent_v3(query: str) -> str:
    decimalai.log_tool_call(
        name="lookup_ticker", input={"ticker": "AAPL"}, output={"price": 178.52}
    )
    # NEW: a second tool this version can call.
    decimalai.log_tool_call(
        name="get_earnings", input={"ticker": "AAPL"}, output={"eps": 6.42}
    )
    return "AAPL: $178.52, EPS: $6.42"


agent_v3("AAPL price and earnings")

traces_accepted(1, "v3 — added get_earnings")
print()
_ = show_impact()
'''),

    md("""
## Scenario 3: Tool Removed

Removing a tool that existing traces used is the most disruptive change: those
traces reference a capability that no longer exists.
"""),

    code('''
# ── v4: Removed lookup_ticker, kept only get_earnings ──
@decimalai.trace(agent_name=AGENT)
def agent_v4(query: str) -> str:
    decimalai.log_tool_call(
        name="get_earnings", input={"ticker": "AAPL"}, output={"eps": 6.42}
    )
    return "AAPL EPS: $6.42"


agent_v4("AAPL earnings")

traces_accepted(1, "v4 — removed lookup_ticker")
print()
_ = show_impact()
'''),

    md("""
## Scenario 4: Schema Change

Now change the *arguments* a tool takes rather than its name — `get_earnings`
starts requiring a `period` parameter. Run the cell and read what comes back
before assuming a new version appeared.
"""),

    code('''
# ── v5: Same tool name, different arguments ──
@decimalai.trace(agent_name=AGENT)
def agent_v5(query: str) -> str:
    decimalai.log_tool_call(
        name="get_earnings",
        input={"ticker": "AAPL", "period": "quarterly"},  # NEW: period param
        output={"eps": 1.52, "period": "Q4 2025"},        # richer output
    )
    return "AAPL Q4 2025 EPS: $1.52"


agent_v5("AAPL quarterly earnings")

traces_accepted(1, "v5 — get_earnings called with a new argument")
print()
_ = show_impact()
'''),

    md("""
### What Auto-Detection Records — and What It Doesn't

`log_tool_call()` contributes the tool's **name** to the auto-detected manifest.
The arguments you pass are stored on the trace as a preview, not as a declared
schema. So if the cell above reported "no NEW manifest version", that is why:
calling `get_earnings` with an extra `period` argument hashes to the same
manifest as calling it without one.

That is a deliberate limit, not a gap to route around — DecimalAI will not
invent a contract you never declared. To version a tool's schema, declare it.
That is what the next section does.
"""),

    md("""
## Declaring the Contract Explicitly

`register_manifest()` states your agent's configuration outright: tool schemas,
models, prompts. Because the schema is now declared, a change to it *is* a
version change, and the impact report can classify traces against it.
"""),

    code('''
# Explicit manifest registration — this is how a SCHEMA change gets versioned.
result = decimalai.register_manifest(
    agent_name=AGENT,
    tools=[
        {"name": "get_earnings", "schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "period": {"type": "string", "enum": ["quarterly", "annual"]},
            },
            "required": ["ticker", "period"],
        }},
    ],
    models={"default": {"provider": "openai", "model": "gpt-4o"}},
    prompts={"system": "You are a financial analyst assistant."},
)

# register_manifest() raises DecimalAPIError on any non-2xx response, so
# reaching this line means the backend accepted the manifest. Print the whole
# response rather than a hand-picked summary with "N/A" fallbacks.
print("POST /api/v1/manifests returned:")
for key, value in result.items():
    print(f"  {key}: {value}")
print()
_ = show_impact()
'''),

    code('''
print(f"Agent name for this run: {AGENT}")
print(f"Impact report API:       GET {BASE_URL}/api/v1/agents/{AGENT}/impact-report")
if BASE_URL == "https://api.decimal.ai":
    print(f"Dashboard:               https://app.decimal.ai/agents/{AGENT}")
else:
    print(f"Dashboard:               n/a — you are pointed at {BASE_URL}, not the hosted API")
'''),

    md("""
## Summary: The Four Classifications

These are the definitions the backend classifies against. What *your* run
produced is printed in the cells above.

| Classification | Trigger | Fix | Cost |
|---------------|---------|-----|------|
| **Keep** | Trace doesn't use any changed component | None needed | Free |
| **Repair** | Tool renamed, prompt template changed | Mechanical string replacement | Free |
| **Replay** | Schema changed, model swapped | Re-run through updated agent | LLM cost |
| **Drop** | Tool removed entirely | Cannot fix — exclude from dataset | Free |

### Why This Matters for Training

If you build a fine-tuning dataset from traces without checking compatibility:
- **Stale tool calls** teach the model to call non-existent tools
- **Wrong schemas** teach the model to pass outdated parameters
- **Mixed versions** create inconsistent training signal

The impact report is how you find those traces before they reach a dataset.

## Next Steps

- 📖 [Quickstart](../quickstart/quickstart.ipynb) — Start from the basics
- 📖 [Evaluations](../evaluations/builtin_evaluators.ipynb) — Score your traces
- 📖 [Build Datasets](../datasets-and-training/build_sft_dataset.ipynb) — Create clean training data
"""),
]

nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 4}
path = os.path.join(os.path.dirname(__file__), "manifest_change.ipynb")
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"✅ {path}")
