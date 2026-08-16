# Framework conformance

One contract. Every framework. Asserted on the wire.

## Why this exists

This repo has 626 adapter tests across 49 files and **not one of them makes an
HTTP-level assertion**. Every one drives a mock, so "it works" has only ever
meant "it called our fake as expected". That is how the LlamaIndex adapter
shipped for months emitting zero traces with a fully green suite.

A conformance test asserts that an implementation satisfies a **contract**, not
that it called a function. The contract here is:

> Running the framework's documented snippet produces a trace **on the wire**
> that satisfies these properties.

One spec, many implementations. Four rules make that real:

1. **One shared set of assertions, applied to every framework.** They all live
   in `contract.py`. A framework gets a *driver* — how to run its documented
   snippet — and never its own assertions. `test_drivers_contain_no_assertions`
   enforces this structurally.
2. **Assert the wire.** The adapter talks real HTTP to `probe.py`, a real server
   that validates the ingest contract the way the backend does. No mock
   transports, no monkeypatched senders.
3. **Hermetic by default.** Tier A needs no provider key and no platform
   backend, so it runs on every commit. Nearly every defect worth catching —
   emits nothing, missing `manifest_id`, flat spans, previews that are role
   names, `routing_id` absent, concurrency cross-contamination, cwd pollution —
   is visible at the wire with a stub model.
4. **Nobody gets left out.** `test_coverage.py` fails when a framework is
   advertised and has no driver. There is no ledger, no allow-list and no
   "we'll add it later" — see [Coverage guard](#coverage-guard-the-anti-drift-part).

**Red rows are the product, not a bug.** Most frameworks fail several contract
items today. Each one is a finding about an adapter, recorded and left red on
purpose. Do not "fix" a red row by weakening `contract.py` unless the contract
itself is wrong for *every* framework.

## Running it

There is no Makefile in this repo. The one command is:

```bash
pip install -e ".[dev,conformance-tests]"    # the eleven frameworks the drivers drive
python -m pytest tests/conformance -q -rs -m conformance   # ~30s; -rs prints every N/A reason
# -m conformance is required — the marker is deselected by default so this suite
# never shares a process with the unit suite (it mutates adapter globals on purpose).
```

Other useful forms:

```bash
pytest tests/conformance/test_coverage.py -m conformance   # the anti-drift guard alone; no framework needed
pytest tests/conformance -k langchain -m conformance       # one framework's whole column
pytest -m 'not conformance'                  # everything else, without this
```

A bare `pytest` from the repo root **does** collect Tier A (the default
`addopts` excludes `integration`, `live_llm` and `conformance_live`, but not
`conformance`). Only the GitHub `test` job passes `--ignore=tests/conformance`,
and only because `.[dev]` installs three of the eleven frameworks — a partial
matrix reads as coverage and is not.

Every run ends with the conformance matrix: item × driver, PASS / FAIL / N/A,
each with the reason. Set `DECIMAL_CONFORMANCE_REPORT_JSON=<path>` to also get
it as JSON (the release gate does this so its report can name the drivers that
did not run).

## Where it runs

| Lane | What it is | Blocks? |
|---|---|---|
| `.github/workflows/ci.yml` job `conformance` | Tier A on every push and PR. No secrets. | yes |
| `make ci-workflows` (platform) | the same job, executed locally from the same YAML | yes |
| `make release-gate` → phase P0 | Tier A inside the pinned release gate | yes |
| `make release-gate-conformance` | phase P0 alone, ~90s, no key, no backend | yes |
| `make release-gate-latest` (canary) | Tier A against the **newest** frameworks | no — advisory |

the local CI runner needed no change to pick the job up: it parses each
repo's workflow YAML and runs the same `run:` blocks, so adding the job was
enough. Verify with `python3 infra/ci_local.py --list --repos decimalai-python`.

The CI job runs the coverage guard **before** installing the frameworks —
"somebody added a framework and skipped the driver" then fails in seconds rather
than after a multi-minute install — and sets
`DECIMAL_CONFORMANCE_REQUIRE_ALL=1` for the matrix, which turns "a driver could
not import its framework" from a silent skip into a failure.

## Adding a framework

You write a driver; you do not write assertions.

1. Create `drivers/<name>.py` with a stub model, a `run(ctx)` that executes the
   framework's **documented** snippet, and a module-level `DRIVER = Driver(...)`.
2. Add `"<name>"` to `DRIVER_MODULES` in `drivers/__init__.py`.
3. If the docs capability table gained a row, add its slug to
   `ADVERTISED_SNAPSHOT` in `drivers/__init__.py`. If the framework arrived as a
   `pyproject.toml` extra or an `init()` flag, add it to `EXTRA_SLUGS` /
   `FLAG_SLUGS` in `test_coverage.py`.
4. Add its distribution to the `conformance-tests` extra in `pyproject.toml`, so
   CI actually installs it. (A missing install means the driver reports NOT RUN,
   which `DECIMAL_CONFORMANCE_REQUIRE_ALL=1` turns into a CI failure.)
5. Run `pytest tests/conformance -m conformance`. Whatever the adapter gets wrong now fails.
   **That is the correct outcome, not a broken test.**

You do not have to remember any of that: every coverage failure prints the whole
recipe, with the file to create and the lines to add.

The whole driver surface:

| Field | Meaning |
|---|---|
| `name`, `covers`, `requires`, `entrypoint` | identity, docs rows covered, imports needed, entry point exercised |
| `capabilities` | which contract items apply; every `False` needs a printed reason |
| `run(ctx)` | the documented snippet, once |
| `run_concurrent(ctxs)` | N lanes at once — `fanout_threads(run)` is one line |
| `run_error(ctx)` | a run that fails partway through |
| `run_degenerate(ctx)` | a run with no model and no tools |
| `run_skills(ctxs)` | the skills rail, one run per lane |

Everything a driver needs arrives on `Ctx`, and the contract asserts against the
**same** `Ctx`. The sentinels are load-bearing: `ctx.prompt_sentinel` must go
into the prompt, `ctx.reply_sentinel` must be what the stub model answers,
`ctx.tool_name` must be the tool called. That is how "the preview said
`system`" and "the preview was `<object at 0x7f…>`" get caught without the
contract knowing anything about the framework.

Use the shared helpers rather than inventing your own text — `stub_script(ctx)`,
`tool_result(ctx, query)`, `user_message(ctx)`, `SYSTEM_PROMPT`,
`STUB_MODEL_NAME`, `fanout_threads(run)`. `stub_script` returns the scripted
turns (tool call, then answer) with fixed token counts, so **every** framework's
stub answers the same script and a contract item means the same thing
everywhere. The only genuinely framework-specific code left in a driver is
mapping `StubTurn` onto that framework's message object — in
`drivers/langchain.py` that is the ~40-line `_stub_model`, and the whole rest of
the file is the documented snippet itself. Two drivers do not even need that:
anything reaching its model through the `openai` SDK reuses
`drivers/_openai_wire.py`, a real HTTP server speaking the OpenAI wire format.

If a framework seems to need a special assertion, **the contract is wrong**.
Fix `contract.py` so every framework gets the fix.

## One driver, one process

Every adapter installs **process-global** instrumentation: module ContextVars,
langchain-core's global configure-hook list, an OTel global `TracerProvider`
that can only be set once, monkeypatched `__init__`s that cannot be undone. Run
every driver's phases in one process and whichever goes first decides what the
ones after it are allowed to observe — which produced *false* results in
both directions: a framework that emits fine graded as emitting nothing, a
known-red item graded green, and one driver's `manifest_id` stamped on another
driver's traces (a C2 rejection blamed on the wrong adapter).

So `isolation.py` runs each driver's phases in a **child process** — one per
driver, `sys.executable` with `PYTHONPATH` inherited — and brings only the
capture back as JSON. The probe lives in the child, because that is the process
the SDK's HTTP calls come out of. The parent imports no framework and runs no
driver code; it deserialises and grades, so `contract.py` stays the single place
assertions live.

Consequences worth knowing:

- Children run a few at a time (`DECIMAL_CONFORMANCE_JOBS`, default 4). Each has
  its own probe on its own port, so they cannot interfere; the cap keeps the
  timing-sensitive items (C8, C9) off a saturated machine.
- Only the drivers whose items were collected are run. `-k langchain` is one
  child and ~6s, not the whole matrix.
- A child that crashes or times out (`DECIMAL_CONFORMANCE_DRIVER_TIMEOUT`,
  default 900s) makes **every** item for that driver a hard error naming the
  driver, and the matrix prints `NOT GRADED — the driver process died`. It is
  never a skip: an ungraded driver reported as passed or skipped is the failure
  mode this tier exists to remove.
- What crosses the boundary is exactly what the contract reads: every phase's
  recorded requests (bodies verbatim), plus the probe state C6 and C8 query.
  `isolation.dump_payload` re-parses its own JSON and compares before shipping,
  so a field that does not survive the round-trip crashes the child instead of
  being graded.

## The phases

Run in order, in one process — the driver's OWN process — each in its own temp
cwd. Adapter module globals are deliberately **not** reset between them: running
two agents back to back is exactly where process-global state bites, and that is
a property of one driver's run, not something another framework may inject.

| Phase | What it is | Feeds |
|---|---|---|
| `main` | the documented snippet | C1 C3 C4 C5 |
| `repeat` | the same agent again | C7 |
| `second_agent` | `run` with a *differently named* agent | C6 |
| `degenerate` | no model, no tools | C7b |
| `error` | a run that fails | C10 |
| `concurrent` | N lanes, distinct agents | C9 |
| `skills` | the rail, N lanes, one agent | C8 |

`skills` runs last on purpose: on several adapters, enabling the rail is an
irreversible process-wide monkey-patch that would double-trace every other
phase. `DRIVER_MODULES` is ordered for the same reason — the drivers that enable
a process-wide OpenInference instrumentor run last.

C2, C11 and C12 grade **every** phase.

## The probe

`probe.py` is a real `ThreadingHTTPServer` implementing the endpoints the SDK
actually calls — discovered by reading `decimalai/_client.py` and
`decimalai/skill_router.py`, not guessed:

`GET /auth/verify` · `POST /traces` · `POST /traces/batch` ·
`POST|GET /manifests` · `GET /manifests/{id}` · `POST /skills/route` ·
`GET /skills/menu` · `GET /skills/{name}/body` · `GET /skills/hashes` ·
`GET /skills` · `GET /skills/{name}` · `POST /skills/sync`

Anything else 404s and lands in the request log, so a driver that depends on an
unmodelled endpoint shows up as a 404 rather than a mystery.

Two behaviours matter:

**It rejects what the backend rejects.** `validate_trace_payload` is a port of
`the platform's trace-ingest validator` (plus the
manifest-exists / trace-id-shape / duplicate-id checks at the top of
`ingest_trace`). "The backend 400s this" is the defect class a mock can never
catch.

**The port cannot silently rot.** `test_backend_validator_has_not_drifted`
re-derives a fingerprint of the backend's validator whenever the platform repo
is on disk, and fails when it moves. Scope caveat, stated plainly: the
fingerprint covers the two validator functions and the four constants they read,
**not** the manifest/trace-id rules that live inside the 200-line `ingest_trace`
— hashing that whole function would churn on every unrelated edit and the guard
would be switched off within a month. Re-read those by hand when the guard
fires.

**Routing provenance.** The probe records which query each `routing_id` was
minted for. That turns "two lanes happened to collide" into a deterministic
statement: *this run reports a routing decision the router made for that other
run*.

## The contract (v1)

| Item | Asserts |
|---|---|
| C1 `emits` | at least one trace reaches the wire |
| C2 `ingest_valid` | every trace passes the backend's own validation |
| C3 `llm_calls` | model name present; token fields present and plausible |
| C4 `content` | previews and `rendered_input` carry the real prompt/completion text |
| C5 `structure` | >1 span, ≥1 parent link, distinguishable names |
| C6 `identity` | the trace names the agent asked for, carrying *that* agent's manifest |
| C7 `manifest_stable` | the same agent run twice mints one manifest version |
| C7b `manifest_no_fabrication` | a degenerate run fabricates no manifest change |
| C8 `skills_rail` | `routing_id` present and per-run; offered names recorded and actually in the prompt |
| C9 `isolation` | N concurrent runs, N traces, no cross-contamination |
| C10 `error_path` | a failing run produces exactly one trace, marked errored |
| C11 `no_side_effects` | nothing written into the working directory |
| C12 `loud_failure` | a phase that emits nothing, or a trace the backend refuses, is never silent |

C7b is the second clause of C7, split out so a framework with no degenerate form
can declare that one clause N/A without silencing the first.

Exact token counts are **not** asserted here — a stub model's numbers are
arbitrary. Presence and plausibility are Tier A's job; exact counts are Tier B's.

## Tiers

**Tier A (hermetic)** — the default, marker `conformance`. Stub model, probe
server, no keys, no backend. Runs everywhere, every commit.

**Tier B (live)** — marker `conformance_live`, excluded from the default run.
Real provider, real backend, **the same contract functions**. Not implemented
yet. The intended mechanism, so nobody forks the assertions to build it: give
`Probe` a `forward_to=<real backend base_url>` mode in which it records and
validates exactly as now and then *proxies* the request onward, returning the
real response. Tier B is then the same harness with a forwarding probe and
drivers that build a real model instead of a stub — `Ctx` already carries
`base_url` and `api_key`, so nothing else changes. Do **not** build Tier B by
reading traces back through `GET /api/v1/traces`; that grades what the backend
stored, not what the adapter sent, and the two differ exactly where the bugs
are.

## Version policy

The **pinned** lane is the gate; the **floating** lane is advisory. Both already
existed in `platform/release_gate`, and conformance rides them rather than
growing a parallel copy. Nothing here is a second matrix, a second lockfile or a
second set of assertions.

| | Pinned lane | Floating lane |
|---|---|---|
| Entry point | `run_gate.py` (phase **P0**) | `canary.py --latest` |
| Versions from | `release_gate/constraints.txt` + the SDK's declared floors | newest, resolved live |
| Which frameworks | the `conformance-tests` extra | the names in `release_gate/frameworks-matrix.txt` |
| Baseline diffed against | — | `release_gate/frameworks-lock.txt` |
| A red row | **blocks the release** | opens an issue; blocks nothing |

The mechanism is one function. `run_gate.run_conformance()` shells
`pytest tests/conformance -m conformance` with `cwd=SDK_DIR` against the venv it
was given. `canary.py --latest` upgrades every name in `frameworks-matrix.txt`
to newest inside a clean-room wheel venv and then calls **that same
orchestrator**, so the floating lane is the identical phase with different
versions underneath. The phase does not know which lane it is in — the advisory/
blocking distinction is structural (the canary is its own scheduled lane and was
never a PR check), not a flag the assertions read.

`frameworks-matrix.txt` grew a second block for the frameworks that have a
conformance driver but no live-matrix lane (crewai, google-adk, llama-index-core,
both AutoGen lineages, pydantic-ai, anthropic, and the OpenInference
instrumentors the OTel-routed drivers read). Floating those is cheap precisely
because Tier A is hermetic — no model spend, no backend.

Two commands worth knowing:

```bash
# pinned, ~30s, no key, no backend — "did I break an adapter?"
make release-gate-conformance

# floating: newest frameworks vs the contract, still no key and no backend
python release_gate/canary.py --latest --conformance-only
```

### Which versions we support, and why

Floors are claims we have RUN, not numbers we picked. Three in this repo were
fiction before anyone checked: `llama-index-core>=0.10.20` (the module the
adapter imports does not exist there), `openai-agents>=0.17.2` (true alone,
false against an `openai` our own extra would select), and
`langchain-core>=1.3.0` (documented "CVE-clean" while admitting three patch
releases that carry CVE-2026-44843). Verify a floor by installing at it and
running that framework's driver — the suite makes this cheap.

Two traps a version range cannot express, both found by resolving and then
running:

- **A floor on one package can be falsified by a floor on another**, in a
  different extra. Always check what `decimalai[<extra>]` and `decimalai[all]`
  actually resolve to, not what the line says.
- **A cap is a guess about breakage that has not happened yet.** `openai<3`
  held users a minor behind on openai-agents for six weeks: openai-agents
  0.21.0 requires `openai>=3`, so our cap made it unreachable. Keep a cap only
  where you can name what it protects and when to review it. The conformance
  job is the better control, because it tests the thing the cap is guessing at.

Support window, per framework, from the observed release cadence:

| Framework | Floor | Cap | Note |
|---|---|---|---|
| langchain-core | `>=1.3.3` | `<2` | Floor is the CVE fix, not compatibility — the adapter grades identically from 0.3.67 to 1.5.5. New minor every 6-7 weeks. |
| langgraph | `>=0.6.0` | none | Never imported by the SDK; the floor is a statement about what we test. langgraph `>=1.2` forces langchain-core `>=1.4.7` — a pairing rule our floors cannot express. |
| openai | `>=2.26.0` | none | Cap removed; see above. A new minor almost weekly. |
| openai-agents | `>=0.18.1` | none | Minors are the breaking unit, every ~2.5 weeks; a meaningful cap would need editing fortnightly. Hold the line with conformance. |
| llama-index-core | `>=0.12.0` | none | Verified at the floor after the previous one proved fictional. |
| google-adk | `>=2.0.0` | none | 2.x is young; watch it. |
| crewai | `>=1.15.3` for the test set | `<2` | Below 1.15 the conformance set silently resolved crewai back to 1.6.1 in an OTel fight with google-adk — so the job was grading a 2025 build users never see. Breaks in PATCH releases, so a major cap protects little. |
| autogen (classic) | `>=0.11.0` | `<1` | The `autogen` name is frozen at 0.14.1 forever — upstream moved to `ag2` 1.x and did not republish. The cap is load-bearing, not boilerplate. |
| autogen-core/agentchat | — | — | Microsoft's lineage: last release 10.5 months ago, fails 7 of 13 items. Recommend reclassifying as generic-OTel-only rather than an advertised integration. |
| pydantic-ai, claude-agent-sdk | `>=0.1.0` | none | Both floors are unverified `>=0.1.0` — the shape of a floor nobody has run. Verify before claiming them. |

### When a new framework version breaks a row

1. The floating lane goes red and opens an issue. **The pinned gate stays
   green**, so releases are not blocked. This is the whole point of the split.
2. Read the canary report's version diff (`changed` / `added` against
   `frameworks-lock.txt`) to see which upstream release moved.
3. Decide which of two things is true:
   - **The adapter is wrong.** Fix `decimalai/<adapter>.py`. The row goes green
     in both lanes and nothing else changes.
   - **The new upstream behaviour is correct and ours was.** Record a new
     baseline with `python release_gate/canary.py --latest --update-lock`, and
     raise the floor in `pyproject.toml` if the old version can no longer work.
4. Never make a red row green by editing the driver. A driver contains no
   assertions, so there is nothing in it to relax; if you find yourself wanting
   to, the change belongs in `contract.py` and applies to every framework.

If the pinned lane goes red instead, an adapter stopped satisfying the contract
at versions that used to work — that is a regression in this repo, not upstream,
and it blocks.

## Coverage guard (the anti-drift part)

`test_coverage.py` fails when the product advertises a framework that no driver
exercises. It reads three independent sources, so the guard cannot be defeated
by editing one file:

| Source | Where | Bites when the docs repo is absent? |
|---|---|---|
| the docs capability table | `decimalai-docs/sdk/python/frameworks.mdx`, vendored into `drivers.ADVERTISED_SNAPSHOT` | snapshot yes, re-derivation no |
| `decimalai.init()`'s framework flags | this repo | **yes** |
| the framework extras in `pyproject.toml` | this repo | **yes** |

Adding `init(haystack=True)`, or a `[haystack]` extra, or a capability-table row,
fails on the same commit that adds it. The failure message names the file to
create, the three lines to add and the two things not to do — because a guard
whose remedy takes twenty minutes to work out gets deleted.

Two more guards ride along: `test_every_driver_module_on_disk_is_registered`
(a driver file nobody runs reads like coverage and grades nothing) and
`test_no_driver_is_silently_unavailable` (opt-in via
`DECIMAL_CONFORMANCE_REQUIRE_ALL=1`; CI sets it after installing the frameworks).

**There is deliberately no debt ledger.** An earlier draft had a
`KNOWN_MISSING` set for frameworks whose driver "was not written yet". That
turns a hard guard into a formality — add the docs row, add the ledger line,
stay green — which is the exact outcome the guard exists to prevent. A framework
that genuinely cannot satisfy part of the contract declares those **items** N/A
in its driver's `Capabilities`, with a reason that is printed in the matrix. It
never opts out of having a driver.

## What v1 does not cover

Stated so nobody mistakes a green cell for a guarantee.

- **`init(google=True)`.** The raw-Google provider rail has no driver: the
  hermetic tier needs a stub that speaks the provider's wire format, and
  `google.genai`'s client has no `base_url` seam to point at one the way `openai`
  and `anthropic` do. Recorded in `PROVIDER_FLAG_DRIVERS` and printed by
  `test_recorded_provider_gaps_are_visible` on every run, rather than hidden.
- **The `disk_sync` path.** `instrument()` resolves `disk_sync=False` when the
  skill loader is on, so the `skills` phase never exercises the disk mirror.
  C11 therefore grades the per-call-handler and rail paths but not
  `init(langchain=True)` with `skill_authority="harness"`, which writes
  `SKILL.md` files and `.decimal/skills.lock` into the cwd. Covering it needs a
  second process, because `instrument()` is process-wide and irreversible.
- **Structured-JSON previews.** C4 flags a preview that is a bare role name,
  empty, or an unambiguous Python repr (`<X object at 0x…>` / `Class(kwarg=…`).
  A preview that embeds a JSON blob still carries the text and is not flagged —
  a stricter bar than every incumbent adapter can meet turns C4 into noise.
- **Tool-call argument fidelity.** Nothing asserts that
  `llm_calls[].tool_calls[].args` is the argument dict the model produced. On
  LangChain it is currently `{"input": "{'query': '…'}"}` — the dict
  stringified under an invented key. Worth a C13; deliberately out of v1 scope.
- **Exact token counts, cost, latency.** Tier B's job.
- **Cross-driver isolation.** Fixed: each driver's phases now run in their own
  process (see [One driver, one process](#one-driver-one-process)), so a late
  trace or a global instrumentor from one adapter can no longer reach another's
  probe. What remains is isolation *within* one driver's process, which is
  deliberate — C6, C7 and C9 exist to grade it.
- **This README does not carry the result matrix.** It would be stale within a
  week. Run the suite; the matrix it prints is the only authoritative one.
