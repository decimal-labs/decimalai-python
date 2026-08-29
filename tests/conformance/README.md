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

A bare `pytest` from the repo root does **not** collect Tier A: the default
`addopts` deselects `conformance` along with `integration`, `live_llm` and
`conformance_live`. That is why every command above passes `-m conformance`,
and why the CI job does too — without the marker, `pytest tests/conformance`
selects zero tests and exits green.

The deselection is not the "exists but never runs" trap that let an adapter
ship emitting nothing. This suite runs as its own required CI job on every
push and PR, and nightly against newest frameworks. It is deselected from the
unit run for a specific reason: it deliberately does not reset adapter globals
between phases — sticky process-global state is a defect it exists to catch —
and those globals leak into the unit suite, turning 11 pre-existing failures
into 18. Each driver already runs in its own child process so drivers cannot
contaminate *each other*; the remaining boundary is with the unit suite.

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
enough.

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
`GET /skills` · `GET /skills/{name}` · `POST /skills/sync` ·
`GET /agents` · `GET /agents/{name}/skills` · `GET /agents/{name}/prompt`

The last three are the journey's (J1): the three calls `decimalai init <name>`
makes, ported from the platform's own agent handlers rather than invented.
The list one is deliberately **unpaginated**, because `init` never sends a
`limit` on purpose — one activates a truncating path whose ordering drops the
never-traced UI-created agent the command exists to find. And
`/agents/{name}/skills` deliberately returns subscriptions **without bodies**,
exactly as the backend does, which is what makes J1's body-sentinel clause
falsifiable: the scaffold cannot learn a skill body from that route, so a body
in front of the model got there through the rail.

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
| C13 `skills_activation` | nothing is recorded as activated that the model did not itself ask for |
| C13b `skills_activation_recorded` | a body the model *did* pull is not silently dropped |
| C14 `skills_body_delivered` | a skill's **body** reached the model, by any channel |
| D1 `delivery_channel` | …and each channel delivers **on its own**, with the other switched off |
| J1 `journey` | the whole path: an agent on the platform → `decimalai init` → a file that runs → its prompt and a skill body in front of the model |

C7b is the second clause of C7, split out so a framework with no degenerate form
can declare that one clause N/A without silencing the first.

C13 and C13b grade the third rung of the skills ladder — *offered*, *delivered*,
*activated* — from opposite sides, and the asymmetry is deliberate. A fabricated
activation is strictly worse than a missing one: downstream it is
indistinguishable from a real one, it becomes a `TraceSkillActivation` row, and
it is blended back into ranking, so a skill that was merely pasted into a prompt
gets promoted over one that was used. C13 therefore applies to every rail; C13b
is declared N/A on a prompt-injection-only rail, where the model has no way to
ask for a body at all. Neither claims more than it measures: the strongest thing
either proves is that the model **asked for the body**, not that the skill
changed the output.

Exact token counts are **not** asserted here — a stub model's numbers are
arbitrary. Presence and plausibility are Tier A's job; exact counts are Tier B's.

### The delivery axis (D1)

C1–C14 grade one capture per driver, in whatever configuration that adapter
resolves to on its own. D1 is a **second axis over the same capture style**: one
child process per *(driver, body channel)*, with the **other channel switched
off** in that process's environment.

| | `injected` | `tool_loaded` |
|---|---|---|
| what carries the body | the router pastes it into the prompt | the model pulls it with `load_skill` |
| what is switched off | `DECIMALAI_LOAD_SKILL_TOOL=0` | `DECIMALAI_INJECT_SKILL_BODY=0` |
| langchain / anthropic | graded | N/A — framework limit, re-proven every run |
| openai-agents / pydantic-ai | graded | graded |

Why it exists: C14 asks whether a body reached the model **at all**, which closes
"zero channels". The defect that produced C14 was *arithmetic* — langchain had
`inject_skill_body` defaulting False **and** no `load_skill` tool, and each half
was defensible alone. D1 closes the next shape of the same hole: one channel
contributing nothing while the other covers for it. Until this axis existed,
`inject_skill_body` appeared **zero times** in the whole suite — the body channel
had never been varied, so there was no cell for it to be wrong in.

Both settings are public SDK surface (`init(inject_skill_body=…)`,
`init(load_skill_tool=False)` / the two env vars), so every cell is a
configuration a user can actually be in — and every cell is also a **kill-switch
test**: the off channel must really be off, or the caller cut a token cost they
did not cut.

A separate process per mode is not tidiness. `DecimalConfig` reads the
environment once, at construction, and each adapter freezes the answer into a
module-level `SkillRouter` singleton, so a second mode in the same process would
be graded against the first mode's router.

**An N/A here has to prove itself on the run.** `Capabilities.__post_init__`
enforces that a reason *exists*, never that it is *true* — which is how C13b came
to be N/A on langchain by a sentence that was accurate and *was the defect*. So a
`FrameworkLimit` carries a file and a marker instead of prose, the driver ASKS
the adapter for the channel, and `contract._grade_framework_limit` refuses the
N/A unless the adapter emits its documented refusal **and** delivers nothing by
that channel. An adapter that grows the capability stops being excused.

### The journey axis (J1)

C1–C14 and D1 all grade **one adapter**, handed a `Ctx` by a driver that already
knows the agent's name, already holds the skills, and never once runs the
product's own entry point. J1 grades what a user actually does:

```
an agent exists on the platform, with a prompt somebody typed and skills somebody attached
   → decimalai init <name>            the REAL console entry point, as a subprocess
   → python agent.py                  the REAL generated source, under runpy, as __main__
   → the skill's knowledge is in the context the model is handed
```

It is hermetic for the same reason the rest of the tier is: the probe already
stands in for `api.decimal.ai`, so it was taught the three routes `decimalai
init` calls — `GET /api/v1/agents` (the unpaginated list it resolves the name
against), `…/{name}/skills` and `…/{name}/prompt` — and `journey.JourneyModel`
stands in for the provider the same way `drivers/_openai_wire.py` does. No
backend, no key, no network, no cost. Verified: with every non-loopback route
pointed at a dead proxy, both cells still pass in the same time.

Five clauses, in the order a user meets them:

1. `decimalai init` exits 0 and makes all three platform calls, all accepted.
2. It wrote a non-empty file.
3. That file runs, reaches the model, and its own stdout carries the answer —
   which is what catches the template that emitted a bare chat model with no
   loop, where `run()` returned `""` on every call with no error anywhere.
4. **The agent's stored prompt reached the model.** The scaffold deliberately
   never copies the prompt text into the file, so the sentinel can only be there
   because the generated file read it at run time. A file that drops its
   `load_agent()` call still runs, still traces and still delivers skills — and
   fails only here.
5. **The skill's body sentinel reached the model.** A menu row cannot contain it,
   which is what tells "the skill was offered" from "the skill was readable".
   Plus one weak trailer: the run produced an accepted trace naming the agent,
   because "The trace appears at → …" is the last thing `decimalai init` prints.

The stub model's script is derived from the **request**, never from the
framework: if the request offers a `load_skill` tool and no body has come back
yet, it asks for one, otherwise it answers. langchain's adapter registers no such
tool and must deliver by injection; openai-agents' does and delivers through the
loop; neither is scripted for by name. A per-framework script would be a driver
artifact in the one cell that exists to be channel-agnostic.

**Which frameworks get a cell** is read off `decimalai/cli/scaffold.py`
directly — `SUPPORTED_FRAMEWORKS` today is langchain and openai-agents. The other
seven drivers are a declared N/A whose reason is the SDK's own
`UNSCAFFOLDED_WITH_SEAM` / `NO_PROMPT_SEAM` classification, recomputed and
compared by `test_coverage.test_every_journey_na_is_declared_and_counted`. Write
a template and the exemption fails until the ledger line is deleted; the cell
starts being graded on the same commit.

**What it does not cover:** the model's *answer*. The model is a stub, so this
tier can prove the skill's body was in the context and can prove nothing about
whether a real model then used it. That needs a real model and belongs to
the live end-to-end tier that runs against a real backend and a real model.

### Nothing is skipped, and the holes are counted

Two ledgers, both in `na_ledger.py`, both compared to the computed set in both
directions by `test_coverage.py`:

* `DECLARED_NA` — every `driver:item` the drivers would grade N/A, with the flag
  granting it, plus `NA_BUDGET` as a number a reviewer sees move in a diff.
* `DECLARED_DELIVERY_NA` — the same for delivery channels.
* `DECLARED_JOURNEY_NA` — the same for frameworks `decimalai init` writes no file
  for. Unlike the other two, no line in it is this suite's judgement: the set is
  recomputed from the shipped scaffold ledger.

And every skip in the package goes through `na_ledger.skip_declared`, which
**fails** rather than skips when the key is not declared; the only exceptions are
the three `ENVIRONMENT_SKIPS` (the docs repo, the platform repo, an opt-in env
var), and `test_no_undeclared_skip_call_sites` fails on any new `pytest.skip`
that is neither. `pytest -q` printing `102 skipped` next to a green bar is read
by nobody, so a skip has to be one somebody declared.

`has_skills_rail` is additionally cross-checked against the SDK's *own* ledger of
seam-carrying frameworks — `decimalai/cli/scaffold.py`'s `SUPPORTED_FRAMEWORKS` /
`UNSCAFFOLDED_WITH_SEAM` / `NO_PROMPT_SEAM`. Two independent sources for the same
claim, one of them the code users actually run.

`known_delivery_failures.txt` is the delivery axis's debt ledger, with the same
contract as `known_failures.txt`: recorded once, self-cleaning (a listed cell
that starts passing fails the build), and never a pressure valve for a new red.

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

The **pinned** lane is the gate; the **floating** lane is advisory. Both ride the
release tooling that already existed rather than growing a parallel copy. Nothing here is a second matrix, a second lockfile or a
second set of assertions.

| | Pinned lane | Floating lane |
|---|---|---|
| Runs in | `ci.yml`, every push and PR | `conformance-latest.yml`, nightly |
| Versions from | pinned constraints + the SDK's declared floors | newest, resolved live |
| Which frameworks | the `conformance-tests` extra | the full floating matrix |
| A red row | **blocks the release** | opens an issue; blocks nothing |

The mechanism is one command. Both lanes shell
`pytest tests/conformance -m conformance`; the floating lane simply upgrades
every framework to newest inside a clean-room wheel venv first, then runs the
identical phase with different versions underneath. The tests do not know which
lane they are in — the advisory/blocking distinction is structural (the nightly
is its own scheduled lane and was never a PR check), not a flag the assertions
read.

The floating matrix grew a second block for the frameworks that have a
conformance driver but no live-matrix lane (crewai, google-adk, llama-index-core,
pydantic-ai, anthropic, and the OpenInference instrumentors the OTel-routed
drivers read). Floating those is cheap precisely
because Tier A is hermetic — no model spend, no backend.

Two commands worth knowing:

```bash
# pinned, ~30s, no key, no backend — "did I break an adapter?"
python -m pytest tests/conformance -q -m conformance

# floating: newest frameworks vs the contract, still no key and no backend
#   → the nightly `Conformance (latest frameworks)` workflow, or
#     workflow_dispatch it from the Actions tab
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
| pydantic-ai, claude-agent-sdk | `>=0.1.0` | none | Both floors are unverified `>=0.1.0` — the shape of a floor nobody has run. Verify before claiming them. |

### When a new framework version breaks a row

1. The floating lane goes red and opens an issue. **The pinned gate stays
   green**, so releases are not blocked. This is the whole point of the split.
2. Read the nightly run's version diff (`changed` / `added` against the
   recorded baseline) to see which upstream release moved.
3. Decide which of two things is true:
   - **The adapter is wrong.** Fix `decimalai/<adapter>.py`. The row goes green
     in both lanes and nothing else changes.
   - **The new upstream behaviour is correct and ours was.** Record a new
     baseline, and raise the floor in `pyproject.toml` if the old version can
     no longer work.
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
| the docs capability table | `https://docs.decimal.ai/sdk/python/frameworks`, vendored into `drivers.ADVERTISED_SNAPSHOT` | snapshot yes, re-derivation no |
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
  stringified under an invented key. Worth a C14; deliberately out of v1 scope.
  (Renumbered from "C13", which now names the activation item above.)
- **What the model DOES with a delivered skill.** C14, D1 and J1 all stop at the
  same line: the body was in the context the model was handed. The model here is
  a stub, so nothing in Tier A says a real one read it, followed it, or answered
  differently for having it. That question needs a real model and belongs to
  the live end-to-end tier that runs against a real backend and a real model.
- **The journey on a framework with no scaffold.** J1 only exists where
  `decimalai init` writes a file — two frameworks today. The other seven are a
  declared N/A, and the exemption is only as good as the day somebody writes
  their template.
- **The journey's own install set.** J1's langchain cell needs `langchain` and
  `langchain-openai` — the umbrella package and the provider binding, neither of
  which `decimalai[langchain]` pulls (that extra is `langchain-core` only). They
  come from `[dev]`, and both Tier A jobs install `[dev]` alongside
  `[conformance-tests]`, so the cell does run in CI. A job that installed
  `[conformance-tests]` *alone* would skip it — loudly under
  `DECIMAL_CONFORMANCE_REQUIRE_ALL=1`, silently without. If the extras are ever
  reorganised, `journey_requirements()` names exactly what is missing.
- **Exact token counts, cost, latency.** Tier B's job.
- **Cross-driver isolation.** Fixed: each driver's phases now run in their own
  process (see [One driver, one process](#one-driver-one-process)), so a late
  trace or a global instrumentor from one adapter can no longer reach another's
  probe. What remains is isolation *within* one driver's process, which is
  deliberate — C6, C7 and C9 exist to grade it.
- **This README does not carry the result matrix.** It would be stale within a
  week. Run the suite; the matrix it prints is the only authoritative one.
