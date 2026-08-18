# Releasing decimalai

How a new version of the `decimalai` package reaches PyPI. Releases are cut by maintainers with
push access; this file documents the process so a contributor can see what a change has to survive.

> **PyPI is append-only.** A version number can never be reused, overwritten, or re-uploaded — even after
> you "delete" a release. If you ship a mistake, the only remedy is a *new* version. Treat the publish as
> irreversible.

## How publishing works

Publishing from CI via PyPI Trusted Publishing is the **intended end state**, and
`.github/workflows/publish.yml` already implements it correctly: a published GitHub Release is the
trigger, the workflow exchanges a short-lived OIDC identity against a publisher row naming this repo,
`publish.yml`, and the `pypi` environment — no token anywhere — and PyPI records an attestation
automatically. That path is better than a laptop upload for two reasons that are not matters of taste:
the tag *is* the trigger (so a shipped version can never be untagged), and provenance is free.

**It has one precondition: GitHub Actions must actually run for this repo.** Today it does not, so
**0.10.3 ships from `scripts/release.sh`.** Check which regime you are in before you cut anything
(see below) — this section is a policy with a switch, not a standing preference.

### The precondition, and why it is currently unmet

Actions jobs on `decimal-labs/decimalai-python` are **not started at all**: billing for this account
has failed, and this is the last repo in the org that is still **private**. Private repos consume
billable Actions minutes; public repos get them free. That single difference is why the OIDC path
succeeded on `decimalai-mcp 0.1.3`, `agentversion 0.2.3`, and `skillevaluation 0.7.1` — all three of
those repos are **public**. Those releases are real proof that the mechanism works; they are *not*
evidence that it works here.

So the precondition is met when **either** of these becomes true:

- this repo is made **public** (free Actions minutes), or
- **org billing is restored** (a valid payment method on the `decimal-labs` billing settings).

Until then, cutting a GitHub Release for a new version does nothing useful: `publish.yml` fires, its
`test` job never starts, `publish` (`needs: test`) is therefore skipped, and **nothing reaches PyPI**.
You are left with a dangling tag and a Release for a version that does not exist.

### Check which regime you are in — before cutting a release

The failure reason is **not** in `gh run list` (which just says `failure`) and **not** in the job logs
(there are none). It appears only in the check-run *annotations*:

```bash
run=$(gh run list -R decimal-labs/decimalai-python -w CI -L1 --json databaseId -q '.[0].databaseId')
job=$(gh api repos/decimal-labs/decimalai-python/actions/runs/$run/jobs -q '.jobs[0].id')
gh api repos/decimal-labs/decimalai-python/check-runs/$job/annotations -q '.[].message'
```

- Prints *"The job was not started because recent account payments have failed or your spending limit
  needs to be increased"* → **precondition unmet.** Publish locally with `scripts/release.sh`, then
  tag by hand.
- Prints nothing and jobs run real steps → **precondition met.** Run the local live-LLM gate, then cut
  the GitHub Release and let CI upload.

> **Diagnostic worth remembering.** A CI job that fails in a few seconds with **zero steps executed and
> no logs** has not run your tests — `.jobs[].steps` is an empty list, and the log API returns
> `BlobNotFound` because nothing ran to produce any. That is an infrastructure failure, not a test
> failure; the annotations command above is what tells you *which* infrastructure failure. Don't spend
> release attempts debugging a red test that passes in a clean venv on every supported Python.

### What the local path costs

Both costs are real, and both are why this reverts to CI as soon as the precondition is met:

- **No attestation.** A `twine` upload from a laptop cannot produce one. Of 22 published `decimalai`
  files, only **2** carry provenance — `0.4.0`'s wheel and sdist, published 2026-06-08 through this
  very workflow. Everything since has none. (That one success is also proof the PyPI publisher row for
  this repo is configured correctly; Actions availability is the only thing missing.)
- **Tagging is best-effort and fails quietly.** `scripts/release.sh` attempts `gh release create` as
  its last step with stderr suppressed, and treats failure as non-fatal because the upload already
  happened. When it fails you get a shipped, untagged version. The repo has exactly one tag, `v0.10.0`,
  and two versions in the 0.10 line went out without one:
  - **0.10.2** — on PyPI (uploaded 2026-08-15), never tagged;
  - **0.10.1** — never tagged *and never uploaded*; it has a CHANGELOG entry and a "Release 0.10.1"
    commit but no artifact on PyPI, so the version was skipped entirely.

  **So after a local publish, verify the tag exists and create it yourself if it doesn't.**

### The gates

The live-LLM gate is local in **both** regimes — CI has no provider keys and no backend, so it can
never make real model calls. Only the *upload* moves.

| Gate | Where | Checks | Blocks the release? |
|---|---|---|---|
| **Live-LLM** | **Local — `scripts/release.sh`** | real model calls through a clean-room wheel | yes — always run before publishing, in either regime |
| **No-model** | CI — `publish.yml` `test` job | unit + contract tests on Python 3.10–3.12 | only when Actions runs. **Today it cannot start, so nothing gates the unit tests automatically.** |

> **While Actions is blocked, run the no-model suite yourself — `scripts/release.sh` does not.**
> The script's step 2 only builds the wheel and smoke-tests it (import, `__version__`, CLI); it never
> invokes `pytest`. So the unit and contract tests are currently checked by *no one* unless you run
> them. Do it before publishing:
>
> ```bash
> pytest tests/ -q      # expect: NNNN passed, NN skipped
> ```
>
> Verified 2026-08-16 on Python 3.12: `1414 passed, 23 skipped, 192 deselected in 19.62s`. Note that
> `uv run --extra dev pytest` currently fails to resolve (the `adk` and `crewai-tests` extras pin
> incompatible `opentelemetry-api` ranges), so use an environment that already has the test deps.
> This is one gate CI would give you across 3.10/3.11/3.12 for free — a local run covers only your
> interpreter.

## Prerequisites (for the local live gate)

1. [`uv`](https://docs.astral.sh/uv/) installed (the script uses `uv build` and `uv run`).
2. `gh` installed and authenticated (`gh auth status`) — the script cuts the Release with it.
3. A **running DecimalAI backend** the gate can send traces to. Point `DECIMAL_BACKEND_URL` at it
   (default `http://localhost:8000`).
4. A provider key exported in your environment — `GEMINI_API_KEY` and/or `OPENAI_API_KEY`.
5. The **live-gate harness**, which builds a clean-room wheel and drives it against real models. It is
   maintainer-only tooling and is not part of this repo; point `DECIMAL_RELEASE_GATE_DIR` at your
   checkout of it. Without it, `scripts/release.sh` stops and tells you to either set that variable or
   pass `SKIP_LIVE_LLM_GATE=1` (see below).

## Versioning model

The package version lives in **two places that must match**:

| Number | Lives in |
|---|---|
| `version` | `pyproject.toml` |
| `__version__` | `decimalai/__init__.py` |

Bump **both** to the same SemVer string. You do not have to keep them in sync by eye — the release
script's clean-env smoke test imports the built wheel and asserts `decimalai.__version__` equals the
`pyproject.toml` version, so a mismatch fails the release before anything is published.

The package is pre-1.0 (`0.x`) while the API settles.

## Cutting a release

Steps 1–6 are identical in both regimes. Only the last step differs: **today** (Actions blocked) run
`scripts/release.sh` and then confirm the tag; **once the precondition is met**, run the gate, then cut
the GitHub Release and let `publish.yml` do the upload.

1. **Pick the next version** (SemVer).
2. **Bump it in both places**: `pyproject.toml` `version` and `decimalai/__init__.py` `__version__`.
   `tests/test_audit_improvements.py::TestVersion::test_version_matches_pyproject` asserts the two
   *agree*, so bumping only one is caught. Do not "improve" that test into pinning the literal current
   version — it used to do exactly that, which made every release fail CI until someone remembered to
   edit the line, and it blocked the 0.9.1 cut in exactly that way.
3. *(Optional)* add a `CHANGELOG.md` entry — `## [X.Y.Z] — YYYY-MM-DD`. The script warns if a CHANGELOG
   exists without an entry for the version, but does not require one.
4. **If you touched the README**: make every link an **absolute** `https://github.com/...` URL — relative
   `./` links render broken on PyPI.
5. **Docs sync** (the docs site's source, ships with or right after the release):
   - add an `<Update>` entry to `changelog.mdx` — the page promises SDK release tracking;
   - if the release adds/changes CLI commands, update `sdk/cli.mdx`;
   - if `requires-python` changed, update the floor noted in `quickstart.mdx`, `faq.mdx`, and this
     repo's README.
6. **Commit and push** the release commit. The script refuses to release a commit that is not on the
   remote (otherwise `gh` would tag the wrong revision).
7. **Publish**, per the regime you confirmed above:
   - **Actions blocked (today, and the path for 0.10.3)** — run the release script from the repo root,
     then verify the tag landed:
     ```bash
     ./scripts/release.sh
     ```
   - **Actions healthy** — use the script for its gates only. It has no gate-only flag: run it, let
     the live-LLM gate go green, then answer anything other than `yes` at the
     `Type 'yes' to publish:` prompt, which aborts before the upload (`Aborted — nothing released.`).
     Then cut the Release, which is the trigger:
     ```bash
     gh release create "v$VERSION" --generate-notes   # CI uploads via OIDC and attests
     ```

### What the script does

In order — cheap checks first, so a trivial error never wastes model budget:

1. resolves the version and refuses to proceed if it already exists on PyPI, if `vX.Y.Z` is already
   tagged, or if HEAD is not on the remote;
2. builds the wheel, runs `twine check`, and smoke-tests the built wheel (import, `__version__`, and the
   `decimalai` CLI) in a throwaway environment;
3. runs the **live-LLM release gate** (see prerequisite 5) — a clean-room wheel, installed as a user
   would install it, exercised against **real models**;
4. pauses for a typed `yes`, then **uploads to PyPI with `twine`**, verifies
   `pypi.org/pypi/<name>/<version>/json` returns 200, and finally tags **GitHub Release `vX.Y.Z`** at the
   current commit. A failure to tag is reported but is NOT fatal — the package is already published at
   that point.

The `twine upload` is the irreversible step. `twine` itself refuses if the version already exists, so a
re-run after a partial failure is safe.

**Step 4's tag is the half that silently doesn't stick** (`gh release create` runs with stderr
suppressed). Always confirm it afterwards, and create it by hand if it's missing:

```bash
git fetch --tags origin && git tag -l "v$VERSION"          # expect: v<version>
gh release create "v$VERSION" --generate-notes             # if the tag is absent
```

Creating that Release also fires `publish.yml`. While Actions is billing-blocked the run fails
instantly with zero steps — that is cosmetic here, not a failed publish; the package is already live.
Once Actions works again, the run is a genuine no-op because the version already exists on PyPI.

## Providers the live gate exercises

By default the gate runs **Google/Gemini only**, because the OpenAI test key is currently quota-blocked.
To include OpenAI once it has quota:

```bash
RELEASE_GATE_PROVIDERS="google openai" ./scripts/release.sh
```

Provider-native frameworks are only tested against their own model (e.g. `openai_agents` → OpenAI,
`adk` → Gemini); generic frameworks are tested against both. So a Gemini-only run does not exercise the
OpenAI-native lanes — use `"google openai"` for a full release when the key is available.

## Skipping the live gate

The live-LLM gate needs the harness from prerequisite 5 and at least one provider with quota. Two cases
leave you without it: you don't have the harness, or **every** provider is quota-blocked
(`429 RESOURCE_EXHAUSTED` from Gemini *and* OpenAI — easy to hit on a free tier's 20-requests/day limit).
For those there is an explicit escape hatch:

```bash
SKIP_LIVE_LLM_GATE=1 ./scripts/release.sh
```

This bypasses **only** the real-model T2 check (step 3). The cheap structural gates — step 1
(version/tag/remote) and step 2 (build, `twine check`, clean-env import + CLI + `__version__` parity) —
**always** run, and the typed-`yes` confirmation still gates the irreversible publish. Use it sparingly:
the SDK then ships without live-provider validation, so reserve it for changes that don't touch
provider/runtime behavior (routing-id plumbing, schema/typing, docs) or for a hotfix that can't wait for
quota to reset. Prefer a real gate run once quota clears. The script prints a loud warning and records
the skip in its output.

## Manual fallback

`scripts/release.sh` is the supported path. If you must upload by hand, you need a PyPI API token in
`~/.pypirc` (`chmod 600`, never commit it):

```ini
[pypi]
  username = __token__
  password = pypi-<your-token-here>
```

```bash
rm -rf dist && uv build
uvx twine check dist/*
uvx twine upload dist/*        # PERMANENT — cannot be undone
# verify (the version endpoint updates fastest; expect 200):
curl -s -o /dev/null -w '%{http_code}\n' https://pypi.org/pypi/decimalai/<version>/json
```

A manual upload skips every gate — run `./scripts/release.sh` instead wherever you can.

## Notes & gotchas

- **Check links logged out.** README badges, `examples/` Colab links and docs-site cards that point
  at `github.com/decimal-labs/decimalai-python` should be verified in a signed-out browser, not your
  own — a link that resolves for you can still be broken for everyone else.
- **PyPI cache lag.** The top-level `https://pypi.org/pypi/decimalai/json` can stay cached on the previous
  version for a minute or two after upload; the version-specific `.../<version>/json` endpoint reflects
  new releases almost immediately. The top-level `releases` map lags much longer — it can omit a version
  that shipped days ago. For an authoritative version list use the simple index:
  `curl -H 'Accept: application/vnd.pypi.simple.v1+json' https://pypi.org/simple/decimalai/`.
- **Checking whether a release got an attestation.** The `provenance` field in the PyPI JSON API is
  **not** a reliable signal — it reads `null` even for files that demonstrably have one. Use the
  integrity endpoint, which returns 200 when an attestation exists and 404 when it does not:
  ```bash
  curl -s -o /dev/null -w '%{http_code}\n' \
    https://pypi.org/integrity/decimalai/<version>/decimalai-<version>-py3-none-any.whl/provenance
  ```
- **Trusted Publisher setup.** Because the project already existed on PyPI before the workflow did, the
  OIDC publisher has to be attached by hand under PyPI → *Manage* → *Publishing* (workflow `publish.yml`,
  environment `pypi`).
