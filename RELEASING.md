# Releasing decimalai

How a new version of the `decimalai` package reaches PyPI. Releases are cut by maintainers with
push access; this file documents the process so a contributor can see what a change has to survive.

> **PyPI is append-only.** A version number can never be reused, overwritten, or re-uploaded — even after
> you "delete" a release. If you ship a mistake, the only remedy is a *new* version. Treat the publish as
> irreversible.

## How publishing works

`decimalai` publishes **from CI, via PyPI Trusted Publishing**, triggered by publishing a GitHub
Release. There is no token anywhere: the workflow exchanges a short-lived OIDC identity that PyPI
verifies against a publisher row naming this repo, `publish.yml`, and the `pypi` environment.

> **Inverted 2026-08-16.** This used to publish by `twine` from a maintainer's machine, on the
> reasoning that *"CI availability must never block a release."* That reasoning came from a period
> when CI was unavailable here for reasons unrelated to the code — that is no longer the case, and
> the OIDC path has now been proven three times (`decimalai-mcp 0.1.3`,
> `agentversion 0.2.3`, `skillevaluation 0.7.1`, each with a verified attestation).
>
> What the local path cost, measured: **0 of 22 published `decimalai` files carry provenance**, and
> **0.10.1 and 0.10.2 were never tagged** — two shipped versions with no commit marked in the repo,
> because nothing in a laptop upload forces a tag. The CI path cannot have either problem: the tag
> *is* the trigger, and the attestation is automatic.
>
> The one durable reason for a local step remains and is unchanged — **CI has no provider keys, so
> the live-LLM gate cannot run there.** That argues for running the gate locally and letting CI do
> the *upload*; the two are separable. Run `scripts/release.sh`'s gate, see it green, then cut the
> Release. `twine` stays documented below as the fallback for when PyPI or Actions is unavailable.

It used to work the other way round: a published GitHub Release triggered
`.github/workflows/publish.yml`, which uploaded via OIDC Trusted Publishing. That made every release
depend on a hosted CI service being available — and when CI was unavailable the release simply could not
happen, even though every check it ran also runs locally. So the upload moved to the maintainer's
machine. If CI is healthy it still runs `publish.yml` on the release event; that upload is a harmless
no-op, because the version already exists. **CI availability must never block a release.** (The sibling
`agentversion` and `skillevaluation` packages already published this way — this package was the odd one
out.)

> **Diagnostic worth remembering.** A CI job that fails with **zero steps executed and no logs** has not
> run your tests — the log API returns `BlobNotFound` because nothing ran to produce any. That is an
> infrastructure failure, not a test failure. Don't spend release attempts debugging a red test that
> passes in a clean venv on every supported Python.

That leaves two gates:

| Gate | Where | Checks | Blocks the release? |
|---|---|---|---|
| **No-model** | CI — `publish.yml` `test` job | unit + contract tests on Python 3.10–3.12 | yes, but only *after* the Release is cut (it runs on the release event). `ci.yml` runs the same tests on every PR, which is where you actually want to see them go green. |
| **Live-LLM** | **Local — `scripts/release.sh`** | real model calls through a clean-room wheel | yes — the maintainer runs it **before** cutting the Release |

CI has no provider keys and no backend, so it cannot make real model calls. That is the whole reason the
live gate is a **local required step**: you run it yourself, see it go green, and only then cut the
GitHub Release.

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
5. **Docs sync** (the `decimalai-docs` repo, ships with or right after the release):
   - add an `<Update>` entry to `changelog.mdx` — the page promises SDK release tracking;
   - if the release adds/changes CLI commands, update `sdk/cli.mdx`;
   - if `requires-python` changed, update the floor noted in `quickstart.mdx`, `faq.mdx`, and this
     repo's README.
6. **Commit and push** the release commit. The script refuses to release a commit that is not on the
   remote (otherwise `gh` would tag the wrong revision).
7. **Run the release script** from the repo root:
   ```bash
   ./scripts/release.sh
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

- **Public repo, public package.** Both the GitHub repo and the PyPI package are public, so the README
  badges, the `examples/` Colab links and the docs-site Colab cards all resolve. If any of them 404, that
  is a real breakage.
- **PyPI cache lag.** The top-level `https://pypi.org/pypi/decimalai/json` can stay cached on the previous
  version for a minute or two after upload; the version-specific `.../<version>/json` endpoint reflects
  new releases almost immediately.
- **Trusted Publisher setup.** Because the project already existed on PyPI before the workflow did, the
  OIDC publisher has to be attached by hand under PyPI → *Manage* → *Publishing* (workflow `publish.yml`,
  environment `pypi`).
