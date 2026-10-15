# Native skill loaders

Do the eight native skill loaders read the folders a DecimalAI install writes,
under the right names, with the stamp intact, the resources reachable and the
scripts runnable? This suite answers that on every `pytest tests/` run, offline,
with no keys.

| loader | environment | documented entry |
|---|---|---|
| `claude-agent-sdk` (the bundled Claude Code CLI) | openai3 | `ClaudeAgentOptions(plugins=[{"type": "local", "path": str(decimalai.skills_dir().parent)}])` |
| `google-adk` | openai3 | `SkillToolset(skills=load_skills_from_dir(decimalai.skills_dir()), code_executor=...)` |
| `openai-agents` (sandbox `Skills`) | openai3 | `Skills(lazy_from=LocalDirLazySkillSource(source=LocalDir(src=decimalai.skills_dir(relative_to=Path.cwd()))))` |
| `deepagents` | openai3 | `create_deep_agent(model, skills=[str(decimalai.skills_dir())], backend=LocalShellBackend(..., virtual_mode=False))` |
| `agent-framework` (agent-framework-core) | openai3 | `SkillsProvider.from_paths(str(decimalai.skills_dir()), script_runner=...)` |
| `pydantic-ai-harness` | openai3 | `Skills(str(decimalai.skills_dir(relative_to=workspace)))` inside `LocalWorkspace(workspace)` |
| `strands-agents` | openai2 | `AgentSkills(skills=str(decimalai.skills_dir()), strict=True)` |
| `crewai` | openai2 | `Agent(..., skills=[decimalai.skills_dir()])` |

`decimalai.skills_dir()` is stood in for by `source.skills_dir()` until the SDK
exports it; switching is a one-file change.

## Why it is not a conformance column

`tests/conformance` grades the traces an adapter sends on the wire. This grades
files, read by loaders that mostly have no DecimalAI adapter at all, and the Claude
check must run the real CLI binary (the CLI is the loader), where conformance stubs
the CLI transport. So it copies conformance's rules instead of joining it:

- **One file holds every assertion**: `contract.py`. A loader module only observes
  and returns a dict; `test_native_coverage.py` fails on an `assert` or a test
  framework import in `loaders/` or `stubs/`.
- **Every check is seen to fail**: `test_contract.py` hands the contract synthetic
  defects (a bare `Bash(*)` on a skill the runtime run never loads, a delivery cut
  short after the sentinel, a plugin that does not load from a subdirectory) and
  requires each to fail, so a check weakened on a day every loader is clean still
  turns the build red.
- **One loader, one child process** (`_child.py`, started by `runner.py`), JSON
  back. CrewAI's event bus, Strands' plugin registry, ADK's runners and every
  OpenTelemetry provider are process-global. The parent imports no framework.
- **Every loader belongs to an environment** (`LOADER_ENVS` in
  `tests/conformance/environments.py`, beside `DRIVER_ENVS`), whose extras must
  install it; the loaders graded in the other environment are named on every run.
- **Nothing skips silently.** Every skip goes through `ledger.skip_declared` or is
  a declared environment skip; an undeclared `pytest.skip` here fails the build.
- **Every exception re-proves itself on every run** (the ledger, below).
- **Collected by default**, not behind a marker: it rides the `pytest tests/` run
  that CI, the publish gate and the landing gate already make in both environments.

## Environments

Six loaders run in openai3 and two in openai2. CrewAI pins `openai<3`. Strands
needs no openai, but it caps `mcp<2.2`: in openai3 that would move `mcp` (2.3.0 to
2.1.1) under claude-agent-sdk and openai-agents, while openai2 already holds an
`mcp` it accepts, so there it only adds packages. The three openai3 additions
(`deepagents`, `agent-framework-core`, `pydantic-ai-harness`) and Strands are in the
`conformance-tests` / `conformance-tests-openai2` extras, so every gate installs
them. A venv built from the extras as they were before this suite **refuses to
start** under `DECIMAL_TEST_ENV` (by design: a stale venv says it is stale);
rebuild it:

```bash
uv pip install -p .venv/bin/python -e ".[dev,conformance-tests]"
uv pip install -p .venv-openai2/bin/python -e ".[conformance-tests-openai2]"
```

Run it alone:

```bash
DECIMAL_TEST_ENV=openai3 .venv/bin/python -m pytest tests/native_loaders -q -rsx
DECIMAL_TEST_ENV=openai2 .venv-openai2/bin/python -m pytest tests/native_loaders -q -rsx
```

About 7-10 s per environment for both projects, four children at a time
(`DECIMAL_NATIVE_LOADER_JOBS`; per-child timeout `DECIMAL_NATIVE_LOADER_TIMEOUT`,
default 120 s, after which the cells ERROR, naming the loader; never a skip).

## What is graded

Two projects per run (`conftest.py`):

- **the graded project**, which every loader must read cleanly. Today it is the
  oracle (`oracle.py`): an independent, deliberately plain rendering of the
  on-disk contract, format 1. When `decimalai skills add` and
  `decimalai skills install --frozen` exist, `source.graded_project` writes the
  folder with them instead, and the oracle becomes the byte-for-byte reference.
- **the negative control** (`fixtures/v0139_project`): what SDK 0.13.9's own
  `disk_export.py` writes for the same three skills, frozen. It is graded against
  the same expectation and must fail, in the cells `ledger.NEGATIVE_CONTROL_PREDICTED`
  names (and exactly the cells `NEGATIVE_CONTROL_RED` records). It stays for good:
  weaken N1, N6 or N10 and it turns red. The fixture stores `.agents/` and
  `.decimal/` as `agents/` and `decimal/`, because this repository's `.gitignore`
  excludes those names at every depth; `source.negative_control_project` restores them.

The three skills: `acme-refund-policy` (registry name `acme/refund-policy`;
references, an executable script, a scoped `allowed-tools`); `acme-returns-desk`
(user-only); and a registry name over 64 characters, installed as 57 characters
plus six hex of `sha256(skill_id)`, imported from GitHub and so `foreign` (its
stored `Bash` and `WebFetch` dropped). Each body has its own sentinel.

| item | asserts |
|---|---|
| N1 | the loader lists exactly the lock's install names (7 of the 8 drop or rename a bad folder without raising), and the model's listing holds every model-invocable one |
| N2 | the loader's own skill-warning channel is quiet, apart from a self-cleaning allowlist |
| N3 | the model got the target's own body (its sentinel, and the loader's parsed body verbatim, so a delivery cut short fails), and `sha256(body.strip())` equals the stamp |
| N4 | the resource, read through the loader's own route, equals the pinned file |
| N5 | the script runs through the loader's own route and prints the expected line |
| N6 | the stamp is a flat map of strings and equals the lock (ids, version, body hash, registry name, format, invocation) |
| N7 | the user-only skill is listed but not offered, and a model call for it is refused (Claude Code); elsewhere a ledgered upstream fact |
| N8 | Claude Code: the scoped rule pre-approves the target's script; a command outside it is denied; the foreign skill's own script is denied; and in every installed `SKILL.md`, no bare Bash (`Bash`, `Bash(*)`, `Bash(*:*)`: all unrestricted to Claude Code) and no comma between tools (import damage: Claude Code reads `Bash,` as `Bash`) |
| N9 | Claude Code: the plugin is `decimalai-skills` at `lock-<12 hex of the recomputed lock_hash>`, loaded from the project's `.agents` (as the CLI reports it), with no plugin errors; and again, with every skill, when the snippet runs from a subdirectory of the project |
| N10 | the script is executable in the copy the loader runs |

The plugin is `decimalai-skills`, not `decimalai`: that is the marketplace plugin's
name, and two plugins with one name replace each other in Claude Code.

Every scripted model uses only what its loader tells it: a skill directory, base
directory or `Location:` line from the loader's own output, never a path the test
already knows. Where a loader dropped the target, the steps after it have nothing to
act on, and N3-N5 say so.

## The ledger (`ledger.py`)

| kind | today | rule |
|---|---|---|
| `DECLARED_NA` | CrewAI N4 (no resource reader), N5 (no script runner) | skipped only after its proof holds this run; the day CrewAI ships the capability, the cell fails |
| `UPSTREAM_GAPS` | N7 on the seven loaders that ignore `disable-model-invocation`; openai-agents N10 (materialization drops the exec bit) | xfail while the strict check fails *in the documented way*; if it passes, the cell fails until the line is deleted |
| `PROOFS` | MAF ships no runner (N5 uses the suite's); openai-agents and pydantic expose no metadata (N6 reads the file the loader reports); Strands' strict parse (N1) | checked inside the cell |
| `NOISE_ALLOWLIST` | pydantic-ai-harness's "Ignoring unsupported ... frontmatter fields" | must still occur on the graded project |

The loader x single-defect matrix is the release gate's, not the per-commit run's
(its cells move with upstream releases):

```bash
DECIMAL_TEST_ENV=openai3 .venv/bin/python -m tests.native_loaders.matrix --check-ledger
DECIMAL_TEST_ENV=openai2 .venv-openai2/bin/python -m tests.native_loaders.matrix --check-ledger
```

Fourteen defects (list-typed fields, nested or unquoted metadata, namespaced,
uppercase, over-long and double-hyphen names, folder and name disagreeing, symlinked
folders and resources, an unknown key, a second `SKILL.md` inside a folder) in
`ledger_matrix.json`, with the versions they were measured at. A changed cell
exits 1 and is a finding; re-record with `--update-ledger`. `decimalai skills
verify` must reject every defect that any column does not mark `OK`. Per commit, a
one-cell run in each environment proves the command works and fails on a flipped
cell.

## Hazards each child handles (`runner.child_env` and the loader modules)

- A parent Claude Code session leaks `CLAUDE*` / `ANTHROPIC*` into the CLI: all
  dropped; the Claude child gets a local Messages stub and a dummy key.
- The CLI's default permission mode ("auto") makes a classifier side call:
  `permission_mode="default"`, `setting_sources=[]`.
- `~/.claude`, and CrewAI writing under `$HOME`: a fresh `HOME` and
  `CLAUDE_CONFIG_DIR` per child.
- Network: a closed-port proxy; and git over SSH ignores proxies, so
  `GIT_SSH_COMMAND=/usr/bin/false`, no global or system git config, and Claude
  Code's official marketplace auto-install off.
- The CLI sends the model a budgeted subset of skills: N1 checks the model-facing
  listing, not only `init.skills`.
- A relative `--plugin-dir .agents` from a subdirectory loads nothing, with exit 0
  and no stderr (only a `path-not-found` in `init.plugin_errors`). The snippet
  passes the absolute path `skills_dir().parent`, and the Claude child runs it a
  second time from `<project>/src` (made for the run, removed after it) for N9.
- Machine-level Claude Code managed settings cannot be isolated by `HOME`: the
  check asserts only on its own plugin and skills, and N9 prints `init.plugins`.
- macOS `sandbox-exec` (openai-agents): this venv's `bin/` first on `PATH`, and,
  because each child's `HOME` is empty, a read-only grant for this interpreter's own
  install (a real user's HOME grants it; the documented snippet needs none).
- openai-agents lists nothing for a source outside the cwd, behind a symlink or the
  `/tmp` alias: every path is resolved and the child's cwd is the project.
- Strands' default model is Bedrock: always the scripted model. `file_editor`
  rejects relative paths: an absolute root.
- CrewAI's event bus is asynchronous: `crewai_event_bus.flush()` before reading.
- Tracing and telemetry are off for every framework (LangSmith, CrewAI,
  OpenTelemetry, the CLI).

## The docs table

`NATIVE_LOADERS_SNAPSHOT` (in `loaders/__init__.py`) is what the docs page
`guides/native-skill-loaders.mdx` advertises: a literal set of ids, edited by hand
when the docs table changes, and required to equal the registered loaders (derived
from them, it could never disagree). When the docs checkout is beside this one and
the page exists, `test_loader_table_matches_the_docs` reads the page's
table (header `| Loader |`, each loader's id in inline code in the first column,
for example ``| Claude Code (`claude-agent-sdk`) | ... |``) and requires the two to
agree. Until then the test skips with that reason, and the in-repo guards stand in.

## Adding a loader

1. `loaders/<module>.py` with `NAME`, `discover(ctx)` and `observe(ctx)`, running
   the documented snippet built on `skills_dir()`; its stub model in `stubs/`.
2. A `Loader` row in `loaders/__init__.py`, and its environment in `LOADER_ENVS`;
   its distribution in that environment's conformance extra.
3. Run both environments. A cell the loader cannot be graded on gets a ledger line
   with its proof, never a skip.
4. `--update-ledger` in its environment.

## What this does not tell you

- Whether a real model uses the skill: the models are scripted.
- In-process refresh after an update (MAF's cache, Strands' mid-process rewrite,
  Claude's "already loaded"): each check is one cold process.
- Coding-agent runtimes with no offline loader to drive (Cursor, Copilot, Codex).
- Linux behaviour of the openai-agents sandbox (no confinement there) and Claude
  Code's Linux managed-settings path: the gate runs on macOS.
- Whether activations report the right identity: that is the adapters', in
  `tests/conformance`.
