# Changelog

All notable changes to `decimalai` are documented here. This project follows
[Semantic Versioning](https://semver.org/); pre-1.0, minor releases add features
and patch releases are fixes.

## 0.10.1 — 2026-08-13

### Fixed

- **`init(enabled=False)` is now a real kill switch.** The framework flags
  (`langchain`, `openai_agents`, `adk`, `llamaindex`, `claude_agent_sdk`,
  `otel`, `crewai`, `autogen`) were applied *outside* the `enabled` gate, so
  `init(enabled=False, langchain=True)` skipped only the client and then
  installed the adapter anyway — which calls the DecimalAI API to sync and pull
  skills, and writes `SKILL.md` files into your project. Turning the SDK off
  produced network traffic and new files on disk. The flags are inside the gate
  now: a disabled init installs no adapter, makes no API call and writes
  nothing. `enabled=True` is unchanged.

  ```python
  decimalai.init(api_key=..., enabled=False, langchain=True)  # now truly a no-op
  ```

### Changed

- **LLM judges now fail closed.** `Relevance`, `Factuality`, `Faithfulness`,
  `Toxicity` and `Conciseness` used to answer *any* evaluator failure — network
  error, revoked key, rate limit, exhausted quota, empty or unparseable
  completion — with `score=0.5, passed=True`. An outage therefore reported a
  PASS on every trace, including `Toxicity`, whose category is
  `quality:safety`. Those paths now return `passed=False, score=0.0`, keep a
  reason naming the error type, and set `metadata["evaluator_error"]` so you
  can tell an evaluator outage apart from a genuine quality failure. Same for
  the server-side path (`use_server=True`), and a backend check that omits
  `passed` no longer defaults to `True`. **If you gate a release or alert on
  these scores, expect judge outages to surface as failures instead of passing
  silently** — filter on `metadata["evaluator_error"]` to separate them.

- **The SDK no longer sends anything that identifies your machine.** Two places
  did. `skills sync` / `skills status` stamped an `install_label` that defaulted
  to your hostname; there is no default label now, and the key is omitted from
  the request body entirely unless you opt in with `DECIMALAI_INSTALL_LABEL`
  (the anonymous per-checkout `install_id` still does the drift attribution, so
  nothing about `skills status` changes). And registering an `@eval` sent the
  absolute path of your source file plus the process working directory
  (`source_location_extra.abs_path` / `.cwd_at_decoration`); only the line
  number is sent now. The repo-relative `source_location` is unchanged.

  ```bash
  export DECIMALAI_INSTALL_LABEL=ci-runner-7   # opt in, e.g. on a shared runner
  ```

- **`install()` on the framework integrations is now `instrument()`.** One word
  was doing two unrelated jobs inside one package: on a framework module it
  turned on TRACING, on `SkillRouter` it added a SKILL to a workspace. The skill
  sense is the one people arrive with — it is what every extension marketplace
  means — so the tracing one moved. `decimalai.providers.instrument()` already
  used the new name; this makes the other nine agree.

  ```python
  from decimalai.langchain import instrument   # was: install
  instrument()
  ```

  `install()` still works and still does exactly what it did, with a
  `DeprecationWarning`. `decimalai.init(langchain=True)` is unaffected and does
  not trip the warning. `SkillRouter.install()` keeps its name.

### Added

- **`router.export()`** writes a skill's files to disk without forking it —
  Export and Fork are separate questions now, so you can export something you
  only linked. See the vocabulary guide.

## 0.10.0

### Changed

- **A plan quota 429 is now terminal — `DecimalQuotaExceededError`.** A quota 429
  and a rate-limit 429 share a status code and nothing else: a rate limit clears
  in seconds, a quota does not clear until the billing period rolls over. The
  retry loop treated both as transient, so an over-quota call burned three
  attempts with backoff and then raised `DecimalRateLimitError` — a misleading
  error for a condition retrying cannot fix, after which the payload was gone.
  Measured against high-volume ingest clients: an over-quota call silently
  discarded its payload after three retries.

  The server now marks the difference with an `X-Quota-Exceeded` header naming
  the dimension. On that header the client raises immediately — no sleep, no
  retry — carrying `.dimension`, `.plan` and `.resets_in_seconds`:

  ```python
  from decimalai._client import DecimalQuotaExceededError

  try:
      client.ingest_trace(trace)
  except DecimalQuotaExceededError as e:
      print(f"{e.dimension} quota exhausted on {e.plan}; resets in {e.resets_in_seconds}s")
  ```

  **This is the breaking part:** `DecimalQuotaExceededError` is deliberately NOT
  a subclass of `DecimalRateLimitError`. Code that catches a rate limit in order
  to sleep and retry must not silently swallow a quota and re-create the
  drop-the-payload behaviour. If you catch `DecimalRateLimitError` around an
  ingest call and want the old catch-all, add `DecimalQuotaExceededError` to it.

  Older SDK versions are unaffected: the server deliberately sends no
  `Retry-After` on a quota 429, because the honest value is up to ~31 days and
  this client's `max(retry_after, …)` feeds an uncapped `time.sleep`.

### Deprecated

- **`project=` no longer reaches the wire.** `decimalai.init(project=...)`,
  `DecimalAIClient(project=...)` and the CLI's `--project` flag used to send an
  `X-Decimal-Project` header. The platform reads no such header — a trace's
  `project_id` is set only for a project-scoped API key — so every value was
  discarded on arrival and `project=` never grouped anything. Passing it now
  emits a `DeprecationWarning` and sends nothing.

  The kwarg is still accepted, so existing code keeps running. It will be
  removed in a future release. To group traces, use workspaces (resolved from
  your API key).

## [0.9.1] — 2026-07-31

### Fixed

- **Registry lookups now match the skill NAME exactly, instead of taking the top
  semantic-search hit.** `q=` ranks the whole corpus and essentially always returns
  something, so `skills pull`, `SkillRouter.fork`, `.use` and `.preview` all resolved a
  typo, a rename, or a retired skill to an *unrelated* skill — and reported success:

  ```
  $ decimalai skills pull pdf-procesing      # one-character typo
    ✓ Pulled minimax-pdf v1                  # not what was asked for
  ```

  A `SKILL.md` is instructions the agent loads and follows, so silently substituting a
  different one is a correctness bug and a supply-chain one. All four call sites now
  resolve through `_registry_resolve.find_exact` and raise `ValueError` when there is no
  exact match, listing the closest names as hints rather than guessing.

## [0.9.0] — 2026-07-23

- **Dependency split**: framework/provider integrations moved out of the
  core install into extras — `[langchain]`, `[openai]`, `[openai-agents]`,
  `[llamaindex]`, `[claude-agent-sdk]`, `[pydantic-ai]`, `[adk]`, `[langgraph]`,
  `[evals]`, `[all]`. Core deps are now just pydantic, httpx, click,
  opentelemetry, skillevaluation. Every framework import is guarded with an
  ImportError naming the extra to install.
- **`skills benchmark --trials` removed → `--runs N`** (skillevaluation ADR-0007).
  `--trials` worked by stamping per-case `trials:` into the uploaded eval.yaml for
  pass^k (a case passed only if all k rollouts passed) — an author-chosen, per-case
  knob that could inflate the delta. It is replaced by `--runs N`, a **run-level**
  query parameter on the hosted run endpoint: the whole suite runs N times, uniformly,
  and the per-case results are averaged by **mean** (the headline's expected value is
  independent of N). `--trials` is not silently ignored — passing it exits with a clear
  error echoing the equivalent `--runs` value. `--runs` no longer touches your eval.yaml.
- **`register_manifest(behavioral_policy=…)` re-documented** as a generic versioned
  policy-document surface (docstring only; the field is unchanged and still opaque —
  hashed whole or by `policy_hash`). It no longer describes itself in terms of the
  removed conversation-mode `policy_check`.
- **Progressive disclosure: native `load_skill` tool** (topk design phase 1). On
  `openai_agents` and `pydantic_ai` (they own their tool loop), enabling the skill
  loader now also registers a `load_skill(name)` tool so the agent can pull a
  surfaced skill's full body on demand — descriptions stay the cheap always-on
  tier; bodies arrive mid-turn as tool results. `anthropic`/`langchain` accept
  `enable_load_skill_tool` but stay prompt-injection (no tool loop to route the
  result); their `inject_skill_body` path now trims and budgets bodies.
- **Body guardrail**: new `SkillRouter` knobs `max_loaded_bodies` (3),
  `body_token_budget` (6000), `per_body_char_limit` (8192),
  `body_load_deadline_s` (20s) cap what body loads can add to one turn's
  context; budgets reset per fresh prompt fragment. `get_skill_body` gained
  `max_chars`/`agent_name` (server-side trim + exact offered-version resolution).
- New public `SkillRouter.load_skill(name)`, `load_skill_tool_spec()`, and
  `estimate_tokens()`; loads are recorded on the trace (`skills_loaded_by_agent`).
- Config: `init(load_skill_tool=...)` / `DECIMALAI_LOAD_SKILL_TOOL=0` kill switch.

## [0.8.0] — 2026-07-05

- `decimalai skills scan [PATH]` — local static SkillSafety scan of SKILL.md files,
  the same deterministic Tier-1 scanner the registry publish gate runs (via
  `skillevaluation.safety`). Free, no network, no API key. `--format text|json|
  github|sarif`, `--fail-on blocked|flagged|never`; exit 1 on findings.
- `decimalai skills review SKILL_NAME` — run a metered SkillSafety **deep review**
  (LLM Tier-2 intent + Tier-3 content) on a synced skill, to catch a rejection
  before publishing.
- Floors `skillevaluation>=0.4.0` (the `safety` module).

## [0.7.1] — 2026-06-26

### Added
- **Router `set_routing_id` plumbing.** The skill router/skills path now exposes
  `set_routing_id` so traces can stamp the router decision id, enabling
  offered→activated attribution (older SDK pins silently dropped `routing_id`,
  leaving the registry's `router_activated_count` at zero). Minor `cli` tweak.

## [0.7.0] — 2026-06-25

### Added
- **agentversion 0.2.0 contract-surface parity.** `extract_from_config()` and
  the public `register_manifest()` / `flush_manifest_for_ci()` now accept
  `behavioral_policy=`, `environment=`, `skills=`, and `workflow=`, and the
  exporter emits the `behavioral_policy` (multi-turn policy — a rule change diffs
  as breaking → replay/drop) and `environment` (region/infra/runtime) surfaces.
  `tool_calling_mode` and `runtime_version` are now lifted to `model_runtime`
  top-level so a tool-calling-mode change is classified breaking. The SDK
  exporter and `agentversion.contract.contract_from_components` were edited
  identically and are pinned byte-for-byte by `test_shared_contract_assembly`.
  *Cross-repo note:* the matching `agentversion` change must ship (a new release)
  and the platform must run it for the canonical jcs-sha256 hash to agree in
  production; the new surfaces are additive-only, so existing manifests keep
  their hash.
- **OTel/CrewAI/AutoGen manifests now capture the system prompt.** The OTel
  exporter harvests the system/developer prompt from span attributes
  (OpenInference `llm.input_messages.*` and GenAI `gen_ai.system_instructions`/
  `gen_ai.prompt`) into the manifest, so prompt drift is versioned for those
  frameworks. `decimalai.otel.install(prompts={...})` overrides the harvested
  (rendered) prompt with a static template to avoid per-run hash churn.
- **Repair surface (`decimalai repair` + SDK methods).** The killer workflow
  (detect → impact → **repair** → export) can now run headless. New
  `DecimalAIClient.repair_preview()` / `repair_apply()` / `get_repair_batch()`,
  top-level `decimalai.repair_preview()` / `decimalai.repair_apply()`, and a
  `decimalai repair preview|apply` CLI group over the platform `/repair`
  endpoints. `repair_apply(..., approved_rule_indices=[...])` applies a subset of
  the preview's rules. `compat-check` now points at the real `repair preview`
  command.

### Fixed
- **Programmatic / auto skill-sync no longer blind-clobbers newer remote
  edits.** `sync_to_platform()` on the background/auto path now defaults to
  `newer_wins` (was `local_wins`) and attaches a git-aware `local_updated_at` —
  file mtime, but only when it reflects a real edit. A fresh checkout resets every
  file's mtime to "now", which would spoof local as always-newer, so without a
  trustworthy timestamp the backend's `newer_wins` falls back to local-wins and a
  clone never overwrites the dashboard.
- **`init(crewai=True)` / `init(autogen=True)` / `init(otel=True)` now register
  a manifest (manifest versioning was silently off for them).** These flags wired
  a manifest-blind exporter that also fragmented one agent run into many traces
  (per-span `SimpleSpanProcessor`); they now use the manifest-capable
  `decimalai.otel` exporter (root-span buffering + manifest registration), as
  `autogen.install()` does. The LlamaIndex span handler also now registers a
  manifest from the run's model config.
- **`SkillRouter.status()` no longer reports every synced skill as
  `modified_locally`.** It read `content_hash` from a top-level key the backend
  never emits (it's nested under `latest_version`) and compared a 12-char disk
  prefix against the full 64-char hash with `==`. Now reads the nested hash and
  compares by prefix, matching `pull_missing()`.
- **`ingest_raw_trace` / `ingest_raw_traces_batch` now scrub lone UTF-16
  surrogates** before sending, like the model-based ingest paths. Raw payloads
  come from non-Python sources where un-encodable text is most likely, and an
  unscrubbed surrogate raised `UnicodeEncodeError` client-side before the
  request was built.
- **`BackgroundSender` guards its pending-futures list with a lock.** Concurrent
  `submit()` calls from multiple caller threads could lose futures via the
  read-modify-write on the pending list, so `flush()` could exit before all work
  was awaited (silent trace drop on a fast exit).
- **CLI `traces import` reports the real counts.** It read `imported`/`failed`
  from the response; the backend returns `imported_count`/`error_count`, so a
  successful import always printed "Imported 0 traces (0 failed)".
- **`compat-check` no longer prints a non-existent `decimalai repair` command**
  when it finds repairable traces; it points to the impact report instead.
- **SKILL.md frontmatter parsing handles `---` inside a value.** The closing
  fence is now matched as a line equal to `---` (YAML delimiter semantics)
  rather than a raw substring search, which previously truncated frontmatter
  whose description contained `---`.
- **`resolve_version_id` detects UUIDs strictly.** A human-friendly version
  label longer than 8 chars containing a hyphen (e.g. `release-1`) was passed to
  the backend as a version ID; it now falls through to the `vN`/number lookup
  and raises a clear error instead of an opaque 404.
- **Missing-dependency error hints point at a real install.** `install()` for
  LangChain / OpenAI Agents / OTel and `providers.instrument()` suggested
  non-existent pip extras (`decimalai[langchain]`, `decimalai[openai-agents]`,
  `decimalai[otel]`); since those deps are core, the hint is now
  `pip install decimalai`.
- **`set_routing_id()` logs a debug line when it no-ops** (no active trace), so a
  silently-dropped routing id — which would break the offered→activated join — is
  discoverable. Behavior unchanged (still a deliberate no-op, unlike its
  `log_skill_*` siblings, now documented as such).

### Changed
- **`ManifestDiffResponse` type hint + `diff_manifest()` docstring** now describe
  the real `{"diff": <ManifestDiff | None>}` envelope the route returns (with
  `message`/`verdict` variants), instead of the inner diff fields at top level.
  Type-only; no runtime behavior change.
- **`dev` extra floors `agentversion>=0.2.0`** (was `>=0.1.0`). The SDK now uses
  0.2.0-only modules (`agentversion.contract`, `agentversion.a2a`), so a 0.1.0
  resolve fails test collection.

### Removed
- **Deleted empty stub modules** that shipped but were never imported:
  `decimalai/manifest/{__init__,detector,extractor,hasher}.py` (the real manifest
  logic lives in `decimalai/schema/manifest.py`) and the unused per-command CLI
  stubs `decimalai/cli/{manifest_cmd,dataset_cmd,replay_cmd}.py` (commands live in
  `decimalai/cli/main.py`). These presented a hollow, misleading import surface.

### Documentation
- README: fixed the `@decimalai.eval` decorator path (→ `@decimalai.evals.eval`),
  the manual-tracing snippet (undefined `msgs`, raw response in `log_llm_call`),
  the OTel install row (`init(otel=True)`), added Google ADK / Anthropic Claude
  Agent SDK / direct-provider rows, and dropped the meaningless "pre-`demo`
  release" note.
- `register_manifest()` docstring now documents its `guardrails` and
  `context_config` parameters; the flagship SFT notebook uses the real
  `pull_dataset()` instead of a non-existent `export_dataset(format="openai_jsonl")`.

## [0.6.0] — 2026-06-24

### Removed
- Removed the unimplemented experiment()/compare_experiments() API. The
  agent/dataset experiment runner (`experiment()`, `run_experiment()`,
  `compare_experiments()`, the offline `Eval()` helper) and the matching
  client methods (`create_experiment`, `get_experiment`,
  `get_experiment_results`, `submit_experiment_result(s)`,
  `complete_experiment`, `compare_experiments`) backed the now-removed
  `/api/v1/experiments` endpoints, which were retired on 2026-06-24. This
  release stops the published wheel from shipping the corresponding client
  methods.

## [0.5.0] — 2026-06-14

### Added
- **Expanded CLI** (`decimalai/cli/main.py`): broader command surface for the
  SDK's day-to-day flows (+269 lines).

### Changed
- `SkillRouter` is now a **lazy top-level re-export** — importing `decimalai`
  no longer pulls the skills client (and its dependencies) unless you use it,
  keeping the base import lightweight.

### Fixed
- `disk_export` refuses to **silently overwrite** an existing export file
  (covered by `tests/test_disk_export_no_overwrite.py`).

## [0.4.0] — 2026-06-08

### Added
- **One-command demo sandbox.** `decimalai demo regression` seeds a realistic
  v1→v2 agent + trace corpus into your workspace, runs the regression check, and
  prints a deep link straight to the impact report. `decimalai demo skills`
  seeds three skills with varied effectiveness and links to the ranked registry.
  See the value in ~2 minutes, before instrumenting your own agent.
- `decimalai demo reset` removes all seeded demo data (exact `[Demo] ` match, so
  your own skills/agents are untouched).
- `decimalai init` now surfaces the demo commands in its next-steps output.

### Changed
- README leads with the demo sandbox as the fastest path to first value.
