# Changelog

All notable changes to `decimalai` are documented here. This project follows
[Semantic Versioning](https://semver.org/); pre-1.0, minor releases add features
and patch releases are fixes.

## [Unreleased]

### Fixed

- **The LangChain adapter turned one refused manifest registration into a
  process-long outage for that agent.** `_register_snapshot` made a single
  `POST /api/v1/manifests` attempt and, on any failure, cached the snapshot's
  LOCAL id together with its hash — the exact pair its own dedupe check reads
  as "already registered". Every later trace from that agent then cited an id
  the platform had never stored, and ingest answered
  `manifest_id '…' does not exist` until the process restarted. Measured on
  production 2026-09-03, with the backend refusing requests at admission: 9% of
  all trace POSTs were that 400.

  The adapter now retries a refused registration on the same short ladder the
  generic adapter uses (three attempts, 0.6s in total, none for a 401/403), and
  a registration that still fails is recorded as UNREGISTERED: the trace about
  to ship still carries the local id and `routing_status()`'s sibling
  `export_status().last_manifest_error` names the cause, but the next trace
  from that agent re-attempts the registration instead of inheriting the
  failure. An empty run no longer adopts a failed registration's local id as
  the agent's manifest either.

## [0.13.0] — 2026-09-03

### Fixed

- **The hot-path routing budget was below what the platform can serve, so agents
  silently ran without skills.** 0.12.0 capped `/skills/route` and `/skills/menu`
  at a 2-second read. Measured against production: healthy, that endpoint serves
  p95 1.39s — so 2s left 1.4x of headroom, inside ordinary variance — and
  degraded it serves 8-30s. The DecimalAI fleet, 93 synthetic users driving this
  SDK, went from delivering a skill body in 69.4% of sessions to 3.1% in the hour
  it restarted onto 0.12.0, with `no_skill_offered` going 7.9% to 77.6%.

  The failure is silent by design and that is the sharp edge: a hot-path timeout
  does not raise. `smart_route` returns an empty menu, the agent answers without
  skills, and the request succeeds. No exception, no 5xx — answers just quietly
  get worse. That state ran for 21 hours across 93 agents with every other health
  signal green.

  The budget is now **5 seconds** (3.6x the healthy p95, still a small share of an
  agent turn), overridable with `DECIMALAI_SKILL_ROUTE_TIMEOUT_S`, clamped to
  [0.5, 30], and read per call rather than frozen at import.

- **The circuit breaker turned a slow minute into a silent one.** Its cooldown was
  a flat 30s, so three slow calls bought a guaranteed half-minute of skill-less
  answers on a platform whose latency is bursty. It now starts at 5s and doubles
  per consecutive open to a 30s ceiling; one success resets the ladder. Opening
  logs at ERROR, not WARNING — while it is open, every agent in the process is
  answering without its skills.

- **`ManifestTracker` re-registered manifests this process had already
  registered.** It kept a single hash slot, so a snapshot that oscillates
  (A -> B -> A) sent three requests for two manifests. It now remembers every hash
  it has registered, in a bounded LRU. A genuinely new hash still registers —
  an agent that discovers a tool mid-run has a different manifest and the platform
  must be told.

### Added

- **`decimalai.routing_status()`** — the companion to `export_status()`, and the
  answer to a question a 200 cannot give you: are my agents actually getting their
  skills right now?

  ```python
  st = decimalai.routing_status()
  if not st.healthy:
      alert_oncall(f"agents running without skills: {st.last_error!r}")
  ```

  Reports `healthy`, `breaker_open`, `consecutive_failures`, `timeouts`, `opens`,
  `suppressed`, `read_budget_s`, `last_error`, `last_error_at`,
  `last_success_at`. Counters are cumulative for the process — a health check
  wants "has this ever degraded", not a gauge a recovery quietly resets.

## [0.12.0] — 2026-08-30

### Fixed

- **`decimalai init` produced an agent that could not read any of its skills.**
  On `langchain` — the framework you get by default — `inject_skill_body`
  defaulted to `False` and the adapter registers no `load_skill` tool, so both
  body channels were off. The model received a menu of skill *titles* with no
  mechanism to read them: asked a question only a skill body could answer, it
  invented a confident 15% where the body said 23.5%. `inject_skill_body` is now
  tri-state and resolved per adapter — no tool loop means inject, because it is
  the only channel; a tool loop means don't double-deliver; an explicit setting
  always wins. `openai_agents` infers this from the registration *outcome*
  rather than the config flag, because a flag saying "register the tool" plus a
  registration that failed adds up to zero channels again.
- **A single 5xx could destroy 50 buffered traces.** `_request_with_retry`
  returned or raised immediately for every non-429, so 500/502/503/504 got zero
  retries, and `flush()`'s bare `except Exception` cleared a buffer that
  auto-flushes at 50. `502/503/504` are now retried on the existing ladder,
  `500` is opt-in per call site (passed by the four trace-ingest POSTs and
  nothing else), and the buffer survives 5xx and `httpx.RequestError` — it is
  cleared only on 4xx and serialization failures.
- **The generated `langchain` file was not an agent.** It emitted
  `agent = init_chat_model(MODEL)` — a chat completion named `agent`, with no
  tool loop — under a docstring inviting you to add tools. Binding a tool to a
  bare chat model returns `tool_calls` and empty `.content`, so following that
  advice produced an empty string. It now emits `create_agent(...)` with a real
  loop. It uses `langchain.agents.create_agent`, not
  `langgraph.prebuilt.create_react_agent`, which raises
  `LangGraphDeprecatedSinceV10` on every call and is removed in langgraph 2.0.
- **The generated `openai-agents` file died with `MaxTurnsExceeded`** on a
  realistic ticket. It now passes an explicit `max_turns` and names the cause.
- **Google ADK can deliver a skill body.** It was listed as having no prompt
  seam and refused a scaffold on that basis; the entry was wrong.

### Added

- **`decimalai init --framework pydantic-ai`** writes a runnable agent — the
  third scaffoldable framework. Pydantic AI owns a real tool loop, so it sits at
  the top delivery tier: a live run of the generated file answered with a figure
  that exists only in the skill body the model pulled through `load_skill`.

### Internal

- Version floors are enforced rather than assumed: an unfloored dev dependency
  resolved to a 2022 release whose top-level `tests/` package shadowed this
  repo's own.
- The post-publish smoke no longer asserts a defect it never observed — a
  failure to *fetch* the artifact is now reported as inconclusive rather than as
  a broken release.
- The conformance and notebook advisory lanes close their own issues when they
  go green, instead of only ever filing new ones.

## [0.11.1] — 2026-08-25

### Added

- **`decimalai.load_agent(name)` reads the system prompt you configured in the
  dashboard**, so changing it there reaches production on your agent's next run
  with no redeploy.

  ```python
  config = decimalai.load_agent("refund-bot")
  agent  = Agent(name="refund-bot", instructions=config.system_prompt)
  ```

  It is **explicit** — the prompt is never injected for you. That asymmetry with
  skills is deliberate: a skill menu is additive, so at worst it wastes a few
  hundred tokens, while a system prompt is your agent's core instruction. If we
  swapped it silently, your repo could say *"Never issue refunds over $500"*
  while the model received *"You are a helpful assistant"*, and you would have no
  way to see it.

  `config.system_prompt` is `None` only when the agent genuinely has no prompt
  set. Every failure — unknown agent, bad key, network error, timeout, an older
  backend without the route, a pin that no longer resolves — **raises** instead.
  A prompt that silently came back empty would leave your agent running with no
  instructions at all, which is worse than failing at startup.

- **`decimalai init <agent-name>` now wires the prompt into the file it writes.**
  Previously the generated LangChain agent sent no system message at all and the
  openai-agents one hardcoded *"You are &lt;name&gt;. Use the skills you are given."*,
  quietly discarding what you had typed in the dashboard. Both now use your
  actual prompt, and an agent with none still produces a file that runs.

- **A one-time warning when the agent name you passed was renamed.** Its prompt
  and skills follow the rename, but trace ingest does not — so traces keep
  landing under the old name until you update `instrument(agent_name=...)`.

## [0.11.0] — 2026-08-24

### Added

- **Skills now arrive as a cacheable prefix plus a one-line hint, instead of one
  block that changed on every request.** The routed menu is the same information
  it always was, but it is now split: a *prefix* listing every skill available to
  the agent, byte-identical from turn to turn, and a *tail* naming the one or two
  that look relevant to this particular request.

  Why it matters: the old single block was rebuilt per query, so on providers
  that merge system content it invalidated the prompt cache for everything behind
  it. A caller's 2,000-token system prompt that should have been a cache hit
  became a full miss on every request. The cost was never our own tokens — it was
  theirs. The stable half can now sit inside the cached region.

  This is automatic on LangChain; no code change. `SkillRouter.build_prompt_parts()`
  exposes it directly for anyone assembling prompts by hand. Against an older
  platform it degrades to exactly the previous behaviour.

  One visible change on LangChain: skills are injected as **two** system messages
  rather than one (stable first, hint second), both placed immediately after your
  own system message. They are kept adjacent to your own deliberately —
  `langchain_anthropic` raises `Received multiple non-consecutive system messages`
  for a system message positioned after any human or AI turn, so the intuitive
  "put the hint next to the question" placement breaks every ChatAnthropic caller.

- **`decimalai init <agent-name>` writes a runnable `agent.py`.** Creating an
  agent in the dashboard used to store a name, a description and a set of
  skills, then hand back a snippet to paste into an agent you had to have
  written already — so the product stored configuration and the user still
  built the agent. `init` now turns that configuration into a file that runs:
  the agent name bound, its assigned skills named in a comment, a model on one
  editable line, and a single example call. We generate; they run — nothing
  executes on DecimalAI's side, and the file is the user's from the moment it
  lands.

  - `--framework langchain` (default) and `--framework openai-agents`.
    Frameworks whose adapter has no prompt seam — llamaindex,
    claude_agent_sdk, crewai/autogen/otel, adk — are **refused, with the
    reason**, rather than scaffolded: a generated file for those would trace
    perfectly and deliver none of the agent's skills, silently, which is worse
    than no scaffold at all.
  - Every generated file passes `enable_skill_loader=True`. It defaults to
    False on both adapters, so a scaffold that omitted it would hand the model
    a list of skill titles it cannot read — the one thing this command has to
    get right.
  - The agent is **resolved against the API**, never invented. An unknown name
    exits with the workspace's agents (or a "did you mean" on a near miss like
    `refund_bot` vs `refund-bot`) and a link to `/agents/new`.
  - `--dry-run` prints the file instead of writing it, byte for byte.
    `--out` chooses the path; an existing file is never overwritten without
    `--force`.
  - No API key is ever written to the file: it reads `DECIMAL_API_KEY` from
    the environment. A non-default `--base-url` **is** baked in, so a file
    scaffolded against a local or self-hosted backend does not silently point
    at production.
  - `decimalai init` with no argument is unchanged — it still verifies the key
    and sends a test trace.

  The templates are checked as code, not as strings: the suite `compile()`s
  every generated file and asserts against its AST, and a subprocess smoke test
  actually runs them with only the LLM call stubbed, so a keyword the adapter
  no longer accepts fails the build instead of the user's first run.

## 0.10.4 — 2026-08-24

### Fixed

- **Skill messages the SDK injects are no longer mistaken for your system
  prompt.** Auto-detection keeps the first system message it sees and warns when
  a later one differs, on the theory that a prompt changing mid-trace is a
  dynamic template you should pin. That is right for your prompts and wrong for
  ours: the skill menu and the routing hint are two system messages by design,
  so every call emitted *"Auto-detected system prompt changed within trace"* and
  advised `install(prompts=...)` — a fix that cannot work, because the message
  that "changed" was one we added. A bare `invoke` went from 0 warnings to 1 and
  a three-turn conversation from 3 to 6.

  The same change fixes something older and worse: an agent whose code passes no
  system message of its own had the routed skill menu recorded **as** its system
  prompt — a prompt the user never wrote. Now nothing is recorded, which is the
  truth. Injected messages carry a marker in `additional_kwargs`; it reaches no
  model and costs you nothing.

- **`pydantic>=2.3`, corrected from `>=2.0`.** Before 2.3 two of our model
  fields raise `NameError` at class-definition time rather than warning, so the
  old floor admitted four releases (2.0, 2.0.3, 2.1, 2.2) on which
  `import decimalai` could not succeed at all — pip resolved happily and the
  first import died. Bisected; 2.3 is the true floor.

- **`openinference-instrumentation-anthropic` is capped below 2.0.0.** Its 2.0
  major requires `anthropic>=1.0.0` and, being uncapped, dragged `anthropic` past
  the `<1.0.0` cap three lines above it — silently reinstating the failure that
  cap exists to prevent, where the Anthropic instrumentor stops capturing the
  model turn entirely. Affects the test extras only; `anthropic` is in no runtime
  extra, so nothing a user resolves was constrained either way.

- **Registry skills attached to a single agent are now actually offered on
  `langchain` and `anthropic`.** An agent's offered set is built from org-owned
  skills scoped `workspace` plus `SkillSubscription` rows, and an agent-scope row
  matches on `agent_name`. A skill pulled from the registry belongs to another
  org, so it is never in the ambient set and reaches one agent only through that
  agent-scope row — which the resolver can match only when the router is told
  which agent it serves. Both auto-inject adapters injected without sending the
  name, so **every registry skill attached to a single agent was silently never
  offered**: no error, no empty-menu warning, just a prompt missing the skills
  the user picked. `openai_agents` and `pydantic_ai` already sent it. LangChain
  now reads the installed name at call time rather than baking it into the router
  singleton, so a later `instrument(agent_name=...)` is not ignored by a router
  built before it; the anthropic adapter's `instrument()` gains an optional
  `agent_name` keyword, and a bare repeat call deliberately does not clear a name
  already set.

- **Skills are delivered on `.stream()` and `.astream()`.** Injection ran on
  `invoke`/`ainvoke` only, so a production chat agent — which streams — received
  no skills at all, and nothing said so. Both are now patched as generator
  functions rather than plain wrappers, so the rails stay open for the whole
  consumption of the stream; a plain wrapper closes them when it hands the
  generator back, before the model call runs. `generate`/`agenerate` are
  deliberately left alone: `invoke` calls `generate` internally, so patching it
  would double-inject, and any model implementing `_stream` bypasses it anyway.
  One further delegation is guarded — when a model cannot really stream,
  LangChain's own `stream` yields `self.invoke(...)`, running the stream path
  through the invoke patch.

- **A near-miss agent name warns instead of failing silently.** A name that does
  not match an installed agent previously produced an empty menu and no signal.

- **LangChain files skill rails per run, so one run's `load_skill` is not
  another's.** Two concurrent runs shared the router singleton: run A's tool call
  loaded a skill, run B loaded nothing, B shipped first — and B reported an
  activation it never made while A's real one was lost. Both halves fired at
  once. The router now takes an ambient scope resolver: adapters register a
  callable answering which run is executing, LangChain's reads its own runnable
  config, and the reader accepts only run ids it already indexed as its own
  callbacks'. The unscoped drain could not simply be deleted — LangChain
  dispatches callbacks under `copy_context()`, so ContextVars written inside the
  runnable are invisible at trace-send, and it registers no native `load_skill`
  tool, so the call arrives from user code with no scope to pass.

- **"View full report" links no longer break on an agent name containing spaces
  or brackets.** Two sites interpolated the name raw, so the default demo agent —
  literally named `[Demo] support-agent` — produced a URL nothing can follow. Two
  other sites in the same file already quoted correctly; all four now share one
  pattern.

- **The routed skill menu no longer sits in front of your system prompt, so the
  provider prompt cache survives.** The menu is built per query, so it differs
  on every request. It was injected at position zero of the system block, which
  meant a caller's own system prompt — stable, and the thing a provider would
  otherwise cache — sat behind ~115 tokens of varying text and missed the cache
  on every call. The cost was never the menu's own tokens; it was everything
  behind it. The menu now lands after the caller's leading system instructions
  on `langchain`, after the caller's `system` on `anthropic` (content blocks and
  their `cache_control` hints are copied through untouched, so a positional
  breakpoint still marks the same prefix), and after the agent's `instructions`
  on `openai_agents`. `pydantic_ai` already behaved this way; only its docstring
  claimed otherwise. **The model now reads your instructions before the skill
  menu rather than after** — a visible change, and the one to notice if a prompt
  depended on the old order. Contract pinned in cache terms rather than index
  terms: given one caller prompt and two different queries, the bytes before the
  menu must be byte-identical.

- **`developer` counts as a system role.** OpenAI's current name for the system
  role converts to a `SystemMessage`, but the reordering above matched only
  `"system"` — so for anyone on the modern spelling the fix silently did nothing
  and the varying menu went back to byte zero.

- **Cache tokens are reported instead of being folded away.** The Claude Agent
  SDK adapter added `cache_read_input_tokens` and `cache_creation_input_tokens`
  into `input_tokens` and returned one number, so the split was destroyed at the
  source and no consumer could see cache behaviour. They are now carried as
  their own fields. `input_tokens` on that adapter is therefore SMALLER than
  before for any cached run — it is now what the provider reported. Run totals
  are no longer stamped onto the last call when per-turn frames already recorded
  cache counts, which double-counted the cache in any sum across calls.

- **Per-run skill attribution now has a floor that is true on Python 3.10.** The
  `[langchain]` extra floored `langchain-core` at 1.3.3 for every interpreter,
  but per-run attribution does not hold there on 3.10. `langchain-core` carries
  its config `ContextVar` into an async runnable body through
  `runnables/utils.py::coro_with_context`, which before 1.4.8 could only do so
  via `asyncio.create_task(..., context=)` — an argument that does not exist
  below Python 3.11. On 3.10 it fell back to a plain `create_task` and dropped
  the context, so `decimalai.langchain._ambient_run_scope()` read no
  `parent_run_id` and the Router filed every `load_skill()` unscoped. Two
  concurrent runs then cross-contaminated: measured on 1.3.3 / 3.10, one run's
  activation was lost and another reported one it never made. 1.4.8 replaced
  that branch with `context.run(asyncio.create_task, coro)`, which works on
  every interpreter. Bisected: 1.3.0–1.4.7 fail on 3.10, 1.4.8+ pass, 3.11+
  passes throughout. The floor is now stated per interpreter — `>=1.3.3` on
  3.11+, `>=1.4.8` on 3.10 — so the declared minimum is true wherever it is
  declared. Nothing changes for 3.11+ installs.

- **The `[dev]` extra no longer resolves a CVE-affected `langchain-core`.** It
  floored at 1.3.0 and described that as "CVE-clean"; CVE-2026-44843 (High,
  unsafe deserialization) affects 1.0.0–1.3.2 and is fixed in 1.3.3. The CI
  floors job had therefore been installing a vulnerable version, and `pip-audit`
  did not catch it because the audit job installs `.`, not `.[dev]`. The dev
  extra now mirrors `[langchain]` exactly, marker included.

### Changed

- **The package summary now matches the repo description.** PyPI said
  "manifest-aware agent change management" while GitHub said the lift sentence —
  two different products to a skimming reader, on the two surfaces most likely to
  be skimmed. Both now carry the lift sentence. This release is the first to
  carry it to PyPI.

- **The floors job runs on 3.10 and 3.13.** A floor can be true on one
  interpreter and false on another, and only the 3.10 leg could see the
  `langchain-core` case above. The 3.13 leg covers the mirror case: every other
  job resolves latest dependencies rather than lowest, so a floor that breaks
  only on a newer Python had nothing watching it.

## 0.10.3 — 2026-08-17

Published from CI through Trusted Publishing rather than a maintainer machine —
the first release of this package to carry a PyPI attestation.

### Added

- **A framework conformance suite that asserts the wire.** Each driver runs in
  its own process, with the real baseline recorded rather than assumed; 18 cells
  closed across span parenting, agent identity and content. A separate floating
  lane runs the same suite against latest framework releases on a schedule, so an
  upstream break surfaces without gating the merge queue.
- **A written support policy for framework versions**, and why each floor is
  where it is.

### Fixed

- **A raw-provider run produces one trace instead of one trace per call**, with
  the run boundary declared on the raw Anthropic rail.
- **Instrumenting an agent no longer writes to the user's repo.**
- **`llamaindex` no longer consumes the stream it is observing.**
- **`otel` no longer drops traces that declare nothing**, and flushes at exit.
- **`openai_agents` routes on the real query** and scopes routing per run.
- **`langchain` keys trace state per run**, and traces worker threads.
- Two floors that were fiction were corrected, and the `openai` major cap dropped.

### Removed

- **AutoGen/AG2 is no longer claimed.** The flag stays; the claim and the advice
  that installed the wrong package are gone.

## 0.10.2 — 2026-08-15

### Fixed

- **`init()` now accepts `DECIMALAI_API_KEY` as well as `DECIMAL_API_KEY`.**
  The CLI has taken both spellings since it shipped
  (`envvar=["DECIMAL_API_KEY", "DECIMALAI_API_KEY"]`), and the copyable
  install snippet on decimal.ai exports the alias — but the library read only
  `DECIMAL_API_KEY`, so following that snippet and calling `decimalai.init()`
  raised `DecimalConfigError: No API key provided` with a usable key sitting in
  the environment. Both spellings resolve now, in the same order the CLI uses:
  an explicit `api_key=` argument wins, then `DECIMAL_API_KEY`, then
  `DECIMALAI_API_KEY`. An exported-but-blank primary falls through to the alias
  instead of shadowing it. The same resolution applies to the auto-init path,
  so `DECIMAL_AUTO_TRACE=<framework>` and the bare auto-init both work with
  either name.

  `DECIMAL_API_KEY` remains the documented primary and is still the variable
  the error message names — nothing that already worked changes.

  ```bash
  export DECIMALAI_API_KEY="dai_sk_..."   # was: "No API key provided"
  python -c "import decimalai; decimalai.init()"
  ```

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
