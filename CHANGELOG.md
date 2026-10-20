# Changelog

All notable changes to `decimalai` are documented here. This project follows
[Semantic Versioning](https://semver.org/); pre-1.0, minor releases add features
and patch releases are fixes.

## [Unreleased]

### Changed (breaking): `load_agent()` is now `agent_config()`, and each message is a `Run`

The function never loaded an agent: you build your own, and it returns DecimalAI's half of
it. The names below are the ones the Anthropic and LangChain releases will use too.

- **`load_agent()` is `agent_config()`** in `decimalai`, `decimalai.openai_agents`,
  `decimalai.pydantic_ai` and `decimalai.adk`, with the same arguments and the same result.
  `load_agent` still imports; calling it raises `TypeError` naming `agent_config()`.
- **`decimalai.adk.DecimalaiPlugin` is `DecimalAIPlugin`**; the old name raises `TypeError`
  naming the new one.
- **The neutral config starts a run per user message.** `support.run(message, variables=...)`
  returns a `Run` (`decimalai.Run`) with `system`, `tool_spec` and `load_skill(name)`. They
  replace `support.system(message, context=...)`, `support.tool_spec` and
  `support.load_skill()`, which now raise `AttributeError` saying where they went. A run serves
  its own `load_skill` calls from any thread or task; before, a load was served against the
  latest `system()` of the thread or task that called it.
- **`variables=`** is the parameter for the prompt's `{{name}}` values, and a
  `MissingPromptVariable` says so. The framework modules keep their framework's own source:
  `Runner.run(context=...)`, Pydantic AI's `deps`, ADK's session state.
- **`prompt_source`** (and the `decimal.prompt.source` trace attribute) says what happened:
  `"decimalai"`, `"decimalai (cached copy)"`, `"fallback (unreachable)"` and
  `"fallback (none saved)"`, which were `"decimal"`, `"decimal (cached copy)"`, `"fallback"`
  and `"code"`.
- `decimalai init --framework openai-agents`, `pydantic-ai` and `adk` write the new names.

Migration:

```python
# before
from decimalai.openai_agents import load_agent
from decimalai.adk import DecimalaiPlugin, load_agent
support = decimalai.load_agent("support", fallback=BASE)
system = support.system(question, context=customer)
tools = [*my_tools, support.tool_spec]
text = support.load_skill(name)

# after
from decimalai.openai_agents import agent_config
from decimalai.adk import DecimalAIPlugin, agent_config
support = decimalai.agent_config("support", fallback=BASE)
run = support.run(question, variables=customer)
system, tools = run.system, [*my_tools, run.tool_spec]
text = run.load_skill(name)
```

### Fixed

- **Claude Agent SDK: one LLM call per model request.** The Claude Code CLI streams one API
  request as several `AssistantMessage`s, one per content block (thinking, text, each tool use),
  all carrying the request's `message_id` and the same `usage`. Each was recorded as its own LLM
  call, so a two-request tool-using run showed five calls, input and cache tokens repeated per
  message (46 input recorded against 18 used in the measured run), and output tokens
  under-reported (17 against 194: the stream reports a request's output count only as it starts). Messages sharing an id are now one
  call: input and cache counts are taken once, exactly as the request reported them; output is
  the run total from the `ResultMessage`, shared across the requests by how much each one
  generated, so the calls add up to the run. The platform estimates cost from these token
  counts, so Claude Agent SDK costs were understated too. When this ships, drop the "Known
  issue in SDK 0.20.0 and earlier" caveat from the docs' Claude Agent SDK page.

## [0.20.0] — 2026-10-20

### Changed (breaking): Google ADK agents get their prompt and skills from `load_agent()`

- **New:** `decimalai.adk.load_agent(name, fallback=..., on_unavailable="continue")` returns
  `instruction`, an ADK instruction provider for `LlmAgent(instruction=...)`: the agent's saved
  prompt (`{{placeholders}}` filled from the session's state), then the skill list routed on the
  invocation's message. Its `tools` is `load_skill`, limited to the agent's skills: ADK agents
  can fetch a skill's text mid-turn for the first time. Each invocation is routed once, on its
  own message, the way OpenAI Agents runs are. Built only on ADK's public instruction and tool
  APIs.
- `DecimalaiPlugin()` goes in `App(plugins=[...])`, ADK's own place for plugins. An invocation
  whose agent has `load_agent()`'s instruction is one trace filed under that DecimalAI agent,
  with its prompt version, placeholder names and the skills offered, loaded and delivered.
  `load_skill`'s results are in the trace's model inputs.
- **Removed:** `decimalai.adk.instrument()`, `init(adk=True)` and the `Runner.__init__` patch
  behind both, which added the plugin to every runner in the process; the plugin's skill
  injection (a menu and a body pasted into `system_instruction`); and its `enable_skill_loader`
  and `enable_load_skill_tool` parameters. `DECIMAL_AUTO_TRACE=adk` now logs a warning that
  says to add the plugin, and traces nothing.
- An agent that carries ADK's own `SkillToolset`, or another tool named `load_skill`, gets no
  DecimalAI skills, with one warning.
- `decimalai init --framework adk` writes the new shape.

Migration:

```python
# before
decimalai.init(adk=True, agent_name="support")
agent = LlmAgent(name="support", model=MODEL, instruction=BASE, tools=[lookup_order])
runner = InMemoryRunner(agent=agent)

# after
from decimalai.adk import DecimalaiPlugin, load_agent
support = load_agent("support", fallback=BASE)
agent = LlmAgent(name="support", model=MODEL, instruction=support.instruction,
                 tools=[lookup_order, *support.tools])
runner = InMemoryRunner(app=App(name="support", root_agent=agent, plugins=[DecimalaiPlugin()]))
```

### Fixed: OpenAI Agents traces record OpenAI's cached input tokens

- Each model call's `cache_read_tokens` now carries the cached-input count OpenAI reports
  (`usage.input_tokens_details.cached_tokens`), so the trace's `total_cache_read_tokens` is no
  longer always null. It is a subset of `input_tokens`, as OpenAI counts it.
- On the Responses API (the default model) a reported 0 is recorded as 0, a measured cache miss.
  On chat-completions models, LiteLLM, and streamed responses only the span's usage is available,
  and `openai-agents` writes 0 there when the provider reported nothing; those record a positive
  count only, and leave 0 as not reported.

### Changed: a failed skill route serves the agent's menu instead of no skills

- When `POST /api/v1/skills/route` times out or fails with a 500 or 504, the SDK asks once for the
  agent's own menu (`GET /api/v1/skills/menu`) and the turn gets the agent's skills, unranked,
  instead of none. Before, the agent answered without skills and only the trace showed it. This
  covers every adapter that routes through `SkillRouter.build_prompt_fragment`, including
  `load_agent()`.
- The extra request has its own 2 s read budget and does not wait out a `Retry-After`, so a failed
  route costs a turn at most 2 s more. It is not made after a 429 or 503 (the platform is shedding
  load), a connection failure, or while the circuit breaker is open.
- After a fallback, routes from the same host and API key are skipped for 30 s and turns go
  straight to the menu, so they do not each wait out the route's timeout again.
- `routing_status().menu_fallbacks` counts the turns served this way. They had their skills, so
  they do not make `healthy` False; a rising count says routing itself is failing.

## [0.19.0] — 2026-10-20

### Changed (breaking): Pydantic AI agents get their prompt and skills from `load_agent()`

- **New:** `decimalai.pydantic_ai.load_agent(name, fallback=..., on_unavailable="continue")`
  returns a Pydantic AI capability for `Agent(capabilities=[...])`. Its instructions are the
  agent's saved prompt (`{{placeholders}}` filled from the run's `deps`), then the skill list
  routed on the run's prompt; its toolset is `load_skill`, limited to the agent's skills; and its
  `wrap_run` opens one DecimalAI run span per run, so a run's model calls are one trace filed under
  the DecimalAI agent, whatever the `Agent` or `init(agent_name=...)` is called: under
  `Agent.instrument_all()`, Pydantic AI's own run span is the trace's root, and it is named for the
  DecimalAI agent too. Every run is routed on its own prompt, the way OpenAI Agents runs are.
  Built only on Pydantic AI's capability API.
- **Removed:** `decimalai.pydantic_ai.instrument()` and the three patches behind it
  (`Agent.__init__` twice, `Agent.iter`): the skills instructions function and `load_skill` it
  added to every Agent in the process, the `decimalai_load_skill` rename, the run span it opened
  around every run, and `get_current_routing_id()`. Tracing is DecimalAI's OpenTelemetry exporter:
  `decimalai.init(otel=True)` with `Agent.instrument_all()`, or a provider instrumentor
  (`decimalai.init(openai=True)`).
- An agent that also has the Pydantic AI harness's `Skills` gets no DecimalAI skills from the
  capability, with one warning. An agent with another tool named `load_skill` fails its run with
  Pydantic AI's own error.
- The harness's observed `Skills` (`decimalai.pydantic_ai_harness.skills()`) needs no
  `instrument()` either: with `init(otel=True)` and `Agent.instrument_all()`, each run's skill
  catalog and every load go on that run's trace.
- `decimalai init --framework pydantic-ai` writes the new shape.

Migration:

```python
# before
from decimalai.pydantic_ai import instrument
instrument(enable_skill_loader=True)
agent = Agent(model, name="support", system_prompt=BASE, tools=[lookup_order])

# after
from decimalai.pydantic_ai import load_agent
support = load_agent("support", fallback=BASE)
agent = Agent(model, name=support.name, tools=[lookup_order], capabilities=[support])
```

### Changed (breaking): `decimalai.load_agent()` returns the agent's prompt and skills

- `decimalai.load_agent(name, fallback=..., on_unavailable="continue")` returns an `AgentConfig`
  that works like `decimalai.openai_agents.load_agent()`, for code that calls a model itself:
  `system(message, context=...)` is this turn's system text (the saved prompt with its
  `{{placeholders}}` filled from `context`, then the skill list routed on `message`), `tool_spec`
  is the `load_skill` tool as `{name, description, parameters}`, and `load_skill(name)` returns a
  skill's text. `prompt`, `prompt_version`, `prompt_hash`, `prompt_source`, `variables` and
  `skill_names` describe what it holds. Inside an OpenTelemetry run, each turn's record (prompt
  version, placeholder names, skills offered and loaded) goes on that run's trace.
- **Removed:** the prompt-only `AgentConfig` it returned before (`system_prompt`,
  `version_number`, `content_hash`, `is_fallback`, …) and `load_agent()`'s `version=` and
  `cache_ttl_seconds=`. `support.prompt` is what `config.system_prompt` was, except that it is
  the `fallback=` text when no prompt is saved.
- `decimalai init` files and the Support project read `support.prompt`.
- `decimalai.MissingPromptVariable` is importable from the top level.
- The OpenTelemetry exporter keeps DecimalAI's own record attributes on the exported span:
  `decimal.agent` and `decimal.prompt.*` (version, hash, placeholder names, never their values).

Migration:

```python
# before
config = decimalai.load_agent("support")
agent = create_agent(model, tools=tools, system_prompt=config.system_prompt)

# after
support = decimalai.load_agent("support")
agent = create_agent(model, tools=tools, system_prompt=support.prompt)
```

### Fixed: a refusing platform gets fewer retries, spread out

- A retry after a 429 or a 502/503/504 now waits a random time under a ceiling that doubles
  (1 s, 2 s, 4 s, capped at 8 s), on top of any `Retry-After`, instead of exactly 1, 2 and 4 s.
  Processes refused at the same moment no longer come back at the same moment.
- When a manifest registration is refused, the background sender's re-registration makes one
  request per step instead of running the client's own retries inside each step, and waits
  before its first step. One trace whose registration stays refused now sends at most 9
  registrations, down from 24 on most adapters, 32 on the generic tracer and 72 on LangChain.
  The steps still span 31-62 s, long enough to bridge a backend restart.
- The generic tracer and the LangChain adapter no longer repeat a registration that the client
  has already retried (429, 502-504) or that a retry cannot fix (other 4xx, a plan quota).

### Added: assign a skill to an agent, and create it restricted, from `SkillRouter`

- `SkillRouter.assign_skill(agent_name, skill_id, pin_version_id=None)` assigns one of your
  skills, or a public registry skill, to one agent (`POST /api/v1/agents/{name}/skills/{id}`).
  Assigning again updates the pin. `unassign_skill(agent_name, skill_id)` removes it (with a key
  that has admin permission). Before, this took a raw REST call.
- `create_skill(..., offer_scope="restricted")` creates a skill offered only to the agents it
  is assigned to, in one call, so it is never offered org-wide in between. Against an older
  backend, which ignores the field, it follows up with `update_skill(offer_scope=...)`, the
  call you used to make yourself.
- `get_trace()`'s type lists the skill fields the API returns. `skills_delivered` holds names
  only; the delivered versions and where each body sits in the prompts are
  `skills_delivered_versions` and `skills_delivered_sites` (older backends mixed them into
  `skills_delivered`).

## [0.18.0] — 2026-10-20

### Changed (breaking): OpenAI Agents SDK agents get their prompt and skills from `load_agent()`

- **New:** `decimalai.openai_agents.load_agent(name, fallback=..., on_unavailable="continue")`
  returns `instructions` (the agent's saved prompt, with `{{placeholders}}` filled from
  `Runner.run(context=...)`, then its skill list) and `tools` (`load_skill`, limited to the
  agent's skills), which you pass into your own `Agent(...)`. Every run is routed on its own
  latest message (read from `RunContextWrapper.turn_input`; without it, a fixed order and one
  warning): the first run before its model call, later runs from a local copy while their own
  decision is recorded in the background, and an agent whose menu is cut to fit before every
  model call. The prompt is re-read in the background at most once a minute. The prompt
  version, its placeholder names (not values) and any skill outage are on the agent's span,
  and the trace is filed under the `load_agent()` name. An unknown agent name raises.
- **Removed:** every patch on the Agents SDK — `Agent.__init__`, `Agent.get_system_prompt`,
  `Agent.get_all_tools`, `agents.run_internal.turn_resolution.check_for_final_output_from_tools`
  and the sandbox `Skills.load_skill` / `Skills.instructions`. They gave every agent in the
  process skills and relied on SDK internals that may change in any patch release.
- `init(openai_agents=True)` and `instrument()` now trace only. `openai_agents.instrument()` no
  longer takes `enable_skill_loader`, and `init()`'s `skills=` applies to the `langchain`, `adk`
  and `crewai` flags only.
- DecimalAI's skills turn off for one agent, with one warning, when its `tool_use_behavior`
  is `"stop_on_first_tool"` (a skill's text would become the answer), it loads skills with the
  sandbox `Skills` capability, it has another tool named `load_skill`, or its instructions are
  `support.instructions` but its tools leave out `*support.tools`. The prompt still runs.
- The tool's kill switch still works: with `DECIMALAI_LOAD_SKILL_TOOL=0` or
  `init(load_skill_tool=False)`, the text of the skills picked for each run's message goes into
  the instructions instead (`inject_skill_body`, which defaults to on once the tool is off).
- `init(enabled=False)` after `load_agent()` stops routing and skills for that config; the
  prompt it read keeps running.
- `init(openai_agents=True)` and `instrument()` read the project's own `SKILL.md` files (and
  sync the ones you wrote) only when asked: `disk_sync=True`, `skill_dirs=`, or the harness
  skill authority.
- `sandbox_skills()` still records what it lists and loads, in its own `Skills` subclass; a
  plain `Skills(...)` is no longer observed.
- `decimalai init --framework openai-agents` writes the new shape.

Migration:

```python
# before
decimalai.init(openai_agents=True, agent_name="support")
agent = Agent(name="support", instructions=BASE_INSTRUCTIONS, tools=[lookup_order])

# after
from decimalai.openai_agents import load_agent

support = load_agent("support", fallback=BASE_INSTRUCTIONS)
agent = Agent(name=support.name, instructions=support.instructions,
              tools=[lookup_order, *support.tools])
```

### Removed (breaking): what only existed to ease the move from older releases

- **`install()`**, the deprecated alias of `instrument()`, in all nine framework modules
  (`adk`, `anthropic`, `autogen`, `claude_agent_sdk`, `langchain`, `llamaindex`,
  `openai_agents`, `otel`, `pydantic_ai`). Call `instrument()`.
- **The 0.14 disk verbs**, deprecated since 0.15. There is one way a registry skill reaches
  disk: `decimalai skills add` (`router.export(name)` in Python).

  | removed | use |
  |---|---|
  | `decimalai skills export <ref>`, `decimalai skills install <ref>` | `decimalai skills add <ref>` |
  | `SkillRouter.install()`, `install_skill()`, `export_to_disk()` | `router.export(name)` |
  | `SkillRouter.pull_missing()` | `decimalai skills install` |
  | `SkillRouter.pull_skill()` | `router.preview(name)` |
  | `export(agents=..., scope=...)` | nothing: there is one root, `.agents/skills` |
  | `decimalai.disk_export` (`export_skill_to_disk()`, `export_skills_to_disk()`, `AGENT_PATHS`) | `decimalai skills add` |

  `decimalai skills install` with no name is unchanged; its hidden 0.14 options `--agent`,
  `--scope` and `--out` are gone.
- **The version 1 lock fallback.** `router.export()` and `router.update_skills()` refuse a
  project whose `.decimal/skills.lock` decimalai 0.14 wrote, as every `decimalai skills`
  command already did, and name the fix: run `decimalai skills migrate` once.
- `SkillRouter.search_registry`, a second name for `search()`.
- `skill_router.should_auto_pull_to_disk()`, which nothing called, and the one-time warning
  that `instrument(disk_sync=True)` no longer writes skills to disk. `disk_sync` decides
  whether the project's own `SKILL.md` files are read and synced; it never wrote registry
  skills in 0.15 or later.
- **`project=`**, deprecated since 0.10.0 and never read by the platform (traces are scoped
  by the API key; group them with workspaces): on `decimalai.init()`, `DecimalAIClient()`, the
  LangChain `CallbackHandler`, ADK's `DecimalaiPlugin()`, and the Claude Agent SDK's
  `trace_stream()` and `traced_query()`, plus the CLI's hidden `--project` option.
  (`decimalai init --project <dir>`, which writes a Support project, is unchanged.) The trace
  schema keeps its `project` field, unset.
- `annotate_trace()`'s `text=` (pass `notes=`), and `annotation_type=` and `span_id=`, which
  the endpoint never accepted.
- `decimalai.skills.detect_skill_activations()`, the deprecated alias of
  `detect_skills_present_in_prompt()`: matching prompt text shows a skill was present, not
  that it was activated.

## [0.17.0] — 2026-10-19

### Changed: `init()` is the default agent name and API key

- `init(agent_name=...)` (or the new `DECIMAL_AGENT_NAME`) is now the agent name wherever
  one is not given: every adapter's `instrument()`, `decimalai.start_trace()`,
  `@decimalai.trace` (before the function's name) and `SkillRouter`. Before, only the
  adapters switched on in the same `init()` call saw it, so a later `instrument()` or a
  `start_trace()` filed traces under another name and routed without the agent's skills.
- `SkillRouter()` no longer needs `api_key=`: it takes `init()`'s key (and its `base_url`),
  then `DECIMAL_API_KEY` / `DECIMALAI_API_KEY`.
- **OpenAI Agents:** a DecimalAI agent name (from `instrument(agent_name=)`, `init()` or
  `DECIMAL_AGENT_NAME`) now wins over `Agent(name=...)` for both the trace and routing, as
  `instrument()`'s docstring always said; with none, `Agent(name=)` is used as before. A name
  given together with `agent=` applies to that agent only. An INFO line says so once when the
  two names differ. Pydantic AI keeps its documented rule (`Agent(name=)` first).

### Changed: `decimalai init` files name the agent once

- A generated `agent.py` assigns `AGENT_NAME` once and passes it to
  `decimalai.init(agent_name=AGENT_NAME)`; `instrument()`, `load_agent()` and the
  framework's own `Agent(name=...)` read it from there. Before, the name was written in
  up to four places, and renaming the agent in one of them filed traces under the old
  name with no error. The ADK file derives its node name from `AGENT_NAME` when it runs.
- Because the name now reaches `init()`, its check for a near-miss agent name (a likely
  typo of an existing agent) runs at startup: one extra `GET /api/v1/agents` when
  `verify` is on.

### Fixed

- **LangChain:** a tool's result is recorded as what the tool said. From a LangGraph
  tool node (`create_agent`, `create_react_agent`) the output is a `ToolMessage`, and the
  record held its repr (`content='...' name=... tool_call_id=...`).
- **LangChain:** an LLM call from a chat model that names no model in its invocation
  params (a custom model class, or a wrapper that drops the field) is recorded under the
  model's class name. Before, it was recorded with no model name and the platform
  rejected the whole trace.

### Deprecated (unchanged, and kept one release longer)

The 0.14 disk verbs (`decimalai skills export <ref>`, `decimalai skills install <ref>`,
`SkillRouter.install()`, `install_skill()`, `pull_missing()`, `export_to_disk()` and the
`disk_export` writers) and the version 1 lock fallback still work in 0.17.0, with the same
warning. The 0.16.0 notes said they would stop working in 0.17; they now stop working in
0.18, and the warnings say so.

## [0.16.0] — 2026-10-18

### Added: native mode — a framework's own skill loader, seen by DecimalAI

Eight frameworks load the skills `decimalai skills add` installed with a loader of
their own. Each now has a one-line helper that points that loader at
`.agents/skills`, and DecimalAI records every skill the loader hands the model as an
activation carrying the exact skill id and version id of the bytes the model
received. Each helper is in its framework's module (`import decimalai.adk` first):

| framework | helper |
|---|---|
| Claude Agent SDK | `ClaudeAgentOptions(plugins=[decimalai.claude_agent_sdk.skills_plugin()])` |
| Google ADK | `SkillToolset(skills=decimalai.adk.skills(), ...)` |
| CrewAI | `Agent(skills=decimalai.crewai.skills(), tools=decimalai.crewai.skill_tools())` |
| OpenAI Agents (sandbox) | `SandboxAgent(capabilities=[Shell(), decimalai.openai_agents.sandbox_skills()], ...)` |
| Deep Agents | `create_deep_agent(model, **decimalai.deepagents.skills())` |
| Microsoft Agent Framework | `Agent(..., context_providers=[decimalai.agent_framework.skills()])` |
| Pydantic AI harness | `Agent(model, capabilities=[LocalWorkspace(project), decimalai.pydantic_ai_harness.skills(project), FileSystem(), Shell()])` |
| Strands | `Agent(plugins=[decimalai.strands.skills()])` |

- **The version is the one the model received, not the one on disk.** ADK,
  Microsoft Agent Framework and Strands keep serving a skill they already loaded after
  `decimalai skills update` wrote a new version; the activation then names the old
  version, because that is what the model read. The version comes from the delivered
  copy's stamp, else the lock by the body's hash, else any version this process has
  seen. A body that is none of them is reported with its skill and hash and no
  version, and the platform resolves it if it can.
- **A skill loaded twice in one run is one activation.**
- **The router stays out of an agent whose own loader serves its skills** (ADK with a
  `SkillToolset`, CrewAI with `skills=`, an OpenAI `SandboxAgent` with `Skills`, Deep
  Agents, a Pydantic AI agent with the harness's `Skills`): no menu, no
  `load_skill` of DecimalAI's, so the model sees one catalog. Before, the OpenAI
  adapter renamed its own tool to `decimalai_load_skill` beside the sandbox's, and
  CrewAI warned that two catalogs reached the model.
- **Gap-fillers.** `decimalai.crewai.skill_tools()` gives CrewAI the resource reader and
  script runner it lacks (confined to installed skills; `scripts=False` leaves the
  runner out). `decimalai.agent_framework.skills()` adds the script runner Microsoft
  Agent Framework does not ship, and runs skill loads and reads without asking for
  approval (scripts still ask unless `auto_approve_scripts=True`).
  `sandbox_skills()` answers an unknown skill name instead of aborting the run, and
  warns when the sandbox would list no skills (it skips symlinked folders silently).
- Microsoft Agent Framework and Strands need no adapter: with `decimalai.init(otel=True)`
  their own OpenTelemetry spans become the trace, and the load is recorded on it.
- The Pydantic AI harness is graded at 0.57.0 (the `conformance-tests` floor), which keeps a
  user-only skill (`disable-model-invocation: true`) out of the model's catalog, as only
  Claude Code did before.

### Deprecated (unchanged, and kept one release longer)

The 0.14 disk verbs (`decimalai skills export <ref>`, `decimalai skills install <ref>`,
`SkillRouter.install()`, `install_skill()`, `pull_missing()`, `export_to_disk()` and the
`disk_export` writers) and the version 1 lock fallback still work in 0.16.0, with the same
warning. The 0.15.0 notes said they would stop working in 0.16; they now stop working in
0.17, and the warnings say so.

### Fixed

- `decimalai.flush()` now also exports the spans the `init(otel=True)` pipeline still
  holds. It drained only CrewAI's, so a short script could exit before a run traced
  through OpenTelemetry became a trace.
- The OpenTelemetry exporter joins each tool span's result onto the model's tool call
  (by `gen_ai.tool.call.id`), so a trace shows what a tool returned even when its
  input, cut at 200 characters, does not.

## [0.15.0] — 2026-10-15

### Added: registry skills on disk, managed like packages

A project lists the registry skills it wants in `.decimal/skills.toml`. Each one
is installed as a real folder, `.agents/skills/<install-name>/`, holding the
registry's own bytes, and `.decimal/skills.lock` (lock version 2) records every
file and hash it wrote. Commit the manifest and the lock; CI and Docker get the
same bytes back with `install --frozen`. `.agents/.claude-plugin/plugin.json`
makes `.agents` a Claude Code plugin named `decimalai-skills`.

- `decimalai skills add <ref>[@N|@latest]` resolves a registry name, url_slug,
  install name or skill id, records it in the manifest (`latest`, or pinned to
  version N), installs it and locks it. Adding it again with another version
  changes the version. `--no-install` only records it.
- `decimalai skills install` makes the folders match the manifest and the lock:
  new entries are installed, entries the manifest dropped are removed, and
  missing folders are fetched at their locked version. It never moves a version
  (that is `update`), and when every folder matches the lock it makes no
  network call.
- `decimalai skills install --frozen` is for CI and Docker. Nothing is
  resolved, and the manifest and the lock are never written; a missing folder
  is fetched by its exact lock entry and every byte is checked against it.
  `--offline` fails instead of fetching. The API key is sent only to the
  configured platform URL, never to a URL read from a committed lock.
- `decimalai skills update [<name>[@N|@latest]]` moves every `latest` entry to
  the newest version and leaves pinned entries alone. Each changed folder is
  replaced whole, so files a new version dropped disappear. If one skill cannot
  be updated, none is. `--dry-run` changes nothing; `--json` prints the plan.
- `decimalai skills remove <name>` deletes the folder and its manifest and lock
  entries. Nothing changes on the registry: Use links and forks stay.
- `decimalai skills status` shows each installed folder against the lock
  (clean, modified, missing, unmanaged), the manifest against the lock, an
  interrupted run, and a version 1 lock that needs `migrate`. `--remote` also
  asks the registry for newer versions, descriptions changed upstream and
  revoked versions. When `./skills` exists and a key is set, a second section
  compares the skills you wrote with the platform, as `status` did before.
  `--check` exits 1 on any drift, for CI.
- `decimalai skills verify [PATH] [--strict]` checks folders the way the native
  skill loaders read them, and writes nothing. In folders the lock lists every
  rule is an error. In other folders, and in `PATH`, the format rules every
  loader needs are errors and the SDK's own rules are warnings (`--strict`
  makes them errors).
- `decimalai skills migrate` converts a project with a version 1 lock (what
  0.14 and older wrote) or an npx lock: it writes the manifest and a version 2
  lock and installs every skill again from the registry. Old folders inside
  `.agents/skills` move to `.decimal/migrated-v1/`, old folders anywhere else
  stay and are listed, nothing is deleted, and a second run does nothing.
- A skill whose safety review is `caution` or `unreviewed` is asked about on a
  terminal and refused without one unless `--yes`. Blocked skills are always
  refused, and withheld ones unless your org owns or uses them. Skills your org
  owns skip the question.
- Without an API key, `add`, `install` and `update` work anonymously, for
  public skills only. A platform too old for these commands is refused with an
  error naming the missing route (exit 7); there is no silent fallback.
- The commands above take `--directory` (the project; by default the nearest
  folder with `.decimal/`, up to the repository root), and `status`, `verify`
  and `update` take `--json`. The exit codes are stable and listed in
  `decimalai skills --help`:

  | code | meaning |
  |---|---|
  | 0 | ok |
  | 1 | `status --check` or `verify` found problems; a journal conflict; another run holds the project |
  | 2 | usage error |
  | 3 | the lock or the manifest is missing or unreadable, or the lock is a version 1 or npx lock |
  | 4 | the lock is out of date with `.decimal/skills.toml` |
  | 5 | local edits, an unmanaged folder or a symlink block the operation (`--force` backs the folder up to `.decimal/backup/` and replaces it; symlinks are refused even then) |
  | 6 | integrity: served bytes differ from the declared or locked hashes, or `lock_hash` does not recompute |
  | 7 | unavailable: not found, ambiguous, withheld, blocked, version unsafe, no access, name taken, render gone or invalid |
  | 8 | network failure, or `--offline` needed a fetch |

- The same engine in Python: `from decimalai import skillpkg`, then
  `skillpkg.add()`, `install(frozen=True)`, `update()`, `remove()`, `status()`,
  `verify()` and `migrate()`. Each returns a `Result` of `changes`, `problems`
  and `exit_code`. Importing it loads no agent framework.
- **`decimalai.skills_dir()`** returns the project's managed root, absolute and
  symlink-resolved, for a framework's own skill loader;
  `skills_dir(relative_to=Path.cwd())` gives a relative path for loaders that
  want one. It raises when the project has no version 2 lock, naming
  `decimalai skills migrate` for a version 1 lock. The SDK's test suite runs
  eight native loaders on a project laid out as `install` writes it: Claude
  Code (through the Claude Agent SDK's `plugins=`), Google ADK's
  `SkillToolset`, the OpenAI Agents sandbox `Skills` capability, deepagents,
  Microsoft Agent Framework, Strands, pydantic-ai-harness and CrewAI. Claude
  Code also loads it with
  `claude --plugin-dir "$(git rev-parse --show-toplevel)/.agents"`.
- `log_skill_activation()` and `log_skill_loaded()` take `skill_id`,
  `version_id` and `body_sha256`, for a framework with no DecimalAI adapter
  that reads installed folders itself. Each installed `SKILL.md` carries them
  under `metadata` (`decimalai-skill-id`, `decimalai-version-id`,
  `decimalai-body-sha256`), and `discover_skills()` returns them on each
  descriptor.

### Changed: traces name the registry skill that ran

- A native loader reports an installed skill by its folder name, by its plugin
  name (`decimalai-skills:<install-name>`) or by its path, none of which the
  platform knows. Before a trace is sent, the SDK looks the name up in the
  project's version 2 lock and adds the registry name, skill id, version id and
  body hash. An entry that already carries ids, or whose hash names another
  body, is sent as reported. A folder edited locally is sent with the hash of
  the body on disk, `modified: true` and no version id, so the edit is not
  credited to the registry version. This covers the traces the SDK builds;
  `ingest_raw_trace()` sends its payload as given. A failed lookup never costs
  a trace.
- Installing a skill to disk no longer counts as serving it. `skills add`,
  `install`, `update`, `migrate`, `pull` and `SkillRouter.export()` fetch the
  registry's bundle, which the platform records as a fetch, separately, and
  the skill's served count does not move. In 0.14, `skills export` read the body through
  the delivery route and was counted as served; 0.14 clients still are.

### Breaking: no adapter writes skills to disk

- `instrument()` on LangChain and OpenAI Agents no longer copies platform skills
  into the working directory, and an existing `.agents/skills` folder no
  longer counts as permission to do so. Registry skills reach disk only when a
  `decimalai skills` command (or `skillpkg`) runs. If you asked for skills on
  disk (`disk_sync=True`, or `skill_authority="harness"`), a WARNING says so
  once per process and names `decimalai skills add` and `install`.
- `disk_sync` now only decides whether the adapter reads local SKILL.md files
  and uploads the ones you wrote. That upload, `decimalai skills sync`,
  `sync_to_platform()` and the authored section of `status` skip folders that
  carry the registry's stamp (`metadata.decimalai-skill-id`: installed, pulled
  or exported ones), so a registry skill is never copied into your org.

### Deprecated: the 0.14 disk verbs, which now run the package manager

The deprecated verbs below still work in 0.15, warn once per process and name
their replacement. Unless noted, each stops working in 0.16.

- `decimalai skills export <ref>` is `skills add <ref>`. `--agent` is ignored
  with a warning (there is one root), `--scope global` exits 2 and names
  `decimalai skills pull <ref> --out ~/.claude/skills`, and `--out` is taken as
  the project directory.
- `decimalai skills install <ref>` (with a reference) is `skills add <ref>` and
  **no longer forks** the skill into your workspace. For an editable copy use
  `router.fork()` or the dashboard. `install` with no reference is the new
  command above.
- `SkillRouter.install()` and `install_skill()` add the skill like `skills add`
  and **no longer fork**: the result's `fork` and `install` are None.
- `SkillRouter.pull_missing()` runs `skills install` when the project has
  `.decimal/skills.toml`. Without one it writes nothing and warns once with the
  command to run. It never copies every org skill to disk again.
- `SkillRouter.export_to_disk(skills=[...])` adds each named skill (deprecated,
  no removal version set). `export_to_disk()` with no `skills` adds every skill
  `list_skills()` returns, each listed in the manifest by name; that form stops
  working in 0.16.
- `disk_export.export_skill_to_disk()` and `export_skills_to_disk()` render
  SKILL.md locally in the registry's format and write folders no lock vouches
  for. They need the skill's registry ids: a 0.14-shaped `{name, description,
  body_markdown}` dict raises `ValueError`. They never create a lock, only keep
  an existing version 1 lock up to date, and never touch a version 2 lock.
- Not deprecated, but running the new engine: `SkillRouter.export(name)` is
  `skills add` (no fork, nothing read through the delivery route) and returns
  its old keys plus `lock_entry` and `result`; `SkillRouter.status()` reports
  the package manager's state for `project_root` and reaches the network only
  with `remote=True`; `SkillRouter.update_skills()` is `skills update`.
- **`update_skills(force=True)` means force.** A folder edited by hand stops
  the update (none is applied) unless `force=True`, which backs it up to
  `.decimal/backup/<txid>/` and replaces it. In 0.14 `force=True` re-wrote every
  skill whether or not it had changed upstream.
- Outside the version 1 fallback below, `agents=` on these methods is ignored
  with a one-time warning. `scope="global"` on `export()`, `install()`,
  `export_to_disk()` and `update_skills()` raises `ValueError` naming
  `decimalai skills pull <name> --out ~/.claude/skills`.
- In a project whose lock is still version 1, the old verbs keep working the
  0.14 way (folders where 0.14 put them, the version 1 lock updated) and warn
  once to run `decimalai skills migrate`. 0.16 removes this fallback.
- `decimalai skills status DIR` stays: it prints the authored section for DIR.

### Changed: `decimalai skills pull`

- It writes the registry's exact files to `<out>/<install-name>/`: SKILL.md
  with the registry's frontmatter and stamp, the bundled files, and
  `eval.yaml` when the skill has one. `--stdout` prints the served SKILL.md,
  frontmatter included. It is still a loose copy that nothing updates; `skills
  add` is the managed way.
- A folder that already holds different files is left alone with exit 5
  unless `--force`, which leaves files that are not part of the skill in
  place. 0.14 overwrote it silently. A folder inside a project's managed
  `.agents/skills` is refused with exit 5, with or without `--force`.
- Exit codes: 7 when the skill is not found, not public or withheld, 8 when
  the platform cannot be reached (after one retry on a timeout), 6 when a
  served file would land outside the folder or does not match its hash. 0.14
  exited 1 for every failure.

### Upgrading from 0.14

- A project with a `.decimal/skills.lock` that 0.14 or older wrote: run
  `decimalai skills migrate` once and commit `.decimal/skills.toml` and
  `.decimal/skills.lock`.
- **Do not run decimalai 0.14 or older in a project 0.15 manages** where it
  writes skills: `skills export`, `skills install <ref>`,
  `SkillRouter.export*()` / `install()` / `pull_missing()`, or an adapter with
  disk sync on (0.14's adapters write into `.agents/skills` when it exists).
  The old writer merges its own entries into `.decimal/skills.lock`;
  `install --frozen` then exits 3, `status` warns about the entries, and
  `skills install` drops them and leaves their folders unmanaged. Upgrade
  every checkout, CI image and agent host that touches the project together.

### Changed: reading SKILL.md and finding the project

- Frontmatter is read with PyYAML: quoted descriptions with escapes are
  decoded, list values are read and then joined into strings for callers, a
  value YAML would retype (`yes`, `404`, `1.10`, dates) keeps its written text,
  and invalid YAML falls back to the simple reader with a warning. A name may
  start with a digit (`2slides-ppt-generator` was skipped). Category, skill
  type, skill scope and invocation are read from `metadata` (`decimalai-*`
  keys) first, then the top level; `skills scan` reads the category the same
  way.
- Descriptors from `discover_skills()` carry `managed`, `registry_name`,
  `skill_id`, `version_id` and `body_sha256`. Discovery includes the managed
  root, searches the working directory before the project root (a monorepo
  package keeps its own skills), and lists a skill once.
- The project directory (for discovery, install identity and every
  `decimalai skills` command) is the nearest `.decimal/` with a manifest or
  lock, then any `.decimal/`, then the repository root. The search never goes
  above the repository, and never reaches the home directory from a folder
  under it, so run from such a folder outside a repository, the SDK no longer
  reads or uploads the personal skills in your home directory.
- `decimalai skills status` writes nothing. Without a sync baseline
  (`.decimal/install.json`, which `skills sync` creates) it asks the platform
  nothing and says to sync once; 0.14 created an install identity on the spot.
- `.decimal/.gitignore` is kept listing `install.json` and the engine's working
  files (`staging/`, `backup/`, `migrated-v1/`, the journal and the process
  lock); missing lines are appended and existing ones kept.
- `decimalai skills benchmark` sends a category only when it is text; a list
  made the platform refuse the request.

### Dependencies

- New core dependencies: `PyYAML>=6.0.2` (SKILL.md frontmatter) and `tomli>=2` on
  Python 3.10 (`.decimal/skills.toml`; 3.11+ uses `tomllib`).
- `skillevaluation>=0.8.0` (was `>=0.4.0`), for `skillevaluation.skillfmt`, the
  on-disk format module the registry renders with.

### Fixed

- **OpenAI Agents: a model turn that calls a tool records the call.** The
  `llm_call` of a turn whose output is a function call (a `load_skill` call, or
  any tool) carried `output: null`, no `tool_calls` and `finish_reason: "stop"`,
  so the trace hid what the model decided. Each call is now on `tool_calls`
  (tool name and argument dict) with `finish_reason: "tool_calls"`, and on
  `output.tool_calls` in the chat-completions shape with its call id. When the
  tool's span ends, the same entry gets its `status`, `result` and
  `latency_ms`, and takes the span's id, so a reader that joins tool calls with
  tool spans by id counts one call, not two. A handoff (`transfer_to_<agent>`)
  is closed by its handoff span the same way, with a result naming the target
  agent. A call never seen to run stays `running`. Both the Responses path and
  the chat-completions / LiteLLM path.
- **OpenAI Agents: tool span previews keep 2000 characters instead of 200**,
  which is what the platform stores, so a loaded skill body shows its first
  paragraphs.
- **Evals see the tool calls a trace recorded.** `trace_to_trace_data` read a
  tool call's name only from `name` or `function.name`, and its arguments only
  from a dict `function.arguments`. Every adapter records `ToolCallRecord`
  (`tool_name` + `args`), so each recorded call reached an eval as
  `ToolCallView(name="", args={})`. The reader now accepts `tool_name`, `name`
  and `function.name`, takes arguments from `args`, `arguments` or
  `function.arguments`, and parses a JSON-string `arguments` into a dict.
- **`decimalai init <agent>` no longer presents the attached skills as
  everything the agent uses.** The header listed `GET /agents/{name}/skills`
  as "Skills this agent will use", but an agent is also offered every skill its
  workspace owns unless that skill is set to "Only agents it's assigned to", so
  a skill created a moment ago was missing, and an agent with no attachments was
  told it would "run as a plain model call". The header now says "Skills
  attached to this agent" and states the workspace-wide offer, and the summary
  line counts "skills attached".

## [0.14.0] — 2026-10-14

### Breaking: every framework extra floors the framework's latest release

Older framework releases are no longer supported, and the code paths that
existed only for them are gone.

- New floors: langchain-core 1.6.10, langchain 1.4.5, langgraph 1.2.15,
  openai 3.30.0, openai-agents 0.23.1, pydantic-ai-slim 2.56.0, google-adk
  2.12.0, anthropic 1.13.0, claude-agent-sdk 0.2.166, llama-index-core 0.14.25,
  google-genai 2.30.1, litellm 1.105.0 (`[evals]`), crewai 1.15.27 (test
  extras). The `anthropic<1.0.0` and `langchain-core<2` caps are removed.
- `decimalai[pydantic-ai]` installs `pydantic-ai-slim[openai,google,anthropic]`
  instead of the `pydantic-ai` meta package, whose logfire extra cannot be
  installed beside google-adk.
- `[evals]` is no longer part of `[all]`: litellm pins `openai<3`, and
  `[openai]`, `[openai-agents]` and `[pydantic-ai]` need `openai>=3`.
- `[adk]`, `[pydantic-ai]` and `[all]` need Python 3.11+, because the google-adk
  and pydantic-ai-slim releases they floor require it. The core package and
  the other extras still install on Python 3.10.
- pip reports neither conflict. Asked for `[evals]` beside `[openai]`,
  `[openai-agents]`, `[pydantic-ai]` or `[all]`, or for `[adk]`,
  `[pydantic-ai]` or `[all]` on Python 3.10, it installs decimalai 0.13.9, the
  newest release whose requirements still resolve, and prints nothing (uv does
  the same). `pip show decimalai` says which release you got. A floor on
  decimalai itself, such as `pip install "decimalai[adk]>=0.14"`, makes pip
  fail with an error instead (on Python 3.10, "No matching distribution found
  for google-adk>=2.12.0"). `decimalai init --framework adk`
  or `--framework pydantic-ai`, run on Python 3.10, says so under the install
  line it prints.

### Breaking: the `init()` framework flags and `DECIMAL_AUTO_TRACE` deliver skills

- `init(langchain=True)`, `init(openai_agents=True)`, `init(adk=True)` and
  `init(crewai=True)` now turn the skill loader on, the same install as that
  adapter's `instrument(agent_name=..., enable_skill_loader=True)`. Through
  0.13.9 they installed tracing only, so the setup the docs teach traced well
  and delivered no skills. The provider flags (`openai=`, `anthropic=`,
  `google=`) stay tracing-only.
- `DECIMAL_AUTO_TRACE=langchain`, `openai-agents`, `adk` and `crewai` call those
  same flags at import, so a deployment set up with environment variables and
  no code starts delivering skills when it upgrades.
- With the loader on, those adapters send the turn's message (on CrewAI, the
  task description) to the platform's router before the model sees it, and add
  a skills menu to the prompt, plus a skill body on the adapters that inject
  one. The routing call has a 5 s read budget by default, and a repeat within
  30 s reuses the answer.
- Two opt-outs keep a flag to tracing only: `init(..., skills=False)` in code,
  and the new `DECIMALAI_SKILLS=0` environment variable, which is the default
  of `init(skills=...)` and the opt-out on the `DECIMAL_AUTO_TRACE` path, where
  there is no `init()` call to pass an argument to. `0`, `false`, `no` and
  `off` turn the loader off. A value the SDK does not recognise turns it off
  too, and logs a warning. An explicit `skills=` argument wins over the
  variable, and the variable does not change an explicit
  `instrument(enable_skill_loader=True)`.
- Re-running `init()` with a different API key, base URL, `inject_skill_body`
  or `load_skill_tool` drops every adapter's cached skill router. Before, a
  second `init(api_key=B)` sent traces as org B while routing org A's skills.

### Fixed: silent non-delivery and crashes

- Google ADK: `Runner(app=App(...))` and `InMemoryRunner(app=...)`, the shape
  `adk web` builds, no longer raise `ValueError` after `init(adk=True)` or
  `instrument()`. The DecimalAI plugin is added to a copy of the App, and an
  explicit `DecimalaiPlugin` under any name is no longer doubled by the global
  one.
- OpenAI Agents: `SandboxAgent` gets the skills menu and a working `load_skill`
  tool. When the sandbox's own Skills capability already registers
  `load_skill`, DecimalAI's tool is exposed as `decimalai_load_skill`.
  `instrument()` is safe to call more than once, including after
  `init(openai_agents=True)`: one trace processor, one trace per run.
  DecimalAI's `load_skill` result is never taken as the run's answer: an Agent
  with `tool_use_behavior="stop_on_first_tool"` used to end its run with a
  skill body, and a custom `ToolsToFinalOutputFunction` was handed DecimalAI's
  tool. Both now see only the agent's own tools.
- OpenAI Agents: an agent whose `instructions` is your own function (sync or
  async, on `Agent` or `SandboxAgent`) is routed. Your function runs as
  before, and the skills menu is appended to what it returns. Through 0.13.9
  such an agent got the `load_skill` tool but no menu and no routing
  decision, so its loads named none and the platform served any public skill
  by name. If your function already appends a fragment from your own
  `SkillRouter` (the 0.13.9 docs' workaround), remove that, or the menu
  appears twice.
- OpenAI Agents: runs with tracing disabled (`RunConfig(tracing_disabled=True)`,
  `set_tracing_disabled(True)` or `OPENAI_AGENTS_DISABLE_TRACING`) each get
  their own run scope. Before, every untraced run shared one (the Agents SDK
  gives them all the trace id `no-op`): one body budget between them, and each
  run's loads named whichever run had routed last. The adapter also kept a
  record of those runs that nothing read or released.
- Pydantic AI: the skills menu is an instructions function routed once per run,
  so every turn of a conversation (`message_history=`) is routed and traced,
  not only the first. A second tool named `load_skill` no longer makes every
  run raise `UserError`; DecimalAI's tool is exposed as `decimalai_load_skill`
  then. Routing uses the agent name the trace is filed under, not the variable
  name Pydantic AI infers for an unnamed `Agent`. `load_skill` sends that same
  agent name, so the body is the agent's pinned version, and a skill assigned
  to the agent loads even when the public registry hides it.
- Anthropic: `messages.stream()` and `messages.parse()` (sync and async) get
  skills, not only `messages.create()`. A run named with
  `decimalai.providers.agent_run(name)` is routed for that agent, the one its
  trace is filed under; it used to route as `instrument(agent_name=...)`, or as
  no agent, so the agent's assignments and pins were ignored.
- Google ADK: `Runner(agent=...)`, `Runner(node=...)` and
  `InMemoryRunner(agent=...)` no longer raise ADK's `plugins` DeprecationWarning
  after `init(adk=True)` when you passed no plugins. The DecimalAI plugin goes
  in through the `App` ADK wraps the agent in, so a suite run with
  `-W error::DeprecationWarning` can build a runner again.
- Google ADK: tool results reach the trace. Every ADK tool call used to be
  recorded as `success` with no result: the record built from the model turn
  never got one, and the tool callback's record, which had it, was dropped
  because a record of that name already existed. The result now lands on the
  matching record, found by the function-call id ADK assigns, so parallel
  calls to one tool each get their own. It is stored as bounded JSON (as text
  when it has no JSON form), and a tool that raises marks its own record.
  Results of ADK's own `SkillToolset` (`load_skill`, skill resources and
  scripts) keep the skill's name and frontmatter metadata whole. Those native
  loads are not yet counted as skill activations.
- `init(enabled=False)` also stops a skill loader an earlier `init()`
  installed: no message is routed and no skill is injected or loaded while the
  SDK is disabled. Before, traces stopped but every user message still went to
  the platform. `init(enabled=True)` turns it back on.
- LangChain: skills are routed for the agent the run's trace is filed under (a
  per-call `CallbackHandler(agent_name=...)` or a named graph), not only for
  `instrument(agent_name=...)`. Unnamed runs used to route as no agent, so
  agent assignments and version pins were ignored.
- LangChain on Python 3.10: an async `create_agent` run (`ainvoke` /
  `astream`) now logs a warning the first time its model call arrives without
  the run. asyncio on 3.10 cannot hand a run's config to an async LangGraph
  node, and `create_agent`'s model node calls the model without passing it on,
  so those calls route without the agent the trace is filed under (that
  agent's skills and pins are not offered) and are missing from the trace.
  This was silent, and is unchanged otherwise: it is LangChain's limit, and
  0.13.9 behaves the same. It corrects the 0.11.0 note below, which says
  per-run attribution holds on 3.10 with `langchain-core>=1.4.8`; that holds
  for runnables that pass their config on, not for this node. Sync `.invoke()`
  runs, nodes that pass `config` to the model, and Python 3.11+ are
  unaffected.
- The blocking routing call no longer runs on the event loop in the async
  paths of the LangChain, OpenAI Agents, Pydantic AI and ADK adapters, nor in
  `AsyncAnthropic`'s `messages.create()`, `messages.stream()` and
  `messages.parse()`. CrewAI hooks are synchronous by CrewAI's design; the
  adapter warns once when CrewAI runs one on a running loop (only the
  deprecated `CrewAgentExecutor` under `Crew.akickoff()` does).
- `AsyncAnthropic().messages.stream()` routes when its `async with` block is
  entered, not when it is called. It returns a manager with the same
  `async with` interface, which yields the SDK's own `AsyncMessageStream`, but
  is not an `AsyncMessageStreamManager` instance.
- Google ADK: model turns of one invocation that ask the same question at the
  same moment (a `ParallelAgent`'s branches) share one routing call, so a user
  turn is still one routing decision, and it is the one the trace names.
- CrewAI warns once when `Agent(skills=...)` / `Crew(skills=...)` puts CrewAI's
  own skill catalog in the same model call as DecimalAI's menu.

### Changed: skill routing

- A skill body is injected only when the platform says it ranked the result
  by this turn's message (`query_ranked`). A degraded result still injects
  when it is ranked: lexical-only retrieval during an embedding outage, or a
  fallback menu whose top row matched the message. The top row of a menu that
  nothing in the message matched no longer lands in turns like "thanks!".
  From a platform that does not send `query_ranked`, a `full_menu` result
  counts as unranked and retrieval as ranked. Two bodies are injected ranked
  or not, because there is no wrong one to pick: the `priority_skills` the
  platform offered, and the agent's only skill (a show-all menu whose
  `rows_total` is 1), so a reworded question in a one-skill workspace still
  gets the body on the adapters with no `load_skill` tool. That skill's body
  also reaches small talk.
- `load_skill` names the run's routing decision (`routing_id`) on each body
  request, and the platform decides what to serve. Skills the org owns, has
  installed or has assigned to the agent load whatever the menu showed,
  including one the menu cap left off this turn's menu. A skill reachable only
  through the public registry loads just when the run was offered it, so a
  prompt injection cannot load an arbitrary registry skill by name. The SDK
  refuses no name itself: a load from a router that has never routed (a
  hand-built menu, a direct `SkillRouter.load_skill()` call) names no decision
  and is not confined, and neither is a load against a platform that does not
  yet read `routing_id` on `GET /skills/{name}/body`.
- The in-turn `/body` fetch uses the 5 s hot-path budget and the circuit
  breaker, like `/route` and `/menu`. Disk export keeps the cold budget.
- `routing_status()` turns unhealthy on 401, 402, 403, 429 and 5xx responses
  (new `http_errors` and `last_status_code` fields), not only on timeouts. A
  404 for an unknown skill name is not a failure. It reports the failures of
  every API key and base URL the process routes with; one the process stopped
  using stops counting 60 s after its last call, once another is in use.
- The routing circuit breaker is per base URL and API key for failed
  responses (401, 402, 403, 429, 5xx), and per host for transport failures
  (timeouts, refused connections). One key's revoked credential, plan limit or
  rate limit no longer pauses routing for the other keys in the process, and
  the ERROR line names the likely cause instead of always suggesting a longer
  timeout.
- A slow failed routing call (a timeout, or any failure that took a second or
  more, such as a 429 whose `Retry-After` waits ran out) is cached for the
  turn, so the other model calls of that turn do not each wait it out again.
  For a caller that passes no `scope=` (LangChain, a hand-written loop) it is
  held 5 s, not the 30 s a success is kept, so one timeout no longer strips
  skills from other conversations sending the same message. A fast failure
  (an edge 429, a 5xx, a 401) is not cached: the next call retries, and three
  in a row open the breaker. An open breaker is not cached.
- `load_skill` tells the model when the platform was unavailable instead of
  saying the skill does not exist.
- Routing no longer asks the platform for skill attachments it discarded.
- Skill names with `#` or `?` are percent-encoded in `/body` requests.

### Added

- `decimalai agents create NAME` creates an agent from a terminal, the same
  call as the dashboard's create form. `--pack ROLE` starts it on the pack's
  starter prompt and starter skills (`--no-starters` keeps only the prompt);
  `--skill` attaches your own or public registry skills by name, slug or id,
  repeatable; `--prompt-file` sets the system prompt. Skill names resolve
  exactly. An unknown one stops the command before anything is created, and a
  skill the server refuses (such as a plan limit) is named in the output. The
  same call is available as `DecimalAIClient.create_agent()`. `decimalai init`
  now points at it when the agent doesn't exist yet.

### Conformance suite

- Three contract items, each red for at least one framework before this
  release's fixes: C17 (a tool call recorded as done carries what the tool
  returned), C18 (the agent a run is routed for is the agent its trace is
  filed under, including an agent the framework does not name) and C19 (every
  turn of a conversation is routed, on that turn's own message).
- An entry-shape axis (S1) grades the second documented ways into a
  framework: ADK `Runner(app=App(...))`, OpenAI `SandboxAgent`,
  `init(openai_agents=True)` followed by `instrument()`, and Anthropic
  `messages.stream()`. Each was red before this release's fixes.
- The matrix runs in two environments, `DECIMAL_TEST_ENV=openai3` and
  `openai2`, because no single environment can hold openai-agents beside
  crewai and litellm. A driver whose framework is installed below the SDK's
  floor reports NOT RUN instead of grading a release the SDK no longer
  supports.
- The probe answers routing in the platform's own shapes (`strategy`,
  `query_ranked`, the split between the stable menu and the per-turn tail),
  and the CrewAI driver sets the telemetry variables CrewAI reads.

## [0.13.9] — 2026-10-10

- OpenAI Agents native skill loaders now honor the executing agent's saved installation scope and version pin, including prebuilt agents, clones and handoffs. Custom user loaders remain unchanged.
- Parallel skill loads retain every delivered immutable version witness on the shared run, so the trace records the bodies that actually ran.
- Full skill menus preserve the executing agent's scope.

- ADK synchronous `Runner.run()` preserves the enclosing DecimalAI trace as
  its parent across ADK's worker thread. The handoff belongs to each invocation,
  including reused and concurrent runners; explicit plugin parents still win
  and standalone invocations remain roots.
- Skill delivery keeps complete immutable bodies or omits them when they do
  not fit. The default per-body character cap now follows the existing
  6,000-token body budget (24,000 characters with the same estimate); explicit
  caller caps still apply. Prompt injection and `load_skill` reject partial
  server responses and enforce the total budget on the first body too. A
  refused body emits no delivery or load witness. Previously bodies above
  8,192 characters were clipped while retaining their full version's hash,
  preventing the recorded prompt from proving which complete version ran.
- CrewAI gets automatic skill delivery (crewai>=1.15.3).
  `from decimalai.crewai import instrument;
  instrument(agent_name="support", enable_skill_loader=True)` routes every agent
  turn through `SkillRouter` and inserts the menu and skill bodies after CrewAI's
  own system prompt, using CrewAI's public `before_llm_call` hook. Same telemetry
  as the LangChain and ADK loaders: the trace carries the `routing_id`, the
  skills offered in the prompt and the skills whose body was delivered, and
  nothing is recorded as activated (there is no `load_skill` tool on this rail).
  Routing uses the task description, so CrewAI's own wrapper text and reflection
  prompts don't skew it; a task's later model calls replace the block instead of
  stacking it. `instrument()` also turns on CrewAI tracing — the exporter
  `init(crewai=True)` installs — once per process, so `init(crewai=True)`
  followed by `instrument(enable_skill_loader=True)` adds only the loader. Calls
  CrewAI makes outside an agent executor (output conversion, guardrails,
  planning) are left alone. `decimalai init --framework crewai` now says there
  is no template yet instead of "no prompt seam".
- `decimalai.crewai.instrument(on_trace=fn)` hands you each CrewAI run's trace,
  on the contract the LangChain and ADK observers already keep: `fn` is called
  once per run, failed runs included, with a detached copy of the finished
  `RunTrace` (its `id`, `parent_trace_id`, routing id, delivered skills and
  model calls), before export. An observer that raises is logged; the run and
  its trace carry on. Each `instrument()` call sets the observer for runs that
  start afterwards. Before this, nothing a caller held could name the CrewAI
  trace, which is assembled from OpenTelemetry spans under an id minted at
  export.
- A CrewAI run made inside `decimalai.start_trace()` is now filed as that
  trace's child (`parent_trace_id`), captured when the run starts, the way
  LangChain and ADK runs are. This applies to `init(crewai=True)` as well. The
  skill router also stamps a delivery onto the enclosing trace, and the
  platform credits the use once only when the run's trace is that trace's
  linked child (and the enclosing trace made no model call of its own), so a
  CrewAI run wrapped in `start_trace()` was credited twice.
- `decimalai.flush()` first exports CrewAI runs still queued in the
  OpenTelemetry batch processor, waiting at most five seconds, and only then
  drains the sender. A short-lived process can now report on a CrewAI trace
  that exists. The SIGTERM handler does the same. Processes that never set up
  CrewAI tracing are unaffected.
- The `conformance-tests` extra's CrewAI floor moves from 1.6.1 to 1.15.3, the
  lowest version that passes the CrewAI conformance column. 1.6.1 could not be
  traced (the OpenInference instrumentor needs crewai>=1.10.1), crewai
  1.7–1.14.6 cannot be installed beside decimalai (they pin `click<8.2`), and
  1.14.7 and 1.15.0–1.15.2 lose the model's reply text from the trace.
- Long-running processes no longer start logging "skill rail overflow" after
  about 4,096 runs. The ADK, Anthropic, OpenAI Agents and Pydantic AI skill
  loaders each left a per-run entry on the skill router that nothing removed:
  from run 4,097 on, every new run evicted an older entry and warned, and a
  LangChain run in flight in the same process could lose its routing decision.
  Every loader now releases its entry as soon as the routing call returns, and
  the OpenAI Agents `load_skill` tool does the same for each skill it loads.
- `DECIMAL_AUTO_TRACE=crewai` now traces CrewAI. It used to install only the
  generic OpenTelemetry exporter, which CrewAI never sends spans to, so it
  captured nothing; it now does what `init(crewai=True)` does.
- `init(crewai=True)` no longer reports "tracing enabled" when the OpenInference
  CrewAI instrumentor refuses the installed CrewAI (it instruments only
  crewai>=1.10.1). It warns that no CrewAI traces will be captured and names the
  version conflict. The same applies to the provider instrumentors that
  `init(crewai=True)` and `init(openai=True)` / `anthropic=` / `google=` turn
  on: a refused instrumentor is no longer reported as enabled or recorded as
  instrumented.
- Test extras: `crewai-tests` floors crewai at 1.15.3 (was 1.14.6, which cannot
  be installed beside decimalai), and `providers-tests` floors
  `openinference-instrumentation-openai` at 0.1.45 (was 0.1.18). Every release
  up to 0.1.44 fails to instrument under wrapt 2.x, which every fresh install now
  gets, so no OpenAI call was traced.

## [0.13.8] — 2026-10-06

- `decimalai starter support --project ./support-agent` downloads a frozen,
  runnable Support project without a DecimalAI account. The standard-library
  runtime uses your OpenAI key locally and uploads no traces;
  `python3 agent.py --check-setup` verifies files and `--checks` makes two task
  and two judge calls. When a selected skill can't be bundled, the error names
  it and prints the command that builds the project without it. Uses the
  platform's public starter endpoint (live since 2026-10-06).
- Platform requests retry within a 30-second total budget and never before a
  server-directed `Retry-After`. Project checks name DecimalAI quota, rate-limit,
  credential and backend errors separately from provider quota and provider
  429s, so the suggested next step matches the cause.
- Generated Support project READMEs explain returning customer work and the
  public usage contract, including fleet activity, publication timing and
  incomplete historical coverage.

## [0.13.7] — 2026-09-24

- Generated Pydantic AI agents recover from temporary 429/503 provider failures
  inside the model request, preserving completed tool results. Each request has
  up to three attempts within 30 seconds; the generated turn has a 90-second limit.
- The opt-in `decimalai.pydantic_ai.retry_model_requests` helper distinguishes
  depleted billing from throttling, respects Retry-After, and propagates terminal
  failures and cancellation. Provider SDK retries share the same time budget.
  Streaming requests are passed through unchanged.
- Request outcomes and attempt counts now survive OpenTelemetry export into
  DecimalAI span attributes, including failed attempts that later recover.

## [0.13.6] — 2026-09-23

### Fixed
- The Support sample judge could reject a correct conditional billing-review path,
  although that path is explicitly in the supplied ticket facts. Clarify the same
  rubric and calibrate against the exact production reply that was falsely rejected.
  Promises to perform reviews or account actions remain failures.
- Failed behavior checks now print each failed criterion and its reason beside the
  case result, so users can diagnose a failed check without opening the JSON receipt.

## [0.13.5] — 2026-09-23

### Added
- `decimalai init <agent> --project <new-directory>` generates a runnable LangChain
  Support draft agent, two sample behavior checks, dependency pins and a deployment
  handoff README. The generated agent and checks use the same prompt and selected
  skill bodies. Requires the Support starters and a platform with setup checks enabled.
- `python check_agent.py` records each attempt separately and confirms its exact
  stored traces and delivered configuration. `--resume` retries delayed trace
  confirmation without another model call; edited configuration makes evidence stale.
- Optional LangChain `instrument(runtime_policy=...)` appends application constraints
  after skill bodies. Generated draft agents use it with a bounded reply guard to
  reject unsupported action promises. This is a narrow draft guard, not a general
  safety guarantee or permission to take customer actions.

### Fixed
- LlamaIndex 0.14.25 starts streamed synthesis after `query()` returns. Lazy model
  spans now stay in the original query's trace while each stream chunk is consumed,
  retaining token counts and answer text without mixing interleaved requests.

### Scope
- The new project defaults to `gpt-5.4-2026-03-05` and two clearly marked sample
  support tickets. Passing these checks does not certify broader behavior, skill
  lift, hosted deployment, or a five-minute onboarding claim. Existing single-file
  scaffolds and other framework defaults are unchanged.

## [0.13.4] — 2026-09-13

### Added
- **`skills sync` carries the bundle.** Text files one level under `references/`, `scripts/`,
  `templates/` and `assets/` beside a `SKILL.md` are sent as the sync item's `attachments`
  and mirrored onto the new version (server caps: 50 files, 500 KB each, text only). Until
  now the CLI synced `SKILL.md` and `eval.yaml` and silently left the bundle behind, while
  `skills pull` delivered bundles an author had never been able to push. Programmatic
  `sync_to_platform` does the same. (CUJ 27)

## [0.13.3] — 2026-09-08

### Fixed
- **A LangChain run inside a generic trace is now its child, not an unrelated root.** An app
  can wear both tracers at once — `@decimalai.trace` / `start_trace()` around the call and the
  LangChain adapter installed process-wide by `instrument()`, the composition the README
  documents — but no adapter consulted the generic context, so one logical run shipped as TWO
  root traces. The generic envelope only collects what `log_llm_call` puts in it, so with the
  model calls going through LangChain it arrived with zero `llm_calls` and zero spans. Measured
  on DecimalAI's own fleet (2026-09-07): 3,604 such traces in 3 days, 31.9% of the langchain
  path against 0.00% on every other framework, each an empty envelope beside a full sibling.
  Downstream, an empty envelope reads as having used no components and is graded `keep` at
  score 1.0 against every manifest (100% keep, versus 80.1% for real traces), so those runs
  could never be flagged for replay. The adapter now captures the live generic trace id when a
  root run OPENS and sends it as `parent_trace_id`; the platform already walks children when
  grading, so the envelope's grade becomes evidence-based with no server change. An explicit
  sub-agent `parent_trace_id` still wins — the enclosing trace is a fallback, never an
  override — and the probe is fail-open (a raise while labelling a trace is worse than an
  unlabelled trace).
  **Known limit:** the generic context is a `ContextVar`, and LangChain copies the context for
  parallel work, so a run started through `.batch()`, `RunnableParallel` or an executor thread
  cannot see the enclosing trace. Absent means "unlinked, exactly as before" — never a wrong
  parent.
- **`decimalai traces import <file>.jsonl` answered 422 on every call.** `POST /api/v1/traces/import`
  takes multipart form-data (`file` + `agent_name`), and the CLI posted the raw bytes as
  `application/x-ndjson` — the command the endpoint's own docs sample prints, and the one the
  LangSmith/Braintrust migration guide gives, ended in an httpx traceback for everyone who ran
  it. It now uploads as multipart through a client that drops the SDK's pinned
  `Content-Type: application/json` (otherwise httpx's boundary never reaches the server),
  posts to the canonical `/api/v1/traces/import` / `/api/v1/traces/import-bulk` paths instead
  of the deprecated `/api/v1/import/*` aliases, gains `--agent-name` (required for JSONL unless
  every row carries `agent_name`), and prints the server's `detail` on a 4xx/5xx instead of a
  bare status code.
- **`decimalai skills sync` could overwrite uncommitted edits with the remote copy.** The local
  timestamp preferred git's commit time over the file mtime, which for a SKILL.md with
  uncommitted edits reports the author's newest work as the oldest state on disk. `newer_wins`
  then answered "remote wins" and — `--apply-pulls` defaults on — wrote the older body back over
  the edit with `↓ <name> → SKILL.md (v1)` as the only notice. The commit time is now used only
  while the file is clean; a dirty or untracked file reports its mtime. A fresh clone is still
  clean, so the checkout protection this existed for is untouched.
- **`decimalai skills sync` counted a frontmatter-only edit as nothing.** The backend reports
  `reconciled_metadata` for a body whose hash matched but whose frontmatter diverged — a real
  write that mints no new version. The summary knew five actions and not that one, so an author
  who edited a description saw every counter read zero. It now prints a `metadata` count and
  `~ <name> (metadata only — no new version)` per skill. Filing it under `no_change` would have
  said the edit did nothing.
- **`decimalai skills benchmark` crashed before printing the report URL on a run with no
  measured lift.** The backend always emits `pass_rate` and nulls its members when nothing was
  measured (every case's LLM call refused, or every expectation calibration-gated — the normal
  first-draft `eval.yaml` outcome), so `delta_pts >= 0` raised `TypeError`. Null members now
  render as `—`, and a run whose verdict is `error` wears `✗`, not the `✓` a passing run gets.

### Added
- `update_skill(offer_scope=...)`: set who the trigger router offers a skill to. `"workspace"` (the default) is every agent in the org; `"restricted"` is only the agents it has been assigned to. Until now a skill could not be taken off the ambient menu at all, so assigning it to one agent never stopped the router considering it for the others. Metadata only — it does not mint a new version.

### Docs
- README: the built-in deterministic scores (`completion`, `has_output`, `tool_compliance`,
  `latency`, `token_efficiency`) are attached by the **LangChain adapter only** — the kill
  switch is `instrument(..., builtin_evals=False)`, not `install(...)`; `inject_skill_body` is
  documented as tri-state (defaults on for `langchain` / `anthropic` / `adk`, off for
  `openai_agents` / `pydantic_ai`), and the delivery and framework tables gain their `adk` and
  Pydantic AI rows.
- RELEASING.md no longer says releases carry no attestation: every release since 0.10.3 is
  uploaded by `publish.yml` over Trusted Publishing and carries provenance.

### Internal
- The pre-commit ruff hook (`scripts/install-git-hooks.sh`) lints the **staged** files under
  `decimalai/` rather than the whole tree, so one session's half-finished edit no longer blocks
  another session's commit in a shared worktree. Its flags are literal and pinned against
  `ci.yml`'s Lint job by `tests/test_local_lint_hook.py`.
- The `openai_agents` conformance driver's comment stops claiming that seeding the registry
  grades the fabricated-activation mutation; what grades it is
  `tests/test_inference_never_writes_the_activation_rung.py`.

## [0.13.2] — 2026-09-05

### Added
- `smart_route()` now names WHY it returned an empty menu: `degraded_reason` is `http_<status>` / `timeout` / `circuit_open` / `transport`, so a caller can tell a Cloud Run edge abort (a 0 s 429) from a slow read or an open breaker instead of filing every empty menu as "no skill offered".

- **`decimalai init --framework adk`** writes a runnable Google ADK agent:
  Gemini by default (`gemini-3.5-flash`; a non-Gemini `--model` is refused up
  front, because `decimalai[adk]` installs no other provider), a `GOOGLE_API_KEY`
  guard on line one, and skills delivered by prompt injection through
  `instrument(agent_name=..., enable_skill_loader=True)`. The template was
  withheld on the premise that the hermetic J1 journey could not stand in for
  Gemini; that premise was false — google-genai honours `GOOGLE_GEMINI_BASE_URL`
  (since v1.72, google-adk 2.0.0's own floor), so the journey stub now speaks
  `:generateContent` and grades the ADK cell like the other three. Nothing in
  the generated file knows about the stub.
- **Every generated file refuses to run without the provider key its `MODEL`
  needs**, naming the variable and the `Set:` block `decimalai init` printed.
  Before, a missing `OPENAI_API_KEY` was a traceback from inside the provider
  client — after `init()`, `instrument()` and `load_agent()` had already made
  their network calls (on `openai-agents`, only at the first `Runner.run_sync`).
  The guard is derived from the same tables as the `Set:` block and is omitted
  where those tables only guess (a local provider, Pydantic AI's `test` model, an
  unknown prefix such as `azure_openai:`), so it never refuses a correct setup.
- The `langchain` template says at the call site that `create_agent` runs with
  `recursion_limit=9_999` (not langchain-core's 25) and how to cap it once a tool
  is added — as visible as `MAX_TURNS` / `MAX_REQUESTS` on the other templates.
- The conformance probe serves a `content_hash` with every skill body, the way
  the platform does, and C13b now requires a model-pulled activation to carry
  that hash. Before, no conformance run ever saw a `skill_hash`, and the one
  fidelity clause that existed skipped a null silently — an adapter that stopped
  stamping versions would have stayed green.

### Fixed
- A refused manifest registration no longer costs the trace. The caller-thread registration
  ladder is milliseconds by necessity, which cannot outlast a Cloud Run revision with no
  available instance; the trace was then shipped carrying a LOCAL id the platform had never
  stored and rejected with `400 manifest_id '...' does not exist`. Production took 1,315 of
  exactly those in the 48 h to 2026-09-04, all from this SDK. Re-registration now happens on
  the background sender, where waiting costs the caller nothing: five jittered attempts,
  bounded, and the trace is held until an id exists rather than sent under one that does not.
  Applies to every adapter that registers a manifest — langchain, the generic
  `@decimalai.trace` path, ADK, LlamaIndex, OpenAI Agents and the Claude Agent SDK. The
  last four had the same defect with no retry ladder at all, in two shapes: two stamped
  the local id and took `400 manifest_id ... does not exist`, two sent no id and took
  `400 manifest_id is required`. Both lose the trace.
- `_poll_for_trace` in the framework end-to-end tests raises `TraceNeverArrived` instead of a
  bare `AssertionError` whose message contained "Timed out" — the substring the live-test
  helper matches to classify a failure as provider-unavailable and SKIP it. A lost trace was
  being reported as a quota skip, the same defect class as the 2026-09-03 journeys finding.

- **`decimalai init` gave up on the first admission abort.** A single-instance
  backend at its concurrency limit answers HTTP 429 in ~0 ms with no Retry-After
  (Cloud Run's "no available instance"); measured on production 2026-09-03 that
  was 84-92% of requests in a ten-minute window. `init` made one
  `/api/v1/auth/verify` call — and `decimalai init <agent>` one bare
  `GET /api/v1/agents` and one `GET /api/v1/agents/{name}/skills` — and exited on
  any non-2xx, so the quickstart failed outright, and the scaffold wrote no
  `agent.py`, for anyone who ran it while the platform was busy (the fleet's
  scaffold canary filed those as `decimalai_init_wrote_no_agent_file`). All three
  calls now retry a 429/502/503/504 twice (0.5 s, then 1.5 s; a short
  `Retry-After` wins) before reporting the status. A 401/403 is a real answer and
  is never retried. The prompt read was already on the SDK's retry ladder and
  still only costs a note in the file, never the file.
- **The `openai-agents` scaffold died with a traceback when the model called a
  tool the file does not define.** The Support pack's starter prompt named
  `kb_search`; a model that called it raised
  `agents.exceptions.ModelBehaviorError: Tool kb_search not found` out of
  `Runner.run_sync` (the fleet's scaffold canary, 2026-09-04). `run()` now catches
  `ModelBehaviorError` the way it already caught `MaxTurnsExceeded` and returns a
  message naming the mismatch between the prompt's Tools line and the tools
  attached to the Agent. The hermetic J1 journey now replays that class: a second
  run of the generated file against a stub that calls an unknown tool must exit 0
  without a traceback.

## [0.13.1] — 2026-09-03

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

- **A sub-agent's trace was built and never sent.** `on_chain_end` closed a run
  only when `parent_run_id is None`. That is the outermost callback of a
  top-level `chain.invoke()`, and it is the right boundary for that case — but
  it is not the only way a run finishes. The documented multi-agent pattern
  gives a nested `.invoke()` its own `CallbackHandler`, and that inner invoke
  runs inside the outer run tree: LangChain hands the sub-agent's handler
  exactly ONE chain end, carrying `run_id == parent_run_id == root`, and never
  one with a null parent. So every sub-agent trace, on every multi-agent run,
  was assembled and dropped. Runs are now closed on an open-chain balance —
  chain starts minus chain ends on that run — which reaches zero exactly when
  the run is done and cannot be tripped early by langchain-core 1.5.x reusing
  the root run_id for child steps.
- **The ADK adapter left the entire system prompt out of its traces.** It built
  `rendered_input` from `llm_request.contents` — the conversation turns — while
  ADK keeps the agent's instruction, and every routed skill body the adapter
  appends, in `config.system_instruction`. A separate field, and one nothing
  recorded. The adapter could route a skill, append its BODY, verify the append
  survived, and emit a trace with no skill text anywhere in it: delivery that
  was real and unwitnessable, which is the one state indistinguishable from
  never delivering at all. Every downstream signal reads that field — the SDK's
  own prompt-rung inference, the platform's activation detection, and the
  conformance contract's delivery items.

### Added

- **`decimalai.adk.instrument()` and `DecimalaiPlugin` accept
  `enable_load_skill_tool`.** Accepted and DORMANT, as on the LangChain
  adapter: ADK registers no `load_skill` tool, so the model cannot ask for a
  body and the strongest rung this rail reaches is DELIVERED. Passing True logs
  a warning naming the adapters that do have a tool loop and stays on prompt
  injection. Accepted rather than rejected on purpose — a flag silently dropped
  is how an adapter ends up credited with a channel it does not have.

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

### Removed

Recorded retroactively on 2026-09-05. These three `skills` subcommands were dropped in
`57d64a6` and shipped gone in this release with no entry, so a reader upgrading from
0.10.0 found them missing and nothing said why:

- `decimalai skills verify` — pushed a local benchmark bundle for the platform to
  re-judge. It moved server-side: a fleet-attested or replayed run now carries its own
  verification method, and a client-supplied bundle was a trust boundary in the wrong
  place.
- `decimalai skills attest` (hidden) — the fleet's own attestation path, never a
  customer-facing command.
- `decimalai skills spotcheck` — sampled a published run for re-grading; the same
  server-side move covers it.

Nothing replaced them in the CLI on purpose. Verification is a property of how a run
was produced, not something a caller can assert about a run afterwards.

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
