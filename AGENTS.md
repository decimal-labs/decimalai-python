# AGENTS.md — decimalai (Python SDK)

Machine-readable guide for AI coding agents working with or on this package.

## What this package is

`decimalai` is the open source Python SDK for [DecimalAI](https://decimal.ai): manifest-aware
agent change management. It captures traces and structural manifests (tools + prompts + models +
skills) from agent frameworks, sends them to a DecimalAI workspace, and lets you run a structural
regression check of an agent change against recorded traces. It also manages SKILL.md skills
(discover, sync, install from the public registry).

- Package name: `decimalai` — one word, never a hyphen or an underscore between "decimal" and "ai".
  (CI greps the whole tree for the separated spellings, so don't write one out here — not even as
  an example of what not to do. The guard can't tell a use from a mention.)
- Language: Python, `requires-python >= 3.10`, fully typed (`py.typed` ships)
- License: MIT
- CLI entry point: `decimalai` (Click-based; `decimalai --help` lists commands)

## Install

```bash
pip install decimalai          # core: tracing, CLI, manifests, skills
uv pip install decimalai       # same, via uv
pip install "decimalai[all]"   # all framework/provider adapters
```

Adapter extras (core stays thin; install the one matching the user's stack):
`[langchain]`, `[langgraph]`, `[openai]`, `[openai-agents]`, `[llamaindex]`,
`[claude-agent-sdk]`, `[pydantic-ai]`, `[adk]`, `[evals]`, `[all]`.

## Demo / smoke test

```bash
export DECIMAL_API_KEY="dai_sk_..."   # free key from https://app.decimal.ai/settings
decimalai demo regression             # seeds a reference agent, runs the real regression pipeline
decimalai demo skills                 # seeds the skills demo
decimalai demo reset                  # removes all demo data
```

The demo needs an API key (it seeds data into a workspace). Its numbers come from a seeded
reference agent — illustrative, not customer traffic. Keyless commands: `decimalai skills pull
<slug>` (fetch a published SKILL.md, no account) and the `agentversion` manifest diff flow
(fully local).

## Core API surface

```python
import decimalai
decimalai.init(langchain=True)        # or openai_agents / adk / llamaindex / claude_agent_sdk / ...
@decimalai.trace(agent_name="...")    # manual tracing for any framework
decimalai.export_manifest(snapshot)   # → an agentversion manifest dict (works offline)
```

Environment variables: `DECIMAL_API_KEY`, `DECIMAL_BASE_URL` (default `https://api.decimal.ai`),
`DECIMAL_AUTO_TRACE`.

## Documentation

- Docs site: https://docs.decimal.ai (Python SDK: https://docs.decimal.ai/sdk/python)
- LLM-friendly full docs: https://docs.decimal.ai/llms-full.txt (index: https://docs.decimal.ai/llms.txt)
- Skill registry index for agents: https://app.decimal.ai/llms.txt
- Changelog: https://docs.decimal.ai/changelog

## Related open specs and tools

- [`agentversion`](https://pypi.org/project/agentversion/) — open spec for agent version
  manifests, diffing, and compatibility classification. The SDK's manifests conform to it.
- [`skillevaluation`](https://pypi.org/project/skillevaluation/) — open A/B spec + runner for
  measuring a skill's lift; anyone can re-run a published number.
- [`regression-check`](https://github.com/decimal-labs/regression-check) — GitHub Action that
  runs the per-PR structural regression check.

## Developing in this repo

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q --ignore=tests/test_langchain_compat.py --ignore=tests/conformance
ruff check decimalai/ --select I,E,W,F --ignore E501,E402,F821,F841
git grep -nE 'decimal[-_]ai'   # must find nothing
```

Default pytest run excludes `integration` and `live_llm` markers (they need a live backend /
provider keys). CI additionally runs the LangChain compatibility matrix.

### Framework conformance (`tests/conformance`)

One contract, graded against every framework adapter on real HTTP payloads — the suite that
answers "does this adapter actually emit a valid trace?", which 626 mock-driven adapter tests
never could. Tier A is hermetic (stub models, a local probe server, no key, no backend), so it
gates every push. This repo has no Makefile; the one command is:

```bash
pip install -e ".[dev,conformance-tests]"    # the eleven frameworks the drivers drive
python -m pytest tests/conformance -q -rs -m conformance   # ~90s; ends with the matrix
```

`pip install -e ".[dev]"` alone is enough to run `tests/conformance/test_coverage.py`, the guard
that fails when a framework is advertised with no driver. Read `tests/conformance/README.md`
before adding a framework, changing an assertion, or reacting to a red row.
