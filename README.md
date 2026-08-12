# DecimalAI Python SDK

The open source SDK for [DecimalAI](https://decimal.ai) — the manifest-aware platform for agent change management.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/support-agent/support_agent.ipynb)

## Installation

```bash
pip install decimalai
```

Requires **Python 3.10+** (`pip` won't install current releases on older Pythons).

The core install is deliberately thin (tracing, CLI, manifests, skills). Framework and provider adapters ship as extras — install the one matching your stack:

```bash
pip install "decimalai[langchain]"       # LangChain (add [langgraph] for LangGraph)
pip install "decimalai[openai-agents]"   # OpenAI Agents SDK
pip install "decimalai[all]"             # everything
```

Available extras: `[langchain]`, `[langgraph]`, `[openai]`, `[openai-agents]`, `[llamaindex]`, `[claude-agent-sdk]`, `[pydantic-ai]`, `[adk]`, `[evals]`, `[all]`.

## See it in 2 minutes

Both flagship workflows ship with a one-command sandbox — realistic data seeded into your workspace, so you don't have to wait to accumulate your own. Set your API key, then run either demo:

```bash
export DECIMAL_API_KEY="dai_sk_..."   # from app.decimal.ai/settings
```

**For engineers — catch regressions before they ship**

```bash
decimalai demo regression   # → impact report: what your next change would break
```

**For prompt engineers — find skills that actually work**

```bash
decimalai demo skills       # → registry ranked by real production effectiveness
```

Browsing without an account? Explore the [public skill registry](https://app.decimal.ai/skills) — no signup required.

## Quick Start

### LangChain / LangGraph — Zero-Code Tracing

```python
import decimalai

decimalai.init(langchain=True)  # That's it — all LLM calls auto-traced

# Use LangChain as normal — nothing else changes
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
result = llm.invoke("Hello!")  # ← Auto-captured by DecimalAI
```

### OpenAI Agents SDK

```python
import decimalai

decimalai.init(openai_agents=True)
```

### Any Framework — Manual Tracing

```python
import decimalai

decimalai.init()

@decimalai.trace(agent_name="my-agent")
def run_agent(query):
    msgs = [{"role": "user", "content": query}]
    resp = openai.chat.completions.create(model="gpt-4o", messages=msgs)
    decimalai.log_llm_call(
        model="gpt-4o",
        input=msgs,
        output={"content": resp.choices[0].message.content},
    )
    return resp.choices[0].message.content
```

### Environment Variable Setup (No Code Changes)

```bash
export DECIMAL_API_KEY=dai_sk_...
export DECIMAL_AUTO_TRACE=langchain   # or "openai-agents"
# Just run your app — tracing activates on import
python app.py
```

## What It Does

- **Auto-tracing** — Captures LLM calls, tool calls, and agent steps with zero code changes
- **Agent versioning** — Auto-detects tool schemas, prompts, models, and graph topology
- **Change detection** — Detects when your agent configuration drifts
- **Inline evals** — Run eval functions on every trace with `@decimalai.evals.eval`
- **Built-in deterministic scores** — `completion`, `has_output`, `tool_compliance`, `latency`, `token_efficiency` attached to every trace by the SDK (disable with `install(..., builtin_evals=False)`). These run **in your process**, not server-side — bare HTTP `POST /traces` does not auto-score.
- **Batch evals** — Run evals offline across historical traces
- **Dataset pull** — One-liner to download versioned training data: `decimalai.pull_dataset("ds_abc", "./data.jsonl")`
- **HuggingFace Hub** — Push datasets to HF Hub for instant Axolotl/Unsloth/TRL compatibility
- **Fine-tuning** — Launch fine-tuning jobs on OpenAI, Together.AI, or Gemini from the platform
- **Skills management** — Auto-discover SKILL.md files, sync to platform, install from registry
- **OTel compatible** — Export spans to any OpenTelemetry backend

## Skills

DecimalAI auto-discovers your existing [SKILL.md](https://agentskills.io) files and provides observability — tracking which skills activate, how effective they are, and how they change over time.

### Auto-Discovery (Bring Your Own Skills)

If you already have SKILL.md files (from `npx skills add`, your team's repo, or hand-written), the SDK discovers them automatically:

```python
import decimalai
decimalai.init(api_key="dai_sk_...")

from decimalai.openai_agents import instrument
instrument()  # Scans .claude/skills/, .agents/skills/, etc. → syncs to dashboard
```

Supports 32 agent runtimes: Claude Code, Cursor, Copilot, Windsurf, Continue, and more.

### Registry Search & Install

Find community skills and install them in one call:

```python
from decimalai.skill_router import SkillRouter
router = SkillRouter(api_key="dai_sk_...")

# Search the public registry
results = router.search("code review security")

# Install a skill — a LINK into your workspace, tracking the author's updates
router.use("pdf")

# Write the files to disk, for runtimes that load SKILL.md themselves
router.export("pdf", agents=["claude-code", "cursor"])

# Take an editable copy, only if you intend to change it
router.fork("pdf")
```

### Status & Update

```python
# Check sync status between local files and platform
status = router.status()
# → {"synced": [...], "modified_locally": [...], "untracked": [...]}

# Pull upstream updates
router.update_skills()
```

### Skill Delivery at Runtime (`enable_skill_loader`)

Discovery and sync (above) get skills *into the platform*. To get them **into your agent's context at runtime**, enable the skill loader on your adapter's `instrument()`:

```python
import decimalai
decimalai.init()

from decimalai.openai_agents import instrument  # or .langchain / .anthropic / .pydantic_ai
instrument(enable_skill_loader=True)
```

With the loader on, the router adds a ranked **menu** of relevant skills to the prompt (one short row per skill: name + when to use it). A menu row alone is only an *offer* — the skill's actual content (its body) reaches the model through one of three delivery mechanisms:

- **Body injection** (opt-in) — `decimalai.init(inject_skill_body=True)` (or `DECIMALAI_INJECT_SKILL_BODY=1`) injects the top-routed skill's full body into the prompt, trimmed to a token budget. Works on adapters that route on the user query (`openai_agents`, `langchain`, `anthropic`); Pydantic AI builds its prompt in full-menu mode (no query available), so bodies arrive via `load_skill` there instead.
- **`load_skill` tool** — on adapters that own their tool loop, a `load_skill` tool registers automatically whenever the loader is enabled, so the model can fetch any offered skill's body mid-turn. On by default; kill switch: `decimalai.init(load_skill_tool=False)` or `DECIMALAI_LOAD_SKILL_TOOL=0`. It is **not** an `instrument()` parameter on these adapters.
- **Export to disk** — for runtimes that natively load skills from files (Claude Code, Cursor, ...), write them out with `router.export(...)` or `decimalai skills export` and let the runtime deliver them. Export takes no copy and needs no fork; you can export a skill you only linked.

Delivery support differs per adapter — the asymmetry is structural (`load_skill` needs a tool loop the adapter controls):

| Adapter | Menu + body injection | `load_skill` tool | Notes |
|---|---|---|---|
| `decimalai.openai_agents` | ✅ | ✅ | tool auto-registers with the loader |
| `decimalai.pydantic_ai` | ✅ menu / ❌ body injection | ✅ | full-menu mode (no query at prompt-build time) — bodies arrive via `load_skill` |
| `decimalai.langchain` | ✅ | ❌ injection-only | `enable_load_skill_tool` accepted but dormant (warns) |
| `decimalai.anthropic` | ✅ | ❌ injection-only | patches a single `messages.create()` — no loop to route a tool result back |
| `decimalai.claude_agent_sdk` | ❌ disk-only | ❌ | tracing-only adapter; Claude Code loads skills itself from `.claude/skills/` |
| generic (`@decimalai.trace`) | ❌ disk-only | ❌ | no prompt-assembly hook; use disk install |

Honest-measurement note: menu-only (loader on, no body injection, no `load_skill` tool) means the model sees that a skill *exists* but never its content. Usage from that channel counts as **offered**, not **activated** — don't expect activation stats from prompt-injection-only setups.

### Other Ways to Use Skills (No SDK)

Every published skill is reachable without installing anything:

- **Web copy-paste** — open any skill's scorecard page on [decimal.ai](https://decimal.ai/skills), copy the SKILL.md, and paste it into your repo (*live at launch*).
- **Raw URLs** — `https://decimal.ai/s/<slug>/SKILL.md` serves the raw markdown (version-pinned: `/s/<slug>@<version>/SKILL.md`); `https://decimal.ai/s/<slug>.json` serves machine-readable metadata (lift summary, benchmark models, trust/safety bands); `https://decimal.ai/llms.txt` indexes the registry for agents (*live at launch*).
- **CLI pull (no account)** — `decimalai skills pull <slug>` writes just the file to disk; no fork, no signup. (`decimalai skills install <slug>` forks + syncs if you do have a key.)
- **MCP server** — search and install skills from any MCP client (*publishing at launch*).
- **Claude Code plugin** — `/decimalai:install <slug>` from inside Claude Code (*publishing at launch*).

See the full [SDK Skills Reference](https://docs.decimal.ai/sdk/python) for all methods.

## Datasets & Training

### Pull Training Data

```python
import decimalai
decimalai.init()

# Pull the latest version to a local file
result = decimalai.pull_dataset("ds_abc123", "./training_data.jsonl")
print(f"Wrote {result['row_count']} rows")

# Pull a specific version
result = decimalai.pull_dataset("ds_abc123", "./data.jsonl", version="v2")
```

### Push to HuggingFace Hub

```python
# Push to HF Hub — instantly loadable by Axolotl, Unsloth, TRL
result = decimalai.push_to_hub("ds_abc123", "my-org/support-agent-sft")

# Now usable everywhere:
# from datasets import load_dataset
# ds = load_dataset("my-org/support-agent-sft")
```

### Load as HuggingFace Dataset (In-Memory)

```python
# Skip files — load directly into your training script
ds = decimalai.load_hf_dataset("ds_abc123")
# → Dataset({features: ['messages'], num_rows: 500})
```

### CLI

```bash
# Pull latest version
decimalai datasets pull ds_abc123 -o ./training_data.jsonl

# Pull specific version as Parquet
decimalai datasets pull ds_abc123 -o ./data.parquet --version v2

# Push to HuggingFace Hub
decimalai datasets push-to-hub ds_abc123 my-org/support-agent-sft
```

### Fine-Tuning Providers

| Provider | Models | Setup |
|----------|--------|-------|
| OpenAI | GPT-4o, GPT-4.1-mini | Dashboard or API |
| Together.AI | Llama 4, Qwen 3, DeepSeek R1, Mistral | Dashboard or API |
| Gemini | Gemini 2.5 Flash/Pro | Dashboard or API |

## Supported Frameworks

| Framework | Status | Setup |
|-----------|--------|-------|
| LangChain / LangGraph | ✅ | `init(langchain=True)` |
| OpenAI Agents SDK | ✅ | `init(openai_agents=True)` |
| Google ADK | ✅ (native) | `init(adk=True)` |
| Anthropic Claude Agent SDK | ✅ (native) | `init(claude_agent_sdk=True)` |
| LlamaIndex | ✅ | `init(llamaindex=True)` |
| CrewAI | ✅ | `init(crewai=True)` |
| AutoGen / AG2 | ✅ | `init(autogen=True)` |
| Generic (any framework) | ✅ | `@decimalai.trace()` |
| OpenTelemetry | ✅ | `init(otel=True)` |

Tracing a direct LLM SDK with no agent framework? Use the provider flags: `init(openai=True)`, `init(anthropic=True)`, or `init(google=True)`.

## Examples

See the [`examples/`](examples/) directory for runnable notebooks with **Open in Colab** badges:

| Notebook | Description | Colab |
|----------|-------------|-------|
| [Quickstart](examples/quickstart/quickstart.ipynb) | Full version-aware loop — no LLM key needed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart.ipynb) |
| [LangChain](examples/quickstart/quickstart_langchain.ipynb) | Instrument a LangChain agent | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart_langchain.ipynb) |
| [OpenAI Agents](examples/quickstart/quickstart_openai_agents.ipynb) | Instrument an OpenAI Agents app | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart_openai_agents.ipynb) |
| [Evaluations](examples/evaluations/builtin_evaluators.ipynb) | Run built-in evaluators on traces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/evaluations/builtin_evaluators.ipynb) |
| [Datasets](examples/datasets-and-training/build_sft_dataset.ipynb) | Build SFT training datasets | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/datasets-and-training/build_sft_dataset.ipynb) |
| [Pull & Push](examples/datasets-and-training/pull_and_push.py) | Pull datasets locally, push to HuggingFace Hub | — |
| [Version-Aware Loop](examples/version-aware-loop/manifest_change.ipynb) | Detect manifest changes and impact | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/version-aware-loop/manifest_change.ipynb) |

## Open standard: agentversion

The manifests this SDK captures are [`agentversion`](https://pypi.org/project/agentversion/) manifests — the open spec for agent versioning, diffing, and compatibility decisions that DecimalAI is built on. `export_manifest` hands a captured manifest to the OSS tooling, so you can diff and gate it in CI with **no platform account**:

```python
import decimalai
from decimalai.schema.manifest import extract_from_config
from agentversion.diff import diff_manifests              # pip install agentversion
from agentversion.compatibility import classify_compatibility

snap = extract_from_config(agent_name="support-agent", prompts={...}, models={...})
manifest = decimalai.export_manifest(snap)                # → an agentversion manifest dict
print(classify_compatibility(diff_manifests(last_prod, manifest)).recommended_decision)
```

You can reproduce the platform's diffs and verdicts entirely outside DecimalAI — the SDK is the convenience layer over the open standard.

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q --ignore=tests/test_langchain_compat.py
ruff check decimalai/ --select I,E,W,F --ignore E501,E402,F821,F841
git grep -nE 'decimal[-_]ai'   # must find nothing — the package is 'decimalai', no separator
```

Run these before opening a PR. CI runs the same commands, plus the LangChain compatibility matrix in `tests/test_langchain_compat.py`.

## Documentation

Full docs at [docs.decimal.ai](https://docs.decimal.ai)

## License

MIT — see [LICENSE](LICENSE) for details.
