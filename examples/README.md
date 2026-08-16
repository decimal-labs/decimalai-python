# DecimalAI Examples & Cookbook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart.ipynb)

Runnable examples demonstrating how to use [DecimalAI](https://decimal.ai) — the manifest-aware platform for agent change management.

## Start here — no account, no install

| Notebook | Description | Colab |
|----------|-------------|-------|
| [**Support agent from a skill**](support-agent/support_agent.ipynb) | Pull one skill into a LangChain agent and watch it stop announcing a deletion it never performed — with the controls that show it was the skill's content and not just extra prompt text. **Anonymous pull; one free Google AI Studio key to run the model cells.** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/support-agent/support_agent.ipynb) |
| [**Measure a skill**](measure-a-skill/measure_a_skill.ipynb) | Audit a registry benchmark end to end — the claim, the per-case transcripts, the safety scan, and what we refuse to show you. **Zero credentials, runs in about a second.** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/measure-a-skill/measure_a_skill.ipynb) |
| [**Reproduce the benchmark**](measure-a-skill/reproduce_the_benchmark.ipynb) | Re-run that same ablation on **your** model against a prompt you write, blind-judged, and see how far your number lands from ours. **One free Google AI Studio key, no framework.** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/measure-a-skill/reproduce_the_benchmark.ipynb) |
| [**Route and trace**](measure-a-skill/route_and_trace.ipynb) | Splice three registry skills into one prompt, watch the token bill multiply, then route instead — and record which skill actually changed an answer. **Model key first; the DecimalAI key is asked for at cell 2.1, after the numbers.** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/measure-a-skill/route_and_trace.ipynb) |

Every other notebook below calls `decimalai.init()` in its second code cell and needs a
`DECIMAL_API_KEY` to get past it. Start with the four above if you do not have one yet —
each runs to the end with no key at all, skipping only the cells that need one.

## Quickstart

| Notebook | Description | Colab |
|----------|-------------|-------|
| [**Quickstart**](quickstart/quickstart.ipynb) | Your first traces + the version-aware loop in 5 minutes. **No LLM key needed.** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart.ipynb) |
| [LangChain Quickstart](quickstart/quickstart_langchain.ipynb) | Instrument a LangChain ReAct agent with 2 lines of code | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart_langchain.ipynb) |
| [OpenAI Agents Quickstart](quickstart/quickstart_openai_agents.ipynb) | Instrument an OpenAI Agents SDK application | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart_openai_agents.ipynb) |

> **New to DecimalAI?** Start with the [Quickstart notebook](quickstart/quickstart.ipynb) — it demonstrates the full version-aware loop (trace → manifest change → impact report) with zero external dependencies.

## Version-Aware Loop (What Makes DecimalAI Different)

| Notebook | Description | Colab |
|----------|-------------|-------|
| [**Manifest Changes**](version-aware-loop/manifest_change.ipynb) | Deep-dive: tool renames, additions, removals, schema changes → impact reports | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/version-aware-loop/manifest_change.ipynb) |

## Evaluations

| Notebook | Description | Colab |
|----------|-------------|-------|
| [**Built-in Evaluators**](evaluations/builtin_evaluators.ipynb) | Deterministic checks, custom `@eval` decorator, LLM judges, batch eval | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/evaluations/builtin_evaluators.ipynb) |

## Datasets & Training

| Notebook / Script | Description | Colab |
|----------|-------------|-------|
| [**Build SFT Dataset**](datasets-and-training/build_sft_dataset.ipynb) | Convert agent traces to OpenAI fine-tuning format, continuous improvement loop | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/datasets-and-training/build_sft_dataset.ipynb) |
| [**Pull & Push**](datasets-and-training/pull_and_push.py) | Pull datasets locally, push to HuggingFace Hub for Axolotl/Unsloth/TRL | — |

## What You'll Learn

The notebooks are organized as a learning path:

```
1. Quickstart          → Get traces in your dashboard
2. Version-Aware Loop  → Understand manifest changes and impact reports
3. Evaluations         → Score your traces (deterministic + LLM)
4. Datasets            → Build clean training data from scored, version-filtered traces
5. Pull & Push         → Download datasets, push to HF Hub for open-source training
```

## Reference Application

| Project | Description |
|---------|-------------|
| [**SupportBot**](reference-app/) | Complete end-to-end sample app with 4 runnable scenarios: generate traces → update agent → evaluate → build dataset. `make demo` runs everything. |

## Skills

Skills (auto-discovery of `SKILL.md` files + registry search/install) are shipped
today — see [**Support agent from a skill**](support-agent/support_agent.ipynb) above for the
shortest end-to-end path (pull a skill, wire it into LangChain, watch it change an answer),
the **Skills** section of the [main README](../README.md#skills), the `decimalai skills` CLI,
and [docs.decimal.ai/sdk/python](https://docs.decimal.ai/sdk/python).

## Coming Soon

- **Framework Guides** — LlamaIndex and CrewAI integration notebooks

## Requirements

```bash
pip install decimalai          # Core SDK
pip install 'decimalai[evals]' # With LLM evaluators
```

Get your API key at [app.decimal.ai/settings](https://app.decimal.ai/settings).

## Documentation

- [Full Documentation](https://docs.decimal.ai)
- [SDK Reference](https://docs.decimal.ai/sdk)
- [API Reference](https://docs.decimal.ai/api-reference)
