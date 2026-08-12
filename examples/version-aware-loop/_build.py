#!/usr/bin/env python3
"""Build version-aware-loop notebook."""
import json, os

def _s(source):
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": _s(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": _s(s)}

META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}}

cells = [
    md("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/version-aware-loop/manifest_change.ipynb)\n"
       "\n"
       "# Version-Aware Loop: Manifest Changes & Impact Reports\n"
       "\n"
       "**DecimalAI's core differentiator: automatic detection of agent changes and their impact on your training data.**\n"
       "\n"
       "This notebook dives deeper than the quickstart into:\n"
       "- How manifests are auto-detected from tools, models, and prompts\n"
       "- What triggers a new manifest version\n"
       "- How the impact report classifies every trace as **keep / repair / replay / drop**\n"
       "- When to use mechanical repair vs. LLM replay\n"
       "\n"
       "> **No LLM API key required.** All examples use mock functions."),

    md("## What Is a Manifest?\n"
       "\n"
       "A **manifest** is a snapshot of your agent's configuration at a point in time:\n"
       "- **Tools**: names, schemas, descriptions\n"
       "- **Models**: provider, model name, temperature\n"
       "- **Prompts**: system prompts, templates\n"
       "- **Subagents**: names and configurations\n"
       "\n"
       "DecimalAI hashes this snapshot. When the hash changes, a new manifest version is created\n"
       "and all existing traces are classified against the new version."),

    md("## Setup"),

    code("!pip install -q decimalai"),

    code("import os\n"
         "os.environ[\"DECIMAL_API_KEY\"] = \"dai_sk_...\"  # ← Replace with your key\n"
         "\n"
         "import decimalai\n"
         "decimalai.init()"),

    md("## Scenario 1: Tool Rename\n"
       "\n"
       "The most common change. You rename a tool for clarity but the\n"
       "functionality stays the same. DecimalAI classifies affected traces\n"
       "as **repair** — they can be mechanically fixed without re-running."),

    code("# ── v1: Original agent ──\n"
         "\n"
         "@decimalai.trace(agent_name=\"demo-agent\")\n"
         "def agent_v1(query: str) -> str:\n"
         "    decimalai.log_tool_call(name=\"get_stock_price\", input={\"ticker\": \"AAPL\"}, output={\"price\": 178.52})\n"
         "    return f\"AAPL is at $178.52\"\n"
         "\n"
         "# Generate 3 traces with v1\n"
         "for q in [\"What is AAPL price?\", \"Check TSLA stock\", \"NVDA current price\"]:\n"
         "    agent_v1(q)\n"
         "\n"
         "print(\"✅ 3 traces sent with tool: get_stock_price\")"),

    code("# ── v2: Renamed tool ──\n"
         "\n"
         "@decimalai.trace(agent_name=\"demo-agent\")\n"
         "def agent_v2(query: str) -> str:\n"
         "    # Same function, better name\n"
         "    decimalai.log_tool_call(name=\"lookup_ticker\", input={\"ticker\": \"AAPL\"}, output={\"price\": 178.52})\n"
         "    return f\"AAPL is at $178.52\"\n"
         "\n"
         "agent_v2(\"What is AAPL price?\")\n"
         "\n"
         "print(\"✅ Manifest v2 auto-detected!\")\n"
         "print(\"   Changed: get_stock_price → lookup_ticker\")\n"
         "print(\"   Impact: 3 traces classified as 'repair'\")"),

    md("### What Happened\n"
       "\n"
       "```\n"
       "Manifest v1: tools=[get_stock_price]     → 3 traces\n"
       "Manifest v2: tools=[lookup_ticker]        → 1 trace\n"
       "\n"
       "Impact Report:\n"
       "  keep:   0  (no traces are unaffected — all used the renamed tool)\n"
       "  repair: 3  (tool name changed, traces can be mechanically updated)\n"
       "  replay: 0  (no schema changes requiring re-execution)\n"
       "  drop:   0  (no tools removed entirely)\n"
       "```\n"
       "\n"
       "**Repair** means DecimalAI can automatically rewrite `get_stock_price` → `lookup_ticker`\n"
       "in the trace data. No LLM cost, instant."),

    md("## Scenario 2: Tool Added\n"
       "\n"
       "Adding a new tool doesn't break existing traces — they just didn't use it.\n"
       "All existing traces are classified as **keep**."),

    code("# ── v3: Added a new tool ──\n"
         "\n"
         "@decimalai.trace(agent_name=\"demo-agent\")\n"
         "def agent_v3(query: str) -> str:\n"
         "    decimalai.log_tool_call(name=\"lookup_ticker\", input={\"ticker\": \"AAPL\"}, output={\"price\": 178.52})\n"
         "    # NEW: Also log the new tool\n"
         "    decimalai.log_tool_call(name=\"get_earnings\", input={\"ticker\": \"AAPL\"}, output={\"eps\": 6.42})\n"
         "    return f\"AAPL: $178.52, EPS: $6.42\"\n"
         "\n"
         "agent_v3(\"AAPL price and earnings\")\n"
         "\n"
         "print(\"✅ Manifest v3: added get_earnings\")\n"
         "print(\"   Impact: existing traces → keep (new tool doesn't invalidate old data)\")"),

    md("## Scenario 3: Tool Removed\n"
       "\n"
       "Removing a tool that existing traces used is the most disruptive change.\n"
       "Traces that called the removed tool are classified as **drop** — they\n"
       "reference a capability that no longer exists and can't be mechanically fixed."),

    code("# ── v4: Removed lookup_ticker, kept only get_earnings ──\n"
         "\n"
         "@decimalai.trace(agent_name=\"demo-agent\")\n"
         "def agent_v4(query: str) -> str:\n"
         "    # Only uses get_earnings now\n"
         "    decimalai.log_tool_call(name=\"get_earnings\", input={\"ticker\": \"AAPL\"}, output={\"eps\": 6.42})\n"
         "    return f\"AAPL EPS: $6.42\"\n"
         "\n"
         "agent_v4(\"AAPL earnings\")\n"
         "\n"
         "print(\"✅ Manifest v4: removed lookup_ticker\")\n"
         "print(\"   Impact:\")\n"
         "print(\"     - Traces using lookup_ticker → drop\")\n"
         "print(\"     - Traces using only get_earnings → keep\")"),

    md("## Scenario 4: Schema Change (Replay)\n"
       "\n"
       "When a tool's **schema** changes (new required parameter, different output format),\n"
       "traces can't be mechanically fixed. They need to be **replayed** — re-executed\n"
       "through the updated agent so the LLM learns the new schema."),

    code("# ── v5: Changed tool schema ──\n"
         "\n"
         "@decimalai.trace(agent_name=\"demo-agent\")\n"
         "def agent_v5(query: str) -> str:\n"
         "    # get_earnings now requires a 'period' parameter\n"
         "    decimalai.log_tool_call(\n"
         "        name=\"get_earnings\",\n"
         "        input={\"ticker\": \"AAPL\", \"period\": \"quarterly\"},  # NEW: period param\n"
         "        output={\"eps\": 1.52, \"period\": \"Q4 2025\"},  # Richer output\n"
         "    )\n"
         "    return f\"AAPL Q4 2025 EPS: $1.52\"\n"
         "\n"
         "agent_v5(\"AAPL quarterly earnings\")\n"
         "\n"
         "print(\"✅ Manifest v5: get_earnings schema changed\")\n"
         "print(\"   Impact: old traces using get_earnings → replay\")\n"
         "print(\"   (schema changed, mechanical repair isn't sufficient)\")"),

    md("## Using the SDK to Query Impact\n"
       "\n"
       "You can also use the `register_manifest` API to explicitly declare\n"
       "your agent's configuration and see the impact report programmatically."),

    code("# Explicit manifest registration\n"
         "result = decimalai.register_manifest(\n"
         "    agent_name=\"demo-agent\",\n"
         "    tools=[\n"
         "        {\"name\": \"get_earnings\", \"schema\": {\n"
         "            \"type\": \"object\",\n"
         "            \"properties\": {\n"
         "                \"ticker\": {\"type\": \"string\"},\n"
         "                \"period\": {\"type\": \"string\", \"enum\": [\"quarterly\", \"annual\"]},\n"
         "            },\n"
         "            \"required\": [\"ticker\", \"period\"],\n"
         "        }},\n"
         "    ],\n"
         "    models={\"default\": {\"provider\": \"openai\", \"model\": \"gpt-4o\"}},\n"
         "    prompts={\"system\": \"You are a financial analyst assistant.\"},\n"
         ")\n"
         "\n"
         "print(f\"Manifest ID: {result.get('manifest_id', 'N/A')}\")\n"
         "print(f\"Compatibility: {result.get('compatibility', {})}\")\n"
         "print(f\"Impact: {result.get('impact_summary', {})}\")\n"
         "print()\n"
         "print(\"📊 Open dashboard: https://app.decimal.ai/agents/demo-agent\")"),

    md("## Summary: The Four Classifications\n"
       "\n"
       "| Classification | Trigger | Fix | Cost |\n"
       "|---------------|---------|-----|------|\n"
       "| **Keep** | Trace doesn't use any changed component | None needed | Free |\n"
       "| **Repair** | Tool renamed, prompt template changed | Mechanical string replacement | Free |\n"
       "| **Replay** | Schema changed, model swapped | Re-run through updated agent | LLM cost |\n"
       "| **Drop** | Tool removed entirely | Cannot fix — exclude from dataset | Free |\n"
       "\n"
       "### Why This Matters for Training\n"
       "\n"
       "If you build a fine-tuning dataset from traces without checking compatibility:\n"
       "- **Stale tool calls** teach the model to call non-existent tools\n"
       "- **Wrong schemas** teach the model to pass outdated parameters\n"
       "- **Mixed versions** create inconsistent training signal\n"
       "\n"
       "DecimalAI catches all of this automatically.\n"
       "\n"
       "## Next Steps\n"
       "\n"
       "- 📖 [Quickstart](../quickstart/quickstart.ipynb) — Start from the basics\n"
       "- 📖 [Evaluations](../evaluations/builtin_evaluators.ipynb) — Score your traces\n"
       "- 📖 [Build Datasets](../datasets-and-training/build_sft_dataset.ipynb) — Create clean training data"),
]

nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 4}
path = os.path.join(os.path.dirname(__file__), "manifest_change.ipynb")
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"✅ {path}")
