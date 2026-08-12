#!/usr/bin/env python3
"""Build datasets notebook."""
import json, os

def _s(source):
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": _s(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": _s(s)}

META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}}

cells = [
    md("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/datasets-and-training/build_sft_dataset.ipynb)\n"
       "\n"
       "# Building SFT Datasets from Traces\n"
       "\n"
       "**Turn your agent's best production outputs into fine-tuning training data.**\n"
       "\n"
       "This notebook shows the full trace → dataset pipeline:\n"
       "1. Filter traces by eval verdict (only passing traces)\n"
       "2. Filter by manifest (only current agent version)\n"
       "3. Convert multi-turn conversations to SFT format\n"
       "4. Export for OpenAI fine-tuning or other providers\n"
       "\n"
       "> This notebook demonstrates the API calls and data formats.\n"
       "> The actual dataset building is done through the DecimalAI dashboard."),

    md("## Setup"),

    code("!pip install -q decimalai"),

    code("import os\n"
         "os.environ[\"DECIMAL_API_KEY\"] = \"dai_sk_...\"  # ← Your key\n"
         "\n"
         "import decimalai\n"
         "decimalai.init()"),

    md("## The Training Data Problem\n"
       "\n"
       "Fine-tuning requires clean, consistent training data. But agent traces have problems:\n"
       "\n"
       "| Problem | Example | Impact |\n"
       "|---------|---------|--------|\n"
       "| **Stale tools** | Trace calls `check_inventory` but agent now uses `lookup_stock` | Model learns to call non-existent tool |\n"
       "| **Low quality** | Trace has hallucinated or wrong answer | Model learns bad behavior |\n"
       "| **Mixed versions** | Some traces from v1, some from v3 | Inconsistent training signal |\n"
       "\n"
       "DecimalAI solves all three with manifest filtering + eval scoring."),

    md("## Step 1: Understand the SFT Format\n"
       "\n"
       "OpenAI's fine-tuning expects JSONL with a `messages` array.\n"
       "DecimalAI converts multi-turn agent traces into this format automatically."),

    code("import json\n"
         "\n"
         "# This is what DecimalAI generates from a trace:\n"
         "sft_example = {\n"
         "    \"messages\": [\n"
         "        {\n"
         "            \"role\": \"system\",\n"
         "            \"content\": \"You are a customer support assistant. Use the available tools to help users.\"\n"
         "        },\n"
         "        {\n"
         "            \"role\": \"user\",\n"
         "            \"content\": \"What is the status of my order ORD-12345?\"\n"
         "        },\n"
         "        {\n"
         "            \"role\": \"assistant\",\n"
         "            \"content\": None,\n"
         "            \"tool_calls\": [{\n"
         "                \"id\": \"call_001\",\n"
         "                \"type\": \"function\",\n"
         "                \"function\": {\n"
         "                    \"name\": \"check_order\",\n"
         "                    \"arguments\": json.dumps({\"order_id\": \"ORD-12345\"})\n"
         "                }\n"
         "            }]\n"
         "        },\n"
         "        {\n"
         "            \"role\": \"tool\",\n"
         "            \"tool_call_id\": \"call_001\",\n"
         "            \"content\": json.dumps({\"status\": \"shipped\", \"eta\": \"April 29\"})\n"
         "        },\n"
         "        {\n"
         "            \"role\": \"assistant\",\n"
         "            \"content\": \"Your order ORD-12345 has been shipped and is expected to arrive on April 29.\"\n"
         "        }\n"
         "    ]\n"
         "}\n"
         "\n"
         "print(\"Example SFT training example:\")\n"
         "print(json.dumps(sft_example, indent=2))"),

    md("## Step 2: Build via Dashboard\n"
       "\n"
       "In the DecimalAI dashboard:\n"
       "\n"
       "1. Navigate to **Datasets** in the sidebar\n"
       "2. Click **\"Build Dataset\"**\n"
       "3. Select your agent (e.g., `support-agent`)\n"
       "4. **Filter by manifest**: Use the latest version (ensures current config)\n"
       "5. **Filter by eval verdict**: Select only `pass` verdicts\n"
       "6. **Choose format**: SFT (supervised fine-tuning)\n"
       "7. Click **Build**\n"
       "\n"
       "DecimalAI handles the conversion automatically:\n"
       "- Multi-turn conversations → chat completion format\n"
       "- Tool calls and results are preserved\n"
       "- System prompts are extracted from the manifest"),

    md("## Step 3: Export & Fine-Tune\n"
       "\n"
       "Once built, you can export the dataset and use it for fine-tuning."),

    code("# The export API (called from dashboard or SDK)\n"
         "# dataset = decimalai.export_dataset(\n"
         "#     dataset_id=\"ds-abc123\",\n"
         "#     format=\"openai_jsonl\",\n"
         "# )\n"
         "#\n"
         "# # This gives you a JSONL file ready for:\n"
         "# # openai api fine_tuning.jobs.create \\\n"
         "# #   --training-file dataset.jsonl \\\n"
         "# #   --model gpt-4o-mini\n"
         "\n"
         "print(\"ℹ️  Export your dataset from the dashboard or use the SDK.\")"),

    md("## The Continuous Improvement Loop\n"
       "\n"
       "```\n"
       "Agent runs in production\n"
       "    ↓\n"
       "Traces captured + auto-evaluated\n"
       "    ↓\n"
       "Agent updated → new manifest version\n"
       "    ↓\n"
       "Impact report: keep / repair / replay / drop\n"
       "    ↓\n"
       "Repair stale traces → build clean dataset\n"
       "    ↓\n"
       "Fine-tune → deploy → repeat\n"
       "```\n"
       "\n"
       "Each iteration produces a better model trained on cleaner, version-consistent data.\n"
       "\n"
       "## Next Steps\n"
       "\n"
       "- 📖 [Quickstart](../quickstart/quickstart.ipynb) — Get your first traces\n"
       "- 📖 [Manifest Changes](../version-aware-loop/manifest_change.ipynb) — Version-aware loop\n"
       "- 📖 [Evaluations](../evaluations/builtin_evaluators.ipynb) — Score your traces\n"
       "- 📖 [Training Pipeline](https://docs.decimal.ai/tutorials/training-pipeline) — Full end-to-end tutorial"),
]

nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 4}
path = os.path.join(os.path.dirname(__file__), "build_sft_dataset.ipynb")
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"✅ {path}")
