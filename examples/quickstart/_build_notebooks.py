#!/usr/bin/env python3
"""Build quickstart notebooks programmatically."""
import json
import os

def _split_source(source):
    """Split source into Jupyter-format list: each line ends with \\n except the last."""
    lines = source.split("\n")
    result = [line + "\n" for line in lines[:-1]]
    if lines[-1]:  # Don't add empty trailing line
        result.append(lines[-1])
    return result

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(source)}

def code(source):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": _split_source(source)}

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 1: quickstart.ipynb (no LLM key needed)
# ═══════════════════════════════════════════════════════════════

qs_cells = [
    md("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart.ipynb)\n"
       "\n"
       "# DecimalAI Quickstart\n"
       "\n"
       "**Get your first traces and see the version-aware loop in 5 minutes.**\n"
       "\n"
       "This notebook walks you through DecimalAI's core workflow:\n"
       "\n"
       "1. ✅ Instrument an agent with 2 lines of code\n"
       "2. ✅ Generate traces (no LLM API key needed)\n"
       "3. ✅ Change the agent → automatic manifest versioning\n"
       "4. ✅ See the impact report (keep / repair / replay / drop)\n"
       "\n"
       "> **No LLM API key required.** This example uses mock tool functions to demonstrate the full workflow.\n"
       "> For framework-specific examples with real LLMs, see the\n"
       "> [LangChain quickstart](./quickstart_langchain.ipynb) or\n"
       "> [OpenAI Agents quickstart](./quickstart_openai_agents.ipynb)."),

    md("## Step 1 — Install & Configure"),

    code("# Install the DecimalAI SDK\n"
         "!pip install -q decimalai"),

    code("import os\n"
         "\n"
         "# Set your API key (get one at https://app.decimal.ai/settings)\n"
         "os.environ[\"DECIMAL_API_KEY\"] = \"dai_sk_...\"  # ← Replace with your key\n"
         "\n"
         "# Initialize the SDK\n"
         "import decimalai\n"
         "decimalai.init()"),

    md("## Step 2 — Define Your Agent (v1)\n"
       "\n"
       "We'll create a simple customer support agent with two tools:\n"
       "- `search_docs` — searches a knowledge base\n"
       "- `check_inventory` — checks product stock levels\n"
       "\n"
       "These are mock functions (no real API calls), but DecimalAI traces them\n"
       "the same way it would trace real tools."),

    code("# --- Agent v1: Two tools ---\n"
         "\n"
         "def search_docs(query: str) -> str:\n"
         "    \"\"\"Search the knowledge base for relevant articles.\"\"\"\n"
         "    responses = {\n"
         "        \"password\": \"To reset your password, go to Settings > Security > Reset Password.\",\n"
         "        \"return\": \"Our return policy allows returns within 30 days of purchase.\",\n"
         "        \"shipping\": \"Standard shipping takes 5-7 business days. Express: 1-2 days.\",\n"
         "        \"support\": \"Contact support at help@example.com or call 1-800-555-0123.\",\n"
         "    }\n"
         "    for keyword, response in responses.items():\n"
         "        if keyword in query.lower():\n"
         "            return response\n"
         "    return f\"Found 3 articles related to '{query}'. Please be more specific.\"\n"
         "\n"
         "\n"
         "def check_inventory(product_id: str) -> dict:\n"
         "    \"\"\"Check product stock levels.\"\"\"\n"
         "    inventory = {\n"
         "        \"SKU-1234\": {\"product_id\": \"SKU-1234\", \"name\": \"Wireless Mouse\", \"in_stock\": True, \"quantity\": 42},\n"
         "        \"SKU-5678\": {\"product_id\": \"SKU-5678\", \"name\": \"USB-C Hub\", \"in_stock\": False, \"quantity\": 0},\n"
         "    }\n"
         "    return inventory.get(product_id, {\"product_id\": product_id, \"in_stock\": True, \"quantity\": 100})\n"
         "\n"
         "\n"
         "print(\"✅ Agent v1 defined with 2 tools: search_docs, check_inventory\")"),

    md("## Step 3 — Instrument & Generate Traces\n"
       "\n"
       "The `@decimalai.trace()` decorator captures everything — inputs, outputs,\n"
       "tool calls, and timing — and sends it to your dashboard automatically."),

    code("@decimalai.trace(agent_name=\"support-agent\")\n"
         "def run_agent_v1(query: str) -> str:\n"
         "    \"\"\"Simple support agent that routes to the right tool.\"\"\"\n"
         "    if \"stock\" in query.lower() or \"inventory\" in query.lower():\n"
         "        result = check_inventory(\"SKU-1234\")\n"
         "        decimalai.log_tool_call(name=\"check_inventory\", input={\"product_id\": \"SKU-1234\"}, output=result)\n"
         "        return f\"Stock check: {result['name']} — {'In stock' if result['in_stock'] else 'Out of stock'} ({result['quantity']} units)\"\n"
         "    else:\n"
         "        result = search_docs(query)\n"
         "        decimalai.log_tool_call(name=\"search_docs\", input={\"query\": query}, output=result)\n"
         "        return f\"Here's what I found: {result}\"\n"
         "\n"
         "\n"
         "# Run 5 queries to generate traces\n"
         "queries = [\n"
         "    \"How do I reset my password?\",\n"
         "    \"Check inventory for SKU-1234\",\n"
         "    \"What is your return policy?\",\n"
         "    \"Is product SKU-5678 in stock?\",\n"
         "    \"How do I contact support?\",\n"
         "]\n"
         "\n"
         "print(\"🚀 Running 5 queries through Agent v1...\\n\")\n"
         "for q in queries:\n"
         "    answer = run_agent_v1(q)\n"
         "    print(f\"  Q: {q}\")\n"
         "    print(f\"  A: {answer}\\n\")\n"
         "\n"
         "print(\"✅ 5 traces sent to DecimalAI!\")\n"
         "print(\"📊 Open your dashboard: https://app.decimal.ai/traces\")"),

    md("## Step 4 — Update the Agent (v2)\n"
       "\n"
       "Now let's simulate a real-world agent update:\n"
       "\n"
       "1. **Rename** `check_inventory` → `lookup_stock` (clearer name)\n"
       "2. **Add** a new tool: `process_refund`\n"
       "\n"
       "This is exactly what happens when your team iterates on an agent."),

    code("# --- Agent v2: Renamed tool + new tool ---\n"
         "\n"
         "def lookup_stock(item_id: str) -> dict:\n"
         "    \"\"\"Look up current stock for an item. (Renamed from check_inventory)\"\"\"\n"
         "    inventory = {\n"
         "        \"SKU-1234\": {\"item_id\": \"SKU-1234\", \"name\": \"Wireless Mouse\", \"in_stock\": True, \"quantity\": 42},\n"
         "        \"SKU-5678\": {\"item_id\": \"SKU-5678\", \"name\": \"USB-C Hub\", \"in_stock\": False, \"quantity\": 0},\n"
         "    }\n"
         "    return inventory.get(item_id, {\"item_id\": item_id, \"in_stock\": True, \"quantity\": 100})\n"
         "\n"
         "\n"
         "def process_refund(order_id: str, reason: str) -> dict:\n"
         "    \"\"\"Process a refund for an order. (NEW in v2)\"\"\"\n"
         "    return {\"order_id\": order_id, \"status\": \"refunded\", \"reason\": reason, \"amount\": 29.99}\n"
         "\n"
         "\n"
         "print(\"✅ Agent v2 defined:\")\n"
         "print(\"   - search_docs (unchanged)\")\n"
         "print(\"   - lookup_stock (renamed from check_inventory)\")\n"
         "print(\"   - process_refund (NEW)\")"),

    code("@decimalai.trace(agent_name=\"support-agent\")\n"
         "def run_agent_v2(query: str) -> str:\n"
         "    \"\"\"Updated support agent with renamed + new tools.\"\"\"\n"
         "    if \"stock\" in query.lower() or \"inventory\" in query.lower():\n"
         "        result = lookup_stock(\"SKU-1234\")\n"
         "        decimalai.log_tool_call(name=\"lookup_stock\", input={\"item_id\": \"SKU-1234\"}, output=result)\n"
         "        return f\"Stock check: {result['name']} — {'In stock' if result['in_stock'] else 'Out of stock'}\"\n"
         "    elif \"refund\" in query.lower() or \"return\" in query.lower():\n"
         "        result = process_refund(\"ORD-001\", \"damaged item\")\n"
         "        decimalai.log_tool_call(name=\"process_refund\", input={\"order_id\": \"ORD-001\"}, output=result)\n"
         "        return f\"Refund processed: ${result['amount']} for order {result['order_id']}\"\n"
         "    else:\n"
         "        result = search_docs(query)\n"
         "        decimalai.log_tool_call(name=\"search_docs\", input={\"query\": query}, output=result)\n"
         "        return f\"Here's what I found: {result}\"\n"
         "\n"
         "\n"
         "# Run the updated agent — this triggers a new manifest version\n"
         "print(\"🚀 Running Agent v2 (triggers manifest change)...\\n\")\n"
         "answer = run_agent_v2(\"I need to return a damaged item\")\n"
         "print(f\"  Q: I need to return a damaged item\")\n"
         "print(f\"  A: {answer}\\n\")\n"
         "\n"
         "print(\"✅ Manifest v2 auto-detected! DecimalAI saw the tool changes:\")\n"
         "print(\"   - check_inventory → renamed to lookup_stock\")\n"
         "print(\"   - process_refund → added\")"),

    md("## Step 5 — See the Impact Report\n"
       "\n"
       "Open your **[DecimalAI Dashboard](https://app.decimal.ai)** → **Agents → support-agent**.\n"
       "\n"
       "You'll see the **Impact Report**:\n"
       "\n"
       "| Classification | Count | Why |\n"
       "|---------------|-------|-----|\n"
       "| **Keep** | 3 | Traces that only used `search_docs` — still compatible |\n"
       "| **Repair** | 2 | Traces that used `check_inventory` — can be mechanically renamed |\n"
       "| **Replay** | 0 | No traces need re-running |\n"
       "| **Drop** | 0 | No tools were removed |\n"
       "\n"
       "### Why This Matters\n"
       "\n"
       "If you were fine-tuning on those 5 traces, 2 of them reference a tool\n"
       "(`check_inventory`) that no longer exists. Training on stale data would teach\n"
       "your model to call a non-existent tool. **DecimalAI catches this automatically.**"),

    md("## What Just Happened\n"
       "\n"
       "```\n"
       "Agent v1 runs → 5 traces recorded\n"
       "    ↓\n"
       "Agent changes (tool renamed, tool added)\n"
       "    ↓\n"
       "DecimalAI auto-detects v2 (manifest versioning)\n"
       "    ↓\n"
       "Impact Report: 5 traces classified (keep / repair / replay / drop)\n"
       "    ↓\n"
       "One-click: Repair stale traces + Build clean dataset\n"
       "```\n"
       "\n"
       "This happens **automatically, every time your agent changes.**\n"
       "\n"
       "## Next Steps\n"
       "\n"
       "- 📖 [LangChain Quickstart](./quickstart_langchain.ipynb) — Instrument a real LangChain agent\n"
       "- 📖 [OpenAI Agents Quickstart](./quickstart_openai_agents.ipynb) — Instrument an OpenAI Agents app\n"
       "- 🔗 [Concepts](https://docs.decimal.ai/concepts) — Understand manifests, traces, and datasets\n"
       "- 📊 [Dashboard](https://app.decimal.ai) — Explore your traces"),
]

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 2: quickstart_langchain.ipynb
# ═══════════════════════════════════════════════════════════════

lc_cells = [
    md("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart_langchain.ipynb)\n"
       "\n"
       "# DecimalAI + LangChain Quickstart\n"
       "\n"
       "**Instrument a LangChain agent with 2 lines of code.**\n"
       "\n"
       "This notebook shows how to add DecimalAI tracing to any LangChain / LangGraph\n"
       "application. Every LLM call, tool invocation, and chain step is captured automatically.\n"
       "\n"
       "**Prerequisites:** An OpenAI API key (or any LangChain-supported LLM)."),

    md("## Step 1 — Install & Configure"),

    code("# Install dependencies (the [langchain] extra brings the adapter deps;\n"
         "# `langchain` provides create_agent, `langchain-openai` the LLM binding)\n"
         "!pip install -q \"decimalai[langchain]\" langchain langchain-openai"),

    code("import os\n"
         "\n"
         "# Set your API keys\n"
         "os.environ[\"DECIMAL_API_KEY\"] = \"dai_sk_...\"    # ← Get at https://app.decimal.ai/settings\n"
         "os.environ[\"OPENAI_API_KEY\"] = \"sk-...\"         # ← Your OpenAI key\n"
         "\n"
         "# Initialize DecimalAI with LangChain auto-tracing\n"
         "import decimalai\n"
         "decimalai.init(langchain=True)\n"
         "\n"
         "# That's it! All LangChain calls are now traced automatically."),

    md("## Step 2 — Build a Simple LangChain Agent\n"
       "\n"
       "We'll create a tool-calling agent with a couple of tools using langchain 1.x's\n"
       "`create_agent` (a LangGraph graph under the hood). DecimalAI captures\n"
       "everything — you don't need to add any extra callbacks or wrappers."),

    code("from langchain_openai import ChatOpenAI\n"
         "from langchain.agents import create_agent\n"
         "from langchain_core.tools import tool\n"
         "\n"
         "\n"
         "# Define tools\n"
         "@tool\n"
         "def search_docs(query: str) -> str:\n"
         "    \"\"\"Search the knowledge base. Input: search query.\"\"\"\n"
         "    return f\"Found 3 results for '{query}': [Article 1, Article 2, Article 3]\"\n"
         "\n"
         "@tool\n"
         "def check_order(order_id: str) -> str:\n"
         "    \"\"\"Look up an order status. Input: order ID.\"\"\"\n"
         "    return f\"Order {order_id}: Shipped on April 25, arriving April 29.\"\n"
         "\n"
         "# Create agent\n"
         "llm = ChatOpenAI(model=\"gpt-4o-mini\", temperature=0)\n"
         "\n"
         "agent = create_agent(\n"
         "    llm,\n"
         "    [search_docs, check_order],\n"
         "    system_prompt=\"You are a helpful customer support assistant.\",\n"
         ")\n"
         "\n"
         "print(\"✅ LangChain agent ready with 2 tools\")"),

    md("## Step 3 — Run Queries (Traces Are Auto-Captured)"),

    code("# Every invocation is automatically traced by DecimalAI\n"
         "queries = [\n"
         "    \"How do I reset my password?\",\n"
         "    \"Where is my order ORD-12345?\",\n"
         "    \"What is your return policy?\",\n"
         "]\n"
         "\n"
         "for q in queries:\n"
         "    print(f\"\\n{'='*50}\")\n"
         "    print(f\"Q: {q}\")\n"
         "    result = agent.invoke({\"messages\": [{\"role\": \"user\", \"content\": q}]})\n"
         "    print(f\"A: {result['messages'][-1].content}\")\n"
         "\n"
         "print(\"\\n✅ 3 traces auto-captured and sent to DecimalAI!\")\n"
         "print(\"📊 Open your dashboard: https://app.decimal.ai/traces\")"),

    md("## Step 4 — View in Dashboard\n"
       "\n"
       "Open **[app.decimal.ai/traces](https://app.decimal.ai/traces)**. For each trace you'll see:\n"
       "\n"
       "- The full conversation flow (input → LLM → tool calls → output)\n"
       "- Token usage and latency per LLM call\n"
       "- Tool call inputs and outputs\n"
       "- The auto-detected manifest (tools + model)\n"
       "\n"
       "## Next Steps\n"
       "\n"
       "- 📖 [Main Quickstart](./quickstart.ipynb) — See the version-aware manifest loop (no LLM key needed)\n"
       "- 📖 [Evaluations Guide](https://docs.decimal.ai/guides/evaluations) — Score your traces\n"
       "- 📖 [Training Pipeline](https://docs.decimal.ai/tutorials/training-pipeline) — Build datasets from traces"),
]

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 3: quickstart_openai_agents.ipynb
# ═══════════════════════════════════════════════════════════════

oai_cells = [
    md("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart_openai_agents.ipynb)\n"
       "\n"
       "# DecimalAI + OpenAI Agents SDK Quickstart\n"
       "\n"
       "**Instrument an OpenAI Agents application with 2 lines of code.**\n"
       "\n"
       "This notebook shows how to add DecimalAI tracing to the OpenAI Agents SDK.\n"
       "Agent runs, tool calls, handoffs, and guardrails are captured automatically.\n"
       "\n"
       "**Prerequisites:** An OpenAI API key."),

    md("## Step 1 — Install & Configure"),

    code("# Install dependencies (the [openai-agents] extra brings the OpenAI Agents SDK)\n"
         "!pip install -q \"decimalai[openai-agents]\""),

    code("import os\n"
         "\n"
         "# Set your API keys\n"
         "os.environ[\"DECIMAL_API_KEY\"] = \"dai_sk_...\"    # ← Get at https://app.decimal.ai/settings\n"
         "os.environ[\"OPENAI_API_KEY\"] = \"sk-...\"         # ← Your OpenAI key\n"
         "\n"
         "# Initialize DecimalAI with OpenAI Agents auto-tracing\n"
         "import decimalai\n"
         "decimalai.init(openai_agents=True)\n"
         "\n"
         "# That's it! All Agent runs are now traced automatically."),

    md("## Step 2 — Define an Agent with Tools"),

    code("from agents import Agent, Runner, function_tool\n"
         "\n"
         "\n"
         "@function_tool\n"
         "def search_docs(query: str) -> str:\n"
         "    \"\"\"Search the knowledge base for relevant articles.\"\"\"\n"
         "    return f\"Found 3 results for '{query}': [Article 1, Article 2, Article 3]\"\n"
         "\n"
         "\n"
         "@function_tool\n"
         "def check_order(order_id: str) -> str:\n"
         "    \"\"\"Look up an order status by order ID.\"\"\"\n"
         "    return f\"Order {order_id}: Shipped on April 25, arriving April 29.\"\n"
         "\n"
         "\n"
         "agent = Agent(\n"
         "    name=\"support-agent\",\n"
         "    instructions=\"You are a helpful customer support assistant. Use the tools to answer questions.\",\n"
         "    tools=[search_docs, check_order],\n"
         ")\n"
         "\n"
         "print(\"✅ OpenAI Agent ready with 2 tools\")"),

    md("## Step 3 — Run the Agent (Traces Are Auto-Captured)"),

    code("import asyncio\n"
         "\n"
         "async def run_queries():\n"
         "    queries = [\n"
         "        \"How do I reset my password?\",\n"
         "        \"Where is my order ORD-12345?\",\n"
         "        \"What is your return policy?\",\n"
         "    ]\n"
         "    for q in queries:\n"
         "        print(f\"\\nQ: {q}\")\n"
         "        result = await Runner.run(agent, q)\n"
         "        print(f\"A: {result.final_output}\")\n"
         "\n"
         "    print(\"\\n✅ 3 traces auto-captured and sent to DecimalAI!\")\n"
         "    print(\"📊 Open your dashboard: https://app.decimal.ai/traces\")\n"
         "\n"
         "await run_queries()"),

    md("## What Gets Captured\n"
       "\n"
       "For each Agent run, DecimalAI records:\n"
       "- Agent name, instructions, and model\n"
       "- Every LLM call with token usage\n"
       "- Tool call inputs and outputs\n"
       "- Agent handoffs (if using multi-agent)\n"
       "- Guardrail evaluations\n"
       "- The full manifest (tools + model + instructions)\n"
       "\n"
       "## Next Steps\n"
       "\n"
       "- 📖 [Main Quickstart](./quickstart.ipynb) — See the version-aware manifest loop\n"
       "- 📖 [Concepts](https://docs.decimal.ai/concepts) — Understand manifests, traces, and datasets\n"
       "- 📖 [Dashboard](https://app.decimal.ai) — Explore your traces"),
]


# ═══════════════════════════════════════════════════════════════
# Write notebooks
# ═══════════════════════════════════════════════════════════════

def write_notebook(cells, filename):
    nb = {"cells": cells, "metadata": METADATA, "nbformat": 4, "nbformat_minor": 4}
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"✅ Written {path}")

write_notebook(qs_cells, "quickstart.ipynb")
write_notebook(lc_cells, "quickstart_langchain.ipynb")
write_notebook(oai_cells, "quickstart_openai_agents.ipynb")
print("\nDone! All 3 notebooks created.")
