#!/usr/bin/env python3
"""Build the three quickstart notebooks programmatically.

    python _build_notebooks.py

overwrites quickstart.ipynb, quickstart_langchain.ipynb and
quickstart_openai_agents.ipynb in this directory. Those three files are BUILD
OUTPUT: edit THIS file and re-run it, never the .ipynb. A hand-edit to a
notebook survives exactly until the next person runs this script.

Cell sources are raw triple-quoted strings, so a backslash in the notebook
(``print("...\\n")``) is written here the way it appears there, and the block
can be diffed against the generated cell line for line.
"""
import json
import os

def _split_source(source):
    """Split source into Jupyter-format list: each line ends with \\n except the last."""
    lines = source.split("\n")
    result = [line + "\n" for line in lines[:-1]]
    if lines[-1]:  # Don't add empty trailing line
        result.append(lines[-1])
    return result

def _clean(source):
    """Drop the newlines that open and close a triple-quoted block."""
    return source.strip("\n")

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": _split_source(_clean(source))}

def code(source):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
            "source": _split_source(_clean(source))}

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

# ═══════════════════════════════════════════════════════════════
# THE KEYLESS START — shared by all three notebooks
# ═══════════════════════════════════════════════════════════════
#
# Every notebook here used to open its second code cell with
#
#     os.environ["DECIMAL_API_KEY"] = "dai_sk_..."
#     decimalai.init()
#
# which hard-stops a reader who hits "Runtime → Run all" without an account:
# init() verifies the key by default, the placeholder is not a key, and the
# notebook ends in a traceback at cell two. This block resolves a key from the
# environment, then Colab secrets, then (opt-in) a hidden prompt, treats the
# placeholder literal as "no key", and falls back to `init(enabled=False)` with
# a labelled OFFLINE banner. One template — do not fork it per notebook.

_KEYLESS_HEAD = r'''
import os

import decimalai

# A key is OPTIONAL here. This cell looks for one, and if it doesn't find a real
# one it initializes the SDK in offline mode instead of raising: `enabled=False`
'''

_KEYLESS_BODY = r'''
#
# To send traces to your own dashboard, get a key at
# https://app.decimal.ai/settings (Settings -> General -> API keys) and either:
#   * add it as a Colab secret named DECIMAL_API_KEY (the key icon, left sidebar),
#   * export DECIMAL_API_KEY before launching Jupyter, or
#   * flip ASK_FOR_KEY to True below and paste it into the hidden prompt.
ASK_FOR_KEY = False

# The literal this notebook used to assign to DECIMAL_API_KEY. Treated as "no key"
# rather than passed through, so a reader who pastes the snippet from the docs and
# forgets to edit it lands in offline mode instead of on a 401.
PLACEHOLDER = "dai_sk_..."


def find_key(*names) -> str:
    """First real key among env vars, Colab secrets, and (opt-in) a prompt."""
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value and value != PLACEHOLDER:
            return value

    try:
        from google.colab import userdata  # only exists inside Colab
    except ImportError:
        pass
    else:
        for name in names:
            try:
                value = (userdata.get(name) or "").strip()
            except Exception:
                continue  # secret not set, or notebook denied access to it
            if value and value != PLACEHOLDER:
                return value

    if ASK_FOR_KEY:
        import getpass
        # getpass, never input(): a key typed into a cell is a key in the
        # browser history and in the .ipynb you later share.
        return getpass.getpass(f"{names[0]} (input is hidden): ").strip()

    return ""


# DECIMALAI_API_KEY is the alias the CLI also accepts; init() reads both.
API_KEY = find_key("DECIMAL_API_KEY", "DECIMALAI_API_KEY")
ONLINE = False

if API_KEY:
    try:
        decimalai.init(api_key=API_KEY__INIT_KWARGS__)
        ONLINE = True
    except Exception as exc:
        # init(verify=True) raises on 401/403 or an unreachable backend. An
        # expired key is not a reason to end the notebook in a traceback.
        print(f"!  that key was rejected: {type(exc).__name__}: {exc}")
        print("!  falling back to offline mode — every cell below still runs.\n")

if not ONLINE:
    decimalai.init(enabled=False)
'''


def keyless_start(offline_effect, banner, init_kwargs="", llm_key_block=""):
    """The 'use a key if there is one, otherwise run offline' cell.

    offline_effect: comment lines finishing the sentence started in _KEYLESS_HEAD,
        i.e. what `enabled=False` costs THIS notebook.
    banner:         the print() that tells the reader which mode they are in.
    init_kwargs:    extra kwargs for the online init(), e.g. ", langchain=True".
    llm_key_block:  for the framework notebooks, the OpenAI-key half.
    """
    # The head, the notebook-specific sentence ending and the body are one
    # continuous comment-then-code run, so they butt up against each other; the
    # blocks after them are separated by a blank line.
    text = "\n".join([
        _clean(_KEYLESS_HEAD),
        _clean(offline_effect),
        _clean(_KEYLESS_BODY).replace("__INIT_KWARGS__", init_kwargs),
    ])
    if llm_key_block:
        text += "\n\n" + _clean(llm_key_block)
    return text + "\n\n" + _clean(banner)


# The OpenAI half of the framework notebooks' config cell. The DecimalAI key is
# optional; this one is not — Step 2 onward drives a real LLM — so it resolves the
# key the same three ways and sets a flag the later cells check instead of raising.
_OPENAI_KEY_BLOCK = r'''

# The LLM key is a different question. A DecimalAI key only decides whether runs
# are recorded; without an OpenAI key there is no agent run to record at all. Same
# three places, and exported to the environment because that is where the LLM
# client looks for it.
OPENAI_KEY = find_key("OPENAI_API_KEY")
if OPENAI_KEY.endswith("..."):
    OPENAI_KEY = ""  # the placeholder this cell used to assign, not a key
if OPENAI_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_KEY
LLM_READY = bool(OPENAI_KEY)
'''

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 1: quickstart.ipynb (no LLM key needed, and no DecimalAI key either)
# ═══════════════════════════════════════════════════════════════

qs_cells = [
    md(r'''
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart.ipynb)

# DecimalAI Quickstart

**Get your first traces and see the version-aware loop in 5 minutes.**

This notebook walks you through DecimalAI's core workflow:

1. Instrument an agent with 2 lines of code
2. Generate traces (no LLM API key needed)
3. Change the agent -> automatic manifest versioning
4. See exactly what changed between versions, and which traces it invalidates

> **Runtime -> Run all works with no account and no keys.** The tool functions are
> mocks, and if no DecimalAI key is present the SDK runs in offline mode: it still
> records traces and still builds a manifest for each version of the agent, it just
> doesn't send anything. Steps 1-5 all run that way. The only thing a key adds is
> the hosted dashboard — every cell that needs one says so in its first line.
>
> For framework-specific examples with real LLMs, see the
> [LangChain quickstart](./quickstart_langchain.ipynb) or
> [OpenAI Agents quickstart](./quickstart_openai_agents.ipynb).
'''),

    md(r'''
## Step 1 — Install & Configure
'''),

    code(r'''
# Install the DecimalAI SDK
!pip install -q decimalai
'''),

    code(keyless_start(
        offline_effect=r'''
# is a real kill switch (no client, no network call), but tracing, tool
# registration and manifest versioning all still run locally — which is the whole
# of Steps 2-5.
''',
        banner=r'''
print("ONLINE  — traces will be sent to https://app.decimal.ai/traces"
      if ONLINE else
      "OFFLINE — running without a usable DecimalAI key. Traces and manifests\n"
      "          are built locally and sent nowhere. Steps 2-5 run in full;\n"
      "          only the dashboard links at the end need an account.")
''')),

    md(r'''
## Step 2 — Define Your Agent (v1)

We'll create a simple customer support agent with two tools:
- `search_docs` — searches a knowledge base
- `check_inventory` — checks product stock levels

These are mock functions (no real API calls), but DecimalAI traces them
the same way it would trace real tools.

`init()` and `@decimalai.trace()` are the two lines that record a run.
`@decimalai.tool` is a third, optional one, and it is what makes versioning work:
it does not change what the function does — it registers the function's name and
argument schema so that **every** trace this agent produces declares the same tool
set, whether or not a given run happened to call every tool. Without it, a manifest
can only describe the tools that fired in that one trace, so an agent that routes
each question to one of two tools mints two alternating manifests instead of one
version.
'''),

    code(r'''
# --- Agent v1: Two tools ---

@decimalai.tool
def search_docs(query: str) -> str:
    """Search the knowledge base for relevant articles."""
    responses = {
        "password": "To reset your password, go to Settings > Security > Reset Password.",
        "return": "Our return policy allows returns within 30 days of purchase.",
        "shipping": "Standard shipping takes 5-7 business days. Express: 1-2 days.",
        "support": "Contact support at help@example.com or call 1-800-555-0123.",
    }
    for keyword, response in responses.items():
        if keyword in query.lower():
            return response
    return f"Found 3 articles related to '{query}'. Please be more specific."


@decimalai.tool
def check_inventory(product_id: str) -> dict:
    """Check product stock levels."""
    inventory = {
        "SKU-1234": {"product_id": "SKU-1234", "name": "Wireless Mouse", "in_stock": True, "quantity": 42},
        "SKU-5678": {"product_id": "SKU-5678", "name": "USB-C Hub", "in_stock": False, "quantity": 0},
    }
    return inventory.get(product_id, {"product_id": product_id, "in_stock": True, "quantity": 100})


from decimalai.decorators import get_registered_tools

print("Agent v1 declares:", ", ".join(sorted(t["name"] for t in get_registered_tools())))
'''),

    md(r'''
## Step 3 — Instrument & Generate Traces

The `@decimalai.trace()` decorator captures everything — inputs, outputs,
tool calls, and timing. With a key it ships that to your dashboard; without one
it still builds the trace and the manifest, and drops them at the network edge.

Each query below routes to **one** tool, so the traces differ from each other —
that is the point, and it is what makes the impact analysis in Step 5 non-trivial.
The *manifest* is the same for all six, because the tool registry, not the
individual run, is what declares the agent's surface.
'''),

    code(r'''
from decimalai.schema.manifest import extract_from_config
from decimalai.skills import discover_skills

# The six questions. Both versions of the agent answer the same six, so the only
# thing that differs between v1 and v2 is the agent — not the workload.
QUERIES = [
    "How do I reset my password?",
    "Check inventory for SKU-1234",
    "What is your return policy?",
    "Is product SKU-5678 in stock?",
    "How do I contact support?",
    "I need a refund for a damaged item",
]

# Which tool each query actually reached, kept so Step 5 can count the traces
# that a version change strands. The backend keeps this per trace; offline we
# have to keep it ourselves.
TOOL_USED = {"v1": {}, "v2": {}}


@decimalai.trace(agent_name="support-agent")
def run_agent_v1(query: str) -> str:
    """Simple support agent that routes to the right tool."""
    if "stock" in query.lower() or "inventory" in query.lower():
        result = check_inventory("SKU-1234")
        decimalai.log_tool_call(name="check_inventory", input={"product_id": "SKU-1234"}, output=result)
        TOOL_USED["v1"][query] = "check_inventory"
        return f"Stock check: {result['name']} — {'In stock' if result['in_stock'] else 'Out of stock'} ({result['quantity']} units)"
    else:
        result = search_docs(query)
        decimalai.log_tool_call(name="search_docs", input={"query": query}, output=result)
        TOOL_USED["v1"][query] = "search_docs"
        return f"Here's what I found: {result}"


def manifest_now(agent_name="support-agent"):
    """The manifest snapshot the tracer builds at the end of a trace.

    Same call the SDK makes internally (decimalai/generic.py), over the same
    inputs: the @decimalai.tool registry plus any skills found on disk. Recomputed
    here so the diff in Step 5 works with no account.
    """
    return extract_from_config(
        agent_name=agent_name,
        tools=get_registered_tools(),
        skills=discover_skills() or None,
    )


print(f"Running {len(QUERIES)} queries through Agent v1...\n")
for q in QUERIES:
    answer = run_agent_v1(q)
    print(f"  Q: {q}")
    print(f"  A: {answer}")
    print(f"     [tool: {TOOL_USED['v1'][q]}]\n")

MANIFEST_V1 = manifest_now()
V1_TOOLS = sorted(c.component_name for c in MANIFEST_V1.components
                  if c.component_type == "tool")

print(f"{len(QUERIES)} traces recorded" + (" and sent to DecimalAI." if ONLINE
                                           else " locally (offline: nothing sent)."))
print(f"manifest v1  {MANIFEST_V1.manifest_hash[:12]}  tools: {', '.join(V1_TOOLS)}")
print("Every one of those traces reports the same manifest — one version, not one per branch.")
'''),

    md(r'''
## Step 4 — Update the Agent (v2)

Now let's simulate a real-world agent update:

1. **Rename** `check_inventory` -> `lookup_stock` (clearer name)
2. **Add** a new tool: `process_refund`

This is exactly what happens when your team iterates on an agent.

One notebook-specific wrinkle: in production, v2 is a **new process** — the old
code is gone and the tool registry starts empty. A notebook keeps one Python
process alive across both versions, so the cell below clears the registry first.
Without that, `check_inventory` would still be registered, v2's manifest would
report four tools instead of three, and the rename would look like an addition.
'''),

    code(r'''
# --- Agent v2: Renamed tool + new tool ---

from decimalai.decorators import _registered_tools

# Stand in for the process restart that a real deploy is. The registry is a
# module-level dict that only ever grows within a process; a tool that is deleted
# from your source is only "gone" once the process that registered it is.
_registered_tools.clear()

# Unchanged in v2 — re-registered because the line above cleared everything.
search_docs = decimalai.tool(search_docs)


@decimalai.tool
def lookup_stock(item_id: str) -> dict:
    """Look up current stock for an item. (Renamed from check_inventory)"""
    inventory = {
        "SKU-1234": {"item_id": "SKU-1234", "name": "Wireless Mouse", "in_stock": True, "quantity": 42},
        "SKU-5678": {"item_id": "SKU-5678", "name": "USB-C Hub", "in_stock": False, "quantity": 0},
    }
    return inventory.get(item_id, {"item_id": item_id, "in_stock": True, "quantity": 100})


@decimalai.tool
def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund for an order. (NEW in v2)"""
    return {"order_id": order_id, "status": "refunded", "reason": reason, "amount": 29.99}


print("Agent v2 declares:", ", ".join(sorted(t["name"] for t in get_registered_tools())))
print("   - search_docs    unchanged")
print("   - lookup_stock   renamed from check_inventory")
print("   - process_refund new")
'''),

    code(r'''
@decimalai.trace(agent_name="support-agent")
def run_agent_v2(query: str) -> str:
    """Updated support agent with renamed + new tools."""
    if "stock" in query.lower() or "inventory" in query.lower():
        result = lookup_stock("SKU-1234")
        decimalai.log_tool_call(name="lookup_stock", input={"item_id": "SKU-1234"}, output=result)
        TOOL_USED["v2"][query] = "lookup_stock"
        return f"Stock check: {result['name']} — {'In stock' if result['in_stock'] else 'Out of stock'}"
    elif "refund" in query.lower():
        result = process_refund("ORD-001", "damaged item")
        decimalai.log_tool_call(name="process_refund", input={"order_id": "ORD-001"}, output=result)
        TOOL_USED["v2"][query] = "process_refund"
        return f"Refund processed: ${result['amount']} for order {result['order_id']}"
    else:
        result = search_docs(query)
        decimalai.log_tool_call(name="search_docs", input={"query": query}, output=result)
        TOOL_USED["v2"][query] = "search_docs"
        return f"Here's what I found: {result}"


# The SAME six queries. Re-running the workload is what makes the two versions
# comparable: any difference below is the agent changing, not the questions.
print(f"Running the same {len(QUERIES)} queries through Agent v2...\n")
for q in QUERIES:
    answer = run_agent_v2(q)
    print(f"  Q: {q}")
    print(f"  A: {answer}")
    print(f"     [tool: {TOOL_USED['v1'][q]} -> {TOOL_USED['v2'][q]}]\n")

MANIFEST_V2 = manifest_now()
V2_TOOLS = sorted(c.component_name for c in MANIFEST_V2.components
                  if c.component_type == "tool")

if ONLINE:
    decimalai.flush()  # drain the background sender before you go look

print(f"manifest v2  {MANIFEST_V2.manifest_hash[:12]}  tools: {', '.join(V2_TOOLS)}")
print("changed:", MANIFEST_V1.manifest_hash != MANIFEST_V2.manifest_hash)
'''),

    md(r'''
## Step 5 — What Changed, and What It Costs You

Two manifests, one diff. This runs with no account: the hashes and the tool sets
below were computed by the same extractor the tracer uses, from the same registry
the tracer reads.
'''),

    code(r'''
v1, v2 = set(V1_TOOLS), set(V2_TOOLS)
removed, added, kept = sorted(v1 - v2), sorted(v2 - v1), sorted(v1 & v2)

print(f"manifest v1  {MANIFEST_V1.manifest_hash[:12]}  {', '.join(sorted(v1))}")
print(f"manifest v2  {MANIFEST_V2.manifest_hash[:12]}  {', '.join(sorted(v2))}\n")
for label, names in (("gone   ", removed), ("new    ", added), ("kept   ", kept)):
    for n in names:
        print(f"  {label} {n}")

# Which of the v1 traces are stranded by the change. A trace is stranded when it
# called a tool the new manifest no longer declares: replayed against v2 it would
# ask for something that does not exist, and used as training data it would teach
# the model to do the same.
stranded = [q for q, t in TOOL_USED["v1"].items() if t in removed]

print(f"\n{len(stranded)} of {len(QUERIES)} v1 traces called a tool that v2 removed:")
for q in stranded:
    print(f"  - {q!r}  (called {TOOL_USED['v1'][q]})")
print(f"The other {len(QUERIES) - len(stranded)} only used tools that survived the change.")

print(f"\nAnd the same {len(QUERIES)} questions now route differently:")
for q in QUERIES:
    if TOOL_USED["v1"][q] != TOOL_USED["v2"][q]:
        print(f"  {TOOL_USED['v1'][q]:<16} -> {TOOL_USED['v2'][q]:<16} {q!r}")
'''),

    md(r'''
### The same thing, with an account

**Needs a key.** With one, the six traces above are stored against their manifest,
and **[your dashboard](https://app.decimal.ai)** -> **Agents -> support-agent**
turns the diff you just printed into an **Impact Report** over every trace you have
ever recorded for this agent — not only the six in this session:

| Classification | What it means |
|---|---|
| **Keep** | The trace only touched parts of the agent the change didn't affect. |
| **Repair** | The trace references something that moved but has a mechanical equivalent — e.g. a rename that can be rewritten in place. |
| **Replay** | The change is behavioural; the trace has to be re-run before it can be trusted. |
| **Drop** | What the trace used is gone with no equivalent. |

### Why This Matters

The cell above named the traces that called `check_inventory`. If you fine-tuned on
your raw trace history today, those rows would teach your model to call a tool that
no longer exists. The manifest is what makes that question answerable at all: without
one, a trace is just text that looks fine.
'''),

    md(r'''
## What Just Happened

```
Agent v1 answers 6 questions -> 6 traces, all reporting ONE manifest
    |
Agent changes (tool renamed, tool added)
    |
Agent v2 answers the SAME 6 questions -> a second manifest, auto-detected
    |
Diff the two -> which tools moved, and which stored traces are stranded
    |
With an account: keep / repair / replay / drop over your whole trace history,
then repair the stale ones and build a clean dataset
```

This happens **automatically, every time your agent changes** — the only thing you
wrote was `@decimalai.tool` and `@decimalai.trace()`.

## Next Steps

- [LangChain Quickstart](./quickstart_langchain.ipynb) — Instrument a real LangChain agent
- [OpenAI Agents Quickstart](./quickstart_openai_agents.ipynb) — Instrument an OpenAI Agents app
- [Concepts](https://docs.decimal.ai/concepts) — Understand manifests, traces, and datasets
- [Dashboard](https://app.decimal.ai) — Explore your traces (needs a free account)
'''),
]

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 2: quickstart_langchain.ipynb
# ═══════════════════════════════════════════════════════════════

lc_cells = [
    md(r'''
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart_langchain.ipynb)

# DecimalAI + LangChain Quickstart

**Instrument a LangChain agent with 2 lines of code.**

This notebook shows how to add DecimalAI tracing to any LangChain / LangGraph
application. Every LLM call, tool invocation, and chain step is captured automatically.

**Prerequisites:** an OpenAI API key (or any LangChain-supported LLM) for Steps 2-3 —
that is the LLM the agent runs on, and there is no agent run without it. A DecimalAI
key is **optional**: with one, the runs are traced to your dashboard; without one the
notebook initializes the SDK with tracing switched off and still runs, rather than
stopping at the second cell. Neither missing key raises — each cell says which one it
would need.
'''),

    md(r'''
## Step 1 — Install & Configure
'''),

    code(r'''
# Install dependencies (the [langchain] extra brings the adapter deps;
# `langchain` provides create_agent, `langchain-openai` the LLM binding)
!pip install -q "decimalai[langchain]" langchain langchain-openai
'''),

    code(keyless_start(
        init_kwargs=", langchain=True",
        offline_effect=r'''
# is a real kill switch: no client, no network call, and the langchain=True flag
# is ignored, so no adapter is installed and nothing is captured. The LangChain
# agent below still runs — it just runs untraced.
''',
        llm_key_block=_OPENAI_KEY_BLOCK,
        banner=r'''

print("DecimalAI: ONLINE  — LangChain runs will be traced to https://app.decimal.ai/traces"
      if ONLINE else
      "DecimalAI: OFFLINE — no usable key, so the SDK is initialized with tracing off.\n"
      "                     Nothing is captured; every cell below still runs.")
print("OpenAI:    ready — Steps 2-3 will call the model." if LLM_READY else
      "OpenAI:    missing — Steps 2-3 drive a real LLM, so they will print what they\n"
      "                     skipped instead of running. Set OPENAI_API_KEY (env var or\n"
      "                     Colab secret) and re-run this cell.")
'''),
    ),

    md(r'''
## Step 2 — Build a Simple LangChain Agent

We'll create a tool-calling agent with a couple of tools using langchain 1.x's
`create_agent` (a LangGraph graph under the hood). DecimalAI captures
everything — you don't need to add any extra callbacks or wrappers.
'''),

    code(r'''
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.tools import tool


# Define tools
@tool
def search_docs(query: str) -> str:
    """Search the knowledge base. Input: search query."""
    return f"Found 3 results for '{query}': [Article 1, Article 2, Article 3]"

@tool
def check_order(order_id: str) -> str:
    """Look up an order status. Input: order ID."""
    return f"Order {order_id}: Shipped on April 25, arriving April 29."

# Create agent. The tools above are plain Python and need no key; the model
# binding is the first thing that does, so that is what the guard is around.
agent = None

if LLM_READY:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    agent = create_agent(
        llm,
        [search_docs, check_order],
        system_prompt="You are a helpful customer support assistant.",
    )
    print("LangChain agent ready with 2 tools")
else:
    print("No OpenAI key, so no model to bind — the agent is not built.")
    print("The two tools are defined either way; set OPENAI_API_KEY and re-run")
    print("Step 1 and this cell to get the real thing.")
'''),

    md(r'''
## Step 3 — Run Queries (Traces Are Auto-Captured)
'''),

    code(r'''
queries = [
    "How do I reset my password?",
    "Where is my order ORD-12345?",
    "What is your return policy?",
]

if agent is None:
    print("Skipped: this step sends the 3 queries above to a real LLM, which needs")
    print("an OpenAI key. Nothing else in the notebook depends on it.")
else:
    # Every invocation is automatically traced by DecimalAI
    for q in queries:
        print(f"\n{'='*50}")
        print(f"Q: {q}")
        result = agent.invoke({"messages": [{"role": "user", "content": q}]})
        print(f"A: {result['messages'][-1].content}")

    if ONLINE:
        decimalai.flush()  # drain the background sender before you go look
        print("\n3 traces auto-captured and sent to DecimalAI!")
        print("Open your dashboard: https://app.decimal.ai/traces")
    else:
        print("\n3 agent runs finished. DecimalAI is offline, so none of them were")
        print("captured — add a key in Step 1 to see them in the dashboard.")
'''),

    md(r'''
## Step 4 — View in Dashboard

**Needs a DecimalAI key.** Open **[app.decimal.ai/traces](https://app.decimal.ai/traces)**.
For each trace you'll see:

- The full conversation flow (input → LLM → tool calls → output)
- Token usage and latency per LLM call
- Tool call inputs and outputs
- The auto-detected manifest (tools + model)

## Next Steps

- 📖 [Main Quickstart](./quickstart.ipynb) — See the version-aware manifest loop (runs with no keys at all)
- 📖 [Evaluations Guide](https://docs.decimal.ai/guides/evaluations) — Score your traces
- 📖 [Training Pipeline](https://docs.decimal.ai/tutorials/training-pipeline) — Build datasets from traces
'''),
]

# ═══════════════════════════════════════════════════════════════
# NOTEBOOK 3: quickstart_openai_agents.ipynb
# ═══════════════════════════════════════════════════════════════

oai_cells = [
    md(r'''
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/quickstart/quickstart_openai_agents.ipynb)

# DecimalAI + OpenAI Agents SDK Quickstart

**Instrument an OpenAI Agents application with 2 lines of code.**

This notebook shows how to add DecimalAI tracing to the OpenAI Agents SDK.
Agent runs, tool calls, handoffs, and guardrails are captured automatically.

**Prerequisites:** an OpenAI API key for Step 3 — that is the model the agent runs
on. A DecimalAI key is **optional**: with one, the runs are traced to your dashboard;
without one the notebook initializes the SDK with tracing switched off and still runs,
rather than stopping at the second cell. Neither missing key raises — each cell says
which one it would need.
'''),

    md(r'''
## Step 1 — Install & Configure
'''),

    code(r'''
# Install dependencies (the [openai-agents] extra brings the OpenAI Agents SDK)
!pip install -q "decimalai[openai-agents]"
'''),

    code(keyless_start(
        init_kwargs=", openai_agents=True",
        offline_effect=r'''
# is a real kill switch: no client, no network call, and the openai_agents=True
# flag is ignored, so no adapter is installed and nothing is captured. The Agent
# below still runs — it just runs untraced.
''',
        llm_key_block=_OPENAI_KEY_BLOCK,
        banner=r'''

print("DecimalAI: ONLINE  — Agent runs will be traced to https://app.decimal.ai/traces"
      if ONLINE else
      "DecimalAI: OFFLINE — no usable key, so the SDK is initialized with tracing off.\n"
      "                     Nothing is captured; every cell below still runs.")
print("OpenAI:    ready — Step 3 will call the model." if LLM_READY else
      "OpenAI:    missing — Step 3 drives a real LLM, so it will print what it skipped\n"
      "                     instead of running. Set OPENAI_API_KEY (env var or Colab\n"
      "                     secret) and re-run this cell.")
'''),
    ),

    md(r'''
## Step 2 — Define an Agent with Tools
'''),

    code(r'''
from agents import Agent, Runner, function_tool


@function_tool
def search_docs(query: str) -> str:
    """Search the knowledge base for relevant articles."""
    return f"Found 3 results for '{query}': [Article 1, Article 2, Article 3]"


@function_tool
def check_order(order_id: str) -> str:
    """Look up an order status by order ID."""
    return f"Order {order_id}: Shipped on April 25, arriving April 29."


# Defining the agent needs no key — it is a description of one. The key is what
# running it costs, so the guard is in Step 3.
agent = Agent(
    name="support-agent",
    instructions="You are a helpful customer support assistant. Use the tools to answer questions.",
    tools=[search_docs, check_order],
)

print("OpenAI Agent ready with 2 tools")
'''),

    md(r'''
## Step 3 — Run the Agent (Traces Are Auto-Captured)
'''),

    code(r'''
async def run_queries():
    queries = [
        "How do I reset my password?",
        "Where is my order ORD-12345?",
        "What is your return policy?",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        result = await Runner.run(agent, q)
        print(f"A: {result.final_output}")

    if ONLINE:
        decimalai.flush()  # drain the background sender before you go look
        print("\n3 traces auto-captured and sent to DecimalAI!")
        print("Open your dashboard: https://app.decimal.ai/traces")
    else:
        print("\n3 agent runs finished. DecimalAI is offline, so none of them were")
        print("captured — add a key in Step 1 to see them in the dashboard.")


if LLM_READY:
    await run_queries()
else:
    print("Skipped: running the agent calls a real LLM, which needs an OpenAI key.")
    print("Set OPENAI_API_KEY (env var or Colab secret), re-run Step 1, then this cell.")
'''),

    md(r'''
## What Gets Captured

**Needs a DecimalAI key.** For each Agent run, DecimalAI records:
- Agent name, instructions, and model
- Every LLM call with token usage
- Tool call inputs and outputs
- Agent handoffs (if using multi-agent)
- Guardrail evaluations
- The full manifest (tools + model + instructions)

## Next Steps

- 📖 [Main Quickstart](./quickstart.ipynb) — See the version-aware manifest loop (runs with no keys at all)
- 📖 [Concepts](https://docs.decimal.ai/concepts) — Understand manifests, traces, and datasets
- 📖 [Dashboard](https://app.decimal.ai) — Explore your traces
'''),
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
