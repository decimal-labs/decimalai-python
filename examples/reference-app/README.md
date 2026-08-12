# SupportBot — DecimalAI Reference Application

A complete, realistic customer support agent that demonstrates the full DecimalAI workflow end-to-end.

## What This Shows

```
Agent v1 → Generate traces → Evaluate → Agent v2 (add tool) →
  → Impact Report → Repair stale traces → Build clean dataset
```

Each step is a standalone script in `scenarios/` that you run in order.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/decimal-labs/decimalai-python.git
cd decimalai-python/examples/reference-app

# 2. Set up environment
cp .env.example .env
# Edit .env with your DecimalAI API key

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full demo
make demo
# Or run scenarios individually:
python scenarios/01_generate_traces.py
python scenarios/02_update_agent.py
python scenarios/03_run_evals.py
python scenarios/04_build_dataset.py
```

## Project Structure

```
reference-app/
├── README.md                    # You're here
├── .env.example                 # Template for API keys
├── requirements.txt             # Dependencies
├── Makefile                     # make setup, make demo, make clean
├── agent/
│   ├── __init__.py
│   ├── support_agent.py         # The agent (v1 and v2)
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── search_docs.py       # Search knowledge base
│   │   ├── check_order.py       # Order status lookup
│   │   └── process_refund.py    # Refund processing (added in v2)
│   └── skills/
│       └── SKILL.md             # Agent skill definition
└── scenarios/
    ├── 01_generate_traces.py    # Run agent v1, send 10 traces
    ├── 02_update_agent.py       # Switch to v2, trigger manifest change
    ├── 03_run_evals.py          # Score traces with evaluators
    └── 04_build_dataset.py      # View impact report, build dataset
```

## The SupportBot Agent

A customer support chatbot that helps users with:
- **Password resets** — searches the knowledge base
- **Order status** — looks up orders by ID
- **Returns & refunds** — processes refund requests (v2 only)

### Agent v1 (2 tools)
- `search_docs` — Search the knowledge base
- `check_order` — Look up order status

### Agent v2 (3 tools — triggers manifest change)
- `search_docs` — Search the knowledge base (unchanged)
- `check_order` — Look up order status (unchanged)
- `process_refund` — Process refund requests (**NEW**)

## Scenario Walkthrough

### Scenario 1: Generate Traces
Runs 10 queries through Agent v1. Traces are captured automatically.

### Scenario 2: Update Agent
Switches to Agent v2 (adds `process_refund` tool). Runs 3 queries.
DecimalAI auto-detects the manifest change.

### Scenario 3: Run Evaluations
Scores all traces with deterministic evaluators:
- `answered_question` — Did the agent actually answer?
- `response_quality` — Is the response substantial?
- `no_hedging` — No uncertain language?

### Scenario 4: Build Dataset
Queries the impact report and builds a clean SFT dataset from
passing, version-compatible traces.

## Requirements

- Python 3.10+
- DecimalAI API key ([app.decimal.ai/settings](https://app.decimal.ai/settings))
- No LLM API key needed (uses mock responses)

## Documentation

- [DecimalAI Docs](https://docs.decimal.ai)
- [Quickstart Notebook](../quickstart/quickstart.ipynb)
- [Version-Aware Loop](../version-aware-loop/manifest_change.ipynb)
