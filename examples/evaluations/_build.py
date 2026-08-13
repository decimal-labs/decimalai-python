#!/usr/bin/env python3
"""Build evaluations notebook."""
import json, os

def _s(source):
    lines = source.split("\n")
    return [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": _s(s)}
def code(s): return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": _s(s)}

META = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"}}

cells = [
    md("[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/decimal-labs/decimalai-python/blob/main/examples/evaluations/builtin_evaluators.ipynb)\n"
       "\n"
       "# Evaluations: Built-in & Custom Evaluators\n"
       "\n"
       "**Score your agent's traces automatically with deterministic checks and LLM-powered judges.**\n"
       "\n"
       "This notebook covers:\n"
       "- 5 built-in deterministic evaluators (no LLM needed)\n"
       "- 5 LLM-powered evaluators (requires an API key)\n"
       "- Writing custom evaluators with the `@eval` decorator\n"
       "- Online evaluation (auto-score production traffic)\n"
       "- Batch evaluation (score existing traces)"),

    md("## Setup"),

    code("!pip install -q 'decimalai[evals]'"),

    code("import os\n"
         "os.environ[\"DECIMAL_API_KEY\"] = \"dai_sk_...\"  # ← Your DecimalAI key\n"
         "# os.environ[\"OPENAI_API_KEY\"] = \"sk-...\"    # ← Needed for LLM evaluators\n"
         "\n"
         "import decimalai\n"
         "decimalai.init()"),

    md("## Part 1: Deterministic Evaluators (No LLM Needed)\n"
       "\n"
       "These run instantly, cost nothing, and are great for structural checks."),

    code("from decimalai.evals import (\n"
         "    eval, TraceData, EvalResult,\n"
         "    json_valid, contains, not_contains, regex_match, length_check,\n"
         ")\n"
         "\n"
         "# Create a sample trace for testing\n"
         "sample = TraceData(\n"
         "    id=\"test-001\",\n"
         "    input=\"What is the return policy?\",\n"
         "    output='Our return policy allows returns within 30 days. Contact support@example.com for help.',\n"
         "    status=\"success\",\n"
         "    agent_name=\"support-agent\",\n"
         ")\n"
         "\n"
         "print(\"Sample trace:\")\n"
         "print(f\"  Input:  {sample.input}\")\n"
         "print(f\"  Output: {sample.output}\")"),

    code("# ── 1. json_valid: checks if output is valid JSON ──\n"
         "json_check = json_valid\n"
         "result = json_check(sample)\n"
         "print(f\"json_valid: passed={result.passed if result else 'skipped'}\")  # False — output is text\n"
         "\n"
         "# ── 2. contains: checks if output contains a substring ──\n"
         "has_policy = contains([\"return\"])\n"
         "result = has_policy(sample)\n"
         "print(f\"contains('return'): passed={result.passed if result else 'skipped'}\")  # True\n"
         "\n"
         "# ── 3. not_contains: checks output doesn't contain forbidden text ──\n"
         "no_pii = not_contains([\"555-\"])\n"
         "result = no_pii(sample)\n"
         "print(f\"not_contains('555-'): passed={result.passed if result else 'skipped'}\")  # True\n"
         "\n"
         "# ── 4. regex_match: checks output against a regex pattern ──\n"
         "has_email = regex_match(r'[\\w.]+@[\\w.]+')\n"
         "result = has_email(sample)\n"
         "print(f\"regex_match(email): passed={result.passed if result else 'skipped'}\")  # True\n"
         "\n"
         "# ── 5. length_check: checks output length is within bounds ──\n"
         "good_length = length_check(min_chars=20, max_chars=500)\n"
         "result = good_length(sample)\n"
         "print(f\"length_check(20-500): passed={result.passed if result else 'skipped'}\")  # True"),

    md("## Part 2: Custom Evaluators with `@eval`\n"
       "\n"
       "The `@eval` decorator turns any function into an evaluator.\n"
       "Your function receives a `TraceData` object and returns:\n"
       "- `bool` — pass/fail\n"
       "- `float` — score between 0.0 and 1.0\n"
       "- `EvalResult` — full control over score, pass/fail, and reason"),

    code("# ── Custom eval: simple bool ──\n"
         "@eval(name=\"answered_question\")\n"
         "def check_answered(trace: TraceData) -> bool:\n"
         "    \"\"\"Check that the agent actually answered (not just repeated the question).\"\"\"\n"
         "    return len(trace.output) > 20 and trace.input.lower() not in trace.output.lower()\n"
         "\n"
         "result = check_answered(sample)\n"
         "print(f\"answered_question: passed={result.passed}, score={result.score}\")\n"
         "\n"
         "\n"
         "# ── Custom eval: float score ──\n"
         "@eval(name=\"response_quality\")\n"
         "def quality_score(trace: TraceData) -> float:\n"
         "    \"\"\"Score response quality on multiple dimensions.\"\"\"\n"
         "    score = 0.0\n"
         "    if len(trace.output) > 50: score += 0.3       # Substantial response\n"
         "    if \"?\" not in trace.output[-20:]: score += 0.3  # Doesn't end with question\n"
         "    if trace.status == \"success\": score += 0.4      # No errors\n"
         "    return score\n"
         "\n"
         "result = quality_score(sample)\n"
         "print(f\"response_quality: score={result.score}, passed={result.passed}\")\n"
         "\n"
         "\n"
         "# ── Custom eval: EvalResult with reason ──\n"
         "@eval(name=\"no_hallucination\")\n"
         "def check_hallucination(trace: TraceData) -> EvalResult:\n"
         "    \"\"\"Check for common hallucination patterns.\"\"\"\n"
         "    hallucination_phrases = [\"I think\", \"probably\", \"I'm not sure\", \"maybe\"]\n"
         "    found = [p for p in hallucination_phrases if p.lower() in trace.output.lower()]\n"
         "    if found:\n"
         "        return EvalResult(score=0.3, passed=False, reason=f\"Hedging detected: {found}\")\n"
         "    return EvalResult(score=1.0, passed=True, reason=\"No hedging language found\")\n"
         "\n"
         "result = check_hallucination(sample)\n"
         "print(f\"no_hallucination: score={result.score}, passed={result.passed}, reason={result.reason}\")"),

    md("## Part 3: LLM-Powered Evaluators\n"
       "\n"
       "For nuanced quality checks, DecimalAI includes LLM-as-a-judge evaluators.\n"
       "These use `litellm` under the hood and support any LLM provider.\n"
       "\n"
       "> **Requires**: `OPENAI_API_KEY` (or another LLM provider key) set in environment."),

    code("# Uncomment and run if you have an OpenAI API key:\n"
         "\n"
         "# from decimalai.evals import Relevance, Factuality, Faithfulness, Toxicity, Conciseness\n"
         "#\n"
         "# relevance = Relevance()       # Is the output relevant to the input?\n"
         "# factuality = Factuality()     # Are the facts accurate?\n"
         "# faithfulness = Faithfulness() # Does output align with retrieved context?\n"
         "# toxicity = Toxicity()         # Is the output safe and appropriate?\n"
         "# conciseness = Conciseness()   # Is the output concise?\n"
         "#\n"
         "# result = relevance(sample)\n"
         "# print(f\"Relevance: score={result.score}, passed={result.passed}\")\n"
         "#\n"
         "# result = toxicity(sample)\n"
         "# print(f\"Toxicity: score={result.score}, passed={result.passed}\")"),

    md("## Part 4: Online Evaluation (Auto-Score Production Traffic)\n"
       "\n"
       "Pass your evaluators to the `instrument()` function. Every trace is\n"
       "automatically scored before being sent to DecimalAI."),

    code("# Online eval with LangChain\n"
         "# from decimalai.langchain import instrument\n"
         "# instrument(\n"
         "#     agent_name=\"support-agent\",\n"
         "#     evals=[check_answered, quality_score, check_hallucination],\n"
         "# )\n"
         "#\n"
         "# # Now every trace is auto-scored!\n"
         "# agent.invoke({\"input\": \"What is your return policy?\"})\n"
         "\n"
         "print(\"ℹ️  Uncomment the code above in your production agent.\")\n"
         "print(\"   Every trace will be scored by your evaluators automatically.\")"),

    md("## Part 5: Batch Evaluation (Score Existing Traces)\n"
       "\n"
       "Use `batch_eval` to run evaluators against traces already in DecimalAI."),

    code("# Batch eval against existing traces\n"
         "# from decimalai import batch_eval\n"
         "#\n"
         "# results = batch_eval(\n"
         "#     trace_ids=[\"trace-001\", \"trace-002\", \"trace-003\"],\n"
         "#     evals=[check_answered, quality_score, check_hallucination],\n"
         "# )\n"
         "# print(f\"Evaluated: {results['traces_evaluated']} traces\")\n"
         "# print(f\"Summary: {results['summary']}\")\n"
         "\n"
         "print(\"ℹ️  Replace trace_ids with real IDs from your dashboard.\")"),

    md("## Summary\n"
       "\n"
       "| Type | Examples | Cost | Speed |\n"
       "|------|---------|------|-------|\n"
       "| **Deterministic** | json_valid, contains, regex_match, length_check | Free | Instant |\n"
       "| **Custom** | @eval decorator — bool, float, or EvalResult | Free | Instant |\n"
       "| **LLM Judge** | Relevance, Factuality, Toxicity, Conciseness | LLM cost | ~1-3s |\n"
       "\n"
       "## Next Steps\n"
       "\n"
       "- 📖 [Quickstart](../quickstart/quickstart.ipynb) — Get your first traces\n"
       "- 📖 [Manifest Changes](../version-aware-loop/manifest_change.ipynb) — Version-aware loop\n"
       "- 📖 [Build Datasets](../datasets-and-training/build_sft_dataset.ipynb) — Create training data from scored traces"),
]

nb = {"cells": cells, "metadata": META, "nbformat": 4, "nbformat_minor": 4}
path = os.path.join(os.path.dirname(__file__), "builtin_evaluators.ipynb")
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"✅ {path}")
