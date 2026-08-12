"""Built-in deterministic evals that run on every trace by default.

These are free, structural checks — no LLM calls required.
Category: ``quality:deterministic``

Disabled via ``install(agent_name="...", builtin_evals=False)``.
"""

from __future__ import annotations

from . import DecimalEval, TraceData

# ── Built-in Eval Functions ──────────────────────────────────────


def _check_completion(trace: TraceData) -> bool:
    """Did the trace complete successfully?"""
    return trace.status == "success"


def _check_has_output(trace: TraceData) -> bool:
    """Is there a non-empty output?"""
    output = trace.output
    output = output if isinstance(output, str) else str(output)
    return len(output.strip()) > 0


def _check_tool_compliance(trace: TraceData) -> bool:
    """Did all tool calls produce results?"""
    if not trace.tool_calls:
        return True  # No tools used — pass
    return all(tc.result is not None for tc in trace.tool_calls)


def _check_latency(trace: TraceData) -> float:
    """Score based on response latency. Linear decay: 0ms=1.0, 10s+=0.0."""
    if trace.latency_ms is None:
        return 1.0  # Unknown latency — assume ok
    return max(0.0, 1.0 - trace.latency_ms / 10000)


def _check_token_efficiency(trace: TraceData) -> float:
    """Score based on total token usage. Linear decay: 0=1.0, 5000+=0.0."""
    if trace.total_tokens is None:
        return 1.0  # Unknown tokens — assume ok
    return max(0.0, 1.0 - trace.total_tokens / 5000)


# ── Create DecimalEval wrappers ──────────────────────────────────


BUILTIN_EVALS = [
    DecimalEval(
        _check_completion,
        name="completion",
        category="quality:deterministic",
        builtin=True,
    ),
    DecimalEval(
        _check_has_output,
        name="has_output",
        category="quality:deterministic",
        builtin=True,
    ),
    DecimalEval(
        _check_tool_compliance,
        name="tool_compliance",
        category="quality:deterministic",
        builtin=True,
    ),
    DecimalEval(
        _check_latency,
        name="latency",
        category="quality:deterministic",
        builtin=True,
    ),
    DecimalEval(
        _check_token_efficiency,
        name="token_efficiency",
        category="quality:deterministic",
        builtin=True,
    ),
]
"""List of all built-in evals. Used by ``langchain.py`` when
``builtin_evals=True`` (the default)."""


# ── LLM-as-Judge prebuilts (opt-in) ──────────────────────────────
#
# These are NOT included in BUILTIN_EVALS — they call an LLM and cost
# money. Users opt in explicitly:
#
#     from decimalai.evals.builtin import LLM_JUDGE_BUILTINS
#     decimalai.init(agent_name="...", evals=LLM_JUDGE_BUILTINS)
#
# Or pick individual judges from `decimalai.evals.llm_evaluators`.
#
# The list is deliberately short. Named judges are cheap to write and
# expensive to maintain — every one of them is a prompt that has to keep
# agreeing with human raters as models change. So the bar for inclusion is
# that a judge covers an axis most agent traces are graded on and that it
# needs nothing beyond the trace itself: no embedding model, no schema, no
# reference answer. Anything narrower belongs in the user's own `@eval`.


def _llm_judge_builtins():
    """Lazy factory — instantiating returns a fresh list each call so
    callers can mutate without poisoning shared state. Import only when
    referenced because `llm_evaluators` imports `litellm` lazily but the
    instantiation chain via `LlmEval.__init__` is non-trivial."""
    from .llm_evaluators import (
        Conciseness,
        Factuality,
        Faithfulness,
        Relevance,
        Toxicity,
    )
    return [
        Relevance(),
        Factuality(),
        Faithfulness(),
        Toxicity(),
        Conciseness(),
    ]


# Eager handle for convenience — equivalent to calling `_llm_judge_builtins()`.
# Users who want unique instances per call should use the factory directly.
LLM_JUDGE_BUILTINS = _llm_judge_builtins()
"""Opt-in list of five LLM-as-judge prebuilts. Costs API tokens per trace
evaluated, so it is intentionally NOT in BUILTIN_EVALS.

Each covers one axis an agent answer is commonly graded on, and each needs
only the trace itself:

- ``Relevance`` — does the answer address what was asked?
- ``Factuality`` — are its claims true?
- ``Faithfulness`` — is it grounded in the context the agent retrieved,
  rather than invented?
- ``Toxicity`` — is it safe to show a user?
- ``Conciseness`` — does it answer without padding?

Deliberately excluded: judges that need something the trace doesn't carry
or that duplicate one of the five. ``SemanticSimilarity`` needs an
embedding model, ``JsonSchemaMatch`` needs a schema (and is deterministic,
so it belongs in a plain ``@eval`` rather than an LLM call),
``AnswerCorrectness`` needs a reference answer, and ``Hallucination``
overlaps ``Faithfulness`` closely enough that shipping both would just
double the token cost for one signal."""
