#!/usr/bin/env python3
"""Scenario 3: Run evaluations on all traces.

Defines 3 custom evaluators and scores all existing traces.
Results are pushed back to DecimalAI for dashboard visualization.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import decimalai
decimalai.init()

from decimalai.evals import eval, TraceData, EvalResult


# ── Define evaluators ──


@eval(name="answered_question")
def check_answered(trace: TraceData) -> bool:
    """Did the agent provide a substantive answer (not just echo the question)?"""
    return (
        len(trace.output) > 20
        and trace.input.lower()[:30] not in trace.output.lower()
    )


@eval(name="response_quality")
def quality_score(trace: TraceData) -> float:
    """Score response quality on multiple dimensions (0.0 to 1.0)."""
    score = 0.0
    if len(trace.output) > 30:
        score += 0.3  # Substantial response
    if "?" not in trace.output[-20:]:
        score += 0.3  # Doesn't end with a question
    if trace.status == "success":
        score += 0.4  # No errors
    return min(score, 1.0)


@eval(name="no_hedging")
def check_hedging(trace: TraceData) -> EvalResult:
    """Check that the agent doesn't use uncertain language."""
    hedging = ["I think", "probably", "I'm not sure", "maybe", "might be"]
    found = [p for p in hedging if p.lower() in trace.output.lower()]
    if found:
        return EvalResult(
            score=0.3, passed=False,
            reason=f"Hedging language detected: {', '.join(found)}"
        )
    return EvalResult(score=1.0, passed=True, reason="No hedging language")


def main():
    print("=" * 55)
    print("  Scenario 3: Run Evaluations")
    print("  Evaluators: answered_question, response_quality, no_hedging")
    print("=" * 55)
    print()

    # Demonstrate the evaluators on sample data
    samples = [
        TraceData(
            id="demo-1",
            input="How do I reset my password?",
            output="Go to Settings > Security > Reset Password. You'll receive a confirmation email within 5 minutes.",
            status="success",
            agent_name="support-bot",
        ),
        TraceData(
            id="demo-2",
            input="What is order ORD-10001 status?",
            output="Your order ORD-10001 is currently delivered. Items: Wireless Mouse.",
            status="success",
            agent_name="support-bot",
        ),
        TraceData(
            id="demo-3",
            input="Can I get a refund?",
            output="I think you might be able to return it, but I'm not sure about the policy.",
            status="success",
            agent_name="support-bot",
        ),
    ]

    evals = [check_answered, quality_score, check_hedging]

    for sample in samples:
        print(f"  Trace: \"{sample.input}\"")
        print(f"  Output: \"{sample.output[:60]}...\"")
        for ev in evals:
            result = ev(sample)
            if result:
                status = "✅" if result.passed else "❌"
                print(f"    {status} {ev.name}: score={result.score:.1f} | {result.reason if hasattr(result, 'reason') and result.reason else ''}")
        print()

    print("─" * 55)
    print()

    # In production, you'd use batch_eval against real trace IDs:
    print("  💡 To score production traces, use batch_eval:")
    print()
    print("     from decimalai import batch_eval")
    print("     results = batch_eval(")
    print("         trace_ids=[\"abc\", \"def\", ...],")
    print("         evals=[check_answered, quality_score, check_hedging],")
    print("     )")
    print("     print(results['summary'])")
    print()
    print("📊 View eval results: https://app.decimal.ai/evaluate")
    print()


if __name__ == "__main__":
    main()
