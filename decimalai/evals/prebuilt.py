"""Pre-built deterministic evaluators — free, no LLM required.

These run instantly in the user's process and never make external calls.
Use them with ``decimalai.init(evals=[...])`` or ``run_evals()``.

Example::

    from decimalai.evals.prebuilt import json_valid, contains, length_check

    decimalai.init(
        agent_name="my-bot",
        evals=[json_valid, contains(["source:"]), length_check(min_words=20)],
    )
"""

from __future__ import annotations

import json
import re
from typing import List, Optional

from . import DecimalEval, EvalResult, TraceData

# ── json_valid ───────────────────────────────────────────────────


def _check_json_valid(trace: TraceData) -> EvalResult:
    """Check whether the output is valid JSON."""
    output = trace.output if isinstance(trace.output, str) else str(trace.output)
    output = output.strip()

    if not output:
        return EvalResult(score=0.0, passed=False, reason="Empty output")

    try:
        json.loads(output)
        return EvalResult(score=1.0, passed=True, reason="Valid JSON")
    except (json.JSONDecodeError, ValueError) as e:
        return EvalResult(
            score=0.0, passed=False, reason=f"Invalid JSON: {str(e)[:100]}"
        )


json_valid = DecimalEval(
    _check_json_valid,
    name="json_valid",
    category="quality:structural",
    builtin=True,
    version="1",
)
"""Evaluator that checks if the output is valid JSON."""


# ── contains ─────────────────────────────────────────────────────


def contains(
    patterns: List[str],
    *,
    case_sensitive: bool = False,
    require_all: bool = True,
    name: Optional[str] = None,
) -> DecimalEval:
    """Create an evaluator that checks if output contains required substrings.

    Args:
        patterns: List of substrings to search for.
        case_sensitive: Whether matching is case-sensitive.
        require_all: If True, ALL patterns must be present (AND).
                     If False, at least ONE must be present (OR).
        name: Custom eval name. Defaults to ``contains_<first_pattern>``.

    Returns:
        A ``DecimalEval`` evaluator.

    Example::

        evals=[contains(["source:", "http"], require_all=False)]
    """

    def _check(trace: TraceData) -> EvalResult:
        output = trace.output if isinstance(trace.output, str) else str(trace.output)
        if not case_sensitive:
            output = output.lower()

        found = []
        missing = []
        for p in patterns:
            target = p if case_sensitive else p.lower()
            if target in output:
                found.append(p)
            else:
                missing.append(p)

        if require_all:
            passed = len(missing) == 0
            score = len(found) / len(patterns) if patterns else 1.0
            reason = (
                f"All {len(patterns)} patterns found"
                if passed
                else f"Missing: {', '.join(missing[:5])}"
            )
        else:
            passed = len(found) > 0
            score = 1.0 if passed else 0.0
            reason = (
                f"Found: {', '.join(found[:5])}"
                if passed
                else f"None of {len(patterns)} patterns found"
            )

        return EvalResult(score=score, passed=passed, reason=reason)

    def _sanitize_name(s: str) -> str:
        return re.sub(r'[^a-zA-Z0-9_-]', '', s)

    eval_name = name or (_sanitize_name(f"contains_{patterns[0][:20]}") if patterns else "contains")
    return DecimalEval(
        _check,
        name=eval_name,
        category="quality:structural",
        version="1",
    )


# ── not_contains ─────────────────────────────────────────────────


def not_contains(
    patterns: List[str],
    *,
    case_sensitive: bool = False,
    name: Optional[str] = None,
) -> DecimalEval:
    """Create an evaluator that checks output does NOT contain banned strings.

    Passes only if NONE of the patterns appear in the output.

    Args:
        patterns: List of banned substrings.
        case_sensitive: Whether matching is case-sensitive.
        name: Custom eval name.

    Example::

        evals=[not_contains(["TODO", "FIXME", "hack"])]
    """

    def _check(trace: TraceData) -> EvalResult:
        output = trace.output if isinstance(trace.output, str) else str(trace.output)
        if not case_sensitive:
            output = output.lower()

        violations = []
        for p in patterns:
            target = p if case_sensitive else p.lower()
            if target in output:
                violations.append(p)

        if not violations:
            return EvalResult(
                score=1.0, passed=True,
                reason=f"None of {len(patterns)} banned patterns found",
            )
        else:
            return EvalResult(
                score=0.0, passed=False,
                reason=f"Found banned patterns: {', '.join(violations[:5])}",
            )

    eval_name = name or "not_contains"
    return DecimalEval(
        _check,
        name=eval_name,
        category="quality:safety",
        version="1",
    )


# ── regex_match ──────────────────────────────────────────────────


def regex_match(
    pattern: str,
    *,
    flags: int = 0,
    full_match: bool = False,
    name: Optional[str] = None,
) -> DecimalEval:
    """Create an evaluator that checks if output matches a regex pattern.

    Args:
        pattern: Regular expression pattern.
        flags: Regex flags (e.g., ``re.IGNORECASE``).
        full_match: If True, the ENTIRE output must match. If False,
                    a match anywhere in the output is sufficient.
        name: Custom eval name.

    Example::

        evals=[regex_match(r"\\d{4}-\\d{2}-\\d{2}", name="has_date")]
    """
    compiled = re.compile(pattern, flags)

    def _check(trace: TraceData) -> EvalResult:
        output = trace.output if isinstance(trace.output, str) else str(trace.output)

        if full_match:
            match = compiled.fullmatch(output.strip())
        else:
            match = compiled.search(output)

        if match:
            return EvalResult(
                score=1.0, passed=True,
                reason=f"Pattern matched: '{match.group()[:50]}'",
            )
        else:
            return EvalResult(
                score=0.0, passed=False,
                reason=f"Pattern /{pattern}/ not found in output",
            )

    eval_name = name or "regex_match"
    return DecimalEval(
        _check,
        name=eval_name,
        category="quality:structural",
        version="1",
    )


# ── length_check ─────────────────────────────────────────────────


def length_check(
    *,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    name: Optional[str] = None,
) -> DecimalEval:
    """Create an evaluator that checks output length bounds.

    Args:
        min_words: Minimum word count.
        max_words: Maximum word count.
        min_chars: Minimum character count.
        max_chars: Maximum character count.
        name: Custom eval name.

    Example::

        evals=[length_check(min_words=10, max_words=500)]
    """

    def _check(trace: TraceData) -> EvalResult:
        output = trace.output if isinstance(trace.output, str) else str(trace.output)
        words = output.split()
        word_count = len(words)
        char_count = len(output)

        violations = []

        if min_words is not None and word_count < min_words:
            violations.append(f"too short ({word_count} words < {min_words})")
        if max_words is not None and word_count > max_words:
            violations.append(f"too long ({word_count} words > {max_words})")
        if min_chars is not None and char_count < min_chars:
            violations.append(f"too short ({char_count} chars < {min_chars})")
        if max_chars is not None and char_count > max_chars:
            violations.append(f"too long ({char_count} chars > {max_chars})")

        if violations:
            return EvalResult(
                score=0.0, passed=False,
                reason="; ".join(violations),
            )
        else:
            return EvalResult(
                score=1.0, passed=True,
                reason=f"{word_count} words, {char_count} chars — within bounds",
            )

    eval_name = name or "length_check"
    return DecimalEval(
        _check,
        name=eval_name,
        category="quality:structural",
        version="1",
    )
