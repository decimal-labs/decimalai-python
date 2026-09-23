"""Conservative English reply checks for the tool-free Support starter.

This catches known failure patterns, not every possible unsupported claim. The
semantic acceptance rubric remains independent and must also pass.
"""
from __future__ import annotations

import re

_ACTION = re.compile(
    r"\b(?:i|we)(?:['’]ll| will| can|['’]d like to| would like to)\s+"
    # Only offers of text assistance or explicit negatives are allowed. A verb
    # denylist missed paraphrases such as "we can take a billing review" in live runs.
    r"(?!(?:not|explain|describe|draft|outline|clarify|summarize)\b|help (?:you understand|word)\b)\w+"
    r"|\b(?:i|we)(?:['’]ve| have)?\s+(?:already\s+)?"
    r"(?:reviewed|escalated|refunded|credited|deleted|locked|cancelled|canceled|sent|"
    r"opened|initiated|started|stopped|requested|arranged|processed|contacted)\b",
    re.I,
)


def unsupported_action_claim(answer: str) -> bool:
    """Detect known first-person action claims where the runtime has no tools."""
    return bool(_ACTION.search(answer))


def draft_reply_violation(answer: str) -> str | None:
    """Return repair guidance for a known invalid draft, otherwise None."""
    if not answer.strip():
        return "Return a substantive customer-facing reply."
    if unsupported_action_claim(answer):
        return (
            "Remove claims or promises that you or your team perform actions. No tools are available. "
            "Use only supplied facts and explain a next step the customer can take; do not offer "
            "to request, review, contact, escalate or act for them."
        )
    lines = [line.strip() for line in answer.splitlines() if line.strip()
             and not re.match(r"^(subject:|hi\b|hello\b|dear\b)", line.strip(), re.I)]
    if lines and re.match(r"^(thank you|thanks) for (reaching out|contacting|your (message|email))", lines[0], re.I):
        return "Begin with this customer's concrete issue, not generic thanks for contacting support."
    return None
