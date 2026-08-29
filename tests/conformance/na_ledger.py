"""Every N/A and every skip this suite is allowed to emit, declared and counted.

Why a ledger
------------
An N/A is a hole in the matrix that reads like coverage. The suite already
refuses a SILENT one — ``Capabilities.__post_init__`` demands a reason, and the
reason is printed. What it never did was bound the SET: a reason is written once,
by whoever is adding the exemption, and after that nothing ever looks at the
total again. Twenty-four N/As can become thirty without a single line of the
suite going red, because each new one arrives with its own perfectly good
sentence.

That is not a hypothetical failure mode here. ``C13b`` was declared N/A on
langchain by a reason that was accurate, careful, and *was the defect* — "the
model has no way to ASK for a body, so the strongest rung observable here is
DELIVERED" describes, exactly, a rail that then delivered nothing. Prose cannot
distinguish a framework's limit from an adapter's choice, because both read the
same. A count can at least make the SET move visible in a diff.

So: every N/A the drivers can produce is enumerated here, the totals are stated
as numbers, and ``test_coverage.py`` fails if the computed set differs from this
one in either direction. Adding an exemption now costs a line in this file with
the flag that grants it — a place a reviewer is looking.

Why the skip half lives here too
--------------------------------
Same failure, one layer up. A skip is how a hole reaches the exit code as
success. ``pytest -q`` prints ``102 skipped`` and a green bar, and nobody reads
102. So a skip in this package must declare which KIND it is, and
:func:`skip_declared` turns an undeclared one into a failure at the cell that
tried to take it.
"""

from __future__ import annotations

from typing import Dict, NoReturn

import pytest

# ── capability N/As ──────────────────────────────────────────────────────────

#: ``"<driver>:<item>"`` -> the capability flag that grants the exemption.
#: Recomputed from the drivers by
#: ``test_coverage.test_every_capability_na_is_declared``, which fails on any
#: difference in either direction — an N/A that appears without a line here, and
#: a line here for an N/A that no longer happens (a stale exemption reads like
#: debt that has already been paid).
DECLARED_NA: Dict[str, str] = {
    # A rail that is prompt-injection only: the model cannot ASK for a body, so
    # there is no model-initiated pull for C13b to find. C13 still applies and is
    # graded, and so is C14 — the delivery verdict is NOT exempt here, which is
    # the correction that came out of the 2026-08-28 defect.
    "langchain:C13b": "model_can_load_skill_bodies",
    "anthropic:C13b": "model_can_load_skill_bodies",
    # No skills rail at all: nothing was ever offered, so nothing can be routed,
    # delivered or activated. Cross-checked against the SDK's own ledger of
    # seamless frameworks — see
    # test_coverage.test_rail_declarations_match_the_scaffold_seam_ledger.
    "llamaindex:C8": "has_skills_rail",
    "llamaindex:C13": "has_skills_rail",
    "llamaindex:C13b": "has_skills_rail",
    "llamaindex:C14": "has_skills_rail",
    "adk:C8": "has_skills_rail",
    "adk:C13": "has_skills_rail",
    "adk:C13b": "has_skills_rail",
    "adk:C14": "has_skills_rail",
    "crewai:C8": "has_skills_rail",
    "crewai:C13": "has_skills_rail",
    "crewai:C13b": "has_skills_rail",
    "crewai:C14": "has_skills_rail",
    "generic-otel:C8": "has_skills_rail",
    "generic-otel:C13": "has_skills_rail",
    "generic-otel:C13b": "has_skills_rail",
    "generic-otel:C14": "has_skills_rail",
    "claude-agent-sdk:C8": "has_skills_rail",
    "claude-agent-sdk:C13": "has_skills_rail",
    "claude-agent-sdk:C13b": "has_skills_rail",
    "claude-agent-sdk:C14": "has_skills_rail",
    # No run that emits a trace without a model call, so there is no degenerate
    # form for a manifest to be fabricated from.
    "crewai:C7b": "supports_degenerate",
    "pydantic-ai:C7b": "supports_degenerate",
}

#: The number above, written down. Redundant with ``len(DECLARED_NA)`` on
#: purpose: the set comparison tells you WHICH exemption moved, and this tells a
#: reviewer skimming a diff THAT the total moved, on one line, without counting.
NA_BUDGET = 24

# ── delivery-axis N/As ───────────────────────────────────────────────────────

#: ``"<driver>:<mode>"`` -> the adapter file whose refusal proves the limit.
#: Unlike the capability N/As above, each of these is re-proven at RUNTIME on
#: every run: the driver asks the adapter for the mode, and
#: ``contract._grade_framework_limit`` refuses the N/A unless the adapter emits
#: its documented refusal and delivers nothing by that channel.
DECLARED_DELIVERY_NA: Dict[str, str] = {
    # An invoke-layer callback patch observes one model call; it does not own the
    # turn, so a load_skill result has nowhere to be routed back to.
    "langchain:tool_loaded": "decimalai/langchain.py",
    # A patched `messages.create()` is one provider request; the tool loop is the
    # caller's own while-loop, which this adapter never sees.
    "anthropic:tool_loaded": "decimalai/anthropic.py",
}

DELIVERY_NA_BUDGET = 2

# ── journey-axis N/As ────────────────────────────────────────────────────────

#: ``"<driver>:J1"`` -> which of ``decimalai/cli/scaffold.py``'s OWN ledgers
#: explains why ``decimalai init`` writes no file for this framework:
#:
#: * ``UNSCAFFOLDED_WITH_SEAM`` — the adapter can deliver skills, there is just
#:   no template yet. A journey cell appears the day one is written, and
#:   ``journey.journey_framework`` reads ``SUPPORTED_FRAMEWORKS`` directly, so
#:   nobody has to remember to add it.
#: * ``NO_PROMPT_SEAM`` — ``decimalai init`` REFUSES to scaffold it, in the
#:   product, because "a generated file would trace correctly and deliver NONE of
#:   this agent's skills — silently".
#:
#: Unlike the capability N/As, no line here is a judgement this suite makes. The
#: set is RECOMPUTED from the shipped scaffold ledger by
#: ``test_coverage.test_every_journey_na_is_declared_and_counted``, which fails
#: on any difference in either direction. Teaching `decimalai init` a new
#: framework therefore turns a skip into a graded cell on the same commit, and
#: leaves a line here that must be deleted.
DECLARED_JOURNEY_NA: Dict[str, str] = {
    "anthropic:J1": "UNSCAFFOLDED_WITH_SEAM",
    "pydantic-ai:J1": "UNSCAFFOLDED_WITH_SEAM",
    "llamaindex:J1": "NO_PROMPT_SEAM",
    "claude-agent-sdk:J1": "NO_PROMPT_SEAM",
    "crewai:J1": "NO_PROMPT_SEAM",
    "adk:J1": "NO_PROMPT_SEAM",
    "generic-otel:J1": "NO_PROMPT_SEAM",
}

#: The number above, written down — same reason as ``NA_BUDGET``. It should go
#: DOWN: every entry is a framework whose users get no scaffold today.
JOURNEY_NA_BUDGET = 7

# ── skips ────────────────────────────────────────────────────────────────────

#: Test functions allowed to skip on an ENVIRONMENT condition rather than a
#: declared N/A, each with the condition. These are the cases where skipping is
#: the honest answer — the thing being checked is genuinely not present on this
#: machine — and each already prints why. Enforced by
#: ``test_coverage.test_no_undeclared_skip_call_sites``: any other
#: ``pytest.skip`` in this package must go through :func:`skip_declared`.
ENVIRONMENT_SKIPS: Dict[str, str] = {
    "test_advertised_snapshot_matches_docs": (
        "the decimalai-docs repo is a sibling checkout most clones (and CI) lack; "
        "the two guards beside it do not skip and cover every in-repo way to add a "
        "framework"
    ),
    "test_backend_validator_has_not_drifted": (
        "the platform repo is a private sibling checkout; the port is checked "
        "wherever both repos are present"
    ),
    "test_no_driver_is_silently_unavailable": (
        "opt-in via DECIMAL_CONFORMANCE_REQUIRE_ALL=1 — a developer with only "
        "[dev] installed should get skips, not a red suite"
    ),
}

#: The kinds of skip a graded cell may take. Anything else is a defect.
SKIP_KINDS = ("na", "delivery_na", "journey_na", "unavailable")


def skip_declared(kind: str, key: str, reason: str) -> NoReturn:
    """Skip this cell — but only if the skip is one this suite has declared.

    ``kind``:

    * ``na``          — a capability N/A. Must be in :data:`DECLARED_NA`.
    * ``delivery_na`` — a delivery-axis framework limit. Must be in
      :data:`DECLARED_DELIVERY_NA`.
    * ``journey_na``  — a framework ``decimalai init`` writes no file for. Must
      be in :data:`DECLARED_JOURNEY_NA`, and the set is recomputed from the
      SDK's own scaffold ledger rather than trusted.
    * ``unavailable`` — the framework is not installed. Refused outright when
      ``DECIMAL_CONFORMANCE_REQUIRE_ALL=1`` says this environment was supposed
      to have every framework.

    Anything else FAILS instead of skipping. A skip is how an ungraded cell
    reaches the exit code as success, and "102 skipped" next to a green bar is
    read by nobody.
    """
    import os

    if kind == "na":
        if key not in DECLARED_NA:
            pytest.fail(
                f"{key} was graded N/A but is not declared in "
                f"tests/conformance/na_ledger.py::DECLARED_NA.\n"
                f"  reason given: {reason}\n"
                f"An N/A is a hole in the matrix that reads like coverage. Add the "
                f"line (with the capability flag that grants it) and move "
                f"NA_BUDGET — deliberately, where a reviewer sees it — or, better, "
                f"fix the adapter so the item applies."
            )
    elif kind == "delivery_na":
        if key not in DECLARED_DELIVERY_NA:
            pytest.fail(
                f"{key} was graded N/A but is not declared in "
                f"tests/conformance/na_ledger.py::DECLARED_DELIVERY_NA.\n"
                f"  reason given: {reason}\n"
                f"A delivery channel a framework cannot do is a FRAMEWORK limit and "
                f"must be declared with a FrameworkLimit proof; a channel an "
                f"adapter merely did not implement is a defect, not an exemption."
            )
    elif kind == "journey_na":
        if key not in DECLARED_JOURNEY_NA:
            pytest.fail(
                f"{key} was graded N/A but is not declared in "
                f"tests/conformance/na_ledger.py::DECLARED_JOURNEY_NA.\n"
                f"  reason given: {reason}\n"
                f"A framework `decimalai init` will not scaffold has no journey to "
                f"walk — but that fact belongs to the SDK's scaffold ledger, not to "
                f"this suite. Add the line naming which ledger classifies it "
                f"(UNSCAFFOLDED_WITH_SEAM / NO_PROMPT_SEAM) and move "
                f"JOURNEY_NA_BUDGET, or better: write the template and delete the "
                f"exemption."
            )
    elif kind == "unavailable":
        if os.environ.get("DECIMAL_CONFORMANCE_REQUIRE_ALL") == "1":
            pytest.fail(
                f"{key} cannot run because its framework is not installed, and "
                f"DECIMAL_CONFORMANCE_REQUIRE_ALL=1 says this environment was "
                f"supposed to have every one of them.\n  {reason}"
            )
    else:
        pytest.fail(
            f"skip_declared({kind!r}) — unknown skip kind. Known: "
            f"{', '.join(SKIP_KINDS)}. A skip nobody classified is a hole nobody "
            f"is counting."
        )
    pytest.skip(reason)
