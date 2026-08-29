"""The DELIVERY axis: which channel actually carries a skill body to the model.

Why this file exists
--------------------
The skills rail has two body channels and the suite only ever exercised whichever
one an adapter picked by default. ``inject_skill_body`` appeared ZERO times in the
whole conformance suite, so the channel was never a variable — and the defect that
started this was precisely a channel arithmetic error: on langchain
``inject_skill_body`` defaulted False AND the adapter registers no ``load_skill``
tool, so the sum was zero channels. The model got a menu of titles it had no
mechanism to read. Every contract item was green.

C14 closed the "zero channels" hole by grading the OUTCOME (a body reached the
model, by *any* channel). This closes the next one: **each channel works ON ITS
OWN**. A cell here turns one channel off and asserts the other still delivers, so
an adapter cannot pass by leaning on a channel the user has disabled.

The modes
---------
``injected``     the router pastes the body into the prompt; the ``load_skill``
                 tool is switched OFF (``DECIMALAI_LOAD_SKILL_TOOL=0``).
``tool_loaded``  the model pulls the body through the native ``load_skill`` tool;
                 injection is switched OFF (``DECIMALAI_INJECT_SKILL_BODY=0``).

Both are real, supported configurations, not test-only contortions: they are the
two documented knobs (``init(inject_skill_body=…)`` /
``DECIMALAI_LOAD_SKILL_TOOL=0``) set to the two settings a user can actually
choose. Each mode is also its own KILL-SWITCH test — the mode asserts the OFF
channel really went off, which is the half that catches a flag the SDK ignores.

How a mode reaches the SDK
--------------------------
Environment variables, set on the CHILD PROCESS before it starts. That is not a
convenience: ``DecimalConfig`` reads these in ``default_factory`` at construction,
and every adapter caches its ``SkillRouter`` in a module-level singleton whose
``inject_body`` is fixed at construction
(``decimalai/langchain.py:_get_skill_router``, and the same in ``anthropic.py``,
``openai_agents.py``, ``pydantic_ai.py``). Two modes in one process would
therefore grade the SECOND mode against the FIRST mode's router. One child per
(driver, mode) is the same argument ``isolation.py`` already makes for drivers.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

#: The adapter's own resolved default — what the per-driver matrix runs. Nothing
#: is forced, so ``DecimalConfig.resolve_inject_body`` answers from the adapter.
DEFAULT = "default"

#: Prompt injection is the only body channel. The load_skill tool is killed.
INJECTED = "injected"

#: The native ``load_skill`` tool is the only body channel. Injection is killed.
TOOL_LOADED = "tool_loaded"

#: The axis the delivery matrix is parametrised over. ``DEFAULT`` is deliberately
#: NOT here — it is the per-driver matrix's mode, and a cell that runs whatever
#: the adapter felt like is the thing this axis exists to replace.
DELIVERY_MODES: Tuple[str, ...] = (INJECTED, TOOL_LOADED)

#: mode -> the environment that produces it. Read by ``isolation`` when it spawns
#: the child, so the SDK sees them before it builds a config or a router.
#:
#: Both variables are public, documented SDK surface:
#:   DECIMALAI_INJECT_SKILL_BODY -> DecimalConfig.inject_skill_body (tri-state)
#:   DECIMALAI_LOAD_SKILL_TOOL   -> DecimalConfig.load_skill_tool   (kill switch)
MODE_ENV: Mapping[str, Dict[str, str]] = {
    INJECTED: {
        "DECIMALAI_INJECT_SKILL_BODY": "1",
        "DECIMALAI_LOAD_SKILL_TOOL": "0",
    },
    TOOL_LOADED: {
        "DECIMALAI_INJECT_SKILL_BODY": "0",
        "DECIMALAI_LOAD_SKILL_TOOL": "1",
    },
}

#: mode -> the channel that must carry the body, and the one that must not.
#: ``contract.grade_delivery`` reads this, so the spec and the environment that
#: produces it cannot drift into disagreeing about what the cell means.
MODE_CHANNELS: Mapping[str, Tuple[str, str]] = {
    INJECTED: ("prompt", "tool"),
    TOOL_LOADED: ("tool", "prompt"),
}

#: The two channel names ``contract._delivery_channels`` can report.
PROMPT_CHANNEL = "prompt"
TOOL_CHANNEL = "tool"


def mode_env(mode: str) -> Dict[str, str]:
    """The environment for ``mode``. Raises on an unknown mode rather than
    silently running the adapter's default and grading it as the mode."""
    if mode == DEFAULT:
        return {}
    try:
        return dict(MODE_ENV[mode])
    except KeyError:
        raise ValueError(
            f"unknown delivery mode {mode!r} — known: {DEFAULT}, "
            f"{', '.join(DELIVERY_MODES)}"
        ) from None
