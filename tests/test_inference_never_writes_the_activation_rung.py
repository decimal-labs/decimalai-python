"""Prompt-text inference may write OFFERED and DELIVERED. It must never write ACTIVATED.

The three rungs are not interchangeable. Offered and delivered are statements about
what the router put in front of the model; activated is a statement about what the
MODEL asked for, and it is the one that becomes a `TraceSkillActivation` row, feeds
`router_activated_count` and lands on the activation ledger. Inferring it from prompt
text asserts a choice the model never made — the fabrication C13 exists to forbid.

WHY THIS IS A UNIT TEST AND NOT A CONFORMANCE CASE. Measured 2026-09-05, as the
2026-09-04 inventory asked: fold `delivered` into `skills_loaded_by_agent` inside
`_infer_skill_rungs` — the exact fabrication — and the hermetic matrix stays GREEN,
155 passed, on all nine drivers. Two reasons, and neither is a missing clause:

  1. Every conformance rail has the router report its own rungs, and
     `infer_prompt_rungs` suppresses names the router already accounted for. Logged
     during a full matrix run: inference fires 24 times and returns
     `offered=[] delivered=[]` every single time, so the mutated line has nothing to
     add and C13 grades an empty set. Its own PASS message says so in as many words —
     "recorded no activation" — on every driver.
  2. Where the model DOES pull a body, it pulls the same skill the router delivered,
     so a fold produces a name that has real evidence behind it anyway.

The driver's skills registry is now seeded (`drivers/openai_agents.py`) so the path at
least RUNS — before that it returned early on an empty registry and the line was
unreachable, which is a worse kind of green. Reaching it is not the same as grading
it, so the grading lives here, where the router can be made silent.
"""
from __future__ import annotations

import pytest


class _Call:
    def __init__(self, rendered_input):
        self.rendered_input = rendered_input


class _Acc:
    """The fields `_infer_skill_rungs` reads and writes."""

    def __init__(self, rendered):
        self.llm_calls = [_Call(rendered)]
        self.skills_offered_in_prompt: set = set()
        self.skills_delivered: set = set()
        self.skills_loaded_by_agent: set = set()


BODY = ("# Alpha\n\nSENTINEL-SKILLBODY-ALPHA: opened boxes carry a 23.5% restocking "
        "fee.\nOnly a delivered body carries this line.")
REGISTRY = [{"name": "conformance-skill-alpha",
             "description": "Alpha skill offered by the conformance probe.",
             "body": BODY}]


def _processor():
    from decimalai.openai_agents import DecimalTracingProcessor

    return DecimalTracingProcessor(agent_name="a", skills_registry=REGISTRY)


def _rendered_with_body():
    """A prompt carrying the body and NOTHING from the router — the only shape in
    which inference contributes anything, and the shape no conformance rail has."""
    return [{"role": "system", "content": BODY}, {"role": "user", "content": "hi"}]


def test_inference_reads_a_delivered_body_off_the_prompt():
    """The positive control. Without this, the test below could pass because the
    inference did nothing at all — which is exactly how the conformance matrix
    stayed green under the mutation."""
    acc = _Acc(_rendered_with_body())
    _processor()._infer_skill_rungs(acc)
    assert "conformance-skill-alpha" in acc.skills_delivered, (
        "inference read no delivered body off a prompt that literally contains it — "
        "the negative test below would then be vacuous"
    )


def test_inference_never_records_an_activation():
    """THE CLAUSE. A body in the prompt is the router delivering it. The model has
    not asked for anything."""
    acc = _Acc(_rendered_with_body())
    _processor()._infer_skill_rungs(acc)
    assert acc.skills_loaded_by_agent == set(), (
        "prompt-text inference wrote the ACTIVATION rung. That claims the model asked "
        "for a skill it was merely shown, which becomes a TraceSkillActivation row and "
        "an entry in the activation ledger"
    )


def test_a_name_in_a_user_message_is_not_an_activation_either():
    """The user typing a skill's name is not the model choosing it."""
    acc = _Acc([{"role": "user", "content": "please use conformance-skill-alpha"}])
    _processor()._infer_skill_rungs(acc)
    assert acc.skills_loaded_by_agent == set()


@pytest.mark.parametrize("router_reported", [True, False])
def test_the_rung_stays_empty_whether_or_not_the_router_spoke(router_reported):
    """Suppression is why the conformance matrix could not see this: when the router
    has already accounted for a name, inference returns nothing and a fold has nothing
    to fold. The clause must hold in BOTH states, not just the one the harness
    happens to produce."""
    acc = _Acc(_rendered_with_body())
    if router_reported:
        acc.skills_offered_in_prompt.add("conformance-skill-alpha")
        acc.skills_delivered.add("conformance-skill-alpha")
    _processor()._infer_skill_rungs(acc)
    assert acc.skills_loaded_by_agent == set()
