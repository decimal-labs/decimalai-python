"""Unit tests for the conformance contract's activation items (C13 / C13b).

These run in the DEFAULT suite, not the conformance matrix, and they exist
because of a specific failure mode: on two of the four rail drivers the
activation set is legitimately empty today, so every C13 clause passes without
examining anything. A green matrix cell therefore does not prove the item bites.
These tests feed C13 the fabrication shapes directly — a delivered body recorded
as activated, a menu row promoted, a never-served skill, evidence borrowed from a
neighbouring trace, a wrong hash — and assert it goes red on each.

Nothing here imports a framework. The inputs are the same dicts the probe
records off the wire, so a test that passes here means the item would have
caught the same payload in the matrix.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import pytest

from tests.conformance import contract
from tests.conformance.harness import Observation, Phase
from tests.conformance.probe import Probe, Recorded

ALPHA = "conformance-skill-alpha"
BETA = "conformance-skill-beta"
ALPHA_BODY = "# Alpha\n\nAlpha guidance for the conformance run."
BETA_BODY = "# Beta\n\nBeta guidance for the conformance run."
ALPHA_SIG = "Alpha guidance for the conformance run."

SKILLS = [
    {"name": ALPHA, "description": "Alpha skill.", "body": ALPHA_BODY},
    {"name": BETA, "description": "Beta skill.", "body": BETA_BODY},
]

#: The menu fragment the platform's router actually renders — offered, not
#: delivered. Present in every trace below, so every "fabrication" case is the
#: realistic one: the name IS in the prompt, it just was not asked for.
MENU = (
    "Available skills:\n"
    f"- {ALPHA}: Alpha skill.\n"
    f"- {BETA}: Beta skill.\n"
)


# ── builders ─────────────────────────────────────────────────────────────────


def _trace(
    *,
    messages: Optional[List[Any]] = None,
    active_skills: Optional[List[Any]] = None,
    loaded: Optional[List[str]] = None,
    offered: Optional[List[str]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    spans: Optional[List[Dict[str, Any]]] = None,
    rendered_input: Any = None,
) -> Dict[str, Any]:
    return {
        "routing_id": "rt_" + "0" * 24,
        "skills_offered_in_prompt": [ALPHA, BETA] if offered is None else offered,
        "active_skills": active_skills or [],
        "skills_loaded_by_agent": loaded or [],
        "llm_calls": [
            {
                "rendered_input": (
                    rendered_input if rendered_input is not None
                    else (messages if messages is not None else [_system()])
                ),
                "tool_calls": tool_calls or [],
                "output": "done",
            }
        ],
        "spans": spans or [],
    }


def _system(extra: str = "") -> Dict[str, Any]:
    return {"role": "system", "content": MENU + extra}


def _tool_result(body: str = ALPHA_BODY) -> Dict[str, Any]:
    """The tool RESULT message — the body coming back because the model asked."""
    return {"role": "tool", "name": "load_skill", "content": f"## Skill: {ALPHA}\n\n{body}"}


def _function_call() -> Dict[str, Any]:
    """The openai-agents shape: the model's own request, echoed as a message."""
    return {
        "role": "assistant",
        "type": "function_call",
        "name": "load_skill",
        "content": '{"name": "%s"}' % ALPHA,
    }


def _load_span(name: str = ALPHA, body: str = ALPHA_BODY) -> Dict[str, Any]:
    return {
        "id": "sp1",
        "span_type": "tool",
        "name": "load_skill",
        "input_preview": {"name": name},
        "output_preview": f"## Skill: {name}\n\n{body}",
    }


def _obs(traces: List[Dict[str, Any]], *, served: List[str] = (ALPHA,)) -> Observation:
    reqs: List[Recorded] = []
    seq = 0
    for name in served:
        seq += 1
        reqs.append(
            Recorded(
                seq=seq, method="GET",
                path=f"/api/v1/skills/{name}/body", query={}, body=None, status=200,
            )
        )
    for t in traces:
        seq += 1
        reqs.append(
            Recorded(
                seq=seq, method="POST", path="/api/v1/traces", query={}, body=t,
                status=200,
            )
        )
    probe = Probe()
    probe.skills = [dict(s) for s in SKILLS]
    phase = Phase(name="skills", ctxs=[], requests=reqs)
    return Observation(driver=None, probe=probe, ctx=None, phases={"skills": phase})


def _c13(traces, **kw) -> contract.Result:
    return contract.c13_skills_activation(_obs(traces, **kw))


def _c13b(traces, **kw) -> contract.Result:
    return contract.c13b_skills_activation_recorded(_obs(traces, **kw))


# ── the honest shapes: C13 green ─────────────────────────────────────────────


def test_openai_agents_shape_passes_and_names_its_evidence() -> None:
    """Tool-role body + load_skill span + assistant function_call."""
    t = _trace(
        messages=[_system(), _function_call(), _tool_result()],
        loaded=[ALPHA],
        spans=[_load_span()],
    )
    r = _c13([t] * 8)
    assert r.status == contract.PASS, r.message
    assert "tool-role body" in r.message
    assert "load_skill span input+output" in r.message
    assert "assistant function_call" in r.message
    assert ALPHA in r.message


def test_pydantic_ai_shape_passes_with_no_tool_span_at_all() -> None:
    """The other live shape: tool_calls on the llm_call, no tool span."""
    t = _trace(
        messages=[_system(), _tool_result()],
        loaded=[ALPHA],
        tool_calls=[{"tool_name": "load_skill", "args": {"name": ALPHA}}],
    )
    r = _c13([t])
    assert r.status == contract.PASS, r.message
    assert "model tool_call args" in r.message


def test_a_tool_call_argument_alone_is_enough() -> None:
    """No body came back yet, but the MODEL asked — that is the strongest signal."""
    t = _trace(
        messages=[_system()],
        loaded=[ALPHA],
        tool_calls=[{"tool_name": "load_skill", "args": {"name": ALPHA}}],
    )
    assert _c13([t]).status == contract.PASS


def test_an_empty_activation_set_passes_and_says_only_what_it_graded() -> None:
    """The prompt-injection rails. A vacuous green must announce itself — and
    must not announce something it never looked at.

    This test used to assert the message contained the word "delivered". The
    branch it grades never consults `served`, the rendered text, or anything
    else, so on a rail that delivered NOTHING the matrix still printed "the
    strongest observable rung is delivered". The fixture here passes the body in
    the system prompt, which made the claim true for this case and false for the
    integration path that reuses the same branch. Delivery is now C14's verdict.
    """
    r = _c13([_trace(messages=[_system(ALPHA_BODY)])], served=[])
    assert r.status == contract.PASS
    assert "recorded no activation" in r.message
    assert "C14" in r.message, "it must point at the item that DOES grade delivery"
    assert "verified" not in r.message
    # The load-bearing half: this branch must not claim a delivery it never checked.
    assert "rung is delivered" not in r.message


# ── the fabrication shapes: C13 must go red ──────────────────────────────────


def test_a_delivered_body_in_the_system_prompt_is_not_an_activation() -> None:
    """The live defect: SkillRouter renders an injected body as '## Skill: X'
    into the SYSTEM message, and detect_skill_activations matches it."""
    t = _trace(
        messages=[{"role": "system", "content": f"{MENU}\n## Skill: {ALPHA}\n\n{ALPHA_BODY}"}],
        active_skills=[{"name": ALPHA}],
    )
    r = _c13([t])
    assert r.status == contract.FAIL
    assert "DELIVERED, not activated" in r.message
    assert "system-role message" in r.message


def test_an_offered_menu_row_is_not_an_activation() -> None:
    """One fragment-format change (a bracket menu) trips the detector's Tier 1."""
    t = _trace(
        messages=[{"role": "system", "content": f"Skills: [{ALPHA}] [{BETA}]"}],
        active_skills=[ALPHA],
    )
    r = _c13([t], served=[])
    assert r.status == contract.FAIL
    assert "MODEL asking for it" in r.message


def test_a_negative_instruction_is_not_an_activation() -> None:
    """'Never use [X] for this task.' currently returns ['X'] from the detector."""
    t = _trace(
        messages=[{"role": "system", "content": f"{MENU}\nNever use [{ALPHA}] for this task."}],
        active_skills=[ALPHA],
    )
    assert _c13([t], served=[]).status == contract.FAIL


def test_a_user_typing_the_skill_name_is_not_an_activation() -> None:
    t = _trace(
        messages=[_system(), {"role": "user", "content": f"use {ALPHA} please: {ALPHA_SIG}"}],
        active_skills=[ALPHA],
    )
    r = _c13([t])
    assert r.status == contract.FAIL
    assert "user-role message" in r.message


def test_the_assistant_merely_mentioning_the_name_is_not_a_function_call() -> None:
    """Prose is not a tool call. Only a typed function_call message counts."""
    t = _trace(
        messages=[_system(), {"role": "assistant", "content": f"I could use {ALPHA}."}],
        active_skills=[ALPHA],
    )
    r = _c13([t])
    assert r.status == contract.FAIL
    assert "not activated" in r.message


def test_a_never_served_skill_with_no_evidence_fails_clause_two() -> None:
    """conformance-skill-beta is offered on every lane and served on none."""
    t = _trace(
        messages=[_system(), _tool_result()],
        loaded=[ALPHA, BETA],
        spans=[_load_span()],
    )
    r = _c13([t], served=[ALPHA])
    assert r.status == contract.FAIL
    assert BETA in r.message


def test_a_never_served_skill_fails_the_control_even_WITH_full_evidence() -> None:
    """The standing negative control, isolated from clause 2.

    Beta here has every corroborating channel C13 accepts — the body comes back
    in a tool-role message and a load_skill span names it — but the probe's
    router never served beta's body on this rail. Something produced a body the
    router did not hand over, and an activation joined to a routing decision
    that never delivered it is a fabrication however well corroborated.
    """
    t = _trace(
        messages=[
            _system(),
            {"role": "tool", "name": "load_skill",
             "content": f"## Skill: {BETA}\n\n{BETA_BODY}"},
        ],
        loaded=[BETA],
        spans=[_load_span(BETA, BETA_BODY)],
    )
    r = _c13([t], served=[ALPHA])
    assert r.status == contract.FAIL
    assert "never served a body" in r.message
    assert BETA in r.message


def test_a_tool_result_that_only_NAMES_the_skill_is_not_evidence() -> None:
    """The tool loop ran, but no body came back — a failed load is not a load.

    E1 matches the body's signature line, never the name, because the name is
    in the menu row too: matching it would make an offered row indistinguishable
    from a delivered body the moment it appeared under a tool role.
    """
    t = _trace(
        messages=[
            _system(),
            {"role": "tool", "name": "load_skill", "content": f"error: {ALPHA} not found"},
        ],
        loaded=[ALPHA],
    )
    r = _c13([t])
    assert r.status == contract.FAIL
    assert "not activated" in r.message


def test_an_activation_the_router_never_offered_fails_clause_one() -> None:
    t = _trace(
        messages=[_system(), _tool_result(), _function_call()],
        loaded=[ALPHA],
        offered=[BETA],
        spans=[_load_span()],
    )
    r = _c13([t])
    assert r.status == contract.FAIL
    assert "skills_offered_in_prompt" in r.message


def test_evidence_from_a_NEIGHBOURING_trace_does_not_corroborate_this_one() -> None:
    """C8's delivered check is phase-level; this one is per-trace, which is the
    gap that lets a leaked loaded-name between concurrent lanes look fine."""
    real = _trace(messages=[_system(), _tool_result()], loaded=[ALPHA], spans=[_load_span()])
    borrowed = _trace(messages=[_system()], loaded=[ALPHA])
    r = _c13([real, borrowed])
    assert r.status == contract.FAIL
    assert "DELIVERED, not activated" in r.message
    # …and with the menu gone too, the message says the name is simply absent.
    bare = _trace(messages=[{"role": "system", "content": "no skills here"}], loaded=[ALPHA])
    r2 = _c13([real, bare])
    assert r2.status == contract.FAIL
    assert "nowhere in this trace" in r2.message


def test_a_flattened_string_prompt_is_never_evidence() -> None:
    """An adapter that renders its prompt to one string has destroyed the role,
    so nothing in it can show WHO asked. That must fail, not be inferred."""
    t = _trace(
        rendered_input=f"system: {MENU}\ntool: ## Skill: {ALPHA}\n\n{ALPHA_BODY}",
        loaded=[ALPHA],
    )
    r = _c13([t])
    assert r.status == contract.FAIL
    assert "not activated" in r.message


def test_a_wrong_hash_fails_even_when_the_activation_is_real() -> None:
    """A wrong hash resolves the activation to a version that never ran."""
    t = _trace(
        messages=[_system(), _tool_result(), _function_call()],
        active_skills=[{"name": ALPHA, "hash": "sha256:" + "de" * 32}],
        spans=[_load_span()],
    )
    r = _c13([t])
    assert r.status == contract.FAIL
    assert "never ran" in r.message


def test_the_right_hash_passes_with_or_without_the_sha256_prefix() -> None:
    digest = hashlib.sha256(ALPHA_BODY.encode("utf-8")).hexdigest()
    for value in (f"sha256:{digest}", digest):
        t = _trace(
            messages=[_system(), _tool_result(), _function_call()],
            active_skills=[{"name": ALPHA, "hash": value}],
            spans=[_load_span()],
        )
        assert _c13([t]).status == contract.PASS, value


def test_no_trace_at_all_fails_rather_than_passing_vacuously() -> None:
    r = _c13([])
    assert r.status == contract.FAIL
    assert "no trace at all" in r.message


def test_a_span_that_names_the_skill_but_returns_nothing_is_not_evidence() -> None:
    """A load_skill span whose output does not carry the body proves a call was
    attempted, not that the body reached the model."""
    span = _load_span()
    span["output_preview"] = "error: not found"
    t = _trace(messages=[_system()], loaded=[ALPHA], spans=[span])
    assert _c13([t]).status == contract.FAIL


# ── the union the platform actually persists ─────────────────────────────────


def test_both_wire_fields_are_graded_because_the_backend_merges_them() -> None:
    """skills_loaded_by_agent becomes a TraceSkillActivation row just like
    active_skills, so grading active_skills alone would grade a field, not the
    number. Both must be held to clause 2."""
    via_active = _trace(messages=[_system(ALPHA_BODY)], active_skills=[ALPHA])
    via_loaded = _trace(messages=[_system(ALPHA_BODY)], loaded=[ALPHA])
    assert _c13([via_active], served=[]).status == contract.FAIL
    assert _c13([via_loaded], served=[]).status == contract.FAIL


def test_a_dict_entry_and_a_bare_string_entry_are_read_the_same_way() -> None:
    dict_form = _trace(
        messages=[_system(), _tool_result(), _function_call()],
        active_skills=[{"name": ALPHA}], spans=[_load_span()],
    )
    str_form = _trace(
        messages=[_system(), _tool_result(), _function_call()],
        active_skills=[ALPHA], spans=[_load_span()],
    )
    assert _c13([dict_form]).status == contract.PASS
    assert _c13([str_form]).status == contract.PASS


# ── C13b: the mirror ─────────────────────────────────────────────────────────


def test_c13b_fails_when_a_model_initiated_pull_is_dropped() -> None:
    t = _trace(messages=[_system(), _tool_result()], loaded=[], active_skills=[])
    r = _c13b([t])
    assert r.status == contract.FAIL
    assert "dropped on the floor" in r.message


def test_c13b_passes_when_the_pull_is_recorded() -> None:
    t = _trace(messages=[_system(), _tool_result()], loaded=[ALPHA])
    r = _c13b([t])
    assert r.status == contract.PASS
    assert ALPHA in r.message


def test_c13b_does_not_read_a_delivered_body_as_a_pull() -> None:
    """A body in the SYSTEM prompt was never pulled, so nothing is missing."""
    t = _trace(messages=[_system(ALPHA_BODY)])
    r = _c13b([t], served=[])
    assert r.status == contract.PASS
    assert "no model-initiated body pull" in r.message


def test_c13b_fails_with_no_trace() -> None:
    assert _c13b([]).status == contract.FAIL


# ── registration / gating ────────────────────────────────────────────────────


def test_the_items_are_registered_and_ordered_last() -> None:
    assert contract.ITEMS["C13"] is contract.c13_skills_activation
    assert contract.ITEMS["C13b"] is contract.c13b_skills_activation_recorded
    assert contract.ITEMS["C14"] is contract.c14_skills_body_delivered
    assert contract.ITEM_ORDER[-3:] == ["C13", "C13b", "C14"]


def test_the_rail_flag_gates_both_items_and_the_loader_flag_gates_only_c13b() -> None:
    """A framework with no rail must not be graded on activation; a framework
    with a rail but no loader must still be graded on C13, because forbidding a
    fabricated activation is exactly what a loader-less rail needs."""
    from tests.conformance.drivers import CAPABILITY_ITEMS, Capabilities

    assert "C13" in CAPABILITY_ITEMS["has_skills_rail"]
    assert "C13b" in CAPABILITY_ITEMS["has_skills_rail"]
    assert CAPABILITY_ITEMS["model_can_load_skill_bodies"] == ("C13b",)

    no_rail = Capabilities(has_skills_rail=False, reasons={"has_skills_rail": "none"})
    assert no_rail.na_reason("C13") == "none"
    assert no_rail.na_reason("C13b") == "none"

    no_loader = Capabilities(
        model_can_load_skill_bodies=False,
        reasons={"model_can_load_skill_bodies": "prompt injection only"},
    )
    assert no_loader.na_reason("C13") is None
    assert no_loader.na_reason("C13b") == "prompt injection only"


def test_turning_the_loader_flag_off_without_a_reason_is_refused() -> None:
    from tests.conformance.drivers import Capabilities

    with pytest.raises(ValueError):
        Capabilities(model_can_load_skill_bodies=False)


def test_every_rail_driver_that_declares_no_loader_says_why() -> None:
    """Every prompt-injection rail must print a reason, not skip silently.

    The set is pinned rather than derived on purpose: it is the one line a
    reviewer sees move when an adapter's activation story changes. `adk` joined
    on 2026-09-03 — not because it lost a loader, but because its rail started
    being GRADED at all (it had been declared `has_skills_rail=False` for five
    days after the rail shipped). What is absent there is the loader, which is
    the narrower and permanent fact this set records.
    """
    from tests.conformance.drivers import all_drivers

    off = {
        d.name: d.capabilities.reasons["model_can_load_skill_bodies"]
        for d in all_drivers()
        if not d.capabilities.model_can_load_skill_bodies
    }
    assert set(off) == {"langchain", "anthropic", "adk"}, off
    for name, reason in off.items():
        assert "prompt-injection" in reason, name
        assert "load_skill" in reason, name


# ── adjacent: TraceData.active_skills was declared and never assigned ────────


def test_trace_to_trace_data_fills_the_active_skills_field_it_declares() -> None:
    """``TraceData.active_skills`` is documented for custom evaluators to read
    (guides/evaluations.mdx) and was never assigned by ``trace_to_trace_data`` —
    an always-empty documented field, so every evaluator that branched on it
    took the "no skills" branch on a run where a skill was active."""
    from decimalai.evals import trace_to_trace_data

    payload = {
        "id": "11111111-1111-1111-1111-111111111111",
        "agent_name": "a",
        "active_skills": [{"name": ALPHA, "hash": "sha256:x"}, BETA, {"hash": "no-name"}],
        "skills_loaded_by_agent": ["conformance-skill-gamma"],
    }
    td = trace_to_trace_data(payload)
    assert td.active_skills == [ALPHA, BETA]
    # NOT unioned with skills_loaded_by_agent: a field called "active" must keep
    # meaning "somebody explicitly asserted this", which is the property C13's
    # evidence clause relies on to tell an assertion from an inference.
    assert "conformance-skill-gamma" not in td.active_skills


def test_trace_to_trace_data_active_skills_is_empty_when_the_payload_is() -> None:
    from decimalai.evals import trace_to_trace_data

    assert trace_to_trace_data({"id": "x", "agent_name": "a"}).active_skills == []


# ── D1: the delivery axis ────────────────────────────────────────────────────
#
# Same reason the C13 tests above exist, one axis over. Only two of the four rail
# drivers are installable in a bare checkout, so four of the eight delivery cells
# SKIP there — and a grading function whose only exercise is a cell that skips is
# a grading function nobody has run. These feed `grade_delivery` the shapes
# directly: the body on the right channel, on the wrong one, on neither, and a
# framework limit with and without its proof.

SENTINEL = "SENTINEL-SKILLBODY-ALPHA: opened boxes carry a 23.5% restocking fee."
SENTINEL_BODY = f"# Alpha\n\n{SENTINEL}\nOnly a delivered body carries this line."


def _delivery_driver(**limits: Any) -> Any:
    """A minimal real Driver — the contract reads its name and capabilities."""
    from tests.conformance.drivers import Capabilities, Driver

    noop = lambda *a, **k: None  # noqa: E731 - a hook that is never called
    return Driver(
        name="fake-rail",
        covers=frozenset(),
        requires=(),
        entrypoint="test",
        run=noop,
        run_concurrent=noop,
        run_error=noop,
        run_degenerate=noop,
        run_skills=noop,
        capabilities=Capabilities(delivery_limits=dict(limits)),
    )


def _delivery_obs(
    messages: List[Any],
    mode: str,
    *,
    logs: Optional[List[str]] = None,
    spans: Optional[List[Dict[str, Any]]] = None,
    driver: Any = None,
    lanes: int = 2,
) -> Observation:
    from tests.conformance.drivers import Ctx

    trace = {
        "routing_id": "rt_" + "0" * 24,
        "skills_offered_in_prompt": [ALPHA],
        "active_skills": [],
        "skills_loaded_by_agent": [],
        "skills_delivered": [],
        "llm_calls": [{"rendered_input": messages, "tool_calls": [], "output": "done"}],
        "spans": spans or [],
    }
    reqs = [
        Recorded(seq=i, method="POST", path="/api/v1/traces", query={}, body=trace,
                 status=200)
        for i in range(lanes)
    ]
    probe = Probe()
    probe.skills = [{"name": ALPHA, "description": "Alpha skill.", "body": SENTINEL_BODY}]
    phase = Phase(name="skills", ctxs=[], requests=reqs, logs=logs or [])
    ctx = Ctx(
        base_url="", api_key="", agent_name="a", prompt_sentinel="p",
        reply_sentinel="r", tool_name="t", tool_sentinel="ts", workdir="",
        delivery_mode=mode,
    )
    return Observation(
        driver=driver or _delivery_driver(), probe=probe, ctx=ctx,
        phases={"skills": phase},
    )


def _sys(text: str) -> Dict[str, Any]:
    return {"role": "system", "content": text}


def _tool(text: str) -> Dict[str, Any]:
    return {"role": "tool", "name": "load_skill", "content": text}


def test_injected_passes_when_the_body_is_in_the_system_prompt() -> None:
    from tests.conformance.delivery import INJECTED

    r = contract.grade_delivery(_delivery_obs([_sys(MENU + SENTINEL_BODY)], INJECTED))
    assert r.status == contract.PASS, r.message
    assert "prompt channel alone" in r.message
    assert "DECIMALAI_LOAD_SKILL_TOOL=0" in r.message


def test_injected_fails_when_only_the_MENU_reached_the_model() -> None:
    """The defect, in its purest form: titles, no body, no way to read one."""
    from tests.conformance.delivery import INJECTED

    r = contract.grade_delivery(_delivery_obs([_sys(MENU)], INJECTED))
    assert r.status == contract.FAIL
    assert "delivered NOTHING" in r.message
    # The message must name the configuration, because that is what a reader
    # needs to reproduce it.
    assert "DECIMALAI_INJECT_SKILL_BODY=1" in r.message


def test_injected_fails_when_the_body_arrives_by_the_switched_off_tool() -> None:
    """A kill switch the SDK ignores is worse than one it does not have."""
    from tests.conformance.delivery import INJECTED

    r = contract.grade_delivery(
        _delivery_obs([_sys(MENU), _tool(SENTINEL_BODY)], INJECTED)
    )
    assert r.status == contract.FAIL
    assert "switched OFF" in r.message


def test_tool_loaded_passes_on_a_tool_role_body() -> None:
    from tests.conformance.delivery import TOOL_LOADED

    r = contract.grade_delivery(
        _delivery_obs([_sys(MENU), _tool(SENTINEL_BODY)], TOOL_LOADED)
    )
    assert r.status == contract.PASS, r.message
    assert "tool channel alone" in r.message


def test_tool_loaded_passes_on_a_load_skill_span_alone() -> None:
    """openai-agents carries the body on a span as well as a message; either is
    the tool channel, and an item that hardcodes one breaks when an adapter
    legitimately changes it."""
    from tests.conformance.delivery import TOOL_LOADED

    span = {
        "id": "sp1", "span_type": "tool", "name": "load_skill",
        "input_preview": {"name": ALPHA},
        "output_preview": f"## Skill: {ALPHA}\n\n{SENTINEL_BODY}",
    }
    r = contract.grade_delivery(
        _delivery_obs([_sys(MENU)], TOOL_LOADED, spans=[span])
    )
    assert r.status == contract.PASS, r.message


def test_tool_loaded_fails_when_injection_leaked_through_anyway() -> None:
    from tests.conformance.delivery import TOOL_LOADED

    r = contract.grade_delivery(
        _delivery_obs([_sys(MENU + SENTINEL_BODY), _tool(SENTINEL_BODY)], TOOL_LOADED)
    )
    assert r.status == contract.FAIL
    assert "switched OFF" in r.message


def test_a_user_pasting_the_body_is_not_a_delivery_on_either_channel() -> None:
    """Neither channel, so it neither satisfies a cell nor trips the leak clause
    — the same rule C13 applies to activation evidence."""
    from tests.conformance.delivery import INJECTED, TOOL_LOADED

    msgs = [_sys(MENU), {"role": "user", "content": SENTINEL_BODY}]
    assert contract.grade_delivery(_delivery_obs(msgs, INJECTED)).status == contract.FAIL
    r = contract.grade_delivery(_delivery_obs(msgs, TOOL_LOADED))
    assert r.status == contract.FAIL
    assert "delivered NOTHING" in r.message


def test_grading_refuses_a_capture_taken_in_the_adapters_default_mode() -> None:
    """The cell means "this channel, alone". A default-mode capture cannot say
    that, so grading it would report a channel nobody switched on."""
    from tests.conformance.delivery import DEFAULT

    r = contract.grade_delivery(_delivery_obs([_sys(MENU + SENTINEL_BODY)], DEFAULT))
    assert r.status == contract.FAIL
    assert "DEFAULT mode" in r.message


def test_no_trace_at_all_fails_rather_than_passing_vacuously_on_delivery() -> None:
    from tests.conformance.delivery import INJECTED

    r = contract.grade_delivery(_delivery_obs([_sys(MENU)], INJECTED, lanes=0))
    assert r.status == contract.FAIL
    assert "no trace at all" in r.message


# ── D1: a framework limit must prove itself, on this run ─────────────────────

REFUSAL = "enable_load_skill_tool is not supported on the langchain adapter"


def _limited_driver() -> Any:
    from tests.conformance.delivery import TOOL_LOADED
    from tests.conformance.drivers import FrameworkLimit

    return _delivery_driver(**{
        TOOL_LOADED: FrameworkLimit(
            reason="an invoke-layer patch owns no tool loop.",
            adapter_module="decimalai/langchain.py",
            refusal_marker=REFUSAL,
        )
    })


def test_a_framework_limit_is_granted_only_with_the_adapters_own_refusal() -> None:
    from tests.conformance.delivery import TOOL_LOADED

    r = contract.grade_delivery(_delivery_obs(
        [_sys(MENU)], TOOL_LOADED,
        logs=[f"decimalai.langchain: {REFUSAL} (invoke-layer patch)"],
        driver=_limited_driver(),
    ))
    assert r.status == contract.NA
    assert "PROVEN ON THIS RUN" in r.message


def test_a_framework_limit_with_no_refusal_on_the_run_FAILS_rather_than_skips() -> None:
    """The whole point. `Capabilities.__post_init__` checks a reason EXISTS,
    never that it is TRUE — which is how C13b was N/A'd on langchain by a
    sentence that was the defect. A limit that stops being emitted stops being
    accepted."""
    from tests.conformance.delivery import TOOL_LOADED

    r = contract.grade_delivery(_delivery_obs(
        [_sys(MENU)], TOOL_LOADED, logs=["something else entirely"],
        driver=_limited_driver(),
    ))
    assert r.status == contract.FAIL
    assert "never said so on this run" in r.message


def test_a_framework_limit_fails_when_the_channel_quietly_worked_anyway() -> None:
    """Declared impossible, printed its refusal — and delivered by that channel.
    One of the two is lying, and a false limit is a permanent exemption."""
    from tests.conformance.delivery import TOOL_LOADED

    r = contract.grade_delivery(_delivery_obs(
        [_sys(MENU), _tool(SENTINEL_BODY)], TOOL_LOADED,
        logs=[f"decimalai.langchain: {REFUSAL}"],
        driver=_limited_driver(),
    ))
    assert r.status == contract.FAIL
    assert "reached the model by that very channel" in r.message


def test_a_limit_on_a_mode_that_does_not_exist_is_refused_at_construction() -> None:
    from tests.conformance.drivers import Capabilities, FrameworkLimit

    with pytest.raises(ValueError, match="not a delivery mode"):
        Capabilities(delivery_limits={
            "carrier-pigeon": FrameworkLimit(reason="r", adapter_module="m", refusal_marker="x"),
        })


def test_a_limit_declared_as_bare_PROSE_is_refused_at_construction() -> None:
    """Prose is what failed. A limit carries a file and a marker or it is not
    a limit."""
    from tests.conformance.delivery import TOOL_LOADED
    from tests.conformance.drivers import Capabilities

    with pytest.raises(ValueError, match="checkable proof"):
        Capabilities(delivery_limits={TOOL_LOADED: "no tool loop on this adapter"})


def test_a_framework_with_no_rail_declares_no_delivery_limits() -> None:
    """`has_skills_rail=False` already silences the whole rail; a per-mode limit
    on top would be a second exemption for the same hole, and two exemptions for
    one hole is how a ledger stops meaning anything."""
    from tests.conformance.delivery import DELIVERY_MODES
    from tests.conformance.drivers import Capabilities

    no_rail = Capabilities(has_skills_rail=False, reasons={"has_skills_rail": "none"})
    assert all(no_rail.delivery_limit(m) is None for m in DELIVERY_MODES)


def test_the_delivery_flag_gates_c14_and_the_loader_flag_still_only_gates_c13b() -> None:
    """The split. `model_can_load_skill_bodies` keeps its true, narrow meaning;
    C14 moves onto its own flag, which a guard then forbids turning off."""
    from tests.conformance.drivers import CAPABILITY_ITEMS

    assert CAPABILITY_ITEMS["model_can_load_skill_bodies"] == ("C13b",)
    assert CAPABILITY_ITEMS["rail_can_deliver_bodies"] == ("C14",)
    assert "C14" in CAPABILITY_ITEMS["has_skills_rail"]
    # Ordering is load-bearing: na_reason returns the FIRST False flag, so a
    # framework with no rail keeps printing its rail reason for C14.
    flags = list(CAPABILITY_ITEMS)
    assert flags.index("has_skills_rail") < flags.index("rail_can_deliver_bodies")
