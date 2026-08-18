"""The prompt heuristic writes the DELIVERED/OFFERED rungs — never ACTIVATED —
and it stands down for any skill the router already accounted for on this run.

Two worlds share one trace:

* **Router-injected** skills. The SDK is the injector, so it OBSERVES what
  happened: ``consume_offered_names`` / ``consume_delivered_names`` /
  ``consume_loaded_names``. No inference is needed or wanted.
* **Harness / disk-discovered** skills. Claude Code and Cursor find SKILL.md on
  disk and inject it themselves, so the SDK never sees a selection event. The
  rendered prompt is the only observable, which is what
  ``infer_prompt_rungs`` reads.

``ctx._skills_registry`` is fed from DISK and the rails are fed by the ROUTER,
so both can be live in one run — a user inside Claude Code who also uses
SkillRouter. Skill names are not unique across those two sources, so a name
match cannot tell them apart. That is what the precedence rule is for.
"""

import pytest

from decimalai.skills import (
    EVIDENCE_BODY,
    EVIDENCE_NAME,
    detect_skill_activations,
    detect_skills_present_in_prompt,
    infer_prompt_rungs,
    _skill_appears_in_text,
)


# A body with three lines long enough for Tier-2 (>10 chars each).
BODY = (
    "Refund window is thirty calendar days from delivery.\n"
    "Partial refunds require a supervisor approval code.\n"
    "Always quote the original order identifier in the reply."
)

DISK_REGISTRY = [
    {"name": "refund-policy", "hash": "sha256:aa", "body": BODY},
    {"name": "tone-guide", "hash": "sha256:bb", "body": "House style is warm and brief always."},
]


def _sys(text):
    return [{"role": "system", "content": text}, {"role": "user", "content": "ping"}]


# ── which rung the evidence supports ─────────────────────────


class TestEvidenceTier:
    """Tier 1 (name) is an OFFER. Only Tier 2 (body overlap) is a DELIVERY."""

    def test_menu_row_is_name_evidence_only(self):
        got = detect_skills_present_in_prompt(
            _sys("Available skills:\n[refund-policy] how refunds work"), DISK_REGISTRY
        )
        assert got == [("refund-policy", EVIDENCE_NAME)]

    def test_body_in_prompt_is_body_evidence(self):
        got = detect_skills_present_in_prompt(
            _sys(f"You are helpful.\n\n{BODY}"), DISK_REGISTRY
        )
        assert got == [("refund-policy", EVIDENCE_BODY)]

    def test_header_plus_body_reports_BOTH_facts(self):
        """When both tiers match, both are true and both are reported.

        They are independent observations: the name in the prompt means the
        menu row was shown (OFFERED), the body in the prompt means the body was
        shown (DELIVERED). A SKILL.md pasted whole satisfies both, because its
        own heading carries the name.

        Picking a winner looked tidier and cost a true fact. Collapsing to BODY
        dropped the offered observation, and the caller compensated with a
        blanket delivered->offered fold — which also fired when the name was
        NOT in the prompt, asserting a menu row the model was never shown. That
        fold is gone; this reports the true half without inventing the false
        one.
        """
        got = detect_skills_present_in_prompt(
            _sys(f"## Skill: refund-policy\n{BODY}"), DISK_REGISTRY
        )
        assert sorted(got) == sorted([
            ("refund-policy", EVIDENCE_BODY),
            ("refund-policy", EVIDENCE_NAME),
        ])

    def test_a_body_without_its_name_is_delivered_but_NOT_offered(self):
        """The case the removed fold used to fabricate.

        A harness pastes a SKILL.md whose body never repeats the slug. The body
        reached the model (delivered), but no menu row ever did — so claiming
        `skills_offered_in_prompt` would assert something that did not happen.
        """
        body_without_name = "Always offer a refund within 30 days.\nEscalate disputes to a human."
        registry = [{"name": "refund-policy", "body": body_without_name}]
        offered, delivered = infer_prompt_rungs(
            [_sys(f"You are a support agent.\n{body_without_name}")], registry
        )
        assert delivered == ["refund-policy"]
        assert offered == [], (
            "a body with no menu row was reported as offered — that asserts the "
            "model was shown a menu row it never saw"
        )

    def test_negative_instruction_is_still_only_presence(self):
        """"Never use [X]" is an instruction NOT to — it can never be more
        than presence, and presence is the ceiling of what this module claims."""
        got = detect_skills_present_in_prompt(
            _sys("Never use [refund-policy] for this task."), DISK_REGISTRY
        )
        assert got == [("refund-policy", EVIDENCE_NAME)]

    def test_alias_returns_the_same_names_in_the_same_order(self):
        """``detect_skill_activations`` is kept only for backward
        compatibility; it must not drift from the function it now wraps."""
        prompt = _sys(f"## Skill: refund-policy\n{BODY}\n\n[tone-guide]")
        # Deduped: the wrapped function reports name- and body-evidence
        # independently, so one skill can appear twice. The alias has always
        # returned each name once, and must keep doing so.
        seen, expected = set(), []
        for name, _ in detect_skills_present_in_prompt(prompt, DISK_REGISTRY):
            if name not in seen:
                seen.add(name)
                expected.append(name)
        assert detect_skill_activations(prompt, DISK_REGISTRY) == expected
        assert detect_skill_activations(prompt, DISK_REGISTRY) == [
            "refund-policy",
            "tone-guide",
        ]


# ── the precedence rule ──────────────────────────────────────


class TestPrecedenceUnit:
    def test_router_accounted_name_is_not_inferred(self):
        offered, delivered = infer_prompt_rungs(
            [_sys(f"## Skill: refund-policy\n{BODY}")],
            DISK_REGISTRY,
            router_accounted={"refund-policy"},
        )
        assert (offered, delivered) == ([], [])

    def test_suppression_is_per_name_not_per_run(self):
        """A router-accounted skill must not silence the disk-only ones.

        The cheap gate — "this run has a routing_id, so skip inference" —
        would drop ``tone-guide`` here, and in a mixed Claude-Code-plus-router
        run that is most of the disk skills.
        """
        offered, delivered = infer_prompt_rungs(
            [_sys(f"## Skill: refund-policy\n{BODY}\n\n[tone-guide] house style")],
            DISK_REGISTRY,
            router_accounted={"refund-policy"},
        )
        assert offered == ["tone-guide"]
        assert delivered == []

    def test_no_router_means_nothing_is_suppressed(self):
        offered, delivered = infer_prompt_rungs(
            [_sys(f"## Skill: refund-policy\n{BODY}")], DISK_REGISTRY
        )
        assert (offered, delivered) == ([], ["refund-policy"])

    def test_body_on_one_call_outranks_a_name_on_another(self):
        offered, delivered = infer_prompt_rungs(
            [_sys("[refund-policy] how refunds work"), _sys(BODY)], DISK_REGISTRY
        )
        assert (offered, delivered) == ([], ["refund-policy"])


# ── the overlap case, end to end on a real TraceContext ──────


class TestOverlapOnGenericTrace:
    """One run, both worlds live: disk skills AND the router."""

    def _build(self, ctx):
        from unittest.mock import MagicMock, patch

        with patch("decimalai._config._get_config") as mock_get_config:
            cfg = MagicMock()
            cfg.project = "test"
            mock_get_config.return_value = cfg
            return ctx.build_trace()

    def test_router_delivered_body_is_not_re_reported_as_an_activation(self):
        """The measured defect: the router injects a body for a skill that also
        exists on disk, the heuristic matches the router's OWN injection, and
        the trace gains a fabricated activation."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="overlap")
        ctx._skills_registry = DISK_REGISTRY
        ctx.set_routing_id("rt_overlap01")
        # What the router observed directly.
        ctx.log_skill_offered(names=["refund-policy"])
        ctx.log_skill_delivered(names=["refund-policy"])
        # The router's own fragment, spliced into the prompt by the agent.
        ctx.log_llm_call(
            model="gpt-4o",
            input=_sys(
                "Available skills:\n- refund-policy: refund rules\n\n"
                f"## Skill: refund-policy\n\n{BODY}"
            ),
            output={"content": "ok"},
        )
        ctx._infer_skill_rungs_from_prompt()
        trace = self._build(ctx)

        assert trace.skills_offered_in_prompt == ["refund-policy"]
        assert trace.skills_delivered == ["refund-policy"]
        assert trace.skills_loaded_by_agent == []
        assert trace.active_skills == [], (
            "the model neither mentioned nor loaded the skill; the only "
            "evidence is the router's own injected body, which is DELIVERED"
        )

    def test_a_router_offered_skill_is_not_promoted_to_delivered(self):
        """THE case the precedence rule exists for.

        The router offered ``refund-policy`` as a menu row and never served
        its body. A same-named skill also sits on disk, and the harness put
        that disk body in the prompt. Name-matching cannot tell the two
        sources apart, so without the rule the disk file's body promotes the
        ROUTER's offered-only skill to delivered — and because the trace
        carries the router's ``routing_id``, that inflated delivery is joined
        straight onto the routing decision's offered denominator.

        Delete the ``router_accounted`` subtraction in ``infer_prompt_rungs``
        and this goes red.
        """
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="overlap")
        ctx._skills_registry = DISK_REGISTRY
        ctx.set_routing_id("rt_overlap02")
        # The router offered a menu row and served NO body.
        ctx.log_skill_offered(names=["refund-policy"])
        ctx.log_llm_call(
            model="gpt-4o",
            input=_sys(
                "Available skills:\n- refund-policy: refund rules\n\n"
                f"# Local project conventions\n{BODY}"
            ),
            output={"content": "ok"},
        )
        ctx._infer_skill_rungs_from_prompt()
        trace = self._build(ctx)

        assert trace.skills_offered_in_prompt == ["refund-policy"]
        assert trace.skills_delivered == [], (
            "a skill the router only OFFERED was promoted to delivered by "
            "prompt text that belongs to a same-named disk skill"
        )
        assert trace.active_skills == []

    def test_disk_only_skills_still_reach_the_wire_in_a_mixed_run(self):
        """Precedence must subtract per NAME, never disable the whole run."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="overlap")
        ctx._skills_registry = DISK_REGISTRY
        ctx.set_routing_id("rt_overlap03")
        ctx.log_skill_offered(names=["refund-policy"])
        ctx.log_skill_delivered(names=["refund-policy"])
        ctx.log_llm_call(
            model="gpt-4o",
            input=_sys(
                f"## Skill: refund-policy\n{BODY}\n\n"
                "[tone-guide] house style"
            ),
            output={"content": "ok"},
        )
        ctx._infer_skill_rungs_from_prompt()
        trace = self._build(ctx)

        assert trace.skills_offered_in_prompt == ["refund-policy", "tone-guide"]
        assert trace.skills_delivered == ["refund-policy"]
        assert trace.active_skills == []

    def test_a_load_skill_serve_is_the_activation_signal(self):
        """Activation is not empty because it is unreachable — it is empty
        because nothing asked. When the model DOES call ``load_skill``, the
        direct signal lands, and the inference stands down for that name."""
        from decimalai.generic import TraceContext

        ctx = TraceContext(agent_name="overlap")
        ctx._skills_registry = DISK_REGISTRY
        ctx.log_skill_loaded(name="refund-policy")
        ctx.log_llm_call(
            model="gpt-4o",
            input=_sys(f"## Skill: refund-policy\n{BODY}"),
            output={"content": "ok"},
        )
        ctx._infer_skill_rungs_from_prompt()
        trace = self._build(ctx)

        assert trace.skills_loaded_by_agent == ["refund-policy"]
        assert trace.skills_delivered == ["refund-policy"]
        assert trace.skills_offered_in_prompt == ["refund-policy"]


# ── the two shapes that mean the model chose ─────────────────


class TestModelInitiatedShapesAreInvisibleHere:
    """Structural proof that this module cannot fabricate an activation.

    ``_extract_system_text`` keeps system/developer roles only, so the two
    message shapes that carry a model-initiated choice are discarded before
    any matching runs. This is why retargeting the heuristic to the delivered
    rung is a rewiring and not a redefinition: what it reads is, by
    construction, "text that was put in front of the model".
    """

    @pytest.mark.parametrize(
        "message",
        [
            {"role": "assistant", "content": f"I'll use the [refund-policy] skill.\n{BODY}"},
            {"role": "tool", "name": "load_skill", "content": f"## Skill: refund-policy\n{BODY}"},
        ],
        ids=["assistant-says-so", "tool-result-carries-body"],
    )
    def test_model_initiated_messages_are_never_read(self, message):
        assert detect_skills_present_in_prompt([message], DISK_REGISTRY) == []
        assert infer_prompt_rungs([[message]], DISK_REGISTRY) == ([], [])


# ── the same rule, on each framework rail ────────────────────


class TestOpenAIAgentsRail:
    """The oai processor must merge its run rail BEFORE inferring.

    The inference used to run FIRST, under an earlier name and writing the
    activation rung, so the router-accounted set was empty when it read and
    precedence could not apply even in principle.
    """

    def _acc_with_prompt(self, text):
        from decimalai.openai_agents import _TraceAccumulator
        from decimalai.schema.trace import LlmCallRecord
        from decimalai.schema.common import Status
        from datetime import datetime, timezone

        acc = _TraceAccumulator(trace_id="t-precedence", trace_name="t")
        acc.llm_calls.append(
            LlmCallRecord(
                model_name="gpt-4o",
                rendered_input=_sys(text),
                status=Status.SUCCESS,
                started_at=datetime.now(timezone.utc),
            )
        )
        return acc

    def test_router_offered_name_is_not_promoted_to_delivered(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a", skills_registry=DISK_REGISTRY
        )
        acc = self._acc_with_prompt(f"# Local conventions\n{BODY}")
        acc.skills_offered_in_prompt.add("refund-policy")  # the router's own record

        processor._infer_skill_rungs(acc)

        assert acc.skills_delivered == set()
        assert acc.active_skills == {}

    def test_disk_only_skill_in_the_same_run_still_lands(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a", skills_registry=DISK_REGISTRY
        )
        acc = self._acc_with_prompt(f"# Local conventions\n{BODY}\n\n[tone-guide]")
        acc.skills_offered_in_prompt.add("refund-policy")

        processor._infer_skill_rungs(acc)

        assert "tone-guide" in acc.skills_offered_in_prompt
        assert acc.skills_delivered == set()

    def test_inference_runs_after_both_the_rail_merge_and_the_splice(self):
        """Ordering pin for ``_send_trace``. RETARGETED — the contract moved.

        Was ``test_inference_runs_after_the_rail_merge_and_before_the_splice``,
        asserting ``merge < infer < splice``. The splice/infer half has been
        deliberately REVERSED; the assertion now reads ``merge < splice <
        infer``.

        Why the old order existed: the inference used to write ACTIVATION, so
        splicing the skills menu in first would have reported every offered
        skill as activated. Why it could go: the inference writes
        offered/delivered only, and the precedence rule — which needs the rail
        merge, hence the half of this assertion that did NOT move — already
        subtracts the router's own names wherever in the prompt they sit.

        Why it had to go: on the Responses path ``ResponseSpanData`` carries
        the input items alone, so this splice is the only way the instructions
        ever reach ``rendered_input``. Running it last made a disk skill
        carried in the agent instructions invisible to the inference on the
        SDK's default path — an empty rung where there was a real value.

        Both halves still matter, for different reasons:
          * ``merge < infer`` — precedence needs the router-accounted set, or
            a skill the router only offered can be re-inferred as delivered.
          * ``splice < infer`` — the inference needs the whole prompt, or the
            instructions half is missing from the haystack.

        The behavioural half lives in ``test_openai_agents_system_prompt.py``
        and ``test_openai_agents_instructions_inference.py``, which own the
        span fixtures.
        """
        import inspect
        from decimalai.openai_agents import DecimalTracingProcessor

        src = inspect.getsource(DecimalTracingProcessor._send_trace)
        merge = src.index("acc.skills_offered_in_prompt.update(rail_offered)")
        infer = src.index("self._infer_skill_rungs(acc)")
        splice = src.index("_attach_system_prompts(acc,")
        assert merge < splice < infer, (
            "_send_trace order is rail-merge → splice → infer; got "
            f"merge@{merge} splice@{splice} infer@{infer}"
        )


class TestOTelRail:
    def test_rail_names_are_not_re_inferred(self):
        from decimalai.otel import DecimalSpanExporter
        from decimalai.schema.trace import RunTrace, LlmCallRecord

        exporter = DecimalSpanExporter(agent_name="a", skills=DISK_REGISTRY)
        trace = RunTrace(
            agent_name="a",
            llm_calls=[LlmCallRecord(model_name="gpt-4o",
                                     rendered_input=_sys(f"# Local\n{BODY}"))],
            skills_offered_in_prompt=["refund-policy"],
        )
        exporter._infer_skill_rungs(trace)

        assert trace.skills_delivered == []
        assert trace.active_skills == []

    def test_inference_unions_with_the_rail_rather_than_replacing_it(self):
        """The rail's own delivered names must survive the inference.

        ``_assemble_trace`` returns before the rail is popped, and the rail
        merge ASSIGNS ``skills_delivered``. Writing the inference on either
        side without unioning silently drops one of the two.
        """
        from decimalai.otel import DecimalSpanExporter
        from decimalai.schema.trace import RunTrace, LlmCallRecord

        exporter = DecimalSpanExporter(agent_name="a", skills=DISK_REGISTRY)
        trace = RunTrace(
            agent_name="a",
            llm_calls=[LlmCallRecord(model_name="gpt-4o",
                                     rendered_input=_sys("[tone-guide] house style"))],
            skills_offered_in_prompt=["router-only"],
            skills_delivered=["router-only"],
        )
        exporter._infer_skill_rungs(trace)

        assert trace.skills_delivered == ["router-only"]
        assert trace.skills_offered_in_prompt == ["router-only", "tone-guide"]

    def test_assemble_trace_writes_no_skill_rung_at_all(self):
        """Structural pin for the clobber hazard.

        Nothing may write a skill rung from inside ``_assemble_trace``: the
        rail merge that runs afterwards ASSIGNS ``skills_delivered``, so an
        earlier write is discarded on every run that had a rail.
        """
        import inspect
        from decimalai.otel import DecimalSpanExporter

        # Code only — the method's comments discuss these fields on purpose.
        code = "\n".join(
            line for line in inspect.getsource(
                DecimalSpanExporter._assemble_trace
            ).splitlines()
            if not line.lstrip().startswith("#")
        )
        for field in ("skills_delivered", "skills_offered_in_prompt",
                      "skills_loaded_by_agent"):
            assert field not in code, (
                f"_assemble_trace writes {field}, which the rail merge in "
                f"_finalize_trace overwrites — move it after the merge"
            )


class TestLangChainRail:
    def test_router_offered_name_is_not_promoted_to_delivered(self):
        import decimalai.langchain as lc_mod
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="test")
        state = handler._current
        state.seen_prompts["system"] = f"# Local conventions\n{BODY}"
        state.skills_offered_in_prompt.add("refund-policy")

        old = lc_mod._explicit_manifest_config
        lc_mod._explicit_manifest_config = {"skills": DISK_REGISTRY}
        try:
            handler._infer_skill_rungs_from_prompts(state)
        finally:
            lc_mod._explicit_manifest_config = old

        assert state.skills_delivered == set()
        assert state.active_skills == {}

    def test_disk_only_skill_in_the_same_run_still_lands(self):
        import decimalai.langchain as lc_mod
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="test")
        state = handler._current
        state.seen_prompts["system"] = f"# Local\n{BODY}\n\n[tone-guide] style"
        state.skills_offered_in_prompt.add("refund-policy")

        old = lc_mod._explicit_manifest_config
        lc_mod._explicit_manifest_config = {"skills": DISK_REGISTRY}
        try:
            handler._infer_skill_rungs_from_prompts(state)
        finally:
            lc_mod._explicit_manifest_config = old

        assert "tone-guide" in state.skills_offered_in_prompt
        assert state.skills_delivered == set()

    def test_inference_does_not_run_before_the_rails_are_merged(self):
        """``on_chat_model_start`` fires before ``_capture_call_rails`` has
        named what the router accounted for, so the inference must not run
        there. Pinned structurally — a behavioural pin would need a full
        LangChain model call."""
        import inspect
        from decimalai.langchain import CallbackHandler as _LC

        src = inspect.getsource(_LC.on_chat_model_start)
        assert "_infer_skill_rungs_from_prompts" not in src, (
            "the inference runs at on_chat_model_start, before the run's "
            "router-accounted set is complete — precedence cannot apply"
        )
        build = inspect.getsource(_LC.build_trace)
        assert "_infer_skill_rungs_from_prompts" in build


class TestPrefixNameLeak:
    """A disk skill whose name PREFIXES another must not match its block.

    The precedence rule subtracts router-accounted names exactly, so anything
    that matches under a *different* name slips straight past it — the SDK then
    infers a rung from its own injection. The Tier-1 patterns had no trailing
    boundary, so a disk skill named `refund` matched the router's block for
    `refund-policy`.

    This is the one correction in this area that no test caught when it was
    deliberately reverted, which is the only reason it is written down here.
    """

    ROUTER_BLOCK = "## Skill: refund-policy\n\nAlways offer a refund within 30 days."

    def test_a_prefix_name_does_not_match_a_longer_skills_block(self):
        assert _skill_appears_in_text("refund", self.ROUTER_BLOCK) is False

    def test_the_exact_name_still_matches(self):
        assert _skill_appears_in_text("refund-policy", self.ROUTER_BLOCK) is True

    def test_precedence_is_not_escaped_via_the_prefix(self):
        """End to end: the router accounted for refund-policy; a disk skill
        called `refund` must not be inferred off that same injected block."""
        registry = [{"name": "refund", "body": "Unrelated body text for refund."}]
        offered, delivered = infer_prompt_rungs(
            [_sys(self.ROUTER_BLOCK)],
            registry,
            router_accounted={"refund-policy"},
        )
        assert offered == []
        assert delivered == []

    def test_a_bare_heading_still_matches_its_own_name(self):
        assert _skill_appears_in_text("refund", "## refund\n\nsome text") is True
