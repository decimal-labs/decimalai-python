"""The harness inference must see the INSTRUCTIONS half of the prompt.

On the Agents SDK Responses path — its default — ``ResponseSpanData.__slots__``
is ``("response", "input", "usage")``. There is no instructions slot, so the
span carries the input items ALONE. The instructions reach ``rendered_input``
only through ``_attach_system_prompts``, from one of two sources:

* ``Response.instructions`` — the server's echo of what it was sent, and
* the run rail — the exact string ``Agent.get_system_prompt`` returned.

That splice used to run AFTER ``_infer_skill_rungs``, which meant a harness
that put a skill in its agent instructions — the ordinary way to use one on
this SDK — got an EMPTY offered/delivered rung on every trace. The text was on
the shipped record; the inference had simply already finished.

The fence existed for a real reason: the inference used to write ACTIVATION,
so splicing the skills menu in first would have reported every offered skill
as activated. That reason is gone. ``_infer_skill_rungs`` writes
``skills_offered_in_prompt`` and ``skills_delivered`` and nothing else, and the
router's own names are subtracted from it by the precedence rule in
``infer_prompt_rungs``. So the splice now runs BEFORE the inference.

These tests pin both halves, because moving the fence is only safe while both
hold:

1. a disk skill whose text lives only in the instructions lands on
   offered/delivered (``TestInstructionsAreVisibleToTheInference``), and
2. the router's OWN injected menu is still not re-inferred, and NOTHING —
   from either half of the prompt — ever reaches ``active_skills``
   (``TestRouterMenuIsStillNotReInferred``).

The second is the guard against reopening the defect the fence was built for.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest


# ── Synthetic SDK objects ───────────────────────────────────
# Plain attribute holders, not MagicMock: the handler's isinstance guards must
# see real str/list values or these tests would prove nothing about them.


class _MockSpanData:
    def __init__(self, span_type: str, **kwargs):
        self._type = span_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def type(self) -> str:
        return self._type


class _MockSpan:
    def __init__(self, trace_id, span_data, span_id=None, parent_id=None):
        self.trace_id = trace_id
        self.span_id = span_id or str(uuid4())
        self.parent_id = parent_id
        self.span_data = span_data
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.error = None


class _MockTrace:
    def __init__(self, trace_id: str, name: str = "instructions-workflow"):
        self.trace_id = trace_id
        self.name = name


class _OutputText:
    def __init__(self, text: str):
        self.type = "output_text"
        self.text = text


class _OutputMessage:
    def __init__(self, text: str):
        self.type = "message"
        self.role = "assistant"
        self.content = [_OutputText(text)]


class _Usage:
    def __init__(self, input_tokens: int = 11, output_tokens: int = 3):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens


class _SyntheticResponse:
    """What the ``openai`` client hands back. ``instructions`` is a real field
    on ``openai.types.responses.Response`` — the API's own echo."""

    def __init__(self, *, instructions=None, text: str = "done"):
        self.id = f"resp_{uuid4().hex[:16]}"
        self.model = "gpt-4o-mini-2024-07-18"
        self.usage = _Usage()
        self.output = [_OutputMessage(text)]
        self.temperature = None
        self.max_output_tokens = None
        self.instructions = instructions


# ── Fixtures / helpers ──────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_sdk():
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id",
        "status": "active",
    }

    import decimalai.openai_agents as oai
    from decimalai.schema.manifest import ManifestTracker

    oai._manifest_id = None
    oai._manifest_tracker = ManifestTracker()
    # Rails are process-global and keyed by trace id; a leftover from another
    # test would be exactly the cross-run contamination these tests assume away.
    with oai._run_rails_lock:
        oai._run_rails.clear()
    yield
    with oai._run_rails_lock:
        oai._run_rails.clear()


def _drive(processor, spans, trace_id):
    """Feed spans through the processor; return the ingested RunTrace."""
    import decimalai._config as cfg
    from decimalai._config import _sender

    trace = _MockTrace(trace_id=trace_id)
    processor.on_trace_start(trace)
    for span in spans:
        processor.on_span_end(span)
    processor.on_trace_end(trace)
    _sender.flush()
    cfg._client.ingest_trace.assert_called_once()
    return cfg._client.ingest_trace.call_args[0][0]


def _response_span(trace_id, *, instructions=None, user_text="how do refunds work?"):
    """A Responses-path span. Note what `input` carries: the USER turn only.

    That is the shape the real SDK emits — the instructions never appear here,
    which is the whole reason this file exists.
    """
    return _MockSpan(
        trace_id=trace_id,
        span_data=_MockSpanData(
            "response",
            response=_SyntheticResponse(instructions=instructions),
            input=[{"role": "user", "content": user_text}],
        ),
    )


def _record_rail(monkeypatch, trace_id, **kwargs):
    """Write onto the run rail through the REAL recording function.

    Goes through ``_record_run_rail`` rather than poking the dict, so the
    de-duplication and turn-ordering the fallback depends on are the ones
    under test, not a re-implementation of them.
    """
    import decimalai.openai_agents as oai

    monkeypatch.setattr(oai, "_current_run_key", lambda: trace_id)
    oai._record_run_rail(**kwargs)


def _system_text(trace) -> str:
    out = []
    for call in trace.llm_calls:
        for msg in call.rendered_input or []:
            if msg.get("role") in ("system", "developer"):
                out.append(str(msg.get("content") or ""))
    return "\n".join(out)


#: A body long enough for Tier-2 fuzzy matching to actually see it — the
#: detector needs >=60% of the skill's body LINES to appear in the prompt, and
#: it ignores lines under 10 characters.
DISK_BODY = (
    "Refund window is thirty calendar days from delivery.\n"
    "Partial refunds require a supervisor approval code.\n"
    "Always quote the original order identifier in the reply."
)

#: Two skills on purpose. ``refund-policy`` has a matchable body, so it can
#: demonstrate the DELIVERED rung; ``tone-guide``'s stub body has no line over
#: ten characters, so nothing can ever demonstrate its delivery and it can only
#: ever reach OFFERED. One trace then exercises both rungs.
DISK_REGISTRY = [
    {"name": "refund-policy", "description": "how refunds work", "body": DISK_BODY},
    {"name": "tone-guide", "description": "house style", "body": "y"},
]

#: What a harness that discovered skills on disk writes into its agent
#: instructions: one body pasted whole, one bare menu row.
HARNESS_INSTRUCTIONS = (
    "You are a support agent.\n\n"
    f"## Skill: refund-policy\n{DISK_BODY}\n\n"
    "[tone-guide]"
)


class TestInstructionsAreVisibleToTheInference:
    """A disk skill in the INSTRUCTIONS reaches offered/delivered.

    Both sources ``_attach_system_prompts`` accepts are covered, because they
    fail independently: the server echo dies when
    ``RunConfig.trace_include_sensitive_data=False`` strips
    ``span_data.response``, and the rail dies when there is no active trace to
    key it to.
    """

    def test_server_echoed_instructions_reach_both_rungs(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a", skills_registry=DISK_REGISTRY
        )
        tid = f"trace_{uuid4().hex[:16]}"
        trace = _drive(
            processor,
            [_response_span(tid, instructions=HARNESS_INSTRUCTIONS)],
            tid,
        )

        assert trace.skills_delivered == ["refund-policy"], (
            "a skill body that reached the model only through the agent "
            "instructions read as EMPTY — the inference ran before "
            "`_attach_system_prompts` put the instructions on rendered_input, "
            "so it name-matched over the user turn alone"
        )
        assert trace.skills_offered_in_prompt == ["tone-guide"], (
            "a menu row carried in the instructions read as EMPTY. "
            "`refund-policy` is deliberately NOT expected here: its body "
            "outranks its name (infer_prompt_rungs drops a name-only hit for "
            "a skill already on delivered), and there is no delivered->offered "
            "fold on the inferred path"
        )
        # The other half: the text really is on the shipped record. Without
        # this the assertions above could be satisfied by an inference that
        # invented the names instead of reading them.
        assert "Refund window is thirty calendar days" in _system_text(trace)
        assert trace.active_skills == [], (
            "prompt text reached the ACTIVATION rung. Nothing may write it "
            "but a direct selection event (the model calling load_skill)"
        )
        assert trace.skills_loaded_by_agent == [], (
            "prompt text reached skills_loaded_by_agent, which C13b grades as "
            "an activation"
        )

    def test_rail_only_instructions_reach_both_rungs(self, monkeypatch):
        """No server echo — the run rail is the only witness.

        This is the case ``RunConfig(trace_include_sensitive_data=False)``
        produces: the SDK stamps neither ``span_data.response`` nor (on the
        chat-completions path) ``span_data.input``, so the rail fallback is
        all that is left. It was a gap on BOTH paths at once.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a", skills_registry=DISK_REGISTRY
        )
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, tid, system_prompt=HARNESS_INSTRUCTIONS)
        trace = _drive(processor, [_response_span(tid, instructions=None)], tid)

        assert trace.skills_delivered == ["refund-policy"]
        assert trace.skills_offered_in_prompt == ["tone-guide"]
        assert "Refund window is thirty calendar days" in _system_text(trace)
        assert trace.active_skills == []
        assert trace.skills_loaded_by_agent == []

    def test_input_items_still_work(self):
        """The pre-existing path must not regress.

        Splicing the system message in EARLIER now means the detector sees two
        system messages on a call that has one of its own. The double-render
        guard in ``_attach_system_prompts`` is what keeps that from doubling
        the prompt, and it runs before the inference now.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a", skills_registry=DISK_REGISTRY
        )
        tid = f"trace_{uuid4().hex[:16]}"
        span = _MockSpan(
            trace_id=tid,
            span_data=_MockSpanData(
                "response",
                response=_SyntheticResponse(instructions=HARNESS_INSTRUCTIONS),
                input=[
                    {"role": "system", "content": HARNESS_INSTRUCTIONS},
                    {"role": "user", "content": "ping"},
                ],
            ),
        )
        trace = _drive(processor, [span], tid)

        assert trace.skills_delivered == ["refund-policy"]
        assert trace.skills_offered_in_prompt == ["tone-guide"]
        assert _system_text(trace).count("Refund window is thirty") == 1, (
            "the instructions were spliced in on top of a system message that "
            "already carried them — the double-render guard did not fire"
        )
        assert trace.active_skills == []


class TestRouterMenuIsStillNotReInferred:
    """The guard against reopening the defect the ordering fence was built for.

    The router injects its menu into the same instructions the inference now
    reads. Its names must come from its OWN observation (the run rail), never
    from a name-match over the text it wrote itself — and they must never climb
    a rung it did not observe.
    """

    #: The adversarial shape, not a comfortable one. The menu row's text is the
    #: disk skill's body verbatim, so Tier-2 line-overlap scores 100% and would
    #: promote a bare OFFER to DELIVERED. The blanket subtraction in
    #: ``infer_prompt_rungs`` is the only thing standing in the way. Names
    #: overlap between the router and disk by construction — `decimalai skills
    #: sync` uploads disk SKILL.md files under their own name — so this is the
    #: common case, not the edge.
    ROUTER_MENU = (
        "## Available Skills\n"
        "[refund-policy] how refunds work\n"
        f"{DISK_BODY}\n"
        "Call load_skill to pull a body."
    )

    def test_router_offered_menu_is_not_promoted_to_delivered(self, monkeypatch):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a",
            skills_registry=[
                {"name": "refund-policy", "description": "how refunds work",
                 "body": DISK_BODY},
            ],
        )
        tid = f"trace_{uuid4().hex[:16]}"
        # The router OFFERED the skill and served no body. Recorded the way the
        # instructions callable records it, on this run's rail.
        _record_rail(
            monkeypatch, tid, routing_id="rt_oai01", offered=["refund-policy"]
        )
        trace = _drive(
            processor, [_response_span(tid, instructions=self.ROUTER_MENU)], tid
        )

        assert trace.skills_delivered == [], (
            "the router's own menu row was re-inferred as DELIVERED off the "
            "text the router itself wrote. The rail says it served no body; a "
            "fabricated delivery here joins straight onto routing_id "
            "rt_oai01's offered denominator"
        )
        assert trace.skills_offered_in_prompt == ["refund-policy"], (
            "offered must come from the router's own rail observation"
        )
        assert trace.active_skills == [], (
            "the menu was promoted to ACTIVATED — the defect the ordering "
            "fence was originally built to prevent"
        )
        assert trace.skills_loaded_by_agent == [], (
            "the menu reached skills_loaded_by_agent, which C13b grades as an "
            "activation"
        )
        # The menu is still on the record — the splice happened, it just is
        # not evidence of anything the router did not itself report.
        assert "refund-policy" in _system_text(trace)

    def test_router_delivered_body_stays_delivered(self, monkeypatch):
        """A body the router DID serve is delivered, and stops there.

        The rung it must not reach is activation: reaching the model is
        delivery, and recording it as activated is indistinguishable
        downstream from the model actually asking for the body.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a",
            skills_registry=[
                {"name": "refund-policy", "description": "how refunds work",
                 "body": DISK_BODY},
            ],
        )
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(
            monkeypatch,
            tid,
            routing_id="rt_oai02",
            offered=["refund-policy"],
            delivered=["refund-policy"],
        )
        trace = _drive(
            processor,
            [_response_span(
                tid,
                instructions=f"## Skill: refund-policy\n{DISK_BODY}",
            )],
            tid,
        )

        assert trace.skills_delivered == ["refund-policy"]
        assert trace.skills_offered_in_prompt == ["refund-policy"]
        assert trace.active_skills == []
        assert trace.skills_loaded_by_agent == []

    def test_a_real_load_is_the_only_thing_that_reaches_activation(
        self, monkeypatch
    ):
        """The contrast case, so the assertions above are not vacuous.

        Every test in this class asserts an empty activation rung. That proves
        nothing unless something can fill it — and exactly one thing can: the
        ``loaded`` rail, written by ``_handle_load_skill`` when the router
        actually returned a body for a name the model asked for.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a",
            skills_registry=[
                {"name": "refund-policy", "description": "how refunds work",
                 "body": DISK_BODY},
            ],
        )
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(
            monkeypatch,
            tid,
            routing_id="rt_oai03",
            offered=["refund-policy"],
            loaded=["refund-policy"],
        )
        trace = _drive(
            processor, [_response_span(tid, instructions=self.ROUTER_MENU)], tid
        )

        assert trace.skills_loaded_by_agent == ["refund-policy"], (
            "a direct selection event is the ONE thing that fills the "
            "activation rung, and it did not"
        )
        assert trace.skills_delivered == ["refund-policy"]  # loaded implies delivered
