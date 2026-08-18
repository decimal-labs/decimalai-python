"""The system half of the prompt on the OpenAI Agents Responses path.

``ResponseSpanData.__slots__`` is ``("response", "input", "usage")`` — there is
no ``instructions`` slot. So ``span_data.input`` is the input-items list ALONE,
and the instructions (where the skills menu is injected) were invisible to the
tracer. A trace could therefore truthfully claim ``skills_offered_in_prompt``
while ``rendered_input`` showed no such name anywhere, which is the shape
conformance item C8 fails on: *the record was incomplete, not the claim false*.

Two sources of truth, in preference order, and nothing else is ever allowed:

1. ``Response.instructions`` — the SERVER's echo of what it was sent. A
   round-trip receipt, and immune to a ``RunConfig.call_model_input_filter``
   replacing the instructions after our hook has already run.
2. the run rail — the exact string ``Agent.get_system_prompt`` returned.

Never reconstructed, never inferred, and never guessed onto a call whose owner
is ambiguous. The tests below pin all three of those refusals as hard as they
pin the happy path, because an over-eager version of this feature invents
prompt text, which is worse than recording none.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest


# ── Synthetic SDK objects ───────────────────────────────────
# Plain attribute holders, not MagicMock: the handler's isinstance guards must
# see real str/list values or the test would prove nothing about them.


class _MockSpanData:
    def __init__(self, span_type: str, **kwargs):
        self._type = span_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def type(self) -> str:
        return self._type


class _MockSpan:
    def __init__(self, trace_id, span_data, span_id=None, parent_id=None, error=None):
        self.trace_id = trace_id
        self.span_id = span_id or str(uuid4())
        self.parent_id = parent_id
        self.span_data = span_data
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.error = error


class _MockTrace:
    def __init__(self, trace_id: str, name: str = "sysprompt-workflow"):
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
    on ``openai.types.responses.Response`` — the API's own echo — and it admits
    ``str | list[ResponseInputItem] | None``."""

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
    # test would be exactly the cross-run contamination this file guards.
    with oai._run_rails_lock:
        oai._run_rails.clear()
    yield
    with oai._run_rails_lock:
        oai._run_rails.clear()


def _drive(processor, spans, trace_id=None):
    """Feed spans through the processor; return the ingested RunTrace."""
    import decimalai._config as cfg
    from decimalai._config import _sender

    trace_id = trace_id or f"trace_{uuid4().hex[:16]}"
    trace = _MockTrace(trace_id=trace_id)
    processor.on_trace_start(trace)
    for span in spans:
        processor.on_span_end(span)
    processor.on_trace_end(trace)
    _sender.flush()
    cfg._client.ingest_trace.assert_called_once()
    return cfg._client.ingest_trace.call_args[0][0]


def _response_span(trace_id, *, instructions=None, user_text="ping"):
    return _MockSpan(
        trace_id=trace_id,
        span_data=_MockSpanData(
            "response",
            response=_SyntheticResponse(instructions=instructions),
            input=[{"role": "user", "content": user_text}],
        ),
    )


def _record_rail(monkeypatch, trace_id, *prompts):
    """Write prompts onto the run rail through the REAL recording function.

    Goes through ``_record_run_rail`` rather than poking the dict, so the
    de-duplication and turn-ordering the fallback depends on are the ones
    under test, not a re-implementation of them.
    """
    import decimalai.openai_agents as oai

    monkeypatch.setattr(oai, "_current_run_key", lambda: trace_id)
    for p in prompts:
        oai._record_run_rail(system_prompt=p)


def _system_text(trace) -> str:
    out = []
    for call in trace.llm_calls:
        for msg in call.rendered_input or []:
            if msg.get("role") in ("system", "developer"):
                out.append(str(msg.get("content") or ""))
    return "\n".join(out)


MENU = (
    "You are a fixture.\n\n"
    "Available skills:\n"
    "- refund-policy: how refunds work\n"
    "- tone-guide: house style"
)


# ── The server's echo is the preferred evidence ─────────────


class TestServerEcho:
    def test_echoed_instructions_land_in_rendered_input(self):
        """The menu the model was shown becomes part of the record."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        trace = _drive(processor, [_response_span(tid, instructions=MENU)], tid)

        assert len(trace.llm_calls) == 1
        first = trace.llm_calls[0].rendered_input[0]
        assert first["role"] == "system"
        assert first["content"] == MENU
        # The user turn is still there, and still after the system message.
        assert trace.llm_calls[0].rendered_input[1]["content"] == "ping"

    def test_absent_echo_records_no_system_message(self):
        """No echo, no rail — the record stays honestly incomplete."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        trace = _drive(processor, [_response_span(tid, instructions=None)], tid)

        assert _system_text(trace) == ""

    def test_non_string_instructions_are_not_claimed(self):
        """``instructions`` also admits a list of input items.

        We cannot render that verbatim, so we decline to claim it rather than
        stringify a repr into the prompt record.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        listy = [{"role": "system", "content": MENU}]
        trace = _drive(processor, [_response_span(tid, instructions=listy)], tid)

        assert _system_text(trace) == ""
        rendered = str(trace.llm_calls[0].rendered_input)
        assert "refund-policy" not in rendered


# ── The rail is the fallback, and only where the echo is absent ──


class TestRailFallback:
    def test_rail_fills_in_when_the_server_does_not_echo(self, monkeypatch):
        """An OpenAI-compatible endpoint that echoes nothing still gets a
        record — from the exact string ``get_system_prompt`` returned."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, tid, MENU)
        trace = _drive(processor, [_response_span(tid, instructions=None)], tid)

        assert trace.llm_calls[0].rendered_input[0]["content"] == MENU

    def test_echo_wins_over_the_rail(self, monkeypatch):
        """When both exist, the round-trip receipt is the one recorded.

        This is the ``call_model_input_filter`` case: the runner may legally
        replace the instructions AFTER our hook returned, so the rail's copy
        can be stale in a way the server's echo never is.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, tid, "what our hook returned")
        trace = _drive(
            processor, [_response_span(tid, instructions="what the server got")], tid
        )

        assert _system_text(trace) == "what the server got"

    def test_one_prompt_covers_every_turn(self, monkeypatch):
        """A run whose prompt never changed de-dupes to one string, which is
        unambiguously every call's."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, tid, MENU, MENU)
        trace = _drive(
            processor,
            [_response_span(tid, user_text="one"), _response_span(tid, user_text="two")],
            tid,
        )

        assert len(trace.llm_calls) == 2
        assert all(c.rendered_input[0]["content"] == MENU for c in trace.llm_calls)

    def test_distinct_prompt_per_turn_zips_in_turn_order(self, monkeypatch):
        """``get_system_prompt`` runs once per turn, before that turn's model
        call, so rail order IS turn order when the counts line up."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, tid, "turn one prompt", "turn two prompt")
        trace = _drive(
            processor,
            [_response_span(tid, user_text="one"), _response_span(tid, user_text="two")],
            tid,
        )

        assert trace.llm_calls[0].rendered_input[0]["content"] == "turn one prompt"
        assert trace.llm_calls[1].rendered_input[0]["content"] == "turn two prompt"

    def test_ambiguous_mapping_attaches_nothing(self, monkeypatch):
        """Two distinct prompts, three calls: which prompt belongs to which
        call is a GUESS, so no prompt is recorded at all.

        This is the whole point of the feature. A trace that under-reports is
        a gap; a trace that reports the wrong prompt text is a fabrication,
        and every downstream read of ``rendered_input`` inherits it.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, tid, "prompt A", "prompt B")
        trace = _drive(
            processor,
            [
                _response_span(tid, user_text="one"),
                _response_span(tid, user_text="two"),
                _response_span(tid, user_text="three"),
            ],
            tid,
        )

        assert len(trace.llm_calls) == 3
        assert _system_text(trace) == ""

    def test_a_concurrent_run_does_not_inherit_another_rail(self, monkeypatch):
        """Rails are keyed by trace id, so the run that recorded nothing
        records nothing — it does not pick up the neighbour's prompt."""
        import decimalai._config as cfg
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        first, second = f"trace_{uuid4().hex[:16]}", f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, first, MENU)
        _drive(processor, [_response_span(first)], first)

        cfg._client.ingest_trace.reset_mock()
        trace2 = _drive(processor, [_response_span(second)], second)
        assert _system_text(trace2) == ""

    def test_the_rail_is_consumed_not_reused(self, monkeypatch):
        """``_send_trace`` POPS the rail.

        A peek would leave the prompt behind for whatever arrives under that
        trace id next — the SDK's ids are not guaranteed unique for all time,
        and a rail that is never consumed also never gets collected. Driving
        the same id twice is the only shape that separates pop from peek.
        """
        import decimalai._config as cfg
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, tid, MENU)
        first = _drive(processor, [_response_span(tid)], tid)
        assert _system_text(first) == MENU

        cfg._client.ingest_trace.reset_mock()
        second = _drive(processor, [_response_span(tid)], tid)
        assert _system_text(second) == "", (
            "the rail survived its own run — a later trace inherited a system "
            "prompt that was never resolved for it"
        )


# ── Menu text is an OFFER, never an ACTIVATION ──────────────


class TestMenuIsNotAnActivation:
    """The prompt-presence detector name-matches over SYSTEM text.

    So the splice must happen AFTER the inference runs, or the menu — which
    names every skill that was merely OFFERED — would have its whole contents
    promoted a rung. Before the rewiring that rung was ACTIVATED and the
    offered→activated join would have read 100% every time; now the inference
    writes offered/delivered, so a mis-ordered splice would inflate DELIVERED
    instead. A smaller lie, the same kind, and the ordering still costs
    nothing.

    Measured caveat, so nobody over-claims what this guard buys: the router's
    CURRENT fragment format (``- name: description``) does not match the
    detector's Tier-1 patterns, so today the bug would not fire. The formats
    below (``[name]``, ``## name``) DO match, they are a server-side rendering
    choice that can change without an SDK release. This test pins the
    ordering, not a live incident.
    """

    REGISTRY = [
        {"name": "refund-policy", "description": "how refunds work", "body": "x"},
        {"name": "tone-guide", "description": "house style", "body": "y"},
    ]

    #: A body long enough for Tier-2 fuzzy matching to actually see it. The
    #: stub ``"x"`` above has no line over 10 chars, so no prompt can ever
    #: demonstrate its delivery — which is the point of keeping both.
    REAL_BODY = (
        "Refund window is thirty calendar days from delivery.\n"
        "Partial refunds require a supervisor approval code.\n"
        "Always quote the original order identifier in the reply."
    )

    def test_bracketed_menu_names_are_offered_not_activated(self):
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a", skills_registry=self.REGISTRY
        )
        tid = f"trace_{uuid4().hex[:16]}"
        menu = "Available skills:\n[refund-policy] how refunds work\n[tone-guide] style"
        trace = _drive(processor, [_response_span(tid, instructions=menu)], tid)

        assert trace.active_skills == [], (
            "an offered-but-unused skill was recorded as ACTIVATED — the menu "
            "was spliced into rendered_input before the inference ran"
        )
        assert trace.skills_delivered == [], (
            "a menu row was recorded as DELIVERED — no body appears anywhere "
            "in this prompt, so nothing observed one reaching the model"
        )
        assert trace.skills_offered_in_prompt == [], (
            "the menu was re-read back off the record. `_attach_system_prompts` "
            "RECONSTRUCTS the prompt from the rail, so inferring rungs from it "
            "is the SDK reading its own bookkeeping and calling it evidence — "
            "the router already reported offered directly through the rail"
        )
        # ...and the menu is still on the record, which is the other half.
        assert "refund-policy" in _system_text(trace)

    def test_a_delivered_body_lands_on_delivered_not_activated(self):
        """A body the router injected is DELIVERED. It is not an activation.

        Was ``test_a_delivered_body_still_counts_as_activated``. Its own
        docstring convicted it: "that genuinely reached the model". Reaching
        the model is the delivered rung. Recording it as activated is
        indistinguishable downstream from the model actually asking for the
        body, which is what C13 exists to forbid (contract.py:704-721).

        The fixture body was lengthened from the ``"x"`` stub so Tier-2 can
        genuinely match it — otherwise the case would prove nothing about
        delivery, only about a name header.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a",
            skills_registry=[
                {"name": "refund-policy", "description": "how refunds work",
                 "body": self.REAL_BODY},
                {"name": "tone-guide", "description": "house style", "body": "y"},
            ],
        )
        tid = f"trace_{uuid4().hex[:16]}"
        span = _MockSpan(
            trace_id=tid,
            span_data=_MockSpanData(
                "response",
                response=_SyntheticResponse(instructions="Available skills:\n- x: y"),
                input=[
                    {"role": "system",
                     "content": f"## Skill: refund-policy\n{self.REAL_BODY}"},
                    {"role": "user", "content": "ping"},
                ],
            ),
        )
        trace = _drive(processor, [span], tid)

        assert trace.skills_delivered == ["refund-policy"]
        # NOT implied. This fixture's body never repeats its own slug, so no
        # menu row reached the model — and `skills_offered_in_prompt` means
        # exactly "the menu row was in the prompt the model was shown". The
        # blanket delivered->offered fold that used to make this pass asserted a
        # menu row that was never there; it is gone.
        assert "refund-policy" not in trace.skills_offered_in_prompt
        assert trace.active_skills == [], (
            "a body the router injected reached the model — that is DELIVERED. "
            "Recording it as activated inflates router_activated_count and "
            "promotes a pasted skill over one that was actually used."
        )

    def test_a_rail_offered_skill_is_not_promoted_by_disk_prompt_text(
        self, monkeypatch
    ):
        """The run rail must be merged BEFORE the inference runs.

        The rail records that the router only OFFERED ``refund-policy`` — it
        served no body. A same-named skill also sits on disk, and its body is
        in the prompt. Skill names are not unique across those two sources, so
        without the precedence rule the disk file promotes the ROUTER's
        offered-only skill to delivered — and this trace carries the router's
        ``routing_id``, so that inflated delivery is joined straight onto the
        routing decision's offered denominator.

        Ordering matters as much as the rule: run the inference before the
        rail merge (where ``_detect_skills`` used to sit) and the
        router-accounted set is empty, so precedence cannot apply at all.
        """
        import decimalai.openai_agents as oai
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(
            agent_name="a",
            skills_registry=[
                {"name": "refund-policy", "description": "how refunds work",
                 "body": self.REAL_BODY},
            ],
        )
        tid = f"trace_{uuid4().hex[:16]}"
        monkeypatch.setattr(oai, "_current_run_key", lambda: tid)
        oai._record_run_rail(routing_id="rt_oai01", offered=["refund-policy"])

        span = _MockSpan(
            trace_id=tid,
            span_data=_MockSpanData(
                "response",
                response=_SyntheticResponse(instructions=None),
                input=[
                    {"role": "system",
                     "content": f"# Local project conventions\n{self.REAL_BODY}"},
                    {"role": "user", "content": "ping"},
                ],
            ),
        )
        trace = _drive(processor, [span], tid)

        assert trace.routing_id == "rt_oai01"
        assert trace.skills_offered_in_prompt == ["refund-policy"]
        assert trace.skills_delivered == [], (
            "the router only OFFERED this skill; prompt text belonging to a "
            "same-named DISK skill promoted it to delivered"
        )
        assert trace.active_skills == []


# ── No double-rendering of a prompt already on the record ───


class TestNoDuplication:
    def test_existing_system_message_is_not_prepended_again(self, monkeypatch):
        """The chat-completions path renders the system message itself.

        Its ``generation`` span input already carries the prompt, so the rail
        fallback must recognise it rather than double the text.
        """
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="a")
        tid = f"trace_{uuid4().hex[:16]}"
        _record_rail(monkeypatch, tid, MENU)
        span = _MockSpan(
            trace_id=tid,
            span_data=_MockSpanData(
                "generation",
                model="gpt-4o-mini",
                model_config={},
                usage={"input_tokens": 5, "output_tokens": 2},
                input=[
                    {"role": "system", "content": MENU},
                    {"role": "user", "content": "ping"},
                ],
                output=[{"role": "assistant", "content": "done"}],
            ),
        )
        trace = _drive(processor, [span], tid)

        rendered = trace.llm_calls[0].rendered_input
        systems = [m for m in rendered if m.get("role") == "system"]
        assert len(systems) == 1
        assert systems[0]["content"] == MENU
