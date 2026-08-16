"""Lock in three OTel-exporter fixes found by the CrewAI conformance column.

All three are exporter-side and therefore shared by every framework that rides
``decimalai.otel`` — crewai, generic-otel, anthropic and pydantic-ai:

1. **A blank attribute is not content.** ``_content_from_attrs`` matches
   attribute names by SUBSTRING, so CrewAI's ``crew_inputs`` (empty whenever a
   crew is started without an inputs dict — the common shape) matched the
   ``"input"`` pattern and returned ``""``. ``_assemble_trace`` tested
   ``is not None``, so it accepted the empty string as the trace's
   ``user_input_preview`` AND set ``root_set_input``, which suppressed the LLM
   fallback that had the real prompt. On the wire: ``user_input_preview: ""``.

2. **The ask is the last message, not the whole request truncated.** The
   fallback took ``_preview_from_attrs(attrs, "input")`` — every message joined
   with newlines, cut at 200 chars. A rendered chat request leads with the
   system preamble, so once that preamble passes the cap the preview is 100%
   system prompt and 0% user question. Measured on CrewAI: a 167-char preamble
   puts the user's ask at index 198, leaving two of its characters in the
   preview. ``decimalai.langchain`` already previews
   ``call.rendered_input[-1]["content"]`` for this same field; the OTel exporter
   was the outlier.

3. **Per-run state must be dropped on the exception path too.**
   ``_finalize_trace`` popped the run's skill rail only after ``_assemble_trace``
   succeeded, so a run whose assembly raised stranded its entry in the
   process-global ``_skill_rails`` store. That store is capped with oldest-first
   eviction, so the leak is bounded — but every stranded entry occupies a slot
   and evicts a LIVE run's rail early, losing that run's ``routing_id``.

The concurrency test is the one that reproduces what ``crewai:C9`` was actually
reporting. C9 was never an isolation defect: its message was eight copies of
"<lane>'s trace does not carry its OWN prompt", never the foreign-prompt branch.
It graded text taken only from ``llm_calls``, which arrived empty, so the
own-prompt clause could not be satisfied and the foreign-prompt clause was
vacuous. This test grades both clauses on POPULATED content, under real threads,
through a real ``BatchSpanProcessor`` — i.e. with the trace assembled on a
worker thread that never saw the caller's context.

Real opentelemetry-sdk (a core dependency); no backend — the client is mocked.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from decimalai.otel import (
    DecimalSpanExporter,
    _active_agent_name,
    _AgentNameStamper,
    _ask_from_rendered_input,
    _content_from_attrs,
    _preview_from_attrs,
    _reset_skill_rails,
    _skill_rails,
)

# A system preamble longer than the 200-char preview cap. CrewAI's real one is
# 167 chars and already pushed the ask to index 198 of the joined string; this
# one is deliberately past the cap so the defect is unambiguous rather than
# marginal.
LONG_SYSTEM = (
    "You are Conformance Fixture. You are a conformance fixture whose only job "
    "is to use the tool it is given and then answer. Your personal goal is to "
    "report the looked-up value verbatim, adding nothing and omitting nothing, "
    "and to keep going until the task is done."
)
assert len(LONG_SYSTEM) > 200, "the fixture must exceed the preview cap to test it"


@pytest.fixture(autouse=True)
def _sdk(monkeypatch):
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    saved_config, saved_client = cfg._config, cfg._client
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "m1", "status": "active"}
    _reset_skill_rails()
    yield
    _reset_skill_rails()
    cfg._config, cfg._client = saved_config, saved_client


# ── mock spans, for the assembly-level tests ─────────────────────────────────


class _Ctx:
    def __init__(self, trace_id, span_id):
        self.trace_id = trace_id
        self.span_id = span_id
        self.trace_flags = 1


class _Status:
    status_code = "OK"


class _Resource:
    attributes = {"service.name": "test-service"}


class _Span:
    def __init__(self, name, trace_id, span_id, parent_span_id=None, attributes=None):
        self.name = name
        self.context = _Ctx(trace_id, span_id)
        self.parent = _Ctx(trace_id, parent_span_id) if parent_span_id else None
        self.attributes = attributes or {}
        self.start_time = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.end_time = self.start_time + 100_000_000
        self.status = _Status()
        self.resource = _Resource()


def _assemble(spans):
    exporter = DecimalSpanExporter(agent_name="test-agent")
    result = exporter._assemble_trace(spans)
    assert result is not None
    return result[0]


def _crewai_shaped_spans(trace_id: int, sentinel: str) -> List[_Span]:
    """The exact span shape CrewAI 1.15 + the OpenInference instrumentors emit.

    Captured from a real ``crew.kickoff()`` against the conformance stub, not
    invented: a parentless ``…kickoff`` CHAIN span whose only input-ish
    attribute is an empty ``crew_inputs``, and ``ChatCompletion`` LLM spans from
    the OpenAI instrumentor carrying role-indexed messages and token counts.
    """
    return [
        _Span(
            "Crew_x.kickoff", trace_id, 0x01,
            attributes={
                "crew_inputs": "",
                "crew_id": "cid",
                "openinference.span.kind": "CHAIN",
                "output.mime_type": "application/json",
                "output.value": '{"raw": "REPLY-%s"}' % sentinel,
            },
        ),
        _Span(
            "ChatCompletion", trace_id, 0x02, parent_span_id=0x01,
            attributes={
                "llm.model_name": "conformance-stub-1",
                "llm.system": "openai",
                "llm.token_count.prompt": 17,
                "llm.token_count.completion": 5,
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.0.message.content": LONG_SYSTEM,
                "llm.input_messages.1.message.role": "user",
                "llm.input_messages.1.message.content": (
                    "\nCurrent Task: Please look up PROMPT-%s and report it." % sentinel
                ),
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "REPLY-%s" % sentinel,
            },
        ),
    ]


# ── 1. a blank attribute is not content ──────────────────────────────────────


def test_blank_attribute_is_not_reported_as_content():
    """``crew_inputs=''`` must read as "no input here", not as an empty input."""
    attrs = {"crew_inputs": "", "crew_id": "cid", "openinference.span.kind": "CHAIN"}
    assert _content_from_attrs(attrs, "input") is None
    assert _preview_from_attrs(attrs, "input") is None


def test_blank_attribute_does_not_shadow_a_real_one():
    """Scanning must CONTINUE past the blank, not stop at it.

    Same pattern (``"input"``) matches both keys; the empty one must not win
    just because it sorts first in the attribute dict.
    """
    attrs = {"crew_inputs": "   ", "input.value": "the real ask"}
    assert _content_from_attrs(attrs, "input") == "the real ask"


def test_blank_root_input_does_not_suppress_the_llm_fallback():
    """The whole point: an empty root attribute must not eat the real preview."""
    rt = _assemble(_crewai_shaped_spans(0xC0FFEE, "alpha"))
    assert rt.user_input_preview, "user_input_preview must not be empty or None"
    assert "PROMPT-alpha" in rt.user_input_preview, (
        f"the trace must carry the run's ask; got {rt.user_input_preview!r}"
    )


# ── 2. the ask is the last message, not the truncated join ───────────────────


def test_ask_helper_takes_the_user_message_not_the_preamble():
    rendered = [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "the actual question"},
    ]
    assert _ask_from_rendered_input(rendered) == "the actual question"


def test_ask_helper_takes_the_LAST_user_message_past_the_history():
    """Turn six of a conversation: the ask is the newest user turn, not the first.

    This is the clause that makes the rule more than "raise the truncation cap".
    A bigger cap still puts turn one at the front of the preview.
    """
    rendered = [
        {"role": "system", "content": LONG_SYSTEM},
        {"role": "user", "content": "the FIRST question"},
        {"role": "assistant", "content": "an answer"},
        {"role": "user", "content": "the CURRENT question"},
        {"role": "assistant", "content": None},
        {"role": "tool", "content": "a tool result nobody asked as a question"},
    ]
    assert _ask_from_rendered_input(rendered) == "the CURRENT question"


def test_ask_helper_falls_back_to_the_last_text_when_no_user_turn_exists():
    """Ported fallback: a request carrying only assistant/tool context."""
    rendered = [
        {"role": "assistant", "content": "context one"},
        {"role": "tool", "content": "context two"},
        {"role": "assistant", "content": ""},
    ]
    assert _ask_from_rendered_input(rendered) == "context two"


def test_ask_helper_skips_a_trailing_contentless_message():
    """A tool-call turn carries no text; the ask is the last message that does."""
    rendered = [
        {"role": "user", "content": "the actual question"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
    ]
    assert _ask_from_rendered_input(rendered) == "the actual question"


def test_ask_helper_has_nothing_to_say_about_nothing():
    assert _ask_from_rendered_input(None) is None
    assert _ask_from_rendered_input([]) is None
    assert _ask_from_rendered_input([{"role": "user", "content": "  "}]) is None


def test_long_system_preamble_does_not_push_the_ask_out_of_the_preview():
    """The regression that kept crewai:C4 red even once LLM spans existed.

    Joining every message and cutting at 200 chars means a system preamble
    longer than the cap IS the whole preview. Asserted on the sentinel rather
    than on the exact string, so this stays a statement about the ask surviving
    rather than about the cap's value.
    """
    spans = [
        _Span("agent-run", 0xA11CE, 0x01),
        _Span(
            "llm", 0xA11CE, 0x02, parent_span_id=0x01,
            attributes={
                "llm.model_name": "gpt-4o",
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.0.message.content": LONG_SYSTEM,
                "llm.input_messages.1.message.role": "user",
                "llm.input_messages.1.message.content": "SENTINEL-ASK please",
                "llm.output_messages.0.message.role": "assistant",
                "llm.output_messages.0.message.content": "SENTINEL-REPLY",
            },
        ),
    ]
    rt = _assemble(spans)
    assert "SENTINEL-ASK" in (rt.user_input_preview or ""), (
        f"the ask was truncated away by the system preamble; preview was "
        f"{rt.user_input_preview!r}"
    )


def test_a_span_with_no_messages_still_previews_its_prompt_content():
    """The fallback's fallback: keep working for spans carrying a bare string."""
    spans = [
        _Span("agent-run", 0xB0B, 0x01),
        _Span(
            "llm", 0xB0B, 0x02, parent_span_id=0x01,
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.input": "what is 2+2?",
                "gen_ai.output": "4",
            },
        ),
    ]
    rt = _assemble(spans)
    assert rt.user_input_preview == "what is 2+2?"
    assert rt.final_output_preview == "4"


# ── 3. per-run state is dropped on the exception path ────────────────────────


def test_a_failed_assembly_does_not_strand_its_skill_rail():
    """A run whose assembly raises must not leave its rail behind.

    The rail store is capped with oldest-first eviction, so a stranded entry is
    not an unbounded leak — it is worse than that in one specific way: it holds
    a slot and evicts a LIVE run's rail early, losing that run's routing_id.
    """
    tid = 0xDEAD
    _skill_rails[tid] = {
        "routing_id": "rt_x", "offered": [], "delivered": [], "loaded": [],
    }
    exporter = DecimalSpanExporter(agent_name="a")
    with patch.object(
        DecimalSpanExporter, "_assemble_trace", side_effect=RuntimeError("boom")
    ):
        exporter._finalize_trace(tid, [_Span("s", tid, 0x01)])
    assert tid not in _skill_rails, (
        "the failed run's rail is still in the process-global store"
    )


def test_a_trace_that_assembles_to_nothing_does_not_strand_its_rail():
    """The other silent path: ``_assemble_trace`` returning None, not raising."""
    tid = 0xBEEF
    _skill_rails[tid] = {
        "routing_id": "rt_y", "offered": [], "delivered": [], "loaded": [],
    }
    exporter = DecimalSpanExporter(agent_name="a")
    with patch.object(DecimalSpanExporter, "_assemble_trace", return_value=None):
        exporter._finalize_trace(tid, [_Span("s", tid, 0x01)])
    assert tid not in _skill_rails


# ── 4. concurrency: N lanes, N uncontaminated traces, on REAL spans ──────────


def test_concurrent_runs_each_carry_their_own_ask_and_no_other_lane_s():
    """What crewai:C9 grades, on populated content, through a worker thread.

    Real ``TracerProvider`` + real ``BatchSpanProcessor``, so the trace is
    assembled on the batch worker — the thread that never saw the caller's
    context, which is why the agent name has to ride on the span as an
    attribute rather than be read from a ContextVar at export time.

    Both of C9's clauses are asserted, and the foreign-prompt one is asserted
    over the WHOLE trace payload rather than just ``llm_calls``, which is a
    wider net than C9 itself casts.
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    lanes = [f"lane{i}" for i in range(8)]
    sent: List[Any] = []
    sent_lock = threading.Lock()

    exporter = DecimalSpanExporter(agent_name="fallback-should-not-be-used")

    def _capture(trace):
        with sent_lock:
            sent.append(trace)

    exporter._send = _capture  # type: ignore[method-assign]

    provider = TracerProvider()
    provider.add_span_processor(_AgentNameStamper())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    tracer = provider.get_tracer("conformance-unit")

    barrier = threading.Barrier(len(lanes))

    def _lane(name: str) -> None:
        # What decimalai.otel.instrument() does on the caller's thread.
        _active_agent_name.set(name)
        attrs_root = {
            "crew_inputs": "",  # the CrewAI shape that used to poison the preview
            "openinference.span.kind": "CHAIN",
            "output.value": '{"raw": "REPLY-%s"}' % name,
        }
        with tracer.start_as_current_span("Crew_x.kickoff", attributes=attrs_root):
            barrier.wait(timeout=30)  # force real overlap between the lanes
            with tracer.start_as_current_span(
                "ChatCompletion",
                attributes={
                    "llm.model_name": "conformance-stub-1",
                    "llm.system": "openai",
                    "llm.token_count.prompt": 17,
                    "llm.token_count.completion": 5,
                    "llm.input_messages.0.message.role": "system",
                    "llm.input_messages.0.message.content": LONG_SYSTEM,
                    "llm.input_messages.1.message.role": "user",
                    "llm.input_messages.1.message.content": (
                        "\nCurrent Task: Please look up PROMPT-%s and report it." % name
                    ),
                    "llm.output_messages.0.message.role": "assistant",
                    "llm.output_messages.0.message.content": "REPLY-%s" % name,
                },
            ):
                pass

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        for fut in [pool.submit(_lane, n) for n in lanes]:
            fut.result()
    provider.force_flush()

    assert len(sent) == len(lanes), f"{len(lanes)} runs produced {len(sent)} trace(s)"
    assert sorted(t.agent_name for t in sent) == sorted(lanes)

    problems: List[str] = []
    for trace in sent:
        name = trace.agent_name
        own = "PROMPT-%s" % name
        # C9's own-prompt clause, on the text C9 reads.
        rendered = "\n".join(
            str(m.get("content") or "")
            for call in trace.llm_calls
            for m in (call.rendered_input or [])
        )
        if own not in rendered:
            problems.append(f"{name}: rendered_input does not carry its OWN prompt")
        if own not in (trace.user_input_preview or ""):
            problems.append(
                f"{name}: user_input_preview does not carry its OWN ask "
                f"({trace.user_input_preview!r})"
            )
        # C9's foreign-prompt clause, cast over the ENTIRE payload.
        whole = repr(
            [
                trace.user_input_preview,
                trace.final_output_preview,
                [(c.rendered_input, c.output) for c in trace.llm_calls],
                [(s.input_preview, s.output_preview) for s in trace.spans],
            ]
        )
        foreign = [f"PROMPT-{other}" for other in lanes if other != name]
        leaked = [f for f in foreign if f in whole]
        if leaked:
            problems.append(f"{name}: trace carries another lane's prompt: {leaked}")
        # C3's clause, so this test also fails if the LLM detail goes away.
        if not trace.llm_calls:
            problems.append(f"{name}: no llm_calls")
        for call in trace.llm_calls:
            if not call.model_name:
                problems.append(f"{name}: llm_call has no model_name")
            for field in ("input_tokens", "output_tokens"):
                value = getattr(call, field)
                if not isinstance(value, int) or value <= 0:
                    problems.append(f"{name}: {field}={value!r}")

    assert not problems, "\n".join(problems)


def test_concurrent_runs_do_not_share_span_ids() -> None:
    """The other half of C9: no span may appear under two agents."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    lanes = [f"lane{i}" for i in range(8)]
    sent: List[Any] = []
    lock = threading.Lock()

    exporter = DecimalSpanExporter(agent_name="fallback")

    def _capture(trace):
        with lock:
            sent.append(trace)

    exporter._send = _capture  # type: ignore[method-assign]

    provider = TracerProvider()
    provider.add_span_processor(_AgentNameStamper())
    provider.add_span_processor(BatchSpanProcessor(exporter))
    tracer = provider.get_tracer("conformance-unit")
    barrier = threading.Barrier(len(lanes))

    def _lane(name: str) -> None:
        _active_agent_name.set(name)
        with tracer.start_as_current_span("root"):
            barrier.wait(timeout=30)
            with tracer.start_as_current_span(
                "llm", attributes={"llm.model_name": "m"}
            ):
                pass

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        for fut in [pool.submit(_lane, n) for n in lanes]:
            fut.result()
    provider.force_flush()

    owner: Dict[Any, str] = {}
    for trace in sent:
        for span in trace.spans:
            assert owner.setdefault(span.id, trace.agent_name) == trace.agent_name, (
                f"span {span.id} appears under two agents"
            )
    assert len(sent) == len(lanes)
