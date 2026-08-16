"""Values the OTel exporter puts on a trace must come from the span, not a default.

Each test here corresponds to a value that was being REPORTED without being
OBSERVED. That failure mode is worse than an empty field: downstream, a
defaulted value is indistinguishable from a real measurement, so it does not
read as missing data — it reads as data.

Every one of these was found by adversarial review of a change that had already
been declared green by its author and passed the whole conformance matrix.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from decimalai.otel import (
    DecimalSpanExporter,
    _ask_from_rendered_input,
    _content_from_attrs,
)
from decimalai.schema.common import FinishReason, Status



@pytest.fixture(autouse=True)
def _sdk_enabled():
    """A configured, mocked SDK — the exporter refuses to assemble without one."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    prev_config, prev_client = cfg._config, cfg._client
    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {
        "manifest_id": "test-manifest-id", "status": "active",
    }
    yield cfg
    cfg._config, cfg._client = prev_config, prev_client


class _Span:
    """The minimum ReadableSpan surface the exporter reads."""

    def __init__(
        self,
        name: str,
        attributes: Dict[str, Any],
        span_id: int = 1,
        parent_id: Optional[int] = None,
        trace_id: int = 0xABC,
    ):
        self.name = name
        self.attributes = attributes
        self.context = type("C", (), {"span_id": span_id, "trace_id": trace_id})()
        self.parent = (
            None if parent_id is None else type("P", (), {"span_id": parent_id})()
        )
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000
        self.status = type("S", (), {"status_code": "OK"})()
        self.kind = None
        self.events = []
        self.links = []
        self.resource = None
        self.instrumentation_scope = None


def _assemble(spans: List[Any]):
    exporter = DecimalSpanExporter(agent_name="t")
    assembled = exporter._assemble_trace(spans)
    assert assembled is not None, "the exporter produced no trace at all"
    return assembled[0]


# ── finish_reason ────────────────────────────────────────────────────────────


def test_finish_reason_comes_from_the_span_not_the_default():
    """A span saying `tool_calls` must not produce a record saying `stop`.

    The reader only knew the OTel semantic key. OpenInference — which is what
    CrewAI, raw OpenAI and raw Anthropic all go through — publishes
    `llm.finish_reason`, so every one of those calls fell through to the STOP
    default while its own span said otherwise.
    """
    trace = _assemble([
        _Span("ChatCompletion", {
            "llm.model_name": "gpt-4o",
            "llm.finish_reason": "tool_calls",
            "llm.token_count.prompt": 17,
            "llm.token_count.completion": 5,
        }),
    ])
    assert trace.llm_calls[0].finish_reason == FinishReason.TOOL_CALLS


def test_finish_reason_still_defaults_when_the_span_is_silent():
    """The other direction — absence of the attribute is not a regression."""
    trace = _assemble([
        _Span("ChatCompletion", {"llm.model_name": "gpt-4o"}),
    ])
    assert trace.llm_calls[0].finish_reason == FinishReason.STOP


# ── tool call outcomes ───────────────────────────────────────────────────────


def test_a_requested_tool_call_is_not_reported_as_succeeded():
    """These records are read off the LLM span — emitted when the model ASKS.

    The tool has not run at that point and the span carries no result, so
    `status: success` was asserting an outcome nobody observed.
    """
    trace = _assemble([
        _Span("ChatCompletion", {
            "llm.model_name": "gpt-4o",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "lookup",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments": '{"q": "x"}',
        }),
    ])
    (call,) = trace.llm_calls
    assert call.tool_calls, "the tool call was not extracted at all"
    assert call.tool_calls[0].status == Status.RUNNING, (
        f"an unexecuted tool call reported status={call.tool_calls[0].status}"
    )


# ── nested instrumentors ─────────────────────────────────────────────────────


def test_one_model_call_behind_two_instrumentors_is_counted_once():
    """CrewAI <=1.15 goes through LiteLLM, which calls the openai SDK.

    The documented install enables an instrumentor for both, so ONE completion
    emits two nested LLM spans carrying the same model and the same counts.
    Recorded naively that doubles every token count and every cost derived from
    them, and no contract item catches it because both records are valid.
    """
    common = {"llm.model_name": "gpt-4o",
              "llm.token_count.prompt": 17,
              "llm.token_count.completion": 5}
    trace = _assemble([
        _Span("completion", dict(common), span_id=1, parent_id=None),
        _Span("ChatCompletion", dict(common), span_id=2, parent_id=1),
    ])
    assert len(trace.llm_calls) == 1, (
        f"one model call was recorded {len(trace.llm_calls)} times — "
        "token counts and cost are doubled"
    )
    assert trace.llm_calls[0].input_tokens == 17
    assert trace.llm_calls[0].output_tokens == 5


def test_two_sibling_calls_are_still_two_calls():
    """The other direction: de-duplication must not swallow real calls."""
    common = {"llm.model_name": "gpt-4o", "llm.token_count.prompt": 3}
    trace = _assemble([
        _Span("agent", {}, span_id=1, parent_id=None),
        _Span("ChatCompletion", dict(common), span_id=2, parent_id=1),
        _Span("ChatCompletion", dict(common), span_id=3, parent_id=1),
    ])
    assert len(trace.llm_calls) == 2


# ── the run's ask ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("messages,expected,why", [
    ([{"role": "system", "content": "You are X"}, {"role": "user", "content": "ASK"}],
     "ASK", "the plain shape"),
    ([{"role": "system", "content": "You are X"}, {"role": "user", "content": "ASK"},
      {"role": "assistant", "content": ""}, {"role": "user", "content": "TOOL RESULT"}],
     "ASK", "Anthropic renders a tool RESULT as a user turn"),
    ([{"role": "user", "content": "old"}, {"role": "assistant", "content": "replied"},
      {"role": "user", "content": "ASK"}],
     "ASK", "a genuine follow-up in a conversation IS the current ask"),
])
def test_the_ask_is_the_question_not_the_tool_result(messages, expected, why):
    assert _ask_from_rendered_input(messages) == expected, why


# ── metadata is not content ──────────────────────────────────────────────────


def test_a_mime_type_is_never_reported_as_the_prompt():
    """The attribute scan matches key names by SUBSTRING, so `input.mime_type`
    matches the "input" pattern. A preview reading "application/json" cannot be
    told apart downstream from a model that was really shown that string."""
    assert _content_from_attrs(
        {"crew_inputs": "", "input.mime_type": "application/json"}, "input"
    ) is None


def test_a_real_value_beside_a_mime_type_still_wins():
    assert _content_from_attrs(
        {"input.mime_type": "application/json", "input.value": "the real prompt"},
        "input",
    ) == "the real prompt"
