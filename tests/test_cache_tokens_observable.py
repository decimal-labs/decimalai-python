"""Prompt-cache tokens are captured, kept split, and reach the trace payload.

WHY
---
DecimalAI injects a query-routed skill menu at position ZERO of the system
prompt, rebuilt on every query. Providers cache on a stable PREFIX (OpenAI
auto-caches above a token floor, Anthropic matches up to each `cache_control`
breakpoint), so varying bytes at position zero defeat the cache for EVERYTHING
behind them: a customer's 2,000-token system prompt that would have been a hit
becomes a full miss. The cost is not the ~115 tokens of menu, it is the 2,000
behind it.

That regression was unmeasurable. Nothing in the SDK carried a cache count, and
the one place that parsed them destroyed the split
(`claude_agent_sdk._extract_usage` summed cache_read + cache_creation into
`input_tokens`). These tests pin the three properties that make it measurable:

  1. the counts are CAPTURED wherever provider usage is parsed;
  2. they stay SPLIT — never folded into `input_tokens`, because a sum cannot
     distinguish "180k cached + 4k fresh" from "184k fresh";
  3. None and 0 stay DIFFERENT — "the provider never told us" vs "the provider
     told us, and the cache was cold".

Covers the OTEL exporter (the path raw OpenAI/Anthropic SDK calls take via
`decimalai.providers.instrument()`) and the manual `decimalai.trace` tracer.
The Claude Agent SDK adapter is covered in
tests/test_claude_agent_sdk_handler.py.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest


class _MockContext:
    def __init__(self, trace_id, span_id):
        self.trace_id = trace_id
        self.span_id = span_id
        self.trace_flags = 1


class _MockStatus:
    def __init__(self, code="OK"):
        self.status_code = code


class _MockSpan:
    def __init__(self, name, trace_id, span_id, parent_span_id=None, attributes=None):
        self.name = name
        self.context = _MockContext(trace_id, span_id)
        self.parent = _MockContext(trace_id, parent_span_id) if parent_span_id else None
        self.attributes = attributes or {}
        self.start_time = int(datetime.now(timezone.utc).timestamp() * 1e9)
        self.end_time = self.start_time + 100_000_000
        self.status = _MockStatus("OK")
        self.resource = None


@pytest.fixture(autouse=True)
def _reset_sdk():
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "m", "status": "active"}
    yield


def _assemble(spans):
    from decimalai.otel import DecimalSpanExporter

    exporter = DecimalSpanExporter(agent_name="test-agent")
    result = exporter._assemble_trace(spans)
    assert result is not None
    run_trace, _seen_model, _seen_tools, _seen_prompts = result
    return run_trace


# ── OTEL exporter: OpenInference (raw provider SDKs) ─────────────────────────

def test_openinference_cache_details_are_captured():
    """`decimalai.init(anthropic=True)` routes through here.

    OpenInference names Anthropic's cache CREATION `cache_write`, which reads
    like a second kind of read and is easy to drop on the floor.
    """
    spans = [
        _MockSpan(
            "AnthropicMessages", 0xB1, 0x01,
            attributes={
                "llm.model_name": "claude-sonnet-4-5",
                "llm.provider": "anthropic",
                "llm.token_count.prompt": 4_000,
                "llm.token_count.completion": 900,
                "llm.token_count.prompt_details.cache_read": 180_000,
                "llm.token_count.prompt_details.cache_write": 16_000,
            },
        ),
    ]
    call = _assemble(spans).llm_calls[0]

    assert call.input_tokens == 4_000          # NOT 200_000 — no folding
    assert call.output_tokens == 900
    assert call.cache_read_tokens == 180_000
    assert call.cache_creation_tokens == 16_000


def test_genai_semconv_cache_keys_are_captured():
    """OTel-native instrumentations use the gen_ai.* spelling instead."""
    spans = [
        _MockSpan(
            "chat claude-sonnet-4-5", 0xB2, 0x01,
            attributes={
                "gen_ai.system": "anthropic",
                "gen_ai.request.model": "claude-sonnet-4-5",
                "gen_ai.usage.input_tokens": 500,
                "gen_ai.usage.output_tokens": 60,
                "gen_ai.usage.cache_read_input_tokens": 12_000,
                "gen_ai.usage.cache_creation_input_tokens": 0,
            },
        ),
    ]
    call = _assemble(spans).llm_calls[0]

    assert call.input_tokens == 500
    assert call.cache_read_tokens == 12_000
    assert call.cache_creation_tokens == 0


def test_openai_cached_tokens_are_a_subset_and_stay_that_way():
    """OpenAI's `prompt_tokens_details.cached_tokens` is ALREADY inside
    `prompt_tokens`. Storing it verbatim (rather than normalising to
    Anthropic's additive convention) is what keeps
    `input_tokens + cache_read_tokens` from double-counting on this provider.
    """
    spans = [
        _MockSpan(
            "ChatCompletion", 0xB3, 0x01,
            attributes={
                "llm.model_name": "gpt-5.4-mini",
                "llm.provider": "openai",
                "llm.token_count.prompt": 2_115,          # includes the cached part
                "llm.token_count.completion": 300,
                "llm.token_count.prompt_details.cache_read": 2_048,
            },
        ),
    ]
    call = _assemble(spans).llm_calls[0]

    assert call.input_tokens == 2_115
    assert call.cache_read_tokens == 2_048
    assert call.cache_read_tokens < call.input_tokens   # subset, not additive
    # OpenAI's auto-cache reports no creation step — absent, not zero.
    assert call.cache_creation_tokens is None


def test_no_cache_attributes_leaves_both_none():
    """An instrumentor that reports no cache detail must not read as a MISS.

    `0` here would be a fabricated measurement, and would make every
    pre-existing integration look like it had a permanently cold cache.
    """
    spans = [
        _MockSpan(
            "chat gpt-4o", 0xB4, 0x01,
            attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 42,
                "gen_ai.usage.output_tokens": 7,
            },
        ),
    ]
    call = _assemble(spans).llm_calls[0]

    assert call.input_tokens == 42
    assert call.cache_read_tokens is None
    assert call.cache_creation_tokens is None


def test_reported_zero_survives_as_zero():
    """The other half of the distinction — a measured cold cache is data.

    This is the value a cache-defeating change actually produces, so a
    truthiness test anywhere on the path (`attrs.get(k) or None`) would erase
    the regression in precisely the case it exists to catch.
    """
    spans = [
        _MockSpan(
            "AnthropicMessages", 0xB5, 0x01,
            attributes={
                "llm.model_name": "claude-sonnet-4-5",
                "llm.token_count.prompt": 184_000,
                "llm.token_count.completion": 900,
                "llm.token_count.prompt_details.cache_read": 0,
                "llm.token_count.prompt_details.cache_write": 0,
            },
        ),
    ]
    call = _assemble(spans).llm_calls[0]

    assert call.cache_read_tokens == 0
    assert call.cache_creation_tokens == 0
    assert call.cache_read_tokens is not None


def test_warm_and_cold_runs_are_distinguishable():
    """The property the whole change exists for.

    Same effective context, opposite cache outcomes. A single folded input
    number reports these identically; the split does not.
    """
    warm = _assemble([
        _MockSpan("AnthropicMessages", 0xB6, 0x01, attributes={
            "llm.model_name": "claude-sonnet-4-5",
            "llm.token_count.prompt": 4_000,
            "llm.token_count.completion": 900,
            "llm.token_count.prompt_details.cache_read": 180_000,
            "llm.token_count.prompt_details.cache_write": 0,
        }),
    ]).llm_calls[0]
    cold = _assemble([
        _MockSpan("AnthropicMessages", 0xB7, 0x01, attributes={
            "llm.model_name": "claude-sonnet-4-5",
            "llm.token_count.prompt": 184_000,
            "llm.token_count.completion": 900,
            "llm.token_count.prompt_details.cache_read": 0,
            "llm.token_count.prompt_details.cache_write": 0,
        }),
    ]).llm_calls[0]

    folded_warm = warm.input_tokens + (warm.cache_read_tokens or 0)
    folded_cold = cold.input_tokens + (cold.cache_read_tokens or 0)
    assert folded_warm == folded_cold                  # what the old code saw
    assert warm.cache_read_tokens != cold.cache_read_tokens   # what it now sees


# ── Manual tracer (decimalai.trace / generic.py) ──────────────────────────────

def test_generic_tracer_accepts_and_forwards_the_split():
    from decimalai.generic import TraceContext

    ctx = TraceContext(agent_name="manual-agent")
    ctx.log_llm_call(
        model="claude-sonnet-4-5",
        provider="anthropic",
        input_tokens=4_000,
        output_tokens=900,
        cache_read_tokens=180_000,
        cache_creation_tokens=16_000,
    )
    call = ctx._llm_calls[0]

    assert call.input_tokens == 4_000
    assert call.cache_read_tokens == 180_000
    assert call.cache_creation_tokens == 16_000


def test_generic_tracer_defaults_to_unknown_not_zero():
    """Callers who do not pass the split get NULL, not a fabricated miss."""
    from decimalai.generic import TraceContext

    ctx = TraceContext(agent_name="manual-agent")
    ctx.log_llm_call(model="gpt-5.4-mini", input_tokens=100, output_tokens=10)
    call = ctx._llm_calls[0]

    assert call.cache_read_tokens is None
    assert call.cache_creation_tokens is None


# ── The wire ─────────────────────────────────────────────────────────────────

def test_split_is_on_the_serialized_trace_payload():
    """`_client.ingest_trace` sends `trace.model_dump(mode="json")`.

    If the keys are not in that dump, the platform's new nullable columns
    receive nothing and the whole chain is decorative.
    """
    from decimalai.schema.trace import LlmCallRecord, RunTrace

    payload = RunTrace(
        agent_name="wire-agent",
        llm_calls=[
            LlmCallRecord(
                model_name="claude-sonnet-4-5",
                input_tokens=4_000, output_tokens=900,
                cache_read_tokens=180_000, cache_creation_tokens=0,
            ),
            LlmCallRecord(model_name="gpt-4o", input_tokens=10),
        ],
    ).model_dump(mode="json")

    reported, silent = payload["llm_calls"]
    assert reported["cache_read_tokens"] == 180_000
    assert reported["cache_creation_tokens"] == 0      # measured miss
    assert silent["cache_read_tokens"] is None         # never measured
    assert silent["cache_creation_tokens"] is None


# ── Client-side evals see the split too ──────────────────────────────────────

def test_eval_trace_view_carries_the_split():
    """`TraceData.llm_calls[i]` is what a user's eval function reads.

    Additive only: `prompt_tokens` / `total_tokens` keep their existing
    meaning, so no eval's score moves because of this field. But an eval that
    WANTS to assert "my prefix stayed cached" now has the number to assert on.
    """
    from decimalai.evals import trace_to_trace_data

    data = trace_to_trace_data({
        "id": "t1",
        "status": "success",
        "llm_calls": [
            {
                "model_name": "claude-sonnet-4-5",
                "input_tokens": 4_000,
                "output_tokens": 900,
                "cache_read_tokens": 180_000,
                "cache_creation_tokens": 0,
            },
            {"model_name": "gpt-4o", "input_tokens": 10, "output_tokens": 2},
        ],
    })
    warm, silent = data.llm_calls

    assert warm.prompt_tokens == 4_000          # unchanged meaning
    assert warm.cache_read_tokens == 180_000
    assert warm.cache_creation_tokens == 0      # measured miss, kept as 0
    assert silent.cache_read_tokens is None     # never measured
    assert silent.cache_creation_tokens is None
