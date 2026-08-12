"""Lock in: OpenAI-Agents wrapper spans preserve parent/child hierarchy.

Deep-audit finding (sdk-integrations): handlers computed
``span_id = _coerce_span_id(span.span_id)`` but then built wrapper
TraceSpans with ``id=uuid4()`` — so the computed span_id was dead and a
child's ``parent_span_id`` (a coerced id) never matched its parent's
random uuid4 ``id``. The span tree was flattened.

The fix uses ``id=span_id`` on every wrapper TraceSpan so the coerced
ids line up.

No backend / no real OpenAI Agents SDK — spans are mocked.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from decimalai.openai_agents import _coerce_span_id


class _MockSpanData:
    def __init__(self, span_type, **kwargs):
        self._type = span_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def type(self):
        return self._type


class _MockSpan:
    def __init__(self, trace_id, span_id=None, parent_id=None, span_data=None):
        self.trace_id = trace_id
        self.span_id = span_id or str(uuid4())
        self.parent_id = parent_id
        self.span_data = span_data
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at = datetime.now(timezone.utc).isoformat()
        self.error = None


class _MockTrace:
    def __init__(self, trace_id, name="test-workflow"):
        self.trace_id = trace_id
        self.name = name


@pytest.fixture(autouse=True)
def _reset_sdk():
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "m", "status": "active"}
    import decimalai.openai_agents as oai
    oai._manifest_id = None
    yield


def test_child_span_links_to_parent_span_id():
    """A function (tool) span whose parent_id is the agent span's span_id
    must produce a TraceSpan whose parent_span_id == the agent TraceSpan's id.
    Pre-fix the agent span had a random uuid4 id, breaking the link.
    """
    from decimalai.openai_agents import DecimalTracingProcessor
    import decimalai._config as cfg

    processor = DecimalTracingProcessor(agent_name="test-agent")
    trace_id = "trace_abc123"
    agent_span_id = "span_agent_001"
    child_span_id = "span_func_002"

    processor.on_trace_start(_MockTrace(trace_id))

    # Parent: agent span
    agent_span = _MockSpan(
        trace_id=trace_id,
        span_id=agent_span_id,
        parent_id=None,
        span_data=_MockSpanData("agent", name="planner", tools=[], handoffs=[]),
    )
    processor.on_span_end(agent_span)

    # Child: function span whose parent_id points at the agent span
    func_span = _MockSpan(
        trace_id=trace_id,
        span_id=child_span_id,
        parent_id=agent_span_id,
        span_data=_MockSpanData("function", name="get_weather", input="{}", output="{}"),
    )
    processor.on_span_end(func_span)

    processor.on_trace_end(_MockTrace(trace_id))

    from decimalai._config import _sender
    _sender.flush()

    cfg._client.ingest_trace.assert_called_once()
    run_trace = cfg._client.ingest_trace.call_args[0][0]

    spans_by_name = {s.name: s for s in run_trace.spans}
    agent_ts = spans_by_name["planner"]
    func_ts = spans_by_name["get_weather"]

    # The fix: both ids are the coerced span ids, so the child links up.
    assert agent_ts.id == _coerce_span_id(agent_span_id), (
        "agent wrapper span must use its coerced span_id as its id, not a random uuid4"
    )
    assert func_ts.parent_span_id == agent_ts.id, (
        "child span's parent_span_id must match the parent span's id — "
        "the hierarchy was broken when wrappers used id=uuid4()"
    )
