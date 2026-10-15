"""`trace_to_trace_data` lists the tool calls a trace recorded.

Every adapter in this SDK records a tool call as a ``ToolCallRecord``, which
serializes as ``{"tool_name": ..., "args": {...}}``. The reader looked only for
``name`` and ``function.name`` (and only for ``function.arguments``), so every
recorded tool call reached an eval as ``ToolCallView(name="", args={})``: an
eval asking "did the agent call lookup_order?" answered no on a trace that did.

No network: the adapter path below is driven with stand-in spans and the
backend client is a mock.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from decimalai.evals import trace_to_trace_data
from decimalai.schema.common import Status
from decimalai.schema.trace import LlmCallRecord, RunTrace, ToolCallRecord


def _dict_trace(*tool_calls):
    return {"id": "t-1", "llm_calls": [{"model_name": "m", "tool_calls": list(tool_calls)}]}


@pytest.mark.parametrize("entry", [
    pytest.param({"tool_name": "lookup_order", "args": {"order": "A-1"}}, id="ToolCallRecord"),
    pytest.param({"name": "lookup_order", "args": {"order": "A-1"}}, id="name+args"),
    pytest.param({"name": "lookup_order", "arguments": '{"order": "A-1"}'}, id="name+arguments"),
    pytest.param({"id": "call_1", "type": "function",
                  "function": {"name": "lookup_order", "arguments": '{"order": "A-1"}'}},
                 id="chat-completions"),
    pytest.param({"function": {"name": "lookup_order", "arguments": {"order": "A-1"}}},
                 id="chat-completions-dict-args"),
])
def test_every_recorded_shape_is_listed(entry):
    data = trace_to_trace_data(_dict_trace(entry))

    assert [(tc.name, tc.args) for tc in data.tool_calls] == [("lookup_order", {"order": "A-1"})]
    (call,) = data.llm_calls
    assert [(tc.name, tc.args) for tc in call.tool_calls] == [("lookup_order", {"order": "A-1"})]


def test_arguments_that_are_not_a_json_object_give_no_args():
    data = trace_to_trace_data(_dict_trace(
        {"function": {"name": "lookup_order", "arguments": "not json"}},
    ))
    assert [(tc.name, tc.args) for tc in data.tool_calls] == [("lookup_order", {})]


@pytest.mark.parametrize("as_json", [False, True], ids=["RunTrace", "wire-dict"])
def test_a_tool_call_record_is_listed_with_its_args_and_result(as_json):
    """The model and its JSON form — the second is what the platform hands
    back for a stored trace."""
    trace = RunTrace(
        agent_name="support",
        llm_calls=[LlmCallRecord(model_name="m", tool_calls=[ToolCallRecord(
            tool_name="lookup_order", args={"order": "A-1"}, result="refunded",
        )])],
    )
    data = trace_to_trace_data(trace.model_dump(mode="json") if as_json else trace)

    (tc,) = data.tool_calls
    assert (tc.name, tc.args, tc.result) == ("lookup_order", {"order": "A-1"}, "refunded")


# ── Through a real adapter ───────────────────────────────────


@pytest.fixture
def sdk(monkeypatch):
    import decimalai._config as cfg
    import decimalai.openai_agents as oai
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    monkeypatch.setattr(cfg, "_config", DecimalConfig(
        api_key="dai_sk_test", base_url="http://127.0.0.1:9", enabled=True,
    ))
    client = MagicMock()
    client.register_manifest.return_value = {"manifest_id": "m-1", "status": "active"}
    monkeypatch.setattr(cfg, "_client", client)
    monkeypatch.setattr(oai, "_manifest_id", None)
    monkeypatch.setattr(oai, "_manifest_tracker", ManifestTracker())
    return client


def _span(trace_id, span_type, **data):
    now = datetime.now(timezone.utc).isoformat()
    return SimpleNamespace(
        trace_id=trace_id, span_id=str(uuid4()), parent_id=None, error=None,
        started_at=now, ended_at=now, span_data=SimpleNamespace(type=span_type, **data),
    )


def test_a_trace_from_the_openai_agents_adapter_lists_its_tool_call(sdk):
    """A model turn that calls `lookup_order`, then the tool's own span — the
    ToolCallRecord the adapter writes for it must reach the eval by name."""
    from openai.types.responses import ResponseFunctionToolCall

    from decimalai._config import _sender
    from decimalai.openai_agents import DecimalTracingProcessor

    processor = DecimalTracingProcessor(agent_name="support")
    tid = f"trace_{uuid4().hex[:16]}"
    args = '{"order": "A-1"}'
    response = SimpleNamespace(
        id="resp_1", model="gpt-4o-mini", usage=None, temperature=None,
        max_output_tokens=None, instructions=None,
        output=[ResponseFunctionToolCall(id="fc_1", call_id="call_1", type="function_call",
                                         name="lookup_order", arguments=args)],
    )
    processor.on_trace_start(SimpleNamespace(trace_id=tid, name="support"))
    processor.on_span_end(_span(tid, "response", response=response,
                                input="where is order A-1?", usage=None))
    processor.on_span_end(_span(tid, "function", name="lookup_order",
                                input=args, output="order A-1: refunded"))
    processor.on_trace_end(SimpleNamespace(trace_id=tid, name="support"))
    _sender.flush()
    (trace,) = [c.args[0] for c in sdk.ingest_trace.call_args_list]
    assert trace.llm_calls[0].tool_calls[0].status == Status.SUCCESS  # the adapter's half

    for shipped in (trace, trace.model_dump(mode="json")):
        data = trace_to_trace_data(shipped)
        assert [(tc.name, tc.args, tc.result) for tc in data.tool_calls] == [
            ("lookup_order", {"order": "A-1"}, "order A-1: refunded"),
        ]
