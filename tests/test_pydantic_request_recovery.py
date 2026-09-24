"""Real agent loop: recovery after a tool must never execute that tool again."""
import asyncio

import pytest

pytest.importorskip("pydantic_ai")
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from decimalai._provider_recovery import _failure
from decimalai.pydantic_ai import retry_model_requests


@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_tool_preserves_result_and_records_attempts(status, monkeypatch):
    calls, effects = [], []
    async def respond(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            return ModelResponse(parts=[ToolCallPart("charge_once", {"amount": 5})])
        if len(calls) == 2:
            raise ModelHTTPError(status, "fixture", "temporary capacity shortage")
        assert any(type(part).__name__ == "ToolReturnPart" for m in messages for part in m.parts)
        return ModelResponse(parts=[TextPart("charged")])
    def charge_once(amount: int) -> str:
        effects.append(amount)
        return "receipt-123"

    agent = Agent(retry_model_requests(FunctionModel(respond)), tools=[charge_once])
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("test").start_as_current_span("run"):
        assert agent.run_sync("charge 5").output == "charged"
    assert effects == [5] and len(calls) == 3
    events = exporter.get_finished_spans()[0].events
    assert [(e.attributes["attempt"], e.attributes["outcome"]) for e in events] == [
        (1, "success"), (1, "retry"), (2, "recovered"),
    ]
    from decimalai import _config
    from decimalai._config import DecimalConfig
    from decimalai.otel import DecimalSpanExporter
    monkeypatch.setattr(_config, "_config", DecimalConfig(api_key="test", enabled=True))
    wire_trace = DecimalSpanExporter()._assemble_trace(list(exporter.get_finished_spans()))[0]
    attempts = wire_trace.spans[0].attributes["decimalai.model_requests"]
    assert [a["outcome"] for a in attempts] == ["success", "retry", "recovered"]
    assert wire_trace.status.value == "success"
    provider.shutdown()


@pytest.mark.parametrize("status,body", [(401, "bad key"), (403, "denied"),
                                         (429, "prepayment credits are depleted")])
def test_permanent_failure_is_not_retried(status, body):
    calls = []
    async def fail(messages, info):
        calls.append(1)
        raise ModelHTTPError(status, "fixture", body)
    agent = Agent(retry_model_requests(FunctionModel(fail)))
    with pytest.raises(ModelHTTPError):
        agent.run_sync("hello")
    assert calls == [1]


def test_repeated_unavailability_propagates_after_three_attempts():
    calls = []
    async def fail(messages, info):
        calls.append(1)
        raise ModelHTTPError(503, "fixture", "capacity")
    agent = Agent(retry_model_requests(FunctionModel(fail)))
    with pytest.raises(ModelHTTPError):
        agent.run_sync("hello")
    assert len(calls) == 3


def test_total_deadline_cancels_hung_provider():
    calls, cancelled = [], []
    async def hang(messages, info):
        calls.append(1)
        try:
            await asyncio.sleep(60)
        finally:
            cancelled.append(1)
    agent = Agent(retry_model_requests(FunctionModel(hang), timeout_s=0.03))
    with pytest.raises(TimeoutError):
        agent.run_sync("hello")
    assert calls == cancelled == [1]


def test_retry_after_larger_than_budget_does_not_retry():
    calls = []
    async def fail(messages, info):
        calls.append(1)
        exc = ModelHTTPError(429, "fixture", "capacity")
        exc.headers = {"Retry-After": "60"}
        raise exc
    agent = Agent(retry_model_requests(FunctionModel(fail), timeout_s=1))
    with pytest.raises(ModelHTTPError):
        agent.run_sync("hello")
    assert calls == [1]


def test_external_cancellation_is_not_retried():
    calls = []
    async def cancel(messages, info):
        calls.append(1)
        raise asyncio.CancelledError()
    agent = Agent(retry_model_requests(FunctionModel(cancel)))
    with pytest.raises(asyncio.CancelledError):
        agent.run_sync("hello")
    assert calls == [1]


def test_generic_quota_with_billing_help_link_is_transient():
    assert _failure(ModelHTTPError(429, "fixture", "RESOURCE_EXHAUSTED: see billing for limits")) == ("throttle", None)


@pytest.mark.parametrize("kwargs", [{"max_attempts": 0}, {"max_attempts": 1.5},
                                    {"timeout_s": 0}, {"timeout_s": float("inf")}])
def test_invalid_bounds_rejected(kwargs):
    with pytest.raises(ValueError):
        retry_model_requests(None, **kwargs)
