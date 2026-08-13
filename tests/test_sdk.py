"""Tests for SDK schema models and callback handler."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from decimalai.schema.common import (
    CallRole, FinishReason, SourceType, SpanType, Status,
)
from decimalai.schema.trace import (
    LlmCallRecord, RunTrace, ToolCallRecord, TraceSpan,
)
from decimalai._config import DecimalConfig
from decimalai.langchain import CallbackHandler


# ── Schema model tests ────────────────────────────────

class TestEnums:
    def test_span_types(self):
        assert SpanType.LLM == "llm"
        assert SpanType.TOOL == "tool"
        assert SpanType.AGENT == "agent"

    def test_status_values(self):
        assert Status.SUCCESS == "success"
        assert Status.ERROR == "error"

    def test_source_types(self):
        assert SourceType.PRODUCTION == "production"
        assert SourceType.REPLAYED == "replayed"
        assert SourceType.SYNTHETIC == "synthetic"


class TestTraceModels:
    def test_run_trace_minimal(self):
        trace = RunTrace()
        assert trace.id is not None
        assert trace.status == Status.SUCCESS
        assert trace.source_type == SourceType.PRODUCTION
        assert trace.spans == []
        assert trace.llm_calls == []

    def test_run_trace_full(self):
        now = datetime.now(timezone.utc)
        trace = RunTrace(
            project="my-agent",
            agent_name="finance-agent",
            status=Status.SUCCESS,
            source_type=SourceType.PRODUCTION,
            started_at=now,
            ended_at=now,
            user_input_preview="What is...",
            final_output_preview="The answer is...",
        )
        assert trace.project == "my-agent"
        assert trace.agent_name == "finance-agent"

    def test_run_trace_roundtrip(self):
        trace = RunTrace(
            project="test",
            agent_name="test-agent",
            spans=[TraceSpan(name="root", span_type=SpanType.AGENT)],
            llm_calls=[LlmCallRecord(model_name="gpt-4o")],
        )
        data = json.loads(trace.model_dump_json())
        restored = RunTrace.model_validate(data)
        assert restored.project == "test"
        assert len(restored.spans) == 1
        assert len(restored.llm_calls) == 1
        assert restored.spans[0].name == "root"
        assert restored.llm_calls[0].model_name == "gpt-4o"

    def test_trace_span(self):
        span = TraceSpan(
            name="llm_call",
            span_type=SpanType.LLM,
            status=Status.RUNNING,
            input_preview="Hello",
        )
        assert span.name == "llm_call"
        assert span.span_type == SpanType.LLM

    def test_llm_call_record(self):
        call = LlmCallRecord(
            model_name="gpt-4o",
            provider="openai",
            call_role=CallRole.RESPONDER,
            rendered_input=[{"role": "user", "content": "Hi"}],
            output={"content": "Hello!"},
            input_tokens=10,
            output_tokens=5,
            finish_reason=FinishReason.STOP,
        )
        assert call.provider == "openai"
        assert call.input_tokens == 10
        assert call.finish_reason == FinishReason.STOP

    def test_tool_call_record(self):
        tc = ToolCallRecord(
            tool_name="get_population",
            args={"city": "Paris"},
            result="2161000",
            status=Status.SUCCESS,
        )
        assert tc.tool_name == "get_population"
        assert tc.args == {"city": "Paris"}

    def test_llm_call_with_tool_calls(self):
        call = LlmCallRecord(
            model_name="gpt-4o",
            tool_calls=[
                ToolCallRecord(tool_name="search", args={"q": "test"}),
                ToolCallRecord(tool_name="calc", args={"expr": "1+1"}),
            ],
        )
        assert len(call.tool_calls) == 2
        data = json.loads(call.model_dump_json())
        assert len(data["tool_calls"]) == 2


# ── Config tests ──────────────────────────────────────

class TestConfig:
    def test_defaults(self):
        config = DecimalConfig()
        assert config.project is None
        assert config.enabled is True
        assert config.base_url == "https://api.decimal.ai"

    def test_api_headers(self):
        config = DecimalConfig(api_key="dai_sk_test", project="my-project")
        headers = config.api_headers
        assert headers["Authorization"] == "Bearer dai_sk_test"
        assert headers["Content-Type"] == "application/json"
        # `project` is deprecated and inert: the platform reads no such header,
        # so emitting it made project= look like it grouped traces when it did
        # nothing. Must not appear on the wire.
        assert "X-Decimal-Project" not in headers


# ── Callback handler tests ─────────────────────────────

class TestCallbackHandler:
    def test_create_handler(self):
        handler = CallbackHandler(
            agent_name="test-agent", auto_send=False
        )
        assert handler.agent_name == "test-agent"
        assert handler.auto_send is False

    def test_chain_lifecycle(self):
        handler = CallbackHandler(agent_name="agent", auto_send=False)
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "AgentExecutor"},
            inputs={"input": "Hello World"},
            run_id=run_id,
        )
        assert run_id in handler._spans
        assert handler._spans[run_id].status == Status.RUNNING

        handler.on_chain_end(
            outputs={"output": "Hi there"},
            run_id=run_id,
        )
        assert handler._spans[run_id].status == Status.SUCCESS
        assert handler._user_input_preview is not None

    def test_llm_lifecycle(self):
        handler = CallbackHandler(agent_name="agent")
        llm_id = uuid4()

        handler.on_llm_start(
            serialized={"name": "ChatOpenAI"},
            prompts=["What is 1+1?"],
            run_id=llm_id,
            invocation_params={"model_name": "gpt-4o", "temperature": 0.7},
        )
        assert llm_id in handler._llm_calls
        call = handler._llm_calls[llm_id]
        assert call.model_name == "gpt-4o"
        assert call.temperature == 0.7
        assert call.status == Status.RUNNING

    def test_tool_lifecycle(self):
        handler = CallbackHandler(agent_name="agent")
        tool_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "get_population"},
            input_str='{"city": "Paris"}',
            run_id=tool_id,
        )
        assert tool_id in handler._spans
        assert tool_id in handler._tool_calls
        assert handler._tool_calls[tool_id].tool_name == "get_population"

        handler.on_tool_end(output="2161000", run_id=tool_id)
        assert handler._spans[tool_id].status == Status.SUCCESS
        assert handler._tool_calls[tool_id].result == "2161000"

    def test_build_trace(self):
        handler = CallbackHandler(
            agent_name="agent", auto_send=False
        )
        chain_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "pipeline"},
            inputs={"input": "test"},
            run_id=chain_id,
        )
        handler.on_llm_start(
            serialized={"name": "ChatOpenAI"},
            prompts=["test"],
            run_id=llm_id,
            parent_run_id=chain_id,
            invocation_params={"model_name": "gpt-4o"},
        )
        handler.on_chain_end(outputs={"result": "done"}, run_id=chain_id)

        trace = handler.build_trace()
        assert trace.agent_name == "agent"
        assert len(trace.spans) == 1
        assert len(trace.llm_calls) == 1
        assert trace.user_input_preview is not None

    def test_reset(self):
        handler = CallbackHandler()
        handler.on_chain_start(
            serialized={"name": "test"},
            inputs={"x": 1},
            run_id=uuid4(),
        )
        assert len(handler._spans) == 1
        handler.reset()
        assert len(handler._spans) == 0

    def test_get_completed_trace(self):
        handler = CallbackHandler(agent_name="agent")
        handler.on_chain_start(
            serialized={"name": "pipeline"},
            inputs={"input": "test"},
            run_id=uuid4(),
        )
        trace = handler.get_completed_trace()
        assert trace.agent_name == "agent"
        # After get_completed_trace, handler should be reset
        assert len(handler._spans) == 0

    def test_chain_error(self):
        handler = CallbackHandler()
        root_id = uuid4()
        run_id = uuid4()
        handler.on_chain_start(
            serialized={"name": "RootAgent"}, inputs={}, run_id=root_id
        )
        handler.on_chain_start(
            serialized={"name": "test"}, inputs={}, run_id=run_id,
            parent_run_id=root_id,
        )
        handler.on_chain_error(
            error=ValueError("boom"), run_id=run_id
        )
        assert handler._spans[run_id].status == Status.ERROR

    def test_llm_error(self):
        handler = CallbackHandler()
        llm_id = uuid4()
        handler.on_llm_start(
            serialized={}, prompts=["test"], run_id=llm_id,
            invocation_params={"model_name": "gpt-4o"},
        )
        handler.on_llm_error(error=RuntimeError("timeout"), run_id=llm_id)
        call = handler._llm_calls[llm_id]
        assert call.status == Status.ERROR
        assert call.finish_reason == FinishReason.ERROR

    def test_llm_error_drains_streaming_buffer(self):
        """A streaming LLM call that errors mid-stream must drop its
        buffered tokens from `_streaming_buffers`. Without this, the
        global handler accumulates one entry per errored streaming call
        for the lifetime of the process — multi-hour memory leak under
        flaky providers / timeouts / content-filter rejections.
        """
        handler = CallbackHandler()
        llm_id = uuid4()
        handler.on_llm_start(
            serialized={}, prompts=["test"], run_id=llm_id,
            invocation_params={"model_name": "gpt-4o"},
        )
        # Simulate a few streamed tokens arriving before the error.
        handler.on_llm_new_token(token="Hel", run_id=llm_id)
        handler.on_llm_new_token(token="lo", run_id=llm_id)
        assert llm_id in handler._streaming_buffers, (
            "on_llm_new_token should have populated the streaming buffer"
        )

        handler.on_llm_error(error=RuntimeError("content_filter"), run_id=llm_id)

        # Buffer for this call must be drained.
        assert llm_id not in handler._streaming_buffers, (
            "_streaming_buffers still holds the entry for an errored streaming "
            "call. on_llm_error must drop the buffered tokens, or the global "
            "handler leaks one entry per errored stream for the life of the "
            "process."
        )


# ── Bounded _resolve_agent_name walk ──


def test_resolve_agent_name_terminates_on_cycle():
    """If `_spans` ever contains a parent_span_id cycle (malformed
    parent_run_id from LangChain, or a corrupted custom callback), the
    old `while span_id and span_id in self._spans` looped forever and
    blocked the dispatcher thread. The new walk has a `seen` guard and
    a 32-hop cap.
    """
    from decimalai.langchain import CallbackHandler
    from decimalai.schema.trace import TraceSpan

    handler = CallbackHandler(agent_name="fallback-agent")
    a_id = uuid4()
    b_id = uuid4()
    # Build a cycle: a → b → a
    handler._spans[a_id] = TraceSpan(
        id=a_id, name="a", span_type=SpanType.LLM, parent_span_id=b_id,
    )
    handler._spans[b_id] = TraceSpan(
        id=b_id, name="b", span_type=SpanType.LLM, parent_span_id=a_id,
    )
    # Should return the fallback `agent_name` instead of looping forever.
    result = handler._resolve_agent_name(a_id)
    assert result == "fallback-agent", (
        f"A parent_span_id cycle should fall through to the handler's "
        f"agent_name instead of walking forever; got {result!r}"
    )


def test_resolve_agent_name_returns_nearest_agent_ancestor():
    """Regression guard: the cycle break must NOT break the
    happy-path walk. A chain `llm → tool → agent` should still resolve
    to the agent's name."""
    from decimalai.langchain import CallbackHandler
    from decimalai.schema.trace import TraceSpan

    handler = CallbackHandler(agent_name="root-agent")
    agent_id = uuid4()
    tool_id = uuid4()
    llm_id = uuid4()
    handler._spans[agent_id] = TraceSpan(
        id=agent_id, name="researcher", span_type=SpanType.AGENT,
        parent_span_id=None,
    )
    handler._spans[tool_id] = TraceSpan(
        id=tool_id, name="search", span_type=SpanType.TOOL,
        parent_span_id=agent_id,
    )
    handler._spans[llm_id] = TraceSpan(
        id=llm_id, name="llm-call", span_type=SpanType.LLM,
        parent_span_id=tool_id,
    )

    # Resolving from llm should walk llm → tool → agent and return "researcher".
    assert handler._resolve_agent_name(llm_id) == "researcher"


# ── Tool description flows through to manifest ──


def test_on_tool_start_captures_description():
    """`on_tool_start` previously read `description` into a local
    variable and discarded it — `_seen_tools[name]` only had `name` and
    `schema`. A description change like "Search the web" → "Search the
    corporate intranet" produced NO manifest delta. Now `_seen_tools`
    includes the `description` field, which `extract_from_config` reads
    and the content_hash incorporates."""
    from decimalai.langchain import CallbackHandler
    from decimalai.schema.manifest import extract_from_config

    handler = CallbackHandler(agent_name="search-agent")
    tool_id = uuid4()

    handler.on_tool_start(
        serialized={
            "name": "web_search",
            "description": "Search the web for current information",
            "schema": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
        input_str='{"q": "test"}',
        run_id=tool_id,
    )

    # _seen_tools should now carry the description.
    seen = handler._seen_tools["web_search"]
    assert seen["description"] == "Search the web for current information", (
        f"on_tool_start must record the tool's description in _seen_tools "
        f"so it reaches the manifest; got {seen}"
    )

    # Extract a manifest snapshot — the description should be in schema_json
    # AND the content_hash should be different for two tools with different
    # descriptions but same name.
    tools = list(handler._seen_tools.values())
    snapshot_a = extract_from_config(agent_name="search-agent", tools=tools)
    tool_comp_a = next(c for c in snapshot_a.components if c.component_type == "tool")
    assert tool_comp_a.schema_json["description"] == "Search the web for current information"

    # Mutate the description and re-extract — content_hash MUST differ
    tools_b = [
        {**tools[0], "description": "Search the corporate intranet only"},
    ]
    snapshot_b = extract_from_config(agent_name="search-agent", tools=tools_b)
    tool_comp_b = next(c for c in snapshot_b.components if c.component_type == "tool")
    assert tool_comp_a.content_hash != tool_comp_b.content_hash, (
        "Changing a tool's description must change its content_hash — "
        "otherwise the manifest pipeline sees no delta and the edit is "
        "silently dropped"
    )


# ── _warn_once_then_debug pattern ──


def test_warn_once_then_debug_pattern(caplog):
    """The first call per category should log at WARNING; subsequent
    calls for the same category drop to DEBUG. Surfaces user-actionable
    SDK failures once-per-process without spamming the log."""
    import logging
    from decimalai.langchain import _warn_once_then_debug, _warned_once

    # Reset the module-level state so this test is hermetic.
    _warned_once.clear()

    caplog.set_level(logging.DEBUG, logger="decimalai.langchain")
    try:
        # First call → WARNING
        _warn_once_then_debug("test-cat", "first call")
        first = [r for r in caplog.records if r.message == "first call"]
        assert len(first) == 1
        assert first[0].levelname == "WARNING"

        caplog.clear()

        # Second call → DEBUG (same category)
        _warn_once_then_debug("test-cat", "second call")
        second = [r for r in caplog.records if r.message == "second call"]
        assert len(second) == 1
        assert second[0].levelname == "DEBUG"

        caplog.clear()

        # Different category → WARNING again (independent counters)
        _warn_once_then_debug("other-cat", "other category call")
        other = [r for r in caplog.records if r.message == "other category call"]
        assert len(other) == 1
        assert other[0].levelname == "WARNING"
    finally:
        _warned_once.clear()
