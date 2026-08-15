"""Tests for the OpenAI Agents SDK integration (decimalai.openai_agents)."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest


class _MockSpanData:
    """Lightweight mock for OpenAI Agents SpanData subclasses."""

    def __init__(self, span_type: str, **kwargs):
        self._type = span_type
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def type(self) -> str:
        return self._type


class _MockSpan:
    """Lightweight mock for OpenAI Agents Span."""

    def __init__(
        self,
        trace_id: str,
        span_id: str | None = None,
        parent_id: str | None = None,
        span_data: _MockSpanData | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        error: dict | None = None,
    ):
        self.trace_id = trace_id
        self.span_id = span_id or str(uuid4())
        self.parent_id = parent_id
        self.span_data = span_data
        self.started_at = started_at or datetime.now(timezone.utc).isoformat()
        self.ended_at = ended_at or datetime.now(timezone.utc).isoformat()
        self.error = error


class _MockTrace:
    """Lightweight mock for OpenAI Agents Trace."""

    def __init__(self, trace_id: str, name: str = "test-workflow"):
        self.trace_id = trace_id
        self.name = name


# ── Setup/Teardown ──────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_sdk():
    """Reset global SDK state before each test."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    # register_manifest must return a dict with a string manifest_id
    cfg._client.register_manifest.return_value = {"manifest_id": "test-manifest-id", "status": "active"}

    # Reset the module-level manifest_id so stale MagicMocks don't leak
    try:
        import decimalai.openai_agents as oai
        oai._manifest_id = None
    except Exception:
        pass
    yield


# ── Processor Tests ─────────────────────────────────────────


class TestDecimalTracingProcessor:
    """Tests for DecimalTracingProcessor."""

    def test_generation_span_creates_llm_call(self):
        """A generation span should map to an LlmCallRecord."""
        from decimalai.openai_agents import DecimalTracingProcessor
        import decimalai._config as cfg

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        # Start trace
        trace = _MockTrace(trace_id=trace_id, name="test-workflow")
        processor.on_trace_start(trace)

        # Simulate a generation span
        gen_data = _MockSpanData(
            "generation",
            model="gpt-4o",
            input=[{"role": "user", "content": "Hello"}],
            output=[{"role": "assistant", "content": "Hi there!"}],
            model_config={"temperature": 0.7},
            usage={"input_tokens": 5, "output_tokens": 8},
        )
        span = _MockSpan(trace_id=trace_id, span_data=gen_data)
        processor.on_span_end(span)

        # End trace → should send
        processor.on_trace_end(trace)

        # Flush background sender
        from decimalai._config import _sender
        _sender.flush()

        # Verify ingest_trace was called
        cfg._client.ingest_trace.assert_called_once()
        run_trace = cfg._client.ingest_trace.call_args[0][0]

        assert run_trace.agent_name == "test-agent"
        assert len(run_trace.llm_calls) == 1

        llm_call = run_trace.llm_calls[0]
        assert llm_call.model_name == "gpt-4o"
        assert llm_call.provider == "openai"
        assert llm_call.input_tokens == 5
        assert llm_call.output_tokens == 8
        assert llm_call.temperature == 0.7

    def test_function_span_creates_tool_span(self):
        """A function span should map to a TraceSpan with type TOOL."""
        from decimalai.openai_agents import DecimalTracingProcessor
        import decimalai._config as cfg

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        trace = _MockTrace(trace_id=trace_id)
        processor.on_trace_start(trace)

        func_data = _MockSpanData(
            "function",
            name="get_weather",
            input='{"city": "NYC"}',
            output='{"temp": 72}',
        )
        span = _MockSpan(trace_id=trace_id, span_data=func_data)
        processor.on_span_end(span)

        processor.on_trace_end(trace)
        from decimalai._config import _sender
        _sender.flush()

        cfg._client.ingest_trace.assert_called_once()
        run_trace = cfg._client.ingest_trace.call_args[0][0]

        # Should have a TOOL span
        tool_spans = [s for s in run_trace.spans if s.span_type.value == "tool"]
        assert len(tool_spans) == 1
        assert tool_spans[0].name == "get_weather"
        assert tool_spans[0].input_preview is not None

    def test_agent_span_auto_detects_name(self):
        """An agent span should auto-populate the agent_name on the trace."""
        from decimalai.openai_agents import DecimalTracingProcessor
        import decimalai._config as cfg

        # No default agent_name — let it auto-detect
        processor = DecimalTracingProcessor()
        trace_id = f"trace_{uuid4().hex[:16]}"

        trace = _MockTrace(trace_id=trace_id)
        processor.on_trace_start(trace)

        agent_data = _MockSpanData(
            "agent",
            name="finance-assistant",
            tools=["get_stock_price", "get_news"],
            handoffs=[],
            output_type="str",
        )
        span = _MockSpan(trace_id=trace_id, span_data=agent_data)
        processor.on_span_end(span)

        processor.on_trace_end(trace)
        from decimalai._config import _sender
        _sender.flush()

        run_trace = cfg._client.ingest_trace.call_args[0][0]
        assert run_trace.agent_name == "finance-assistant"

        # Should have AGENT span with attributes
        agent_spans = [s for s in run_trace.spans if s.span_type.value == "agent"]
        assert len(agent_spans) == 1
        assert agent_spans[0].attributes["tools"] == ["get_stock_price", "get_news"]

    def test_handoff_span(self):
        """A handoff span should create a TraceSpan with from/to agent info."""
        from decimalai.openai_agents import DecimalTracingProcessor
        import decimalai._config as cfg

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        trace = _MockTrace(trace_id=trace_id)
        processor.on_trace_start(trace)

        handoff_data = _MockSpanData(
            "handoff",
            from_agent="triage",
            to_agent="billing",
        )
        span = _MockSpan(trace_id=trace_id, span_data=handoff_data)
        processor.on_span_end(span)

        processor.on_trace_end(trace)
        from decimalai._config import _sender
        _sender.flush()

        run_trace = cfg._client.ingest_trace.call_args[0][0]
        handoff_spans = [s for s in run_trace.spans if "handoff" in s.name]
        assert len(handoff_spans) == 1
        assert handoff_spans[0].attributes["from_agent"] == "triage"
        assert handoff_spans[0].attributes["to_agent"] == "billing"

    def test_guardrail_span(self):
        """A guardrail span should create a TraceSpan with triggered status."""
        from decimalai.openai_agents import DecimalTracingProcessor
        import decimalai._config as cfg

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        trace = _MockTrace(trace_id=trace_id)
        processor.on_trace_start(trace)

        guard_data = _MockSpanData(
            "guardrail",
            name="pii_filter",
            triggered=True,
        )
        span = _MockSpan(trace_id=trace_id, span_data=guard_data)
        processor.on_span_end(span)

        processor.on_trace_end(trace)
        from decimalai._config import _sender
        _sender.flush()

        run_trace = cfg._client.ingest_trace.call_args[0][0]
        guardrail_spans = [s for s in run_trace.spans if "guardrail" in s.name]
        assert len(guardrail_spans) == 1
        assert guardrail_spans[0].attributes["triggered"] is True

    def test_error_handling_on_generation(self):
        """A generation span with an error should map to ERROR status."""
        from decimalai.openai_agents import DecimalTracingProcessor
        import decimalai._config as cfg

        processor = DecimalTracingProcessor(agent_name="test-agent")
        trace_id = f"trace_{uuid4().hex[:16]}"

        trace = _MockTrace(trace_id=trace_id)
        processor.on_trace_start(trace)

        gen_data = _MockSpanData(
            "generation",
            model="gpt-4o",
            input=[{"role": "user", "content": "Hello"}],
            output=None,
            model_config={},
            usage={},
        )
        span = _MockSpan(
            trace_id=trace_id,
            span_data=gen_data,
            error={"message": "Rate limit exceeded"},
        )
        processor.on_span_end(span)

        processor.on_trace_end(trace)
        from decimalai._config import _sender
        _sender.flush()

        run_trace = cfg._client.ingest_trace.call_args[0][0]
        assert len(run_trace.llm_calls) == 1
        assert run_trace.llm_calls[0].status.value == "error"
        assert run_trace.llm_calls[0].output["error"] == "Rate limit exceeded"

    def test_multi_span_trace(self):
        """A complete trace with agent + generation + function spans."""
        from decimalai.openai_agents import DecimalTracingProcessor
        import decimalai._config as cfg

        processor = DecimalTracingProcessor()
        trace_id = f"trace_{uuid4().hex[:16]}"

        trace = _MockTrace(trace_id=trace_id)
        processor.on_trace_start(trace)

        # Agent span
        agent_data = _MockSpanData(
            "agent", name="research-agent",
            tools=["search"], handoffs=[], output_type="str",
        )
        processor.on_span_end(_MockSpan(trace_id=trace_id, span_data=agent_data))

        # Generation span
        gen_data = _MockSpanData(
            "generation", model="gpt-4o",
            input=[{"role": "user", "content": "search for AI news"}],
            output=[{"role": "assistant", "content": "Here are results..."}],
            model_config={"temperature": 0.0},
            usage={"input_tokens": 20, "output_tokens": 50},
        )
        processor.on_span_end(_MockSpan(trace_id=trace_id, span_data=gen_data))

        # Function span
        func_data = _MockSpanData(
            "function", name="search",
            input="AI news", output="Some results",
        )
        processor.on_span_end(_MockSpan(trace_id=trace_id, span_data=func_data))

        processor.on_trace_end(trace)
        from decimalai._config import _sender
        _sender.flush()

        run_trace = cfg._client.ingest_trace.call_args[0][0]
        assert run_trace.agent_name == "research-agent"
        assert len(run_trace.llm_calls) == 1
        assert len(run_trace.spans) == 3  # agent + llm wrapper + tool

    def test_on_span_start_is_noop(self):
        """on_span_start should be a no-op (not error)."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor()
        # Should not raise
        processor.on_span_start(MagicMock())

    def test_force_flush_is_noop(self):
        """force_flush should not error."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor()
        processor.force_flush()  # Should not raise

    def test_shutdown_sends_remaining(self):
        """shutdown should send any in-flight traces."""
        from decimalai.openai_agents import DecimalTracingProcessor
        import decimalai._config as cfg

        processor = DecimalTracingProcessor(agent_name="test")
        trace_id = f"trace_{uuid4().hex[:16]}"

        trace = _MockTrace(trace_id=trace_id)
        processor.on_trace_start(trace)

        # Don't call on_trace_end — call shutdown instead
        processor.shutdown()

        from decimalai._config import _sender
        _sender.flush()

        cfg._client.ingest_trace.assert_called_once()

    def test_unknown_trace_id_on_end(self):
        """on_trace_end with unknown trace_id should not error."""
        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor()
        trace = _MockTrace(trace_id="nonexistent-id")
        # Should not raise
        processor.on_trace_end(trace)

    def test_disabled_sdk_skips_send(self):
        """When SDK is disabled, traces should not be sent."""
        import decimalai._config as cfg
        cfg._config.enabled = False

        from decimalai.openai_agents import DecimalTracingProcessor

        processor = DecimalTracingProcessor(agent_name="test")
        trace_id = f"trace_{uuid4().hex[:16]}"

        trace = _MockTrace(trace_id=trace_id)
        processor.on_trace_start(trace)
        processor.on_trace_end(trace)

        from decimalai._config import _sender
        _sender.flush()

        cfg._client.ingest_trace.assert_not_called()


# ── Install Tests ───────────────────────────────────────────


class TestInstall:
    """Tests for the install() function."""

    def test_install_calls_add_trace_processor(self):
        """install() should call add_trace_processor by default."""
        mock_add = MagicMock()
        mock_set = MagicMock()

        with patch.dict("sys.modules", {
            "agents": MagicMock(),
            "agents.tracing": MagicMock(
                add_trace_processor=mock_add,
                set_trace_processors=mock_set,
            ),
        }):
            from decimalai.openai_agents import install
            install()

            mock_add.assert_called_once()
            mock_set.assert_not_called()

            # Verify the processor is a DecimalTracingProcessor
            from decimalai.openai_agents import DecimalTracingProcessor
            processor = mock_add.call_args[0][0]
            assert isinstance(processor, DecimalTracingProcessor)

    def test_install_exclusive_calls_set_trace_processors(self):
        """install(exclusive=True) should call set_trace_processors."""
        mock_add = MagicMock()
        mock_set = MagicMock()

        with patch.dict("sys.modules", {
            "agents": MagicMock(),
            "agents.tracing": MagicMock(
                add_trace_processor=mock_add,
                set_trace_processors=mock_set,
            ),
        }):
            from decimalai.openai_agents import install
            install(exclusive=True)

            mock_set.assert_called_once()
            mock_add.assert_not_called()

    def test_install_passes_agent_name(self):
        """install(agent_name=...) should set the processor's default_agent_name."""
        mock_add = MagicMock()

        with patch.dict("sys.modules", {
            "agents": MagicMock(),
            "agents.tracing": MagicMock(
                add_trace_processor=mock_add,
                set_trace_processors=MagicMock(),
            ),
        }):
            from decimalai.openai_agents import install
            install(agent_name="my-custom-agent")

            processor = mock_add.call_args[0][0]
            assert processor.default_agent_name == "my-custom-agent"

    def test_install_without_openai_agents_raises(self):
        """install() should raise ImportError if openai-agents is not installed."""
        saved = {}
        # Only the third-party `agents` package — a substring match also swept
        # out `decimalai.openai_agents`, and re-importing THAT under the patch
        # builds a second module object and rebinds it on the `decimalai`
        # package. Restoring sys.modules does not restore the package
        # attribute, so later tests reset globals on one object while the
        # adapter under test reads the other.
        for key in list(sys.modules.keys()):
            if key == "agents" or key.startswith("agents."):
                saved[key] = sys.modules.pop(key)

        try:
            with patch.dict("sys.modules", {"agents.tracing": None}):
                from decimalai.openai_agents import install
                with pytest.raises(ImportError, match="openai-agents"):
                    install()
        finally:
            sys.modules.update(saved)


# ── Utility Tests ───────────────────────────────────────────


class TestUtilities:
    """Tests for helper functions."""

    def test_infer_provider(self):
        from decimalai.openai_agents import _infer_provider

        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("claude-3-sonnet") == "anthropic"
        assert _infer_provider("gemini-2.0-flash") == "google"
        assert _infer_provider("mistral-large") == "mistral"
        assert _infer_provider("llama-3.1-70b") == "meta"
        assert _infer_provider("o1-mini") == "openai"
        assert _infer_provider("o3-mini") == "openai"
        assert _infer_provider(None) is None
        # Default for unknown models in OpenAI Agents SDK context
        assert _infer_provider("some-unknown-model") == "openai"

    def test_introspect_agent_resolves_model_name_from_object(self):
        """A non-string Agent.model (e.g. an OpenAIChatCompletionsModel pointed
        at Gemini's OpenAI-compatible endpoint) must record the real model NAME
        and inferred provider — not a useless `<...object at 0x...>` repr."""
        from decimalai.openai_agents import _introspect_agent

        class _FakeModel:  # duck-types agents.* Model: exposes `.model`
            model = "gemini-2.5-flash"

        class _FakeAgent:
            name = "gem-agent"
            instructions = "help"
            tools = []
            handoffs = []
            model = _FakeModel()

        models = _introspect_agent(_FakeAgent())["models"]
        assert models["default"]["model"] == "gemini-2.5-flash"
        assert models["default"]["provider"] == "google"
        assert "object at 0x" not in models["default"]["model"]

        # String models keep working unchanged.
        class _StrAgent(_FakeAgent):
            model = "gpt-5-mini"

        str_models = _introspect_agent(_StrAgent())["models"]
        assert str_models["default"] == {"provider": "openai", "model": "gpt-5-mini"}

    def test_parse_iso(self):
        from decimalai.openai_agents import _parse_iso

        # Standard ISO
        dt = _parse_iso("2025-01-01T00:00:00+00:00")
        assert dt is not None
        assert dt.year == 2025

        # Z suffix
        dt = _parse_iso("2025-06-15T12:30:00Z")
        assert dt is not None

        # None
        assert _parse_iso(None) is None
        assert _parse_iso("") is None

    def test_preview(self):
        from decimalai.openai_agents import _preview

        assert _preview(None) is None
        assert _preview("hello") == "hello"
        assert _preview([{"content": "Hi"}]) == "Hi"
        assert _preview({"content": "Hello"}) == "Hello"
        assert len(_preview("x" * 500)) <= 200

    def test_normalize_messages(self):
        from decimalai.openai_agents import _normalize_messages

        assert _normalize_messages(None) is None
        result = _normalize_messages([
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ])
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"


# ── Init Integration Test ──────────────────────────────────


class TestInitIntegration:
    """Test that init(openai_agents=True) works."""

    def test_init_with_openai_agents_flag(self):
        """init(openai_agents=True) should call install()."""
        mock_add = MagicMock()

        with patch.dict("sys.modules", {
            "agents": MagicMock(),
            "agents.tracing": MagicMock(
                add_trace_processor=mock_add,
                set_trace_processors=MagicMock(),
            ),
        }):
            import decimalai
            import decimalai._config as cfg
            cfg._config = None
            cfg._client = None

            decimalai.init(
                api_key="dai_sk_test",
                base_url="http://localhost:8000",
                openai_agents=True,
            )

            mock_add.assert_called_once()
