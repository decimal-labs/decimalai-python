"""Tests for the OpenTelemetry SpanExporter (decimalai.otel)."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


# ── Mock OTEL objects ────────────────────────────────────────


class _MockContext:
    """Lightweight mock for OTEL SpanContext."""

    def __init__(self, trace_id: int, span_id: int):
        self.trace_id = trace_id
        self.span_id = span_id
        self.trace_flags = 1  # sampled


class _MockStatus:
    """Lightweight mock for OTEL StatusCode."""

    def __init__(self, code: str = "OK"):
        self.status_code = code


class _MockResource:
    """Lightweight mock for OTEL Resource."""

    def __init__(self, service_name: str = "test-service"):
        self.attributes = {"service.name": service_name}


class _MockSpan:
    """Lightweight mock for OTEL ReadableSpan."""

    def __init__(
        self,
        name: str,
        trace_id: int,
        span_id: int,
        parent_span_id: int | None = None,
        attributes: dict | None = None,
        start_time_ns: int | None = None,
        end_time_ns: int | None = None,
        status_code: str = "OK",
        resource: _MockResource | None = None,
    ):
        self.name = name
        self.context = _MockContext(trace_id, span_id)
        self.parent = (
            _MockContext(trace_id, parent_span_id) if parent_span_id else None
        )
        self.attributes = attributes or {}
        self.start_time = start_time_ns or int(
            datetime.now(timezone.utc).timestamp() * 1e9
        )
        self.end_time = end_time_ns or (self.start_time + 100_000_000)  # +100ms
        self.status = _MockStatus(status_code)
        self.resource = resource or _MockResource()


def _now_ns() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1e9)


# ── Setup ────────────────────────────────────────────────────


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
    # register_manifest must return a dict with string manifest_id
    cfg._client.register_manifest.return_value = {"manifest_id": "test-manifest-id", "status": "active"}

    # Reset module-level manifest_id to prevent MagicMock leaks
    try:
        import decimalai.otel as otel_mod
        if hasattr(otel_mod, '_manifest_id'):
            otel_mod._manifest_id = None
    except Exception:
        pass
    yield


# ── Mock SpanExportResult so tests work without otel installed ──

_mock_export_result = MagicMock()
_mock_export_result.SUCCESS = "SUCCESS"


# ── Exporter Tests ───────────────────────────────────────────


class TestDecimalSpanExporter:
    """Tests for DecimalSpanExporter."""

    def test_export_single_trace(self):
        """A batch of spans with one trace_id → one RunTrace."""
        from decimalai.otel import DecimalSpanExporter
        import decimalai._config as cfg

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter(agent_name="test-agent")

            tid = 0x1234567890ABCDEF
            spans = [
                _MockSpan("agent-run", tid, 0x01),
                _MockSpan(
                    "llm-call",
                    tid,
                    0x02,
                    parent_span_id=0x01,
                    attributes={
                        "gen_ai.request.model": "gpt-4o",
                        "gen_ai.system": "openai",
                        "gen_ai.usage.input_tokens": 10,
                        "gen_ai.usage.output_tokens": 20,
                        "gen_ai.request.temperature": 0.7,
                    },
                ),
            ]

            result = exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            cfg._client.ingest_trace.assert_called_once()
            run_trace = cfg._client.ingest_trace.call_args[0][0]

            assert run_trace.agent_name == "test-agent"
            assert len(run_trace.llm_calls) == 1
            assert run_trace.llm_calls[0].model_name == "gpt-4o"
            assert run_trace.llm_calls[0].provider == "openai"
            assert run_trace.llm_calls[0].input_tokens == 10
            assert run_trace.llm_calls[0].output_tokens == 20
            assert run_trace.llm_calls[0].temperature == 0.7

    def _otel_mock(self):
        return patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        })

    def _run_and_get_snapshot(self, exporter, spans):
        import decimalai._config as cfg

        with self._otel_mock():
            exporter.export(spans)
        from decimalai._config import _sender

        _sender.flush()
        cfg._client.register_manifest.assert_called()
        return cfg, cfg._client.register_manifest.call_args[0][0]

    def _prompt_contents(self, snapshot):
        return [
            (c.schema_json or {}).get("content")
            for c in snapshot.components
            if c.component_type == "prompt"
        ]

    def test_genai_system_instructions_become_prompt(self):
        """The GenAI-semconv system instruction lands in the manifest."""
        from decimalai.otel import DecimalSpanExporter

        exporter = DecimalSpanExporter(agent_name="a")
        tid = 0xABC1
        spans = [
            _MockSpan("root", tid, 0x01),
            _MockSpan("llm", tid, 0x02, parent_span_id=0x01, attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.system": "openai",
                "gen_ai.system_instructions": "You are a triage bot.",
            }),
        ]
        cfg, snap = self._run_and_get_snapshot(exporter, spans)
        assert "You are a triage bot." in self._prompt_contents(snap)
        # And the manifest_id is stamped on the ingested trace.
        run_trace = cfg._client.ingest_trace.call_args[0][0]
        assert run_trace.manifest_id == "test-manifest-id"

    def test_openinference_input_messages_system_prompt(self):
        """The OpenInference role-indexed system message is captured
        (the CrewAI/LlamaIndex dialect), and the user message is NOT."""
        from decimalai.otel import DecimalSpanExporter

        exporter = DecimalSpanExporter(agent_name="a")
        tid = 0xABC2
        spans = [
            _MockSpan("root", tid, 0x01),
            _MockSpan("llm", tid, 0x02, parent_span_id=0x01, attributes={
                "llm.model_name": "gpt-4o",
                "llm.input_messages.0.message.role": "system",
                "llm.input_messages.0.message.content": "SYSTEM TEXT",
                "llm.input_messages.1.message.role": "user",
                "llm.input_messages.1.message.content": "hello there",
            }),
        ]
        _cfg, snap = self._run_and_get_snapshot(exporter, spans)
        contents = self._prompt_contents(snap)
        assert "SYSTEM TEXT" in contents
        assert "hello there" not in contents

    def test_first_system_prompt_wins_within_trace(self):
        """Hash-stability guard: only the FIRST system prompt in a
        trace is captured (later rendered ones would flip the hash)."""
        from decimalai.otel import DecimalSpanExporter

        exporter = DecimalSpanExporter(agent_name="a")
        tid = 0xABC3
        spans = [
            _MockSpan("root", tid, 0x01),
            _MockSpan("llm-1", tid, 0x02, parent_span_id=0x01, attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.system_instructions": "FIRST",
            }),
            _MockSpan("llm-2", tid, 0x03, parent_span_id=0x01, attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.system_instructions": "SECOND",
            }),
        ]
        _cfg, snap = self._run_and_get_snapshot(exporter, spans)
        contents = self._prompt_contents(snap)
        assert "FIRST" in contents
        assert "SECOND" not in contents

    def test_explicit_prompts_override_harvested(self):
        """install(prompts=...) / DecimalSpanExporter(prompts=...) wins over the
        rendered prompt auto-harvested from spans."""
        from decimalai.otel import DecimalSpanExporter

        exporter = DecimalSpanExporter(agent_name="a", prompts={"system": "STATIC TEMPLATE"})
        tid = 0xABC4
        spans = [
            _MockSpan("root", tid, 0x01),
            _MockSpan("llm", tid, 0x02, parent_span_id=0x01, attributes={
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.system_instructions": "RENDERED",
            }),
        ]
        _cfg, snap = self._run_and_get_snapshot(exporter, spans)
        contents = self._prompt_contents(snap)
        assert "STATIC TEMPLATE" in contents
        assert "RENDERED" not in contents

    def test_export_groups_by_trace_id(self):
        """Spans from 2 different traces → 2 separate RunTraces."""
        from decimalai.otel import DecimalSpanExporter
        import decimalai._config as cfg

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter(agent_name="test")

            tid_a = 0xAAAA
            tid_b = 0xBBBB

            spans = [
                _MockSpan("root-a", tid_a, 0x01),
                _MockSpan("root-b", tid_b, 0x02),
                _MockSpan(
                    "llm-a",
                    tid_a,
                    0x03,
                    parent_span_id=0x01,
                    attributes={"gen_ai.request.model": "gpt-4o"},
                ),
            ]

            exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            assert cfg._client.ingest_trace.call_count == 2

    def test_genai_attributes_mapped(self):
        """gen_ai.* attributes should map to LlmCallRecord fields."""
        from decimalai.otel import DecimalSpanExporter
        import decimalai._config as cfg

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter()
            tid = 0xCCCC

            spans = [
                _MockSpan(
                    "generation",
                    tid,
                    0x01,
                    attributes={
                        "gen_ai.request.model": "claude-3-sonnet",
                        "gen_ai.system": "anthropic",
                        "gen_ai.usage.input_tokens": 100,
                        "gen_ai.usage.output_tokens": 50,
                        "gen_ai.request.temperature": 0.3,
                        "gen_ai.request.max_tokens": 1024,
                    },
                ),
            ]

            exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            run_trace = cfg._client.ingest_trace.call_args[0][0]
            llm = run_trace.llm_calls[0]
            assert llm.model_name == "claude-3-sonnet"
            assert llm.provider == "anthropic"
            assert llm.input_tokens == 100
            assert llm.output_tokens == 50
            assert llm.temperature == 0.3
            assert llm.max_output_tokens == 1024

    def test_fallback_attribute_keys(self):
        """Fallback keys (llm.request.model, etc.) should work."""
        from decimalai.otel import DecimalSpanExporter
        import decimalai._config as cfg

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter()
            tid = 0xDDDD

            spans = [
                _MockSpan(
                    "llm-call",
                    tid,
                    0x01,
                    attributes={
                        "llm.request.model": "gpt-4o-mini",
                        "llm.system": "openai",
                        "llm.usage.prompt_tokens": 15,
                        "llm.usage.completion_tokens": 25,
                    },
                ),
            ]

            exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            run_trace = cfg._client.ingest_trace.call_args[0][0]
            llm = run_trace.llm_calls[0]
            assert llm.model_name == "gpt-4o-mini"
            assert llm.provider == "openai"
            assert llm.input_tokens == 15
            assert llm.output_tokens == 25

    def test_span_classification(self):
        """Non-LLM spans should be classified by name pattern."""
        from decimalai.otel import DecimalSpanExporter
        import decimalai._config as cfg

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter(agent_name="test")
            tid = 0xEEEE

            spans = [
                _MockSpan("tool_call: search", tid, 0x01),
                _MockSpan("agent: researcher", tid, 0x02),
                _MockSpan("retrieve_documents", tid, 0x03),
                _MockSpan("pipeline_step", tid, 0x04),
                _MockSpan("something_else", tid, 0x05),
            ]

            exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            run_trace = cfg._client.ingest_trace.call_args[0][0]
            types = [s.span_type.value for s in run_trace.spans]
            assert "tool" in types
            assert "agent" in types
            assert "retrieval" in types
            # pipeline and something_else both map to "other"
            assert types.count("other") == 2

    def test_error_status_propagation(self):
        """OTEL ERROR status should map to DecimalAI ERROR."""
        from decimalai.otel import DecimalSpanExporter
        import decimalai._config as cfg

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter(agent_name="test")
            tid = 0xFFFF

            spans = [
                _MockSpan(
                    "failed-call",
                    tid,
                    0x01,
                    attributes={"gen_ai.request.model": "gpt-4o"},
                    status_code="ERROR",
                ),
            ]

            exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            run_trace = cfg._client.ingest_trace.call_args[0][0]
            assert run_trace.llm_calls[0].status.value == "error"
            assert run_trace.status.value == "error"

    def test_auto_detect_agent_name_from_root(self):
        """Without default agent_name, use root span name."""
        from decimalai.otel import DecimalSpanExporter
        import decimalai._config as cfg

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter()  # no default agent_name
            tid = 0x1111

            spans = [
                _MockSpan(
                    "customer-service-agent",
                    tid,
                    0x01,
                    resource=_MockResource("my-cool-service"),
                ),
            ]

            exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            run_trace = cfg._client.ingest_trace.call_args[0][0]
            # Should use service name from resource
            assert run_trace.agent_name == "my-cool-service"

    def test_export_empty_batch(self):
        """Empty span batch should return SUCCESS."""
        from decimalai.otel import DecimalSpanExporter

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter()
            result = exporter.export([])
            assert result == "SUCCESS"

    def test_disabled_sdk_skips_send(self):
        """When SDK is disabled, traces should not be sent."""
        import decimalai._config as cfg
        cfg._config.enabled = False

        from decimalai.otel import DecimalSpanExporter

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter(agent_name="test")
            tid = 0x2222

            spans = [_MockSpan("root", tid, 0x01)]
            exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            cfg._client.ingest_trace.assert_not_called()

    def test_force_flush_returns_true(self):
        """force_flush should return True."""
        from decimalai.otel import DecimalSpanExporter

        exporter = DecimalSpanExporter()
        assert exporter.force_flush() is True

    def test_shutdown_is_noop(self):
        """shutdown should not error."""
        from decimalai.otel import DecimalSpanExporter

        exporter = DecimalSpanExporter()
        exporter.shutdown()  # Should not raise

    def test_finish_reason_parsing(self):
        """gen_ai.response.finish_reasons should map to FinishReason."""
        from decimalai.otel import DecimalSpanExporter
        import decimalai._config as cfg

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(),
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.trace": MagicMock(),
            "opentelemetry.sdk.trace.export": MagicMock(
                SpanExportResult=_mock_export_result
            ),
        }):
            exporter = DecimalSpanExporter(agent_name="test")
            tid = 0x3333

            spans = [
                _MockSpan(
                    "llm",
                    tid,
                    0x01,
                    attributes={
                        "gen_ai.request.model": "gpt-4o",
                        "gen_ai.response.finish_reasons": ["tool_calls"],
                    },
                ),
            ]

            exporter.export(spans)

            from decimalai._config import _sender
            _sender.flush()

            run_trace = cfg._client.ingest_trace.call_args[0][0]
            assert run_trace.llm_calls[0].finish_reason.value == "tool_calls"


# ── Install Tests ───────────────────────────────────────────


class TestOtelInstall:
    """Tests for the install() function."""

    def test_install_creates_provider(self):
        """install() should set up a TracerProvider with our exporter."""
        mock_provider_cls = MagicMock()
        mock_batch_cls = MagicMock()
        mock_trace_api = MagicMock()

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(trace=mock_trace_api),
            "opentelemetry.trace": mock_trace_api,
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(
                Resource=MagicMock(), SERVICE_NAME="service.name"
            ),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls
            ),
            "opentelemetry.sdk.trace.export": MagicMock(
                BatchSpanProcessor=mock_batch_cls
            ),
        }):
            from decimalai.otel import install
            install(agent_name="my-agent", service_name="test-svc")

            mock_provider_cls.assert_called_once()
            mock_batch_cls.assert_called_once()

    def test_install_without_otel_raises(self):
        """install() should raise ImportError if opentelemetry-sdk is missing."""
        saved = {}
        for key in list(sys.modules.keys()):
            if "opentelemetry" in key:
                saved[key] = sys.modules.pop(key)

        try:
            with patch.dict("sys.modules", {
                "opentelemetry": None,
                "opentelemetry.sdk": None,
                "opentelemetry.sdk.trace": None,
            }):
                from decimalai.otel import install
                with pytest.raises(ImportError, match="opentelemetry-sdk"):
                    install()
        finally:
            sys.modules.update(saved)


# ── Utility Tests ───────────────────────────────────────────


class TestOtelUtilities:
    """Tests for OTEL utility functions."""

    def test_classify_span(self):
        from decimalai.otel import _classify_span

        assert _classify_span("tool_call", {}).value == "tool"
        assert _classify_span("agent_run", {}).value == "agent"
        assert _classify_span("retrieve_docs", {}).value == "retrieval"
        assert _classify_span("pipeline_step", {}).value == "other"
        assert _classify_span("chat_completion", {}).value == "llm"
        assert _classify_span("unknown_op", {}).value == "other"

    def test_infer_provider(self):
        from decimalai.otel import _infer_provider

        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("claude-3") == "anthropic"
        assert _infer_provider("gemini-2.0") == "google"
        assert _infer_provider("mistral-large") == "mistral"
        assert _infer_provider("llama-3.1") == "meta"
        assert _infer_provider("command-r") == "cohere"
        assert _infer_provider("unknown-model") is None
        assert _infer_provider(None) is None

    def test_ns_to_datetime(self):
        from decimalai.otel import _ns_to_datetime

        assert _ns_to_datetime(None) is None
        assert _ns_to_datetime(0) is None

        ts = int(datetime(2025, 6, 15, tzinfo=timezone.utc).timestamp() * 1e9)
        dt = _ns_to_datetime(ts)
        assert dt is not None
        assert dt.year == 2025
        assert dt.month == 6

    def test_get_first(self):
        from decimalai.otel import _get_first

        attrs = {"gen_ai.system": "openai", "llm.system": "fallback"}
        assert _get_first(attrs, "gen_ai.system") == "openai"
        assert _get_first(attrs, "missing", "llm.system") == "fallback"
        assert _get_first(attrs, "missing1", "missing2") is None


# ── Init Integration ────────────────────────────────────────


class TestInitOtelIntegration:
    """Test that init(otel=True) works."""

    def test_init_with_otel_flag(self):
        """init(otel=True) should call otel.install()."""
        mock_provider_cls = MagicMock()
        mock_batch_cls = MagicMock()
        mock_trace_api = MagicMock()

        with patch.dict("sys.modules", {
            "opentelemetry": MagicMock(trace=mock_trace_api),
            "opentelemetry.trace": mock_trace_api,
            "opentelemetry.sdk": MagicMock(),
            "opentelemetry.sdk.resources": MagicMock(
                Resource=MagicMock(), SERVICE_NAME="service.name"
            ),
            "opentelemetry.sdk.trace": MagicMock(
                TracerProvider=mock_provider_cls
            ),
            "opentelemetry.sdk.trace.export": MagicMock(
                BatchSpanProcessor=mock_batch_cls
            ),
        }):
            import decimalai
            import decimalai._config as cfg
            cfg._config = None
            cfg._client = None

            decimalai.init(
                api_key="dai_sk_test",
                base_url="http://localhost:8000",
                otel=True,
            )

            mock_provider_cls.assert_called_once()
