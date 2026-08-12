"""Tests for SDK future-proofing features.

Covers: streaming token buffering, multi-modal content_type detection,
structured output response_format capture, session aggregation,
eval versioning, and batch_eval.
"""

import json
import time
from datetime import datetime, timezone
from threading import Thread
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


# ── Setup ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_sdk_state():
    """Reset all SDK global state before each test."""
    import decimalai._config as cfg
    from decimalai._config import DecimalConfig

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    cfg._sender._pending = []
    yield
    cfg._config = None
    cfg._client = None


# ── 1. Streaming Support ─────────────────────────────────────────


class TestStreaming:
    """Streaming token buffering in LangChain handler."""

    def test_token_buffering(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="stream-test", auto_send=False)
        root = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {"input": "hello"}, run_id=root)
        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "openai"]},
            [[MagicMock(type="human")]],
            run_id=llm_id,
            parent_run_id=root,
            invocation_params={"model_name": "gpt-4o"},
        )

        # Stream tokens
        for token in ["Hello", " ", "world", "!"]:
            handler.on_llm_new_token(token, run_id=llm_id)

        # End the LLM call
        handler.on_llm_end(MagicMock(generations=[], llm_output={}), run_id=llm_id)
        handler.on_chain_end({"output": "Hello world!"}, run_id=root)

        trace = handler.get_trace()
        call = trace.llm_calls[0]

        assert call.streaming is True
        assert call.streaming_token_count == 4
        assert call.output.get("streaming_content") == "Hello world!"

    def test_non_streaming_stays_false(self):
        from decimalai.schema.trace import LlmCallRecord

        call = LlmCallRecord(model_name="gpt-4o")
        assert call.streaming is False
        assert call.streaming_token_count is None

    def test_streaming_serializes(self):
        from decimalai.schema.trace import LlmCallRecord

        call = LlmCallRecord(
            model_name="gpt-4o",
            streaming=True,
            streaming_token_count=42,
        )
        data = json.loads(call.model_dump_json())
        assert data["streaming"] is True
        assert data["streaming_token_count"] == 42


# ── 2. Multi-Modal Content Type ─────────────────────────────────


class TestMultiModal:
    """Content type detection for multi-modal messages."""

    def test_default_is_text(self):
        from decimalai.schema.trace import LlmCallRecord

        call = LlmCallRecord(model_name="gpt-4o")
        assert call.content_type == "text"

    def test_image_content_type(self):
        from decimalai.schema.trace import LlmCallRecord

        call = LlmCallRecord(model_name="gpt-4o-vision", content_type="image")
        data = json.loads(call.model_dump_json())
        assert data["content_type"] == "image"

    def test_multimodal_content_type(self):
        from decimalai.schema.trace import LlmCallRecord

        call = LlmCallRecord(model_name="gpt-4o", content_type="multimodal")
        assert call.content_type == "multimodal"

    def test_generic_log_llm_call_content_type(self):
        import decimalai

        with decimalai.start_trace(agent_name="mm-test", auto_send=False) as ctx:
            ctx.log_llm_call(
                model="gpt-4o-vision",
                content_type="multimodal",
                input_tokens=100,
                output_tokens=50,
            )

        trace = ctx.build_trace()
        assert trace.llm_calls[0].content_type == "multimodal"


# ── 3. Structured Output Capture ─────────────────────────────────


class TestStructuredOutput:
    """response_format field on LlmCallRecord."""

    def test_response_format_field(self):
        from decimalai.schema.trace import LlmCallRecord

        fmt = {"type": "json_schema", "json_schema": {"name": "Response", "schema": {"type": "object"}}}
        call = LlmCallRecord(model_name="gpt-4o", response_format=fmt)
        data = json.loads(call.model_dump_json())
        assert data["response_format"]["type"] == "json_schema"
        assert data["response_format"]["json_schema"]["name"] == "Response"

    def test_response_format_none_by_default(self):
        from decimalai.schema.trace import LlmCallRecord

        call = LlmCallRecord(model_name="gpt-4o")
        assert call.response_format is None

    def test_response_format_from_invocation_params(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="struct-test", auto_send=False)
        root = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "Agent"}, {"input": "hello"}, run_id=root)
        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "openai"]},
            [[MagicMock(type="human")]],
            run_id=llm_id,
            parent_run_id=root,
            invocation_params={
                "model_name": "gpt-4o",
                "response_format": {"type": "json_object"},
            },
        )
        handler.on_llm_end(MagicMock(generations=[], llm_output={}), run_id=llm_id)
        handler.on_chain_end({"output": "ok"}, run_id=root)

        trace = handler.get_trace()
        call = trace.llm_calls[0]
        assert call.response_format == {"type": "json_object"}

    def test_generic_log_with_response_format(self):
        import decimalai

        with decimalai.start_trace(agent_name="fmt-test", auto_send=False) as ctx:
            ctx.log_llm_call(
                model="gpt-4o",
                response_format={"type": "json_object"},
            )

        trace = ctx.build_trace()
        assert trace.llm_calls[0].response_format == {"type": "json_object"}


# ── 4. Session Aggregation ───────────────────────────────────────


class TestSessionAggregation:
    """Session metadata and turn_index on traces."""

    def test_session_metadata_on_trace(self):
        from decimalai.schema.trace import RunTrace

        trace = RunTrace(
            agent_name="support",
            session_id="sess-123",
            session_metadata={"user_id": "u-42", "channel": "web"},
            turn_index=3,
        )
        data = json.loads(trace.model_dump_json())
        assert data["session_id"] == "sess-123"
        assert data["session_metadata"]["user_id"] == "u-42"
        assert data["turn_index"] == 3

    def test_session_metadata_default_empty(self):
        from decimalai.schema.trace import RunTrace

        trace = RunTrace(agent_name="test")
        assert trace.session_metadata == {}
        assert trace.turn_index is None

    def test_start_trace_with_session(self):
        import decimalai

        with decimalai.start_trace(
            agent_name="session-test",
            session_id="sess-001",
            session_metadata={"user_id": "u-99"},
            turn_index=2,
            auto_send=False,
        ) as ctx:
            ctx.set_input("hello")

        trace = ctx.build_trace()
        assert trace.session_id == "sess-001"
        assert trace.session_metadata == {"user_id": "u-99"}
        assert trace.turn_index == 2

    def test_set_session_metadata(self):
        import decimalai

        with decimalai.start_trace(agent_name="meta-test", auto_send=False) as ctx:
            ctx.set_session_metadata({"user_id": "u-1"})
            ctx.set_session_metadata({"channel": "slack"})

        trace = ctx.build_trace()
        assert trace.session_metadata == {"user_id": "u-1", "channel": "slack"}

    def test_trace_decorator_with_session(self):
        import decimalai

        @decimalai.trace(
            agent_name="dec-sess",
            session_id="sess-dec",
            session_metadata={"org": "acme"},
            turn_index=1,
            auto_send=False,
        )
        def my_func():
            return "ok"

        my_func()
        # No assertion needed — just verify it doesn't crash


# ── 5. Eval Versioning ──────────────────────────────────────────


class TestEvalVersioning:
    """Eval version field on DecimalEval."""

    def test_default_version_is_1(self):
        from decimalai.evals import DecimalEval, TraceData

        @lambda f: DecimalEval(f)
        def my_eval(trace: TraceData) -> bool:
            return True

        assert my_eval.version == "1"

    def test_custom_version(self):
        from decimalai.evals import eval, TraceData

        @eval(name="check_v2", version="2")
        def check_v2(trace: TraceData) -> bool:
            return True

        assert check_v2.version == "2"

    def test_version_in_score_dicts(self):
        from decimalai.evals import eval, TraceData

        @eval(name="scored_eval", version="3")
        def scored(trace: TraceData) -> float:
            return 0.9

        td = TraceData(id="test", input="hello", output="world", status="success")
        scores = scored.to_score_dicts(td)
        assert len(scores) == 1
        assert scores[0]["eval_version"] == "3"

    def test_version_in_repr(self):
        from decimalai.evals import eval, TraceData

        @eval(name="repr_test", version="2.1")
        def my_check(trace: TraceData) -> bool:
            return True

        assert "version='2.1'" in repr(my_check)


# ── 6. Batch Eval ───────────────────────────────────────────────


class TestBatchEval:
    """batch_eval() function for offline eval passes."""

    def test_batch_eval_import(self):
        from decimalai.evals import batch_eval
        assert callable(batch_eval)

    def test_batch_eval_re_exported(self):
        import decimalai
        assert hasattr(decimalai, "batch_eval")
        assert callable(decimalai.batch_eval)

    def test_batch_eval_runs(self):
        """batch_eval should fetch traces, run evals, and push scores."""
        from decimalai.evals import batch_eval, eval, TraceData

        @eval(name="always_pass")
        def always_pass(trace: TraceData) -> bool:
            return True

        # Mock client
        mock_client = MagicMock()

        # Mock trace response
        trace_data = {
            "id": "trace-1",
            "agent_name": "test",
            "status": "success",
            "user_input_preview": "hello",
            "final_output_preview": "world",
            "llm_calls": [],
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = trace_data
        mock_resp.raise_for_status = MagicMock()
        mock_client._request_with_retry.return_value = mock_resp

        results = batch_eval(
            trace_ids=["trace-1"],
            evals=[always_pass],
            client=mock_client,
        )

        assert results["traces_evaluated"] == 1
        assert results["traces_requested"] == 1
        assert results["total_scores"] == 1
        assert results["summary"]["always_pass"]["passed"] == 1
        assert results["summary"]["always_pass"]["failed"] == 0

        # Should have pushed scores
        mock_client.push_eval_scores.assert_called_once()

    def test_batch_eval_handles_fetch_failure(self):
        """batch_eval should handle trace fetch failures gracefully."""
        from decimalai.evals import batch_eval, eval, TraceData

        @eval(name="check")
        def check(trace: TraceData) -> bool:
            return True

        mock_client = MagicMock()
        mock_client._request_with_retry.side_effect = Exception("network error")

        results = batch_eval(
            trace_ids=["bad-id"],
            evals=[check],
            client=mock_client,
        )

        assert results["traces_evaluated"] == 0
        assert results["traces_requested"] == 1
        assert results["total_scores"] == 0
