"""Tests for SDK audit improvements.

Covers: ContextVar tracing, background sender, handler isolation,
new span types, cost_usd, score() shorthand, structured TraceData,
eval adapter move, A2A card export, and atexit flush.
"""

import asyncio
import os
import json
import time
from datetime import datetime, timezone
from threading import Thread
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


# ── Setup: reset global state before each test ────────────────────


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
    # Reset the sender
    cfg._sender._pending = []
    yield
    # Teardown
    cfg._config = None
    cfg._client = None


# ── ContextVar trace context ──────────────────────────────────────


class TestContextVar:
    """ContextVar-based trace context is async-safe."""

    def test_independent_traces_in_threads(self):
        """Two threads should get independent trace contexts."""
        import decimalai
        from decimalai.generic import _get_current_trace

        traces_seen = {}

        def run_in_thread(name):
            with decimalai.start_trace(agent_name=name, auto_send=False) as ctx:
                ctx.set_input(f"input-{name}")
                traces_seen[name] = ctx.agent_name
                time.sleep(0.05)  # overlap
                # Should still be our trace
                current = _get_current_trace()
                assert current is ctx
                assert current.agent_name == name

        t1 = Thread(target=run_in_thread, args=("agent-a",))
        t2 = Thread(target=run_in_thread, args=("agent-b",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert traces_seen == {"agent-a": "agent-a", "agent-b": "agent-b"}

    def test_contextvar_restores_previous(self):
        """Nested trace contexts should restore the previous context."""
        import decimalai
        from decimalai.generic import _get_current_trace

        with decimalai.start_trace(agent_name="outer", auto_send=False) as outer:
            assert _get_current_trace() is outer
            with decimalai.start_trace(agent_name="inner", auto_send=False) as inner:
                assert _get_current_trace() is inner
            assert _get_current_trace() is outer
        assert _get_current_trace() is None


# ── Background sender ─────────────────────────────────────────────


class TestBackgroundSender:
    """Background sender should not block the calling thread."""

    def test_sender_submits_work(self):
        from decimalai._config import BackgroundSender

        results = []
        sender = BackgroundSender()
        sender.submit(lambda: results.append("done"))
        sender.flush()
        assert results == ["done"]

    def test_sender_flush_waits(self):
        from decimalai._config import BackgroundSender

        results = []
        sender = BackgroundSender()
        sender.submit(lambda: (time.sleep(0.1), results.append("late")))
        assert results == []  # Not yet done
        sender.flush()
        assert "late" in [r for r in results if r == "late"]

    def test_trace_uses_background_send(self):
        """@trace should use background sender, not block."""
        import decimalai
        import decimalai._config as cfg

        @decimalai.trace(agent_name="bg-test")
        def my_func():
            return "ok"

        my_func()
        # The sender should have been used
        cfg._sender.flush()
        # ingest_trace should have been called
        cfg._client.ingest_trace.assert_called_once()


# ── atexit flush ─────────────────────────────────────────────────


class TestAtexitFlush:
    """The _shutdown function should flush and close."""

    def test_shutdown_flushes_sender(self):
        from decimalai._config import _shutdown, _sender

        # Submit something
        results = []
        _sender.submit(lambda: results.append("flushed"))
        _shutdown()
        assert results == ["flushed"]


# ── Handler isolation ────────────────────────────────────────────


class TestHandlerIsolation:
    """Handler should reset state for each root invocation."""

    def test_handler_resets_on_new_root(self):
        from decimalai.langchain import CallbackHandler
        from uuid import uuid4

        handler = CallbackHandler(agent_name="test", auto_send=False)

        # First invocation
        root1 = uuid4()
        handler.on_chain_start(
            {"name": "Chain1"}, {"input": "hello"}, run_id=root1
        )
        handler.on_chain_end({"output": "world"}, run_id=root1)
        trace1 = handler.get_trace()

        # Second invocation — should get fresh state
        root2 = uuid4()
        handler.on_chain_start(
            {"name": "Chain2"}, {"input": "foo"}, run_id=root2
        )
        handler.on_chain_end({"output": "bar"}, run_id=root2)
        trace2 = handler.get_trace()

        assert trace1.spans[0].name == "Chain1"
        assert trace2.spans[0].name == "Chain2"
        assert len(trace2.spans) == 1  # No leftover from trace1


# ── New span types ──────────────────────────────────────────────


class TestNewSpanTypes:
    """New span types should exist and serialize correctly."""

    def test_handoff_span(self):
        from decimalai.schema.common import SpanType
        assert SpanType.HANDOFF == "handoff"

    def test_guardrail_span(self):
        from decimalai.schema.common import SpanType
        assert SpanType.GUARDRAIL == "guardrail"

    def test_memory_span(self):
        from decimalai.schema.common import SpanType
        assert SpanType.MEMORY == "memory"

    def test_planning_span(self):
        from decimalai.schema.common import SpanType
        assert SpanType.PLANNING == "planning"

    def test_span_in_trace(self):
        from decimalai.schema.common import SpanType
        from decimalai.schema.trace import TraceSpan

        span = TraceSpan(
            name="transfer_to_billing",
            span_type=SpanType.HANDOFF,
        )
        data = json.loads(span.model_dump_json())
        assert data["span_type"] == "handoff"


# ── cost_usd field ───────────────────────────────────────────────


class TestCostField:
    """LlmCallRecord should support cost_usd."""

    def test_cost_field_serializes(self):
        from decimalai.schema.trace import LlmCallRecord

        call = LlmCallRecord(
            model_name="gpt-4o",
            cost_usd=0.003,
            input_tokens=100,
            output_tokens=50,
        )
        data = json.loads(call.model_dump_json())
        assert data["cost_usd"] == 0.003

    def test_log_llm_call_with_cost(self):
        import decimalai

        with decimalai.start_trace(agent_name="cost-test", auto_send=False) as ctx:
            ctx.log_llm_call(
                model="gpt-4o",
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.003,
            )

        trace = ctx.build_trace()
        assert trace.llm_calls[0].cost_usd == 0.003


# ── score() shorthand ───────────────────────────────────────────


class TestScoreShorthand:
    """decimalai.score() should be a convenient way to push a score."""

    def test_score_calls_push(self):
        import decimalai
        import decimalai._config as cfg

        cfg._client.push_eval_scores = MagicMock(return_value={"ok": True})

        result = decimalai.score("trace-123", "accuracy", 0.85)

        cfg._client.push_eval_scores.assert_called_once()
        call_args = cfg._client.push_eval_scores.call_args
        assert call_args[1]["trace_id"] == "trace-123"
        assert call_args[1]["scores"][0]["name"] == "accuracy"
        assert call_args[1]["scores"][0]["score"] == 0.85

    def test_score_with_reason(self):
        import decimalai
        import decimalai._config as cfg

        cfg._client.push_eval_scores = MagicMock(return_value={})

        decimalai.score("trace-123", "quality", 0.9, reason="Good output")

        call_args = cfg._client.push_eval_scores.call_args
        assert call_args[1]["scores"][0]["passed"] is True


# ── Structured TraceData ─────────────────────────────────────────


class TestStructuredTraceData:
    """TraceData.input and output should accept dicts."""

    def test_dict_input_output(self):
        from decimalai.evals import TraceData

        td = TraceData(
            id="test",
            input={"query": "hello", "context": "world"},
            output={"result": "answer", "confidence": 0.95},
            status="success",
        )
        assert isinstance(td.input, dict)
        assert td.input["query"] == "hello"
        assert td.output["confidence"] == 0.95

    def test_str_input_still_works(self):
        from decimalai.evals import TraceData

        td = TraceData(
            id="test",
            input="hello",
            output="world",
            status="success",
        )
        assert td.input == "hello"
        assert td.output == "world"


# ── Eval adapter move ───────────────────────────────────────────


class TestEvalAdapterMove:
    """Eval adapters should be importable from new location."""

    def test_import_from_new_path(self):
        from decimalai.evals.adapters import (
            push_deepeval_results,
            push_langsmith_scores,
            push_custom_scores,
        )
        assert callable(push_deepeval_results)
        assert callable(push_langsmith_scores)
        assert callable(push_custom_scores)

    def test_re_export_from_init(self):
        """They should still be accessible from decimalai directly."""
        import decimalai
        assert callable(decimalai.push_deepeval_results)
        assert callable(decimalai.push_langsmith_scores)
        assert callable(decimalai.push_custom_scores)

    def test_old_path_removed(self):
        """The old decimalai.eval module should not exist."""
        with pytest.raises(ModuleNotFoundError):
            import importlib
            importlib.import_module("decimalai.eval")


# ── A2A card export ─────────────────────────────────────────────


class TestA2ACard:
    """ManifestSnapshot.to_a2a_card() should export valid card."""

    def test_basic_card(self):
        from decimalai.schema.manifest import ManifestSnapshot, ComponentSnapshot

        snapshot = ManifestSnapshot(
            agent_name="support-agent",
            manifest_hash="abc123def456",
            version_label="v1.2.0",
            components=[
                ComponentSnapshot(
                    component_type="tool",
                    component_name="search_db",
                ),
                ComponentSnapshot(
                    component_type="tool",
                    component_name="send_email",
                ),
                ComponentSnapshot(
                    component_type="model",
                    component_name="gpt-4o",
                ),
                ComponentSnapshot(
                    component_type="prompt",
                    component_name="system_prompt",
                ),
            ],
        )

        card = snapshot.to_a2a_card(url="https://agent.example.com/a2a")

        assert card["name"] == "support-agent"
        assert card["version"] == "v1.2.0"
        assert card["protocolVersion"]
        assert card["url"] == "https://agent.example.com/a2a"
        # A2A-canonical shape: capabilities is a feature dict, not a tool list.
        assert isinstance(card["capabilities"], dict)
        assert {"streaming", "pushNotifications", "stateTransitionHistory"} <= set(card["capabilities"])
        # The x-agentversion provenance block pins the exact versioned manifest.
        prov = card["x-agentversion"]
        assert prov["overall_hash"].startswith("sha256:")  # canonical jcs-sha256
        assert prov["manifest_id"].startswith("amf_")
        assert prov["spec_version"]

    def test_card_without_version_label(self):
        from decimalai.schema.manifest import ManifestSnapshot

        snapshot = ManifestSnapshot(
            agent_name="test",
            manifest_hash="abc123def456789",
        )
        card = snapshot.to_a2a_card()
        assert card["version"] == "abc123def456"  # First 12 chars of hash (vlabel fallback)


# ── Thread-safe manifest ────────────────────────────────────────


class TestThreadSafeManifest:
    """Manifest registration lock should exist."""

    def test_lock_exists(self):
        from decimalai.langchain import _manifest_lock
        import threading
        assert isinstance(_manifest_lock, type(threading.Lock()))


# ── Auto-detect agent name ──────────────────────────────────────


class TestAutoDetectAgentName:
    """Handler should auto-detect agent name from root chain."""

    def test_detects_from_chain_name(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(auto_send=False)
        assert handler.agent_name is None

        handler.on_chain_start(
            {"name": "SupportAgent"},
            {"input": "hello"},
            run_id=uuid4(),
        )
        assert handler.agent_name == "SupportAgent"


# ── init(langchain=True) ────────────────────────────────────────


class TestInitLangchain:
    """init(langchain=True) should auto-install LangChain tracing."""

    def test_init_with_langchain(self):
        import decimalai._config as cfg

        # Reset
        cfg._config = None
        cfg._client = None

        with patch("decimalai.langchain.instrument") as mock_install:
            with patch("decimalai.langchain.register_configure_hook", create=True):
                import decimalai
                os.environ["DECIMAL_API_KEY"] = "dai_sk_test"
                try:
                    decimalai.init(langchain=True, agent_name="test-agent")
                    mock_install.assert_called_once_with(agent_name="test-agent")
                finally:
                    os.environ.pop("DECIMAL_API_KEY", None)


# ── Version bump ────────────────────────────────────────────────


class TestVersion:
    def test_version_matches_pyproject(self):
        """`__init__.__version__` and pyproject's `version` must agree.

        This used to hard-code the literal current version, which meant EVERY release
        failed CI until someone remembered to edit this line — it blocked decimalai
        0.9.1 in exactly that way, and the CI logs were unavailable, so the cause took
        a clean-room reproduction to find.

        Asserting the two sources AGREE keeps the real protection and removes the
        release blocker. The real bug is them drifting apart: the version lives in two
        files, and bumping only pyproject builds a wheel that reports the old version
        to every caller — in User-Agent headers and telemetry — which PyPI cannot
        correct in place. That drift happened during this same release and the release
        script's wheel smoke caught it; this test now catches it earlier, and for free.
        """
        # A regex, not tomllib: tomllib is stdlib only from 3.11 and CI matrixes 3.10,
        # so importing it here fails the whole build on the oldest supported Python.
        # (Found the hard way — the first version of this fix did exactly that.)
        # Not importlib.metadata either: that reads INSTALLED metadata, which can be
        # stale against the working tree, which is the very staleness that made this
        # test pass locally while failing in CI.
        import re
        from pathlib import Path

        import decimalai

        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.M)
        assert m, "could not find `version` in pyproject.toml"
        declared = m.group(1)
        assert decimalai.__version__ == declared, (
            f"__init__.py says {decimalai.__version__}, pyproject says {declared} — "
            "bump both or the published wheel misreports its own version"
        )
