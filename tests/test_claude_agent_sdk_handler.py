"""Unit tests for the Claude Agent SDK tracer (decimalai.claude_agent_sdk).

Drives ``trace_stream`` with a *synthetic* message stream — no claude-agent-sdk
install, no Claude Code CLI, no network, no API key. The tracer dispatches on
``type(message).__name__``, so plain local classes named ``SystemMessage`` /
``AssistantMessage`` / ``UserMessage`` / ``ResultMessage`` (and the block types)
reproduce a real query() stream by hand. We then assert on the RunTrace that
lands at the (mocked) backend client.

Mirrors test_llamaindex_handler.py: a mock client on the global config, read the
RunTrace from ingest_trace.call_args. The wrapper is an async generator, so each
test drives it via ``asyncio.run`` (the repo sets no pytest-asyncio mode).
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import MagicMock

import pytest

from decimalai.claude_agent_sdk import trace_stream, traced_query


# ── Synthetic Claude Agent SDK messages (dispatch is on class name) ──

class SystemMessage:
    def __init__(self, subtype="init", data=None):
        self.subtype = subtype
        self.data = data or {}


class AssistantMessage:
    def __init__(self, content, model="claude-haiku-4-5"):
        self.content = content
        self.model = model


class UserMessage:
    def __init__(self, content):
        self.content = content


class ResultMessage:
    def __init__(self, result=None, usage=None, total_cost_usd=None,
                 session_id=None, is_error=False):
        self.result = result
        self.usage = usage
        self.total_cost_usd = total_cost_usd
        self.session_id = session_id
        self.is_error = is_error


class TextBlock:
    def __init__(self, text):
        self.text = text


class ThinkingBlock:
    def __init__(self, thinking="...", signature=""):
        self.thinking = thinking
        self.signature = signature


class ToolUseBlock:
    def __init__(self, id, name, input):
        self.id = id
        self.name = name
        self.input = input


class ToolResultBlock:
    def __init__(self, tool_use_id, content, is_error=False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class FakeOptions:
    """Stand-in for ClaudeAgentOptions — only the fields _build_manifest reads."""
    def __init__(self, model=None, system_prompt=None, allowed_tools=None, agents=None):
        self.model = model
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools
        self.agents = agents


def _full_run_messages():
    """A canonical tool-using run: init → plan(tool_use) → tool_result →
    final answer → result."""
    return [
        SystemMessage(subtype="init", data={"session_id": "sess_123"}),
        AssistantMessage(content=[
            TextBlock("Let me search."),
            ToolUseBlock(id="tu_1", name="search", input={"q": "eiffel height"}),
        ]),
        UserMessage(content=[ToolResultBlock(tool_use_id="tu_1", content="330 meters")]),
        AssistantMessage(content=[TextBlock("The Eiffel Tower is 330 meters tall.")]),
        ResultMessage(
            result="The Eiffel Tower is 330 meters tall.",
            usage={"input_tokens": 120, "output_tokens": 18},
            total_cost_usd=0.0021,
            session_id="sess_123",
            is_error=False,
        ),
    ]


# ── Async driving helpers ────────────────────────────────────────────

async def _stream(messages):
    for m in messages:
        yield m


def _drive(messages, **kwargs):
    """Run the wrapped stream to completion, returning the messages it yielded."""
    async def run():
        out = []
        async for m in trace_stream(_stream(messages), **kwargs):
            out.append(m)
        return out
    return asyncio.run(run())


# ── SDK reset (mirror test_llamaindex_handler._reset_sdk) ────────────

@pytest.fixture(autouse=True)
def _reset_sdk():
    import decimalai._config as cfg
    import decimalai.claude_agent_sdk as cas
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {"manifest_id": "man_test"}
    # Reset module-level manifest + install state so tests don't bleed into each other.
    cas._manifest_tracker = ManifestTracker()
    cas._manifest_id = None
    cas._install_agent_name = None
    cas._query_patched = False
    yield


def _flush_and_get_trace():
    """Drain the background sender and return the single captured RunTrace."""
    import decimalai._config as cfg
    from decimalai._config import _sender

    _sender.flush()
    assert cfg._client.ingest_trace.called, "ingest_trace was never called"
    return cfg._client.ingest_trace.call_args[0][0]


# ── Stream → RunTrace ─────────────────────────────────────────────────

class TestTraceStream:
    def test_full_run_produces_one_trace(self):
        """A complete tool-using run yields one RunTrace with both model turns
        as llm_calls, the tool call captured, and session/output recorded."""
        import decimalai._config as cfg
        _drive(_full_run_messages(), agent_name="claude-dev",
               user_input="How tall is the Eiffel Tower?")

        trace = _flush_and_get_trace()
        cfg._client.ingest_trace.assert_called_once()

        assert trace.agent_name == "claude-dev"
        assert trace.status.value == "success"
        assert trace.session_id == "sess_123"
        assert trace.user_input_preview == "How tall is the Eiffel Tower?"
        assert trace.final_output_preview == "The Eiffel Tower is 330 meters tall."

        # Two model turns → two LLM calls.
        assert len(trace.llm_calls) == 2
        plan, answer = trace.llm_calls
        assert plan.call_role.value == "planner"
        assert plan.finish_reason.value == "tool_calls"
        assert plan.provider == "anthropic"
        assert plan.model_name == "claude-haiku-4-5"
        assert len(plan.tool_calls) == 1
        assert plan.tool_calls[0].tool_name == "search"
        assert plan.tool_calls[0].args == {"q": "eiffel height"}
        assert plan.tool_calls[0].result == "330 meters"
        assert plan.tool_calls[0].latency_ms is not None

        assert answer.call_role.value == "responder"
        assert answer.finish_reason.value == "stop"
        assert answer.output["content"] == "The Eiffel Tower is 330 meters tall."

        # One TOOL span, closed by the ToolResultBlock.
        tool_spans = [s for s in trace.spans if s.span_type.value == "tool"]
        assert len(tool_spans) == 1
        assert tool_spans[0].name == "search"
        assert tool_spans[0].status.value == "success"
        assert tool_spans[0].output_preview == "330 meters"

    def test_cumulative_usage_attaches_to_last_call(self):
        """ResultMessage usage + cost are run totals → attached to the final
        LLM call only (the first turn carries no token counts)."""
        _drive(_full_run_messages(), agent_name="cas")
        trace = _flush_and_get_trace()

        plan, answer = trace.llm_calls
        assert plan.input_tokens is None and plan.output_tokens is None
        assert answer.input_tokens == 120
        assert answer.output_tokens == 18
        assert answer.cost_usd == 0.0021

    def test_tool_error_marks_span_not_run(self):
        """A ToolResultBlock is_error marks that tool/span ERROR but does NOT
        fail the run — the agent recovered and ResultMessage is_error is False."""
        msgs = [
            SystemMessage(data={"session_id": "s"}),
            AssistantMessage(content=[ToolUseBlock(id="tu_1", name="search", input={})]),
            UserMessage(content=[ToolResultBlock(tool_use_id="tu_1", content="boom", is_error=True)]),
            AssistantMessage(content=[TextBlock("recovered, here is the answer")]),
            ResultMessage(result="recovered, here is the answer", session_id="s", is_error=False),
        ]
        _drive(msgs, agent_name="cas")
        trace = _flush_and_get_trace()

        assert trace.status.value == "success"  # run recovered
        tool_span = [s for s in trace.spans if s.span_type.value == "tool"][0]
        assert tool_span.status.value == "error"
        assert trace.llm_calls[0].tool_calls[0].status.value == "error"

    def test_result_is_error_marks_run_error(self):
        """ResultMessage.is_error is the authoritative run status."""
        msgs = [
            SystemMessage(data={"session_id": "s"}),
            AssistantMessage(content=[TextBlock("partial")]),
            ResultMessage(result="failed", session_id="s", is_error=True),
        ]
        _drive(msgs, agent_name="cas")
        trace = _flush_and_get_trace()
        assert trace.status.value == "error"

    def test_thinking_block_ignored(self):
        """ThinkingBlocks carry no trace-relevant fields and are skipped without
        breaking the surrounding text capture."""
        msgs = [
            SystemMessage(data={"session_id": "s"}),
            AssistantMessage(content=[ThinkingBlock("hmm, let me reason"), TextBlock("answer")]),
            ResultMessage(result="answer", session_id="s"),
        ]
        _drive(msgs, agent_name="cas")
        trace = _flush_and_get_trace()
        assert len(trace.llm_calls) == 1
        assert trace.llm_calls[0].output["content"] == "answer"

    def test_default_agent_name(self):
        """No agent_name and no install() name → the 'claude-agent' default."""
        _drive([
            SystemMessage(data={"session_id": "s"}),
            ResultMessage(result="x", session_id="s"),
        ])
        trace = _flush_and_get_trace()
        assert trace.agent_name == "claude-agent"


# ── Manifest registration ─────────────────────────────────────────────

class TestManifest:
    def test_manifest_registered_from_options(self):
        """A run carrying ClaudeAgentOptions (model/prompt/tools/agents)
        registers a manifest and stamps its id on the trace."""
        import decimalai._config as cfg
        opts = FakeOptions(
            model="claude-haiku-4-5",
            system_prompt="You are a helpful assistant.",
            allowed_tools=["search", "calc"],
            agents={"researcher": object()},
        )
        _drive(_full_run_messages(), agent_name="cas", options=opts)
        trace = _flush_and_get_trace()

        cfg._client.register_manifest.assert_called_once()
        assert trace.manifest_id == "man_test"

    def test_manifest_registered_from_init_data(self):
        """When no options are supplied, model + tools fall back to the init
        SystemMessage data."""
        import decimalai._config as cfg
        msgs = [
            SystemMessage(data={"session_id": "s", "model": "claude-haiku-4-5", "tools": ["search"]}),
            AssistantMessage(content=[TextBlock("done")]),
            ResultMessage(result="done", session_id="s"),
        ]
        _drive(msgs, agent_name="cas")
        trace = _flush_and_get_trace()

        cfg._client.register_manifest.assert_called_once()
        assert trace.manifest_id == "man_test"

    def test_no_manifest_when_no_config(self):
        """Init data without model/tools and no options → no manifest registered."""
        import decimalai._config as cfg
        msgs = [
            SystemMessage(data={"session_id": "s"}),
            AssistantMessage(content=[TextBlock("hi")]),
            ResultMessage(result="hi", session_id="s"),
        ]
        _drive(msgs, agent_name="cas")
        trace = _flush_and_get_trace()

        cfg._client.register_manifest.assert_not_called()
        assert trace.manifest_id is None


# ── Stream lifecycle edge cases ───────────────────────────────────────

class TestLifecycle:
    def test_stream_exception_marks_error_and_reraises(self):
        """An exception from the run is recorded as ERROR, the partial trace is
        still flushed, and the exception propagates to the caller."""
        async def boom_stream():
            yield SystemMessage(data={"session_id": "s"})
            yield AssistantMessage(content=[TextBlock("working")])
            raise RuntimeError("kaboom")

        async def run():
            async for _ in trace_stream(boom_stream(), agent_name="cas"):
                pass

        with pytest.raises(RuntimeError, match="kaboom"):
            asyncio.run(run())

        trace = _flush_and_get_trace()
        assert trace.status.value == "error"
        assert "kaboom" in (trace.error_message or "")
        assert len(trace.llm_calls) == 1  # the turn captured before the error

    def test_early_break_still_flushes(self):
        """Closing the stream early (consumer break) still finalizes a partial
        trace via the teardown fallback."""
        async def run():
            gen = trace_stream(_stream(_full_run_messages()), agent_name="cas")
            await gen.__anext__()  # init
            await gen.__anext__()  # first assistant turn (tool use)
            await gen.aclose()     # GeneratorExit → finally → finalize

        asyncio.run(run())
        trace = _flush_and_get_trace()
        assert len(trace.llm_calls) == 1

    def test_disabled_sdk_skips_send(self):
        """With the SDK disabled, the stream still passes through but no trace
        is sent."""
        import decimalai._config as cfg
        from decimalai._config import _sender
        cfg._config.enabled = False

        out = _drive(_full_run_messages(), agent_name="cas")
        assert len(out) == 5  # every message still yielded to the caller
        _sender.flush()
        cfg._client.ingest_trace.assert_not_called()


# ── Public install / convenience wrappers ─────────────────────────────

class TestInstall:
    def _inject_fake_sdk(self, monkeypatch):
        fake = types.ModuleType("claude_agent_sdk")

        async def fake_query(*, prompt, options=None):
            for m in _full_run_messages():
                yield m

        fake.query = fake_query
        monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
        return fake, fake_query

    def test_install_patches_query(self, monkeypatch):
        """install() replaces claude_agent_sdk.query with a tracing wrapper that
        captures the agent_name and prompt."""
        import decimalai.claude_agent_sdk as cas
        fake, original = self._inject_fake_sdk(monkeypatch)

        cas.install(agent_name="installed-agent")
        assert fake.query is not original  # patched

        async def run():
            async for _ in fake.query(prompt="How tall is the Eiffel Tower?"):
                pass

        asyncio.run(run())
        trace = _flush_and_get_trace()
        assert trace.agent_name == "installed-agent"
        assert trace.user_input_preview == "How tall is the Eiffel Tower?"
        assert len(trace.llm_calls) == 2

    def test_install_is_idempotent(self, monkeypatch):
        """A second install() doesn't double-wrap query."""
        import decimalai.claude_agent_sdk as cas
        fake, _ = self._inject_fake_sdk(monkeypatch)

        cas.install(agent_name="a")
        once = fake.query
        cas.install(agent_name="b")
        assert fake.query is once

    def test_traced_query_wraps(self, monkeypatch):
        """traced_query() calls query() under the hood and traces the stream."""
        self._inject_fake_sdk(monkeypatch)

        async def run():
            async for _ in traced_query(prompt="hi", agent_name="tq-agent"):
                pass

        asyncio.run(run())
        trace = _flush_and_get_trace()
        assert trace.agent_name == "tq-agent"
        assert len(trace.llm_calls) == 2


class TestUsageExtraction:
    """_extract_usage — Anthropic's input_tokens is the uncached remainder;
    effective input must add cache read/creation tokens (parity with the
    OpenAI handler, whose prompt_tokens includes cached tokens)."""

    def test_cache_tokens_added_to_input(self):
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": 4_000,
            "cache_read_input_tokens": 180_000,
            "cache_creation_input_tokens": 16_000,
            "output_tokens": 900,
        })
        assert inp == 200_000
        assert out == 900

    def test_plain_usage_unchanged(self):
        from decimalai.claude_agent_sdk import _extract_usage

        assert _extract_usage({"input_tokens": 120, "output_tokens": 18}) == (120, 18)
        assert _extract_usage(None) == (None, None)

    def test_object_shaped_usage_with_cache(self):
        from types import SimpleNamespace

        from decimalai.claude_agent_sdk import _extract_usage

        usage = SimpleNamespace(
            input_tokens=10, cache_read_input_tokens=90,
            cache_creation_input_tokens=0, output_tokens=5,
        )
        assert _extract_usage(usage) == (100, 5)

    def test_text_generation_without_cache(self):
        """Pure text generation (no cache): uncached input + output unchanged."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": 500,
            "output_tokens": 250,
        })
        assert inp == 500  # no cache tokens to add
        assert out == 250

    def test_text_generation_with_only_cache_read(self):
        """Cache read without creation: input = uncached + read only."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": 100,
            "cache_read_input_tokens": 900,  # warm cache hit
            "output_tokens": 50,
        })
        assert inp == 1_000  # 100 + 900 + 0
        assert out == 50

    def test_text_generation_with_only_cache_creation(self):
        """Cache creation without prior read: input = uncached + created."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": 200,
            "cache_creation_input_tokens": 1_800,  # cache fill
            "output_tokens": 75,
        })
        assert inp == 2_000  # 200 + 0 + 1_800
        assert out == 75

    def test_text_generation_large_tokens(self):
        """Large token counts (context-window scale) are summed correctly."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": 50_000,
            "cache_read_input_tokens": 900_000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 2_000,
        })
        assert inp == 950_000
        assert out == 2_000

    def test_missing_cache_fields_treated_as_zero(self):
        """Omitted cache fields default to 0 (not an error)."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": 300,
            # cache_read_input_tokens omitted
            # cache_creation_input_tokens omitted
            "output_tokens": 100,
        })
        assert inp == 300
        assert out == 100

    def test_zero_cache_fields_unchanged(self):
        """Explicit zeros in cache fields are honored."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": 400,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 150,
        })
        assert inp == 400
        assert out == 150

    def test_object_shaped_usage_cache_read_only(self):
        """SimpleNamespace with only cache_read (cache_creation omitted)."""
        from types import SimpleNamespace

        from decimalai.claude_agent_sdk import _extract_usage

        usage = SimpleNamespace(
            input_tokens=50,
            cache_read_input_tokens=450,
            output_tokens=30,
        )
        # cache_creation_input_tokens is not defined — defaults to None, treated as 0
        assert _extract_usage(usage) == (500, 30)

    def test_none_input_tokens_returns_none(self):
        """If input_tokens itself is None, input stays None (no summing with cache)."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": None,
            "cache_read_input_tokens": 100,
            "output_tokens": 50,
        })
        assert inp is None  # uncached remainder is None → stop
        assert out == 50

    def test_dict_with_all_fields_present(self):
        """Complete dict with all token fields (typical Anthropic response)."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": 1_000,
            "cache_read_input_tokens": 10_000,
            "cache_creation_input_tokens": 2_000,
            "output_tokens": 500,
        })
        assert inp == 13_000
        assert out == 500

    def test_non_integer_token_values_ignored(self):
        """Non-integer token values (strings, floats) are skipped."""
        from decimalai.claude_agent_sdk import _extract_usage

        inp, out = _extract_usage({
            "input_tokens": "1000",  # string → ignored, becomes None
            "cache_read_input_tokens": 100,
            "output_tokens": 50,
        })
        # input_tokens was None after type check, so input is None (no summing)
        assert inp is None
        assert out == 50
