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
    """_extract_usage keeps the four Anthropic counts SPLIT.

    It used to return two numbers, with the cache counts summed into input:

        inp += cache_read_input_tokens + cache_creation_input_tokens

    That fold made the one thing DecimalAI most needs to see unobservable.
    DecimalAI injects a query-routed skill menu at position ZERO of the system
    prompt, rebuilt per query; varying bytes at position zero defeat the
    provider's prefix cache for everything behind them. A call that goes from
    "180k cached + 4k fresh" to "184k fresh" sums to 184k either way — the
    regression is invisible in exactly the number that was being reported.

    So `input_tokens` is now Anthropic's UNCACHED REMAINDER, verbatim, and the
    cache counts ride as their own fields. This is a real behaviour change on
    the Claude path (see the function's docstring for what moves downstream).
    """

    def test_cache_tokens_are_not_folded_into_input(self):
        """The headline change: input stays 4,000, not 200,000."""
        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage({
            "input_tokens": 4_000,
            "cache_read_input_tokens": 180_000,
            "cache_creation_input_tokens": 16_000,
            "output_tokens": 900,
        })
        assert usage.input_tokens == 4_000          # was 200_000 before
        assert usage.output_tokens == 900
        assert usage.cache_read_tokens == 180_000
        assert usage.cache_creation_tokens == 16_000

    def test_absent_cache_fields_stay_none_not_zero(self):
        """A provider that never reported cache tokens must not read as a MISS.

        None ("never measured") and 0 ("measured, cache was cold") are
        different facts, and the platform column that receives them is nullable
        precisely so the difference survives. Defaulting absent → 0 would
        manufacture a cache miss on every un-instrumented call.
        """
        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage({"input_tokens": 120, "output_tokens": 18})
        assert usage.input_tokens == 120
        assert usage.output_tokens == 18
        assert usage.cache_read_tokens is None
        assert usage.cache_creation_tokens is None

    def test_reported_zero_is_kept_as_zero(self):
        """The other half of the same distinction: 0 is a measurement."""
        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage({
            "input_tokens": 400,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 150,
        })
        assert usage.cache_read_tokens == 0
        assert usage.cache_creation_tokens == 0
        assert usage.cache_read_tokens is not None

    def test_no_usage_at_all(self):
        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage(None)
        assert usage == (None, None, None, None)

    def test_object_shaped_usage(self):
        """The CLI hands back objects as well as dicts."""
        from types import SimpleNamespace

        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage(SimpleNamespace(
            input_tokens=10, cache_read_input_tokens=90,
            cache_creation_input_tokens=0, output_tokens=5,
        ))
        assert usage == (10, 5, 90, 0)

    def test_object_shaped_usage_with_field_missing(self):
        """An attribute the object simply does not define reads as None."""
        from types import SimpleNamespace

        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage(SimpleNamespace(
            input_tokens=50, cache_read_input_tokens=450, output_tokens=30,
        ))
        assert usage.cache_read_tokens == 450
        assert usage.cache_creation_tokens is None   # not defined ≠ zero

    def test_warm_cache_run_is_now_distinguishable_from_a_cold_one(self):
        """Two runs the old fold reported IDENTICALLY.

        Both charge the same effective context. Only the split says whether
        the prefix was cacheable — which is the whole point of the change.
        """
        from decimalai.claude_agent_sdk import _extract_usage

        warm = _extract_usage({
            "input_tokens": 4_000,
            "cache_read_input_tokens": 180_000,
            "cache_creation_input_tokens": 0,
            "output_tokens": 900,
        })
        cold = _extract_usage({
            "input_tokens": 184_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "output_tokens": 900,
        })
        # The old two-number contract collapsed these to the same pair.
        assert (warm.input_tokens + (warm.cache_read_tokens or 0)) == (
            cold.input_tokens + (cold.cache_read_tokens or 0)
        )
        # The new one does not.
        assert warm != cold
        assert warm.input_tokens != cold.input_tokens
        assert warm.cache_read_tokens != cold.cache_read_tokens

    def test_none_input_tokens_stays_none(self):
        """A missing uncached remainder is not inferred from the cache counts."""
        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage({
            "input_tokens": None,
            "cache_read_input_tokens": 100,
            "output_tokens": 50,
        })
        assert usage.input_tokens is None
        assert usage.output_tokens == 50
        assert usage.cache_read_tokens == 100     # still captured

    def test_non_integer_token_values_ignored(self):
        """Strings/floats are not coerced — a wrong number beats no number never."""
        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage({
            "input_tokens": "1000",
            "cache_read_input_tokens": 3.5,
            "output_tokens": 50,
        })
        assert usage.input_tokens is None
        assert usage.cache_read_tokens is None
        assert usage.output_tokens == 50

    def test_bool_is_not_a_token_count(self):
        """`bool` subclasses `int`; True must not be stored as 1 token."""
        from decimalai.claude_agent_sdk import _extract_usage

        usage = _extract_usage({
            "input_tokens": 100,
            "cache_read_input_tokens": True,
            "output_tokens": 50,
        })
        assert usage.cache_read_tokens is None


class TestCacheTokensReachTheRecord:
    """End-to-end within the adapter: split counts land on LlmCallRecord.

    The extractor being right is necessary but not sufficient — the record is
    what gets serialized onto the trace payload, so it is what the platform's
    new `llm_call.cache_read_tokens` / `cache_creation_tokens` columns
    actually receive.
    """

    def test_per_turn_usage_lands_on_the_call_record(self):
        from types import SimpleNamespace

        import decimalai.claude_agent_sdk as cas
        from decimalai.schema.trace import LlmCallRecord

        state = cas._RunState(
            agent_name="cache-agent", project=None, parent_trace_id=None,
            user_input="hi", options=None,
        )
        # Blocks are matched by class NAME, so reuse the module's fake TextBlock.
        message = SimpleNamespace(
            model="claude-sonnet-4-5",
            content=[TextBlock("done")],
            usage={
                "input_tokens": 4_000,
                "cache_read_input_tokens": 180_000,
                "cache_creation_input_tokens": 16_000,
                "output_tokens": 900,
            },
        )
        cas._ingest_assistant(state, message)

        assert len(state.llm_calls) == 1
        call = state.llm_calls[0]
        assert isinstance(call, LlmCallRecord)
        assert call.input_tokens == 4_000
        assert call.cache_read_tokens == 180_000
        assert call.cache_creation_tokens == 16_000

    def test_run_total_fallback_carries_the_split(self):
        """When only ResultMessage reports usage, the cache totals ride along."""
        from types import SimpleNamespace

        import decimalai.claude_agent_sdk as cas
        from decimalai.schema.trace import LlmCallRecord

        state = cas._RunState(
            agent_name="cache-agent", project=None, parent_trace_id=None,
            user_input="hi", options=None,
        )
        state.llm_calls.append(LlmCallRecord(model_name="claude-sonnet-4-5"))
        cas._ingest_result(state, SimpleNamespace(
            session_id="s1", result="done", is_error=False, total_cost_usd=None,
            usage={
                "input_tokens": 4_000,
                "cache_read_input_tokens": 180_000,
                "cache_creation_input_tokens": 0,
                "output_tokens": 900,
            },
        ))
        call = state.llm_calls[-1]
        assert call.input_tokens == 4_000
        assert call.cache_read_tokens == 180_000
        assert call.cache_creation_tokens == 0

    def test_record_serializes_the_split_onto_the_payload(self):
        """`_client` sends `model_dump(mode="json")` — the keys must be there."""
        from decimalai.schema.trace import LlmCallRecord

        payload = LlmCallRecord(
            model_name="claude-sonnet-4-5",
            input_tokens=4_000, output_tokens=900,
            cache_read_tokens=180_000, cache_creation_tokens=0,
        ).model_dump(mode="json")

        assert payload["input_tokens"] == 4_000
        assert payload["cache_read_tokens"] == 180_000
        assert payload["cache_creation_tokens"] == 0

    def test_unreported_split_serializes_as_null_not_zero(self):
        from decimalai.schema.trace import LlmCallRecord

        payload = LlmCallRecord(model_name="gpt-5.4-mini").model_dump(mode="json")
        assert payload["cache_read_tokens"] is None
        assert payload["cache_creation_tokens"] is None
