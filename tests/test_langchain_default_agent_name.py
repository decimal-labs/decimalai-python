"""A LangChain trace never ships a null `agent_name`.

`decimalai.init(langchain=True)` with no `agent_name` leaves the handler
relying on `on_chain_start`'s auto-detection, and that can't fire for an
LCEL chain: `prompt | llm` starts as a RunnableSequence, which
`_SKIP_CHAIN_TYPES` returns on before the detection block. (A bare
`llm.invoke()` emits no chain callback at all; a LangGraph `create_agent`
is detected as "LangGraph" and was never affected.) The handler's
`agent_name` therefore stayed None all the way into the ingest payload,
which the API rejects — "Trace validation failed: 'agent_name' is
required" — so every LCEL trace was dropped.

The resolution ladder is: explicit handler name → name detected from the
root chain → the global `instrument(agent_name=...)` → "langchain-agent".
"""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def reset_sdk_state(monkeypatch):
    """Reset all SDK global state before each test."""
    import decimalai._config as cfg
    import decimalai.langchain as lc_mod
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    cfg._config = DecimalConfig(
        api_key="dai_sk_test",
        base_url="http://localhost:8000",
        enabled=True,
    )
    cfg._client = MagicMock()
    cfg._client.register_manifest.return_value = {}
    cfg._sender._pending = []
    monkeypatch.setattr(lc_mod, "_manifest_id", None)
    monkeypatch.setattr(lc_mod, "_manifest_tracker", ManifestTracker())
    monkeypatch.setattr(lc_mod, "_install_agent_name", None)
    # `instrument()` in any earlier test module leaves a handler in this
    # ContextVar and its configure hook registered for the life of the
    # process — the real-langchain runs below would then be traced twice.
    token = lc_mod._decimal_callback_var.set(None)
    yield
    lc_mod._decimal_callback_var.reset(token)
    cfg._config = None
    cfg._client = None


def _sent_traces():
    import decimalai._config as cfg

    cfg._sender.flush()
    return [c[1][0] for c in cfg._client.method_calls if c[0] == "ingest_trace"]


def _run_lcel_events(handler, root):
    """Replay the callback events an LCEL `prompt | llm` chain emits."""
    handler.on_chain_start(
        {"name": "RunnableSequence"}, {"topic": "cats"},
        run_id=root, parent_run_id=None,
    )
    handler.on_chain_start(
        {"name": "ChatPromptTemplate"}, {"topic": "cats"},
        run_id=root, parent_run_id=root,
    )
    handler.on_chain_end({"output": "prompt"}, run_id=root, parent_run_id=root)
    handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
        [[MagicMock(type="human")]],
        run_id=root, parent_run_id=root,
        invocation_params={"model_name": "gpt-4o"},
    )
    handler.on_llm_end(
        MagicMock(generations=[], llm_output={}), run_id=root, parent_run_id=root,
    )
    handler.on_chain_end({"output": "the joke"}, run_id=root, parent_run_id=None)


class TestDefaultAgentName:

    def test_build_trace_without_agent_name_is_not_null(self):
        """The regression: a trace built with no explicit agent_name."""
        from decimalai.langchain import DEFAULT_AGENT_NAME, CallbackHandler

        handler = CallbackHandler(auto_send=False)
        trace = handler.build_trace()

        assert trace.agent_name is not None
        assert trace.agent_name == DEFAULT_AGENT_NAME

    def test_lcel_chain_sends_a_named_trace(self):
        """`prompt | llm` — the root is skipped, so nothing detects a name."""
        from decimalai.langchain import DEFAULT_AGENT_NAME, CallbackHandler

        handler = CallbackHandler(auto_send=True)
        _run_lcel_events(handler, uuid4())

        traces = _sent_traces()
        assert len(traces) == 1
        assert traces[0].agent_name == DEFAULT_AGENT_NAME

    def test_llm_call_records_are_named_too(self):
        """Per-call attribution must not be null while the trace is named."""
        from decimalai.langchain import DEFAULT_AGENT_NAME, CallbackHandler

        handler = CallbackHandler(auto_send=True)
        _run_lcel_events(handler, uuid4())

        trace = _sent_traces()[0]
        assert [lc.agent_name for lc in trace.llm_calls] == [DEFAULT_AGENT_NAME]

    def test_manifest_uses_the_same_name_as_the_trace(self):
        """A manifest registered as "unknown" belongs to a different agent
        than the traces that carry its manifest_id."""
        import decimalai._config as cfg
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(auto_send=True)
        _run_lcel_events(handler, uuid4())

        trace = _sent_traces()[0]
        snapshots = [
            c[1][0] for c in cfg._client.method_calls if c[0] == "register_manifest"
        ]
        assert len(snapshots) == 1
        assert snapshots[0].agent_name == trace.agent_name

    def test_instrument_agent_name_wins_over_the_default(self):
        """A handler built without a name inherits the global one."""
        import decimalai.langchain as lc_mod
        from decimalai.langchain import CallbackHandler

        lc_mod._install_agent_name = "global-agent"
        handler = CallbackHandler(auto_send=False)

        assert handler.build_trace().agent_name == "global-agent"

    def test_explicit_agent_name_wins(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="support-agent", auto_send=False)

        assert handler.build_trace().agent_name == "support-agent"

    def test_detected_chain_name_wins(self):
        """The default must not pre-empt `on_chain_start` auto-detection —
        which only fires because the fallback is never written back onto
        `self.agent_name`."""
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(auto_send=True)
        root = uuid4()
        handler.on_chain_start(
            {"name": "SupportAgent"}, {"input": "hello"},
            run_id=root, parent_run_id=None,
        )
        handler.on_chain_end({"output": "hi"}, run_id=root, parent_run_id=None)

        traces = _sent_traces()
        assert len(traces) == 1
        assert traces[0].agent_name == "SupportAgent"

    def test_default_does_not_stick_across_invocations(self):
        """An unnamed first run must not poison a named second run."""
        from decimalai.langchain import DEFAULT_AGENT_NAME, CallbackHandler

        handler = CallbackHandler(auto_send=True)
        _run_lcel_events(handler, uuid4())

        second = uuid4()
        handler.on_chain_start(
            {"name": "SupportAgent"}, {"input": "hello"},
            run_id=second, parent_run_id=None,
        )
        handler.on_chain_end({"output": "hi"}, run_id=second, parent_run_id=None)

        traces = _sent_traces()
        assert [t.agent_name for t in traces] == [DEFAULT_AGENT_NAME, "SupportAgent"]


# ── End-to-end against real langchain-core ─────────────────────

try:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.prompts import ChatPromptTemplate

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
class TestQuickstartPathEndToEnd:
    """The docs' Quickstart: no agent_name anywhere, real LCEL chain."""

    def test_unnamed_handler_ships_a_named_trace(self):
        from decimalai.langchain import DEFAULT_AGENT_NAME, CallbackHandler

        handler = CallbackHandler()
        llm = FakeListChatModel(responses=["A cat joke."])
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        chain = prompt | llm

        chain.invoke({"topic": "cats"}, config={"callbacks": [handler]})

        traces = _sent_traces()
        assert len(traces) == 1
        assert traces[0].agent_name == DEFAULT_AGENT_NAME
