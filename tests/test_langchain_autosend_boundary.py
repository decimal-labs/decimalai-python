"""Auto-send fires at the outermost chain end, not at the root run_id.

langchain-core 1.5.x reuses the root run_id for child steps: in an LCEL
`prompt | llm` sequence, ChatPromptTemplate's chain events AND the chat
model events all arrive with run_id == parent_run_id == the root's run_id.
The old auto-send check (`span_id == self._root_run_id`) therefore fired at
the PROMPT step's on_chain_end — sending a trace with 0 llm_calls before
the model was even called, and orphaning the real LLM call. Only
parent_run_id=None marks the true outermost callback, on every
langchain-core line, so that is now the boundary.

The simulated-event tests replay both observed event shapes directly
against the handler (no langchain-core needed); the end-to-end test runs a
real `prompt | llm` chain when langchain-core is installed.
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
    # Auto-send registers a manifest (the tests see a model); an empty dict
    # makes `result.get("manifest_id", snapshot.id)` fall back to the
    # snapshot's real string id instead of leaking a MagicMock into RunTrace.
    cfg._client.register_manifest.return_value = {}
    cfg._sender._pending = []
    monkeypatch.setattr(lc_mod, "_manifest_id", None)
    monkeypatch.setattr(lc_mod, "_manifest_tracker", ManifestTracker())
    yield
    cfg._config = None
    cfg._client = None


def _sent_traces():
    import decimalai._config as cfg

    cfg._sender.flush()
    return [c[1][0] for c in cfg._client.method_calls if c[0] == "ingest_trace"]


class TestAutoSendBoundarySimulated:
    """Replay the raw callback event shapes the two langchain-core lines emit."""

    def test_15x_reused_run_id_sends_one_complete_trace(self):
        """langchain-core 1.5.x shape: every child event reuses the root run_id."""
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="joke-bot", auto_send=True)
        root = uuid4()

        handler.on_chain_start(
            {"name": "RunnableSequence"}, {"topic": "cats"},
            run_id=root, parent_run_id=None,
        )
        handler.on_chain_start(
            {"name": "ChatPromptTemplate"}, {"topic": "cats"},
            run_id=root, parent_run_id=root,
        )
        handler.on_chain_end({"output": "prompt"}, run_id=root, parent_run_id=root)

        # The old check fired here (run_id == root) — before the LLM ran.
        assert _sent_traces() == []

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

        traces = _sent_traces()
        assert len(traces) == 1
        assert len(traces[0].llm_calls) == 1
        assert traces[0].llm_calls[0].model_name == "gpt-4o"

    def test_subagent_handler_ships_without_ever_seeing_a_null_parent(self):
        """A sub-agent invoked INSIDE an orchestrator's tool still ships.

        THE BUG (2026-09-03). `parent_run_id is None` was not just the
        boundary, it was the ONLY boundary. The documented multi-agent
        pattern gives the nested `.invoke()` its own CallbackHandler:

            @tool
            def consult_specialist(q):
                h = CallbackHandler(agent_name=SPEC,
                                    parent_trace_id=orch_h.get_trace_id())
                return specialist.invoke(..., config={"callbacks": [h]})

        That inner invoke runs inside the outer run tree, so LangChain hands
        the specialist's handler exactly ONE chain end and it carries
        `run_id == parent_run_id == root` — never a null parent. The
        specialist's trace was therefore built, held, and never sent. Journey
        C caught it as `spec_trace=None` on every run; the in-repo live test
        hit the same wall but its "Timed out waiting for 1 trace(s)" message
        matched the `"timed out"` provider marker and was reported as a QUOTA
        SKIP, so the live matrix stayed green through it.

        Note the shape below is byte-identical to the 1.5.x reused-run_id case
        above except for what is MISSING: no final `parent_run_id=None` end.
        That is the whole difference, and it is why the fix could not simply
        re-match on the root run_id — both shapes look the same by id, so the
        close signal has to be the open-chain balance instead.
        """
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="refund-specialist", auto_send=True)
        root = uuid4()

        handler.on_chain_start(
            {"name": "LangGraph"}, {"messages": []},
            run_id=root, parent_run_id=root,      # nested: never None
        )
        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
            [[MagicMock(type="human")]],
            run_id=root, parent_run_id=root,
            invocation_params={"model_name": "gpt-4o"},
        )
        handler.on_llm_end(
            MagicMock(generations=[], llm_output={}), run_id=root, parent_run_id=root,
        )
        handler.on_chain_end({"output": "refund is 100%"}, run_id=root, parent_run_id=root)

        traces = _sent_traces()
        assert len(traces) == 1, (
            "the sub-agent's trace was never sent — its only chain end carries "
            "a non-null parent_run_id, so a `parent_run_id is None` boundary "
            "never fires for it"
        )
        assert traces[0].agent_name == "refund-specialist"
        assert len(traces[0].llm_calls) == 1

    def test_subagent_does_not_ship_before_its_own_work_finishes(self):
        """The balance must not close on an INNER step of the sub-agent's run.

        The counterpart to the test above: having removed the null-parent
        requirement, the close must still wait for the sub-agent's own nested
        steps. A prompt step ending inside the specialist's run leaves one
        chain open, so nothing ships until the outer one ends.
        """
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="refund-specialist", auto_send=True)
        root, inner = uuid4(), uuid4()

        handler.on_chain_start(
            {"name": "LangGraph"}, {"messages": []}, run_id=root, parent_run_id=root,
        )
        handler.on_chain_start(
            {"name": "ChatPromptTemplate"}, {}, run_id=inner, parent_run_id=root,
        )
        handler.on_chain_end({"output": "prompt"}, run_id=inner, parent_run_id=root)

        assert _sent_traces() == [], (
            "shipped at the sub-agent's INNER step — the model call had not "
            "happened yet, which is the 1.5.x early-send bug in a new place"
        )

        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
            [[MagicMock(type="human")]],
            run_id=root, parent_run_id=root,
            invocation_params={"model_name": "gpt-4o"},
        )
        handler.on_llm_end(
            MagicMock(generations=[], llm_output={}), run_id=root, parent_run_id=root,
        )
        handler.on_chain_end({"output": "done"}, run_id=root, parent_run_id=root)

        traces = _sent_traces()
        assert len(traces) == 1
        assert len(traces[0].llm_calls) == 1

    def test_distinct_run_ids_sends_one_complete_trace(self):
        """Pre-1.5 shape: skipped RunnableSequence root, distinct child run_ids.

        The old code set _root_run_id to the first non-skipped span — the
        prompt — so its end auto-sent early here too.
        """
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="joke-bot", auto_send=True)
        root, prompt_id, llm_id = uuid4(), uuid4(), uuid4()

        handler.on_chain_start(
            {"name": "RunnableSequence"}, {"topic": "cats"},
            run_id=root, parent_run_id=None,
        )
        handler.on_chain_start(
            {"name": "ChatPromptTemplate"}, {"topic": "cats"},
            run_id=prompt_id, parent_run_id=root,
        )
        handler.on_chain_end({"output": "prompt"}, run_id=prompt_id, parent_run_id=root)

        assert _sent_traces() == []

        handler.on_chat_model_start(
            {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
            [[MagicMock(type="human")]],
            run_id=llm_id, parent_run_id=root,
            invocation_params={"model_name": "gpt-4o"},
        )
        handler.on_llm_end(
            MagicMock(generations=[], llm_output={}), run_id=llm_id, parent_run_id=root,
        )
        handler.on_chain_end({"output": "the joke"}, run_id=root, parent_run_id=None)

        traces = _sent_traces()
        assert len(traces) == 1
        assert len(traces[0].llm_calls) == 1

    def test_outermost_error_sends_one_trace(self):
        from decimalai.langchain import CallbackHandler
        from decimalai.schema.common import Status

        handler = CallbackHandler(agent_name="err-bot", auto_send=True)
        root = uuid4()

        handler.on_chain_start(
            {"name": "AgentExecutor"}, {"input": "x"},
            run_id=root, parent_run_id=None,
        )
        handler.on_chain_error(RuntimeError("boom"), run_id=root, parent_run_id=None)

        traces = _sent_traces()
        assert len(traces) == 1
        assert traces[0].status == Status.ERROR

    def test_all_skipped_run_sends_nothing(self):
        """A run made only of skipped wrappers must not send an empty trace."""
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="noop", auto_send=True)
        root = uuid4()

        handler.on_chain_start(
            {"name": "RunnablePassthrough"}, {}, run_id=root, parent_run_id=None,
        )
        handler.on_chain_end({}, run_id=root, parent_run_id=None)

        assert _sent_traces() == []


# ── End-to-end against real langchain-core ─────────────────────

try:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.prompts import ChatPromptTemplate

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


@pytest.mark.skipif(not HAS_LANGCHAIN, reason="langchain-core not installed")
class TestLcelChainEndToEnd:
    """The docs' per-call joke-bot: `prompt | llm` with a CallbackHandler."""

    def _make_chain(self, responses):
        llm = FakeListChatModel(responses=responses)
        prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
        return prompt | llm

    def test_per_call_handler_sends_one_trace_with_llm_call(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="joke-bot")
        chain = self._make_chain(["A cat joke."])
        chain.invoke({"topic": "cats"}, config={"callbacks": [handler]})

        traces = _sent_traces()
        assert len(traces) == 1
        assert len(traces[0].llm_calls) == 1
        assert traces[0].agent_name == "joke-bot"

    def test_handler_reused_across_invocations(self):
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(agent_name="joke-bot")
        chain = self._make_chain(["A cat joke.", "A dog joke."])
        chain.invoke({"topic": "cats"}, config={"callbacks": [handler]})
        chain.invoke({"topic": "dogs"}, config={"callbacks": [handler]})

        traces = _sent_traces()
        assert len(traces) == 2
        assert all(len(t.llm_calls) == 1 for t in traces)
        assert traces[0].id != traces[1].id
