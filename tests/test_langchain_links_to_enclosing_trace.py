"""A LangChain run inside a generic trace must be its CHILD, not a stranger.

THE BUG (measured on prod 2026-09-07). An app can wear both tracers at once —
`@decimalai.trace` / `start_trace()` around the call, and the LangChain adapter
installed process-wide by `instrument()`. The README documents exactly that
composition. But no adapter consulted the generic context, so one logical run
shipped as TWO unrelated ROOT traces.

The generic envelope is the casualty: it only collects what `log_llm_call` puts
in it, so when the model calls go through LangChain it ends up with zero
llm_calls and zero spans. On DecimalAI's own fleet that was 3,604 traces in 3
days — 31.9% of the langchain path, against 0.00% on all six other frameworks.

It is not cosmetic. `compat_service._get_used_components` decides "which
components did this run use" by walking a trace's llm_calls and spans, so an
empty envelope reads as having used NOTHING and is graded `keep` at score 1.0
against every manifest: 100% keep, versus 80.1% for real traces.

Linking fixes it for free — that same function already traverses CHILDREN when
a trace has no parent, so a correct `parent_trace_id` makes the child's
llm_calls reachable from the envelope.
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def reset_sdk_state(monkeypatch):
    import decimalai._config as cfg
    import decimalai.langchain as lc_mod
    from decimalai._config import DecimalConfig
    from decimalai.schema.manifest import ManifestTracker

    cfg._config = DecimalConfig(
        api_key="dai_sk_test", base_url="http://localhost:8000", enabled=True,
    )
    cfg._client = MagicMock()
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


def _run_lcel_events(handler, root):
    """Replay the callback events an LCEL `prompt | llm` chain emits."""
    handler.on_chain_start(
        {"name": "RunnableSequence"}, {"topic": "cats"},
        run_id=root, parent_run_id=None,
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
    handler.on_chain_end({"output": "the joke"}, run_id=root, parent_run_id=None)


def _langchain_trace():
    """The one trace the adapter shipped (the envelope, if any, is separate)."""
    sent = [t for t in _sent_traces() if t.llm_calls]
    assert len(sent) == 1, f"expected exactly one adapter trace, got {len(sent)}"
    return sent[0]


class TestLinksToEnclosingTrace:

    def test_a_run_inside_a_generic_trace_is_linked_to_it(self):
        """THE FIX. Two tracers, one logical run — parent and child, not strangers."""
        import decimalai
        from decimalai.langchain import CallbackHandler

        with decimalai.start_trace(agent_name="envelope-agent") as ctx:
            enclosing_id = ctx.get_trace_id()
            handler = CallbackHandler(auto_send=True, agent_name="inner")
            _run_lcel_events(handler, uuid4())

        assert _langchain_trace().parent_trace_id == enclosing_id

    def test_without_an_enclosing_trace_it_stays_a_root(self):
        """The adapter used alone must be unchanged: a root, parent None."""
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(auto_send=True, agent_name="inner")
        _run_lcel_events(handler, uuid4())

        assert _langchain_trace().parent_trace_id is None

    def test_an_explicit_subagent_parent_still_wins(self):
        """The documented multi-agent pattern must not be overridden.

        A sub-agent handler is constructed with the ORCHESTRATOR's trace id.
        That is a deliberate parent and outranks whatever context happens to
        be live.
        """
        import decimalai
        from decimalai.langchain import CallbackHandler

        explicit = str(uuid4())
        with decimalai.start_trace(agent_name="envelope-agent") as ctx:
            assert ctx.get_trace_id() != explicit
            handler = CallbackHandler(
                auto_send=True, agent_name="sub", parent_trace_id=explicit,
            )
            _run_lcel_events(handler, uuid4())

        assert _langchain_trace().parent_trace_id == explicit

    def test_the_link_is_captured_at_run_START_not_at_close(self):
        """Why capture on the open callback: the close may land elsewhere.

        `_current_trace` is a ContextVar. If the link were read when the run
        closes, any run whose close is driven from outside the caller's
        context would silently lose its parent. Opening inside the context and
        closing outside it is the cheapest way to pin that ordering.
        """
        import decimalai
        from decimalai.langchain import CallbackHandler

        handler = CallbackHandler(auto_send=True, agent_name="inner")
        root = uuid4()

        with decimalai.start_trace(agent_name="envelope-agent") as ctx:
            enclosing_id = ctx.get_trace_id()
            handler.on_chain_start(
                {"name": "RunnableSequence"}, {"topic": "cats"},
                run_id=root, parent_run_id=None,
            )
            handler.on_chat_model_start(
                {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
                [[MagicMock(type="human")]],
                run_id=root, parent_run_id=root,
                invocation_params={"model_name": "gpt-4o"},
            )
            handler.on_llm_end(
                MagicMock(generations=[], llm_output={}),
                run_id=root, parent_run_id=root,
            )

        # Context is gone; the run closes outside it.
        handler.on_chain_end({"output": "done"}, run_id=root, parent_run_id=None)

        assert _langchain_trace().parent_trace_id == enclosing_id

    def test_a_probe_failure_never_breaks_tracing(self, monkeypatch):
        """Fail-open: a raising context probe must not lose the trace."""
        import decimalai.langchain as lc_mod
        from decimalai.langchain import CallbackHandler

        def boom():
            raise RuntimeError("context machinery is broken")

        monkeypatch.setattr(lc_mod, "_enclosing_generic_trace_id", boom)

        handler = CallbackHandler(auto_send=True, agent_name="inner")
        with pytest.raises(RuntimeError):
            _run_lcel_events(handler, uuid4())
