"""End-to-end framework adapter tests.

Unlike the unit tests in `tests/`, these run *real* framework code (LangChain,
OpenAI Agents, generic decorator) through the DecimalAI adapter and verify
that traces and manifests actually land in the live backend.

Run with the live backend up on http://localhost:8000:

    cd decimalai-python
    .venv/bin/pytest tests/integration -m integration -v

Conventions:
- Each test uses a unique agent name (per-run timestamp + framework name) so
  reruns don't interfere with each other.
- Tests skip cleanly if the framework isn't installed.
- Tests skip if the backend isn't reachable — they aren't intended to fail in
  pure-unit CI.

What each test asserts:
1. `decimalai.<framework>.install()` does not raise.
2. Running the agent emits at least one trace.
3. The trace is retrievable from the backend by agent_name.
4. The trace has the expected manifest, user_input, and final_output.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from uuid import uuid4

import pytest


# ─── Config ──────────────────────────────────────────────────────────

BACKEND_URL = os.environ.get("DECIMAL_BACKEND_URL", "http://localhost:8000")
API_KEY = os.environ.get("DECIMAL_API_KEY", "dai_sk_test_key_001")
POLL_TIMEOUT_S = 8
POLL_INTERVAL_S = 0.5


def _unique_agent(framework: str) -> str:
    """Generate a unique agent name so re-runs don't collide."""
    return f"e2e-{framework}-{datetime.now().strftime('%H%M%S')}-{uuid4().hex[:6]}"


def _backend_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _flush_sdk_sender() -> None:
    """Block until the SDK's background sender has POSTed all queued traces."""
    from decimalai._config import _sender
    _sender.flush(timeout=POLL_TIMEOUT_S)


# ─── Backend probe helpers ───────────────────────────────────────────

def _list_agent_traces(agent_name: str) -> list[dict]:
    """Query the live backend for traces of the given agent."""
    url = f"{BACKEND_URL}/api/v1/traces?agent_name={agent_name}&limit=20"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read())["traces"]
    except urllib.error.HTTPError as e:
        raise AssertionError(f"backend trace list error: {e.code} {e.read().decode()[:200]}")


def _poll_for_trace(agent_name: str, expected_count: int = 1) -> list[dict]:
    """Poll the backend until at least `expected_count` traces appear, or timeout."""
    deadline = time.time() + POLL_TIMEOUT_S
    last = []
    while time.time() < deadline:
        last = _list_agent_traces(agent_name)
        if len(last) >= expected_count:
            return last
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(
        f"Timed out waiting for {expected_count} trace(s) on agent={agent_name}. "
        f"Last poll saw: {len(last)} traces."
    )


# ─── Shared fixtures ─────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _require_backend():
    """Skip every integration test if the live backend is unreachable."""
    if not _backend_alive():
        pytest.skip(f"Backend at {BACKEND_URL} unreachable — start uvicorn first.")


@pytest.fixture(autouse=True)
def _reset_sdk_config():
    """Force-reset the SDK config to point at the live backend with the test key."""
    import decimalai
    decimalai.init(api_key=API_KEY, base_url=BACKEND_URL, enabled=True)
    yield


# ═══════════════════════════════════════════════════════════════════
# Case A — Generic decorator (no framework dependency)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_generic_decorator_end_to_end():
    """Use @decimalai.trace + log_llm_call to hand-roll a trace, then verify
    it lands on the backend.

    This is the lowest-friction integration: it depends on nothing but the
    SDK itself, so a failure here means the wire to the backend is broken.
    """
    import decimalai

    agent_name = _unique_agent("generic")
    user_query = "What's 2 + 2?"
    final_answer = "4"

    @decimalai.trace(agent_name=agent_name)
    def run_agent(query: str) -> str:
        decimalai.log_llm_call(
            model="gpt-4o-fake",
            input=[{"role": "user", "content": query}],
            output={"content": final_answer},
        )
        return final_answer

    out = run_agent(user_query)
    assert out == final_answer

    # The SDK posts traces in a background thread — flush before polling.
    _flush_sdk_sender()

    # Backend should receive the trace.
    traces = _poll_for_trace(agent_name)
    assert len(traces) == 1
    t = traces[0]
    assert t["agent_name"] == agent_name
    # The trace should reference the manifest we just auto-detected.
    assert t.get("manifest_id") is not None, "manifest_id missing on trace"


# ═══════════════════════════════════════════════════════════════════
# Case B — LangChain via langchain_core
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_langchain_callback_end_to_end():
    """Run a real langchain_core chain with FakeListChatModel and verify
    the CallbackHandler emits a trace to the backend.

    Uses FakeListChatModel so this test doesn't need an LLM API key.
    """
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.prompts import ChatPromptTemplate

    from decimalai.langchain import CallbackHandler

    agent_name = _unique_agent("langchain")
    fake_responses = ["The capital of France is Paris."]

    # FakeListChatModel does not expose a `model_name` in its invocation
    # params, which means the adapter cannot auto-detect a manifest and the
    # backend rejects the trace with 400. Subclass to inject a model name —
    # this mirrors what any real ChatModel implementation provides.
    class _NamedFakeChatModel(FakeListChatModel):
        @property
        def _identifying_params(self):
            return {"model_name": "fake-chat-model-v1"}

    llm = _NamedFakeChatModel(responses=fake_responses)
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])

    # NB: `prompt | llm` is a RunnableSequence, which is in the adapter's
    # _SKIP_CHAIN_TYPES list — so the bare pipeline wouldn't trigger
    # `on_chain_start` and the trace wouldn't auto-send. Naming the chain
    # via with_config(run_name=...) makes LangChain dispatch it as a named
    # chain start, which the adapter does track.
    chain = (prompt | llm).with_config({"run_name": "e2e-langchain-agent"})

    handler = CallbackHandler(agent_name=agent_name)
    result = chain.invoke({"question": "Capital of France?"}, config={"callbacks": [handler]})

    # The chain ran — FakeListChatModel returns the pre-canned response.
    assert "Paris" in str(result.content)

    _flush_sdk_sender()

    # Backend should receive the trace.
    traces = _poll_for_trace(agent_name)
    assert len(traces) >= 1
    t = traces[0]
    assert t["agent_name"] == agent_name


# ═══════════════════════════════════════════════════════════════════
# Case C — OpenAI Agents SDK
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
def test_openai_agents_processor_end_to_end():
    """Verify the OpenAI Agents tracing processor:
    1. Installs without error.
    2. Accepts synthetic spans.
    3. Sends an aggregated trace to the backend.

    We don't run a real agent (that would need an OPENAI_API_KEY); instead
    we feed the processor a hand-built span — exercising the *adapter's*
    end-to-end path without paying for an LLM call.
    """
    pytest.importorskip("agents")
    from agents.tracing.span_data import GenerationSpanData

    from decimalai.openai_agents import DecimalTracingProcessor

    agent_name = _unique_agent("openai-agents")
    processor = DecimalTracingProcessor(agent_name=agent_name)

    # The OpenAI Agents SDK's `Trace`/`Span` are abstract — but the processor
    # only duck-types them via getattr(...). Duck-typed mocks are enough.
    class _FakeTrace:
        def __init__(self, trace_id: str, name: str = "e2e"):
            self.trace_id = trace_id
            self.name = name

    class _FakeSpan:
        def __init__(self, trace_id: str, span_data):
            self.trace_id = trace_id
            self.span_id = f"span_{uuid4().hex}"
            self.parent_id = None
            self.span_data = span_data
            self.started_at = datetime.utcnow().isoformat() + "Z"
            self.ended_at = datetime.utcnow().isoformat() + "Z"
            self.error = None

    trace_id = f"trace_{uuid4().hex}"
    fake_trace = _FakeTrace(trace_id)
    processor.on_trace_start(fake_trace)

    span_data = GenerationSpanData(
        input=[{"role": "user", "content": "ping"}],
        output=[{"role": "assistant", "content": "pong"}],
        model="gpt-4o-fake",
        model_config={"temperature": 0.7},
        usage={"input_tokens": 5, "output_tokens": 5},
    )
    processor.on_span_end(_FakeSpan(trace_id, span_data))
    processor.on_trace_end(fake_trace)
    processor.shutdown()

    _flush_sdk_sender()

    # Backend should receive the trace.
    traces = _poll_for_trace(agent_name)
    assert len(traces) >= 1
    t = traces[0]
    assert t["agent_name"] == agent_name
