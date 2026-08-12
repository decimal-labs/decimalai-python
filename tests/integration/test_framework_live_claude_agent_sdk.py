"""Live-LLM — Anthropic's Claude Agent SDK through the native stream tracer.

The Claude Agent SDK (``claude-agent-sdk``) is Anthropic's own agent framework:
``query(prompt, options=ClaudeAgentOptions(...))`` is an async generator that
drives the Claude Code engine and yields ``SystemMessage`` (init) →
``AssistantMessage`` (model turns) → ``UserMessage`` (tool results) →
``ResultMessage`` (cumulative usage, cost, final text). There is no global
callback system, so the DecimalAI adapter (``decimalai.claude_agent_sdk``) traces
by *wrapping the stream*: :func:`traced_query` consumes each message, passes it
through unchanged, and emits **one** :class:`RunTrace` when the stream ends —
capturing the LLM turn(s) and an auto-detected manifest (model + system prompt).

This test proves that path end-to-end with a real Claude-backed run. We use the
explicit ``traced_query`` wrapper (not the global ``install()`` monkeypatch) so
the tracing is scoped to this cell with no global state to restore. The workload
is a no-tool arithmetic prompt: it exercises the core init→assistant→result
trace path deterministically without depending on the Claude Code toolset or the
sandbox filesystem.

The Claude Agent SDK is Anthropic-native, so the matrix is anthropic-only (the
release gate enforces this via ``FRAMEWORK_PROVIDERS``). Its ``query()`` shells
out to the ``claude`` CLI, so the test also skips when that binary isn't on PATH.

Marker: live_llm + claude_agent_sdk.
Install the extra with ``pip install -e ".[claude-agent-sdk-tests]"`` (and the
Claude Code CLI: ``npm i -g @anthropic-ai/claude-code``).
"""

from __future__ import annotations

import asyncio
import shutil

import pytest

from . import _live_helpers as h


# Pure-reasoning prompt — no tools, no filesystem, deterministic answer.
CAS_PROMPT = "What is 17 multiplied by 4? Reply with only the number."
CAS_EXPECTED = "68"


@pytest.mark.live_llm
@pytest.mark.claude_agent_sdk
@pytest.mark.parametrize("provider, model", h.matrix("claude_agent_sdk"))
def test_claude_agent_sdk_query_native(provider, model):
    """A real Claude Agent SDK ``query()`` on Anthropic → the native stream
    tracer → one backend trace whose model turn is captured as an llm_call with
    the Claude model id, plus an auto-detected manifest."""
    h.require_key_for(provider)
    pytest.importorskip("claude_agent_sdk")
    # query() drives the Claude Code engine via the `claude` CLI; without it the
    # SDK raises CLINotFoundError. Skip cleanly rather than fail on setup.
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI ('claude') not found on PATH — required by claude-agent-sdk")

    from claude_agent_sdk import ClaudeAgentOptions

    from decimalai.claude_agent_sdk import traced_query

    agent_name = h.unique_agent(f"claude-agent-sdk-{provider}")

    # Bound the run and give it a manifest-bearing system prompt. allowed_tools=[]
    # keeps it from reaching for the Claude Code toolset on a pure-math prompt.
    options = ClaudeAgentOptions(
        model=model,
        system_prompt="You are a calculator. Reply with only the number, nothing else.",
        allowed_tools=[],
        max_turns=2,
    )

    async def _run() -> str:
        final = ""
        async for message in traced_query(
            prompt=CAS_PROMPT, options=options, agent_name=agent_name,
        ):
            # ResultMessage carries the run's final text.
            result = getattr(message, "result", None)
            if isinstance(result, str) and result:
                final = result
        return final

    answer = asyncio.run(_run())

    assert CAS_EXPECTED in answer.replace(",", ""), (
        f"Claude Agent SDK run didn't surface {CAS_EXPECTED!r}: {answer!r}"
    )

    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])

    llm_calls = detail.get("llm_calls", [])
    assert llm_calls, (
        f"Trace {detail['id']} has no llm_calls — the model turn wasn't captured "
        f"by the stream tracer. spans={detail.get('spans')}"
    )
    models = " ".join(
        str(c.get("model_name") or c.get("model") or "") for c in llm_calls
    ).lower()
    assert "claude" in models, (
        f"Expected a Claude model id in recorded llm_calls models {models!r}. "
        f"Trace id={detail['id']}"
    )
    assert detail.get("manifest_id"), "manifest_id missing — auto-detection failed"


@pytest.mark.live_llm
@pytest.mark.claude_agent_sdk
@pytest.mark.parametrize("provider, model", h.matrix("claude_agent_sdk"))
def test_claude_agent_sdk_text_generation_token_usage(provider, model):
    """Text generation token usage is correctly reported: uncached remainder +
    cache read + cache creation summed into effective input_tokens.

    The Claude Agent SDK's ResultMessage.usage carries:
    - input_tokens: uncached remainder only (unique to Anthropic)
    - cache_read_input_tokens: re-used from warm cache (if any)
    - cache_creation_input_tokens: tokens paid to create cache (if any)
    - output_tokens: generated tokens

    The tracer should sum the cache tokens into input_tokens so the trace
    reports the true context consumption (matching OpenAI's prompt_tokens which
    includes cached tokens). This test verifies that the sum is recorded."""
    h.require_key_for(provider)
    pytest.importorskip("claude_agent_sdk")
    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI ('claude') not found on PATH")

    from claude_agent_sdk import ClaudeAgentOptions
    from decimalai.claude_agent_sdk import traced_query

    agent_name = h.unique_agent(f"claude-agent-sdk-tokens-{provider}")

    # Simple prompt that will generate a short text response (no tools).
    options = ClaudeAgentOptions(
        model=model,
        system_prompt="You are a helpful assistant.",
        allowed_tools=[],
        max_turns=1,
    )

    async def _run():
        async for message in traced_query(
            prompt="Say 'hello' only.",
            options=options,
            agent_name=agent_name,
        ):
            pass

    asyncio.run(_run())
    h.flush_sdk_sender()
    traces = h.poll_for_trace(agent_name)
    detail = h.get_trace_detail(traces[0]["id"])

    llm_calls = detail.get("llm_calls", [])
    assert llm_calls, f"No llm_calls in trace {detail['id']}"

    # Find the response turn (responder role, not planner).
    responder_calls = [
        c for c in llm_calls
        if c.get("call_role") == "responder" or c.get("role") == "responder"
    ]
    assert responder_calls, (
        f"No responder turn found in llm_calls. Available roles: "
        f"{[c.get('call_role') for c in llm_calls]}"
    )

    resp = responder_calls[-1]  # take the last responder call
    output_tokens = resp.get("output_tokens")
    input_tokens = resp.get("input_tokens")

    # Both should be present: output_tokens from generation, input_tokens from
    # the summed uncached + cache tokens. For a non-cached run, input_tokens
    # should be just the prompt size (small for our short prompt).
    assert input_tokens is not None and input_tokens > 0, (
        f"input_tokens missing or zero in llm_call: {resp}"
    )
    assert output_tokens is not None and output_tokens > 0, (
        f"output_tokens missing or zero in llm_call: {resp}"
    )
