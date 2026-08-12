#!/usr/bin/env python3
"""Mock LLM Demo — Finance Research Agent.

Exercises the full DecimalAI pipeline WITHOUT needing a real LLM API key.
Uses mock LLM responses to simulate a ReAct agent with tools.

Usage:
    # 1. Point the SDK at a DecimalAI backend (the hosted API, or your own
    #    deployment). Defaults to http://localhost:8000 if unset.
    export DECIMAL_BASE_URL=https://api.decimal.ai
    export DECIMAL_API_KEY=dai_sk_...

    # 2. Run this demo
    python examples/demo_mock_agent.py

Trajectory Types Covered:
    1. Simple Q&A (no tools)
    2. Single tool call (ReAct)
    3. Multi-tool chain (sequential)
    4. Parallel tool calls
    5. Error mid-execution (tool failure + recovery)
    6. Multi-turn session (3 exchanges)
    7. Manifest evolution (same agent, different tools)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from uuid import UUID, uuid4

# Add parent dir to path so SDK is importable without installing
sys.path.insert(0, ".")

from decimalai._client import DecimalAIClient
from decimalai.langchain import CallbackHandler
from decimalai.schema.trace import RunTrace

# ── Configuration ──────────────────────────────────────────────
# Defaults target a backend on localhost; override both with the standard SDK
# env vars to run against the hosted API or your own deployment.
API_KEY = os.environ.get("DECIMAL_API_KEY", "dai_sk_test_key_001")
BASE_URL = os.environ.get("DECIMAL_BASE_URL", "http://localhost:8000")
AGENT_NAME = "finance-research-agent"


def mock_msg(type_: str, content: str, tool_calls=None):
    """Create a mock LangChain message object."""
    return type("Msg", (), {
        "type": type_,
        "content": content,
        "tool_calls": tool_calls or [],
    })()


def mock_llm_response(text: str, tool_calls=None, prompt_tokens=20, completion_tokens=15):
    """Create a mock LangChain LLM response."""
    msg = mock_msg("assistant", text, tool_calls=tool_calls)
    gen = type("Gen", (), {"text": text, "message": msg})()
    return type("Response", (), {
        "generations": [[gen]],
        "llm_output": {"token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }},
    })()


# ── Trajectory Generators ─────────────────────────────────────

def trajectory_1_simple_qa(handler: CallbackHandler) -> None:
    """Trajectory 1: Simple Q&A — no tools, just LLM."""
    print("  📝 Trajectory 1: Simple Q&A")
    chain_id = uuid4()
    llm_id = uuid4()

    handler.on_chain_start(
        serialized={"name": "FinanceQAChain"},
        inputs={"input": "What is a P/E ratio?"},
        run_id=chain_id,
    )
    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("user", "What is a P/E ratio?")]],
        run_id=llm_id, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash", "temperature": 0.1},
    )
    handler.on_llm_end(
        response=mock_llm_response(
            "The Price-to-Earnings (P/E) ratio is a valuation metric that compares "
            "a company's stock price to its earnings per share (EPS). A high P/E may "
            "indicate expected growth, while a low P/E may suggest undervaluation.",
            prompt_tokens=15, completion_tokens=55,
        ),
        run_id=llm_id,
    )
    handler.on_chain_end(
        outputs={"output": "The P/E ratio compares stock price to earnings per share."},
        run_id=chain_id,
    )


def trajectory_2_single_tool(handler: CallbackHandler) -> None:
    """Trajectory 2: Single tool call — LLM → get_stock_price → answer."""
    print("  🔧 Trajectory 2: Single tool call (get_stock_price)")
    chain_id = uuid4()
    llm_id_1 = uuid4()
    tool_id = uuid4()
    llm_id_2 = uuid4()

    handler.on_chain_start(
        serialized={"name": "FinanceReActAgent"},
        inputs={"input": "What is the current price of AAPL?"},
        run_id=chain_id,
    )

    # LLM decides to call a tool
    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("user", "What is the current price of AAPL?")]],
        run_id=llm_id_1, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash", "temperature": 0.0},
    )
    handler.on_llm_end(
        response=mock_llm_response(
            "I need to look up the current stock price.",
            tool_calls=[{"name": "get_stock_price", "args": {"ticker": "AAPL"}}],
        ),
        run_id=llm_id_1,
    )

    # Tool execution
    handler.on_tool_start(
        serialized={"name": "get_stock_price"},
        input_str='{"ticker": "AAPL"}',
        run_id=tool_id, parent_run_id=chain_id,
    )
    time.sleep(0.05)  # Simulate latency
    handler.on_tool_end(
        output='{"price": 178.52, "change": "+1.23", "change_pct": "+0.69%"}',
        run_id=tool_id,
    )

    # LLM generates final answer
    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[
            mock_msg("user", "What is the current price of AAPL?"),
            mock_msg("assistant", "I need to look up the current stock price."),
            mock_msg("tool", '{"price": 178.52, "change": "+1.23"}'),
        ]],
        run_id=llm_id_2, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash", "temperature": 0.0},
    )
    handler.on_llm_end(
        response=mock_llm_response(
            "Apple (AAPL) is currently trading at $178.52, up $1.23 (+0.69%) today.",
            prompt_tokens=45, completion_tokens=25,
        ),
        run_id=llm_id_2,
    )

    handler.on_chain_end(
        outputs={"output": "AAPL is at $178.52, up 0.69% today."},
        run_id=chain_id,
    )


def trajectory_3_multi_tool(handler: CallbackHandler) -> None:
    """Trajectory 3: Multi-tool chain — get_stock_price → search_news → answer."""
    print("  🔗 Trajectory 3: Multi-tool chain (price + news)")
    chain_id = uuid4()
    llm1 = uuid4()
    tool1 = uuid4()
    llm2 = uuid4()
    tool2 = uuid4()
    llm3 = uuid4()

    handler.on_chain_start(
        serialized={"name": "FinanceReActAgent"},
        inputs={"input": "Should I buy TSLA? Check the price and recent news."},
        run_id=chain_id,
    )

    # Step 1: LLM → get price
    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("user", "Should I buy TSLA?")]],
        run_id=llm1, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response("Let me check the price first.",
            tool_calls=[{"name": "get_stock_price"}]),
        run_id=llm1,
    )
    handler.on_tool_start(serialized={"name": "get_stock_price"},
        input_str='{"ticker": "TSLA"}', run_id=tool1, parent_run_id=chain_id)
    handler.on_tool_end(output='{"price": 245.30, "change": "-3.10"}', run_id=tool1)

    # Step 2: LLM → search news
    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("user", "Should I buy TSLA?"),
                   mock_msg("tool", '{"price": 245.30}')]],
        run_id=llm2, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response("Now let me check the news.",
            tool_calls=[{"name": "search_news"}]),
        run_id=llm2,
    )
    handler.on_tool_start(serialized={"name": "search_news"},
        input_str='{"query": "TSLA Tesla news"}', run_id=tool2, parent_run_id=chain_id)
    handler.on_tool_end(
        output='[{"title": "Tesla Q4 deliveries beat expectations", "sentiment": "positive"}]',
        run_id=tool2,
    )

    # Step 3: Final answer
    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("user", "Should I buy TSLA?"),
                   mock_msg("tool", "price and news data")]],
        run_id=llm3, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response(
            "TSLA is at $245.30 (-1.25%). Recent news is positive with Q4 deliveries "
            "beating expectations. The stock looks reasonably valued at current levels.",
            prompt_tokens=80, completion_tokens=40,
        ),
        run_id=llm3,
    )
    handler.on_chain_end(
        outputs={"output": "TSLA at $245.30. Positive news. Reasonably valued."},
        run_id=chain_id,
    )


def trajectory_4_parallel_tools(handler: CallbackHandler) -> None:
    """Trajectory 4: Parallel tool calls — LLM requests 2 tools at once."""
    print("  ⚡ Trajectory 4: Parallel tool calls (price + calculate)")
    chain_id = uuid4()
    llm1 = uuid4()
    tool_price = uuid4()
    tool_calc = uuid4()
    llm2 = uuid4()

    handler.on_chain_start(
        serialized={"name": "FinanceReActAgent"},
        inputs={"input": "If I invest $10,000 in NVDA, how many shares do I get?"},
        run_id=chain_id,
    )

    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("user", "If I invest $10K in NVDA, how many shares?")]],
        run_id=llm1, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response("I need the price and to calculate.",
            tool_calls=[
                {"name": "get_stock_price", "args": {"ticker": "NVDA"}},
                {"name": "calculate", "args": {"expression": "10000 / price"}},
            ]),
        run_id=llm1,
    )

    # Both tools execute "in parallel"
    handler.on_tool_start(serialized={"name": "get_stock_price"},
        input_str='{"ticker": "NVDA"}', run_id=tool_price, parent_run_id=chain_id)
    handler.on_tool_start(serialized={"name": "calculate"},
        input_str='{"expression": "10000 / 875.50"}', run_id=tool_calc, parent_run_id=chain_id)
    handler.on_tool_end(output='{"price": 875.50}', run_id=tool_price)
    handler.on_tool_end(output='{"result": 11.42}', run_id=tool_calc)

    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("tool", "price=875.50, shares=11.42")]],
        run_id=llm2, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response(
            "With NVDA at $875.50, a $10,000 investment would buy approximately "
            "11.42 shares (11 full shares + $367.50 remaining).",
        ),
        run_id=llm2,
    )
    handler.on_chain_end(
        outputs={"output": "$10K → ~11 shares of NVDA at $875.50"},
        run_id=chain_id,
    )


def trajectory_5_error_recovery(handler: CallbackHandler) -> None:
    """Trajectory 5: Tool error mid-execution → agent recovers."""
    print("  💥 Trajectory 5: Error + recovery")
    chain_id = uuid4()
    llm1 = uuid4()
    tool_fail = uuid4()
    llm2 = uuid4()
    tool_retry = uuid4()
    llm3 = uuid4()

    handler.on_chain_start(
        serialized={"name": "FinanceReActAgent"},
        inputs={"input": "Get the price history for GOOG"},
        run_id=chain_id,
    )

    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("user", "Get price history for GOOG")]],
        run_id=llm1, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response("I'll look that up.",
            tool_calls=[{"name": "get_stock_price"}]),
        run_id=llm1,
    )

    # Tool fails!
    handler.on_tool_start(serialized={"name": "get_stock_price"},
        input_str='{"ticker": "GOOG", "history": true}',
        run_id=tool_fail, parent_run_id=chain_id)
    handler.on_tool_error(
        error=ConnectionError("API rate limit exceeded. Retry after 1s."),
        run_id=tool_fail,
    )

    # LLM handles the error
    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("tool", "Error: API rate limit exceeded")]],
        run_id=llm2, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response("Let me retry with a simpler request.",
            tool_calls=[{"name": "get_stock_price"}]),
        run_id=llm2,
    )

    # Retry succeeds
    handler.on_tool_start(serialized={"name": "get_stock_price"},
        input_str='{"ticker": "GOOG"}', run_id=tool_retry, parent_run_id=chain_id)
    handler.on_tool_end(output='{"price": 175.20, "change": "+0.80"}', run_id=tool_retry)

    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("tool", '{"price": 175.20}')]],
        run_id=llm3, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response("GOOG is at $175.20, up $0.80 today."),
        run_id=llm3,
    )
    handler.on_chain_end(
        outputs={"output": "GOOG is at $175.20 (recovered from rate limit error)."},
        run_id=chain_id,
    )


def trajectory_6_multi_turn(handler: CallbackHandler, session_id: str) -> list:
    """Trajectory 6: Multi-turn session — 3 user exchanges, same session."""
    print("  💬 Trajectory 6: Multi-turn conversation (3 turns)")
    traces = []

    turns = [
        ("What is MSFT's market cap?", "Microsoft's market cap is approximately $3.1 trillion."),
        ("How does that compare to Apple?", "Apple's market cap is $3.4T, so MSFT is slightly smaller."),
        ("Which is a better buy?", "Both are strong, but AAPL has slightly better margins."),
    ]

    for i, (question, answer) in enumerate(turns):
        h = CallbackHandler(agent_name=AGENT_NAME)
        chain_id = uuid4()
        llm_id = uuid4()

        h.on_chain_start(
            serialized={"name": "FinanceReActAgent"},
            inputs={"input": question},
            run_id=chain_id,
        )
        h.on_chat_model_start(
            serialized={"name": "ChatGemini"},
            messages=[[mock_msg("user", question)]],
            run_id=llm_id, parent_run_id=chain_id,
            invocation_params={"model_name": "gemini-2.0-flash"},
        )
        h.on_llm_end(
            response=mock_llm_response(answer, prompt_tokens=20 + i * 30),
            run_id=llm_id,
        )
        h.on_chain_end(outputs={"output": answer}, run_id=chain_id)

        trace = h.build_trace()
        trace.session_id = session_id
        traces.append(trace)
        print(f"    Turn {i+1}: {question[:40]}...")

    return traces


def trajectory_7_manifest_evolution(handler: CallbackHandler) -> None:
    """Trajectory 7: Different tools → new manifest version.

    Uses 'calculate' tool instead of 'search_news' to trigger
    a different manifest hash.
    """
    print("  🔄 Trajectory 7: Manifest evolution (different tools)")
    chain_id = uuid4()
    llm1 = uuid4()
    tool1 = uuid4()
    llm2 = uuid4()

    handler.on_chain_start(
        serialized={"name": "FinanceReActAgent"},
        inputs={"input": "Calculate 15% annual return on $50,000 for 10 years"},
        run_id=chain_id,
    )
    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("user", "Calculate compound interest")]],
        run_id=llm1, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response("Let me calculate that.",
            tool_calls=[{"name": "calculate"}]),
        run_id=llm1,
    )
    handler.on_tool_start(serialized={"name": "calculate"},
        input_str='{"expression": "50000 * (1.15 ** 10)"}',
        run_id=tool1, parent_run_id=chain_id)
    handler.on_tool_end(output='{"result": 202277.91}', run_id=tool1)

    handler.on_chat_model_start(
        serialized={"name": "ChatGemini"},
        messages=[[mock_msg("tool", '{"result": 202277.91}')]],
        run_id=llm2, parent_run_id=chain_id,
        invocation_params={"model_name": "gemini-2.0-flash"},
    )
    handler.on_llm_end(
        response=mock_llm_response(
            "$50,000 at 15% annual return for 10 years = $202,277.91"
        ),
        run_id=llm2,
    )
    handler.on_chain_end(
        outputs={"output": "$50K → $202K after 10 years at 15%"},
        run_id=chain_id,
    )


# ── Main ─────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DecimalAI Mock Agent Demo")
    print("  Finance Research Agent — 7 Trajectory Types")
    print("=" * 60)
    print()

    client = DecimalAIClient(
        api_key=API_KEY,
        project="demo",
        base_url=BASE_URL,
    )

    # Verify connection
    print("🔌 Connecting to backend...")
    try:
        result = client.verify_auth()
        print(f"   ✅ Authenticated: {result}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        print(f"   Could not reach a DecimalAI backend at {BASE_URL}.")
        print(f"   Set DECIMAL_BASE_URL (and DECIMAL_API_KEY) to point at one.")
        sys.exit(1)

    print()
    print("🚀 Running trajectories...\n")
    ingested_traces = []

    # ── Trajectories 1-5: single traces ──
    for i, trajectory_fn in enumerate([
        trajectory_1_simple_qa,
        trajectory_2_single_tool,
        trajectory_3_multi_tool,
        trajectory_4_parallel_tools,
        trajectory_5_error_recovery,
    ], start=1):
        handler = CallbackHandler(agent_name=AGENT_NAME)
        trajectory_fn(handler)
        trace = handler.get_completed_trace()

        result = client.ingest_trace(trace)
        ingested_traces.append(str(trace.id))
        print(f"     → Trace {trace.id} ingested (manifest: {result.get('manifest_id', 'N/A')})")
        print()

    # ── Trajectory 6: multi-turn ──
    session_id = f"session_{uuid4().hex[:8]}"
    multi_turn_handler = CallbackHandler(agent_name=AGENT_NAME)
    session_traces = trajectory_6_multi_turn(multi_turn_handler, session_id)
    for trace in session_traces:
        result = client.ingest_trace(trace)
        ingested_traces.append(str(trace.id))
    print(f"     → {len(session_traces)} turns ingested (session: {session_id})")
    print()

    # ── Trajectory 7: manifest evolution ──
    handler7 = CallbackHandler(agent_name=AGENT_NAME + "-v2")
    trajectory_7_manifest_evolution(handler7)
    trace7 = handler7.get_completed_trace()
    result7 = client.ingest_trace(trace7)
    ingested_traces.append(str(trace7.id))
    print(f"     → Trace {trace7.id} ingested (new manifest: {result7.get('manifest_id', 'N/A')})")
    print()

    # ── Verification ──
    print("=" * 60)
    print("  📊 Verification")
    print("=" * 60)
    print()

    # List traces to verify
    all_traces = client.list_traces(limit=50, agent_name=AGENT_NAME)
    print(f"  Traces for '{AGENT_NAME}': {all_traces.get('total', len(all_traces.get('traces', [])))}")

    # List manifests
    manifests = client.list_manifests(agent_name=AGENT_NAME)
    manifest_list = manifests.get("manifests", [])
    print(f"  Manifests for '{AGENT_NAME}': {len(manifest_list)}")
    for m in manifest_list:
        print(f"    - {m['id'][:12]}... hash={m.get('manifest_hash', '?')[:16]}...")

    # Verify a specific trace
    if ingested_traces:
        sample = client.get_trace(ingested_traces[1])  # Single tool trace
        print(f"\n  Sample trace (trajectory 2):")
        print(f"    Agent: {sample.get('agent_name')}")
        print(f"    Status: {sample.get('status')}")
        print(f"    Spans: {len(sample.get('spans', []))}")
        print(f"    LLM calls: {len(sample.get('llm_calls', []))}")
        print(f"    Manifest: {sample.get('manifest_id', 'N/A')}")
        print(f"    Input: {sample.get('user_input_preview', '')[:60]}")
        print(f"    Output: {sample.get('final_output_preview', '')[:60]}")

    print()
    print("✅ Demo complete! Open http://localhost:3000 to see traces in the UI.")
    client.close()


if __name__ == "__main__":
    main()
