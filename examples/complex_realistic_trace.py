#!/usr/bin/env python3
"""Complex Realistic Trace — a deep, multi-tool, retry-laden agent run.

Designed to produce a single trace with:
  - 10+ tool calls across distinct domains (stock data, news, analysis)
  - Nested LLM calls (a tool that itself calls Gemini to summarize)
  - One deliberate retry path (a "flaky" tool that fails on first invocation)
  - Multi-step reasoning that forces the agent to chain tool outputs

The result is a single trace with a dense waterfall, nested LLM spans and a
visible retry — a good way to see how a real multi-step agent renders end to end.

Prerequisites
-------------
- GEMINI_API_KEY exported (or in a .env — see DECIMAL_ENV_FILE below)
- DECIMAL_API_KEY (default: dai_sk_test_key_001)
- DECIMAL_BASE_URL pointed at a DecimalAI backend — the hosted API, or your
  own deployment (default: http://localhost:8000)

Optional: DECIMAL_ENV_FILE — path to a .env outside this repo to load keys
from, so you don't have to copy them into the checkout.

Usage
-----
    python examples/complex_realistic_trace.py
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from typing import Any, Dict, List

# ── Load .env if available ────────────────────────────────
# DECIMAL_ENV_FILE points at a .env outside this repo — e.g. the one your
# backend already uses — so keys don't have to be copied into the checkout.
try:
    from dotenv import load_dotenv
    load_dotenv()
    _extra_env = os.environ.get("DECIMAL_ENV_FILE")
    if _extra_env and os.path.exists(_extra_env):
        load_dotenv(_extra_env, override=False)
except ImportError:
    pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DECIMAL_API_KEY = os.environ.get("DECIMAL_API_KEY", "dai_sk_test_key_001")
DECIMAL_BASE_URL = os.environ.get("DECIMAL_BASE_URL", "http://localhost:8000")
AGENT_NAME = "complex-research-agent"

if not GEMINI_API_KEY:
    print("✗ GEMINI_API_KEY not set. Export it, or put it in a .env and point "
          "DECIMAL_ENV_FILE at that file.", file=sys.stderr)
    sys.exit(1)

# ── DecimalAI SDK setup ───────────────────────────────────
import decimalai
decimalai.init(api_key=DECIMAL_API_KEY, base_url=DECIMAL_BASE_URL)

from decimalai.langchain import install
install(agent_name=AGENT_NAME)

# ── LangGraph / LangChain-core imports (langchain itself isn't installed) ──
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent


# ─────────────────────────────────────────────────────────────────
# Tool 1: get_stock_price
# Returns a flat record. Simple, fast — used 3x in a single run.
# ─────────────────────────────────────────────────────────────────
_PRICES = {
    "AAPL": {"price": 184.27, "change": 1.85, "change_pct": 1.01, "market_cap_b": 2820},
    "MSFT": {"price": 421.50, "change": -2.10, "change_pct": -0.50, "market_cap_b": 3130},
    "NVDA": {"price": 920.31, "change": 18.40, "change_pct": 2.04, "market_cap_b": 2260},
    "GOOGL": {"price": 178.45, "change": 0.95, "change_pct": 0.53, "market_cap_b": 2210},
}

@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price and market cap for a ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, MSFT, NVDA)
    """
    ticker = ticker.upper().strip()
    if ticker not in _PRICES:
        return json.dumps({"error": f"Unknown ticker {ticker}"})
    return json.dumps({"ticker": ticker, **_PRICES[ticker]})


# ─────────────────────────────────────────────────────────────────
# Tool 2: get_company_financials
# Returns a large object — agent often follows up with comparisons.
# ─────────────────────────────────────────────────────────────────
_FINANCIALS = {
    "AAPL": {"revenue_yoy_pct": 6.0, "gross_margin": 0.45, "operating_margin": 0.31, "pe_ratio": 31.2, "debt_to_equity": 1.45},
    "MSFT": {"revenue_yoy_pct": 18.0, "gross_margin": 0.69, "operating_margin": 0.45, "pe_ratio": 36.5, "debt_to_equity": 0.27},
    "NVDA": {"revenue_yoy_pct": 122.0, "gross_margin": 0.78, "operating_margin": 0.65, "pe_ratio": 78.3, "debt_to_equity": 0.21},
    "GOOGL": {"revenue_yoy_pct": 15.0, "gross_margin": 0.58, "operating_margin": 0.32, "pe_ratio": 28.1, "debt_to_equity": 0.10},
}

@tool
def get_company_financials(ticker: str) -> str:
    """Pull current financial metrics for a company: revenue growth, margins, P/E ratio, debt-to-equity.

    Args:
        ticker: Stock ticker symbol
    """
    ticker = ticker.upper().strip()
    if ticker not in _FINANCIALS:
        return json.dumps({"error": f"No financials for {ticker}"})
    return json.dumps({"ticker": ticker, "fiscal_year": 2025, **_FINANCIALS[ticker]})


# ─────────────────────────────────────────────────────────────────
# Tool 3: get_industry_peers
# ─────────────────────────────────────────────────────────────────
_PEERS = {
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "ORCL"],
    "NVDA": ["AMD", "INTC", "AVGO", "TSM"],
    "GOOGL": ["MSFT", "META", "AAPL", "AMZN"],
}

@tool
def get_industry_peers(ticker: str) -> str:
    """Return a list of industry-peer ticker symbols for comparison."""
    return json.dumps({"ticker": ticker.upper(), "peers": _PEERS.get(ticker.upper(), [])})


# ─────────────────────────────────────────────────────────────────
# Tool 4: search_recent_news (flaky — retries on first call)
# Deliberately raises once to demonstrate error+retry spans in the trace.
# ─────────────────────────────────────────────────────────────────
_NEWS_CALL_COUNT: Dict[str, int] = {}

_NEWS = {
    "AAPL": [
        "Apple unveils on-device LLM features for iPhone 17, beating Samsung to market by 4 months.",
        "DOJ antitrust case enters discovery phase; analysts split on revenue impact.",
        "Apple Pay processed $1.4T in transactions last fiscal year, up 28%.",
    ],
    "MSFT": [
        "Azure AI revenue grew 70% YoY, driven by Copilot enterprise adoption.",
        "Microsoft acquires three AI data infrastructure startups in Q3.",
        "OpenAI partnership extension confirmed through 2030 with broader IP-sharing terms.",
    ],
    "NVDA": [
        "Blackwell GPU shipments outpace forecast by 22%; supply still constrained through 2026.",
        "China export controls tightened; NVDA estimates $5B revenue impact for fiscal 2026.",
        "Earnings call: data center segment 87% of revenue, gross margins hit all-time high.",
    ],
}

@tool
def search_recent_news(ticker: str, days_back: int = 7) -> str:
    """Search for recent news headlines about a company.

    NOTE: This tool may rate-limit on first call — the agent should retry.

    Args:
        ticker: Stock ticker symbol
        days_back: How many days of news to search (default: 7)
    """
    ticker = ticker.upper().strip()
    # Flaky behavior: fail the first call to AAPL with a retryable error
    # response (NOT a raised exception — we want the agent to read it and retry).
    _NEWS_CALL_COUNT[ticker] = _NEWS_CALL_COUNT.get(ticker, 0) + 1
    if ticker == "AAPL" and _NEWS_CALL_COUNT[ticker] == 1:
        return json.dumps({
            "error": "rate_limited",
            "retry_after_seconds": 1,
            "message": "News API quota briefly hit. Call this tool again with the same args to retry.",
        })

    if ticker not in _NEWS:
        return json.dumps({"ticker": ticker, "headlines": [], "note": "no coverage"})
    time.sleep(0.4)  # simulate API latency
    return json.dumps({"ticker": ticker, "days_back": days_back, "headlines": _NEWS[ticker]})


# ─────────────────────────────────────────────────────────────────
# Tool 5: summarize_with_llm — calls Gemini, producing a NESTED LLM call
# ─────────────────────────────────────────────────────────────────
_summarizer = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)

@tool
def summarize_news_sentiment(ticker: str, headlines_json: str) -> str:
    """Summarize news headlines into a 1-sentence sentiment + score.

    Args:
        ticker: Stock ticker symbol
        headlines_json: JSON string with a "headlines" array of strings
    """
    try:
        data = json.loads(headlines_json)
        headlines = data.get("headlines", [])
    except json.JSONDecodeError:
        headlines = []
    if not headlines:
        return json.dumps({"ticker": ticker, "sentiment": "neutral", "score": 0.0, "summary": "No news available."})

    prompt = (
        f"Headlines for {ticker} (last 7 days):\n"
        + "\n".join(f"  - {h}" for h in headlines)
        + "\n\nReturn STRICT JSON: {\"sentiment\": \"positive|neutral|negative\", \"score\": -1.0..1.0, \"summary\": \"one sentence\"}"
    )
    resp = _summarizer.invoke(prompt)
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    # Best-effort JSON extraction
    try:
        start = content.find("{")
        end = content.rfind("}")
        return content[start : end + 1] if start >= 0 and end > start else json.dumps({"ticker": ticker, "summary": content[:200]})
    except Exception:
        return json.dumps({"ticker": ticker, "summary": str(content)[:200]})


# ─────────────────────────────────────────────────────────────────
# Tool 6: compare_growth — pure compute, no LLM
# ─────────────────────────────────────────────────────────────────
@tool
def compare_growth_rates(tickers_csv: str) -> str:
    """Compare revenue growth rates and return them sorted high-to-low.

    Args:
        tickers_csv: Comma-separated ticker symbols (e.g. "AAPL,MSFT,NVDA")
    """
    tickers = [t.strip().upper() for t in tickers_csv.split(",") if t.strip()]
    rows = []
    for t in tickers:
        if t in _FINANCIALS:
            rows.append({"ticker": t, "revenue_yoy_pct": _FINANCIALS[t]["revenue_yoy_pct"]})
    rows.sort(key=lambda r: -r["revenue_yoy_pct"])
    return json.dumps({"comparison": rows, "winner": rows[0]["ticker"] if rows else None})


# ─────────────────────────────────────────────────────────────────
# Tool 7: build_recommendation — final synthesis (also calls LLM)
# ─────────────────────────────────────────────────────────────────
@tool
def build_recommendation(ticker: str, evidence_json: str) -> str:
    """Synthesize all evidence into a buy/hold/sell recommendation with reasoning.

    Args:
        ticker: Stock ticker symbol
        evidence_json: JSON string with collected financial + news evidence
    """
    prompt = (
        f"You are a CFA-trained equity analyst. Given the evidence below for {ticker},\n"
        f"return STRICT JSON: {{\"action\": \"buy|hold|sell\", \"confidence\": 0.0..1.0, \"reasoning\": \"2-3 sentences\"}}.\n\n"
        f"Evidence:\n{evidence_json[:1500]}"
    )
    resp = _summarizer.invoke(prompt)
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = " ".join(str(c) for c in content)
    try:
        start = content.find("{")
        end = content.rfind("}")
        return content[start : end + 1] if start >= 0 and end > start else json.dumps({"action": "hold", "confidence": 0.5, "reasoning": content[:200]})
    except Exception:
        return json.dumps({"action": "hold", "confidence": 0.5, "reasoning": str(content)[:200]})


# ─────────────────────────────────────────────────────────────────
# Tool 8: get_market_indices — context
# ─────────────────────────────────────────────────────────────────
@tool
def get_market_indices() -> str:
    """Get current values for major market indices (S&P 500, NASDAQ, VIX)."""
    return json.dumps({
        "S&P 500": {"value": 5870.25, "change_pct": 0.42},
        "NASDAQ": {"value": 19120.50, "change_pct": 0.78},
        "VIX": {"value": 13.85, "change_pct": -2.10},
    })


TOOLS = [
    get_stock_price,
    get_company_financials,
    get_industry_peers,
    search_recent_news,
    summarize_news_sentiment,
    compare_growth_rates,
    build_recommendation,
    get_market_indices,
]


# ─────────────────────────────────────────────────────────────────
# Agent setup — LangGraph ReAct prebuilt with Gemini Flash
# ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a senior equity-research analyst. Use the tools to gather "
    "evidence before recommending. If a tool returns a rate-limit error, "
    "retry it once before giving up. Your final answer should be a markdown "
    "report with one section per company + a buy/hold/sell verdict."
)

QUERY = (
    "Compare AAPL, MSFT, and NVDA for a 12-month investment horizon. "
    "For each: fetch current price, financials, recent news (summarize the sentiment), "
    "then call build_recommendation with all collected evidence. "
    "Also fetch the broader market indices for context."
)


def main() -> int:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )
    # LangGraph's prebuilt ReAct agent — instrument-friendly: each LLM call
    # and tool invocation becomes a span the DecimalAI adapter captures.
    app = create_react_agent(
        llm,
        tools=TOOLS,
        prompt=SYSTEM_PROMPT,
    )

    print(f"\n→ Agent: {AGENT_NAME}")
    print(f"→ Query: {QUERY}\n")
    print("=" * 80)

    t0 = time.time()
    final_messages: List[Any] = []
    try:
        for chunk in app.stream(
            {"messages": [HumanMessage(content=QUERY)]},
            stream_mode="values",
        ):
            final_messages = chunk.get("messages", [])
    except Exception as exc:
        print(f"\n✗ Agent run failed: {exc}")
        import traceback; traceback.print_exc()
        return 1
    elapsed = time.time() - t0

    print("=" * 80)
    final_text = final_messages[-1].content if final_messages else "<no output>"
    if isinstance(final_text, list):
        final_text = " ".join(str(c) for c in final_text)
    print(f"\n✓ Agent completed in {elapsed:.1f}s")
    print(f"  Total messages: {len(final_messages)}")
    print(f"  Output length: {len(final_text)} chars\n")
    print("─" * 80)
    print(final_text[:2000])
    print("─" * 80)

    # Give the SDK a moment to flush
    time.sleep(2)

    print(f"\n→ Trace ingested to {DECIMAL_BASE_URL}")
    print(f"→ View at:")
    print(f"     http://localhost:3000/agents/{AGENT_NAME}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
