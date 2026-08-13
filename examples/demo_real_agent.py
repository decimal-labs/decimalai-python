#!/usr/bin/env python3
"""Real LLM Demo — Finance Research Agent with LangChain + Gemini.

Runs a real LangChain tool-calling agent (langchain 1.x `create_agent`,
a LangGraph graph under the hood) with actual Gemini API calls and
sends the resulting traces to the DecimalAI backend.

Prerequisites:
    pip install "decimalai[langchain]" langchain langchain-google-genai

Environment Variables:
    GEMINI_API_KEY        — Gemini API key
    DECIMAL_API_KEY       — DecimalAI API key (default: dai_sk_test_key_001)
    DECIMAL_BASE_URL      — Backend URL (default: http://localhost:8000)
    DECIMAL_ENV_FILE      — optional path to a .env outside this repo to load
                            keys from, so they don't have to be copied in

Usage:
    # 1. Point the SDK at a DecimalAI backend — the hosted API, or your own
    #    deployment. Unset, it falls back to http://localhost:8000.
    export DECIMAL_BASE_URL=https://api.decimal.ai
    export DECIMAL_API_KEY=dai_sk_...

    # 2. Run
    GEMINI_API_KEY=your_key python examples/demo_real_agent.py
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone

# Load .env if available (so GEMINI_API_KEY doesn't need manual export).
# DECIMAL_ENV_FILE lets you point at a .env that lives outside this repo — e.g.
# the one your backend already uses — without copying keys around.
try:
    from dotenv import load_dotenv

    load_dotenv()  # loads .env from cwd
    _extra_env = os.environ.get("DECIMAL_ENV_FILE")
    if _extra_env and os.path.exists(_extra_env):
        load_dotenv(_extra_env, override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on shell env

sys.path.insert(0, ".")

# ── Configuration ──────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
API_KEY = os.environ.get("DECIMAL_API_KEY", "dai_sk_test_key_001")
BASE_URL = os.environ.get("DECIMAL_BASE_URL", "http://localhost:8000")
AGENT_NAME = "finance-research-agent-live"

# ── NEW SDK SETUP (2 lines!) ──
import decimalai
decimalai.init()

from decimalai.langchain import instrument
instrument(agent_name=AGENT_NAME)

# ── Tool Definitions ──────────────────────────────────────────

def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol (e.g., AAPL, TSLA, GOOG)

    Returns:
        JSON string with price, change, and change_pct
    """
    # Simulated stock data (in production, call a real API)
    prices = {
        "AAPL": {"price": 178.52, "change": 1.23, "change_pct": 0.69},
        "TSLA": {"price": 245.30, "change": -3.10, "change_pct": -1.25},
        "GOOG": {"price": 175.20, "change": 0.80, "change_pct": 0.46},
        "MSFT": {"price": 415.60, "change": 2.40, "change_pct": 0.58},
        "NVDA": {"price": 875.50, "change": 15.20, "change_pct": 1.77},
        "META": {"price": 505.75, "change": -1.90, "change_pct": -0.37},
        "AMZN": {"price": 185.60, "change": 0.95, "change_pct": 0.51},
    }
    data = prices.get(ticker.upper(), {"price": 100.00, "change": 0.00, "change_pct": 0.00})
    return json.dumps({
        "ticker": ticker.upper(),
        "price": data["price"],
        "change": f"{data['change']:+.2f}",
        "change_pct": f"{data['change_pct']:+.2f}%",
        "currency": "USD",
        "as_of": datetime.now(timezone.utc).isoformat(),
    })


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like '10000 / 178.52' or '50000 * 1.15 ** 10'

    Returns:
        The result of the calculation
    """
    # Safe eval with limited namespace
    allowed = {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "pow": pow, "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"expression": expression, "result": round(float(result), 2)})
    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


def search_news(query: str) -> str:
    """Search for recent financial news articles.

    Args:
        query: Search query about a company or financial topic

    Returns:
        JSON list of recent news articles with title, summary, and sentiment
    """
    # Simulated news (in production, call a real news API)
    news_db = {
        "AAPL": [
            {"title": "Apple Vision Pro sales exceed expectations in Q1",
             "summary": "Apple's new headset sold 600K units in its first quarter.",
             "sentiment": "positive", "source": "Reuters"},
            {"title": "Apple announces new AI features for iPhone 17",
             "summary": "Siri to get major upgrade with on-device LLM.",
             "sentiment": "positive", "source": "Bloomberg"},
        ],
        "TSLA": [
            {"title": "Tesla Q4 deliveries beat analyst expectations",
             "summary": "Tesla delivered 484,507 vehicles, beating the 473K estimate.",
             "sentiment": "positive", "source": "CNBC"},
            {"title": "Tesla faces increasing competition in China",
             "summary": "BYD and Nio gaining market share in key EV market.",
             "sentiment": "negative", "source": "WSJ"},
        ],
    }

    # Try to find matching news
    for key, articles in news_db.items():
        if key.lower() in query.lower():
            return json.dumps(articles)

    return json.dumps([{
        "title": f"No specific news found for '{query}'",
        "summary": "Try searching for a specific ticker symbol.",
        "sentiment": "neutral", "source": "DecimalAI",
    }])


# ── Agent Setup ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful finance research assistant. Answer questions about "
    "stocks, markets, and financial calculations. Use the available tools "
    "when needed."
)


def _make_react_agent(model, tools, system_prompt):
    """Build a tool-calling agent across langchain / langgraph versions.

    langgraph >=1.0 moved `create_react_agent` to `langchain.agents.create_agent`
    (renaming `prompt=` to `system_prompt=`) and deprecated the old symbol.
    Prefer the new one; fall back for langgraph-only installs. Same idiom as
    tests/integration/_live_helpers.py:make_react_agent.
    """
    try:
        from langchain.agents import create_agent
    except ImportError:
        from langgraph.prebuilt import create_react_agent
        return create_react_agent(model, tools, prompt=system_prompt)
    return create_agent(model, tools, system_prompt=system_prompt)


def build_agent():
    """Create a LangChain tool-calling agent with Gemini and finance tools."""
    try:
        from langchain_core.tools import tool
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print('   pip install "decimalai[langchain]" langchain langchain-google-genai')
        sys.exit(1)

    if not GEMINI_API_KEY:
        print("❌ GEMINI_API_KEY not set.")
        print("   Export it: export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    # Wrap the plain functions as tools — schema + description come from
    # the type hints and docstrings.
    tools = [tool(get_stock_price), tool(calculate), tool(search_news)]

    # Create LLM. gemini-2.0-flash was retired for new users (2026-Q1);
    # the platform's canonical judge model is `gemini-2.5-flash`. This demo
    # pins it to that name so the example keeps working on freshly-created
    # Google projects.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,  # langchain SDK param name (kw arg, not env var)
        temperature=0.1,
    )

    try:
        return _make_react_agent(llm, tools, SYSTEM_PROMPT)
    except ImportError:
        print("❌ Missing dependencies. Install with:")
        print('   pip install "decimalai[langchain]" langchain langchain-google-genai')
        sys.exit(1)


# ── Test Scenarios ────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "Simple Q&A (no tools)",
        "input": "What does P/E ratio mean in simple terms?",
    },
    {
        "name": "Single tool call",
        "input": "What is Apple's current stock price?",
    },
    {
        "name": "Multi-tool chain",
        "input": "What is Tesla's stock price and what are the latest news about it?",
    },
    {
        "name": "Calculation with tools",
        "input": "If I have $10,000 to invest in NVDA, how many shares can I buy?",
    },
    {
        "name": "Complex analysis",
        "input": "Compare AAPL and MSFT stock prices. Which one had a better day today?",
    },
]


def main():
    print("=" * 60)
    print("  DecimalAI Real Agent Demo")
    print("  Finance Research Agent — LangChain + Gemini")
    print("=" * 60)
    print()

    # Create agent
    print("🤖 Creating LangChain tool-calling agent with Gemini...")
    agent = build_agent()
    print("   ✅ Agent ready with 3 tools: get_stock_price, calculate, search_news")
    print("   ✅ DecimalAI tracing registered globally via instrument()")

    # Run scenarios
    print(f"\n🚀 Running {len(SCENARIOS)} scenarios...\n")

    for i, scenario in enumerate(SCENARIOS, start=1):
        print(f"  [{i}/{len(SCENARIOS)}] {scenario['name']}")
        print(f"  Question: {scenario['input']}")
        print("-" * 40)

        try:
            # No callbacks needed — instrument() handles it!
            result = agent.invoke(
                {"messages": [{"role": "user", "content": scenario["input"]}]}
            )
            answer = str(result["messages"][-1].content)
            print(f"  Answer: {answer[:100]}...")
            print(f"  📤 Trace auto-sent!")
        except Exception as e:
            print(f"  ❌ Agent error: {e}")

        print()

    print("=" * 60)
    print("✅ Demo complete! Open http://localhost:3000 to see traces in the UI.")


if __name__ == "__main__":
    main()

