#!/usr/bin/env python3
"""Scenario 1: Generate traces with Agent v1.

Runs 10 customer queries through SupportBot v1 (search_docs + check_order).
All traces are captured and sent to DecimalAI automatically.
"""

import os
import sys

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

import decimalai
decimalai.init()

from agent.support_agent import run_v1


QUERIES = [
    "How do I reset my password?",
    "What is the status of order ORD-10001?",
    "What is your return policy?",
    "Where is my order ORD-10002?",
    "How much does shipping cost?",
    "Check order ORD-10003 status",
    "How do I update my account email?",
    "Has order ORD-10005 shipped yet?",
    "What payment methods do you accept?",
    "I need help with order ORD-10004",
]


def main():
    print("=" * 55)
    print("  Scenario 1: Generate Traces (Agent v1)")
    print("  Tools: search_docs, check_order")
    print("=" * 55)
    print()

    for i, query in enumerate(QUERIES, 1):
        answer = run_v1(query)
        print(f"  [{i:2d}/10] Q: {query}")
        print(f"          A: {answer[:80]}{'...' if len(answer) > 80 else ''}")
        print()

    print(f"✅ {len(QUERIES)} traces sent to DecimalAI!")
    print("📊 View them at: https://app.decimal.ai/traces")
    print()


if __name__ == "__main__":
    main()
