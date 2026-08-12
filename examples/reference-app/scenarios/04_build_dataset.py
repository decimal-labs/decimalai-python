#!/usr/bin/env python3
"""Scenario 4: Build a clean SFT dataset.

Demonstrates the end of the DecimalAI loop:
- View the impact report (which traces are compatible)
- Build a training dataset from passing, version-compatible traces
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass


def main():
    print("=" * 55)
    print("  Scenario 4: Build Clean Dataset")
    print("=" * 55)
    print()

    # ── Show the impact report ──
    print("  📋 Impact Report (support-bot: v1 → v2)")
    print()
    print("   Manifest v1: tools=[search_docs, check_order]")
    print("   Manifest v2: tools=[search_docs, check_order, process_refund]")
    print("   Change: process_refund ADDED")
    print()
    print("   ┌────────────┬───────┬────────────────────────────────────┐")
    print("   │ Category   │ Count │ Reason                             │")
    print("   ├────────────┼───────┼────────────────────────────────────┤")
    print("   │ keep       │  10   │ No changed tools used              │")
    print("   │ repair     │   0   │                                    │")
    print("   │ replay     │   0   │                                    │")
    print("   │ drop       │   0   │                                    │")
    print("   └────────────┴───────┴────────────────────────────────────┘")
    print()

    # ── Show dataset format ──
    print("  📦 Dataset Build")
    print()
    print("   Filters applied:")
    print("   ✓ Manifest: v2 (current)")
    print("   ✓ Eval verdict: pass only")
    print("   ✓ Format: SFT (chat completion)")
    print()

    example = {
        "messages": [
            {
                "role": "system",
                "content": "You are a friendly customer support assistant.",
            },
            {
                "role": "user",
                "content": "What is the status of order ORD-10001?",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "check_order",
                        "arguments": json.dumps({"order_id": "ORD-10001"}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_001",
                "content": json.dumps({
                    "order_id": "ORD-10001",
                    "status": "delivered",
                    "eta": "April 25",
                    "items": ["Wireless Mouse"],
                }),
            },
            {
                "role": "assistant",
                "content": "Your order ORD-10001 is currently delivered. "
                           "Items: Wireless Mouse.",
            },
        ],
    }

    print("   Example training row (SFT format):")
    print("   " + "-" * 50)
    print(json.dumps(example, indent=4).replace("\n", "\n   "))
    print()

    # ── Summary ──
    print("  ─" * 27)
    print()
    print("  ✅ The full DecimalAI workflow:")
    print()
    print("     1. Agent v1 ran → 10 traces captured")
    print("     2. Agent updated to v2 → manifest change detected")
    print("     3. Traces evaluated → quality scored")
    print("     4. Impact report → all 10 traces compatible (keep)")
    print("     5. Dataset built → clean SFT data for fine-tuning")
    print()
    print("  📊 Build datasets at: https://app.decimal.ai/datasets")
    print("  📖 Docs: https://docs.decimal.ai/tutorials/training-pipeline")
    print()


if __name__ == "__main__":
    main()
