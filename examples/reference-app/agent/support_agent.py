"""SupportBot agent — v1 and v2 implementations.

v1: search_docs + check_order
v2: search_docs + check_order + process_refund (NEW)
"""

from __future__ import annotations

import sys
import os

# Ensure the reference-app root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import decimalai
from agent.tools.search_docs import search_docs
from agent.tools.check_order import check_order


# ── Agent v1 ────────────────────────────────────────────────


@decimalai.trace(agent_name="support-bot")
def run_v1(query: str) -> str:
    """SupportBot v1: search_docs + check_order.

    Routes to the appropriate tool based on the query content.
    """
    q = query.lower()

    if any(kw in q for kw in ["order", "status", "tracking", "shipped", "delivery"]):
        # Extract order ID or use default
        order_id = "ORD-10001"
        for word in query.split():
            if word.upper().rstrip("?.,!").startswith("ORD-"):
                order_id = word.upper().rstrip("?.,!")
                break

        result = check_order(order_id)
        decimalai.log_tool_call(name="check_order", input={"order_id": order_id}, output=result)

        if result.get("found"):
            return (
                f"Your order {order_id} is currently {result['status']}. "
                f"{'Expected delivery: ' + result['eta'] + '.' if result.get('eta') else ''} "
                f"Items: {', '.join(result.get('items', []))}."
            )
        return result.get("error", f"Order {order_id} not found.")

    else:
        result = search_docs(query)
        decimalai.log_tool_call(name="search_docs", input={"query": query}, output=result)

        if result["results"]:
            top = result["results"][0]
            return f"{top['title']}: {top['snippet']}"
        return f"I couldn't find specific information about '{query}'. Please contact support."


# ── Agent v2 ────────────────────────────────────────────────


def run_v2(query: str) -> str:
    """SupportBot v2: search_docs + check_order + process_refund.

    Same as v1 but adds refund processing capability.
    Uses start_trace for explicit control.
    """
    from agent.tools.process_refund import process_refund

    q = query.lower()

    with decimalai.start_trace(agent_name="support-bot") as trace:
        trace.set_input(query)

        if any(kw in q for kw in ["refund", "return", "money back", "cancel order"]):
            # Extract order ID
            order_id = "ORD-10001"
            for word in query.split():
                if word.upper().startswith("ORD-"):
                    order_id = word.upper()
                    break

            reason = "customer requested" if "return" in q else "refund requested"
            result = process_refund(order_id, reason)
            trace.log_tool_call(name="process_refund", input={"order_id": order_id, "reason": reason}, output=result)

            answer = result["message"]

        elif any(kw in q for kw in ["order", "status", "tracking", "shipped"]):
            order_id = "ORD-10001"
            for word in query.split():
                if word.upper().startswith("ORD-"):
                    order_id = word.upper()
                    break

            result = check_order(order_id)
            trace.log_tool_call(name="check_order", input={"order_id": order_id}, output=result)

            if result.get("found"):
                answer = (
                    f"Your order {order_id} is currently {result['status']}. "
                    f"{'Expected delivery: ' + result['eta'] + '.' if result.get('eta') else ''}"
                )
            else:
                answer = result.get("error", f"Order {order_id} not found.")

        else:
            result = search_docs(query)
            trace.log_tool_call(name="search_docs", input={"query": query}, output=result)

            if result["results"]:
                top = result["results"][0]
                answer = f"{top['title']}: {top['snippet']}"
            else:
                answer = f"I couldn't find information about '{query}'."

        trace.set_output(answer)
        return answer
