"""Process a refund for an order. Added in Agent v2."""


def process_refund(order_id: str, reason: str) -> dict:
    """Process a refund request for an order.

    Args:
        order_id: The order ID to refund.
        reason: Reason for the refund request.

    Returns:
        Dict with refund status and details.
    """
    # Mock refund processing
    return {
        "order_id": order_id,
        "refund_id": f"REF-{order_id.split('-')[1]}",
        "status": "approved",
        "reason": reason,
        "amount": 29.99,
        "estimated_days": 3,
        "message": f"Refund of $29.99 approved for order {order_id}. "
                   f"Expect the credit within 3 business days.",
    }
