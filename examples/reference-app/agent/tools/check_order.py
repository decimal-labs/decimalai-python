"""Look up order status by order ID."""

# Mock order database
_ORDERS = {
    "ORD-10001": {"status": "delivered", "eta": "April 25", "items": ["Wireless Mouse"], "total": 29.99},
    "ORD-10002": {"status": "shipped", "eta": "April 30", "items": ["USB-C Hub", "HDMI Cable"], "total": 54.98},
    "ORD-10003": {"status": "processing", "eta": "May 3", "items": ["Mechanical Keyboard"], "total": 89.99},
    "ORD-10004": {"status": "cancelled", "eta": None, "items": ["Laptop Stand"], "total": 45.00},
    "ORD-10005": {"status": "shipped", "eta": "April 29", "items": ["Monitor Arm"], "total": 79.99},
}


def check_order(order_id: str) -> dict:
    """Look up the status of an order by its ID.

    Args:
        order_id: The order ID (e.g., 'ORD-10001').

    Returns:
        Dict with order status, ETA, items, and total.
    """
    if order_id in _ORDERS:
        order = _ORDERS[order_id]
        return {
            "order_id": order_id,
            "found": True,
            **order,
        }

    return {
        "order_id": order_id,
        "found": False,
        "error": f"Order {order_id} not found. Please check the order ID and try again.",
    }
