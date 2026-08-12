"""Search the knowledge base for relevant articles."""

# Mock knowledge base
_ARTICLES = {
    "password": {
        "title": "How to Reset Your Password",
        "content": "Go to Settings > Security > Reset Password. You'll receive a "
                   "confirmation email within 5 minutes.",
    },
    "return": {
        "title": "Return Policy",
        "content": "We accept returns within 30 days of purchase. Items must be in "
                   "original condition. Start a return at Settings > Orders.",
    },
    "shipping": {
        "title": "Shipping Information",
        "content": "Standard shipping: 5-7 business days ($4.99). Express: 1-2 "
                   "business days ($12.99). Free shipping on orders over $50.",
    },
    "account": {
        "title": "Account Management",
        "content": "Update your profile at Settings > Account. You can change your "
                   "email, name, and notification preferences.",
    },
    "payment": {
        "title": "Payment Methods",
        "content": "We accept Visa, Mastercard, AMEX, and PayPal. Add or update "
                   "payment methods at Settings > Billing.",
    },
}


def search_docs(query: str) -> dict:
    """Search the knowledge base for articles matching the query.

    Args:
        query: Search query string.

    Returns:
        Dict with matching articles and metadata.
    """
    query_lower = query.lower()
    results = []

    for keyword, article in _ARTICLES.items():
        if keyword in query_lower or any(
            word in query_lower for word in article["title"].lower().split()
        ):
            results.append({
                "title": article["title"],
                "snippet": article["content"][:100] + "...",
                "relevance": 0.95,
            })

    if not results:
        # Fuzzy fallback
        results = [{
            "title": "General Help",
            "snippet": f"We found several articles that may help with '{query}'.",
            "relevance": 0.5,
        }]

    return {
        "query": query,
        "results": results,
        "total": len(results),
    }
