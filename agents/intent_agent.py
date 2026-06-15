"""Rule-based intent classifier for SupportPilot queries."""

from __future__ import annotations

import logging

ORDER_KEYWORDS = [
    "order",
    "track",
    "tracking",
    "delivery",
    "delivered",
    "shipment",
    "shipping",
]

REFUND_KEYWORDS = [
    "refund",
    "return",
    "money back",
    "reimburse",
    "credit",
]

PRODUCT_KEYWORDS = [
    "product",
    "food",
    "toy",
    "treat",
    "grain free",
    "sell",
    "stock",
]

ACCOUNT_KEYWORDS = [
    "account",
    "password",
    "login",
    "sign in",
    "email",
    "profile",
]

GENERAL_INTENT = "GENERAL"
INTENT_KEYWORDS = {
    "ORDER": ORDER_KEYWORDS,
    "REFUND": REFUND_KEYWORDS,
    "PRODUCT": PRODUCT_KEYWORDS,
    "ACCOUNT": ACCOUNT_KEYWORDS,
}

logger = logging.getLogger(__name__)


class IntentAgent:
    """Classify support queries with simple keyword rules."""

    def classify(self, query: str) -> str:
        """Return the first supported intent whose keywords match the query."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty or whitespace only")

        normalized_query = query.lower()
        for intent, keywords in INTENT_KEYWORDS.items():
            if any(keyword in normalized_query for keyword in keywords):
                logger.info("Classified query as %s", intent)
                return intent

        logger.info("Classified query as %s", GENERAL_INTENT)
        return GENERAL_INTENT
