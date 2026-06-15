"""Retrieval agent for selecting relevant knowledge repository results."""

from __future__ import annotations

import logging

from repositories.faq_repository import FAQRepository
from repositories.knowledge_repository import KnowledgeRepository

CONFIDENCE_THRESHOLD = 0.7

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Retrieve relevant support knowledge and apply confidence gating."""

    def __init__(self, repository: KnowledgeRepository | None = None) -> None:
        self.repository = repository or FAQRepository()

    def retrieve(self, query: str, top_k: int = 1) -> dict:
        """Return repository matches with confidence and no-match filtering."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty or whitespace only")

        if not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        logger.info("Retrieving knowledge results with top_k=%d", top_k)
        results = self.repository.search(query, top_k=top_k)
        if not results:
            logger.info("Repository returned no results")
            return {
                "match_found": False,
                "confidence": 0.0,
                "results": [],
            }

        best_distance = float(results[0]["distance"])
        confidence = _confidence_from_distance(best_distance)
        match_found = confidence >= CONFIDENCE_THRESHOLD

        if not match_found:
            logger.info(
                "Rejecting result with confidence %.4f below threshold %.4f",
                confidence,
                CONFIDENCE_THRESHOLD,
            )
            return {
                "match_found": False,
                "confidence": confidence,
                "results": [],
            }

        logger.info(
            "Accepted %d result(s) with confidence %.4f",
            len(results),
            confidence,
        )
        return {
            "match_found": True,
            "confidence": confidence,
            "results": [_without_distance(result) for result in results],
        }


def _confidence_from_distance(distance: float) -> float:
    """Convert cosine distance to confidence with clamp(1 - distance / 2)."""
    confidence = 1.0 - (distance / 2.0)
    return min(1.0, max(0.0, confidence))


def _without_distance(result: dict) -> dict:
    """Return a public retrieval result without raw repository distance."""
    return {
        key: value
        for key, value in result.items()
        if key != "distance"
    }
