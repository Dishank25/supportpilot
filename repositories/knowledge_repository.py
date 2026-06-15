"""Repository contract for searchable SupportPilot knowledge sources."""

from __future__ import annotations

from abc import ABC, abstractmethod


class KnowledgeRepository(ABC):
    """Abstract base class for searchable knowledge repositories."""

    @abstractmethod
    def search(self, query: str, top_k: int = 1) -> list[dict]:
        """Return matching knowledge records with distance scores."""
