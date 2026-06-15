"""Repository contract for searchable SupportPilot knowledge sources."""

from __future__ import annotations

from abc import ABC, abstractmethod


class KnowledgeRepository(ABC):
    """Abstract base class for searchable knowledge repositories."""

    @abstractmethod
    def search(self, query: str, top_k: int = 1) -> list[dict]:
        """Return matching knowledge records ordered by similarity.

        Each record is a dict with exactly these keys:

            - ``id``: stable identifier for the source record (str)
            - ``title``: short headline for the record. For FAQs this is the
              question; for other sources it is the chunk/section title.
            - ``content``: the body text. For FAQs this is the answer.
            - ``category``: classification label for the record (str)
            - ``distance``: cosine distance between the query and the record,
              on a bounded scale where lower means a closer (better) match.

        Results are sorted by ``distance`` ascending (lower = more similar).
        The returned list may contain fewer than ``top_k`` records when the
        underlying store holds fewer documents than requested; it is empty
        only when the store itself is empty.

        Implementations must preserve this exact key set and the cosine
        distance semantics so callers can swap one repository for another
        without changes.
        """
