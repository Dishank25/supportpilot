"""ChromaDB-backed repository for SupportPilot FAQ knowledge."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import chromadb

from repositories.knowledge_repository import KnowledgeRepository
from services.embeddings import embed_text, embed_texts

COLLECTION_NAME = "faq_collection"
FAQ_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "faq.json"
VECTORSTORE_PATH = Path(__file__).resolve().parents[1] / "vectorstore"

logger = logging.getLogger(__name__)


class FAQRepository(KnowledgeRepository):
    """Search FAQ entries using persisted ChromaDB question embeddings."""

    def __init__(
        self,
        faq_path: Path = FAQ_PATH,
        vectorstore_path: Path = VECTORSTORE_PATH,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.faq_path = faq_path
        self.vectorstore_path = vectorstore_path
        self.collection_name = collection_name
        self.faqs = self._load_faqs()
        self.faq_hash = self._hash_faqs()

        logger.info("Initializing FAQRepository with ChromaDB at %s", vectorstore_path)
        self.client = chromadb.PersistentClient(path=str(vectorstore_path))

        existing = self._get_existing_collection()
        if self._needs_ingest(existing):
            if existing is not None:
                logger.info(
                    "FAQ collection is stale or incomplete; deleting and re-ingesting"
                )
                self.client.delete_collection(self.collection_name)
            else:
                logger.info("FAQ collection not found; ingesting %d FAQs", len(self.faqs))
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine", "faq_hash": self.faq_hash},
            )
            self._ingest_faqs()
        else:
            self.collection = existing
            logger.info(
                "Reusing existing FAQ collection with %d documents",
                self.collection.count(),
            )

    def search(self, query: str, top_k: int = 1) -> list[dict]:
        """Search FAQ records and return neutral knowledge records.

        Maps FAQ fields onto the shared schema: ``title`` is the question and
        ``content`` is the answer. See ``KnowledgeRepository.search`` for the
        full contract.
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty or whitespace only")

        if not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        logger.info("Searching FAQ collection with top_k=%d", top_k)
        query_embedding = embed_text(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = results.get("ids", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        records: list[dict] = []
        for record_id, metadata, distance in zip(ids, metadatas, distances):
            records.append(
                {
                    "id": record_id,
                    "title": metadata["question"],
                    "content": metadata["answer"],
                    "category": metadata["category"],
                    "distance": float(distance),
                }
            )

        return records

    def _load_faqs(self) -> list[dict]:
        if not self.faq_path.exists():
            raise FileNotFoundError(f"FAQ file not found: {self.faq_path}")

        with self.faq_path.open("r", encoding="utf-8") as faq_file:
            faqs = json.load(faq_file)

        if not isinstance(faqs, list):
            raise ValueError("FAQ file must contain a JSON array")

        required_keys = {"id", "category", "question", "answer"}
        for index, faq in enumerate(faqs):
            if not isinstance(faq, dict):
                raise ValueError(f"FAQ entry at index {index} must be an object")

            missing_keys = required_keys - set(faq)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"FAQ entry at index {index} is missing: {missing}")

        return faqs

    def _hash_faqs(self) -> str:
        """Return a SHA-256 hash of the raw faq.json bytes."""
        return hashlib.sha256(self.faq_path.read_bytes()).hexdigest()

    def _get_existing_collection(self):
        """Return the persisted collection if it exists, else None."""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            # ChromaDB raises when the collection does not exist yet.
            return None

    def _needs_ingest(self, existing) -> bool:
        """Re-ingest when the store is missing, incomplete, or out of date."""
        if existing is None:
            return True
        if existing.count() != len(self.faqs):
            logger.info(
                "FAQ count mismatch: store has %d, JSON has %d",
                existing.count(),
                len(self.faqs),
            )
            return True
        stored_hash = (existing.metadata or {}).get("faq_hash")
        if stored_hash != self.faq_hash:
            logger.info("FAQ hash changed since last ingest")
            return True
        return False

    def _ingest_faqs(self) -> None:
        ids = [str(faq["id"]) for faq in self.faqs]
        questions = [str(faq["question"]) for faq in self.faqs]
        embeddings = embed_texts(questions)
        metadatas: list[dict[str, Any]] = [
            {
                "id": str(faq["id"]),
                "category": str(faq["category"]),
                "question": str(faq["question"]),
                "answer": str(faq["answer"]),
            }
            for faq in self.faqs
        ]

        self.collection.add(
            ids=ids,
            documents=questions,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Ingested %d FAQs into ChromaDB", len(ids))
