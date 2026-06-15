"""Embedding generation helpers for SupportPilot."""

from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the cached SentenceTransformer model, loading it on first use."""
    global _model

    if _model is None:
        logger.info("Loading sentence-transformers model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Loaded sentence-transformers model: %s", MODEL_NAME)

    return _model


def embed_text(text: str) -> list[float]:
    """Generate an embedding vector for a single text string."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    if not text.strip():
        raise ValueError("text must not be empty or whitespace only")

    logger.debug("Generating embedding for one text")
    embedding = get_embedding_model().encode(text, convert_to_numpy=True)
    return [float(value) for value in embedding.tolist()]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for a list of text strings."""
    if not isinstance(texts, list):
        raise ValueError("texts must be a list of strings")

    if not texts:
        raise ValueError("texts must contain at least one string")

    invalid_indexes = [
        index for index, text in enumerate(texts) if not isinstance(text, str)
    ]
    if invalid_indexes:
        raise ValueError(
            "texts must contain only strings; invalid indexes: "
            + ", ".join(str(index) for index in invalid_indexes)
        )

    empty_indexes = [
        index for index, text in enumerate(texts) if not text.strip()
    ]
    if empty_indexes:
        raise ValueError(
            "texts must not contain empty or whitespace-only strings; invalid indexes: "
            + ", ".join(str(index) for index in empty_indexes)
        )

    logger.debug("Generating embeddings for %d texts", len(texts))
    embeddings = get_embedding_model().encode(texts, convert_to_numpy=True)
    return [
        [float(value) for value in embedding]
        for embedding in embeddings.tolist()
    ]
