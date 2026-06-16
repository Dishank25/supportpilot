"""Response generation agent for grounded customer support answers."""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_TOKENS = 250
OPENAI_TIMEOUT_SECONDS = 30

FALLBACK_RESPONSE = (
    "I couldn't find information about that in our knowledge base. "
    "Please contact customer support for further assistance."
)
PLACEHOLDER_API_KEY = "YOUR_API_KEY_HERE"

SYSTEM_INSTRUCTIONS = (
    "You are a concise, helpful pet-care customer support assistant. "
    "Answer using only the provided context. Do not invent facts. "
    "If the answer is not contained in the context, say you do not know."
)

logger = logging.getLogger(__name__)


class ResponseAgent:
    """Generate customer-facing responses from retrieved knowledge context."""

    def __init__(self) -> None:
        load_dotenv()
        self._client: OpenAI | None = None

    def generate(self, question: str, retrieval_result: dict, intent: str) -> str:
        """Generate a grounded answer or return a fallback for no-match results."""
        _validate_question(question)
        _validate_retrieval_result(retrieval_result)
        _validate_intent(intent)

        if not retrieval_result["match_found"]:
            logger.info("Returning fallback response for no-match retrieval result")
            return FALLBACK_RESPONSE

        context = _build_context(retrieval_result["results"])
        prompt = _build_prompt(question, retrieval_result, intent, context)

        logger.info("Generating grounded response with model %s", MODEL_NAME)
        response = self._get_client().responses.create(
            model=MODEL_NAME,
            instructions=SYSTEM_INSTRUCTIONS,
            input=prompt,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
        )

        answer = _extract_response_text(response)
        if not answer:
            raise RuntimeError("OpenAI response did not contain text output")

        return answer

    def _get_client(self) -> OpenAI:
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY is missing; add it to .env before generating "
                    "matched responses"
                )

            if api_key == PLACEHOLDER_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY is still the placeholder value; replace it in "
                    ".env before generating matched responses"
                )

            self._client = OpenAI(
                api_key=api_key,
                timeout=OPENAI_TIMEOUT_SECONDS,
            )

        return self._client


def _validate_question(question: str) -> None:
    if not isinstance(question, str):
        raise ValueError("question must be a string")

    if not question.strip():
        raise ValueError("question must not be empty or whitespace only")


def _validate_intent(intent: str) -> None:
    if not isinstance(intent, str):
        raise ValueError("intent must be a string")

    if not intent.strip():
        raise ValueError("intent must not be empty or whitespace only")


def _validate_retrieval_result(retrieval_result: dict) -> None:
    if not isinstance(retrieval_result, dict):
        raise ValueError("retrieval_result must be a dictionary")

    required_keys = {"match_found", "confidence", "results"}
    missing_keys = required_keys - set(retrieval_result)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"retrieval_result is missing required keys: {missing}")

    if not isinstance(retrieval_result["match_found"], bool):
        raise ValueError("retrieval_result['match_found'] must be a boolean")

    confidence = retrieval_result["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("retrieval_result['confidence'] must be a number")

    results = retrieval_result["results"]
    if not isinstance(results, list):
        raise ValueError("retrieval_result['results'] must be a list")

    if retrieval_result["match_found"] and not results:
        raise ValueError(
            "retrieval_result['results'] must contain at least one result when "
            "match_found is True"
        )

    for index, result in enumerate(results):
        _validate_result_item(result, index)


def _validate_result_item(result: Any, index: int) -> None:
    if not isinstance(result, dict):
        raise ValueError(f"retrieval_result['results'][{index}] must be a dictionary")

    required_keys = {"id", "title", "content", "category"}
    missing_keys = required_keys - set(result)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"retrieval_result['results'][{index}] is missing required keys: {missing}"
        )

    for key in required_keys:
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(
                f"retrieval_result['results'][{index}]['{key}'] must be a non-empty string"
            )


def _build_context(results: list[dict]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"Context item {index}",
                    f"Title: {result['title']}",
                    f"Category: {result['category']}",
                    f"Content: {result['content']}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_prompt(
    question: str,
    retrieval_result: dict,
    intent: str,
    context: str,
) -> str:
    return "\n\n".join(
        [
            f"Customer question: {question}",
            f"Intent: {intent}",
            f"Retrieval confidence: {retrieval_result['confidence']}",
            (
                "Use intent only as response framing/context. Do not use it to "
                "filter retrieval or override retrieved knowledge."
            ),
            "Retrieved context:",
            context,
            "Write the customer-facing answer.",
        ]
    )


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_parts: list[str] = []
    for output_item in getattr(response, "output", []) or []:
        for content_item in getattr(output_item, "content", []) or []:
            text = getattr(content_item, "text", None)
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

    return "\n".join(text_parts).strip()
