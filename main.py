"""FastAPI entry point for SupportPilot chat orchestration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from agents.intent_agent import IntentAgent
from agents.response_agent import ResponseAgent
from agents.retrieval_agent import RetrievalAgent


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    message: str | None = None


class ChatResponse(BaseModel):
    """Outgoing chat response payload."""

    intent: str
    confidence: float
    response: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agents once for the app lifecycle."""
    app.state.intent_agent = IntentAgent()
    app.state.retrieval_agent = RetrievalAgent()
    app.state.response_agent = ResponseAgent()
    yield


app = FastAPI(title="SupportPilot", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, fastapi_request: Request) -> ChatResponse:
    """Classify, retrieve, and generate a support response."""
    message = request.message
    if message is None or not message.strip():
        raise HTTPException(
            status_code=400,
            detail="message is required and must not be empty",
        )

    try:
        intent_agent: IntentAgent = fastapi_request.app.state.intent_agent
        retrieval_agent: RetrievalAgent = fastapi_request.app.state.retrieval_agent
        response_agent: ResponseAgent = fastapi_request.app.state.response_agent

        intent = intent_agent.classify(message)
        retrieval_result: dict[str, Any] = retrieval_agent.retrieve(message, top_k=1)
        response = response_agent.generate(message, retrieval_result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return ChatResponse(
        intent=intent,
        confidence=float(retrieval_result["confidence"]),
        response=response,
    )
