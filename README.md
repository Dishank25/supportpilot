# SupportPilot

Multi-Agent Customer Support API using RAG

## Overview

SupportPilot is a pet-care customer support API that uses a small multi-agent architecture to classify customer intent, retrieve relevant FAQ knowledge with semantic embeddings, and generate grounded LLM responses through a FastAPI `/chat` endpoint.

## Architecture

```text
User Message
↓
Intent Agent
↓
Retrieval Agent
↓
FAQ Repository
↓
ChromaDB
↓
Response Agent
↓
Final Response
```

- `IntentAgent`: rule-based classifier for `ORDER`, `REFUND`, `PRODUCT`, `ACCOUNT`, and `GENERAL`.
- `RetrievalAgent`: confidence-thresholded retrieval layer that decides whether the retrieved context is reliable enough to answer.
- `FAQRepository`: repository implementation backed by `knowledge/faq.json`.
- `ChromaDB`: persisted vector store used for semantic FAQ search.
- `ResponseAgent`: OpenAI grounded generation layer that answers only from retrieved context or returns a fallback when no reliable match is found.
- `main.py`: FastAPI orchestration for the `/chat` endpoint.

## Key Design Decisions

### Why rule-based intent classification?

The current system supports only five intents, so a rule-based classifier is enough for V1. It is deterministic, fast, has zero LLM cost, and is easy to maintain by editing keyword lists.

### Why a repository pattern?

The current knowledge source is FAQ JSON, but the system is designed so future sources such as PDFs or websites can implement the same repository contract. Repositories return a neutral schema:

```json
{
  "id": "...",
  "title": "...",
  "content": "...",
  "category": "..."
}
```

For FAQs, `title` is the question and `content` is the answer. This prevents downstream refactoring because the retrieval and response layers do not need to know whether knowledge came from FAQs, PDFs, or webpages.

### Why confidence-thresholded retrieval?

The repository returns cosine distance from ChromaDB, where lower distance means a better match. The retrieval layer converts distance into a normalized confidence score and applies a confidence threshold. This prevents weak matches from being passed to the LLM as if they were reliable, reducing hallucination risk. When confidence is too low, the system returns a fallback response.

### Why top_k=3?

During review, `top_k=1` discarded the best answer for some queries. The API now retrieves `top_k=3`, allowing the LLM to see nearby candidates while confidence still comes from the best retrieval result.

### Why startup initialization?

The embedding model and ChromaDB initialization are expensive relative to normal request handling. FastAPI initializes the agents once at startup and stores them in app state instead of recreating them per request.

## Example Request

curl:

```bash
curl -X POST http://127.0.0.1:8000/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"message\":\"Where is my order?\"}"
```

PowerShell:

```powershell
$headers = @{ "Content-Type" = "application/json" }
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8000/chat" `
  -Method Post `
  -Headers $headers `
  -Body '{"message":"Where is my order?"}'
```

## Example Response

```json
{
  "intent": "ORDER",
  "confidence": 0.757,
  "response": "To check the status of your order, please visit your account orders page. If tracking is available, a carrier link will appear once the order has shipped."
}
```

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.venv\Scripts\activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Add your OpenAI API key to `.env`:

```text
OPENAI_API_KEY=...
```

5. Run the API:

```powershell
uvicorn main:app --reload
```

## Project Structure

```text
supportpilot/
├── agents/
│   ├── intent_agent.py
│   ├── retrieval_agent.py
│   └── response_agent.py
├── repositories/
│   ├── knowledge_repository.py
│   └── faq_repository.py
├── services/
│   └── embeddings.py
├── knowledge/
│   └── faq.json
├── vectorstore/
├── main.py
├── requirements.txt
└── requirements.lock
```

- `agents/`: intent classification, retrieval decisioning, and response generation.
- `repositories/`: knowledge repository contracts and FAQ-backed ChromaDB implementation.
- `services/`: shared embedding model utilities.
- `knowledge/`: source FAQ dataset.
- `vectorstore/`: persisted ChromaDB data.
- `main.py`: FastAPI app and `/chat` route.

## Future Extensions

Designed to support, but not implemented in V1:

- PDF knowledge bases
- Website knowledge bases
- Escalation agent
- Sentiment agent
- Docker deployment

## Notes

SupportPilot is a portfolio and educational project. It demonstrates practical RAG architecture and agent orchestration, but it is not intended as enterprise production software.
