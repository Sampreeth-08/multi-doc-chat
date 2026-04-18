# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Package manager:** `uv` (not pip). Dependencies in `pyproject.toml`, lockfile in `uv.lock`.

```bash
# Install dependencies
uv sync

# Run the FastAPI server (development)
uv run uvicorn multi_doc_chat.api.app:app --reload

# CLI entry point
uv run python main.py ingest
uv run python main.py query "your question"
uv run python main.py chat
uv run python main.py run        # ingest data/ + MMR retriever + chat

# Run all tests
uv run pytest test/ -v

# Run a single test
uv run pytest test/test_api.py::TestIndexEndpoint::test_returns_200 -v

# Docker
docker-compose up --build
```

**Runtime directories** (`data/`, `logs/`, `sessions/`, `vectorstore/`) are created on demand and excluded from git/Docker.

## Architecture

This is a multi-document RAG (Retrieval-Augmented Generation) chat app. Users upload documents, which are embedded into a FAISS vector store, then queried via a conversational LLM with chain-of-thought reasoning.

**Request flow (web):**
1. `POST /start` — upload files → `IngestionPipeline` loads/chunks/embeds → FAISS index saved to session dir → `ConversationalRAGEngine` created and stored in `session_store`
2. `POST /chat/{session_id}` — question → history-aware retriever rewrites it as standalone query → retrieved chunks → CoT LLM answer → `CoTAnswer` (reasoning + answer)
3. `DELETE /sessions/{session_id}` — cleanup

**Core modules:**

| Module | Responsibility |
|--------|---------------|
| `multi_doc_chat/src/ingestion.py` | `DocumentLoader` → `DocumentChunker` → `VectorStoreBuilder` → `IngestionPipeline` (LCEL chain) |
| `multi_doc_chat/src/retrieval.py` | `RAGQueryEngine` (single-turn), `ConversationalRAGEngine` (multi-turn with history rewrite + CoT), `build_mmr_retriever()` |
| `multi_doc_chat/api/app.py` | FastAPI routes; thin layer over ingestion + retrieval |
| `multi_doc_chat/api/session_store.py` | In-memory dict: `session_id → (Session, ConversationalRAGEngine)` — single-process only |
| `multi_doc_chat/config/settings.py` | Frozen `Settings` dataclass loaded once from env via `get_settings()` |
| `multi_doc_chat/prompts/templates.py` | LangChain prompt builders: QA, conversational QA, CoT, contextualize (standalone-question rewrite) |
| `multi_doc_chat/model/rag_model.py` | `CoTAnswer` (Pydantic, structured LLM output), `IngestionResult`, `QueryRequest/Response` |
| `multi_doc_chat/session/session.py` | `Session` dataclass with UUID-scoped upload + vectorstore dirs |

**Key design decisions:**
- `ConversationalRAGEngine` uses a two-chain pattern: a `contextualize_chain` rewrites follow-up questions into standalone queries before retrieval, then a separate QA chain answers using retrieved chunks.
- CoT is toggled per-engine instance (`cot=True`); structured output is parsed into `CoTAnswer` via `with_structured_output`.
- FAISS indexes are persisted to disk per session; `_load_faiss()` reloads them on demand.
- `session_store` is in-memory — sessions are lost on server restart.

## Environment

Copy `.env` and set:
```
OPENAI_API_KEY=...
OPENAI_BASE_URL=...          # optional, for proxies
LANGCHAIN_TRACING_V2=...     # optional LangSmith tracing
```
