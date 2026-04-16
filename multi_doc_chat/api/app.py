from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from multi_doc_chat.api import session_store
from multi_doc_chat.api.models import ChatRequest, ChatResponse, RunResponse
from multi_doc_chat.config.settings import get_settings
from multi_doc_chat.exception.exceptions import MultiDocChatError
from multi_doc_chat.session.session import create_session
from multi_doc_chat.src.ingestion import IngestionPipeline
from multi_doc_chat.src.retrieval import ConversationalRAGEngine, build_mmr_retriever

_ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(
    title="Multi-Doc Chat API",
    description="Session-scoped RAG chat with chain-of-thought reasoning.",
)

app.mount("/static", StaticFiles(directory=str(_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(_ROOT / "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html")


@app.post("/start", response_model=RunResponse, status_code=201)
async def start(files: List[UploadFile] = File(...)) -> RunResponse:
    """Upload files, ingest them, and start a chat session.

    Accepts multipart/form-data with one or more files (PDF, TXT, DOCX).
    Returns the session_id needed for subsequent /chat calls.
    """
    settings = get_settings()
    session = create_session()

    # Save uploads to the session's upload directory
    saved_paths = []
    for upload in files:
        dest = session.upload_dir / (upload.filename or "file")
        dest.write_bytes(await upload.read())
        saved_paths.append(dest)

    try:
        pipeline = IngestionPipeline(settings)
        result = pipeline.run(files=saved_paths, vectorstore_dir=session.vectorstore_dir)
        session.cleanup_uploads()
    except MultiDocChatError as exc:
        session.cleanup()
        raise HTTPException(status_code=500, detail=str(exc))

    retriever = build_mmr_retriever(session.vectorstore_dir, settings)
    engine = ConversationalRAGEngine(settings=settings, retriever=retriever, cot=True)
    session_store.save(session.id, session, engine)

    return RunResponse(
        session_id=session.id,
        started_at=session.created_at.isoformat(),
        files_count=len(saved_paths),
        files_loaded=result.files_loaded,
        chunks_created=result.chunks_created,
    )


@app.post("/run", response_model=RunResponse, status_code=201)
def run() -> RunResponse:
    """Create a session, ingest data/, build an MMR retriever, and start a CoT engine.

    Returns the session_id needed for subsequent /chat calls.
    """
    settings = get_settings()

    session = create_session()

    try:
        pipeline = IngestionPipeline(settings)
        result = pipeline.run(vectorstore_dir=session.vectorstore_dir)
    except MultiDocChatError as exc:
        session.cleanup()
        raise HTTPException(status_code=500, detail=str(exc))

    retriever = build_mmr_retriever(session.vectorstore_dir, settings)
    engine = ConversationalRAGEngine(settings=settings, retriever=retriever, cot=True)

    session_store.save(session.id, session, engine)

    return RunResponse(
        session_id=session.id,
        started_at=session.created_at.isoformat(),
        files_loaded=result.files_loaded,
        chunks_created=result.chunks_created,
    )


@app.post("/chat/{session_id}", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest) -> ChatResponse:
    """Send a question to an active session and receive a CoT answer.

    Args:
        session_id: Returned by ``POST /run``.
        request: JSON body with a ``question`` field.
    """
    entry = session_store.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    _, engine = entry
    try:
        cot_answer = engine.chat_with_reasoning(request.question)
    except MultiDocChatError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        session_id=session_id,
        question=request.question,
        reasoning=cot_answer.reasoning,
        answer=cot_answer.answer,
    )


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str) -> None:
    """Delete a session and clean up its vector store from disk."""
    entry = session_store.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    session, _ = entry
    session.cleanup()
    session_store.remove(session_id)
