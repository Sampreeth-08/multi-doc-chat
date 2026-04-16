"""In-memory store mapping session_id → (Session, ConversationalRAGEngine).

Intentionally simple: one process, no persistence. Replace with Redis or a DB
if you need multi-process or durable sessions.
"""

from multi_doc_chat.session.session import Session
from multi_doc_chat.src.retrieval import ConversationalRAGEngine

_store: dict[str, tuple[Session, ConversationalRAGEngine]] = {}


def save(session_id: str, session: Session, engine: ConversationalRAGEngine) -> None:
    _store[session_id] = (session, engine)


def get(session_id: str) -> tuple[Session, ConversationalRAGEngine] | None:
    return _store.get(session_id)


def remove(session_id: str) -> None:
    _store.pop(session_id, None)
