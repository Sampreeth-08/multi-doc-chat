from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    session_id: str
    question: str
    reasoning: str
    answer: str


class RunResponse(BaseModel):
    session_id: str
    started_at: str
    files_count: int = 0
    files_loaded: int
    chunks_created: int
