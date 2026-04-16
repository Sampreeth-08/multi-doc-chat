from dataclasses import dataclass, field
from typing import List

from pydantic import BaseModel, Field


class CoTAnswer(BaseModel):
    """Structured LLM output for chain-of-thought QA."""

    reasoning: str = Field(description="Step-by-step reasoning through the provided context")
    answer: str = Field(description="Concise final answer to the question")


@dataclass
class QueryRequest:
    """Input to the RAG query engine."""

    question: str


@dataclass
class QueryResponse:
    """Output from the RAG query engine."""

    question: str
    answer: str
    sources: List[str] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Summary of a completed ingestion run."""

    files_loaded: int
    chunks_created: int
    vectorstore_path: str
    errors: List[str] = field(default_factory=list)
