from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from multi_doc_chat.config.settings import Settings


@pytest.fixture
def sample_settings(tmp_path: Path) -> Settings:
    """Settings with dummy API key and tmp_path-based dirs so no real I/O occurs."""
    return Settings(
        openai_api_key="sk-test-dummy-key",
        openai_base_url="https://api.openai.com/v1",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        chunk_size=200,
        chunk_overlap=20,
        retriever_k=3,
        mmr_fetch_k=10,
        mmr_lambda_mult=0.5,
        data_dir=tmp_path / "data",
        vectorstore_dir=tmp_path / "vectorstore",
        sessions_dir=tmp_path / "sessions",
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """Two short Documents that fit within the default chunk_size=200."""
    return [
        Document(
            page_content="Agentic AI is proactive and goal-oriented.",
            metadata={"source": "test.txt"},
        ),
        Document(
            page_content="It can reason, plan, and use tools autonomously.",
            metadata={"source": "test.txt"},
        ),
    ]


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """A real .txt file in tmp_path with known content."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    f = data_dir / "sample.txt"
    f.write_text("Agentic AI is proactive.\nIt can reason and plan.", encoding="utf-8")
    return f


@pytest.fixture
def mock_faiss_vectorstore() -> MagicMock:
    """A MagicMock that mimics a FAISS vector store."""
    vs = MagicMock()
    retriever = MagicMock()
    retriever.invoke = MagicMock(
        return_value=[
            Document(page_content="mock context", metadata={"source": "mock.txt"})
        ]
    )
    vs.as_retriever.return_value = retriever
    vs.similarity_search.return_value = [
        Document(page_content="mock context", metadata={"source": "mock.txt"})
    ]
    return vs
