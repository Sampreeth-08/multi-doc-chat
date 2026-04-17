from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from multi_doc_chat.api import session_store
from multi_doc_chat.api.app import app
from multi_doc_chat.exception.exceptions import MultiDocChatError
from multi_doc_chat.model.rag_model import CoTAnswer, IngestionResult


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_store():
    """Isolate tests by flushing the in-memory session store."""
    session_store._store.clear()
    yield
    session_store._store.clear()


@pytest.fixture
def mock_session(tmp_path):
    session = MagicMock()
    session.id = "test-session-abc"
    session.created_at = datetime(2024, 6, 1, 10, 0, 0)
    # upload_dir must support the `/` operator and write_bytes (real Path)
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True)
    session.upload_dir = upload_dir
    session.vectorstore_dir = tmp_path / "vectorstore"
    session.session_dir = tmp_path
    return session


@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.chat_with_reasoning.return_value = CoTAnswer(
        reasoning="Step 1: review context. Step 2: conclude.",
        answer="The answer is 42.",
    )
    return engine


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestIndexEndpoint:
    def test_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_returns_html_content_type(self, client):
        response = client.get("/")
        assert "text/html" in response.headers["content-type"]


# ---------------------------------------------------------------------------
# POST /start
# ---------------------------------------------------------------------------

class TestStartEndpoint:
    def test_returns_201_with_session_metadata(self, client, mock_session, mock_engine):
        ingestion_result = IngestionResult(
            files_loaded=2, chunks_created=8, vectorstore_path="/tmp/vs"
        )
        with (
            patch("multi_doc_chat.api.app.get_settings"),
            patch("multi_doc_chat.api.app.create_session", return_value=mock_session),
            patch("multi_doc_chat.api.app.IngestionPipeline") as mock_pipeline_cls,
            patch("multi_doc_chat.api.app.build_mmr_retriever", return_value=MagicMock()),
            patch("multi_doc_chat.api.app.ConversationalRAGEngine", return_value=mock_engine),
        ):
            mock_pipeline_cls.return_value.run.return_value = ingestion_result
            response = client.post(
                "/start",
                files=[
                    ("files", ("a.txt", b"content A", "text/plain")),
                    ("files", ("b.txt", b"content B", "text/plain")),
                ],
            )

        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == "test-session-abc"
        assert data["files_count"] == 2
        assert data["files_loaded"] == 2
        assert data["chunks_created"] == 8
        assert "started_at" in data

    def test_saves_session_to_store_after_start(self, client, mock_session, mock_engine):
        ingestion_result = IngestionResult(
            files_loaded=1, chunks_created=3, vectorstore_path="/tmp/vs"
        )
        with (
            patch("multi_doc_chat.api.app.get_settings"),
            patch("multi_doc_chat.api.app.create_session", return_value=mock_session),
            patch("multi_doc_chat.api.app.IngestionPipeline") as mock_pipeline_cls,
            patch("multi_doc_chat.api.app.build_mmr_retriever", return_value=MagicMock()),
            patch("multi_doc_chat.api.app.ConversationalRAGEngine", return_value=mock_engine),
        ):
            mock_pipeline_cls.return_value.run.return_value = ingestion_result
            client.post(
                "/start",
                files=[("files", ("doc.txt", b"hello", "text/plain"))],
            )

        assert session_store.get("test-session-abc") is not None

    def test_returns_500_and_cleans_up_on_ingestion_error(self, client, mock_session):
        with (
            patch("multi_doc_chat.api.app.get_settings"),
            patch("multi_doc_chat.api.app.create_session", return_value=mock_session),
            patch("multi_doc_chat.api.app.IngestionPipeline") as mock_pipeline_cls,
        ):
            mock_pipeline_cls.return_value.run.side_effect = MultiDocChatError("embed failed")
            response = client.post(
                "/start",
                files=[("files", ("bad.txt", b"", "text/plain"))],
            )

        assert response.status_code == 500
        assert "embed failed" in response.json()["detail"]
        mock_session.cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# POST /chat/{session_id}
# ---------------------------------------------------------------------------

class TestChatEndpoint:
    def test_returns_200_with_answer(self, client, mock_session, mock_engine):
        session_store.save("test-session-abc", mock_session, mock_engine)

        response = client.post(
            "/chat/test-session-abc",
            json={"question": "What is the main topic?"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-abc"
        assert data["question"] == "What is the main topic?"
        assert data["answer"] == "The answer is 42."
        assert "reasoning" in data

    def test_returns_404_for_unknown_session(self, client):
        response = client.post(
            "/chat/nonexistent",
            json={"question": "hello"},
        )
        assert response.status_code == 404
        assert "nonexistent" in response.json()["detail"]

    def test_returns_500_when_engine_raises(self, client, mock_session, mock_engine):
        mock_engine.chat_with_reasoning.side_effect = MultiDocChatError("LLM unavailable")
        session_store.save("err-session", mock_session, mock_engine)

        response = client.post("/chat/err-session", json={"question": "test"})

        assert response.status_code == 500
        assert "LLM unavailable" in response.json()["detail"]

    def test_calls_engine_with_correct_question(self, client, mock_session, mock_engine):
        session_store.save("test-session-abc", mock_session, mock_engine)
        client.post("/chat/test-session-abc", json={"question": "Deep question"})
        mock_engine.chat_with_reasoning.assert_called_once_with("Deep question")


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------

class TestDeleteSessionEndpoint:
    def test_returns_204_on_success(self, client, mock_session, mock_engine):
        session_store.save("to-delete", mock_session, mock_engine)
        response = client.delete("/sessions/to-delete")
        assert response.status_code == 204

    def test_removes_session_from_store(self, client, mock_session, mock_engine):
        session_store.save("to-delete", mock_session, mock_engine)
        client.delete("/sessions/to-delete")
        assert session_store.get("to-delete") is None

    def test_calls_session_cleanup(self, client, mock_session, mock_engine):
        session_store.save("to-delete", mock_session, mock_engine)
        client.delete("/sessions/to-delete")
        mock_session.cleanup.assert_called_once()

    def test_returns_404_for_unknown_session(self, client):
        response = client.delete("/sessions/ghost")
        assert response.status_code == 404
        assert "ghost" in response.json()["detail"]
