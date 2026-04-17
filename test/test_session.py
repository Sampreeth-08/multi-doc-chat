from pathlib import Path
from unittest.mock import patch

import pytest

from multi_doc_chat.config.settings import Settings
from multi_doc_chat.session.session import Session, create_session


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_api_key="sk-test",
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


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------

class TestCreateSession:
    def test_returns_session_with_unique_uuid(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            s1 = create_session()
            s2 = create_session()
        assert s1.id != s2.id

    def test_creates_upload_dir_on_disk(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        assert session.upload_dir.is_dir()

    def test_creates_vectorstore_dir_on_disk(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        assert session.vectorstore_dir.is_dir()

    def test_session_dir_is_parent_of_upload_dir(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        assert session.upload_dir.parent == session.session_dir
        assert session.vectorstore_dir.parent == session.session_dir

    def test_created_at_is_set(self, test_settings):
        from datetime import datetime
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        assert isinstance(session.created_at, datetime)


# ---------------------------------------------------------------------------
# Session.cleanup_uploads
# ---------------------------------------------------------------------------

class TestSessionCleanupUploads:
    def test_removes_upload_dir(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        assert session.upload_dir.exists()
        session.cleanup_uploads()
        assert not session.upload_dir.exists()

    def test_leaves_vectorstore_dir_intact(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        session.cleanup_uploads()
        assert session.vectorstore_dir.exists()

    def test_safe_to_call_when_upload_dir_already_gone(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        session.cleanup_uploads()
        session.cleanup_uploads()  # should not raise


# ---------------------------------------------------------------------------
# Session.cleanup
# ---------------------------------------------------------------------------

class TestSessionCleanup:
    def test_removes_entire_session_dir(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        assert session.session_dir.exists()
        session.cleanup()
        assert not session.session_dir.exists()

    def test_safe_to_call_when_session_dir_already_gone(self, test_settings):
        with patch("multi_doc_chat.session.session.get_settings", return_value=test_settings):
            session = create_session()
        session.cleanup()
        session.cleanup()  # should not raise
