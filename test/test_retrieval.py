from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from multi_doc_chat.exception.exceptions import QueryError, VectorStoreNotFoundError
from multi_doc_chat.src.retrieval import RAGQueryEngine, _format_docs


# ---------------------------------------------------------------------------
# _format_docs helper
# ---------------------------------------------------------------------------

class TestFormatDocs:
    def test_joins_page_content_with_double_newline(self):
        docs = [
            Document(page_content="First chunk."),
            Document(page_content="Second chunk."),
        ]
        result = _format_docs(docs)
        assert result == "First chunk.\n\nSecond chunk."

    def test_single_document(self):
        docs = [Document(page_content="Only chunk.")]
        assert _format_docs(docs) == "Only chunk."

    def test_empty_list(self):
        assert _format_docs([]) == ""


# ---------------------------------------------------------------------------
# RAGQueryEngine
# ---------------------------------------------------------------------------

class TestRAGQueryEngine:
    @patch("multi_doc_chat.src.retrieval.FAISS")
    @patch("multi_doc_chat.src.retrieval.OpenAIEmbeddings")
    @patch("multi_doc_chat.src.retrieval.ChatOpenAI")
    def test_query_returns_string(
        self,
        mock_chat_cls,
        mock_embeddings_cls,
        mock_faiss_cls,
        sample_settings,
        tmp_path,
        mock_faiss_vectorstore,
    ):
        # Make vectorstore_dir appear to exist with an index.faiss file
        vs_dir = tmp_path / "vectorstore"
        vs_dir.mkdir()
        (vs_dir / "index.faiss").write_bytes(b"")

        settings_with_vs = sample_settings.__class__(
            **{**sample_settings.__dict__, "vectorstore_dir": vs_dir}
        )

        mock_embeddings_cls.return_value = MagicMock()
        mock_faiss_cls.load_local.return_value = mock_faiss_vectorstore

        # Build a fake chain that returns a known answer string
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = "Agentic AI is autonomous and goal-oriented."

        engine = RAGQueryEngine(settings_with_vs)
        engine._chain = fake_chain  # bypass chain construction

        answer = engine.query("What is Agentic AI?")
        assert isinstance(answer, str)
        assert len(answer) > 0
        fake_chain.invoke.assert_called_once_with("What is Agentic AI?")

    def test_query_raises_vector_store_not_found_when_dir_missing(self, sample_settings, tmp_path):
        # vectorstore_dir does not exist
        engine = RAGQueryEngine(sample_settings)
        with pytest.raises(VectorStoreNotFoundError):
            engine.query("Any question?")

    def test_query_raises_vector_store_not_found_when_index_missing(
        self, sample_settings, tmp_path
    ):
        # Dir exists but index.faiss is absent
        vs_dir = tmp_path / "vectorstore"
        vs_dir.mkdir()
        # No index.faiss written

        settings_without_index = sample_settings.__class__(
            **{**sample_settings.__dict__, "vectorstore_dir": vs_dir}
        )
        engine = RAGQueryEngine(settings_without_index)
        with pytest.raises(VectorStoreNotFoundError):
            engine.query("Any question?")

    def test_query_wraps_chain_errors_in_query_error(self, sample_settings):
        engine = RAGQueryEngine(sample_settings)
        fake_chain = MagicMock()
        fake_chain.invoke.side_effect = RuntimeError("upstream API failure")
        engine._chain = fake_chain  # skip _build_chain

        with pytest.raises(QueryError, match="upstream API failure"):
            engine.query("test?")

    def test_chain_built_only_once_across_multiple_queries(self, sample_settings):
        engine = RAGQueryEngine(sample_settings)
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = "answer"
        engine._chain = fake_chain

        engine.query("question 1")
        engine.query("question 2")

        # _build_chain was never called (chain was pre-set)
        assert fake_chain.invoke.call_count == 2
        # _chain is still the same object
        assert engine._chain is fake_chain

    @patch("multi_doc_chat.src.retrieval.FAISS")
    @patch("multi_doc_chat.src.retrieval.OpenAIEmbeddings")
    def test_query_with_sources_returns_dict_with_answer_and_sources(
        self,
        mock_embeddings_cls,
        mock_faiss_cls,
        sample_settings,
        tmp_path,
        mock_faiss_vectorstore,
    ):
        vs_dir = tmp_path / "vectorstore"
        vs_dir.mkdir()
        (vs_dir / "index.faiss").write_bytes(b"")

        settings_with_vs = sample_settings.__class__(
            **{**sample_settings.__dict__, "vectorstore_dir": vs_dir}
        )

        mock_embeddings_cls.return_value = MagicMock()
        mock_faiss_cls.load_local.return_value = mock_faiss_vectorstore

        fake_chain = MagicMock()
        fake_chain.invoke.return_value = "Mock answer."

        engine = RAGQueryEngine(settings_with_vs)
        engine._chain = fake_chain

        result = engine.query_with_sources("What is AI?")

        assert "answer" in result
        assert "sources" in result
        assert result["answer"] == "Mock answer."
        assert isinstance(result["sources"], list)
