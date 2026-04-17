from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from multi_doc_chat.exception.exceptions import QueryError, VectorStoreNotFoundError
from multi_doc_chat.model.rag_model import CoTAnswer
from multi_doc_chat.src.retrieval import (
    ConversationalRAGEngine,
    RAGQueryEngine,
    _format_docs,
    build_mmr_retriever,
)


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


# ---------------------------------------------------------------------------
# ConversationalRAGEngine
# ---------------------------------------------------------------------------

class TestConversationalRAGEngine:
    def _engine_with_mock_chain(self, settings, cot=False):
        """Return an engine with a pre-built mock chain to skip _build_conv_chain."""
        engine = ConversationalRAGEngine(settings=settings, cot=cot)
        fake_chain = MagicMock()
        engine._conv_chain = fake_chain
        return engine, fake_chain

    # -- chat_with_reasoning --

    def test_chat_with_reasoning_raises_when_cot_false(self, sample_settings):
        engine = ConversationalRAGEngine(settings=sample_settings, cot=False)
        engine._conv_chain = MagicMock()
        with pytest.raises(ValueError, match="cot=True"):
            engine.chat_with_reasoning("What is AI?")

    def test_chat_with_reasoning_returns_cot_answer(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=True)
        fake_chain.invoke.return_value = CoTAnswer(
            reasoning="Step 1: analyse context.", answer="The answer."
        )
        result = engine.chat_with_reasoning("What is AI?")
        assert isinstance(result, CoTAnswer)
        assert result.answer == "The answer."
        assert "Step 1" in result.reasoning

    def test_chat_with_reasoning_appends_to_history(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=True)
        fake_chain.invoke.return_value = CoTAnswer(reasoning="r", answer="ans")
        engine.chat_with_reasoning("Q1")
        assert len(engine.history) == 2
        assert isinstance(engine.history[0], HumanMessage)
        assert isinstance(engine.history[1], AIMessage)

    def test_chat_with_reasoning_wraps_errors_in_query_error(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=True)
        fake_chain.invoke.side_effect = RuntimeError("LLM timeout")
        with pytest.raises(QueryError, match="LLM timeout"):
            engine.chat_with_reasoning("Q?")

    # -- chat (cot=False) --

    def test_chat_returns_string_answer(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=False)
        fake_chain.invoke.return_value = "Plain answer."
        result = engine.chat("What is RAG?")
        assert result == "Plain answer."

    def test_chat_delegates_to_chat_with_reasoning_when_cot_true(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=True)
        fake_chain.invoke.return_value = CoTAnswer(reasoning="r", answer="delegate answer")
        result = engine.chat("What is RAG?")
        assert result == "delegate answer"

    def test_chat_accumulates_history_across_turns(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=False)
        fake_chain.invoke.return_value = "answer"
        engine.chat("Q1")
        engine.chat("Q2")
        assert len(engine.history) == 4  # 2 human + 2 AI messages

    def test_chat_wraps_errors_in_query_error(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=False)
        fake_chain.invoke.side_effect = RuntimeError("chain failure")
        with pytest.raises(QueryError, match="chain failure"):
            engine.chat("Q?")

    # -- clear_history / history property --

    def test_clear_history_resets_to_empty(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=False)
        fake_chain.invoke.return_value = "a"
        engine.chat("Q1")
        assert len(engine.history) == 2
        engine.clear_history()
        assert engine.history == []

    def test_history_property_returns_copy(self, sample_settings):
        engine = ConversationalRAGEngine(settings=sample_settings)
        hist = engine.history
        hist.append(HumanMessage(content="injected"))
        assert engine.history == []  # internal list unaffected

    # -- lazy chain building --

    def test_chain_built_only_once(self, sample_settings):
        engine, fake_chain = self._engine_with_mock_chain(sample_settings, cot=False)
        fake_chain.invoke.return_value = "a"
        engine.chat("Q1")
        engine.chat("Q2")
        # Chain was pre-set; _conv_chain must still be the same object
        assert engine._conv_chain is fake_chain
        assert fake_chain.invoke.call_count == 2


# ---------------------------------------------------------------------------
# build_mmr_retriever
# ---------------------------------------------------------------------------

class TestBuildMmrRetriever:
    def test_raises_when_vectorstore_dir_missing(self, sample_settings, tmp_path):
        missing_dir = tmp_path / "no_such_dir"
        with pytest.raises(VectorStoreNotFoundError):
            build_mmr_retriever(missing_dir, sample_settings)

    def test_raises_when_index_file_absent(self, sample_settings, tmp_path):
        vs_dir = tmp_path / "vectorstore"
        vs_dir.mkdir()
        # No index.faiss written
        with pytest.raises(VectorStoreNotFoundError):
            build_mmr_retriever(vs_dir, sample_settings)

    @patch("multi_doc_chat.src.retrieval.OpenAIEmbeddings")
    @patch("multi_doc_chat.src.retrieval.FAISS")
    def test_returns_retriever_when_index_exists(
        self, mock_faiss_cls, mock_embeddings_cls, sample_settings, tmp_path, mock_faiss_vectorstore
    ):
        vs_dir = tmp_path / "vectorstore"
        vs_dir.mkdir()
        (vs_dir / "index.faiss").write_bytes(b"")

        mock_embeddings_cls.return_value = MagicMock()
        mock_faiss_cls.load_local.return_value = mock_faiss_vectorstore

        retriever = build_mmr_retriever(vs_dir, sample_settings)

        mock_faiss_vectorstore.as_retriever.assert_called_once()
        call_kwargs = mock_faiss_vectorstore.as_retriever.call_args[1]
        assert call_kwargs["search_type"] == "mmr"
        assert call_kwargs["search_kwargs"]["k"] == sample_settings.retriever_k
