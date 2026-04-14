from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from multi_doc_chat.exception.exceptions import (
    DocumentLoadError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from multi_doc_chat.src.ingestion import (
    DocumentChunker,
    DocumentLoader,
    IngestionPipeline,
    VectorStoreBuilder,
)


# ---------------------------------------------------------------------------
# DocumentLoader
# ---------------------------------------------------------------------------

class TestDocumentLoaderLoadFile:
    def test_loads_txt_file(self, sample_settings, sample_txt_file):
        loader = DocumentLoader(sample_settings)
        docs = loader.load_file(sample_txt_file)
        assert len(docs) >= 1
        combined = " ".join(d.page_content for d in docs)
        assert "Agentic AI" in combined

    def test_raises_unsupported_extension(self, sample_settings, tmp_path):
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("a,b,c")
        loader = DocumentLoader(sample_settings)
        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            loader.load_file(csv_file)
        assert "csv" in str(exc_info.value).lower()

    def test_raises_document_load_error_on_broken_file(self, sample_settings, tmp_path):
        # Create a .pdf file with invalid content so PyPDFLoader will fail
        bad_pdf = tmp_path / "bad.pdf"
        bad_pdf.write_bytes(b"not a real pdf")
        loader = DocumentLoader(sample_settings)
        with pytest.raises(DocumentLoadError):
            loader.load_file(bad_pdf)

    def test_unsupported_is_subclass_of_document_load_error(self, sample_settings, tmp_path):
        f = tmp_path / "file.xlsx"
        f.write_bytes(b"PK")
        loader = DocumentLoader(sample_settings)
        with pytest.raises(DocumentLoadError):
            loader.load_file(f)


class TestDocumentLoaderLoadDirectory:
    def test_loads_all_txt_files(self, sample_settings, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "a.txt").write_text("File A content.")
        (data_dir / "b.txt").write_text("File B content.")
        loader = DocumentLoader(sample_settings)
        docs = loader.load_directory(data_dir)
        combined = " ".join(d.page_content for d in docs)
        assert "File A" in combined
        assert "File B" in combined

    def test_continues_past_single_failing_file(self, sample_settings, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        good = data_dir / "good.txt"
        bad = data_dir / "bad.pdf"
        good.write_text("Good content.")
        bad.write_bytes(b"not a pdf")

        loader = DocumentLoader(sample_settings)
        # Should not raise — bad.pdf is skipped with a warning
        docs = loader.load_directory(data_dir)
        assert any("Good content" in d.page_content for d in docs)

    def test_raises_when_zero_docs_loaded(self, sample_settings, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # Only unsupported files
        (data_dir / "skip.csv").write_text("col1,col2")
        loader = DocumentLoader(sample_settings)
        with pytest.raises(DocumentLoadError):
            loader.load_directory(data_dir)


# ---------------------------------------------------------------------------
# DocumentChunker
# ---------------------------------------------------------------------------

class TestDocumentChunker:
    def test_chunk_returns_documents(self, sample_settings, sample_documents):
        chunker = DocumentChunker(sample_settings)
        chunks = chunker.chunk(sample_documents)
        assert len(chunks) >= 1
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunks_respect_size_limit(self, sample_settings):
        # Create a single large document
        large_doc = Document(page_content="word " * 500, metadata={"source": "big.txt"})
        chunker = DocumentChunker(sample_settings)
        chunks = chunker.chunk([large_doc])
        for chunk in chunks:
            assert len(chunk.page_content) <= sample_settings.chunk_size + 50  # small tolerance

    def test_raises_on_empty_input(self, sample_settings):
        chunker = DocumentChunker(sample_settings)
        with pytest.raises(ValueError, match="empty"):
            chunker.chunk([])

    def test_metadata_preserved(self, sample_settings, sample_documents):
        chunker = DocumentChunker(sample_settings)
        chunks = chunker.chunk(sample_documents)
        sources = {c.metadata.get("source") for c in chunks}
        assert "test.txt" in sources


# ---------------------------------------------------------------------------
# VectorStoreBuilder
# ---------------------------------------------------------------------------

class TestVectorStoreBuilder:
    @patch("multi_doc_chat.src.ingestion.FAISS")
    @patch("multi_doc_chat.src.ingestion.OpenAIEmbeddings")
    def test_build_and_save_calls_save_local(
        self, mock_embeddings_cls, mock_faiss_cls, sample_settings, sample_documents, tmp_path
    ):
        # Arrange
        mock_embeddings_cls.return_value = MagicMock()
        mock_vs = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_vs

        settings = sample_settings  # vectorstore_dir is tmp_path/vectorstore
        builder = VectorStoreBuilder(settings)
        builder.build_and_save(sample_documents)

        # Assert FAISS.from_documents was called
        mock_faiss_cls.from_documents.assert_called_once()
        # Assert save_local was called with the correct path
        mock_vs.save_local.assert_called_once_with(str(settings.vectorstore_dir))

    @patch("multi_doc_chat.src.ingestion.FAISS")
    @patch("multi_doc_chat.src.ingestion.OpenAIEmbeddings")
    def test_build_and_save_raises_vector_store_error_on_failure(
        self, mock_embeddings_cls, mock_faiss_cls, sample_settings, sample_documents
    ):
        mock_embeddings_cls.return_value = MagicMock()
        mock_faiss_cls.from_documents.side_effect = RuntimeError("embedding API error")
        builder = VectorStoreBuilder(sample_settings)
        with pytest.raises(VectorStoreError, match="embedding API error"):
            builder.build_and_save(sample_documents)


# ---------------------------------------------------------------------------
# IngestionPipeline (integration-style, all I/O mocked)
# ---------------------------------------------------------------------------

class TestIngestionPipeline:
    @patch("multi_doc_chat.src.ingestion.FAISS")
    @patch("multi_doc_chat.src.ingestion.OpenAIEmbeddings")
    def test_run_returns_ingestion_result(
        self, mock_embeddings_cls, mock_faiss_cls, sample_settings, tmp_path
    ):
        # Set up a real data dir with one txt file
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "doc.txt").write_text("Agentic AI is proactive and goal-oriented.")

        mock_embeddings_cls.return_value = MagicMock()
        mock_vs = MagicMock()
        mock_faiss_cls.from_documents.return_value = mock_vs

        pipeline = IngestionPipeline(sample_settings)
        result = pipeline.run(data_dir)

        assert result.files_loaded >= 1
        assert result.chunks_created >= 1
        assert "vectorstore" in result.vectorstore_path
        mock_vs.save_local.assert_called_once()
