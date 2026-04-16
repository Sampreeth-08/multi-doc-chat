from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from multi_doc_chat.config.settings import Settings, get_settings
from multi_doc_chat.exception.exceptions import (
    DocumentLoadError,
    UnsupportedFileTypeError,
    VectorStoreError,
)
from multi_doc_chat.logger.logger import get_logger
from multi_doc_chat.model.rag_model import IngestionResult
from multi_doc_chat.utils.file_utils import ensure_dir, get_extension, iter_supported_files

logger = get_logger(__name__)


class DocumentLoader:
    """Loads raw documents from a directory, dispatching to format-specific loaders."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def load_file(self, file_path: Path) -> List[Document]:
        """Load a single file and return a list of LangChain Documents.

        Dispatches to the appropriate LangChain loader based on file extension:
        - ``.pdf``  → PyPDFLoader (one Document per page)
        - ``.txt``  → TextLoader (one Document for the whole file)
        - ``.docx`` / ``.doc`` → Docx2txtLoader (one Document for the whole file)

        Args:
            file_path: Path to the document file.

        Returns:
            List[Document]: Loaded documents with populated page_content and metadata.

        Raises:
            UnsupportedFileTypeError: If the extension is not supported.
            DocumentLoadError: If the underlying loader raises any exception.
        """
        ext = get_extension(file_path)
        logger.debug("Loading file: %s (extension: %s)", file_path, ext)

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif ext == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif ext in (".docx", ".doc"):
                loader = Docx2txtLoader(str(file_path))
            else:
                raise UnsupportedFileTypeError(
                    file_path=str(file_path),
                    reason=f"Extension '{ext}' is not supported. "
                    f"Supported: .pdf, .txt, .doc, .docx",
                )

            documents = loader.load()
            logger.debug("Loaded %d document(s) from %s", len(documents), file_path)
            return documents

        except UnsupportedFileTypeError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                file_path=str(file_path),
                reason=str(exc),
            ) from exc

    def load_files(self, file_paths: List[Path]) -> List[Document]:
        """Load a specific list of files and return their Documents.

        Logs a WARNING and continues on per-file errors so one bad file does
        not abort the whole upload batch.

        Args:
            file_paths: Explicit list of file paths to load.

        Returns:
            List[Document]: Combined documents from all successfully loaded files.

        Raises:
            DocumentLoadError: If no documents could be loaded from any file.
        """
        logger.info("Loading %d uploaded file(s)…", len(file_paths))
        all_documents: List[Document] = []
        errors: List[str] = []

        for file_path in file_paths:
            try:
                docs = self.load_file(file_path)
                all_documents.extend(docs)
                logger.info("Successfully loaded: %s (%d doc(s))", file_path.name, len(docs))
            except (DocumentLoadError, UnsupportedFileTypeError) as exc:
                logger.warning("Skipping file due to load error: %s", exc)
                errors.append(str(exc))

        if not all_documents:
            msg = "No documents could be loaded from the uploaded files."
            if errors:
                msg += f" Errors: {'; '.join(errors)}"
            raise DocumentLoadError(file_path="<uploaded files>", reason=msg)

        return all_documents

    def load_directory(self, data_dir: Path | None = None) -> List[Document]:
        """Load all supported files under data_dir.

        Logs a WARNING and continues on per-file errors — the pipeline is not
        aborted by a single bad file. Raises DocumentLoadError only if zero
        documents were successfully loaded.

        Args:
            data_dir: Directory to scan. Defaults to settings.data_dir.

        Returns:
            List[Document]: Combined list of Documents from all loaded files.

        Raises:
            DocumentLoadError: If no documents could be loaded from any file.
        """
        directory = data_dir or self._settings.data_dir
        logger.info("Loading documents from directory: %s", directory)

        all_documents: List[Document] = []
        errors: List[str] = []

        for file_path in iter_supported_files(directory):
            try:
                docs = self.load_file(file_path)
                all_documents.extend(docs)
                logger.info("Successfully loaded: %s (%d doc(s))", file_path.name, len(docs))
            except DocumentLoadError as exc:
                logger.warning("Skipping file due to load error: %s", exc)
                errors.append(str(exc))

        if not all_documents:
            msg = f"No documents could be loaded from '{directory}'."
            if errors:
                msg += f" Errors encountered: {'; '.join(errors)}"
            raise DocumentLoadError(file_path=str(directory), reason=msg)

        logger.info(
            "Finished loading directory: %d total document(s) from %d file(s) (%d error(s))",
            len(all_documents),
            len(all_documents) - len(errors),
            len(errors),
        )
        return all_documents


class DocumentChunker:
    """Splits documents into overlapping chunks suitable for embedding."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            length_function=len,
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into overlapping chunks.

        Args:
            documents: List of LangChain Documents to split.

        Returns:
            List[Document]: Chunked documents with preserved metadata.

        Raises:
            ValueError: If the documents list is empty.
        """
        if not documents:
            raise ValueError("Cannot chunk an empty document list.")

        logger.debug(
            "Chunking %d document(s) with chunk_size=%d, chunk_overlap=%d",
            len(documents),
            self._settings.chunk_size,
            self._settings.chunk_overlap,
        )
        chunks = self._splitter.split_documents(documents)
        logger.info("Created %d chunk(s) from %d document(s)", len(chunks), len(documents))
        return chunks


class VectorStoreBuilder:
    """Creates and persists a FAISS vector store from document chunks."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = OpenAIEmbeddings(
            model=self._settings.openai_embedding_model,
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url,
        )

    def build_and_save(self, chunks: List[Document], vectorstore_dir: Path | None = None) -> None:
        """Embed chunks with OpenAI and persist the FAISS index to disk.

        The index is saved as two files: ``index.faiss`` (the ANN index) and
        ``index.pkl`` (the docstore).

        Args:
            chunks: Chunked documents to embed and index.
            vectorstore_dir: Override the destination directory. Defaults to
                ``settings.vectorstore_dir``.

        Raises:
            VectorStoreError: If embedding or FAISS operations fail.
        """
        logger.info("Building FAISS vector store from %d chunk(s)…", len(chunks))
        try:
            vectorstore = FAISS.from_documents(chunks, self._embeddings)
        except Exception as exc:
            raise VectorStoreError(f"Failed to build FAISS index: {exc}") from exc

        self._save(vectorstore, vectorstore_dir)

    def _save(self, vectorstore: FAISS, vectorstore_dir: Path | None = None) -> None:
        """Persist the FAISS index and docstore pickle to vectorstore_dir."""
        dest = vectorstore_dir or self._settings.vectorstore_dir
        ensure_dir(dest)
        try:
            vectorstore.save_local(str(dest))
            logger.info("Vector store saved to: %s", dest)
        except Exception as exc:
            raise VectorStoreError(f"Failed to save FAISS index to '{dest}': {exc}") from exc


class IngestionPipeline:
    """Orchestrates the full load → chunk → embed → store pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self.loader = DocumentLoader(self._settings)
        self.chunker = DocumentChunker(self._settings)
        self.builder = VectorStoreBuilder(self._settings)

    def _build_ingestion_chain(self, vectorstore_dir: Path | None = None):
        """Build an LCEL chain that chunks documents then builds the vector store.

        Chain structure:
            RunnableLambda(chunk)
            | RunnableLambda(build_and_save → chunk_count)

        Accepts ``List[Document]`` and returns ``int`` (number of chunks created).

        Args:
            vectorstore_dir: Override destination for the FAISS index.
        """
        def _build_and_count(chunks: List[Document]) -> int:
            self.builder.build_and_save(chunks, vectorstore_dir=vectorstore_dir)
            return len(chunks)

        return (
            RunnableLambda(self.chunker.chunk)
            | RunnableLambda(_build_and_count)
        )

    def run(
        self,
        data_dir: Path | None = None,
        files: List[Path] | None = None,
        vectorstore_dir: Path | None = None,
    ) -> IngestionResult:
        """Execute the full ingestion pipeline end-to-end.

        Steps:
          1. Load documents — either from an explicit ``files`` list or from
             all supported files under ``data_dir``
          2. Chunk documents and build+persist the FAISS vector store via LCEL chain

        Args:
            data_dir: Directory to scan for documents. Ignored when ``files``
                is provided. Defaults to ``settings.data_dir``.
            files: Explicit list of file paths to ingest. Takes priority over
                ``data_dir`` when provided.
            vectorstore_dir: Override the directory where the FAISS index is
                written. Defaults to ``settings.vectorstore_dir``.

        Returns:
            IngestionResult: Summary with counts and any per-file error messages.
        """
        errors: List[str] = []
        dest = vectorstore_dir or self._settings.vectorstore_dir

        if files is not None:
            source_label = f"{len(files)} uploaded file(s)"
        else:
            source_label = str(data_dir or self._settings.data_dir)

        logger.info("=== Ingestion pipeline started (source: %s) ===", source_label)

        # Step 1: Load
        try:
            if files is not None:
                documents = self.loader.load_files(files)
            else:
                documents = self.loader.load_directory(data_dir)
        except DocumentLoadError as exc:
            logger.error("Ingestion failed at load stage: %s", exc)
            raise

        files_loaded = len(documents)

        # Steps 2 & 3: Chunk → embed → save via LCEL chain
        chain = self._build_ingestion_chain(vectorstore_dir=vectorstore_dir)
        chunks_created: int = chain.invoke(documents)

        result = IngestionResult(
            files_loaded=files_loaded,
            chunks_created=chunks_created,
            vectorstore_path=str(dest),
            errors=errors,
        )
        logger.info(
            "=== Ingestion pipeline complete: %d doc(s), %d chunk(s), saved to '%s' ===",
            files_loaded,
            chunks_created,
            result.vectorstore_path,
        )
        return result
