class MultiDocChatError(Exception):
    """Base exception for all multi_doc_chat errors."""


class ConfigurationError(MultiDocChatError):
    """Raised when required environment variables are missing or invalid."""


class DocumentLoadError(MultiDocChatError):
    """Raised when a document file cannot be loaded or parsed."""

    def __init__(self, file_path: str, reason: str) -> None:
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to load '{file_path}': {reason}")


class UnsupportedFileTypeError(DocumentLoadError):
    """Raised when an unsupported file extension is encountered."""


class VectorStoreError(MultiDocChatError):
    """Raised when FAISS index creation, save, or load fails."""


class VectorStoreNotFoundError(VectorStoreError):
    """Raised when load_local is called but the vectorstore directory is missing or incomplete."""


class QueryError(MultiDocChatError):
    """Raised when the RAG query pipeline fails."""
