import pytest

from multi_doc_chat.exception.exceptions import (
    ConfigurationError,
    DocumentLoadError,
    MultiDocChatError,
    QueryError,
    UnsupportedFileTypeError,
    VectorStoreError,
    VectorStoreNotFoundError,
)


class TestExceptionHierarchy:
    def test_configuration_error_is_multi_doc_chat_error(self):
        assert issubclass(ConfigurationError, MultiDocChatError)

    def test_document_load_error_is_multi_doc_chat_error(self):
        assert issubclass(DocumentLoadError, MultiDocChatError)

    def test_unsupported_file_type_error_is_document_load_error(self):
        assert issubclass(UnsupportedFileTypeError, DocumentLoadError)

    def test_vector_store_error_is_multi_doc_chat_error(self):
        assert issubclass(VectorStoreError, MultiDocChatError)

    def test_vector_store_not_found_error_is_vector_store_error(self):
        assert issubclass(VectorStoreNotFoundError, VectorStoreError)

    def test_query_error_is_multi_doc_chat_error(self):
        assert issubclass(QueryError, MultiDocChatError)


class TestDocumentLoadError:
    def test_str_contains_file_path_and_reason(self):
        exc = DocumentLoadError(file_path="doc.pdf", reason="corrupt file")
        msg = str(exc)
        assert "doc.pdf" in msg
        assert "corrupt file" in msg

    def test_attributes_are_set(self):
        exc = DocumentLoadError(file_path="/data/foo.txt", reason="permission denied")
        assert exc.file_path == "/data/foo.txt"
        assert exc.reason == "permission denied"

    def test_can_be_raised_and_caught_as_multi_doc_chat_error(self):
        with pytest.raises(MultiDocChatError):
            raise DocumentLoadError("x.pdf", "bad")


class TestUnsupportedFileTypeError:
    def test_str_contains_extension_info(self):
        exc = UnsupportedFileTypeError(file_path="data.csv", reason="Extension '.csv' not supported.")
        assert "data.csv" in str(exc)

    def test_can_be_caught_as_document_load_error(self):
        with pytest.raises(DocumentLoadError):
            raise UnsupportedFileTypeError("data.csv", "not supported")
