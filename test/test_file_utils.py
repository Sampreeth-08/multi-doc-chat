from pathlib import Path

import pytest

from multi_doc_chat.utils.file_utils import (
    SUPPORTED_EXTENSIONS,
    ensure_dir,
    get_extension,
    iter_supported_files,
)


class TestGetExtension:
    def test_returns_lowercase_extension(self):
        assert get_extension(Path("document.PDF")) == ".pdf"
        assert get_extension(Path("report.DOCX")) == ".docx"
        assert get_extension(Path("notes.TXT")) == ".txt"

    def test_already_lowercase_unchanged(self):
        assert get_extension(Path("file.txt")) == ".txt"

    def test_no_extension(self):
        assert get_extension(Path("README")) == ""


class TestIterSupportedFiles:
    def test_yields_txt_file(self, tmp_path: Path):
        (tmp_path / "doc.txt").write_text("hello")
        result = list(iter_supported_files(tmp_path))
        assert len(result) == 1
        assert result[0].name == "doc.txt"

    def test_yields_pdf_docx_doc(self, tmp_path: Path):
        (tmp_path / "a.pdf").write_bytes(b"%PDF")
        (tmp_path / "b.docx").write_bytes(b"PK")
        (tmp_path / "c.doc").write_bytes(b"\xd0\xcf")
        result = list(iter_supported_files(tmp_path))
        names = {p.name for p in result}
        assert names == {"a.pdf", "b.docx", "c.doc"}

    def test_skips_unsupported_extensions(self, tmp_path: Path):
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        (tmp_path / "valid.txt").write_text("text")
        result = list(iter_supported_files(tmp_path))
        assert len(result) == 1
        assert result[0].name == "valid.txt"

    def test_recursive_traversal(self, tmp_path: Path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")
        (tmp_path / "top.txt").write_text("top content")
        result = list(iter_supported_files(tmp_path))
        names = {p.name for p in result}
        assert "nested.txt" in names
        assert "top.txt" in names

    def test_raises_file_not_found_for_missing_dir(self, tmp_path: Path):
        missing = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            list(iter_supported_files(missing))

    def test_empty_directory_yields_nothing(self, tmp_path: Path):
        result = list(iter_supported_files(tmp_path))
        assert result == []


class TestEnsureDir:
    def test_creates_directory(self, tmp_path: Path):
        new_dir = tmp_path / "a" / "b" / "c"
        assert not new_dir.exists()
        ensure_dir(new_dir)
        assert new_dir.is_dir()

    def test_is_idempotent(self, tmp_path: Path):
        existing = tmp_path / "exists"
        existing.mkdir()
        ensure_dir(existing)  # should not raise
        assert existing.is_dir()
