from pathlib import Path
from typing import Generator

SUPPORTED_EXTENSIONS: set[str] = {".pdf", ".txt", ".doc", ".docx"}


def iter_supported_files(data_dir: Path) -> Generator[Path, None, None]:
    """Yield all supported document files under data_dir recursively.

    Only files whose extension (case-insensitive) is in SUPPORTED_EXTENSIONS
    are yielded. Unsupported extensions are silently skipped.

    Args:
        data_dir: Directory to search recursively.

    Raises:
        FileNotFoundError: If data_dir does not exist.

    Yields:
        Path: Absolute path to each supported file.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file_path in sorted(data_dir.rglob("*")):
        if file_path.is_file() and get_extension(file_path) in SUPPORTED_EXTENSIONS:
            yield file_path


def get_extension(file_path: Path) -> str:
    """Return the lowercase file extension including the leading dot.

    Args:
        file_path: Path to the file.

    Returns:
        str: e.g. '.pdf', '.txt', '.docx'
    """
    return file_path.suffix.lower()


def ensure_dir(path: Path) -> None:
    """Create directory (and any parents) if it does not already exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)
