import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from multi_doc_chat.exception.exceptions import ConfigurationError

load_dotenv()

# Resolve project root as the parent of this file's grandparent (config/ -> multi_doc_chat/ -> project/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Settings:
    """Frozen configuration object — single source of truth for all runtime settings."""

    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    data_dir: Path
    vectorstore_dir: Path


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Load and return a singleton Settings instance from environment variables.

    Raises:
        ConfigurationError: If OPENAI_API_KEY is not set.
    """
    global _settings_instance

    if _settings_instance is not None:
        return _settings_instance

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    _settings_instance = Settings(
        openai_api_key=api_key,
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip(),
        openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        retriever_k=int(os.getenv("RETRIEVER_K", "4")),
        data_dir=_PROJECT_ROOT / os.getenv("DATA_DIR", "data"),
        vectorstore_dir=_PROJECT_ROOT / os.getenv("VECTORSTORE_DIR", "vectorstore"),
    )

    return _settings_instance
