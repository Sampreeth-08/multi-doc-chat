import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from multi_doc_chat.config.settings import get_settings
from multi_doc_chat.logger.logger import get_logger
from multi_doc_chat.utils.file_utils import ensure_dir

logger = get_logger(__name__)


@dataclass
class Session:
    """Represents one run of the application, scoped by a unique ID.

    Each session gets its own subdirectory under sessions_dir:
        sessions/{id}/uploads/     — temporary home for user-provided files
        sessions/{id}/vectorstore/ — FAISS index built from those files
    """

    id: str
    created_at: datetime
    session_dir: Path
    upload_dir: Path
    vectorstore_dir: Path

    def cleanup_uploads(self) -> None:
        """Delete the upload directory after ingestion is complete."""
        if self.upload_dir.exists():
            shutil.rmtree(self.upload_dir)
            logger.debug("Cleaned up upload dir: %s", self.upload_dir)

    def cleanup(self) -> None:
        """Delete the entire session directory (uploads + vector store)."""
        if self.session_dir.exists():
            shutil.rmtree(self.session_dir)
            logger.debug("Cleaned up session dir: %s", self.session_dir)


def create_session() -> Session:
    """Generate a new session with a UUID and create its directory layout.

    Returns:
        Session: Fully initialised session with upload_dir and vectorstore_dir
        already created on disk.
    """
    settings = get_settings()
    session_id = str(uuid.uuid4())
    session_dir = settings.sessions_dir / session_id
    upload_dir = session_dir / "uploads"
    vectorstore_dir = session_dir / "vectorstore"

    ensure_dir(upload_dir)
    ensure_dir(vectorstore_dir)

    session = Session(
        id=session_id,
        created_at=datetime.now(),
        session_dir=session_dir,
        upload_dir=upload_dir,
        vectorstore_dir=vectorstore_dir,
    )
    logger.info("Session created: %s", session_id)
    return session
