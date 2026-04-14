import logging
import logging.handlers
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "multi_doc_chat.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured_loggers: set[str] = set()


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return a configured logger instance.

    Sets up handlers only once per named logger (idempotent):
    - RotatingFileHandler: logs/multi_doc_chat.log (5 MB, 3 backups, DEBUG level)
    - StreamHandler: console at WARNING level to keep CLI output clean

    Args:
        name: Logger name, typically __name__ of the calling module.
        level: Base logging level for the logger. Defaults to DEBUG.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    if name in _configured_loggers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # File handler — full DEBUG output, rotating at 5 MB with 3 backups
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler — WARNING only, avoids noisy output during queries
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    _configured_loggers.add(name)
    return logger
