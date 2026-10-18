"""Project-wide logging setup."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
DEFAULT_FORMAT = "%(asctime)s %(levelname)s: %(message)s"
DEFAULT_DATEFMT = "%H:%M:%S"


def setup_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure the root logger once."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=DEFAULT_FORMAT,
        datefmt=DEFAULT_DATEFMT,
        handlers=handlers,
        force=True,
    )
    _CONFIGURED = True


def get_logger(name: str = "dfdetect") -> logging.Logger:
    """Return a configured logger."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
