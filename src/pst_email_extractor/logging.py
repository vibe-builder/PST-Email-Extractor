"""
Logging helpers shared by CLI and GUI surfaces.
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import config, pst_parser


def configure_logging(log_file: Path | None = None, level: int | None = None) -> Path:
    """Configure rotating file logging and return the active log path."""
    config.ensure_app_directories()
    log_path = log_file or config.DEFAULT_LOG_FILE
    return pst_parser.configure_logging(log_path, level)


def get_logger(name: str = "pst_email_extractor") -> logging.Logger:
    """Return a namespaced logger."""
    return logging.getLogger(name)
