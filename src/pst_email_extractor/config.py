"""
Shared configuration helpers and defaults.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "PST Email Extractor"
APP_DIR = Path.home() / APP_NAME
LOG_DIR = APP_DIR / "logs"
DEFAULT_LOG_FILE = LOG_DIR / "pst_parser.log"


def ensure_app_directories() -> None:
    """Create common application directories when missing."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
