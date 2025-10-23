"""
High-level package exports for PST Email Extractor.
"""

from __future__ import annotations

from importlib import metadata
from typing import Final

try:
    __version__: Final[str] = metadata.version("pst-email-extractor")
except metadata.PackageNotFoundError:  # pragma: no cover - local execution
    __version__ = "0.0.0"

# Convenience re-exports
from . import exporters, pst_parser  # noqa: E402
from .core.extraction import perform_extraction  # noqa: E402
from .core.models import ExtractionConfig, ExtractionResult  # noqa: E402
from .id_generator import generate as generate_identifier  # noqa: E402

__all__ = [
    "__version__",
    "ExtractionConfig",
    "ExtractionResult",
    "exporters",
    "pst_parser",
    "generate_identifier",
    "perform_extraction",
]
