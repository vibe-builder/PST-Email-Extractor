"""
High level orchestration services.
"""

from __future__ import annotations

from .extraction import perform_extraction
from .models import ExtractionConfig, ExtractionResult, ProgressCallback


def run_extraction(
    config: ExtractionConfig,
    progress_callback: ProgressCallback | None = None,
) -> ExtractionResult:
    """Convenience wrapper that delegates to the core extraction routine."""
    return perform_extraction(config, progress_callback=progress_callback)
