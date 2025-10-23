"""
Base classes and protocols for PST email exporters.

This module defines the core interfaces and utilities used by all exporters
to ensure consistent behavior and type safety across different export formats.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from pst_email_extractor.ai.pipeline import create_text_pipeline


class Exporter(Protocol):
    """Protocol describing minimal exporter behaviour."""

    path: Path

    def __enter__(self) -> Exporter:
        ...

    def __exit__(self, exc_type, exc, tb) -> None:
        ...

    def close(self) -> None:
        ...


class EmailExporter(Exporter, Protocol):
    """Protocol for exporters that write email records."""

    def write(self, email_row: Mapping[str, Any]) -> None:
        ...


def _normalise_output_path(path_like: str | Path) -> Path:
    """Normalize and ensure output path exists."""
    path = Path(path_like).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Common email field definitions used across exporters
EMAIL_FIELDS = [
    "Email_ID",
    "Date_Received",
    "Date_Sent",
    "From",
    "Sender_Email",
    "To",
    "CC",
    "BCC",
    "Reply_To",
    "Subject",
    "Body",
    "Importance",
    "Attachment_Count",
    "Attachment_Paths",
    "Attachment_Metadata",
    "Attachment_Content",
    "Attachment_Types",
    "Message_ID",
    "Content_Type",
    "Email_Client",
    "Return_Path",
    "Client_Info",
    "Thread_ID",
    "Parent_ID",
    "References",
    "Received_Hops",
    "Full_Headers",
]

ADDRESS_FIELDS = ["Address", "Count", "Roles", "Domains", "Names"]
HOST_FIELDS = ["Host", "Count"]


class AIPipelineMixin:
    """Mixin class that adds AI text processing capabilities to exporters."""

    def __init__(self, ai_sanitize: bool = False, ai_polish: bool = False,
                 ai_language: str = "en-US", ai_neural_model_dir: str | None = None):
        self.ai_pipeline = None
        if ai_sanitize or ai_polish or ai_neural_model_dir:
            try:
                self.ai_pipeline = create_text_pipeline(
                    language=ai_language,
                    enable_sanitize=ai_sanitize,
                    enable_spell=ai_polish,
                    enable_grammar=ai_polish,
                    enable_neural=bool(ai_neural_model_dir),
                    neural_model_dir=ai_neural_model_dir
                )
            except Exception:
                # AI pipeline creation failed - disable AI processing gracefully
                self.ai_pipeline = None
        self.ai_sanitize = ai_sanitize
        self.ai_polish = ai_polish

    def process_text_field(self, text: str) -> str:
        """Apply AI processing to text fields if enabled."""
        if self.ai_pipeline and text:
            try:
                return self.ai_pipeline.process(text, sanitize=self.ai_sanitize, polish=self.ai_polish)
            except Exception:
                # AI processing failed - return original text
                pass
        return text

    def process_email_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Process text fields in an email record for AI sanitization/polishing."""
        processed = dict(record)  # Copy the record

        # Fields that should be processed for PII and grammar
        text_fields = ["Subject", "Body", "From", "To", "CC", "BCC"]

        for field in text_fields:
            if field in processed and isinstance(processed[field], str):
                processed[field] = self.process_text_field(processed[field])

        return processed
