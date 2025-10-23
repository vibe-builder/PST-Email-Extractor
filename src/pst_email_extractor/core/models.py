"""
Typed models used across extraction pipelines.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .attachment_processor import AttachmentContentOptions

Mode = Literal["extract", "addresses"]


@dataclass(slots=True)
class ExtractionConfig:
    pst_path: Path
    output_dir: Path
    formats: Sequence[str]
    mode: Mode = "extract"
    deduplicate: bool = False
    extract_attachments: bool = False
    attachments_dir: Path | None = None
    log_file: Path | None = None
    html_index: bool = False
    # AI pipeline options
    ai_sanitize: bool = False
    ai_polish: bool = False
    ai_language: str = "en-US"
    ai_neural_model_dir: Path | None = None
    # Attachment content extraction options
    extract_attachment_content: bool = False
    attachment_content_options: AttachmentContentOptions | None = None
    # Performance options
    compress: bool = False  # Gzip compression for JSON/CSV exports


@dataclass(slots=True)
class ProgressUpdate:
    current: int
    total: int
    message: str


ProgressCallback = Callable[[ProgressUpdate], None]


@dataclass(slots=True)
class ExtractionResult:
    mode: Mode
    exported_paths: list[Path] = field(default_factory=list)
    log_path: Path | None = None
    unique_run_id: str | None = None
    email_count: int = 0
    attachments_saved: int = 0
    attachments_dir: Path | None = None
    html_index_path: Path | None = None
    address_count: int = 0
    host_count: int = 0


# Folder-aware GUI and attachment preview support
@dataclass(slots=True)
class FolderInfo:
    id: str  # stable identifier, e.g. "/Root/Inbox"
    name: str  # display name
    path: str  # same as id for this backend
    total_count: int = 0
    unread_count: int = 0


@dataclass(slots=True)
class MessageHandle:
    folder_id: str
    message_index: int  # 0-based index within folder


@dataclass(slots=True)
class AttachmentInfo:
    index: int
    filename: str
    size: int | None = None