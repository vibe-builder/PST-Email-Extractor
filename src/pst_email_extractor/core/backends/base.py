"""
Backend protocol for PST providers.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from pst_email_extractor.core.attachment_processor import AttachmentContentOptions
from pst_email_extractor.core.models import AttachmentInfo, FolderInfo, MessageHandle


class PstBackend(Protocol):
    """Protocol describing minimal PST backend behaviour."""

    def open(self, path: Path) -> None:
        ...

    def close(self) -> None:
        ...

    def is_available(self) -> bool:
        ...

    def iter_messages(
        self,
        *,
        deduplicate: bool = False,
        attachments_dir: Path | None = None,
        progress_callback=None,
        attachment_content_options: AttachmentContentOptions | None = None,
    ) -> Iterable[dict]:
        ...

    def analyze_addresses(
        self,
        *,
        deduplicate: bool = False,
        progress_callback=None,
    ) -> dict:
        ...

    # Folder tree and scoped iteration
    def list_folders(self) -> list[FolderInfo]:
        ...

    def iter_folder_messages(
        self,
        folder_id: str,
        *,
        start: int = 0,
        limit: int = 100,
        progress_callback=None,
        attachment_content_options: AttachmentContentOptions | None = None,
        attachments_dir: Path | None = None,
    ) -> Iterable[tuple[dict, MessageHandle]]:
        ...

    # Attachment metadata and content
    def list_attachments(self, handle: MessageHandle) -> list[AttachmentInfo]:
        ...

    def read_attachment_bytes(self, handle: MessageHandle, attachment_index: int) -> bytes:
        ...
