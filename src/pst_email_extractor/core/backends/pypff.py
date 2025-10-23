"""
libpff-backed PST backend.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

from .base import PstBackend
from ..models import FolderInfo, MessageHandle, AttachmentInfo
from ..attachment_processor import AttachmentContentExtractor, AttachmentContentOptions
from ..security import MAX_ATTACHMENT_SIZE, CHUNK_SIZE

logger = logging.getLogger(__name__)


class DependencyError(RuntimeError):
    """Raised when an optional backend dependency is unavailable."""


class PypffBackend(PstBackend):
    """Adaptor around the existing pst_email_extractor.pst_parser module."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._parser: ModuleType | None = None

    def _import_parser(self) -> ModuleType:
        if self._parser is not None:
            return self._parser
        try:
            parser = importlib.import_module("pst_email_extractor.pst_parser")
        except SystemExit as exc:  # pragma: no cover - dependency missing
            raise DependencyError("pypff is not available.") from exc
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise DependencyError("pypff is not available.") from exc
        self._parser = parser
        return parser

    def open(self, path: Path) -> None:
        self._path = path.expanduser().resolve()

    def close(self) -> None:
        self._path = None

    def is_available(self) -> bool:
        try:
            self._import_parser()
        except DependencyError:
            return False
        return True

    def iter_messages(
        self,
        *,
        deduplicate: bool = False,
        attachments_dir: Path | None = None,
        progress_callback=None,
        attachment_content_options: AttachmentContentOptions | None = None,
        parallel: bool = True,
    ) -> Iterable[dict]:
        if self._path is None:
            raise RuntimeError("PST backend not opened before iterating messages")
        parser = self._import_parser()
        
        # If attachment_content_options is provided, use iter_folder_messages on all folders
        if attachment_content_options:
            folders = self.list_folders()
            folders_with_msgs = [f for f in folders if f.total_count > 0]
            
            if parallel and len(folders_with_msgs) > 1:
                # Parallel processing for multiple folders using ThreadPoolExecutor
                # Threads work well here since pypff I/O is often blocking
                max_workers = min(os.cpu_count() or 2, len(folders_with_msgs), 4)
                logger.info(f"Processing {len(folders_with_msgs)} folders with {max_workers} threads")
                
                yield from self._process_folders_parallel(
                    folders_with_msgs,
                    attachment_content_options,
                    progress_callback,
                    max_workers
                )
            else:
                # Sequential processing for single folder or when parallel disabled
                yield from self._process_folders_sequential(
                    folders_with_msgs,
                    attachment_content_options,
                    progress_callback
                )
        else:
            # Standard iteration without attachment content extraction
            yield from parser.iter_emails(
                str(self._path),
                progress_callback=progress_callback,
                deduplicate=deduplicate,
                attachments_dir=str(attachments_dir) if attachments_dir else None,
            )
    
    def _process_folders_sequential(
        self,
        folders: list[FolderInfo],
        options: AttachmentContentOptions,
        progress_callback,
    ) -> Iterable[dict]:
        """Process folders sequentially (original behavior)."""
        total_processed = 0
        for folder_info in folders:
            try:
                for email_dict, handle in self.iter_folder_messages(
                    folder_info.path,
                    start=0,
                    limit=folder_info.total_count,
                    attachment_content_options=options,
                    progress_callback=progress_callback
                ):
                    total_processed += 1
                    if progress_callback and total_processed % 10 == 0:
                        try:
                            progress_callback(total_processed, 0, "Processing attachments")
                        except Exception:
                            try:
                                progress_callback(total_processed)
                            except Exception:
                                pass
                    yield email_dict
            except Exception as e:
                logger.warning(f"Error processing folder {folder_info.path}: {e}")
                continue
    
    def _process_folders_parallel(
        self,
        folders: list[FolderInfo],
        options: AttachmentContentOptions,
        progress_callback,
        max_workers: int,
    ) -> Iterable[dict]:
        """Process folders in parallel using ThreadPoolExecutor."""
        total_processed = 0
        
        def process_folder(folder_info: FolderInfo) -> list[dict]:
            """Worker function to process a single folder."""
            emails = []
            try:
                for email_dict, _ in self.iter_folder_messages(
                    folder_info.path,
                    start=0,
                    limit=folder_info.total_count,
                    attachment_content_options=options,
                    progress_callback=None  # Avoid callback contention in threads
                ):
                    emails.append(email_dict)
            except Exception as e:
                logger.warning(f"Error processing folder {folder_info.path}: {e}")
            return emails
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all folders for processing
            future_to_folder = {
                executor.submit(process_folder, folder): folder
                for folder in folders
            }
            
            # Yield results as they complete
            for future in as_completed(future_to_folder):
                folder = future_to_folder[future]
                try:
                    emails = future.result()
                    for email_dict in emails:
                        total_processed += 1
                        if progress_callback and total_processed % 10 == 0:
                            try:
                                progress_callback(total_processed, 0, f"Processed {total_processed} emails")
                            except Exception:
                                pass
                        yield email_dict
                except Exception as e:
                    logger.error(f"Failed to retrieve results for folder {folder.path}: {e}")

    def analyze_addresses(
        self,
        *,
        deduplicate: bool = False,
        progress_callback=None,
    ) -> dict:
        if self._path is None:
            raise RuntimeError("PST backend not opened before analyzing addresses")

        # Use optimized analysis with Polars when possible
        from ..analysis import analyze_addresses

        parser = self._import_parser()
        email_iter = parser.iter_emails(
            str(self._path),
            progress_callback=progress_callback,
            deduplicate=deduplicate,
        )

        return analyze_addresses(email_iter, progress_callback=progress_callback)

    # --- Folder tree and scoped iteration ---
    def _open_pst(self):
        if self._path is None:
            raise RuntimeError("PST backend not opened")
        parser = self._import_parser()
        try:
            pypff = parser.pypff  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            raise DependencyError("pypff is not available.")
        if not parser.is_pypff_available():
            raise DependencyError("pypff is not available.")
        pst_file = pypff.file()
        pst_file.open(str(self._path))
        return pst_file

    def _folder_path(self, folder) -> str:
        # Build a stable path like /Root/Inbox/...
        segments: list[str] = []
        cur = folder
        while cur is not None:
            try:
                name = getattr(cur, "name", None) or ""
            except Exception:
                name = ""
            segments.append(str(name) if name else "")
            try:
                cur = getattr(cur, "parent", None)
            except Exception:
                break
        # Last element is the root which may be empty; normalise
        segments = list(reversed(segments))
        # Ensure first segment is 'Root'
        if not segments or segments[0] == "":
            segments = ["Root"] + segments[1:]
        return "/" + "/".join(seg if seg else "(Unnamed)" for seg in segments)

    def list_folders(self) -> list[FolderInfo]:
        pst = self._open_pst()
        try:
            root = pst.root_folder
            if root is None:
                return []

            result: list[FolderInfo] = []
            # Stack now holds (folder_object, path_string) tuples
            root_name = getattr(root, "name", None) or "Root"
            stack = [(root, f"/{root_name}")]
            
            while stack:
                f, current_path = stack.pop()
                try:
                    name = getattr(f, "name", None) or current_path.rsplit("/", 1)[-1]
                except Exception:
                    name = current_path.rsplit("/", 1)[-1]
                
                # message count
                total = 0
                try:
                    total = int(getattr(f, "number_of_sub_messages", 0))
                except Exception:
                    try:
                        total = sum(1 for _ in f.sub_messages)
                    except Exception:
                        total = 0
                
                result.append(FolderInfo(
                    id=current_path, 
                    name=str(name), 
                    path=current_path, 
                    total_count=total, 
                    unread_count=0
                ))
                
                # push children with their full paths
                try:
                    for sub in f.sub_folders:
                        sub_name = getattr(sub, "name", None) or "(Unnamed)"
                        sub_path = f"{current_path}/{sub_name}"
                        stack.append((sub, sub_path))
                except Exception as exc:
                    logger.debug("Failed to enumerate subfolders for %s: %s", current_path, exc)
            
            return result
        finally:
            try:
                pst.close()
            except Exception as exc:
                logger.debug("Failed to close PST file: %s", exc)

    def _resolve_folder_by_path(self, pst_file, folder_path: str):
        # folder_path like /Root/Inbox/Sub
        if not folder_path or not folder_path.startswith("/"):
            raise ValueError("folder_id must be an absolute path like /Root/Inbox")
        parts = [p for p in folder_path.split("/") if p]
        if not parts:
            raise ValueError("Invalid folder path")
        cur = pst_file.root_folder
        if cur is None:
            return None
        # If the first part is 'Root', skip it; otherwise traverse from the first part
        start_index = 1 if parts and parts[0] == "Root" else 0
        for name in parts[start_index:]:
            found = None
            try:
                for sub in cur.sub_folders:
                    try:
                        sub_name = getattr(sub, "name", None) or ""
                    except Exception:
                        sub_name = ""
                    if (sub_name or "(Unnamed)") == name:
                        found = sub
                        break
            except Exception:
                found = None
            if found is None:
                return None
            cur = found
        return cur

    def iter_folder_messages(
        self,
        folder_id: str,
        *,
        start: int = 0,
        limit: int = 100,
        progress_callback=None,
        attachment_content_options: AttachmentContentOptions | None = None,
    ) -> Iterable[tuple[dict, MessageHandle]]:
        if limit <= 0:
            return iter(())
        parser = self._import_parser()
        pst = self._open_pst()
        try:
            folder = self._resolve_folder_by_path(pst, folder_id)
            if folder is None:
                return iter(())
            yielded = 0
            idx = -1
            messages_iter = getattr(folder, "sub_messages", None)
            if messages_iter is not None:
                for m in messages_iter:
                    idx += 1
                    if idx < start:
                        continue
                    if yielded >= limit:
                        break
                    try:
                        email = parser._parse_message(m, idx, None)  # type: ignore[attr-defined]
                    except Exception:
                        email = None
                    if email:
                        handle = MessageHandle(folder_id=folder_id, message_index=idx)

                        # Process attachment content if requested
                        if attachment_content_options:
                            email = self._process_attachment_content(email, handle, attachment_content_options, m, parser)

                        yielded += 1
                        yield email, handle
            else:
                # Fallback: index-based access
                try:
                    total = int(getattr(folder, "number_of_sub_messages", 0))
                except Exception:
                    total = 0
                end = min(total, start + limit)
                for i in range(start, end):
                    try:
                        m = folder.get_sub_message(i)
                    except Exception:
                        continue
                    try:
                        email = parser._parse_message(m, i, None)  # type: ignore[attr-defined]
                    except Exception:
                        email = None
                    if email:
                        handle = MessageHandle(folder_id=folder_id, message_index=i)
                        if attachment_content_options:
                            email = self._process_attachment_content(email, handle, attachment_content_options, m, parser)
                        yielded += 1
                        yield email, handle
            return
        finally:
            try:
                pst.close()
            except Exception as exc:
                logger.debug("Failed to close PST file: %s", exc)

    def _process_attachment_content(self, email: dict, handle: MessageHandle,
                                   options: AttachmentContentOptions, message_obj: Any, parser: ModuleType) -> dict:
        """Process attachment content for a message and enrich the email record."""
        try:
            # Get attachment count
            attachment_count = int(getattr(message_obj, "number_of_attachments", 0))
            if attachment_count <= 0:
                return email

            # Initialize content extractor
            extractor = AttachmentContentExtractor(options)

            # Process each attachment
            attachment_metadata = []
            all_extracted_text = []

            for i in range(attachment_count):
                try:
                    # Read attachment data
                    attachment_data = self.read_attachment_bytes(handle, i)
                    if not attachment_data:
                        continue

                    # Get attachment filename from the message record if available
                    filename = f"attachment_{i}"
                    attachment_paths = email.get("Attachment_Paths", [])
                    if isinstance(attachment_paths, list) and i < len(attachment_paths):
                        # Extract filename from path
                        path = attachment_paths[i]
                        if isinstance(path, str):
                            filename = path.split("/")[-1] or f"attachment_{i}"

                    # Process attachment content
                    metadata = extractor.process_attachment(i, filename, len(attachment_data), attachment_data)
                    attachment_metadata.append({
                        "index": metadata.index,
                        "filename": metadata.filename,
                        "size": metadata.size,
                        "mime_type": metadata.mime_type,
                        "content_type": metadata.content_type,
                        "extracted_text": metadata.extracted_text,
                        "text_extraction_method": metadata.text_extraction_method,
                        "page_count": metadata.page_count,
                        "embedded_message_data": metadata.embedded_message_data,
                        "error_message": metadata.error_message
                    })

                    # Collect extracted text
                    if metadata.extracted_text:
                        all_extracted_text.append(metadata.extracted_text)

                except Exception as e:
                    logger.debug(f"Failed to process attachment {i}: {e}")
                    attachment_metadata.append({
                        "index": i,
                        "filename": f"attachment_{i}",
                        "size": 0,
                        "mime_type": "application/octet-stream",
                        "content_type": "binary",
                        "error_message": str(e)
                    })

            # Enrich email record
            enriched_email = dict(email)
            enriched_email["Attachment_Metadata"] = attachment_metadata
            enriched_email["Attachment_Content"] = "\n\n".join(all_extracted_text)
            enriched_email["Attachment_Types"] = list(set(meta["mime_type"] for meta in attachment_metadata if meta.get("mime_type")))

            return enriched_email

        except Exception as e:
            logger.debug(f"Failed to process attachment content for message: {e}")
            return email

    # --- Attachments ---
    def list_attachments(self, handle: MessageHandle) -> list[AttachmentInfo]:
        pst = self._open_pst()
        try:
            folder = self._resolve_folder_by_path(pst, handle.folder_id)
            if folder is None:
                return []
            # Locate message by index
            try:
                # Prefer direct indexing if available
                msg_iter = getattr(folder, "sub_messages", [])
                msg = None
                for i, m in enumerate(msg_iter):
                    if i == handle.message_index:
                        msg = m
                        break
            except Exception:
                msg = None
            if msg is None:
                return []
            try:
                total = int(getattr(msg, "number_of_attachments", 0))
            except Exception:
                total = 0
            items: list[AttachmentInfo] = []
            for i in range(max(0, total)):
                try:
                    att = msg.get_attachment(i)
                except Exception:
                    att = None
                if att is None:
                    continue
                try:
                    name = (
                        getattr(att, "long_filename", "")
                        or getattr(att, "file_name", "")
                        or getattr(att, "name", "")
                    )
                except Exception:
                    name = ""
                try:
                    size = getattr(att, "size", None)
                except Exception:
                    size = None
                items.append(AttachmentInfo(index=i, filename=str(name or f"attachment_{i}"), size=size if isinstance(size, int) else None))
            return items
        finally:
            try:
                pst.close()
            except Exception as exc:
                logger.debug("Failed to close PST file: %s", exc)

    def read_attachment_bytes(self, handle: MessageHandle, attachment_index: int) -> bytes:
        pst = self._open_pst()
        try:
            folder = self._resolve_folder_by_path(pst, handle.folder_id)
            if folder is None:
                return b""
            # message
            msg = None
            try:
                for i, m in enumerate(folder.sub_messages):
                    if i == handle.message_index:
                        msg = m
                        break
            except Exception:
                msg = None
            if msg is None:
                return b""
            # attachment
            try:
                att = msg.get_attachment(int(attachment_index))
            except Exception:
                att = None
            if att is None:
                return b""

            # Security-aligned streaming (shared constants)

            try:
                candidate_size = getattr(att, "size", None) or getattr(att, "get_size", lambda: 0)()
            except Exception:
                candidate_size = None

            data = b""
            try:
                if hasattr(att, "read_buffer") and isinstance(candidate_size, (int, float)) and candidate_size > 0:
                    if candidate_size > MAX_ATTACHMENT_SIZE or candidate_size < 0:
                        return b""
                    bytes_read = 0
                    out = bytearray()
                    while bytes_read < candidate_size:
                        remaining = int(min(CHUNK_SIZE, candidate_size - bytes_read))
                        chunk = att.read_buffer(remaining)
                        if not chunk:
                            break
                        out.extend(chunk)
                        bytes_read += len(chunk)
                    data = bytes(out)
                else:
                    data = getattr(att, "data", b"")
            except Exception:
                try:
                    data = getattr(att, "data", b"")
                except Exception:
                    data = b""

            if isinstance(data, (bytes, bytearray)) and len(data) > MAX_ATTACHMENT_SIZE:
                return b""
            return bytes(data)
        finally:
            try:
                pst.close()
            except Exception:
                pass


def is_available() -> bool:
    return PypffBackend().is_available()
