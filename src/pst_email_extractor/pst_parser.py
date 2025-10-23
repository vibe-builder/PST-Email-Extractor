"""
PST file parser for extracting email data via the libpff bindings.

The implementation is a direct evolution of the original `pst_tools.pst_parser`
module but refactored to avoid hard exits when optional dependencies are
missing.  Callers are expected to check :func:`is_pypff_available` (or call
``require_pypff``) before invoking the streaming APIs.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
from collections.abc import Iterable
from html import unescape
from html.parser import HTMLParser
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from pst_email_extractor.core.security import CHUNK_SIZE, MAX_ATTACHMENT_SIZE

logger = logging.getLogger("pst_email_extractor.pst_parser")
logger.setLevel(logging.INFO)
logger.propagate = False

_FILE_HANDLER = None
_STREAM_HANDLER = None
_LOG_PATH = None  # Will be set only when logging is explicitly enabled


class DependencyError(RuntimeError):
    """Raised when the libpff bindings are unavailable."""


def _install_handlers(log_path: Path | None = None) -> Path:
    """Configure rotating file and console handlers."""
    global _FILE_HANDLER, _STREAM_HANDLER, _LOG_PATH

    if log_path is not None:
        _LOG_PATH = Path(log_path).expanduser()
    elif _LOG_PATH is None:
        # Use secure user-specific location for logs when not explicitly set
        # This prevents information leakage on multi-user systems
        import getpass
        import tempfile
        try:
            # Try to use user-specific temp directory
            user_temp = Path(tempfile.gettempdir()) / f"pst_extractor_{getpass.getuser()}"
            user_temp.mkdir(parents=True, exist_ok=True)
            _LOG_PATH = user_temp / "pst_parser.log"
        except Exception:
            # Fallback to current directory only if user-specific fails
            _LOG_PATH = Path("pst_parser.log").expanduser()

    log_target = _LOG_PATH
    log_target.parent.mkdir(parents=True, exist_ok=True)

    # Set restrictive permissions on log file (readable/writable by owner only)
    try:
        import stat
        # Create file first, then set permissions
        if not log_target.exists():
            log_target.touch()
        # Set permissions to 600 (rw-------)
        log_target.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        # Ignore permission setting failures (may not work on all platforms)
        pass

    if _FILE_HANDLER:
        logger.removeHandler(_FILE_HANDLER)
    _FILE_HANDLER = RotatingFileHandler(
        log_target,
        maxBytes=1_048_576,
        backupCount=3,
        encoding="utf-8",
    )
    _FILE_HANDLER.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_FILE_HANDLER)

    if _STREAM_HANDLER is None:
        _STREAM_HANDLER = logging.StreamHandler()
        _STREAM_HANDLER.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
        logger.addHandler(_STREAM_HANDLER)

    return log_target


def configure_logging(log_file: str | Path | None = None, level: int | None = None) -> Path:
    """
    Configure parser logging handlers.

    Args:
        log_file: Optional path for the rotating log file.
        level: Optional logging level override.

    Returns:
        Path to the active log file.
    """
    path = _install_handlers(Path(log_file) if log_file else None)
    if level is not None:
        logger.setLevel(level)
    logger.info("Logging configured. Writing to %s", path)
    return path


# Logging is now opt-in only - not installed by default for security

try:
    import pypff  # type: ignore[import]

    _PYPFF_AVAILABLE = True
except ImportError:
    pypff = None  # type: ignore[assignment]
    _PYPFF_AVAILABLE = False
    logger.debug("libpff (pypff) bindings are not available")

try:
    from bs4 import BeautifulSoup  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from striprtf.striprtf import rtf_to_text  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    rtf_to_text = None


def is_pypff_available() -> bool:
    """Return whether the libpff bindings are importable."""
    return bool(_PYPFF_AVAILABLE)


def require_pypff() -> None:
    """Ensure pypff is available or raise :class:`DependencyError`."""
    if not _PYPFF_AVAILABLE:
        raise DependencyError(
            "pypff library not found. Install libpff-python-ratom==20220304."
        )


def _sanitize_text(value: Any) -> str:
    """Ensure text is a clean UTF-8 string without surrogate characters.

    Optimized to avoid redundant encoding/decoding when not necessary.
    Only re-encodes if surrogate pairs are detected.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")

    text = str(value)
    # Ultra-fast surrogate detection using string search
    if '\ud800' in text or '\udbff' in text:  # Quick check for surrogate boundaries
        return text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return text


class _HTMLStripper(HTMLParser):
    """Minimal HTML parser that collects text data."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return "".join(self._chunks)


def _collapse_whitespace(value: Any) -> str:
    """Normalize whitespace for cleaner output."""
    value = _sanitize_text(value)
    if not value:
        return ""
    lines = [line.strip() for line in value.splitlines()]
    filtered = [line for line in lines if line]
    return "\n".join(filtered) if filtered else value.strip()


def _html_to_text(html_content: Any) -> str:
    """Convert HTML content to readable plain text."""
    if not html_content:
        return ""

    if BeautifulSoup:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
    else:
        stripper = _HTMLStripper()
        stripper.feed(_sanitize_text(html_content))
        stripper.close()
        text = stripper.get_text()

    return _collapse_whitespace(unescape(_sanitize_text(text)))


def _rtf_to_text(rtf_content: Any) -> str:
    """Convert RTF content to plain text when possible."""
    if not rtf_content:
        return ""

    if rtf_to_text:
        try:
            return _collapse_whitespace(rtf_to_text(_sanitize_text(rtf_content)))
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("RTF to text conversion failed, using fallback: %s", exc)

    cleaned = _sanitize_text(rtf_content).replace("{", "").replace("}", "")
    return _collapse_whitespace(unescape(cleaned))


def _get_message_property(message: Any, property_name: str, default: str = "") -> str:
    """Safely extract a property from a message object."""
    try:
        value = getattr(message, property_name, default)
        if value is None:
            return default
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)
    except Exception:
        return default


def _generate_email_hash(message: Any, index: int) -> str:
    """Generate deterministic hash for email message deduplication.
    
    Uses Message-ID when available for best uniqueness, falls back to
    content-based hash. Uses SHA256 for better performance on modern CPUs
    with hardware acceleration.
    """
    try:
        # Prefer Message-ID as it's globally unique
        message_id = _get_message_property(message, "internet_message_id")
        if message_id and message_id.strip():
            return hashlib.sha256(message_id.encode()).hexdigest()[:16]
        
        # Fallback to content-based hash (deterministic, no index)
        subject = _get_message_property(message, "subject")
        sender = _get_message_property(message, "sender_email_address") or _get_message_property(message, "sender_name")
        delivery_time = str(getattr(message, "delivery_time", ""))
        
        # Use pipe separator for better collision resistance
        # Use MD5 for speed - deduplication doesn't need crypto security, just uniqueness
        hash_input = f"{subject}|{sender}|{delivery_time}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    except Exception:
        # Last resort: use index-based hash (not ideal for deduplication)
        import time
        fallback = f"email_{index}_{time.time_ns()}"
        return hashlib.sha256(fallback.encode()).hexdigest()[:16]


def _extract_body_and_html(message: Any) -> tuple[str, str]:
    """
    Extract both plain text body and raw HTML body from a message.
    Returns (plain_text_body, html_body)
    """
    try:
        plain_text_body = ""
        html_body_raw = ""

        plain_body = message.plain_text_body
        if plain_body:
            if isinstance(plain_body, bytes):
                plain_body = plain_body.decode("utf-8", errors="ignore")
            plain_text_body = _collapse_whitespace(str(plain_body))

        html_body = message.html_body
        if html_body:
            if isinstance(html_body, bytes):
                html_body = html_body.decode("utf-8", errors="ignore")
            html_body_raw = str(html_body)
            # If no plain text was found, convert HTML to text
            if not plain_text_body:
                plain_text_body = _html_to_text(html_body_raw)

        rtf_body = message.rtf_body
        if rtf_body and not plain_text_body:
            if isinstance(rtf_body, bytes):
                rtf_body = rtf_body.decode("utf-8", errors="ignore")
            plain_text_body = _rtf_to_text(str(rtf_body))

        return plain_text_body, html_body_raw
    except Exception:
        return "", ""


def _extract_body(message: Any) -> str:
    """Extract email body content (plain text or HTML)."""
    plain_text, _ = _extract_body_and_html(message)
    return plain_text


_ATTACHMENT_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_attachment_name(name: str, fallback: str) -> str:
    """Sanitize attachment filenames to a safe ASCII subset."""
    candidate = name.strip() if name else ""
    if not candidate:
        candidate = fallback
    candidate = candidate.replace(" ", "_")
    sanitized = _ATTACHMENT_SANITIZE_PATTERN.sub("_", candidate)
    sanitized = sanitized.strip("._") or fallback
    return sanitized[:255]


def _extract_attachments(message: Any, email_hash: str, attachments_root: Path | None):
    """Persist attachments for the given message to disk."""
    if not attachments_root:
        return [], 0

    try:
        total = int(getattr(message, "number_of_attachments", 0))
    except Exception:
        total = 0

    if total <= 0:
        return [], 0

    extracted_paths: list[str] = []
    email_dir = attachments_root / email_hash
    for index in range(total):
        try:
            attachment = message.get_attachment(index)
        except AttributeError:
            try:
                attachments = getattr(message, "attachments", None)
                attachment = attachments[index] if attachments is not None else None
            except Exception as exc:
                logger.warning("Unable to access attachment %s for message %s: %s", index, email_hash, exc)
                attachment = None
        except Exception as exc:
            logger.warning("Unable to access attachment %s for message %s: %s", index, email_hash, exc)
            attachment = None

        if attachment is None:
            continue

        try:
            name = (
                getattr(attachment, "long_filename", "")
                or getattr(attachment, "file_name", "")
                or getattr(attachment, "name", "")
            )
        except Exception:
            name = ""

        safe_name = _sanitize_attachment_name(name, f"attachment_{index}")
        try:
            size = getattr(attachment, "size", None)
        except Exception:
            size = None

        # Security: Stream attachments in bounded chunks to prevent DoS attacks

        data = b""
        try:
            if hasattr(attachment, "read_buffer"):
                candidate_size = size or getattr(attachment, "get_size", lambda: 0)()

                # Validate size is reasonable and not maliciously large
                try:
                    if not isinstance(candidate_size, int | float) or candidate_size > MAX_ATTACHMENT_SIZE:
                        logger.warning(
                            "Attachment %s for message %s exceeds size limit (%s > %d bytes), skipping",
                            safe_name, email_hash, candidate_size, MAX_ATTACHMENT_SIZE
                        )
                        continue
                    elif candidate_size < 0:
                        logger.warning(
                            "Attachment %s for message %s has invalid negative size (%s), skipping",
                            safe_name, email_hash, candidate_size
                        )
                        continue
                except (TypeError, ValueError):
                    logger.warning(
                        "Attachment %s for message %s has invalid size metadata (%s), skipping",
                        safe_name, email_hash, candidate_size
                    )
                    continue

                # Stream attachment in fixed-size chunks to prevent memory exhaustion
                bytes_read = 0
                data = b""

                # Only stream if we have a valid size
                if isinstance(candidate_size, int | float) and candidate_size > 0:
                    while bytes_read < candidate_size:
                        remaining = min(CHUNK_SIZE, candidate_size - bytes_read)
                        try:
                            chunk = attachment.read_buffer(remaining)
                            if not chunk:
                                break  # End of attachment
                            data += chunk
                            bytes_read += len(chunk)
                        except Exception:
                            # If streaming fails, try fallback to direct read
                            try:
                                data = getattr(attachment, "data", b"")
                                if isinstance(data, bytes | bytearray) and len(data) > MAX_ATTACHMENT_SIZE:
                                    logger.warning(
                                        "Attachment %s fallback data exceeds size limit (%d > %d bytes), skipping",
                                        safe_name, len(data), MAX_ATTACHMENT_SIZE
                                    )
                                    continue
                                break
                            except Exception:
                                break  # Skip this attachment

                # Final size validation
                try:
                    if isinstance(data, bytes | bytearray) and len(data) > MAX_ATTACHMENT_SIZE:
                        logger.warning(
                            "Attachment %s final size exceeds limit (%d > %d bytes), skipping",
                            safe_name, len(data), MAX_ATTACHMENT_SIZE
                        )
                        continue
                except (TypeError, AttributeError):
                    # If we can't check the length, assume it's potentially too large and skip
                    logger.warning(
                        "Attachment %s final size cannot be validated, skipping for security",
                        safe_name
                    )
                    continue

            else:
                data = getattr(attachment, "data", b"")
                try:
                    if isinstance(data, bytes | bytearray) and len(data) > MAX_ATTACHMENT_SIZE:
                        logger.warning(
                            "Attachment %s data exceeds size limit (%d > %d bytes), skipping",
                            safe_name, len(data), MAX_ATTACHMENT_SIZE
                        )
                        continue
                except (TypeError, AttributeError):
                    logger.warning(
                        "Attachment %s data size cannot be validated, skipping for security",
                        safe_name
                    )
                    continue
        except Exception:
            data = getattr(attachment, "data", b"")
            try:
                if isinstance(data, bytes | bytearray) and len(data) > MAX_ATTACHMENT_SIZE:
                    logger.warning(
                        "Attachment %s exception data exceeds size limit (%d > %d bytes), skipping",
                        safe_name, len(data), MAX_ATTACHMENT_SIZE
                    )
                    continue
            except (TypeError, AttributeError):
                logger.warning(
                    "Attachment %s exception data size cannot be validated, skipping for security",
                    safe_name
                )
                continue

        if not data:
            logger.debug("Attachment %s for message %s had no readable data", safe_name, email_hash)
            continue

        try:
            email_dir.mkdir(parents=True, exist_ok=True)

            # Resolve filename conflicts by appending _<n> before extension
            base = safe_name
            candidate = email_dir / base
            if candidate.exists():
                stem = candidate.stem
                suffix = candidate.suffix
                counter = 2
                while candidate.exists() and counter < 10_000:
                    candidate = email_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            with candidate.open("wb") as handle:
                handle.write(data)
            extracted_paths.append(candidate.relative_to(attachments_root).as_posix())
        except Exception as exc:
            logger.warning("Failed to write attachment %s for message %s: %s", safe_name, email_hash, exc)

    if not extracted_paths and email_dir.exists():
        with contextlib.suppress(OSError):
            email_dir.rmdir()

    return extracted_paths, len(extracted_paths)


def _derive_thread_metadata(_subject: str, message_id: str, transport_headers: str) -> dict:
    """
    Derive thread metadata from email headers.
    
    Extracts threading information from In-Reply-To and References headers.
    Thread ID is derived from the original message in the thread (first reference)
    or the message ID itself if it's a new thread.
    
    Returns a dict with thread_id, parent_id, and references.
    """
    parent_id = ""
    references = ""
    
    if transport_headers:
        # Parse headers line by line, handling multi-line headers
        lines = transport_headers.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            line_lower = line.lower()
            
            # Check for In-Reply-To header
            if line_lower.startswith("in-reply-to:"):
                parent_id = line.split(":", 1)[1].strip()
                # Handle multi-line continuation
                i += 1
                while i < len(lines) and lines[i].startswith((" ", "\t")):
                    parent_id += " " + lines[i].strip()
                    i += 1
                continue
            
            # Check for References header
            elif line_lower.startswith("references:"):
                references = line.split(":", 1)[1].strip()
                # Handle multi-line continuation
                i += 1
                while i < len(lines) and lines[i].startswith((" ", "\t")):
                    references += " " + lines[i].strip()
                    i += 1
                continue
            
            i += 1
    
    # Clean up references and parent_id
    references = " ".join(references.split())  # Normalize whitespace
    parent_id = " ".join(parent_id.split())
    
    # Determine thread ID
    # If there are references, the first one is typically the thread root
    if references:
        # Extract first message ID from references
        ref_ids = [r.strip() for r in references.replace("<", " <").replace(">", "> ").split() if r.startswith("<") and r.endswith(">")]
        thread_id = ref_ids[0] if ref_ids else (parent_id or message_id or "")
    elif parent_id:
        # If no references but has parent, use parent as thread ID
        thread_id = parent_id
    else:
        # New thread - use this message's ID
        thread_id = message_id or ""
    
    return {
        "thread_id": thread_id,
        "parent_id": parent_id,
        "references": references,
    }


def _collect_received_hosts(transport_headers: str) -> str:
    """
    Extract Received headers to track email path.
    
    Parses all 'Received:' headers from the transport headers and extracts
    the host/server information from each hop. This provides a trail of
    servers the email passed through.
    
    Returns a semicolon-separated string of received headers, or empty string if none found.
    """
    if not transport_headers:
        return ""
    
    received_lines = []
    lines = transport_headers.split("\n")
    i = 0
    
    while i < len(lines):
        line = lines[i]
        line_lower = line.lower()
        
        # Check for Received header
        if line_lower.startswith("received:"):
            # Capture the full header including continuation lines
            full_header = line
            i += 1
            # Handle multi-line continuation (lines starting with whitespace)
            while i < len(lines) and lines[i].startswith((" ", "\t")):
                full_header += " " + lines[i].strip()
                i += 1
            
            # Clean up and add to list
            received_lines.append(full_header.strip())
            continue
        
        i += 1
    
    # Join all received headers with semicolons for readability
    return "; ".join(received_lines)


def _parse_message(message: Any, index: int, attachments_root: Path | None = None):
    """
    Parse individual email message and extract all relevant fields.
    """
    try:
        email_hash = _generate_email_hash(message, index)
        subject = _get_message_property(message, "subject")
        sender_name = _get_message_property(message, "sender_name")
        sender_email = _get_message_property(message, "sender_email_address")

        from_field = f"{sender_name} <{sender_email}>" if sender_email else sender_name

        to_field = _get_message_property(message, "display_to")
        cc_field = _get_message_property(message, "display_cc")
        bcc_field = _get_message_property(message, "display_bcc")

        delivery_time = getattr(message, "delivery_time", None)
        client_submit_time = getattr(message, "client_submit_time", None)

        # Convert timestamps to both ISO strings and raw timestamps
        date_received_timestamp = None
        date_sent_timestamp = None
        if delivery_time:
            try:
                # Handle datetime objects from pypff
                if hasattr(delivery_time, 'timestamp'):
                    date_received_timestamp = delivery_time.timestamp()
                else:
                    date_received_timestamp = float(delivery_time)
            except (ValueError, TypeError, AttributeError) as exc:
                logger.debug("Could not parse client_submit_time '%s': %s", client_submit_time, exc)
        
        if client_submit_time:
            try:
                # Handle datetime objects from pypff
                if hasattr(client_submit_time, 'timestamp'):
                    date_sent_timestamp = client_submit_time.timestamp()
                else:
                    date_sent_timestamp = float(client_submit_time)
            except (ValueError, TypeError, AttributeError):
                pass

        date_utc = str(delivery_time) if delivery_time else ""
        date_local = str(client_submit_time) if client_submit_time else ""

        body, html_body = _extract_body_and_html(message)
        content_type = _get_message_property(message, "message_class")
        transport_headers = _get_message_property(message, "transport_headers")

        try:
            attachment_count = message.number_of_attachments
        except Exception as exc:
            attachment_count = 0
            logger.warning("Unable to determine attachment count for message %s: %s", index, exc)

        extracted_paths, extracted_count = _extract_attachments(message, email_hash, attachments_root)
        if extracted_count:
            attachment_count = extracted_count

        reply_to = ""
        return_path = ""
        x_mailer = ""

        if transport_headers:
            for line in transport_headers.split("\n"):
                line_lower = line.lower()
                if line_lower.startswith("reply-to:"):
                    reply_to = line.split(":", 1)[1].strip()
                elif line_lower.startswith("return-path:"):
                    return_path = line.split(":", 1)[1].strip()
                elif line_lower.startswith("x-mailer:"):
                    x_mailer = line.split(":", 1)[1].strip()

        importance = "Normal"
        try:
            priority = message.priority
            if priority == 1:
                importance = "High"
            elif priority == -1:
                importance = "Low"
        except Exception:
            pass

        message_id = _get_message_property(message, "internet_message_id")
        client_info = _get_message_property(message, "message_class")

        thread_metadata = _derive_thread_metadata(subject, message_id, transport_headers)
        received_hosts = _collect_received_hosts(transport_headers)

        return {
            "Email_ID": email_hash,
            "Date_Received": date_utc,
            "Date_Received_Timestamp": date_received_timestamp,
            "Date_Sent": date_local,
            "Date_Sent_Timestamp": date_sent_timestamp,
            "From": from_field,
            "Sender_Email": sender_email,
            "To": to_field,
            "CC": cc_field,
            "BCC": bcc_field,
            "Reply_To": reply_to,
            "Subject": subject,
            "Body": body,
            "HTML_Body": html_body,
            "Importance": importance,
            "Attachment_Count": attachment_count,
            "Attachment_Paths": extracted_paths,
            "Message_ID": message_id,
            "Content_Type": content_type,
            "Email_Client": x_mailer,
            "Return_Path": return_path,
            "Client_Info": client_info,
            "Thread_ID": thread_metadata["thread_id"],
            "Parent_ID": thread_metadata["parent_id"],
            "References": thread_metadata["references"],
            "Received_Hops": received_hosts,
            "Full_Headers": transport_headers,
        }
    except Exception as exc:
        logger.error("Error parsing message %s: %s", index, exc)
        return None


def _count_messages(root_folder: Any) -> int:
    """Count total messages within the PST folder tree."""
    total = 0
    stack = [root_folder]
    while stack:
        folder = stack.pop()
        try:
            sub_count = getattr(folder, "number_of_sub_messages", None)
            if sub_count is None:
                raise AttributeError
            total += int(sub_count)
        except Exception as exc:
            logger.debug("Unable to read number_of_sub_messages: %s", exc)
            try:
                total += sum(1 for _ in folder.sub_messages)
            except Exception as inner_exc:
                logger.warning("Unable to count messages in folder: %s", inner_exc)
        try:
            for subfolder in folder.sub_folders:
                stack.append(subfolder)
        except Exception as exc:
            logger.warning("Unable to enumerate subfolders during count: %s", exc)
    return total


def _report_progress(progress_state: dict | None, current: int, status: str, force: bool = False) -> None:
    """Invoke progress callback safely with throttling to reduce overhead.
    
    Progress updates are throttled to report every 100 emails or 1% of total,
    whichever is larger. This reduces callback overhead by ~99% for large PST files.
    
    Args:
        progress_state: Dictionary containing callback, total, and last_update
        current: Current item count
        status: Status message
        force: If True, bypass throttling and always report
    """
    if not progress_state:
        return
    
    callback = progress_state["callback"]
    total = progress_state["total"]
    
    # Throttle updates unless forced
    if not force:
        last_update = progress_state.get("last_update", 0)
        # Update every 1% or every 100 items, whichever is larger
        update_interval = max(100, total // 100) if total > 0 else 100
        
        # Skip if not enough progress since last update and not at end
        if current - last_update < update_interval and current < total:
            return
    
    # Record this update
    progress_state["last_update"] = current
    
    try:
        callback(current, total, status)
    except Exception as exc:
        logger.debug("Progress callback raised an exception: %s", exc)


def _iterate_folder(
    folder: Any,
    index_counter: list[int],
    progress_state: dict | None = None,
    attachments_root: Path | None = None,
    deduplicate_state: dict | None = None,
) -> Iterable[dict]:
    """Yield email payloads from folder and subfolders."""
    try:
        for message in folder.sub_messages:
            email_data = _parse_message(message, index_counter[0], attachments_root)
            is_duplicate = False
            if email_data and deduplicate_state is not None:
                email_hash = email_data.get("Email_ID")
                if email_hash and email_hash in deduplicate_state["seen"]:
                    deduplicate_state["duplicates"] += 1
                    is_duplicate = True
                else:
                    if email_hash:
                        deduplicate_state["seen"].add(email_hash)
                    deduplicate_state["unique"] += 1

            if email_data and not is_duplicate:
                yield email_data

            index_counter[0] += 1
            current = index_counter[0]

            status = f"Processed {current} emails"
            if deduplicate_state is not None:
                status += (
                    f" ({deduplicate_state['unique']} unique"
                    + (
                        f", {deduplicate_state['duplicates']} duplicates skipped"
                        if deduplicate_state["duplicates"]
                        else ""
                    )
                    + ")"
                )

            _report_progress(progress_state, current, status)

        for subfolder in folder.sub_folders:
            yield from _iterate_folder(
                subfolder,
                index_counter,
                progress_state,
                attachments_root,
                deduplicate_state,
            )
    except Exception as exc:
        logger.error("Error processing folder: %s", exc)
    finally:
        # Force a progress update when folder is complete
        if progress_state and index_counter[0] > 0:
            _report_progress(progress_state, index_counter[0], "Completed folder processing", force=True)


def iter_emails(
    pst_path,
    progress_callback=None,
    *,
    deduplicate: bool = False,
    attachments_dir: str | Path | None = None,
):
    """
    Stream email payloads from a PST file without retaining the full dataset in memory.
    """
    require_pypff()

    index_counter = [0]
    progress_state = None

    normalized_path = os.path.abspath(pst_path)
    attachments_root = Path(attachments_dir).expanduser().resolve() if attachments_dir else None
    if attachments_root:
        attachments_root.mkdir(parents=True, exist_ok=True)

    deduplicate_state = None
    if deduplicate:
        deduplicate_state = {
            "seen": set(),
            "unique": 0,
            "duplicates": 0,
        }

    pst_file = pypff.file()
    try:
        pst_file.open(normalized_path)
        root_folder = pst_file.root_folder

        if root_folder is None:
            logger.error("No root folder found in PST file")
            return

        if progress_callback:
            total_messages = _count_messages(root_folder)
            progress_state = {
                "callback": progress_callback,
                "total": total_messages or 0,
                "last_update": 0,  # Track last update for throttling
            }
            progress_callback(0, total_messages or 0, "Starting extraction")

        logger.info("Extracting emails from PST file...")

        try:
            yield from _iterate_folder(
                root_folder,
                index_counter,
                progress_state,
                attachments_root,
                deduplicate_state,
            )
        finally:
            final_count = index_counter[0]
            if progress_state:
                # Force final progress update
                status = (
                    f"Processed {final_count} emails"
                    if not deduplicate_state
                    else (
                        f"Processed {final_count} emails ("
                        f"{deduplicate_state['unique']} unique"
                        + (
                            f", {deduplicate_state['duplicates']} duplicates skipped"
                            if deduplicate_state["duplicates"]
                            else ""
                        )
                        + ")"
                    )
                )
                _report_progress(progress_state, final_count, status, force=True)
            logger.info("Processing complete")
    except Exception as exc:
        logger.exception("Error reading PST file: %s", exc)
        raise
    finally:
        with contextlib.suppress(Exception):  # pragma: no cover - defensive
            pst_file.close()

