"""
PST file parser for extracting email data.
Supports extraction of headers, body, attachments from Outlook PST files.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
import hashlib
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger("pst_tools.parser")
logger.setLevel(logging.INFO)
logger.propagate = False

_FILE_HANDLER = None
_STREAM_HANDLER = None
_LOG_PATH = Path("pst_parser.log").expanduser()


def _install_handlers(log_path: Path | None = None) -> Path:
    """Configure rotating file and console handlers."""
    global _FILE_HANDLER, _STREAM_HANDLER, _LOG_PATH

    if log_path is not None:
        _LOG_PATH = Path(log_path).expanduser()
    log_target = _LOG_PATH
    log_target.parent.mkdir(parents=True, exist_ok=True)

    if _FILE_HANDLER:
        logger.removeHandler(_FILE_HANDLER)
    _FILE_HANDLER = RotatingFileHandler(
        log_target,
        maxBytes=1_048_576,
        backupCount=3,
        encoding="utf-8"
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


_install_handlers()


try:
    import pypff
except ImportError:
    logger.error("pypff library not found")
    logger.info("Install with: pip install pypff")
    logger.info("Note: pypff may require additional system dependencies")
    sys.exit(1)

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from striprtf.striprtf import rtf_to_text
except ImportError:
    rtf_to_text = None


def _sanitize_text(value):
    """Ensure text is a clean UTF-8 string without surrogate characters."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode('utf-8', errors='ignore')
    return value.encode('utf-8', 'ignore').decode('utf-8', 'ignore')


class _HTMLStripper(HTMLParser):
    """Minimal HTML parser that collects text data."""

    def __init__(self):
        super().__init__()
        self._chunks = []

    def handle_data(self, data):
        if data:
            self._chunks.append(data)

    def get_text(self):
        return "".join(self._chunks)


def _collapse_whitespace(value):
    """Normalize whitespace for cleaner output."""
    value = _sanitize_text(value)
    if not value:
        return ""
    lines = [line.strip() for line in value.splitlines()]
    filtered = [line for line in lines if line]
    return "\n".join(filtered) if filtered else value.strip()


def _html_to_text(html_content):
    """Convert HTML content to readable plain text."""
    if not html_content:
        return ""

    if BeautifulSoup:
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
    else:
        stripper = _HTMLStripper()
        stripper.feed(html_content)
        stripper.close()
        text = stripper.get_text()

    return _collapse_whitespace(unescape(_sanitize_text(text)))


def _rtf_to_text(rtf_content):
    """Convert RTF content to plain text when possible."""
    if not rtf_content:
        return ""

    if rtf_to_text:
        try:
            return _collapse_whitespace(rtf_to_text(_sanitize_text(rtf_content)))
        except Exception:
            pass

    # fallback: return decoded raw text without braces
    cleaned = rtf_content.replace("{", "").replace("}", "")
    return _collapse_whitespace(unescape(_sanitize_text(cleaned)))


def _get_message_property(message, property_name, default=""):
    """
    Safely extract a property from a message object.
    
    Args:
        message: pypff message object
        property_name: Name of property to extract
        default: Default value if property doesn't exist
        
    Returns:
        Property value or default
    """
    try:
        value = getattr(message, property_name, default)
        if value is None:
            return default
        # handle bytes returned by pypff
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='ignore')
        return str(value)
    except Exception:
        return default


def _generate_email_hash(message, index):
    """
    Generate unique hash for email message.
    
    Args:
        message: pypff message object
        index: Index of message in PST
        
    Returns:
        Unique hash string
    """
    try:
        # combine multiple fields to create unique identifier
        subject = _get_message_property(message, 'subject')
        sender = _get_message_property(message, 'sender_name')
        delivery_time = str(_get_message_property(message, 'delivery_time'))
        
        hash_input = f"{subject}{sender}{delivery_time}{index}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    except Exception:
        # fallback to index-based hash
        return hashlib.md5(f"email_{index}".encode()).hexdigest()


def _extract_body(message):
    """
    Extract email body content (plain text or HTML).
    
    Args:
        message: pypff message object
        
    Returns:
        Email body as string
    """
    try:
        # try plain text first
        plain_body = message.plain_text_body
        if plain_body:
            if isinstance(plain_body, bytes):
                plain_body = plain_body.decode('utf-8', errors='ignore')
            return _collapse_whitespace(str(plain_body))

        # fallback to HTML body
        html_body = message.html_body
        if html_body:
            if isinstance(html_body, bytes):
                html_body = html_body.decode('utf-8', errors='ignore')
            return _html_to_text(str(html_body))

        # fallback to RTF body if available
        rtf_body = message.rtf_body
        if rtf_body:
            if isinstance(rtf_body, bytes):
                rtf_body = rtf_body.decode('utf-8', errors='ignore')
            return _rtf_to_text(str(rtf_body))

        return ""
    except Exception:
        return ""


def _parse_message(message, index):
    """
    Parse individual email message and extract all relevant fields.
    
    Args:
        message: pypff message object
        index: Index of message in PST
        
    Returns:
        Dictionary containing email metadata and content
    """
    try:
        # extract basic metadata
        email_hash = _generate_email_hash(message, index)
        subject = _get_message_property(message, 'subject')
        sender_name = _get_message_property(message, 'sender_name')
        sender_email = _get_message_property(message, 'sender_email_address')
        
        # combine sender info
        from_field = f"{sender_name} <{sender_email}>" if sender_email else sender_name
        
        # recipients
        to_field = _get_message_property(message, 'display_to')
        cc_field = _get_message_property(message, 'display_cc')
        
        # dates
        delivery_time = message.delivery_time
        client_submit_time = message.client_submit_time
        
        # format dates
        date_utc = str(delivery_time) if delivery_time else ""
        date_local = str(client_submit_time) if client_submit_time else ""
        
        # content and metadata
        body = _extract_body(message)
        content_type = _get_message_property(message, 'message_class')
        transport_headers = _get_message_property(message, 'transport_headers')
        
        # attachment count (libpff may raise on corrupt descriptors)
        try:
            attachment_count = message.number_of_attachments
        except Exception as exc:
            attachment_count = 0
            logger.warning("Unable to determine attachment count for message %s: %s", index, exc)
        
        # extract additional headers from transport_headers if available
        reply_to = ""
        return_path = ""
        x_mailer = ""
        
        if transport_headers:
            headers_lower = transport_headers.lower()
            # parse common headers from transport headers string
            for line in transport_headers.split('\n'):
                line_lower = line.lower()
                if line_lower.startswith('reply-to:'):
                    reply_to = line.split(':', 1)[1].strip()
                elif line_lower.startswith('return-path:'):
                    return_path = line.split(':', 1)[1].strip()
                elif line_lower.startswith('x-mailer:'):
                    x_mailer = line.split(':', 1)[1].strip()
        
        # importance level
        importance = "Normal"
        try:
            priority = message.priority
            if priority == 1:
                importance = "High"
            elif priority == -1:
                importance = "Low"
        except Exception:
            pass
        
        # message ID
        message_id = _get_message_property(message, 'internet_message_id')
        
        # client info (application that created the message)
        client_info = _get_message_property(message, 'message_class')
        
        return {
            'Email_ID': email_hash,
            'Date_Received': date_utc,
            'Date_Sent': date_local,
            'From': from_field,
            'To': to_field,
            'CC': cc_field,
            'Reply_To': reply_to,
            'Subject': subject,
            'Body': body,
            'Importance': importance,
            'Attachment_Count': attachment_count,
            'Message_ID': message_id,
            'Content_Type': content_type,
            'Email_Client': x_mailer,
            'Return_Path': return_path,
            'Client_Info': client_info,
            'Full_Headers': transport_headers
        }
    except Exception as e:
        logger.error("Error parsing message %s: %s", index, e)
        return None


def _count_messages(root_folder):
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


def _report_progress(progress_state, current, status):
    """Invoke progress callback safely."""
    if not progress_state:
        return
    callback = progress_state["callback"]
    total = progress_state["total"]
    try:
        callback(current, total, status)
    except Exception as exc:
        logger.debug("Progress callback raised an exception: %s", exc)


def _iterate_folder(folder, index_counter, progress_state=None):
    """Yield email payloads from folder and subfolders."""
    try:
        # process messages in current folder
        for message in folder.sub_messages:
            email_data = _parse_message(message, index_counter[0])
            if email_data:
                yield email_data
            index_counter[0] += 1
            current = index_counter[0]
            _report_progress(
                progress_state,
                current,
                f"Processed {current} emails"
            )
            
            # progress indicator every 100 messages
            if current % 100 == 0:
                logger.info("Processed %s messages...", current)
        
        # recursively process subfolders
        for subfolder in folder.sub_folders:
            yield from _iterate_folder(subfolder, index_counter, progress_state)
            
    except Exception as e:
        logger.error("Error processing folder: %s", e)
    
    return


def iter_emails(pst_path, progress_callback=None):
    """
    Stream email payloads from a PST file without retaining the full dataset in memory.
    """
    import os
    index_counter = [0]
    progress_state = None

    normalized_path = os.path.abspath(pst_path)

    pst_file = pypff.file()
    try:
        pst_file.open(normalized_path)
        root_folder = pst_file.root_folder

        if root_folder is None:
            logger.error("No root folder found in PST file")
            return

        total_messages = None
        if progress_callback:
            total_messages = _count_messages(root_folder)
            progress_state = {
                "callback": progress_callback,
                "total": total_messages or 0
            }
            progress_callback(0, total_messages or 0, "Starting extraction")

        logger.info("Extracting emails from PST file...")

        try:
            for email_payload in _iterate_folder(root_folder, index_counter, progress_state):
                yield email_payload
        finally:
            final_count = index_counter[0]
            if progress_state:
                progress_callback(
                    final_count,
                    progress_state["total"],
                    f"Processed {final_count} emails"
                )
            logger.info("Processing complete")
    except Exception as e:
        logger.exception("Error reading PST file: %s", e)
        raise
    finally:
        try:
            pst_file.close()
        except Exception:
            pass


def extract_emails(pst_path, progress_callback=None):
    """
    Preserve backwards compatibility by materialising all emails in memory.
    """
    emails = {}
    for email in iter_emails(pst_path, progress_callback=progress_callback):
        email_id = email.get("Email_ID")
        if email_id:
            emails[email_id] = email
    return emails

