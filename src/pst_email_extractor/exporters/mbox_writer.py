"""
MBOX exporter for PST email data.

This module provides functionality to export emails to standard MBOX format files.
MBOX is a common archive format used by many email clients for storing collections
of email messages in a single file.
"""

from __future__ import annotations

import logging
import mailbox
import mimetypes
from collections.abc import Mapping
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any

from .base import AIPipelineMixin, EmailExporter, _normalise_output_path

logger = logging.getLogger("pst_email_extractor.exporters.mbox")


class MBOXWriter(EmailExporter, AIPipelineMixin):
    """
    Append emails to a standard MBOX file.

    This exporter creates or appends to MBOX format files, which store multiple
    email messages in a single file with "From " separator lines. Attachments
    are embedded in the MIME structure of each message.
    """

    def __init__(self, output_path: str | Path, attachments_dir: str | Path | None = None,
                 ai_sanitize: bool = False, ai_polish: bool = False,
                 ai_language: str = "en-US", ai_neural_model_dir: str | None = None) -> None:
        AIPipelineMixin.__init__(self, ai_sanitize, ai_polish, ai_language, ai_neural_model_dir)
        self.path = _normalise_output_path(output_path)
        self.attachments_dir = Path(attachments_dir).expanduser() if attachments_dir else None
        self._mbox: mailbox.mbox | None = None

    def __enter__(self) -> MBOXWriter:
        logger.info("Writing MBOX export to: %s", self.path)
        if self.attachments_dir:
            self.attachments_dir = self.attachments_dir.resolve()
        self._mbox = mailbox.mbox(str(self.path))
        self._mbox.lock()
        return self

    def _resolve_attachment_path(self, attachment_path: str) -> Path | None:
        """Resolve attachment path relative to the configured attachments directory."""
        candidate = Path(attachment_path)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if self.attachments_dir:
            joined = (self.attachments_dir / candidate).resolve()
            if joined.exists():
                return joined
        return None

    def write(self, email_row: Mapping[str, Any]) -> None:
        """Write a single email record to the MBOX file."""
        if self._mbox is None:
            raise RuntimeError("MBOXWriter not initialised. Use as a context manager.")

        # Create email message from data
        msg = EmailMessage()

        # Set standard headers
        msg["Subject"] = email_row.get("Subject", "No Subject")
        msg["From"] = email_row.get("From", "unknown@example.com")
        msg["To"] = email_row.get("To", "")
        if email_row.get("CC"):
            msg["Cc"] = email_row["CC"]
        if email_row.get("BCC"):
            msg["Bcc"] = email_row["BCC"]
        msg["Date"] = formatdate(
            float(email_row["Date_Received_Timestamp"]) if email_row.get("Date_Received_Timestamp") else None,
            usegmt=True
        )
        msg["Message-ID"] = email_row.get("Message_ID", make_msgid())

        # Add additional headers from Full_Headers if available
        if email_row.get("Full_Headers"):
            for header_line in email_row["Full_Headers"].splitlines():
                if ":" in header_line and not header_line.startswith(
                    ("Subject", "From", "To", "Cc", "Bcc", "Date", "Message-ID")
                ):
                    try:
                        key, value = header_line.split(":", 1)
                        msg[key.strip()] = value.strip()
                    except ValueError:
                        pass  # Malformed header, skip it

        # Set message body with multipart alternative when HTML is available
        body = email_row.get("Body", "") or ""
        html_body = email_row.get("HTML_Body")
        if html_body:
            try:
                msg.set_content(body or "")
                msg.add_alternative(html_body, subtype="html")
            except Exception:
                msg.set_content(body or "")
        else:
            msg.set_content(body or "")

        # Add attachments
        attachment_paths = email_row.get("Attachment_Paths", [])
        if attachment_paths:
            for attach_path_str in attachment_paths:
                attach_path = self._resolve_attachment_path(attach_path_str)
                if not attach_path:
                    logger.warning("Attachment file not found: %s", attach_path_str)
                    continue
                try:
                    with attach_path.open("rb") as f:
                        data = f.read()

                    # Determine MIME type
                    mime_type, _ = mimetypes.guess_type(attach_path.name)
                    if mime_type:
                        maintype, subtype = mime_type.split("/", 1)
                    else:
                        maintype, subtype = "application", "octet-stream"

                    # Add attachment to message
                    msg.add_attachment(
                        data,
                        maintype=maintype,
                        subtype=subtype,
                        filename=attach_path.name,
                    )
                except OSError as e:
                    logger.warning("Could not add attachment %s to MBOX: %s", attach_path.name, e)

        # Add message to MBOX
        self._mbox.add(msg)

    def close(self) -> None:
        """Finalize MBOX export and close the file."""
        if self._mbox is None:
            return

        try:
            self._mbox.flush()
        finally:
            try:
                self._mbox.unlock()
            finally:
                self._mbox.close()

        self._mbox = None
        logger.info("Finished MBOX export to: %s", self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context manager exit handler."""
        self.close()
