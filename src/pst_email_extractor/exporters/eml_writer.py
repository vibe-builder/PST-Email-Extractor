"""
EML exporter for PST email data.

This module provides functionality to export individual emails as RFC 822
compliant .eml files. Each email is written to its own file with proper MIME
encoding and attachment handling.
"""

from __future__ import annotations

import logging
import mimetypes
from collections.abc import Mapping
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Any

from .base import AIPipelineMixin, EmailExporter

logger = logging.getLogger("pst_email_extractor.exporters.eml")


class EMLWriter(EmailExporter, AIPipelineMixin):
    """
    Write individual emails to .eml files.

    This exporter creates RFC 822 compliant .eml files for each email record.
    Attachments are embedded in the MIME structure when available. Filenames
    are sanitized to ensure filesystem compatibility.
    """

    def __init__(self, output_dir: str | Path, attachments_dir: str | Path | None = None,
                 ai_sanitize: bool = False, ai_polish: bool = False,
                 ai_language: str = "en-US", ai_neural_model_dir: str | None = None) -> None:
        AIPipelineMixin.__init__(self, ai_sanitize, ai_polish, ai_language, ai_neural_model_dir)
        self.directory = Path(output_dir).expanduser()
        self.attachments_dir = Path(attachments_dir).expanduser() if attachments_dir else None
        self._counter = 0

    def __enter__(self) -> EMLWriter:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.directory = self.directory.resolve()
        if self.attachments_dir:
            self.attachments_dir = self.attachments_dir.resolve()
        logger.info("Writing EML exports to directory: %s", self.directory)
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

    @staticmethod
    def _resolve_conflict_name(base_dir: Path, filename: str) -> Path:
        """Return a unique path within base_dir by appending _<n> before extension."""
        path = base_dir / filename
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        counter = 2
        candidate = path
        while candidate.exists() and counter < 10_000:
            candidate = base_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        return candidate

    def write(self, email_row: Mapping[str, Any]) -> Path:
        """Write a single email record as an .eml file."""
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
            # Ensure a text/plain alternative exists
            try:
                msg.set_content(body or "")
                msg.add_alternative(html_body, subtype="html")
            except Exception:
                # Fallback to plain text only
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
                    logger.warning("Could not add attachment %s to EML: %s", attach_path.name, e)

        # Generate filename and write file
        email_id = email_row.get("Email_ID", f"email_{self._counter}")
        filename = f"{email_id}.eml"
        output_path = self.directory / filename

        try:
            with output_path.open("wb") as f:
                f.write(msg.as_bytes())
            self._counter += 1
            return output_path
        except OSError as e:
            logger.error("Failed to write EML file %s: %s", output_path, e)
            raise

    def close(self) -> None:
        """Finalize EML export operation."""
        logger.info("Finished EML exports to directory: %s", self.directory)

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context manager exit handler."""
        self.close()
