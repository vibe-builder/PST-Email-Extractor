"""
JSON exporter for PST email data.

This module provides streaming JSON export functionality that writes email records
to a JSON object keyed by Email_ID. It supports incremental writing without loading
all data into memory, making it suitable for processing large PST files.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from .base import AIPipelineMixin, EmailExporter, _normalise_output_path

logger = logging.getLogger("pst_email_extractor.exporters.json")


class JSONStreamWriter(EmailExporter, AIPipelineMixin):
    """
    Stream emails to JSON (dictionary keyed by Email_ID) incrementally.

    This writer creates a JSON object where each email record is stored as a
    key-value pair with the Email_ID as the key. Records are written incrementally
    to avoid memory issues with large datasets.

    For production use with large datasets, disable pretty printing (indent=None)
    to reduce file size by ~30% and improve write speed by ~20%.
    """

    def __init__(self, output_path: str | Path, pretty_print: bool = False,
                 ai_sanitize: bool = False, ai_polish: bool = False,
                 ai_language: str = "en-US", ai_neural_model_dir: str | None = None,
                 compress: bool = False) -> None:
        AIPipelineMixin.__init__(self, ai_sanitize, ai_polish, ai_language, ai_neural_model_dir)
        self.path = _normalise_output_path(output_path)
        self._compress = compress
        if compress and not str(self.path).endswith('.gz'):
            self.path = Path(str(self.path) + '.gz')
        self._handle = None
        self._first = True
        self._index = 0
        self._indent = 2 if pretty_print else None

    def __enter__(self) -> JSONStreamWriter:
        logger.info("Writing JSON export to %s%s", self.path, " (compressed)" if self._compress else "")
        if self._compress:
            self._handle = gzip.open(self.path, "wt", encoding="utf-8", compresslevel=6)
        else:
            self._handle = self.path.open("w", encoding="utf-8")
        self._handle.write("{\n")
        return self

    def write(self, email_row: Mapping[str, Any]) -> None:
        """Write a single email record to the JSON file."""
        if not self._handle:
            raise RuntimeError("JSONStreamWriter not initialised. Use as a context manager.")

        # Process email record through AI pipeline if enabled
        processed_row = self.process_email_record(email_row)

        # Generate Email_ID if not present
        email_id = processed_row.get("Email_ID")
        if not email_id:
            email_id = f"email_{self._index}"
        self._index += 1

        # Serialize the email record with optional pretty printing
        payload = json.dumps(dict(processed_row), ensure_ascii=False, default=str, indent=self._indent)

        # Write with proper JSON formatting
        if self._indent:
            # Pretty printed format with indentation
            prefix = "" if self._first else ",\n"
            self._handle.write(f'{prefix}  "{email_id}": {payload}')
        else:
            # Compact format for production use
            prefix = "" if self._first else ","
            self._handle.write(f'{prefix}"{email_id}":{payload}')
        self._first = False

    def close(self) -> None:
        """Close the JSON file handle and finalize the JSON structure."""
        if not self._handle:
            return

        # Close the JSON object
        if self._first:
            self._handle.write("}\n")
        else:
            self._handle.write("\n}\n")

        self._handle.close()
        self._handle = None

    def __exit__(self, exc_type, exc, tb) -> None:
        """Context manager exit handler."""
        self.close()


def export_to_json(
    output_path: str | Path,
    email_data: Mapping[str, Mapping[str, Any]],
    pretty_print: bool = False,
) -> Path:
    """
    Export email data to a JSON file.

    This is a convenience function for batch exporting email data that is
    already loaded into memory. For streaming exports from large PST files,
    use JSONStreamWriter directly.
    
    Args:
        output_path: Path to output JSON file
        email_data: Dictionary of email records keyed by Email_ID
        pretty_print: If True, format with indentation (slower, larger files)
    """
    destination = _normalise_output_path(output_path)
    logger.info("Writing JSON export to %s (pretty_print=%s)", destination, pretty_print)
    with JSONStreamWriter(destination, pretty_print=pretty_print) as writer:
        for email_id, email_content in email_data.items():
            payload = dict(email_content)
            payload.setdefault("Email_ID", email_id)
            writer.write(payload)
    return destination
