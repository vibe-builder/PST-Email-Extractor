"""
Shared export helpers for PST email data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, MutableMapping, Any

import unicodecsv


logger = logging.getLogger("pst_tools.parser")

EMAIL_FIELDS = [
    "Email_ID",
    "Date_Received",
    "Date_Sent",
    "From",
    "To",
    "CC",
    "Reply_To",
    "Subject",
    "Body",
    "Importance",
    "Attachment_Count",
    "Message_ID",
    "Content_Type",
    "Email_Client",
    "Return_Path",
    "Client_Info",
    "Full_Headers",
]


def _normalise_output_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def export_to_json(
    output_path: str | Path,
    email_data: Mapping[str, Mapping[str, Any]],
) -> Path:
    """
    Persist email data to JSON with UTF-8 encoding.
    """
    destination = _normalise_output_path(output_path)
    logger.info("Writing JSON export to %s", destination)
    with JSONStreamWriter(destination) as writer:
        for email_id, email_content in email_data.items():
            payload = dict(email_content)
            payload.setdefault("Email_ID", email_id)
            writer.write(payload)
    return destination


def export_to_csv(
    output_path: str | Path,
    email_data: Mapping[str, MutableMapping[str, Any]],
    fields: list[str] | None = None,
) -> Path:
    """
    Persist email data to CSV using unicodecsv for broad compatibility.
    """
    destination = _normalise_output_path(output_path)
    selected_fields = fields or EMAIL_FIELDS
    logger.info("Writing CSV export to %s", destination)
    with CSVStreamWriter(destination, selected_fields) as writer:
        for email_id, email_content in email_data.items():
            payload = dict(email_content)
            payload.setdefault("Email_ID", email_id)
            writer.write(payload)
    return destination


class CSVStreamWriter:
    """Stream emails directly to CSV without holding everything in memory."""

    def __init__(self, output_path: str | Path, fields: list[str] | None = None) -> None:
        self.path = _normalise_output_path(output_path)
        self.fields = fields or EMAIL_FIELDS
        self._handle = None
        self._writer = None

    def __enter__(self) -> "CSVStreamWriter":
        self._handle = self.path.open("wb")
        self._writer = unicodecsv.DictWriter(self._handle, self.fields)
        self._writer.writeheader()
        return self

    def write(self, email_row: Mapping[str, Any]) -> None:
        if not self._writer:
            raise RuntimeError("CSVStreamWriter not initialised. Use as a context manager.")
        email_id = email_row.get("Email_ID", "")
        row = {field: email_row.get(field, email_id) for field in self.fields}
        self._writer.writerow(row)

    def close(self) -> None:
        if self._handle:
            self._handle.close()
            self._handle = None
            self._writer = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class JSONStreamWriter:
    """Stream emails to JSON (dictionary keyed by Email_ID) incrementally."""

    def __init__(self, output_path: str | Path) -> None:
        self.path = _normalise_output_path(output_path)
        self._handle = None
        self._first = True
        self._index = 0

    def __enter__(self) -> "JSONStreamWriter":
        self._handle = self.path.open("w", encoding="utf-8")
        self._handle.write("{\n")
        return self

    def write(self, email_row: Mapping[str, Any]) -> None:
        if not self._handle:
            raise RuntimeError("JSONStreamWriter not initialised. Use as a context manager.")

        email_id = email_row.get("Email_ID")
        if not email_id:
            email_id = f"email_{self._index}"
        self._index += 1

        payload = json.dumps(dict(email_row), ensure_ascii=False, default=str, indent=2)
        prefix = "" if self._first else ",\n"
        self._handle.write(f'{prefix}  "{email_id}": {payload}')
        self._first = False

    def close(self) -> None:
        if not self._handle:
            return
        if self._first:
            self._handle.write("}\n")
        else:
            self._handle.write("\n}\n")
        self._handle.close()
        self._handle = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "EMAIL_FIELDS",
    "export_to_json",
    "export_to_csv",
    "CSVStreamWriter",
    "JSONStreamWriter",
]
