"""
Export helpers for PST email data.

This module provides streaming writers for the various export formats offered
by the application. The writers are implemented such that very large PST
files can be processed without exhausting memory.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .base import ADDRESS_FIELDS, EMAIL_FIELDS, HOST_FIELDS, _normalise_output_path
from .csv_writer import CSVStreamWriter, export_to_csv
from .eml_writer import EMLWriter
from .html_index import generate_html_index
from .json_writer import JSONStreamWriter, export_to_json
from .mbox_writer import MBOXWriter

logger = logging.getLogger("pst_email_extractor.exporters")


def export_address_report_to_json(output_path: str | Path, report: Mapping[str, Any]) -> Path:
    """
    Export address analysis report to JSON file.

    This function handles the serialization of address analysis results,
    which may include complex data structures that need special handling.
    """
    destination = _normalise_output_path(output_path)
    logger.info("Writing address analysis JSON to %s", destination)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    return destination


def export_addresses_to_csv(output_path: str | Path, addresses: list[Mapping[str, Any]]) -> Path:
    """
    Export address data to CSV format.

    Handles the conversion of complex address data structures (lists, sets)
    to CSV-compatible string representations.
    """
    destination = _normalise_output_path(output_path)
    logger.info("Writing address analysis CSV to %s", destination)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ADDRESS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in addresses:
            row = {}
            for field in ADDRESS_FIELDS:
                value = entry.get(field, "")
                if isinstance(value, list | tuple | set):
                    value = "; ".join(str(item) for item in sorted(value) if item)
                row[field] = str(value)
            writer.writerow(row)
    return destination


def export_hosts_to_csv(output_path: str | Path, hosts: list[Mapping[str, Any]]) -> Path:
    """
    Export host analysis data to CSV format.

    Creates a simple CSV with host information and occurrence counts.
    """
    destination = _normalise_output_path(output_path)
    logger.info("Writing transport host CSV to %s", destination)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, HOST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for entry in hosts:
            row = {field: str(entry.get(field, "")) for field in HOST_FIELDS}
            writer.writerow(row)
    return destination


__all__ = [
    # Constants
    "EMAIL_FIELDS",
    "ADDRESS_FIELDS",
    "HOST_FIELDS",
    # Writers
    "CSVStreamWriter",
    "JSONStreamWriter",
    "EMLWriter",
    "MBOXWriter",
    # Functions
    "export_to_json",
    "export_to_csv",
    "export_address_report_to_json",
    "export_addresses_to_csv",
    "export_hosts_to_csv",
    "generate_html_index",
]


