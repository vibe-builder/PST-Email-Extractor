"""
Lightweight PST health analysis helpers.

These routines provide a quick, read-only health summary for a PST file
without materialising messages into memory. They are defensive and will
gracefully degrade when optional dependencies are unavailable.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Check for optional polars dependency
try:
    import polars as pl  # type: ignore
    POLARS_AVAILABLE = True
except ImportError:
    pl = None
    POLARS_AVAILABLE = False

logger = logging.getLogger("pst_email_extractor.core.analysis")


@dataclass(slots=True)
class PstHealth:
    total_emails: int
    folder_count: int
    corrupted_samples: int
    sampled: int
    estimated_size_mb: float

    @property
    def health_score(self) -> float:
        if self.sampled <= 0:
            return 100.0
        ratio = max(0.0, 1.0 - (self.corrupted_samples / max(1, self.sampled)))
        return round(ratio * 100.0, 1)


def _count_tree(root_folder: Any) -> tuple[int, int]:
    """Return (message_count, folder_count) by traversing the tree defensively."""
    total_msgs = 0
    folders = 0
    stack = [root_folder]
    while stack:
        folder = stack.pop()
        folders += 1
        try:
            sub_count = getattr(folder, "number_of_sub_messages", None)
            if sub_count is None:
                raise AttributeError
            total_msgs += int(sub_count)
        except Exception:
            with contextlib.suppress(Exception):
                total_msgs += sum(1 for _ in folder.sub_messages)
        try:
            for sub in folder.sub_folders:
                stack.append(sub)
        except Exception:
            continue
    return total_msgs, max(0, folders - 1)  # exclude the root when possible


def analyze_pst_health(pst_path: Path, *, sample_limit: int = 200) -> PstHealth:
    """
    Produce a quick health overview:
    - total_emails: estimated total messages
    - folder_count: number of folders
    - corrupted_samples: samples that raised while accessing basic fields
    - sampled: number of messages sampled (up to sample_limit)
    - estimated_size_mb: file size in MB
    """
    pst_path = pst_path.expanduser().resolve()
    size_mb = round(pst_path.stat().st_size / (1024 * 1024), 2) if pst_path.exists() else 0.0

    if not pst_path.exists():
        return PstHealth(total_emails=0, folder_count=0, corrupted_samples=0, sampled=0, estimated_size_mb=size_mb)

    try:  # Optional dependency
        import pypff  # type: ignore
    except Exception:
        return PstHealth(total_emails=0, folder_count=0, corrupted_samples=0, sampled=0, estimated_size_mb=size_mb)

    total = 0
    folders = 0
    corrupted = 0
    sampled = 0

    file_obj = pypff.file()
    try:
        file_obj.open(str(pst_path))
        root = file_obj.root_folder
        if root is None:
            return PstHealth(0, 0, 0, 0, size_mb)

        total, folders = _count_tree(root)

        # Sample a small number of messages across the first folders
        stack = [root]
        while stack and sampled < sample_limit:
            folder = stack.pop()
            try:
                for message in folder.sub_messages:
                    if sampled >= sample_limit:
                        break
                    try:
                        _ = getattr(message, "subject", None)
                        _ = getattr(message, "delivery_time", None)
                        _ = getattr(message, "number_of_attachments", 0)
                    except Exception:
                        corrupted += 1
                    sampled += 1
            except Exception:
                # If the folder is unreadable, approximate that all messages are corrupted
                try:
                    maybe = int(getattr(folder, "number_of_sub_messages", 0))
                except Exception:
                    maybe = 0
                remaining = max(0, min(sample_limit - sampled, maybe))
                corrupted += remaining
                sampled += remaining

            try:
                for sub in folder.sub_folders:
                    stack.append(sub)
            except Exception:
                pass
    finally:
        with contextlib.suppress(Exception):
            file_obj.close()

    return PstHealth(total_emails=total, folder_count=folders, corrupted_samples=corrupted, sampled=sampled, estimated_size_mb=size_mb)

"""
Memory-efficient address and host analysis using Polars.

This module provides optimized implementations for analyzing email addresses
and transport hosts from PST files using Polars DataFrames for better memory
efficiency and performance with large datasets.
"""

def _extract_address_rows(email: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return normalised address rows for aggregation."""
    rows: list[dict[str, str]] = []

    def add_address(candidate: Any, role: str) -> None:
        if not candidate:
            return
        text = str(candidate).strip()
        if not text or text.lower() == "none":
            return
        normalised = text.lower()
        domain = normalised.split("@", 1)[1] if "@" in normalised else ""
        local = normalised.split("@", 1)[0] if "@" in normalised else normalised
        rows.append(
            {
                "address": normalised,
                "role": role,
                "domain": domain,
                "name": local,
            }
        )

    add_address(email.get("Sender_Email", ""), "sender")
    add_address(email.get("Reply_To", ""), "reply-to")
    add_address(email.get("Return_Path", ""), "return-path")

    for field, role in (("To", "to"), ("CC", "cc"), ("BCC", "bcc")):
        recipients = email.get(field, "")
        if isinstance(recipients, str):
            candidates = [part.strip() for part in recipients.replace(",", ";").split(";")]
        elif isinstance(recipients, list | tuple | set):
            candidates = [str(part).strip() for part in recipients]
        else:
            candidates = []
        for candidate in candidates:
            if candidate:
                add_address(candidate, role)

    return rows


def _extract_host_values(email: Mapping[str, Any]) -> list[str]:
    """Return normalised received host strings for aggregation."""
    received_hops = email.get("Received_Hops", [])
    if isinstance(received_hops, str):
        hops_iter = [received_hops]
    elif isinstance(received_hops, list | tuple | set):
        hops_iter = received_hops
    else:
        hops_iter = []

    hosts: list[str] = []
    for host in hops_iter:
        if host and isinstance(host, str):
            hosts.append(host.lower().strip())
    return hosts


def analyze_addresses_with_polars(
    email_records: Iterable[Mapping[str, Any]],
    progress_callback=None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Analyze email addresses and hosts using Polars for memory efficiency.

    This implementation uses Polars DataFrames to efficiently aggregate and
    process large volumes of email data without excessive memory usage.

    Args:
        email_records: Iterable of email record dictionaries
        progress_callback: Optional progress reporting function

    Returns:
        Dictionary containing 'addresses' and 'hosts' lists
    """
    if not POLARS_AVAILABLE:
        raise ImportError("Polars is required for this analysis method")

    logger.info("Starting Polars-based address analysis")

    CHUNK_SIZE = 5000  # Process emails in chunks to limit memory usage

    address_chunks: list[pl.DataFrame] = []
    host_chunks: list[pl.DataFrame] = []

    current_address_rows: list[dict[str, Any]] = []
    current_host_rows: list[dict[str, Any]] = []

    # Process records in chunks to limit memory usage
    for email_idx, email in enumerate(email_records):
        if progress_callback and email_idx % 1000 == 0:
            progress_callback(email_idx, 0, "Analyzing addresses")

        current_address_rows.extend(_extract_address_rows(email))
        current_host_rows.extend({"host": host} for host in _extract_host_values(email))

        # Process chunk when it reaches the limit
        if len(current_address_rows) >= CHUNK_SIZE:
            if current_address_rows:
                address_chunks.append(pl.DataFrame(current_address_rows))
            if current_host_rows:
                host_chunks.append(pl.DataFrame(current_host_rows))
            current_address_rows = []
            current_host_rows = []

    # Process remaining rows
    if current_address_rows:
        address_chunks.append(pl.DataFrame(current_address_rows))
    if current_host_rows:
        host_chunks.append(pl.DataFrame(current_host_rows))

    if not address_chunks and not host_chunks:
        return {"addresses": [], "hosts": []}

    if address_chunks:
        # Concatenate all address chunks into a single DataFrame
        addresses_df = address_chunks[0] if len(address_chunks) == 1 else pl.concat(address_chunks, how="vertical")

        # Group by address and aggregate
        addresses_agg = (
            addresses_df
            .group_by("address")
            .agg([
                pl.col("role").unique().alias("roles"),
                pl.col("domain").unique().alias("domains"),
                pl.col("name").unique().alias("names"),
                pl.len().alias("count"),
            ])
            .sort("address")
        )

        # Convert to the expected format
        addresses = []
        for row in addresses_agg.iter_rows(named=True):
            addresses.append({
                "Address": row["address"],
                "Count": row["count"],
                "Roles": sorted(row["roles"]),
                "Domains": sorted(row["domains"]),
                "Names": sorted(row["names"]),
            })
    else:
        addresses = []

    if host_chunks:
        # Concatenate all host chunks into a single DataFrame
        hosts_df = host_chunks[0] if len(host_chunks) == 1 else pl.concat(host_chunks, how="vertical")

        # Group by host and count
        hosts_agg = (
            hosts_df
            .group_by("host")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        # Convert to the expected format
        hosts = []
        for row in hosts_agg.iter_rows(named=True):
            hosts.append({
                "Host": row["host"],
                "Count": row["count"],
            })
    else:
        hosts = []

    logger.info(f"Analysis complete: {len(addresses)} addresses, {len(hosts)} hosts")
    return {"addresses": addresses, "hosts": hosts}


def analyze_addresses_fallback(
    email_records: Iterable[Mapping[str, Any]],
    progress_callback=None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Fallback implementation using standard Python collections.

    This is used when Polars is not available or for comparison purposes.
    """
    from collections import Counter

    logger.info("Using fallback address analysis (Polars not available)")

    address_index: dict[str, dict] = {}
    host_counter: Counter[str] = Counter()

    for i, email in enumerate(email_records):
        if progress_callback and i % 1000 == 0:
            progress_callback(i, 0, "Analyzing addresses")

        for row in _extract_address_rows(email):
            addr = row["address"]
            record = address_index.setdefault(
                addr,
                {
                    "Address": addr,
                    "Count": 0,
                    "Roles": set(),
                    "Domains": set(),
                    "Names": set(),
                },
            )
            record["Count"] += 1
            record["Roles"].add(row["role"])
            if row["domain"]:
                record["Domains"].add(row["domain"])
            if row["name"]:
                record["Names"].add(row["name"])

        # Count hosts
        for host in _extract_host_values(email):
            host_counter[host] += 1

    # Format addresses
    addresses = []
    for record in address_index.values():
        addresses.append({
            "Address": record["Address"],
            "Count": record["Count"],
            "Roles": sorted(record["Roles"]),
            "Domains": sorted(record["Domains"]),
            "Names": sorted(record["Names"]),
        })

    # Format hosts
    hosts = [{"Host": host, "Count": count} for host, count in host_counter.most_common()]

    logger.info(f"Fallback analysis complete: {len(addresses)} addresses, {len(hosts)} hosts")
    return {"addresses": addresses, "hosts": hosts}


def analyze_addresses(
    email_records: Iterable[Mapping[str, Any]],
    progress_callback=None,
    use_polars: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """
    Analyze email addresses and transport hosts with automatic fallback.

    Uses Polars for memory-efficient processing when available, otherwise
    falls back to standard Python collections.

    Args:
        email_records: Iterable of email record dictionaries
        progress_callback: Optional progress reporting function
        use_polars: Whether to prefer Polars implementation

    Returns:
        Dictionary containing 'addresses' and 'hosts' lists
    """
    if use_polars and POLARS_AVAILABLE:
        try:
            return analyze_addresses_with_polars(email_records, progress_callback)
        except Exception as e:
            logger.warning(f"Polars analysis failed, falling back to standard implementation: {e}")
            return analyze_addresses_fallback(email_records, progress_callback)
    else:
        return analyze_addresses_fallback(email_records, progress_callback)
