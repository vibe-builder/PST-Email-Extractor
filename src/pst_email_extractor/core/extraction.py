"""
Extraction orchestration logic.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import ExitStack, suppress
from pathlib import Path

from pst_email_extractor import exporters, id_generator
from pst_email_extractor.core.backends import DependencyError, PypffBackend
from pst_email_extractor.core.models import (
    ExtractionConfig,
    ExtractionResult,
    ProgressCallback,
    ProgressUpdate,
)
from pst_email_extractor.logging import configure_logging, get_logger

from .analysis import analyze_pst_health

# Try to import psutil for dynamic batch sizing
try:
    import psutil  # type: ignore[import-untyped]
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Format-specific batching configuration (moved to module level for performance)
FORMAT_BATCH_SIZES = {"json": 500, "csv": 1000, "eml": 50, "mbox": 100}

# Supported export formats
SUPPORTED_FORMATS = {"json", "csv", "eml", "mbox"}
# Formats supported for address analysis mode
ADDRESS_MODES = {"json", "csv"}

logger = get_logger(__name__)


def _normalise_formats(formats: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported export format requested: {fmt}")
        if fmt_lower not in seen:
            ordered.append(fmt_lower)
            seen.add(fmt_lower)
    return ordered


def _emit_progress(callback: ProgressCallback | None, current: int, total: int, status: str) -> None:
    if not callback:
        return
    update = ProgressUpdate(current=current, total=total, message=status)
    try:
        callback(update)
    except TypeError:
        with suppress(Exception):
            callback(current, total, status)  # type: ignore[misc]
    except Exception:
        pass  # Ignore callback failures


def perform_extraction(
    config: ExtractionConfig,
    progress_callback: ProgressCallback | None = None,
) -> ExtractionResult:
    """Execute extraction or address analysis with the provided configuration."""
    mode = config.mode.lower()
    if mode not in {"extract", "addresses"}:
        raise ValueError(f"Unsupported mode: {mode}")

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = configure_logging(config.log_file)

    selected_formats = _normalise_formats(config.formats)
    if not selected_formats:
        raise ValueError("No export formats selected.")

    if mode == "addresses":
        invalid = [fmt for fmt in selected_formats if fmt not in ADDRESS_MODES]
        if invalid:
            raise ValueError("Address analysis mode supports JSON or CSV exports only.")

    backend = PypffBackend()
    if not backend.is_available():
        raise DependencyError(
            "pypff library not found. Install the Python bindings for libpff "
            "(e.g. pip install libpff-python-ratom==20220304)."
        )
    backend.open(config.pst_path)

    # Pre-flight PST health summary
    health = analyze_pst_health(config.pst_path)
    _emit_progress(
        progress_callback,
        0,
        0,
        f"PST health: messages~{health.total_emails}, folders={health.folder_count}, size={health.estimated_size_mb}MB, health={health.health_score}%",
    )
    # Warn for large PSTs when psutil is unavailable (static batch sizing)
    try:
        size_mb = float(getattr(health, "estimated_size_mb", 0) or 0)
        if not HAS_PSUTIL and size_mb >= 1024:
            logger.warning(
                "Large PST (~%.0f MB) detected and psutil not installed; dynamic batch sizing disabled.\n"
                "Consider installing psutil or using --compress to reduce memory usage.",
                size_mb,
            )
    except Exception:
        pass

    unique_id = id_generator.generate()

    attachments_required = config.extract_attachments or config.attachments_dir is not None or any(
        fmt in {"eml", "mbox"} for fmt in selected_formats
    )
    attachments_path: Path | None = None
    if attachments_required:
        attachments_path = (
            config.attachments_dir.expanduser().resolve()
            if config.attachments_dir
            else output_dir / f"pst_{unique_id}_attachments"
        )
        attachments_path.mkdir(parents=True, exist_ok=True)

    exported_paths: list[Path] = []

    def _addresses_progress(current: int, total: int, status: str) -> None:
        _emit_progress(progress_callback, current, total, status)

    def _normalise_recipients(value: object) -> list[str]:
        """Return a list of normalised recipient strings."""
        if not value:
            return []
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(",", ";").split(";")]
            return [part for part in parts if part]
        if isinstance(value, list | tuple | set):
            parts = [str(part).strip() for part in value if part]
            return [part for part in parts if part]
        return []

    if mode == "addresses":
        try:
            report = backend.analyze_addresses(
                deduplicate=config.deduplicate,
                _progress_callback=_addresses_progress,
            )
        finally:
            backend.close()

        addresses = report.get("addresses", [])
        hosts = report.get("hosts", [])
        if not addresses:
            raise RuntimeError("No addresses discovered in PST file.")

        for fmt in selected_formats:
            if fmt == "json":
                path = output_dir / f"pst_{unique_id}_addresses.json"
                exported_paths.append(exporters.export_address_report_to_json(path, report))
            elif fmt == "csv":
                addresses_path = output_dir / f"pst_{unique_id}_addresses.csv"
                exported_paths.append(exporters.export_addresses_to_csv(addresses_path, addresses))
                if hosts:
                    host_path = output_dir / f"pst_{unique_id}_hosts.csv"
                    exported_paths.append(exporters.export_hosts_to_csv(host_path, hosts))

        return ExtractionResult(
            mode="addresses",
            exported_paths=exported_paths,
            log_path=log_path,
            unique_run_id=unique_id,
            address_count=len(addresses),
            host_count=len(hosts),
        )

    attachments_stats = {"saved": 0}
    email_count = 0
    html_records: list[dict] = []
    html_path: Path | None = None

    progress_state = {"current": 0, "total": 0, "status": ""}

    # Enhanced progress tracker with throughput and ETA
    start_time = None
    last_time = None
    ema_rate = None  # exponential moving average of emails/sec

    def _progress_proxy(current: int, total: int, status: str) -> None:
        nonlocal start_time, last_time, ema_rate
        now = time.monotonic()
        if start_time is None:
            start_time = now
        if last_time is None:
            last_time = now

        # Update EMA rate when progress increases
        elapsed = max(1e-6, now - last_time)
        delta = current - progress_state.get("current", 0)
        inst_rate = (delta / elapsed) if delta > 0 else 0.0
        ema_rate = inst_rate if ema_rate is None else 0.8 * ema_rate + 0.2 * inst_rate

        progress_state.update({"current": current, "total": total, "status": status})

        # Compute ETA
        eta_str = ""
        if total and ema_rate and ema_rate > 0:
            remaining = max(0, total - current)
            seconds = int(remaining / max(1e-6, ema_rate))
            mm = seconds // 60
            ss = seconds % 60
            eta_str = f" | ETA {mm:02d}:{ss:02d}"

        rate_str = f" ({ema_rate:.2f}/sec)" if ema_rate and ema_rate > 0 else ""
        suffix = ""
        if attachments_path:
            suffix = f" | attachments saved: {attachments_stats['saved']}"
        message = f"{status}{rate_str}{eta_str}{suffix}"
        _emit_progress(progress_callback, current, total, message)
        last_time = now

    json_writer = None
    csv_writer = None
    eml_writer = None
    mbox_writer = None

    # Precompute values to avoid repeated calculations
    neural_model_path = str(config.ai_neural_model_dir) if config.ai_neural_model_dir else None
    active_formats = [fmt for fmt in selected_formats if fmt in FORMAT_BATCH_SIZES]
    
    # Dynamic batch sizing based on available RAM
    base_batch_size = min((FORMAT_BATCH_SIZES[fmt] for fmt in active_formats), default=1)
    if HAS_PSUTIL:
        try:
            available_ram_mb = psutil.virtual_memory().available / (1024 ** 2)
            # Scale batch size: 100MB RAM per batch item (conservative estimate)
            # This prevents OOM on memory-constrained systems
            ram_based_limit = max(1, int(available_ram_mb / 100))
            batch_size = min(base_batch_size, ram_based_limit)
            if batch_size < base_batch_size:
                logger.info(f"Adjusted batch size to {batch_size} based on {available_ram_mb:.0f}MB available RAM")
        except Exception:
            batch_size = base_batch_size
    else:
        batch_size = base_batch_size
    
    buffer: list[dict] = []

    with ExitStack() as stack:
        if "json" in selected_formats:
            json_path = output_dir / f"pst_{unique_id}.json"
            json_writer = stack.enter_context(exporters.JSONStreamWriter(
                json_path,
                ai_sanitize=config.ai_sanitize,
                ai_polish=config.ai_polish,
                ai_language=config.ai_language,
                ai_neural_model_dir=neural_model_path,
                compress=config.compress
            ))
            exported_paths.append(json_writer.path)
        if "csv" in selected_formats:
            csv_path = output_dir / f"pst_{unique_id}.csv"
            csv_writer = stack.enter_context(exporters.CSVStreamWriter(
                csv_path,
                ai_sanitize=config.ai_sanitize,
                ai_polish=config.ai_polish,
                ai_language=config.ai_language,
                ai_neural_model_dir=neural_model_path,
                compress=config.compress
            ))
            exported_paths.append(csv_writer.path)
        if "eml" in selected_formats:
            eml_dir = output_dir / f"pst_{unique_id}_eml"
            eml_writer = stack.enter_context(exporters.EMLWriter(
                eml_dir, attachments_path,
                ai_sanitize=config.ai_sanitize,
                ai_polish=config.ai_polish,
                ai_language=config.ai_language,
                ai_neural_model_dir=neural_model_path
            ))
            exported_paths.append(eml_writer.directory)
        if "mbox" in selected_formats:
            mbox_path = output_dir / f"pst_{unique_id}.mbox"
            mbox_writer = stack.enter_context(exporters.MBOXWriter(
                mbox_path, attachments_path,
                ai_sanitize=config.ai_sanitize,
                ai_polish=config.ai_polish,
                ai_language=config.ai_language,
                ai_neural_model_dir=neural_model_path
            ))
            exported_paths.append(mbox_writer.path)

        try:
            def _flush_buffer() -> None:
                nonlocal buffer, email_count
                if not buffer:
                    return
                for email_record in buffer:
                    if csv_writer:
                        csv_writer.write(email_record)
                    if json_writer:
                        json_writer.write(email_record)
                    if eml_writer:
                        eml_writer.write(email_record)
                    if mbox_writer:
                        mbox_writer.write(email_record)
                buffer.clear()

            attachment_content_options = None
            if config.extract_attachment_content:
                if config.attachment_content_options:
                    attachment_content_options = config.attachment_content_options
                else:
                    # Create default options when extract_attachment_content=True but no options provided
                    from .attachment_processor import AttachmentContentOptions
                    attachment_content_options = AttachmentContentOptions()

            for email in backend.iter_messages(
                deduplicate=config.deduplicate,
                attachments_dir=attachments_path,
                progress_callback=_progress_proxy,
                attachment_content_options=attachment_content_options,
            ):
                email_count += 1
                buffer.append(email)
                if len(buffer) >= batch_size:
                    _flush_buffer()

                saved = len(email.get("Attachment_Paths") or [])
                if attachments_path and saved:
                    attachments_stats["saved"] += saved
                    status = progress_state["status"] or f"Processed {email_count} emails"
                    _emit_progress(progress_callback, progress_state["current"], progress_state["total"], status)

                if config.html_index:
                    recipient_values: list[str] = []
                    for field in ("To", "CC", "BCC"):
                        recipient_values.extend(_normalise_recipients(email.get(field)))
                    ", ".join(dict.fromkeys(recipient_values))

                    html_records.append(
                        {
                            "Subject": email.get("Subject", ""),
                            "From": email.get("From", ""),
                            "Sender_Email": email.get("Sender_Email", ""),
                            "To": email.get("To", ""),
                            "CC": email.get("CC", ""),
                            "BCC": email.get("BCC", ""),
                            "Date_Received": email.get("Date_Received"),
                            "Date_Sent": email.get("Date_Sent"),
                            "Attachment_Count": email.get("Attachment_Count", 0),
                            "Attachment_Paths": email.get("Attachment_Paths") or [],
                            "Email_ID": email.get("Email_ID"),
                        }
                    )
            _flush_buffer()
        finally:
            backend.close()

    if config.html_index and html_records:
        attachment_dir = attachments_path if attachments_path else output_dir
        html_path = exporters.generate_html_index(
            html_records,
            output_path=output_dir / f"pst_{unique_id}_index.html",
            attachments_dir=attachment_dir,
        )
        exported_paths.append(html_path)

    return ExtractionResult(
        mode="extract",
        exported_paths=exported_paths,
        log_path=log_path,
        unique_run_id=unique_id,
        email_count=email_count,
        attachments_saved=attachments_stats["saved"],
        attachments_dir=attachments_path,
        html_index_path=html_path,
    )
