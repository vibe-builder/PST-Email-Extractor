"""
Command line interface powered by Typer.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

import typer

from pst_email_extractor import __version__
from pst_email_extractor.core import ADDRESS_MODES, SUPPORTED_FORMATS
from pst_email_extractor.core.attachment_processor import AttachmentContentOptions
from pst_email_extractor.core.models import ExtractionConfig, ProgressUpdate
from pst_email_extractor.core.services import run_extraction

app = typer.Typer(help="Extract email content from Outlook PST archives.", no_args_is_help=True)


def _validate_paths(pst_path: Path) -> None:
    if not pst_path.is_file():
        raise typer.BadParameter(f"PST file not found: {pst_path}", param_hint="--pst")
    # Output directory will be auto-created by the extraction process
    # No need to validate existence here


def _render_progress(update: ProgressUpdate) -> None:
    if update.total:
        percent = (update.current / update.total) * 100 if update.total else 0
        message = f"[{update.current}/{update.total}, {percent:5.1f}%] {update.message}"
    else:
        message = update.message
    typer.echo(f"\r[*] {message}", nl=False)
    if update.total and update.current >= update.total:
        typer.echo("")


def _run_cli(
    *,  # noqa: D401
    pst_path: Path,
    output_dir: Path,
    mode: str,
    formats: list[str],
    deduplicate: bool,
    extract_attachments: bool,
    attachments_dir: Path | None,
    log_file: Path | None,
    html_index: bool,
    open_html_index: bool,
    ai_sanitize: bool,
    ai_polish: bool,
    ai_language: str,
    ai_neural_model_dir: Path | None,
    extract_attachment_content: bool,
    attachment_content_options: AttachmentContentOptions | None,
    compress: bool = False,
) -> None:
    selected_formats = [fmt.lower() for fmt in formats]

    if not selected_formats:
        raise typer.BadParameter("Please specify at least one output format (--json/--csv/--format).")

    if mode == "addresses" and html_index:
        raise typer.BadParameter("HTML index generation is only available in extract mode.")

    attachments_needed = any(fmt in {"eml", "mbox"} for fmt in selected_formats)
    effective_extract_attachments = extract_attachments or attachments_needed
    if attachments_needed and not extract_attachments:
        typer.echo("[*] Attachments will be extracted automatically to support EML/MBOX output.", err=True)

    config = ExtractionConfig(
        pst_path=pst_path,
        output_dir=output_dir,
        formats=selected_formats,
        mode=mode,  # type: ignore[arg-type]
        deduplicate=deduplicate,
        extract_attachments=effective_extract_attachments,
        attachments_dir=attachments_dir,
        log_file=log_file,
        html_index=html_index or open_html_index,
        ai_sanitize=ai_sanitize,
        ai_polish=ai_polish,
        ai_language=ai_language,
        ai_neural_model_dir=ai_neural_model_dir,
        extract_attachment_content=extract_attachment_content,
        attachment_content_options=attachment_content_options,
        compress=compress,
    )

    try:
        result = run_extraction(config, progress_callback=_render_progress)
    except Exception as exc:  # pragma: no cover - surfaced to CLI
        typer.echo(f"\n[!] Extraction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo("")
    typer.echo(f"[*] Logs saved to: {result.log_path}")

    if result.mode == "extract":
        typer.echo(f"[+] Extracted {result.email_count} emails")
        if result.attachments_dir and result.attachments_saved:
            typer.echo(f"[+] Attachments saved to: {result.attachments_dir} ({result.attachments_saved} files)")
        elif effective_extract_attachments:
            typer.echo("[!] No attachments were saved (none detected)")

        if result.html_index_path:
            typer.echo(f"[+] HTML index generated: {result.html_index_path}")
            if open_html_index:
                webbrowser.open(result.html_index_path.as_uri())
    else:
        typer.echo(f"[+] Identified {result.address_count} unique email addresses")
        if result.host_count:
            typer.echo(f"[+] Aggregated {result.host_count} transport hosts from Received headers")

    for path in result.exported_paths:
        typer.echo(f"[+] Export complete: {path}")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(
        False,
        "--version",
        help="Display package version and exit.",
        is_eager=True,
    ),
) -> None:
    """Global options for the CLI."""
    if version:
        typer.echo(__version__)
        raise typer.Exit()


@app.command()
def extract(
    pst: Path = typer.Option(..., "--pst", "-i", help="Input PST file path."),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for exported files."),
    mode: str = typer.Option(
        "extract",
        "--mode",
        case_sensitive=False,
        help="Operation mode: extract full messages or analyze unique addresses.",
        show_default=True,
    ),
    json: bool = typer.Option(False, "--json", "-j", help="Export data to JSON format."),
    csv: bool = typer.Option(False, "--csv", "-c", help="Export data to CSV format."),
    compress: bool = typer.Option(False, "--compress", "-z", help="Compress JSON/CSV exports with gzip (saves 50-70%% disk space)."),
    formats: list[str] = typer.Option(
        [],
        "--format",
        help=f"Additional export format ({', '.join(sorted(SUPPORTED_FORMATS))}). Repeatable.",
    ),
    deduplicate: bool = typer.Option(False, "--deduplicate", help="Skip duplicate messages based on message hash."),
    extract_attachments: bool = typer.Option(
        False,
        "--extract-attachments",
        help="Save attachments alongside exported messages.",
    ),
    attachments_dir: Path | None = typer.Option(
        None,
        "--attachments-dir",
        help="Directory for extracted attachments (defaults to <output>/attachments_<run_id>).",
    ),
    log_file: Path | None = typer.Option(None, "--log-file", help="Path to write the parser log file."),
    html_index: bool = typer.Option(
        False,
        "--html-index",
        help="Generate a lightweight HTML index after extraction (extract mode only).",
    ),
    open_html_index: bool = typer.Option(
        False,
        "--open-html-index",
        help="Automatically open the HTML index in the default browser.",
    ),
    ai_sanitize: bool = typer.Option(
        False,
        "--ai-sanitize",
        help="Mask common PII (emails, phones, URLs, IPs) in text fields.",
    ),
    ai_polish: bool = typer.Option(
        False,
        "--ai-polish",
        help="Polish text using local tools (spell + grammar).",
    ),
    ai_language: str = typer.Option(
        "en-US",
        "--ai-language",
        help="Language code for polishing (e.g., en-US).",
        show_default=True,
    ),
    ai_neural_model_dir: Path | None = typer.Option(
        None,
        "--ai-neural-model-dir",
        help="Optional directory of a small ONNX seq2seq model for rewriting.",
    ),
    extract_attachment_content: bool = typer.Option(
        False,
        "--extract-attachment-content",
        help="Extract text content from attachments (PDF, DOCX, images via OCR, etc.)",
    ),
    attachment_ocr: bool = typer.Option(
        True,
        "--attachment-ocr/--no-attachment-ocr",
        help="Enable OCR for images and scanned documents",
    ),
    ocr_languages: str = typer.Option(
        "eng",
        "--ocr-languages",
        help="Comma-separated list of OCR languages (e.g., 'eng,spa,fra')",
    ),
    max_attachment_size: int = typer.Option(
        50,
        "--max-attachment-size-mb",
        help="Maximum attachment size to process for content extraction (MB)",
    ),
) -> None:
    """Extract emails or analyze address metadata from a PST archive."""
    mode = mode.lower()
    if mode not in {"extract", "addresses"}:
        raise typer.BadParameter("Mode must be either 'extract' or 'addresses'.", param_hint="--mode")

    _validate_paths(pst)

    selected_formats: list[str] = []
    if json:
        selected_formats.append("json")
    if csv:
        selected_formats.append("csv")
    selected_formats.extend(formats)

    # Create attachment content options if extraction is enabled
    attachment_content_options = None
    if extract_attachment_content:
        attachment_content_options = AttachmentContentOptions(
            enable_ocr=attachment_ocr,
            ocr_languages=[lang.strip() for lang in ocr_languages.split(",") if lang.strip()],
            max_file_size_mb=max_attachment_size,
        )

    _run_cli(
        pst_path=pst,
        output_dir=output,
        mode=mode,
        formats=selected_formats,
        deduplicate=deduplicate,
        extract_attachments=extract_attachments,
        attachments_dir=attachments_dir,
        log_file=log_file,
        html_index=html_index,
        open_html_index=open_html_index,
        ai_sanitize=ai_sanitize,
        ai_polish=ai_polish,
        ai_language=ai_language,
        ai_neural_model_dir=ai_neural_model_dir,
        extract_attachment_content=extract_attachment_content,
        attachment_content_options=attachment_content_options,
        compress=compress,
    )


@app.command()
def addresses(
    pst: Path = typer.Option(..., "--pst", "-i", help="Input PST file path."),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for exported files."),
    json: bool = typer.Option(False, "--json", "-j", help="Export address analysis to JSON."),
    csv: bool = typer.Option(False, "--csv", "-c", help="Export address analysis to CSV."),
    formats: list[str] = typer.Option(
        [],
        "--format",
        help=f"Additional export format ({', '.join(sorted(ADDRESS_MODES))}). Repeatable.",
    ),
    deduplicate: bool = typer.Option(False, "--deduplicate", help="Deduplicate addresses using message hash."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Path to write the parser log file."),
) -> None:
    """Analyze unique addresses from a PST archive."""
    _validate_paths(pst)
    selected_formats: list[str] = []
    if json:
        selected_formats.append("json")
    if csv:
        selected_formats.append("csv")
    selected_formats.extend(formats)
    invalid = [fmt for fmt in selected_formats if fmt not in ADDRESS_MODES]
    if invalid:
        raise typer.BadParameter(
            "Address analysis supports JSON or CSV outputs only.",
            param_hint="--format",
        )
    _run_cli(
        pst_path=pst,
        output_dir=output,
        mode="addresses",
        formats=selected_formats or ["json"],
        deduplicate=deduplicate,
        extract_attachments=False,
        attachments_dir=None,
        log_file=log_file,
        html_index=False,
        open_html_index=False,
    )


if __name__ == "__main__":
    app()
