#!/usr/bin/env python3
"""
PST Email Extractor - Extract email headers and body content from Outlook PST files.
Exports data to CSV or JSON format.
"""

import sys
import argparse
import os
from contextlib import ExitStack
from pathlib import Path

# TODO: Implement these modules or install required dependencies
try:
    from pst_tools import pst_parser, id_generator, banner, exporters
except ImportError:
    print("[!] Missing dependencies (pst_tools module)")
    print("[*] Required modules: pst_parser, id_generator, banner")
    sys.exit(1)

try:
    import unicodecsv
except ImportError:
    print("[!] Missing unicodecsv package")
    print("[*] Install with: pip install unicodecsv")
    sys.exit(1)

def validate_paths(pst_path, output_dir):
    """
    Validate input PST file and output directory exist.
    
    Args:
        pst_path: Path to input PST file
        output_dir: Path to output directory
        
    Returns:
        Boolean indicating if paths are valid
    """
    if not pst_path or not os.path.isfile(pst_path):
        print(f"[!] PST file not found: {pst_path}")
        return False
    
    if not output_dir or not os.path.isdir(output_dir):
        print(f"[!] Output directory not found: {output_dir}")
        return False
    
    return True


def main():
    """Parse arguments and orchestrate PST extraction workflow."""
    parser = argparse.ArgumentParser(
        description='Extract email headers and body content from PST files',
        epilog='Example: python main.py -i emails.pst -o output/ -j -c'
    )
    parser.add_argument(
        '-i', '--pst',
        required=True,
        help='Input PST file path'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output directory for exported files'
    )
    parser.add_argument(
        '-j', '--json',
        action='store_true',
        help='Export data to JSON format'
    )
    parser.add_argument(
        '-c', '--csv',
        action='store_true',
        help='Export data to CSV format'
    )
    parser.add_argument(
        '--log-file',
        help='Path to write the parser log file (default: pst_parser.log)'
    )
    
    args = parser.parse_args()
    
    # validate paths before processing
    if not validate_paths(args.pst, args.output):
        sys.exit(1)
    
    # ensure at least one output format is selected
    if not args.json and not args.csv:
        print("[!] Please specify at least one output format (-j for JSON, -c for CSV)")
        sys.exit(1)
    
    # configure logging destination
    log_path = pst_parser.configure_logging(args.log_file)
    print(f"[*] Logs will be saved to: {log_path}")
    
    # generate unique identifier for output files
    unique_id = id_generator.generate()
    output_dir = Path(args.output).resolve()
    json_path = output_dir / f"pst_{unique_id}.json"
    csv_path = output_dir / f"pst_{unique_id}.csv"
    
    def progress_callback(current, total, status):
        """Render determinate progress in the CLI."""
        if total:
            percent = (current / total) * 100 if total else 0
            message = f"\r[*] {status} ({current}/{total}, {percent:5.1f}%)"
        else:
            message = f"\r[*] {status}"
        print(message, end='', flush=True)
        if total and current >= total:
            print()
    
    # parse PST file and extract email data
    print(f"[*] Processing PST file: {args.pst}")
    exported_paths = []
    email_count = 0

    try:
        with ExitStack() as stack:
            csv_writer = stack.enter_context(exporters.CSVStreamWriter(csv_path)) if args.csv else None
            json_writer = stack.enter_context(exporters.JSONStreamWriter(json_path)) if args.json else None

            if csv_writer:
                exported_paths.append(csv_writer.path)
            if json_writer:
                exported_paths.append(json_writer.path)

            for email in pst_parser.iter_emails(args.pst, progress_callback=progress_callback):
                email_count += 1
                if csv_writer:
                    csv_writer.write(email)
                if json_writer:
                    json_writer.write(email)
    except Exception as exc:
        print(f"[!] Extraction failed: {exc}")
        sys.exit(1)
    finally:
        print()

    if email_count == 0:
        print("[!] No email data extracted from PST file")
        sys.exit(1)

    print(f"[+] Extracted {email_count} emails")

    for path in exported_paths:
        print(f"[+] Export complete: {path}")


if __name__ == "__main__":
    # display banner/startup info
    banner.display()
    main()

