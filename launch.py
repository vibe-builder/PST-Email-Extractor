#!/usr/bin/env python3
"""
Smart launcher for PST Email Extractor.
Detects if running in terminal or desktop environment and launches appropriately.
"""

import os
import sys


def is_terminal_environment():
    """Detect if running from terminal with arguments."""
    # if arguments provided, assume CLI mode
    if len(sys.argv) > 1:
        return True

    # check if stdin is a terminal
    if not sys.stdin.isatty():
        return False

    return bool(os.environ.get("TERM"))


def launch_cli():
    """Launch CLI application."""
    try:
        from pst_email_extractor.cli.app import app as cli_app

        cli_app()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def launch_gui():
    """Launch GUI application."""
    try:
        from pst_email_extractor.gui import launch_gui as gui_launch

        gui_launch()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main launcher logic."""
    # check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("""
PST Email Extractor - Launcher

Usage:
  python launch.py              # Opens GUI
  python launch.py [options]    # CLI mode with options

GUI Mode (no arguments):
  - Visual interface 
  - Drag-and-drop support
  - Progress indicators

CLI Mode (with arguments):
  -i, --pst      Input PST file
  -o, --output   Output directory
  -j, --json     Export to JSON
  -c, --csv      Export to CSV
  --mode         extract (default) or addresses
  --deduplicate  Skip duplicate messages
  --extract-attachments  Persist attachments when extracting

Examples:
  python launch.py
  python launch.py -i emails.pst -o output/ -j -c

For detailed help:
  pst-email-extractor --help
        """)
        return

    # decide which mode to launch
    if len(sys.argv) == 1:
        print("Launching GUI mode...")
        launch_gui()
    else:
        print("Launching CLI mode...")
        launch_cli()


if __name__ == "__main__":
    main()

