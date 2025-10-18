#!/usr/bin/env python3
"""
Smart launcher for PST Email Extractor.
Detects if running in terminal or desktop environment and launches appropriately.
"""

import sys
import os


def is_terminal_environment():
    """Detect if running from terminal with arguments."""
    # if arguments provided, assume CLI mode
    if len(sys.argv) > 1:
        return True

    # check if stdin is a terminal
    if not sys.stdin.isatty():
        return False

    # check if TERM environment variable exists (usually in terminal)
    if os.environ.get('TERM'):
        return True

    # default to GUI for Windows desktop users
    return False


def launch_gui():
    """Launch GUI application."""
    try:
        from gui import PSTExtractorGUI
        from PyQt6.QtWidgets import QApplication

        app = QApplication(sys.argv)
        window = PSTExtractorGUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("\nFalling back to CLI mode...")
        print("Usage: python main.py -i <pst_file> -o <output_dir> -j -c")
        sys.exit(1)


def launch_cli():
    """Launch CLI application."""
    try:
        from main import main as cli_main
        cli_main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main launcher logic."""
    # check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print("""
PST Email Extractor - Smart Launcher

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

Examples:
  python launch.py
  python launch.py -i emails.pst -o output/ -j -c

For detailed help:
  python main.py --help  (CLI)
  See README.md (GUI)
        """)
        return

    # decide which mode to launch
    if is_terminal_environment() and len(sys.argv) > 1:
        print("Launching CLI mode...")
        launch_cli()
    else:
        print("Launching GUI mode...")
        launch_gui()


if __name__ == "__main__":
    main()

