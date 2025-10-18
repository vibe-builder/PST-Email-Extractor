# PST Email Extractor

This project was developed entirely with AI assistance in Cursor. The goal is a professional, reliable tool for extracting email data from Microsoft Outlook PST files. It pairs a modern desktop interface with a command-line workflow, exports to CSV and JSON via streaming writers (so huge PSTs are safe), and runs on Windows, Linux, or macOS. A standalone Windows executable is available for users who prefer an installer-free experience.

## Setup and Installation

- Requires Python 3.6+ and the dependencies listed in `requirements.txt`, including `pypff` for PST parsing.
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Launch the GUI for an intuitive workflow:
  ```bash
  python gui.py
  ```
  Select a PST file, choose an output folder, pick JSON and/or CSV, then start extraction.
- Use the CLI for automation (streams emails to disk as they’re parsed):
  ```bash
  python main.py -i <pst_file> -o <output_dir> [-j] [-c]
  ```
- Let the smart launcher choose automatically:
  ```bash
  python launch.py            # GUI when no arguments are given
  python launch.py -i file.pst  # CLI when arguments are present
  ```
- Build a Windows executable that bundles Python and dependencies:
  ```bash
  python build_executable.py
  ```
- Before running, make sure Outlook is closed and the PST file is accessible to avoid permission issues.

## Features and Use Cases

- Extracts 17 key fields per email (sender, subject, body, metadata, etc.) with strong error handling and live progress updates.
- Stream-based exporters write JSON/CSV incrementally, keeping memory usage low even for very large PST archives.
- Typical processing times:
  - Small files (<1,000 emails): roughly 1–2 minutes
  - Medium files (1,000–10,000 emails): about 5–10 minutes
  - Large files (>10,000 emails): around 10–30 minutes
- Outputs timestamped, UTF-8 encoded CSV/JSON files suitable for Excel, databases, or scripted analysis.
- Ideal scenarios include email archiving, migration, e-discovery, compliance reviews, and data analysis.
- Attachment contents are not extracted (only counts are recorded), and corrupted PST data may result in skipped messages—see troubleshooting guidance for dependency or data issues.
