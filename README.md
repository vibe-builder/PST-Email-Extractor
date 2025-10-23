# PST Email Extractor

Extract email content from Outlook PST archives with CLI and GUI workflows.

## Features

- **Multiple Export Formats**: JSON, CSV, EML, MBOX
- **Health Checking**: Pre-flight PST file analysis
- **Progress Tracking**: Real-time ETA and throughput metrics
- **Smart Attachment Handling**: Conflict resolution with unique naming
- **AI Processing**: PII sanitization and text polishing
- **Batch Processing**: Format-specific optimization with dynamic RAM-based sizing
- **GUI & CLI**: User-friendly interfaces for all use cases

## Installation

### Base Installation
```bash
pip install -e .
```

### Optional Feature Sets

Install additional capabilities as needed:

```bash
# Performance optimizations (dynamic batching, memory monitoring)
pip install -e ".[perf]"

# Attachment content extraction (PDF, DOCX, OCR)
pip install -e ".[attachments]"

# AI text processing (PII sanitization, grammar correction)
pip install -e ".[ai]"

# All features
pip install -e ".[perf,attachments,ai]"
```

**Performance Features** (`[perf]`):
- `psutil` - Dynamic batch sizing based on available RAM
- Memory monitoring and profiling utilities

**Attachment Features** (`[attachments]`):
- `python-magic` - Advanced MIME type detection
- `PyMuPDF` - PDF text extraction
- `pytesseract` - OCR for images and scanned documents
- `mammoth` - DOCX text extraction
- `chardet` - Character encoding detection

**AI Features** (`[ai]`):
- `symspellpy` - Spell correction
- `language-tool-python` - Grammar correction
- `transformers` + `onnxruntime` - Neural text polishing

## Usage

### CLI

**Basic extraction:**
```bash
python launch.py extract --pst file.pst --output ./exports --format json csv
```

**With performance optimizations:**
```bash
# Enable compression to save 50-70% disk space
python launch.py extract --pst file.pst --output ./exports --json --compress

# Extract attachment content with OCR
python launch.py extract --pst file.pst --output ./exports --json \
  --extract-attachment-content --enable-ocr

# All features enabled
python launch.py extract --pst file.pst --output ./exports --json --csv \
  --compress --extract-attachment-content --ai-sanitize
```

### GUI
```bash
python launch.py gui
```

### Performance Tuning

For detailed performance optimization guide, see **[PERFORMANCE.md](PERFORMANCE.md)**.

**Quick tips:**
- Install `psutil` for dynamic batch sizing: `pip install psutil`
- Use `--compress` flag for 50-70% disk space savings
- Parallel folder processing automatically enabled for multi-folder PST files
- Expect 2-5x speedup on 4+ core systems with multiple folders

## Requirements

- Python 3.11-3.12
- libpff (PST parsing library)

## Progress callbacks

Both callback shapes are supported throughout the pipeline for compatibility:
- Modern: a single `ProgressUpdate` object (`current`, `total`, `message`).
- Legacy: positional `(current: int, total: int, message: str)`.

When integrating custom progress reporting, prefer the modern object form; the system adapts to legacy callables where possible.

## License

MIT
