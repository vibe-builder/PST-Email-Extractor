"""
Attachment content extraction service for PST email processing.

This module provides automatic extraction of text content from email attachments,
including OCR for images and documents, with support for various file formats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

logger = logging.getLogger(__name__)

# Lazy-loaded optional dependencies for reduced startup time
# Modules are imported on first use to avoid loading heavy libs unnecessarily
HAS_MAGIC = None
HAS_FITZ = None
HAS_TESSERACT = None
HAS_MAMMOTH = None
HAS_CHARDET = None

# Module references (lazy-initialized)
magic = None
fitz = None
pytesseract = None
mammoth = None
chardet = None

def _lazy_import_magic():
    """Lazy import python-magic."""
    global HAS_MAGIC, magic
    if HAS_MAGIC is None:
        try:
            import magic as _magic  # type: ignore
            magic = _magic
            HAS_MAGIC = True
        except ImportError:
            HAS_MAGIC = False
            logger.debug("python-magic not available; MIME detection will be limited")
    return HAS_MAGIC

def _lazy_import_fitz():
    """Lazy import PyMuPDF."""
    global HAS_FITZ, fitz
    if HAS_FITZ is None:
        try:
            import fitz as _fitz  # type: ignore
            fitz = _fitz
            HAS_FITZ = True
        except ImportError:
            HAS_FITZ = False
            logger.debug("PyMuPDF not available; PDF text extraction disabled")
    return HAS_FITZ

def _lazy_import_tesseract():
    """Lazy import pytesseract."""
    global HAS_TESSERACT, pytesseract
    if HAS_TESSERACT is None:
        try:
            import pytesseract as _pytesseract  # type: ignore
            pytesseract = _pytesseract
            HAS_TESSERACT = True
        except ImportError:
            HAS_TESSERACT = False
            logger.debug("pytesseract not available; OCR disabled")
    return HAS_TESSERACT

def _lazy_import_mammoth():
    """Lazy import mammoth."""
    global HAS_MAMMOTH, mammoth
    if HAS_MAMMOTH is None:
        try:
            import mammoth as _mammoth  # type: ignore
            mammoth = _mammoth
            HAS_MAMMOTH = True
        except ImportError:
            HAS_MAMMOTH = False
            logger.debug("mammoth not available; DOCX text extraction disabled")
    return HAS_MAMMOTH

def _lazy_import_chardet():
    """Lazy import chardet."""
    global HAS_CHARDET, chardet
    if HAS_CHARDET is None:
        try:
            import chardet as _chardet  # type: ignore
            chardet = _chardet
            HAS_CHARDET = True
        except ImportError:
            HAS_CHARDET = False
            logger.debug("chardet not available; text encoding detection limited")
    return HAS_CHARDET

ContentType = Literal["text", "image", "document", "archive", "embedded_message", "binary"]
ExtractionMethod = Literal["direct", "ocr", "pdf_text_layer", "docx_native", "docx_to_html", "msg_eml", "charset_detection", "fallback"]


@dataclass
class AttachmentContentOptions:
    """Configuration options for attachment content extraction."""
    enable_ocr: bool = True
    ocr_languages: list[str] = field(default_factory=lambda: ["eng"])
    max_file_size_mb: int = 50
    extract_embedded_messages: bool = True
    supported_types: set[str] = field(default_factory=lambda: {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/csv",
        "text/xml",
        "message/rfc822",
        "application/vnd.ms-outlook",
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/tiff"
    })


@dataclass
class AttachmentMetadata:
    """Structured metadata for a single attachment."""
    index: int
    filename: str
    size: int
    mime_type: str
    content_type: ContentType
    extracted_text: str | None = None
    text_extraction_method: ExtractionMethod | None = None
    page_count: int | None = None
    embedded_message_data: dict | None = None
    error_message: str | None = None


class AttachmentContentExtractor:
    """
    Service for extracting text content from attachment bytes.

    Supports various file formats with automatic MIME detection and
    fallback strategies for content extraction.
    """

    def __init__(self, options: AttachmentContentOptions | None = None):
        self.options = options or AttachmentContentOptions()

        # Configure OCR if available (lazy loaded)
        if _lazy_import_tesseract():
            try:
                # Set OCR languages
                pytesseract.get_tesseract_version()  # Test availability
                self._ocr_config = f'--psm 3 -l {",".join(self.options.ocr_languages)}'
            except Exception as e:
                logger.info(f"Tesseract not properly configured; OCR disabled: {e}")
                self._ocr_config = None
        else:
            self._ocr_config = None

    def detect_mime_type(self, data: bytes, filename: str = "") -> str:
        """Detect MIME type from file content and filename."""
        if _lazy_import_magic():
            try:
                # Use content-based detection first
                mime_type = magic.from_buffer(data[:2048], mime=True)
                if mime_type and mime_type != 'application/octet-stream':
                    return mime_type
            except Exception:
                logger.debug("python-magic detection failed; falling back to extension.")

        # Fallback to filename-based detection (cached for performance)
        return self._mime_from_extension(filename)

    @staticmethod
    @lru_cache(maxsize=256)
    def _mime_from_extension(filename: str) -> str:
        """Cached MIME type lookup by file extension."""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        mime_map = {
            'pdf': 'application/pdf',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt': 'text/plain',
            'csv': 'text/csv',
            'xml': 'text/xml',
            'msg': 'application/vnd.ms-outlook',
            'eml': 'message/rfc822',
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'bmp': 'image/bmp',
            'tiff': 'image/tiff',
            'tif': 'image/tiff'
        }
        return mime_map.get(ext, 'application/octet-stream')
    
    def _determine_content_type(self, mime_type: str) -> ContentType:
        """Map MIME type to content type category."""
        if mime_type.startswith('text/'):
            return "text"
        elif mime_type.startswith('image/'):
            return "image"
        elif mime_type in ('application/pdf',
                          'application/vnd.openxmlformats-officedocument.wordprocessingml.document'):
            return "document"
        elif mime_type in ('application/zip', 'application/x-rar-compressed', 'application/x-7z-compressed'):
            return "archive"
        elif mime_type in ('message/rfc822', 'application/vnd.ms-outlook'):
            return "embedded_message"
        else:
            return "binary"

    def _extract_text_content(self, data: bytes, mime_type: str, filename: str) -> tuple[str | None, ExtractionMethod]:
        """Extract text content from attachment data."""
        content_type = self._determine_content_type(mime_type)

        # Check if this type is supported
        if mime_type not in self.options.supported_types:
            return None, "fallback"

        try:
            if content_type == "text":
                return self._extract_plain_text(data), "direct"

            elif content_type == "document":
                if mime_type == "application/pdf" and _lazy_import_fitz():
                    return self._extract_pdf_text(data)
                elif mime_type.endswith('.document') and _lazy_import_mammoth():  # DOCX
                    return self._extract_docx_text(data), "docx_native"

            elif content_type == "image" and self.options.enable_ocr and _lazy_import_tesseract() and self._ocr_config:
                return self._extract_image_text(data), "ocr"

            elif content_type == "embedded_message":
                # Handled separately in process_attachment
                return None, "msg_eml"

        except Exception as e:
            logger.debug(f"Content extraction failed for {filename}: {e}")

        return None, "fallback"

    def _extract_plain_text(self, data: bytes) -> str:
        """Extract text from plain text files with encoding detection."""
        if _lazy_import_chardet():
            try:
                detected = chardet.detect(data)
                if detected and detected['confidence'] > 0.7:
                    encoding = detected['encoding']
                    return data.decode(encoding, errors='replace')
            except Exception:
                pass

        # Fallback encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                return data.decode(encoding, errors='replace')
            except UnicodeDecodeError:
                continue

        return data.decode('utf-8', errors='replace')

    def _extract_pdf_text(self, data: bytes) -> tuple[str | None, ExtractionMethod]:
        """Extract text from PDF, falling back to OCR if needed."""
        if not _lazy_import_fitz():
            return None, "fallback"

        try:
            doc = fitz.open(stream=data, filetype="pdf")
            text = ""
            has_text = False

            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    has_text = True
                text += page_text + "\n"

            doc.close()

            # If we got minimal text and OCR is enabled, try OCR on images
            if not has_text and self.options.enable_ocr and _lazy_import_tesseract() and self._ocr_config:
                doc = fitz.open(stream=data, filetype="pdf")
                ocr_text = ""
                for page in doc:
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    page_ocr = self._extract_image_text(img_data)
                    if page_ocr:
                        ocr_text += page_ocr + "\n"
                doc.close()

                if ocr_text.strip():
                    return ocr_text.strip(), "ocr"

            return text.strip() if text.strip() else None, "pdf_text_layer"

        except Exception as e:
            logger.debug(f"PDF extraction failed: {e}")
            return None, "fallback"

    def _extract_docx_text(self, data: bytes) -> str | None:
        """Extract text from DOCX files."""
        if not _lazy_import_mammoth():
            return None

        try:
            import io
            result = mammoth.extract_raw_text(io.BytesIO(data))
            return result.value.strip()
        except Exception:
            return None

    def _extract_image_text(self, data: bytes) -> str | None:
        """Extract text from images using OCR."""
        if not _lazy_import_tesseract() or not self._ocr_config:
            return None

        try:
            import io

            import PIL.Image

            img = PIL.Image.open(io.BytesIO(data))
            text = pytesseract.image_to_string(img, config=self._ocr_config)
            return text.strip()
        except Exception as e:
            logger.debug(f"OCR failed: {e}")
            return None

    def process_attachment(self, index: int, filename: str, size: int, data: bytes) -> AttachmentMetadata:
        """Process a single attachment and extract metadata/content."""
        metadata = AttachmentMetadata(
            index=index,
            filename=filename,
            size=size,
            mime_type="application/octet-stream",
            content_type="binary"
        )

        # Skip if too large
        if size > self.options.max_file_size_mb * 1024 * 1024:
            metadata.error_message = f"Attachment too large ({size} bytes > {self.options.max_file_size_mb}MB)"
            return metadata

        try:
            # Detect MIME type
            metadata.mime_type = self.detect_mime_type(data, filename)
            metadata.content_type = self._determine_content_type(metadata.mime_type)

            # Extract text content
            if metadata.content_type == "embedded_message" and self.options.extract_embedded_messages:
                # Handle embedded messages (MSG files)
                embedded_text, method = self._handle_embedded_message(data)
                metadata.extracted_text = embedded_text
                metadata.text_extraction_method = method
            else:
                # Handle other content types
                extracted_text, method = self._extract_text_content(data, metadata.mime_type, filename)
                metadata.extracted_text = extracted_text
                metadata.text_extraction_method = method

                # Get page count for documents
                if metadata.content_type == "document" and metadata.mime_type == "application/pdf" and _lazy_import_fitz():
                    try:
                        doc = fitz.open(stream=data, filetype="pdf")
                        metadata.page_count = len(doc)
                        doc.close()
                    except Exception:
                        pass

        except Exception as e:
            metadata.error_message = str(e)
            logger.debug(f"Attachment processing failed for {filename}: {e}")

        return metadata

    def _handle_embedded_message(self, _data: bytes) -> tuple[str | None, ExtractionMethod]:
        """Handle embedded message attachments (MSG format)."""
        # Best-effort MSG text extraction using extract_msg if available
        try:
            import io

            import extract_msg  # type: ignore[import-untyped]

            with io.BytesIO(_data) as bio:
                msg = extract_msg.Message(bio)
                parts: list[str] = []
                if getattr(msg, 'subject', None):
                    parts.append(str(msg.subject))
                if getattr(msg, 'sender', None):
                    parts.append(str(msg.sender))
                if getattr(msg, 'date', None):
                    parts.append(str(msg.date))
                if getattr(msg, 'body', None):
                    parts.append(str(msg.body))
                text = "\n".join(p.strip() for p in parts if p and str(p).strip())
                return (text if text else None), "msg_text"
        except Exception:
            # Fallback to letting backend convert MSG to EML where supported
            return None, "msg_eml"
