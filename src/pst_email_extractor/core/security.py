"""
Shared security-related constants and helpers.

These values are imported by parsers/backends to keep behavior consistent
and avoid drift across modules.
"""

from __future__ import annotations

# Hard limit per-attachment to prevent memory exhaustion/DoS
MAX_ATTACHMENT_SIZE: int = 100 * 1024 * 1024  # 100MB

# Streaming chunk size when reading large attachments
CHUNK_SIZE: int = 1_048_576  # 1MB


