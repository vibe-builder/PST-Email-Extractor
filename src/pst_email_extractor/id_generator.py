"""
Generate unique identifiers for output files.
"""

from __future__ import annotations

import random
import string
import time


def generate(length: int = 8) -> str:
    """Generate a timestamp-based identifier with a random suffix."""
    # Use integer timestamp for better performance
    timestamp = int(time.time())
    # Pre-compute character set to avoid recreation
    chars = string.ascii_lowercase + string.digits
    random_suffix = "".join(random.choices(chars, k=length))
    return f"{timestamp}_{random_suffix}"
