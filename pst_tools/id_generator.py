"""
Generate unique identifiers for output files.
"""

import time
import random
import string


def generate(length=8):
    """
    Generate a unique identifier combining timestamp and random chars.
    
    Args:
        length: Length of random suffix (default 8)
        
    Returns:
        String containing timestamp + random characters
    """
    timestamp = str(int(time.time()))
    random_suffix = ''.join(
        random.choices(string.ascii_lowercase + string.digits, k=length)
    )
    return f"{timestamp}_{random_suffix}"

