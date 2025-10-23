"""
Simple profiling utilities for performance benchmarking.

This module provides lightweight profiling capabilities to identify bottlenecks
in PST extraction workflows without requiring external dependencies.
"""

from __future__ import annotations

import cProfile
import logging
import pstats
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@contextmanager
def profile_section(section_name: str):
    """Context manager for profiling a code section with timing."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"⏱️  {section_name}: {elapsed:.2f}s")


def timed(func: F) -> F:
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper  # type: ignore


class SimpleProfiler:
    """
    Simple profiler for performance analysis of extraction runs.
    
    Example:
        profiler = SimpleProfiler("extraction_profile.txt")
        with profiler:
            run_extraction(config)
    """
    
    def __init__(self, output_path: str | Path | None = None):
        self.output_path = Path(output_path) if output_path else None
        self.profiler = cProfile.Profile()
        
    def __enter__(self):
        self.profiler.enable()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.disable()
        
        # Print stats to console
        stats = pstats.Stats(self.profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        
        if self.output_path:
            with self.output_path.open('w') as f:
                stats.stream = f
                stats.print_stats(50)  # Top 50 functions
            logger.info(f"Profile saved to {self.output_path}")
        else:
            stats.print_stats(20)  # Top 20 to console


class MemoryMonitor:
    """
    Monitor memory usage during extraction (requires psutil).
    
    Example:
        monitor = MemoryMonitor()
        monitor.start()
        # ... do work ...
        monitor.report()
    """
    
    def __init__(self):
        try:
            import psutil  # type: ignore[import]
            self.psutil = psutil
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("psutil not available; memory monitoring disabled")
        
        self.peak_usage = 0
        self.start_usage = 0
        
    def start(self):
        """Start monitoring memory."""
        if not self.available:
            return
        mem = self.psutil.virtual_memory()
        self.start_usage = mem.used / (1024 ** 2)  # MB
        self.peak_usage = self.start_usage
        
    def update(self):
        """Update peak memory usage."""
        if not self.available:
            return
        mem = self.psutil.virtual_memory()
        current = mem.used / (1024 ** 2)
        self.peak_usage = max(self.peak_usage, current)
        
    def report(self):
        """Log memory usage summary."""
        if not self.available:
            return
        delta = self.peak_usage - self.start_usage
        logger.info(f"Memory: Start={self.start_usage:.1f}MB, Peak={self.peak_usage:.1f}MB, Delta=+{delta:.1f}MB")


# Usage example in CLI:
# python -m cProfile -o profile.stats launch.py extract --pst file.pst --output ./out --json
# python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"

