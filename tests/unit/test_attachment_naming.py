"""
Unit tests for attachment filename conflict resolution.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import Mock

from pst_email_extractor.exporters.eml_writer import EMLWriter


def test_resolve_conflict_name_no_conflict(tmp_path):
    """Test _resolve_conflict_name when no conflict exists."""
    base_dir = tmp_path
    filename = "test.txt"

    result = EMLWriter._resolve_conflict_name(base_dir, filename)
    expected = base_dir / "test.txt"

    assert result == expected
    assert not result.exists()  # Should not create the file


def test_resolve_conflict_name_first_conflict(tmp_path):
    """Test _resolve_conflict_name with first conflict."""
    base_dir = tmp_path
    filename = "test.txt"

    # Create the original file
    original = base_dir / filename
    original.write_text("content")

    result = EMLWriter._resolve_conflict_name(base_dir, filename)

    assert result == base_dir / "test_2.txt"
    assert not result.exists()  # Should not create the file


def test_resolve_conflict_name_multiple_conflicts(tmp_path):
    """Test _resolve_conflict_name with multiple conflicts."""
    base_dir = tmp_path
    filename = "document.pdf"

    # Create multiple conflicting files
    (base_dir / "document.pdf").write_text("original")
    (base_dir / "document_2.pdf").write_text("second")
    (base_dir / "document_3.pdf").write_text("third")

    result = EMLWriter._resolve_conflict_name(base_dir, filename)

    assert result == base_dir / "document_4.pdf"
    assert not result.exists()


def test_resolve_conflict_name_no_extension(tmp_path):
    """Test _resolve_conflict_name with filename without extension."""
    base_dir = tmp_path
    filename = "README"

    # Create the original file
    original = base_dir / filename
    original.write_text("content")

    result = EMLWriter._resolve_conflict_name(base_dir, filename)

    assert result == base_dir / "README_2"
    assert not result.exists()


def test_resolve_conflict_name_complex_filename(tmp_path):
    """Test _resolve_conflict_name with complex filename."""
    base_dir = tmp_path
    filename = "my-file.with.dots.txt"

    # Create the original
    original = base_dir / filename
    original.write_text("content")

    result = EMLWriter._resolve_conflict_name(base_dir, filename)

    assert result == base_dir / "my-file.with.dots_2.txt"


def test_resolve_conflict_name_many_conflicts(tmp_path):
    """Test _resolve_conflict_name with many existing conflicts."""
    base_dir = tmp_path
    filename = "file.txt"

    # Create files up to the limit
    for i in range(2, 10001):  # _2 to _10000
        (base_dir / f"file_{i}.txt").write_text(f"content {i}")

    # The original doesn't exist, so should return file.txt
    result = EMLWriter._resolve_conflict_name(base_dir, filename)

    assert result.name == "file.txt"  # No conflicts, returns original
