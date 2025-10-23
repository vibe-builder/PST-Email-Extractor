"""
Unit tests for attachment extraction failure scenarios.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from pst_email_extractor.pst_parser import _extract_attachments


def test_attachment_extraction_no_attachments():
    """Test attachment extraction when message has no attachments."""
    mock_message = Mock()
    mock_message.number_of_attachments = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        attachments_root = Path(tmpdir)

        paths, count = _extract_attachments(mock_message, "test_email", attachments_root)

        assert paths == []
        assert count == 0


def test_attachment_extraction_corrupted_attachment():
    """Test attachment extraction with corrupted attachment data."""
    mock_message = Mock()
    mock_message.number_of_attachments = 1

    # Mock attachment that fails during access
    mock_attachment = Mock()
    mock_attachment.long_filename = "corrupted.docx"
    mock_attachment.file_name = "corrupted.docx"
    mock_attachment.name = "corrupted.docx"
    mock_attachment.size = 1000

    # Make read_buffer fail
    mock_attachment.read_buffer.side_effect = Exception("Corrupted attachment data")

    mock_message.get_attachment.return_value = mock_attachment

    with tempfile.TemporaryDirectory() as tmpdir:
        attachments_root = Path(tmpdir)

        paths, count = _extract_attachments(mock_message, "test_email", attachments_root)

        # Should handle corruption gracefully - no attachments extracted
        assert paths == []
        assert count == 0


def test_attachment_extraction_write_failure():
    """Test attachment extraction when file write fails."""
    mock_message = Mock()
    mock_message.number_of_attachments = 1

    mock_attachment = Mock()
    mock_attachment.long_filename = "test.docx"
    mock_attachment.file_name = "test.docx"
    mock_attachment.name = "test.docx"
    mock_attachment.size = 100
    mock_attachment.read_buffer.return_value = b"test content"

    mock_message.get_attachment.return_value = mock_attachment

    with tempfile.TemporaryDirectory() as tmpdir:
        attachments_root = Path(tmpdir)

        # Simulate write failure by mocking open to raise exception
        with patch('pathlib.Path.open', side_effect=OSError("Write failed")):
            paths, count = _extract_attachments(mock_message, "test_email", attachments_root)

            # Should handle write errors gracefully
            assert paths == []
            assert count == 0


def test_attachment_extraction_invalid_filename():
    """Test attachment extraction with invalid filename characters."""
    mock_message = Mock()
    mock_message.number_of_attachments = 1

    mock_attachment = Mock()
    # Filename with invalid characters
    mock_attachment.long_filename = "file:with*invalid?chars.txt"
    mock_attachment.file_name = "file:with*invalid?chars.txt"
    mock_attachment.name = "file:with*invalid?chars.txt"
    mock_attachment.size = 50
    mock_attachment.read_buffer.return_value = b"content"

    mock_message.get_attachment.return_value = mock_attachment

    with tempfile.TemporaryDirectory() as tmpdir:
        attachments_root = Path(tmpdir)

        paths, count = _extract_attachments(mock_message, "test_email", attachments_root)

        # Should sanitize filename and extract successfully
        assert count == 1
        assert len(paths) == 1

        # Check that filename was sanitized
        filename = Path(paths[0]).name
        assert "*" not in filename
        assert "?" not in filename
        assert ":" not in filename


def test_attachment_extraction_empty_data():
    """Test attachment extraction with empty attachment data."""
    mock_message = Mock()
    mock_message.number_of_attachments = 1

    mock_attachment = Mock()
    mock_attachment.long_filename = "empty.txt"
    mock_attachment.file_name = "empty.txt"
    mock_attachment.name = "empty.txt"
    mock_attachment.size = 0
    mock_attachment.read_buffer.return_value = b""  # Empty data

    mock_message.get_attachment.return_value = mock_attachment

    with tempfile.TemporaryDirectory() as tmpdir:
        attachments_root = Path(tmpdir)

        paths, count = _extract_attachments(mock_message, "test_email", attachments_root)

        # Should skip empty attachments
        assert paths == []
        assert count == 0


def test_attachment_extraction_multiple_attachments_mixed_failures():
    """Test attachment extraction with multiple attachments, some failing."""
    mock_message = Mock()
    mock_message.number_of_attachments = 3

    # Create three different attachments: good, corrupted, good
    attachments = []

    # Good attachment 1
    good_attachment1 = Mock()
    good_attachment1.long_filename = "good1.txt"
    good_attachment1.file_name = "good1.txt"
    good_attachment1.name = "good1.txt"
    good_attachment1.size = 10
    good_attachment1.read_buffer.return_value = b"content1"
    attachments.append(good_attachment1)

    # Corrupted attachment
    bad_attachment = Mock()
    bad_attachment.long_filename = "bad.txt"
    bad_attachment.file_name = "bad.txt"
    bad_attachment.name = "bad.txt"
    bad_attachment.size = 100
    bad_attachment.read_buffer.side_effect = Exception("Corrupted")
    attachments.append(bad_attachment)

    # Good attachment 2
    good_attachment2 = Mock()
    good_attachment2.long_filename = "good2.txt"
    good_attachment2.file_name = "good2.txt"
    good_attachment2.name = "good2.txt"
    good_attachment2.size = 10
    good_attachment2.read_buffer.return_value = b"content2"
    attachments.append(good_attachment2)

    mock_message.get_attachment.side_effect = attachments

    with tempfile.TemporaryDirectory() as tmpdir:
        attachments_root = Path(tmpdir)

        paths, count = _extract_attachments(mock_message, "test_email", attachments_root)

        # Should extract 2 good attachments, skip 1 corrupted
        assert count == 2
        assert len(paths) == 2

        # Check filenames
        filenames = [Path(p).name for p in paths]
        assert "good1.txt" in filenames
        assert "good2.txt" in filenames
        assert "bad.txt" not in filenames


def test_attachment_extraction_no_attachments_root():
    """Test attachment extraction when no attachments root is provided."""
    mock_message = Mock()
    mock_message.number_of_attachments = 1

    mock_attachment = Mock()
    mock_attachment.long_filename = "test.txt"
    mock_attachment.file_name = "test.txt"
    mock_attachment.name = "test.txt"
    mock_attachment.size = 10
    mock_attachment.read_buffer.return_value = b"content"

    mock_message.get_attachment.return_value = mock_attachment

    # No attachments root provided
    paths, count = _extract_attachments(mock_message, "test_email", None)

    # Should skip extraction
    assert paths == []
    assert count == 0


def test_attachment_extraction_filename_conflicts():
    """Test attachment extraction with filename conflicts."""
    mock_message = Mock()
    mock_message.number_of_attachments = 2

    # Two attachments with same filename
    attachment1 = Mock()
    attachment1.long_filename = "duplicate.txt"
    attachment1.file_name = "duplicate.txt"
    attachment1.name = "duplicate.txt"
    attachment1.size = 10
    attachment1.read_buffer.return_value = b"content1"

    attachment2 = Mock()
    attachment2.long_filename = "duplicate.txt"
    attachment2.file_name = "duplicate.txt"
    attachment2.name = "duplicate.txt"
    attachment2.size = 10
    attachment2.read_buffer.return_value = b"content2"

    mock_message.get_attachment.side_effect = [attachment1, attachment2]

    with tempfile.TemporaryDirectory() as tmpdir:
        attachments_root = Path(tmpdir)

        paths, count = _extract_attachments(mock_message, "test_email", attachments_root)

        # Should extract both with conflict resolution
        assert count == 2
        assert len(paths) == 2

        # Check conflict resolution in filenames
        filenames = [Path(p).name for p in paths]
        assert "duplicate.txt" in filenames
        assert "duplicate_2.txt" in filenames
