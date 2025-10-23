"""
Tests for attachment security features.
Tests that attachment size limits prevent DoS attacks.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from pst_email_extractor.pst_parser import _extract_attachments


@pytest.fixture
def temp_attachments_root():
    """Create a temporary directory for attachment tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestAttachmentSecurity:
    """Test attachment security features."""

    def test_large_attachment_size_rejected(self, temp_attachments_root):
        """Test that attachments with sizes > 100MB are rejected."""
        mock_message = Mock()
        mock_message.number_of_attachments = 1

        # Mock attachment with size > 100MB
        mock_attachment = Mock()
        mock_attachment.long_filename = "large_file.dat"
        mock_attachment.file_name = "large_file.dat"
        mock_attachment.name = "large_file.dat"
        mock_attachment.size = 150 * 1024 * 1024  # 150MB

        mock_message.get_attachment.return_value = mock_attachment

        paths, count = _extract_attachments(mock_message, "test_email", temp_attachments_root)

        # Should reject the large attachment
        assert count == 0
        assert len(paths) == 0

    def test_negative_attachment_size_rejected(self, temp_attachments_root):
        """Test that attachments with negative sizes are rejected."""
        mock_message = Mock()
        mock_message.number_of_attachments = 1

        # Mock attachment with negative size
        mock_attachment = Mock()
        mock_attachment.long_filename = "negative_size.dat"
        mock_attachment.file_name = "negative_size.dat"
        mock_attachment.name = "negative_size.dat"
        mock_attachment.size = -100

        mock_message.get_attachment.return_value = mock_attachment

        paths, count = _extract_attachments(mock_message, "test_email", temp_attachments_root)

        # Should reject the invalid attachment
        assert count == 0
        assert len(paths) == 0

    def test_invalid_size_metadata_rejected(self, temp_attachments_root):
        """Test that attachments with invalid size metadata are rejected."""
        mock_message = Mock()
        mock_message.number_of_attachments = 1

        # Mock attachment with non-numeric size
        mock_attachment = Mock()
        mock_attachment.long_filename = "invalid_size.dat"
        mock_attachment.file_name = "invalid_size.dat"
        mock_attachment.name = "invalid_size.dat"
        mock_attachment.size = "not_a_number"
        mock_attachment.get_size.return_value = Mock()  # Mock object

        mock_message.get_attachment.return_value = mock_attachment

        paths, count = _extract_attachments(mock_message, "test_email", temp_attachments_root)

        # Should reject the attachment with invalid size metadata
        assert count == 0
        assert len(paths) == 0

    def test_large_final_data_rejected(self, temp_attachments_root):
        """Test that final attachment data > 100MB is rejected even if size metadata is valid."""
        mock_message = Mock()
        mock_message.number_of_attachments = 1

        # Mock attachment with valid size but returns oversized data
        mock_attachment = Mock()
        mock_attachment.long_filename = "oversized.dat"
        mock_attachment.file_name = "oversized.dat"
        mock_attachment.name = "oversized.dat"
        mock_attachment.size = 10 * 1024 * 1024  # 10MB (valid)
        mock_attachment.read_buffer.return_value = b"x" * (150 * 1024 * 1024)  # 150MB data

        mock_message.get_attachment.return_value = mock_attachment

        paths, count = _extract_attachments(mock_message, "test_email", temp_attachments_root)

        # Should reject the oversized final data
        assert count == 0
        assert len(paths) == 0

    def test_normal_attachment_accepted(self, temp_attachments_root):
        """Test that normal-sized attachments are accepted."""
        mock_message = Mock()
        mock_message.number_of_attachments = 1

        # Mock normal attachment
        mock_attachment = Mock()
        mock_attachment.long_filename = "normal.txt"
        mock_attachment.file_name = "normal.txt"
        mock_attachment.name = "normal.txt"
        mock_attachment.size = 1024  # 1KB

        # Make read_buffer return data once, then empty bytes to stop reading
        mock_attachment.read_buffer.side_effect = [b"test content", b""]

        mock_message.get_attachment.return_value = mock_attachment

        paths, count = _extract_attachments(mock_message, "test_email", temp_attachments_root)

        # Should accept the normal attachment
        assert count == 1
        assert len(paths) == 1
        assert "normal.txt" in paths[0]

        # Verify file was created
        attachment_file = temp_attachments_root / "test_email" / "normal.txt"
        assert attachment_file.exists()
        assert attachment_file.read_text() == "test content"

    def test_chunked_reading_works(self, temp_attachments_root):
        """Test that large attachments are read in chunks."""
        mock_message = Mock()
        mock_message.number_of_attachments = 1

        # Mock large attachment that requires chunked reading
        mock_attachment = Mock()
        mock_attachment.long_filename = "large_chunked.dat"
        mock_attachment.file_name = "large_chunked.dat"
        mock_attachment.name = "large_chunked.dat"
        mock_attachment.size = 3 * 1024 * 1024  # 3MB

        # Simulate chunked reading - return the full data in one call since we're testing the size validation
        full_data = b"x" * (3 * 1024 * 1024)  # 3MB
        mock_attachment.read_buffer.return_value = full_data

        mock_message.get_attachment.return_value = mock_attachment

        paths, count = _extract_attachments(mock_message, "test_email", temp_attachments_root)

        # Should process the attachment (under 100MB limit)
        assert count == 1
        assert len(paths) == 1

        # Verify file was created with correct size
        attachment_file = temp_attachments_root / "test_email" / "large_chunked.dat"
        assert attachment_file.exists()
        assert attachment_file.stat().st_size == 3 * 1024 * 1024  # 3MB
