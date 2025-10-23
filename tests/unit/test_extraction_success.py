"""
Tests for successful extraction scenarios.
Tests positive paths where extraction should succeed and produce expected outputs.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from pst_email_extractor.core.backends.base import PstBackend
from pst_email_extractor.core.extraction import perform_extraction
from pst_email_extractor.core.models import ExtractionConfig


@pytest.fixture
def mock_pypff_backend():
    """Create a mock PstBackend that yields sample email data."""
    backend = Mock(spec=PstBackend)

    # Sample email data that would be parsed
    mock_emails = [
        {
            "Email_ID": "test_001",
            "Subject": "Test Email 1",
            "From": "sender1@example.com",
            "To": "recipient1@example.com",
            "Body": "This is test email 1",
            "Date_Received": "2024-01-01T10:00:00",
            "Date_Received_Timestamp": 1704105600.0,
            "Date_Sent": "2024-01-01T09:00:00",
            "Date_Sent_Timestamp": 1704102000.0,
            "Attachment_Count": 1,
            "Attachment_Paths": ["/path/to/attachment1.txt"],
            "HTML_Body": "<html><body>This is test email 1</body></html>",
        },
        {
            "Email_ID": "test_002",
            "Subject": "Test Email 2",
            "From": "sender2@example.com",
            "To": "recipient2@example.com",
            "Body": "This is test email 2",
            "Date_Received": "2024-01-02T11:00:00",
            "Date_Received_Timestamp": 1704192000.0,
            "Date_Sent": "2024-01-02T10:00:00",
            "Date_Sent_Timestamp": 1704188400.0,
            "Attachment_Count": 0,
            "Attachment_Paths": [],
            "HTML_Body": "",
        },
    ]

    backend.iter_messages.return_value = iter(mock_emails)
    backend.close = Mock()
    return backend


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestExtractionSuccess:
    """Test successful extraction scenarios."""

    def test_csv_extraction_success(self, mock_pypff_backend, temp_output_dir):
        """Test that CSV extraction produces expected output file."""
        config = ExtractionConfig(
            pst_path=Path("/fake/path.pst"),
            output_dir=temp_output_dir,
            formats=["csv"],
            mode="extract",
        )

        with patch("pst_email_extractor.core.extraction.PypffBackend") as mock_backend_class:
            mock_backend_class.return_value = mock_pypff_backend
            result = perform_extraction(config)

        # Check that result indicates success
        assert result.mode == "extract"
        assert len(result.exported_paths) == 1
        assert result.exported_paths[0].name.endswith(".csv")
        assert result.exported_paths[0].exists()

        # Check CSV content
        csv_content = result.exported_paths[0].read_text()
        assert "Email_ID,Date_Received,Date_Sent,From" in csv_content  # CSV header
        assert "test_001" in csv_content  # First email
        assert "test_002" in csv_content  # Second email
        assert "Test Email 1" in csv_content
        assert "sender1@example.com" in csv_content

    def test_json_extraction_success(self, mock_pypff_backend, temp_output_dir):
        """Test that JSON extraction produces expected output file."""
        config = ExtractionConfig(
            pst_path=Path("/fake/path.pst"),
            output_dir=temp_output_dir,
            formats=["json"],
            mode="extract",
        )

        with patch("pst_email_extractor.core.extraction.PypffBackend") as mock_backend_class:
            mock_backend_class.return_value = mock_pypff_backend
            result = perform_extraction(config)

        # Check that result indicates success
        assert result.mode == "extract"
        assert len(result.exported_paths) == 1
        assert result.exported_paths[0].name.endswith(".json")
        assert result.exported_paths[0].exists()

        # Check JSON content
        import json
        json_content = json.loads(result.exported_paths[0].read_text())
        assert isinstance(json_content, dict)
        assert len(json_content) == 2
        assert "test_001" in json_content
        assert "test_002" in json_content
        assert json_content["test_001"]["Subject"] == "Test Email 1"
        assert json_content["test_002"]["Subject"] == "Test Email 2"

    def test_html_index_generation(self, mock_pypff_backend, temp_output_dir):
        """Test that HTML index generation produces expected output."""
        config = ExtractionConfig(
            pst_path=Path("/fake/path.pst"),
            output_dir=temp_output_dir,
            formats=["csv"],  # Include CSV to trigger HTML index
            mode="extract",
            html_index=True,
        )

        with patch("pst_email_extractor.core.extraction.PypffBackend") as mock_backend_class:
            mock_backend_class.return_value = mock_pypff_backend
            result = perform_extraction(config)

        # Check that HTML index was generated
        html_path = None
        for path in result.exported_paths:
            if path.name.endswith("_index.html"):
                html_path = path
                break

        assert html_path is not None
        assert html_path.exists()

        # Check HTML content contains expected elements
        html_content = html_path.read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "Test Email 1" in html_content
        assert "Test Email 2" in html_content
        assert "sender1@example.com" in html_content

    def test_eml_extraction_with_attachments(self, mock_pypff_backend, temp_output_dir):
        """Test that EML extraction works and creates attachment directories."""
        config = ExtractionConfig(
            pst_path=Path("/fake/path.pst"),
            output_dir=temp_output_dir,
            formats=["eml"],
            mode="extract",
            extract_attachments=True,
        )

        with patch("pst_email_extractor.core.extraction.PypffBackend") as mock_backend_class:
            mock_backend_class.return_value = mock_pypff_backend
            result = perform_extraction(config)

        # Check that EML directory was created
        eml_path = None
        for path in result.exported_paths:
            if "eml" in path.name and path.is_dir():
                eml_path = path
                break

        assert eml_path is not None
        assert eml_path.exists()

        # Check that EML files were created
        eml_files = list(eml_path.glob("*.eml"))
        assert len(eml_files) == 2  # Two emails

        # Check EML content
        eml_content = eml_files[0].read_text()
        assert "Subject: Test Email 1" in eml_content
        assert "From: sender1@example.com" in eml_content

    def test_mbox_extraction_success(self, mock_pypff_backend, temp_output_dir):
        """Test that MBOX extraction produces expected output file."""
        config = ExtractionConfig(
            pst_path=Path("/fake/path.pst"),
            output_dir=temp_output_dir,
            formats=["mbox"],
            mode="extract",
        )

        with patch("pst_email_extractor.core.extraction.PypffBackend") as mock_backend_class:
            mock_backend_class.return_value = mock_pypff_backend
            result = perform_extraction(config)

        # Check that MBOX file was created
        mbox_path = None
        for path in result.exported_paths:
            if path.name.endswith(".mbox"):
                mbox_path = path
                break

        assert mbox_path is not None
        assert mbox_path.exists()

        # Check MBOX content
        mbox_content = mbox_path.read_text()
        assert "From " in mbox_content  # MBOX format marker
        assert "Test Email 1" in mbox_content
        assert "Test Email 2" in mbox_content

    def test_multiple_formats_extraction(self, mock_pypff_backend, temp_output_dir):
        """Test extraction with multiple output formats simultaneously."""
        config = ExtractionConfig(
            pst_path=Path("/fake/path.pst"),
            output_dir=temp_output_dir,
            formats=["csv", "json", "eml"],
            mode="extract",
        )

        with patch("pst_email_extractor.core.extraction.PypffBackend") as mock_backend_class:
            mock_backend_class.return_value = mock_pypff_backend
            result = perform_extraction(config)

        # Should have CSV, JSON, and EML outputs
        assert len(result.exported_paths) == 3

        # Check each format exists
        formats_found = {path.suffix.lstrip('.'): path for path in result.exported_paths}
        assert "csv" in formats_found
        assert "json" in formats_found

        # EML should be a directory
        eml_path = None
        for path in result.exported_paths:
            if path.is_dir() and "eml" in path.name:
                eml_path = path
                break
        assert eml_path is not None
