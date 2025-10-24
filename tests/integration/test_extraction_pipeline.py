"""Integration tests for the full extraction pipeline."""

import tempfile
from pathlib import Path

import pytest

from pst_email_extractor.core.models import ExtractionConfig
from pst_email_extractor.core.services import run_extraction


class TestExtractionPipeline:
    """Test end-to-end extraction with mocked PST data."""

    @pytest.fixture
    def mock_pst_backend(self, monkeypatch):
        """Mock the PST backend to return test data."""
        from unittest.mock import Mock

        # Create a mock backend that yields sample emails
        mock_backend = Mock()
        mock_backend.is_available.return_value = True

        sample_emails = [
            {
                "Email_ID": "test_001",
                "Subject": "Test Email 1",
                "From": "sender1@example.com",
                "To": "recipient1@example.com",
                "Body": "This is test email 1 content.",
                "Date_Received": "2024-01-01T10:00:00",
                "Attachment_Count": 0,
                "Attachment_Paths": [],
            },
            {
                "Email_ID": "test_002",
                "Subject": "Test Email 2",
                "From": "sender2@example.com",
                "To": "recipient2@example.com",
                "Body": "This is test email 2 content.",
                "Date_Received": "2024-01-02T11:00:00",
                "Attachment_Count": 0,
                "Attachment_Paths": [],
            }
        ]

        def mock_iter_messages(**_kwargs):
            yield from sample_emails

        mock_backend.iter_messages = mock_iter_messages

        # Mock backend import
        def mock_import_parser():
            from unittest.mock import Mock
            parser_mock = Mock()
            parser_mock.is_pypff_available.return_value = True
            parser_mock.pypff.file.return_value = mock_backend

            # Mock the iter_emails method to return sample emails
            sample_emails = [
                {
                    "Email_ID": "test_001",
                    "Subject": "Test Email 1",
                    "From": "sender1@example.com",
                    "To": "recipient1@example.com",
                    "Body": "This is test email 1 content.",
                    "Date_Received": "2024-01-01T10:00:00",
                    "Attachment_Count": 0,
                    "Attachment_Paths": [],
                },
                {
                    "Email_ID": "test_002",
                    "Subject": "Test Email 2",
                    "From": "sender2@example.com",
                    "To": "recipient2@example.com",
                    "Body": "This is test email 2 content.",
                    "Date_Received": "2024-01-02T11:00:00",
                    "Attachment_Count": 0,
                    "Attachment_Paths": [],
                }
            ]

            def mock_iter_emails(*_args, **_kwargs):
                return iter(sample_emails)

            parser_mock.iter_emails = mock_iter_emails
            return parser_mock

        monkeypatch.setattr(
            "pst_email_extractor.core.backends.pypff.PypffBackend._import_parser",
            lambda _self: mock_import_parser()
        )

        return mock_backend

    def test_basic_extraction_json(self, _mock_pst_backend):
        """Test basic extraction to JSON format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractionConfig(
                pst_path=Path("dummy.pst"),  # Not used in mock
                output_dir=Path(tmpdir),
                formats=["json"],
                mode="extract",
                deduplicate=False,
            )

            result = run_extraction(config)

            # Check that files were created
            assert len(result.exported_paths) == 1
            json_file = result.exported_paths[0]
            assert json_file.exists()
            assert json_file.suffix == ".json"

            # Check content
            content = json_file.read_text()
            assert "test_001" in content
            assert "test_002" in content
            assert result.email_count == 2

    def test_basic_extraction_csv(self, _mock_pst_backend):
        """Test basic extraction to CSV format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractionConfig(
                pst_path=Path("dummy.pst"),
                output_dir=Path(tmpdir),
                formats=["csv"],
                mode="extract",
                deduplicate=False,
            )

            result = run_extraction(config)

            assert len(result.exported_paths) == 1
            csv_file = result.exported_paths[0]
            assert csv_file.exists()
            assert csv_file.suffix == ".csv"

            content = csv_file.read_text()
            assert "Email_ID" in content
            assert "test_001" in content

    def test_address_analysis_mode(self, _mock_pst_backend):
        """Test address analysis mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractionConfig(
                pst_path=Path("dummy.pst"),
                output_dir=Path(tmpdir),
                formats=["json"],
                mode="addresses",
                deduplicate=False,
            )

            result = run_extraction(config)

            # Should have address and host exports
            assert len(result.exported_paths) >= 1
            assert result.mode == "addresses"
            assert result.address_count > 0

    def test_deduplication(self, mock_pst_backend, monkeypatch):
        """Test email deduplication."""
        # Mock with duplicate emails for this specific test
        def mock_import_parser_duplicate():
            from unittest.mock import Mock
            parser_mock = Mock()
            parser_mock.is_pypff_available.return_value = True
            parser_mock.pypff.file.return_value = mock_pst_backend

            # Duplicate emails for deduplication test (same content, no unique IDs)
            duplicate_emails = [
                {
                    "Subject": "Duplicate",
                    "From": "sender@example.com",
                    "To": "recipient@example.com",
                    "Body": "Same content",
                    "Date_Received": "2024-01-01T10:00:00",
                    "Attachment_Count": 0,
                    "Attachment_Paths": [],
                },
                {
                    "Subject": "Duplicate",  # Same content
                    "From": "sender@example.com",
                    "To": "recipient@example.com",
                    "Body": "Same content",
                    "Date_Received": "2024-01-01T10:00:00",
                    "Attachment_Count": 0,
                    "Attachment_Paths": [],
                }
            ]

            def mock_iter_emails(*_args, **_kwargs):
                return iter(duplicate_emails)

            parser_mock.iter_emails = mock_iter_emails
            return parser_mock

        monkeypatch.setattr(
            "pst_email_extractor.core.backends.pypff.PypffBackend._import_parser",
            lambda _self: mock_import_parser_duplicate()
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExtractionConfig(
                pst_path=Path("dummy.pst"),
                output_dir=Path(tmpdir),
                formats=["json"],
                mode="extract",
                deduplicate=True,  # Enable deduplication
            )

            result = run_extraction(config)

            # Should deduplicate based on content hash
            # With deduplication, we expect only 1 email (duplicates removed)
            assert result.email_count == 1
