"""
Integration tests for AI pipeline with exporters.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from pst_email_extractor.exporters.json_writer import JSONStreamWriter
from pst_email_extractor.exporters.csv_writer import CSVStreamWriter


def test_json_exporter_with_ai_sanitization(tmp_path):
    """Test JSON exporter with AI sanitization enabled."""
    output_file = tmp_path / "test_output.json"

    # Test data with PII that should be masked
    test_email = {
        "Email_ID": "test_001",
        "Subject": "Meeting at 3pm",
        "Body": "Contact john.doe@example.com or call 555-123-4567 for details.",
        "From": "sender@test.com",
        "To": "recipient@test.com",
        "Date_Received": "2024-01-01T10:00:00"
    }

    with JSONStreamWriter(str(output_file), ai_sanitize=True, ai_polish=False) as writer:
        writer.write(test_email)

    # Verify file was created and contains processed data
    assert output_file.exists()

    # Read and verify content
    import json
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    assert "test_001" in data
    email_data = data["test_001"]

    # Check that PII was sanitized (email addresses should be masked)
    assert "<EMAIL>" in email_data["Body"]
    assert "<PHONE>" in email_data["Body"]
    assert "john.doe@example.com" not in email_data["Body"]
    assert "555-123-4567" not in email_data["Body"]


def test_csv_exporter_with_ai_sanitization(tmp_path):
    """Test CSV exporter with AI sanitization enabled."""
    output_file = tmp_path / "test_output.csv"

    # Test data with PII
    test_email = {
        "Email_ID": "test_002",
        "Subject": "Project update",
        "Body": "Please review the document from jane.smith@company.com sent at 202.168.1.100",
        "From": "boss@company.com",
        "To": "team@company.com"
    }

    with CSVStreamWriter(str(output_file), ai_sanitize=True, ai_polish=False) as writer:
        writer.write(test_email)

    # Verify file was created
    assert output_file.exists()

    # Read and verify content
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Should have header + 1 data row
    assert len(lines) == 2

    # Check that CSV contains sanitized data
    data_line = lines[1]
    assert "<EMAIL>" in data_line
    assert "<IP>" in data_line
    assert "jane.smith@company.com" not in data_line
    assert "202.168.1.100" not in data_line


def test_exporter_without_ai_processing(tmp_path):
    """Test exporter without AI processing (baseline)."""
    output_file = tmp_path / "test_no_ai.json"

    test_email = {
        "Email_ID": "test_003",
        "Subject": "Plain email",
        "Body": "This is plain text without any PII like john@example.com",
        "From": "sender@test.com"
    }

    # No AI processing enabled
    with JSONStreamWriter(str(output_file), ai_sanitize=False, ai_polish=False) as writer:
        writer.write(test_email)

    assert output_file.exists()

    import json
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    email_data = data["test_003"]

    # PII should NOT be sanitized when AI is disabled
    assert "john@example.com" in email_data["Body"]
    assert "<EMAIL>" not in email_data["Body"]


def test_ai_pipeline_error_handling(tmp_path):
    """Test AI pipeline error handling in exporters."""
    output_file = tmp_path / "test_error.json"

    test_email = {
        "Email_ID": "test_004",
        "Subject": "Test email",
        "Body": "Test content with email@test.com",
        "From": "sender@test.com"
    }

    # Mock AI pipeline to fail
    with patch('pst_email_extractor.ai.pipeline.create_text_pipeline') as mock_create:
        mock_pipeline = Mock()
        mock_pipeline.process.side_effect = Exception("AI processing failed")
        mock_create.return_value = mock_pipeline

        # Should still export successfully despite AI failure
        with JSONStreamWriter(str(output_file), ai_sanitize=True, ai_polish=False) as writer:
            writer.write(test_email)

    assert output_file.exists()

    import json
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # Should contain sanitized data (regex sanitization happens first, even if other AI processing fails)
        email_data = data["test_004"]
        assert email_data["Body"] == "Test content with <EMAIL>"


def test_ai_polish_with_missing_dependencies(tmp_path):
    """Test AI polishing when dependencies are not available."""
    output_file = tmp_path / "test_missing_deps.json"

    test_email = {
        "Email_ID": "test_005",
        "Subject": "Test email",
        "Body": "This has some gramatical errors and speling mistakes.",
        "From": "sender@test.com"
    }

    # Mock create_text_pipeline to return None (simulating missing dependencies)
    with patch('pst_email_extractor.ai.pipeline.create_text_pipeline', return_value=None):
        with JSONStreamWriter(str(output_file), ai_sanitize=False, ai_polish=True) as writer:
            writer.write(test_email)

    assert output_file.exists()

    import json
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Should contain original data (no AI processing available)
    email_data = data["test_005"]
    assert "gramatical" in email_data["Body"]  # Original text preserved
    assert "speling" in email_data["Body"]


def test_multiple_emails_with_ai_processing(tmp_path):
    """Test processing multiple emails with AI enabled."""
    output_file = tmp_path / "test_multiple.json"

    emails = [
        {
            "Email_ID": "email_001",
            "Subject": "First email",
            "Body": "Contact support@example.com for help.",
            "From": "user1@test.com"
        },
        {
            "Email_ID": "email_002",
            "Subject": "Second email",
            "Body": "Call 1-800-123-4567 for assistance.",
            "From": "user2@test.com"
        }
    ]

    with JSONStreamWriter(str(output_file), ai_sanitize=True, ai_polish=False) as writer:
        for email in emails:
            writer.write(email)

    assert output_file.exists()

    import json
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Check both emails were processed
    assert "email_001" in data
    assert "email_002" in data

    # Check AI processing was applied
    email1_data = data["email_001"]
    email2_data = data["email_002"]

    assert "<EMAIL>" in email1_data["Body"]
    assert "<PHONE>" in email2_data["Body"]
    assert "support@example.com" not in email1_data["Body"]
    assert "1-800-123-4567" not in email2_data["Body"]


def test_ai_neural_processing_with_model_dir(tmp_path):
    """Test AI neural processing when model directory is provided."""
    output_file = tmp_path / "test_neural.json"

    test_email = {
        "Email_ID": "test_006",
        "Subject": "Test neural processing",
        "Body": "This text could be improved with neural processing.",
        "From": "sender@test.com"
    }

    # Test with neural model directory (will fail gracefully if ONNX not available)
    with JSONStreamWriter(str(output_file), ai_sanitize=False, ai_polish=True,
                         ai_neural_model_dir="/fake/model/dir") as writer:
        writer.write(test_email)

    # Should still export successfully
    assert output_file.exists()

    import json
    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Should contain some data (neural processing may or may not work)
    assert "test_006" in data
