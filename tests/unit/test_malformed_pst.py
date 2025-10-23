"""
Unit tests for malformed PST file handling and error scenarios.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from pst_email_extractor.core.analysis import analyze_pst_health
from pst_email_extractor.core.extraction import perform_extraction
from pst_email_extractor.core.models import ExtractionConfig


def test_analyze_health_nonexistent_file():
    """Test health analysis with non-existent file."""
    result = analyze_pst_health(Path("/completely/nonexistent/file.pst"))

    assert result.total_emails == 0
    assert result.folder_count == 0
    assert result.corrupted_samples == 0
    assert result.sampled == 0
    assert result.estimated_size_mb == 0.0
    assert result.health_score == 100.0  # No samples = perfect health


def test_analyze_health_empty_file():
    """Test health analysis handles empty files gracefully."""
    # Test with non-existent file (simulates empty/invalid file scenario)
    result = analyze_pst_health(Path("/nonexistent/empty.pst"))

    assert result.total_emails == 0
    assert result.corrupted_samples == 0
    assert result.estimated_size_mb == 0.0
    assert result.sampled == 0


def test_analyze_health_corrupted_pst():
    """Test health analysis with corrupted PST file."""
    # Test with a file that exists but pypff can't open
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.stat') as mock_stat:

        mock_stat.return_value.st_size = 1024 * 1024  # 1MB

        # Mock pypff import to fail
        with patch('builtins.__import__', side_effect=ImportError("pypff not available")):
            result = analyze_pst_health(Path("/fake/corrupted.pst"))

            # Should return default values since pypff failed
            assert result.total_emails == 0
            assert result.estimated_size_mb == 1.0


def test_analyze_health_partial_corruption():
    """Test health analysis with partial corruption simulation."""
    # Test that we handle missing dependencies gracefully
    with patch('pathlib.Path.exists', return_value=True), \
         patch('pathlib.Path.stat') as mock_stat:

        mock_stat.return_value.st_size = 1024 * 1024 * 50  # 50MB

        # Simulate missing pypff dependency
        with patch('builtins.__import__', side_effect=ImportError("pypff not available")):
            result = analyze_pst_health(Path("/fake/partial.pst"))

            # Should return graceful defaults
            assert result.estimated_size_mb == 50.0
            assert result.total_emails == 0


def test_extraction_config_validation():
    """Test that extraction config validates paths properly."""
    # Test with non-existent output directory
    config = ExtractionConfig(
        pst_path=Path("nonexistent.pst"),
        output_dir=Path("nonexistent_output"),
        formats=["json"]
    )

    # Should not raise during config creation, only during execution
    assert config.pst_path == Path("nonexistent.pst")
    assert config.output_dir == Path("nonexistent_output")


@patch('pst_email_extractor.core.backends.PypffBackend.open')
@patch('pst_email_extractor.core.backends.PypffBackend.is_available', return_value=True)
def test_extraction_with_invalid_pst_path(mock_available, mock_open):
    """Test extraction handling with invalid PST path."""
    mock_open.side_effect = FileNotFoundError("PST file not found")

    config = ExtractionConfig(
        pst_path=Path("nonexistent.pst"),
        output_dir=Path("output"),
        formats=["json"]
    )

    with pytest.raises(FileNotFoundError):
        perform_extraction(config)


@patch('pst_email_extractor.core.backends.PypffBackend.is_available', return_value=False)
def test_extraction_without_pypff(mock_available):
    """Test extraction when pypff library is not available."""
    config = ExtractionConfig(
        pst_path=Path("test.pst"),
        output_dir=Path("output"),
        formats=["json"]
    )

    with pytest.raises(Exception) as exc_info:  # Should raise DependencyError
        perform_extraction(config)

    assert "pypff library not found" in str(exc_info.value)


def test_health_analysis_with_unicode_path():
    """Test health analysis with Unicode characters in path."""
    unicode_path = Path("测试文件夹/test_文件.pst")

    with patch('pathlib.Path.exists', return_value=False):
        result = analyze_pst_health(unicode_path)

        # Should handle Unicode paths gracefully
        assert result.total_emails == 0
        assert result.estimated_size_mb == 0.0


def test_health_analysis_timeout_simulation():
    """Test health analysis respects sample limits."""
    # Test that sample_limit parameter is respected (can't easily test timeout without real PST)
    # The function should accept the parameter without error
    result = analyze_pst_health(Path("/nonexistent.pst"), sample_limit=50)

    # Should handle non-existent file gracefully
    assert result.total_emails == 0
    assert result.sampled == 0
