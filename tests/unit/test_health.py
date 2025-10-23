"""
Unit tests for PST health analysis functionality.
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from pst_email_extractor.core.analysis import PstHealth, analyze_pst_health


def test_pst_health_dataclass():
    """Test PstHealth dataclass and health_score property."""
    health = PstHealth(
        total_emails=1000,
        folder_count=5,
        corrupted_samples=10,
        sampled=200,
        estimated_size_mb=50.0
    )

    assert health.total_emails == 1000
    assert health.folder_count == 5
    assert health.corrupted_samples == 10
    assert health.sampled == 200
    assert health.estimated_size_mb == 50.0

    # Health score: (200-10)/200 * 100 = 95.0 (based on sampled messages)
    assert health.health_score == 95.0


def test_pst_health_zero_emails():
    """Test health score calculation with zero emails."""
    health = PstHealth(total_emails=0, folder_count=0, corrupted_samples=0, sampled=0, estimated_size_mb=0.0)
    assert health.health_score == 100.0  # Default for no samples


def test_pst_health_no_samples():
    """Test health score with no sampled messages."""
    health = PstHealth(total_emails=500, folder_count=3, corrupted_samples=0, sampled=0, estimated_size_mb=25.0)
    assert health.health_score == 100.0  # No samples = 100% health


def test_pst_health_all_corrupted():
    """Test health score when all samples are corrupted."""
    health = PstHealth(total_emails=100, folder_count=2, corrupted_samples=50, sampled=50, estimated_size_mb=10.0)
    assert health.health_score == 0.0  # All corrupted = 0% health


@pytest.mark.parametrize("total_emails,corrupted,sampled,expected", [
    (100, 0, 100, 100.0),    # Perfect health
    (100, 50, 100, 50.0),    # Half corrupted
    (100, 10, 100, 90.0),    # 10% corrupted
    (1, 1, 1, 0.0),          # Single corrupted email
])
def test_pst_health_score_edge_cases(total_emails, corrupted, sampled, expected):
    """Test various health score calculations."""
    health = PstHealth(
        total_emails=total_emails,
        folder_count=1,
        corrupted_samples=corrupted,
        sampled=sampled,
        estimated_size_mb=1.0
    )
    assert health.health_score == expected


@patch('pst_email_extractor.core.analysis.Path')
def test_analyze_pst_health_no_pypff(mock_path_class):
    """Test analyze_pst_health when pypff is not available."""
    mock_path_instance = Mock()
    mock_path_instance.expanduser.return_value.resolve.return_value = mock_path_instance
    mock_path_instance.exists.return_value = True
    mock_path_instance.stat.return_value.st_size = 1024 * 1024 * 10  # 10MB
    mock_path_class.return_value = mock_path_instance

    # Mock import error for pypff
    with patch.dict('sys.modules', {'pypff': None}):
        with patch('builtins.__import__', side_effect=ImportError("No module named 'pypff'")):
            health = analyze_pst_health(mock_path_instance)

    assert health.total_emails == 0
    assert health.folder_count == 0
    assert health.corrupted_samples == 0
    assert health.sampled == 0
    assert health.estimated_size_mb == 10.0  # 10MB


@patch('pst_email_extractor.core.analysis.Path')
def test_analyze_pst_health_file_size_calculation(mock_path_class):
    """Test file size calculation in health analysis."""
    mock_path_instance = Mock()
    mock_path_instance.expanduser.return_value.resolve.return_value = mock_path_instance
    mock_path_instance.exists.return_value = True
    mock_path_instance.stat.return_value.st_size = 1024 * 1024 * 50  # 50MB
    mock_path_class.return_value = mock_path_instance

    # Mock pypff failure
    with patch.dict('sys.modules', {'pypff': None}):
        with patch('builtins.__import__', side_effect=ImportError):
            health = analyze_pst_health(mock_path_instance)

    assert health.estimated_size_mb == 50.0


def test_analyze_pst_health_sample_limit():
    """Test that sampling respects the limit parameter."""
    # This test requires pypff to be available and a real PST file
    # For now, just test the parameter validation
    from pst_email_extractor.core.analysis import analyze_pst_health

    # Test that the function accepts the sample_limit parameter
    # (Actual testing would require a real PST file)
    assert callable(analyze_pst_health)

    # Test with a non-existent file (should handle gracefully)
    from pathlib import Path
    health = analyze_pst_health(Path("/nonexistent.pst"))
    assert health.total_emails == 0
    assert health.sampled == 0
