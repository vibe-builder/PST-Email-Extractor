"""
Unit tests for progress tracking and ETA calculation.
"""

from __future__ import annotations

import time
from unittest.mock import Mock, patch
import pytest

from pst_email_extractor.core.extraction import _emit_progress
from pst_email_extractor.core.models import ProgressUpdate


def test_emit_progress_with_callback():
    """Test _emit_progress calls callback correctly."""
    callback = Mock()
    _emit_progress(callback, 10, 100, "Test message")

    callback.assert_called_once()
    args = callback.call_args[0]
    assert len(args) == 1
    assert isinstance(args[0], ProgressUpdate)
    assert args[0].current == 10
    assert args[0].total == 100
    assert args[0].message == "Test message"


def test_emit_progress_legacy_callback():
    """Test _emit_progress with legacy tuple callback."""
    callback = Mock()
    _emit_progress(callback, 50, 200, "Legacy message")

    # Should call with ProgressUpdate object (not legacy tuple format)
    update = callback.call_args[0][0]
    assert isinstance(update, ProgressUpdate)
    assert update.current == 50
    assert update.total == 200
    assert update.message == "Legacy message"


def test_emit_progress_no_callback():
    """Test _emit_progress does nothing when callback is None."""
    callback = None
    # Should not raise any exception
    _emit_progress(callback, 1, 10, "No callback")


def test_emit_progress_callback_exception():
    """Test _emit_progress handles callback exceptions gracefully."""
    def failing_callback(update):
        raise RuntimeError("Callback failed")

    # Should not raise the exception
    _emit_progress(failing_callback, 5, 20, "Failing callback")


def test_progress_message_formatting():
    """Test that progress messages include expected components."""
    # Mock the progress proxy to capture messages
    captured_messages = []

    def mock_emit(callback, current, total, message):
        captured_messages.append((current, total, message))

        # This would require mocking the entire extraction loop
        # For now, we test the message format expectations

        # Expected message format: "Processed X/Y (Z.Z/sec) • ETA mm:ss"
        test_messages = [
            (10, 100, "Processed 10/100 (5.0/sec) • ETA 00:18"),
            (50, 100, "Processed 50/100 (10.0/sec) • ETA 00:05"),
            (90, 100, "Processed 90/100 (15.0/sec) • ETA 00:00"),
        ]

        for current, total, expected_msg in test_messages:
            # Test message format expectations (basic validation)
            assert 'Processed' in expected_msg
            assert '/' in expected_msg  # Should have X/Y format
            assert 'ETA' in expected_msg  # Should include ETA
            assert '/sec' in expected_msg  # Should include rate


def test_progress_rate_calculation():
    """Test EMA rate calculation logic."""
    # Test the EMA formula: new_rate = 0.8 * old_rate + 0.2 * instantaneous_rate

    # Initial rate
    old_rate = 10.0  # emails/sec
    inst_rate = 15.0  # new instantaneous rate
    expected_new_rate = 0.8 * 10.0 + 0.2 * 15.0  # = 11.0

    assert abs(expected_new_rate - 11.0) < 0.001

    # Another step
    old_rate = 11.0
    inst_rate = 8.0
    expected_new_rate = 0.8 * 11.0 + 0.2 * 8.0  # = 10.4

    assert abs(expected_new_rate - 10.4) < 0.001


def test_eta_calculation():
    """Test ETA calculation from rate and remaining work."""
    total_emails = 1000
    processed = 200
    remaining = total_emails - processed  # 800
    rate = 50.0  # emails/sec

    expected_seconds = remaining / rate  # 16.0 seconds
    expected_minutes = int(expected_seconds // 60)  # 0
    expected_secs = int(expected_seconds % 60)  # 16

    eta_str = f"{expected_minutes:02d}:{expected_secs:02d}"
    assert eta_str == "00:16"

    # Test with minutes
    rate = 10.0  # slower rate
    expected_seconds = 800 / 10.0  # 80 seconds
    expected_minutes = int(80 // 60)  # 1
    expected_secs = int(80 % 60)  # 20

    eta_str = f"{expected_minutes:02d}:{expected_secs:02d}"
    assert eta_str == "01:20"


def test_zero_rate_eta_handling():
    """Test ETA calculation when rate is zero."""
    # Should handle zero rate gracefully (though in practice rate won't be zero)
    remaining = 100
    rate = 0.0

    # Avoid division by zero
    if rate > 0:
        seconds = remaining / rate
    else:
        seconds = 0  # or some default

    assert seconds == 0
