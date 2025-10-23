"""Integration tests for GUI smoke testing."""

import pytest

from pst_email_extractor.gui.main import SettingsDialog


class TestGUISmoke:
    """Smoke tests for GUI components to ensure they can be instantiated without errors."""

    def test_settings_dialog_creation(self):
        """Test that SettingsDialog can be created and basic UI elements work."""
        pytest.importorskip("customtkinter", reason="CustomTkinter not available")

        try:
            import tkinter as tk
            import customtkinter as ctk

            # Create a minimal root window (required for CTk components)
            root = ctk.CTk()
            root.withdraw()  # Hide the root window

            # Test creating settings dialog
            initial_settings = {
                "ai_sanitize": True,
                "ai_polish": False,
                "ai_language": "en-US",
                "ai_neural_model_dir": "",
                "extract_attachment_content": False,
                "attachment_ocr": True,
                "attachment_ocr_languages": ["eng"],
                "attachment_max_size_mb": 10,
                "export_compress": False,
                "export_html_index": True,
                "export_deduplicate": False,
            }

            dialog = SettingsDialog(root, initial_settings)
            assert dialog is not None
            assert hasattr(dialog, 'get_settings')

            # Test getting settings back (simulate Apply button click)
            dialog._apply()
            settings = dialog.get_settings()
            assert isinstance(settings, dict)
            assert "ai_sanitize" in settings

            # Clean up
            dialog.dialog.destroy()
            root.destroy()

        except ImportError:
            pytest.skip("GUI dependencies not available")

    def test_address_analysis_dialog_creation(self):
        """Test that AddressAnalysisDialog can be created."""
        pytest.importorskip("customtkinter", reason="CustomTkinter not available")
        pytest.importorskip("tkinter", reason="Tkinter not available")

        try:
            import tkinter as tk
            import customtkinter as ctk
            from unittest.mock import Mock

            # Create a minimal root window
            root = ctk.CTk()
            root.withdraw()

            # Mock extractor
            mock_extractor = Mock()
            mock_extractor.is_available.return_value = True

            initial_settings = {"deduplicate": False}

            from pst_email_extractor.gui.main import AddressAnalysisDialog
            dialog = AddressAnalysisDialog(root, mock_extractor, initial_settings)
            assert dialog is not None

            # Clean up
            dialog.destroy()
            root.destroy()

        except ImportError:
            pytest.skip("GUI dependencies not available")
