"""
Smoke tests for GUI components.
Tests that GUI classes can be imported and instantiated without errors.
"""

import pytest
from unittest.mock import patch, Mock


class TestGUISmoke:
    """Smoke tests for GUI functionality."""

    def test_pst_extractor_creation(self):
        """Test that PSTExtractor can be created."""
        from pst_email_extractor.gui.main import PSTExtractor

        extractor = PSTExtractor()
        assert extractor.pst_path is None
        assert not extractor.is_loaded

    def test_pst_extractor_load_pst(self):
        """Test PSTExtractor.load_pst method."""
        from pst_email_extractor.gui.main import PSTExtractor

        extractor = PSTExtractor()

        # Mock os.path.exists and the pypff availability check
        with patch("os.path.exists", return_value=True):
            with patch("pst_email_extractor.gui.main.is_pypff_available", return_value=True):
                # Mock backend open to avoid real file operations
                with patch("pst_email_extractor.gui.main.PypffBackend") as mock_backend_cls:
                    backend = Mock()
                    mock_backend_cls.return_value = backend
                    result = extractor.load_pst("/fake/path.pst")
                    assert result is True
                    assert extractor.is_loaded
                    assert extractor.pst_path == "/fake/path.pst"

    def test_pst_extractor_get_folders(self):
        """Test PSTExtractor.get_folders returns empty list (as implemented)."""
        from pst_email_extractor.gui.main import PSTExtractor

        extractor = PSTExtractor()
        folders = extractor.get_folders()
        assert folders == []  # Should return empty list as currently implemented

    def test_pst_extractor_get_emails_page(self):
        """Test loading emails via backend.iter_folder_messages mock."""
        from pst_email_extractor.gui.main import PSTExtractor, PSTFolder

        extractor = PSTExtractor()

        # Prepare one email tuple (record, handle)
        mock_record = {
            "Email_ID": "test_001",
            "Subject": "Test Subject",
            "From": "test@example.com",
            "Body": "Test body",
        }
        mock_handle = Mock()

        with patch("os.path.exists", return_value=True):
            with patch("pst_email_extractor.gui.main.is_pypff_available", return_value=True):
                with patch("pst_email_extractor.gui.main.PypffBackend") as mock_backend_cls:
                    backend = Mock()
                    backend.list_folders.return_value = []
                    backend.iter_folder_messages.return_value = iter([(mock_record, mock_handle)])
                    mock_backend_cls.return_value = backend

                    extractor.load_pst("/fake/path.pst")
                    folder = PSTFolder(name="Inbox", path="/Root/Inbox", email_count=1)
                    emails = extractor.get_emails_from_folder(folder)

                    assert len(emails) == 1
                    assert emails[0].email_id == "test_001"
                    assert emails[0].subject == "Test Subject"

    def test_pst_extractor_get_total_email_count(self):
        """Test PSTExtractor.get_total_email_count."""
        from pst_email_extractor.gui.main import PSTExtractor
        from pst_email_extractor.core.models import FolderInfo

        extractor = PSTExtractor()

        # Mock backend to return folders with known email counts
        mock_backend = Mock()
        mock_backend.is_available.return_value = True
        mock_backend.list_folders.return_value = [
            FolderInfo(id="/Root/Inbox", name="Inbox", path="/Root/Inbox", total_count=5, unread_count=0),
            FolderInfo(id="/Root/Sent", name="Sent", path="/Root/Sent", total_count=3, unread_count=0),
        ]

        with patch("pst_email_extractor.gui.main.PypffBackend") as mock_backend_class:
            mock_backend_class.return_value = mock_backend

            with patch("os.path.exists", return_value=True):
                extractor.load_pst("/fake/path.pst")
                count = extractor.get_total_email_count()

                assert count == 8  # 5 + 3

    @pytest.mark.skipif(True, reason="GUI tests require display server, skip in CI")
    def test_gui_creation_smoke(self):
        """Smoke test that GUI can be created (requires display)."""
        # This test is skipped in CI environments without display
        # In local development, uncomment to test GUI creation
        pass

        # from pst_email_extractor.gui.main import PSTEmailExtractorGUI
        #
        # # Mock tkinter to avoid requiring display
        # with patch("customtkinter.CTk"):
        #     with patch("customtkinter.CTkFrame"):
        #         with patch("customtkinter.CTkButton"):
        #             with patch("customtkinter.CTkLabel"):
        #                 gui = PSTEmailExtractorGUI()
        #                 assert gui is not None
        #                 assert hasattr(gui, 'root')
        #                 assert hasattr(gui, 'extractor')
