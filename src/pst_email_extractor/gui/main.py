"""
PST Email Extractor GUI Application

This module implements a comprehensive graphical user interface for the PST Email Extractor,
providing advanced email processing capabilities with real-time progress monitoring,
attachment content extraction, and AI-powered text processing.

Key Features:
- Interactive PST file loading and folder navigation
- Advanced email preview with attachment handling
- Real-time progress indicators with status-based color theming
- Configurable export options (JSON, CSV, EML, MBOX)
- Address analysis and deduplication capabilities
- AI-powered text sanitization and polishing

Architecture:
- Event-driven GUI using CustomTkinter framework
- Asynchronous processing using background threads
- Memory-efficient streaming for large PST files
- Modular component design for maintainability
"""

import builtins
import contextlib
import logging
import os
import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

from pst_email_extractor.core.attachment_processor import AttachmentContentOptions
from pst_email_extractor.core.backends.pypff import PypffBackend
from pst_email_extractor.core.models import ExtractionConfig, MessageHandle, ProgressUpdate
from pst_email_extractor.core.services import run_extraction
from pst_email_extractor.pst_parser import is_pypff_available

logger = logging.getLogger(__name__)


class StatusProgressBar(ctk.CTkFrame):
    """
    Professional progress bar with status-based color coding.

    A clean, understated progress indicator that provides clear visual feedback
    for long-running operations without excessive styling.
    """

    def __init__(self, parent, width=300, height=6, **kwargs):
        """
        Initialize the progress bar component.

        Args:
            parent: Parent widget container
            width: Progress bar width in pixels
            height: Progress bar height in pixels
            **kwargs: Additional CustomTkinter frame arguments
        """
        super().__init__(parent, fg_color="transparent", **kwargs)

        # Component dimensions and state
        self.progress_bar_width = width
        self.progress_bar_height = height
        self.progress_value = 0.0  # Current progress (0.0 to 1.0)
        self.operation_status = "idle"  # Current status: idle, loading, processing, success, error

        # Create Tkinter canvas for custom rendering
        self.canvas_widget = tk.Canvas(
            self,
            width=width,
            height=height,
            bg=self._get_background_color(),
            highlightthickness=0,
            bd=0
        )
        self.canvas_widget.pack()

        # Create progress fill rectangle
        self.progress_fill_rect = self.canvas_widget.create_rectangle(
            0, 0, 0, height,
            fill=self._get_progress_color(),
            outline="",
            width=0
        )

        # Render initial state
        self._render_progress_bar()

    def _get_background_color(self):
        """
        Retrieve the background color for the progress bar track.

        Returns:
            str: Hex color code for the background track
        """
        return "#E5E7EB"  # Light gray background track

    def _get_progress_color(self):
        """
        Retrieve the progress fill color based on current operation status.

        Returns:
            str: Hex color code for progress fill based on operation status
        """
        status_color_map = {
            "idle": "#D1D5DB",       # Gray - inactive state
            "loading": "#2563EB",    # Professional blue - file loading operations
            "processing": "#7C3AED", # Professional purple - data processing operations
            "success": "#059669",    # Professional green - completed successfully
            "error": "#DC2626"       # Professional red - operation failed
        }
        return status_color_map.get(self.operation_status, "#2563EB")

    def set_progress(self, value: float):
        """
        Update the progress bar fill level.

        Args:
            value: Progress value between 0.0 (empty) and 1.0 (complete)
        """
        self.progress_value = max(0.0, min(1.0, value))
        self._render_progress_bar()

    def set_status(self, status: str):
        """
        Update the operation status for color theming.

        Args:
            status: Operation status ('idle', 'loading', 'processing', 'success', 'error')
        """
        self.operation_status = status
        self._render_progress_bar()

    def _render_progress_bar(self):
        """
        Render the progress bar with current progress value and status theming.

        Updates the progress fill width and color based on current state.
        """
        # Update canvas background track
        self.canvas_widget.configure(bg=self._get_background_color())

        # Calculate pixel width of progress fill
        progress_fill_width = int(self.progress_bar_width * self.progress_value)

        # Update progress fill rectangle
        self.canvas_widget.coords(self.progress_fill_rect, 0, 0, progress_fill_width, self.progress_bar_height)
        self.canvas_widget.itemconfig(self.progress_fill_rect, fill=self._get_progress_color())

# Pre-computed color palette for sender avatar generation (performance optimization)
_AVATAR_COLOR_PALETTE = [
    '#dc3545', '#fd7e14', '#ffc107', '#28a745',
    '#20c997', '#17a2b8', '#6f42c1', '#007bff'
]


@dataclass
class EmailMessage:
    """
    Data model representing an email message in the GUI.

    Encapsulates all email metadata and content required for display
    and interaction within the graphical interface.

    Attributes:
        email_id: Unique identifier for the email message
        subject: Email subject line
        sender: Sender's email address
        recipients: List of recipient email addresses
        sent_date: Timestamp when email was sent
        body: Plain text email body content
        is_read: Read/unread status flag
        attachment_count: Number of attachments
        attachments: Detailed attachment metadata (lazy-loaded)
        _handle: Internal backend handle for attachment access
    """
    email_id: str
    subject: str
    sender: str
    recipients: list[str]
    sent_date: datetime | None
    body: str
    is_read: bool = False
    attachment_count: int = 0
    attachments: list[dict[str, Any]] | None = None
    _handle: MessageHandle | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'EmailMessage':
        """
        Factory method to create EmailMessage instance from parser data dictionary.

        Parses raw email data from the PST parser into a structured GUI model,
        handling date parsing and recipient field normalization.

        Args:
            data: Raw email data dictionary from PST parser

        Returns:
            EmailMessage: Structured email object for GUI display
        """
        # Parse sent date with error handling
        sent_timestamp = None
        if data.get("Date_Sent"):
            with contextlib.suppress(builtins.BaseException):
                # Convert ISO format string to datetime object
                sent_timestamp = datetime.fromisoformat(data["Date_Sent"].replace('Z', '+00:00'))

        # Normalize recipient addresses from To, CC, BCC fields
        recipient_addresses = []
        recipient_fields = ["To", "CC", "BCC"]
        for field_name in recipient_fields:
            field_value = data.get(field_name)
            if field_value:
                # Split comma-separated addresses and clean whitespace
                addresses = [addr.strip() for addr in str(field_value).split(',') if addr.strip()]
                recipient_addresses.extend(addresses)

        return cls(
            email_id=data.get("Email_ID", ""),
            subject=data.get("Subject", ""),
            sender=data.get("From", ""),
            recipients=recipient_addresses,
            sent_date=sent_timestamp,
            body=data.get("Body", ""),
            is_read=False  # GUI assumes unread by default
        )

    @property
    def sender_initials(self) -> str:
        """
        Generate sender initials for avatar display.

        Extracts initials from sender name for visual representation.
        Falls back to '?' for unknown senders.

        Returns:
            str: Two-character initials string
        """
        if not self.sender:
            return "?"

        # Split sender name into parts and extract non-empty components
        name_parts = [part.strip() for part in self.sender.split() if part.strip()]
        if not name_parts:
            return "?"

        # Generate initials: first letter of first name + first letter of last name
        if len(name_parts) >= 2:
            return (name_parts[0][0] + name_parts[-1][0]).upper()
        else:
            # Single name component
            return name_parts[0][0].upper()[:2]

    @property
    def avatar_color(self) -> str:
        """
        Generate consistent avatar background color based on sender identity.

        Uses sender initials to deterministically select from a predefined color palette,
        ensuring visual consistency for the same sender across the interface.

        Returns:
            str: Hex color code for avatar background
        """
        if not self.sender_initials or self.sender_initials == "?":
            return "#6c757d"  # Neutral gray for unknown senders

        # Extract first character for color mapping
        primary_char = self.sender_initials[0]

        # Map uppercase letters to color palette indices
        if 'A' <= primary_char <= 'Z':
            palette_index = (ord(primary_char) - 65) % len(_AVATAR_COLOR_PALETTE)
            return _AVATAR_COLOR_PALETTE[palette_index]
        elif 'a' <= primary_char <= 'z':
            # Handle lowercase (convert to uppercase equivalent)
            palette_index = (ord(primary_char) - 97) % len(_AVATAR_COLOR_PALETTE)
            return _AVATAR_COLOR_PALETTE[palette_index]

        # Fallback for non-alphabetic characters
        palette_index = hash(primary_char) % len(_AVATAR_COLOR_PALETTE)
        return _AVATAR_COLOR_PALETTE[palette_index]


@dataclass
class PSTFolder:
    """
    Data model representing a PST folder/mailbox in the GUI.

    Contains metadata for folder display and navigation within the
    hierarchical folder structure of PST files.

    Attributes:
        name: Display name of the folder
        path: Internal path/identifier for backend access
        email_count: Total number of emails in this folder
        unread_count: Number of unread emails in this folder
        favorite: User preference flag for quick access
    """
    name: str
    path: str
    email_count: int = 0
    unread_count: int = 0
    favorite: bool = False


class PSTExtractor:
    """
    High-level PST file extraction interface for GUI operations.

    Provides a simplified abstraction over the low-level PypffBackend,
    handling PST file loading, folder enumeration, and email retrieval
    with appropriate error handling and resource management.

    Attributes:
        pst_path: Path to currently loaded PST file
        is_loaded: Flag indicating successful PST file loading
        _backend: Internal PypffBackend instance for low-level operations
    """

    def __init__(self):
        """
        Initialize the PST extractor with default state.
        """
        self.pst_path: str | None = None
        self.is_loaded = False
        self._backend: PypffBackend | None = None

    def load_pst(self, file_path: str) -> bool:
        """
        Load and validate a PST file for extraction operations.

        Performs prerequisite checks and initializes the backend parser.
        Raises exceptions for missing dependencies or invalid files.

        Args:
            file_path: Absolute path to the PST file

        Returns:
            bool: True if loading successful

        Raises:
            Exception: If PyPFF library unavailable or file not found
        """
        if not is_pypff_available():
            raise Exception("PyPFF library not available. Please install libpff-python-ratom.")

        if not os.path.exists(file_path):
            raise Exception(f"PST file not found: {file_path}")

        self.pst_path = file_path
        self.is_loaded = True

        # Initialize and open backend with PST file
        self._backend = PypffBackend()
        self._backend.open(Path(file_path))

        return True

    def get_folders(self) -> list[PSTFolder]:
        """
        Retrieve the hierarchical folder structure from the loaded PST file.

        Queries the backend for folder metadata and converts to GUI models.
        Returns empty list on errors to ensure graceful degradation.

        Returns:
            list[PSTFolder]: List of folder objects with metadata
        """
        if not self.is_loaded or not self._backend:
            return []

        gui_folders: list[PSTFolder] = []
        try:
            # Enumerate all folders via backend
            for folder_info in self._backend.list_folders():
                folder_model = PSTFolder(
                    name=str(folder_info.name),
                    path=str(folder_info.id),
                    email_count=int(folder_info.total_count),
                    unread_count=int(folder_info.unread_count)
                )
                gui_folders.append(folder_model)
        except Exception as error:
            logger.error(f"Error enumerating PST folders: {error}")
            return []

        return gui_folders

    def get_total_email_count(self) -> int:
        """
        Calculate total email count across all folders in the PST file.

        Provides a high-level summary statistic for the loaded PST file.
        Used for progress estimation and user feedback.

        Returns:
            int: Total number of emails, or 0 on error
        """
        if not self.is_loaded or not self._backend:
            return 0

        try:
            # Sum email counts across all folders
            return sum(int(folder.total_count) for folder in self._backend.list_folders())
        except Exception:
            return 0

    def get_emails_from_folder(self, folder: PSTFolder) -> list[EmailMessage]:
        """
        Retrieve email messages from a specific folder with pagination.

        Loads a limited number of emails for performance, converting raw
        parser data into GUI-compatible EmailMessage objects.

        Args:
            folder: PSTFolder instance specifying target folder

        Returns:
            list[EmailMessage]: List of email objects for GUI display
        """
        if not self._backend:
            return []

        gui_emails: list[EmailMessage] = []
        try:
            # Retrieve paginated email data from backend
            for email_data, message_handle in self._backend.iter_folder_messages(
                folder.path, start=0, limit=50
            ):
                # Convert raw data to GUI model
                email_model = EmailMessage.from_dict(email_data)
                email_model.attachment_count = int(email_data.get("Attachment_Count", 0) or 0)
                email_model._handle = message_handle
                gui_emails.append(email_model)

        except Exception as error:
            logger.error(f"Error retrieving emails from folder '{folder.name}': {error}")
            return []

        return gui_emails


class PSTEmailExtractorGUI:
    """
    Main graphical user interface for PST Email Extractor.

    Provides a comprehensive desktop application for PST file analysis and email extraction
    with clean UI components, progress indicators, and advanced processing options.

    Key Components:
    - File loading interface with validation and error handling
    - Hierarchical folder navigation with collapsible groups
    - Email list view with read/unread indicators and avatar generation
    - Rich email preview with attachment viewing capabilities
    - Configurable export system (JSON, CSV, EML, MBOX formats)
    - Address analysis and deduplication features
    - AI-powered text processing and sanitization

    Threading Architecture:
    - Main thread handles UI updates and user interactions
    - Background threads perform file I/O and processing operations
    - Thread-safe progress callbacks for status updates
    """

    def __init__(self):
        """
        Initialize the GUI application with default configuration and UI setup.
        """
        # Configure CustomTkinter appearance
        ctk.set_appearance_mode("system")  # Follow system light/dark mode
        ctk.set_default_color_theme("blue")  # Professional blue theme

        # Initialize main application window
        self.root = ctk.CTk()
        self.root.title("PST Email Extractor")
        self.root.geometry("1400x900")  # Default window size
        self.root.minsize(1000, 700)  # Minimum window constraints

        # Title bar: solid title, set app icon from project root, force full opacity
        self.root.title("PST Email Extractor")
        try:
            # Ensure fully opaque window (no transparency effects)
            with contextlib.suppress(Exception):
                self.root.attributes('-alpha', 1.0)

            # Load icon from project root next to 'PST Email Extractor'
            project_root = Path(__file__).expanduser().resolve().parents[3]
            icon_path = project_root / "logo.ico"
            if icon_path.exists():
                self.root.iconbitmap(str(icon_path))
            else:
            logger.debug(f"Window icon not found at: {icon_path}")
        except Exception as e:
            logger.debug(f"Could not set window icon/attributes: {e}")

        # Initialize application state variables
        self.current_pst_path: str | None = None  # Path to loaded PST file
        self.extractor: PSTExtractor | None = None  # PST processing backend
        self.folders: list[PSTFolder] = []  # Available PST folders
        self.current_folder: PSTFolder | None = None  # Currently selected folder
        self.emails: list[EmailMessage] = []  # Emails in current folder
        self.selected_email: EmailMessage | None = None  # Currently viewed email
        self.attachment_content_options: AttachmentContentOptions | None = None

        # Persistent user settings for export operations
        self._settings: dict[str, Any] = {
            "ai_sanitize": False,          # PII sanitization enabled
            "ai_polish": False,            # Text polishing enabled
            "ai_language": "en-US",        # Language for AI processing
            "ai_neural_model_dir": None,   # Path to neural models
            "compress": False,             # Gzip compression for exports
            "html_index": False,           # Generate HTML index files
            "deduplicate": False,          # Remove duplicate emails
        }

        # Create UI components
        self._setup_ui()

        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def _setup_ui(self):
        """
        Initialize and configure all user interface components.

        Sets up the complete GUI layout including toolbar, content areas,
        and status bar with proper grid weight configuration.
        """
        # Create application toolbar with file operations and controls
        self._create_toolbar()

        # Create main content area with folder navigation and email views
        self._create_main_content()

        # Create bottom status bar for application feedback
        self._create_status_bar()

    def _create_toolbar(self):
        """
        Create the application toolbar with file operations and progress indicators.

        Constructs the top toolbar containing:
        - File operation buttons (Open PST, Export Data, Address Analysis)
        - Settings and configuration access
        - Progress bar with status-based coloring
        - Status text and visual feedback elements
        """
        # Toolbar frame
        toolbar_frame = ctk.CTkFrame(self.root, height=50)
        toolbar_frame.grid(row=0, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        toolbar_frame.grid_columnconfigure(10, weight=1)

        # File operations
        self.open_btn = ctk.CTkButton(
            toolbar_frame, text="Open PST",
            command=self._open_pst_file,
            width=120
        )
        self.open_btn.grid(row=0, column=0, padx=5, pady=5)

        # Separator
        separator1 = ctk.CTkFrame(toolbar_frame, width=2, height=30)
        separator1.grid(row=0, column=1, padx=5)

        # Export button (more prominent than dropdown)
        self.export_btn = ctk.CTkButton(
            toolbar_frame, text="Export Data",
            command=self._show_export_dialog,
            width=120,
            fg_color="#007acc",
            hover_color="#005999"
        )
        self.export_btn.grid(row=0, column=2, padx=5, pady=5)
        self.export_btn.configure(state="disabled")  # Disabled until data is loaded

        # Analyze addresses button
        self.address_btn = ctk.CTkButton(
            toolbar_frame, text="Analyze",
            command=self._show_address_dialog,
            width=100
        )
        self.address_btn.grid(row=0, column=3, padx=5, pady=5)
        self.address_btn.configure(state="disabled")

        # Settings button
        self.settings_btn = ctk.CTkButton(
            toolbar_frame, text="Settings",
            command=self._show_settings,
            width=100,
            state="disabled"
        )
        self.settings_btn.grid(row=0, column=4, padx=5, pady=5)

        # Progress indicator with status-based coloring
        self.progress_container = ctk.CTkFrame(toolbar_frame, fg_color="transparent")
        self.progress_container.grid(row=0, column=5, padx=5, pady=5)

        # Custom progress bar with status-based coloring
        self.progress_bar = StatusProgressBar(self.progress_container, width=300, height=6)
        self.progress_bar.pack(pady=(2, 0))

        # Progress label with clear status text
        self.progress_label = ctk.CTkLabel(
            toolbar_frame,
            text="Ready",
            font=ctk.CTkFont(size=11, weight="normal"),
            text_color="#8B949E"
        )
        self.progress_label.grid(row=0, column=6, padx=5, pady=5)

        # Status indicator (small circle)
        self.status_indicator = ctk.CTkLabel(
            toolbar_frame,
            text="●",
            font=ctk.CTkFont(size=12),
            text_color="#10B981"  # Green color
        )
        self.status_indicator.grid(row=0, column=7, padx=(0, 5), pady=5)

        # Start blinking animation for the status indicator
        self.status_blink_state = True
        self._animate_status_indicator()

    def _create_main_content(self):
        """Create the main content area"""
        # Left sidebar (folders/mailboxes)
        self._create_sidebar()

        # Main content area (email list and preview)
        self._create_email_area()

    def _create_sidebar(self):
        """Create the left sidebar with folder navigation and collapsible groups"""
        self.sidebar_frame = ctk.CTkFrame(self.root, width=280)
        self.sidebar_frame.grid(row=1, column=0, sticky="nsw", padx=5, pady=5)
        self.sidebar_frame.grid_rowconfigure(1, weight=1)
        self.sidebar_frame.grid_columnconfigure(0, weight=1)

        # Sidebar title
        sidebar_title = ctk.CTkLabel(
            self.sidebar_frame, text="Mailboxes",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        sidebar_title.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Folder container
        self.folder_container = ctk.CTkScrollableFrame(self.sidebar_frame)
        self.folder_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Initially show no folders message
        self.no_folders_label = ctk.CTkLabel(
            self.folder_container, text="No PST file loaded\n\nClick 'Open PST' to begin",
            font=ctk.CTkFont(size=12)
        )
        self.no_folders_label.pack(pady=50)

        # Initialize group expansion states
        self.group_expanded = {"Favorites": True, "All Folders": True}

    def _create_email_area(self):
        """Create the main email display area"""
        # Main content frame
        content_frame = ctk.CTkFrame(self.root)
        content_frame.grid(row=1, column=1, columnspan=2, sticky="nsew", padx=5, pady=5)
        content_frame.grid_rowconfigure(1, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        # Email list area (top half)
        self._create_email_list(content_frame)

        # Email preview area (bottom half)
        self._create_email_preview(content_frame)

    def _create_email_list(self, parent):
        """Create the email list view"""
        # Email list frame
        list_frame = ctk.CTkFrame(parent, height=300)
        list_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        list_frame.grid_columnconfigure(0, weight=1)

        # Email list title
        list_title = ctk.CTkLabel(
            list_frame, text="Messages",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        list_title.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Email list with scrollbar
        self.email_scrollable = ctk.CTkScrollableFrame(list_frame, height=250)
        self.email_scrollable.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # Initially show no emails message
        self.no_emails_label = ctk.CTkLabel(
            self.email_scrollable, text="No emails to display\n\nSelect a folder to view messages",
            font=ctk.CTkFont(size=12)
        )
        self.no_emails_label.pack(pady=50)

    def _create_email_preview(self, parent):
        """Create the email preview pane"""
        # Preview frame
        preview_frame = ctk.CTkFrame(parent)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        preview_frame.grid_rowconfigure(1, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        # Preview title
        preview_title = ctk.CTkLabel(
            preview_frame, text="Message Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        preview_title.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        # Preview content area
        self.preview_scrollable = ctk.CTkScrollableFrame(preview_frame)
        self.preview_scrollable.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Initially show no preview message
        self.no_preview_label = ctk.CTkLabel(
            self.preview_scrollable, text="No message selected\n\nClick on an email above to preview",
            font=ctk.CTkFont(size=12)
        )
        self.no_preview_label.pack(pady=50)

        # Attachment viewing area (initially hidden)
        self.attachment_frame = ctk.CTkFrame(self.preview_scrollable, fg_color="transparent")
        self.attachment_frame.pack_forget()

    def _create_status_bar(self):
        """Create the bottom status bar"""
        # Status bar frame
        status_frame = ctk.CTkFrame(self.root, height=30)
        status_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=5)
        status_frame.grid_columnconfigure(2, weight=1)

        # Status labels
        self.status_file = ctk.CTkLabel(status_frame, text="No file loaded", anchor="w")
        self.status_file.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        self.status_count = ctk.CTkLabel(status_frame, text="0 messages", anchor="center")
        self.status_count.grid(row=0, column=1, padx=10, pady=5)

        self.status_info = ctk.CTkLabel(status_frame, text="Ready", anchor="e")
        self.status_info.grid(row=0, column=2, padx=10, pady=5, sticky="e")

    def _show_address_dialog(self):
        """Run address analysis and present results with export options."""
        if not self.current_pst_path:
            self._show_error("No PST loaded")
            return
        # Basic modal dialog
        dlg = ctk.CTkToplevel(self.root)
        dlg.title("Address Analysis")
        dlg.geometry("700x500")
        dlg.transient(self.root)
        dlg.grab_set()

        header = ctk.CTkFrame(dlg)
        header.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(header, text="Analyzing unique addresses...", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w")

        body = ctk.CTkScrollableFrame(dlg)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        footer = ctk.CTkFrame(dlg)
        footer.pack(fill="x", padx=10, pady=10)
        export_json_btn = ctk.CTkButton(footer, text="Export JSON")
        export_csv_btn = ctk.CTkButton(footer, text="Export CSV")
        close_btn = ctk.CTkButton(footer, text="Close", command=dlg.destroy)
        close_btn.pack(side="right")
        export_csv_btn.pack(side="right", padx=(0, 8))
        export_json_btn.pack(side="right", padx=(0, 8))

        # Run analysis in background
        def _run_analysis():
            try:
                from pst_email_extractor.core.models import ExtractionConfig
                cfg = ExtractionConfig(
                    pst_path=Path(self.current_pst_path),
                    output_dir=Path("."),  # not used for analysis display
                    formats=["json"],
                    mode="addresses",
                    deduplicate=bool(self._settings.get("deduplicate", False)),
                )
                result = run_extraction(cfg)

                # Render simple summary table
                def _render():
                    for w in body.winfo_children():
                        w.destroy()
                    ctk.CTkLabel(body, text=f"Unique addresses: {result.address_count}").pack(anchor="w")
                    if result.host_count:
                        ctk.CTkLabel(body, text=f"Transport hosts: {result.host_count}").pack(anchor="w")

                    ctk.CTkLabel(body, text="Use export buttons to save full results to disk.").pack(anchor="w", pady=(8, 0))

                    def _export(fmt: str):
                        outdir = filedialog.askdirectory(title="Select Output Directory")
                        if not outdir:
                            return
                        # Re-run to generate files to selected directory
                        out_cfg = ExtractionConfig(
                            pst_path=Path(self.current_pst_path),
                            output_dir=Path(outdir),
                            formats=[fmt],
                            mode="addresses",
                            deduplicate=bool(self._settings.get("deduplicate", False)),
                        )
                        try:
                            export_res = run_extraction(out_cfg)
                            paths = "\n".join(str(p) for p in export_res.exported_paths)
                            messagebox.showinfo("Exported", f"Saved files:\n{paths}")
                        except Exception as e:
                            messagebox.showerror("Export failed", str(e))

                    export_json_btn.configure(command=lambda: _export("json"))
                    export_csv_btn.configure(command=lambda: _export("csv"))

                self.root.after(0, _render)
            except Exception as e:
                self.root.after(0, lambda e=e: self._show_error(f"Address analysis failed: {str(e)}"))

        t = threading.Thread(target=_run_analysis)
        t.daemon = True
        t.start()

    def _open_pst_file(self):
        """Open PST file dialog and load the file"""
        file_path = filedialog.askopenfilename(
            title="Select PST File",
            filetypes=[("PST files", "*.pst"), ("All files", "*.*")]
        )

        if file_path:
            self._load_pst_file(file_path)

    def _load_pst_file(self, file_path: str):
        """Load and process the PST file in a background thread"""
        self.current_pst_path = file_path
        self.status_file.configure(text=f"Loading: {os.path.basename(file_path)}")
        self.status_info.configure(text="Processing...")

        # Show progress bar
        self.progress_bar.set_progress(0)
        self.progress_bar.set_status("loading")
        self.progress_label.configure(text="Loading PST file...", text_color="#3B82F6")
        self.status_indicator.configure(text_color="#3B82F6")  # Blue during loading

        # Disable buttons during processing
        self.open_btn.configure(state="disabled")
        self.export_btn.configure(state="disabled")

        # Start background processing
        thread = threading.Thread(target=self._process_pst_file, args=(file_path,))
        thread.daemon = True
        thread.start()

    def _process_pst_file(self, file_path: str):
        """Process PST file in background thread"""
        try:
            # Update progress
            self.root.after(0, lambda: self.progress_label.configure(text="Initializing extractor...", text_color="#3B82F6"))
            self.root.after(0, lambda: self.status_indicator.configure(text_color="#3B82F6"))
            self.root.after(0, lambda: self.progress_bar.set_progress(0.1))

            # Create extractor
            self.extractor = PSTExtractor()

            # Load PST file
            self.root.after(0, lambda: self.progress_label.configure(text="Loading PST structure...", text_color="#3B82F6"))
            self.root.after(0, lambda: self.progress_bar.set_progress(0.3))

            success = self.extractor.load_pst(file_path)
            if not success:
                raise Exception("Failed to load PST file")

            # Get folders
            self.root.after(0, lambda: self.progress_label.configure(text="Reading folders...", text_color="#3B82F6"))
            self.root.after(0, lambda: self.progress_bar.set_progress(0.6))

            self.folders = self.extractor.get_folders()

            # Update UI on main thread
            self.root.after(0, self._update_folders_display)

            # Auto-select first folder if available to show emails
            if self.folders:
                self.root.after(0, lambda: self._select_folder(self.folders[0]))

            # Complete
            self.root.after(0, lambda: self.progress_bar.set_progress(1.0))
            self.root.after(0, lambda: self.progress_bar.set_status("success"))
            self.root.after(0, lambda: self.progress_label.configure(text="Complete", text_color="#10B981"))
            self.root.after(0, lambda: self.status_indicator.configure(text_color="#10B981"))  # Green for success
            self.root.after(0, lambda: self.status_info.configure(text="Ready"))

            # Enable settings after successful PST load
            self.root.after(0, lambda: self.settings_btn.configure(state="normal"))
            self.root.after(0, lambda: self.address_btn.configure(state="normal"))

            # Hide progress after a moment
            self.root.after(2000, self._hide_progress)

        except Exception as e:
            logger.error(f"Error processing PST file: {e}")
            self.root.after(0, lambda e=e: self._show_error(f"Error loading PST file: {str(e)}"))
            self.root.after(0, self._reset_ui_state)

    def _update_folders_display(self):
        """Update the folder list display with collapsible groups"""
        # Clear existing folders
        for widget in self.folder_container.winfo_children():
            widget.destroy()

        if not self.folders:
            no_folders = ctk.CTkLabel(
                self.folder_container, text="No folders found",
                font=ctk.CTkFont(size=12)
            )
            no_folders.pack(pady=20)
            return

        # Separate favorites and regular folders
        favorites = [f for f in self.folders if f.favorite]
        regular_folders = [f for f in self.folders if not f.favorite]

        # Create Favorites group
        if favorites:
            self._create_folder_group("Favorites", favorites)

        # Create All Folders group
        if regular_folders:
            self._create_folder_group("All Folders", regular_folders)

    def _create_folder_group(self, group_name: str, folders: list[PSTFolder]):
        """Create a collapsible folder group"""
        # Group header
        group_frame = ctk.CTkFrame(self.folder_container, fg_color="transparent")
        group_frame.pack(fill="x", padx=5, pady=(5, 0))

        # Group header with expand/collapse button
        header_frame = ctk.CTkFrame(group_frame, fg_color="transparent")
        header_frame.pack(fill="x")

        # Expand/collapse button
        expand_icon = "▼" if self.group_expanded.get(group_name, True) else "▶"
        expand_btn = ctk.CTkButton(
            header_frame,
            text=f"{expand_icon} {group_name}",
            command=lambda: self._toggle_group(group_name),
            anchor="w",
            height=25,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            font=ctk.CTkFont(size=11, weight="bold")
        )
        expand_btn.pack(fill="x", padx=5, pady=2)

        # Folder list (only if expanded)
        if self.group_expanded.get(group_name, True):
            folders_frame = ctk.CTkFrame(group_frame, fg_color="transparent")
            folders_frame.pack(fill="x", padx=10)

            for folder in folders:
                self._create_folder_button(folders_frame, folder)

    def _create_folder_button(self, parent, folder: PSTFolder):
        """Create an individual folder button with unread count"""
        # Folder button frame
        folder_frame = ctk.CTkFrame(parent, fg_color="transparent")
        folder_frame.pack(fill="x", pady=1)

        # Folder button
        display_text = folder.name
        if folder.unread_count > 0:
            display_text += f" ({folder.unread_count})"

        folder_btn = ctk.CTkButton(
            folder_frame,
            text=display_text,
            command=lambda f=folder: self._select_folder(f),
            anchor="w",
            height=28,
            fg_color="transparent",
            text_color=("gray10", "gray90"),
            font=ctk.CTkFont(size=10),
            hover_color=("#e3f2fd", "#1f538d")
        )
        folder_btn.pack(fill="x", padx=5)

        # Highlight current folder
        if self.current_folder and self.current_folder.path == folder.path:
            folder_btn.configure(
                fg_color=("#e3f2fd", "#1f538d"),
                text_color=("gray10", "gray90")
            )

    def _toggle_group(self, group_name: str):
        """Toggle expansion state of a folder group"""
        self.group_expanded[group_name] = not self.group_expanded.get(group_name, True)
        self._update_folders_display()

    def _select_folder(self, folder: PSTFolder):
        """Select a folder and load its emails"""
        self.current_folder = folder
        self.status_info.configure(text=f"Loading folder: {folder.name}")

        # Refresh folder display to show selection
        self._update_folders_display()

        # Load emails in background
        thread = threading.Thread(target=self._load_folder_emails, args=(folder,))
        thread.daemon = True
        thread.start()

    def _load_folder_emails(self, folder: PSTFolder):
        """Load emails for the selected folder with paging"""
        try:
            logger.debug(f"Loading emails for folder: {folder.name} (path: {folder.path})")
            self.emails = self.extractor.get_emails_from_folder(folder)
            logger.debug(f"Loaded {len(self.emails)} emails")
            # Prefer folder-specific count if available
            self.total_email_count = folder.email_count or self.extractor.get_total_email_count()

            # Update folder statistics
            folder.email_count = len(self.emails)
            folder.unread_count = sum(1 for email in self.emails if not email.is_read)

            # Enable export button if we have emails
            if self.emails:
                self.root.after(0, lambda: self.export_btn.configure(state="normal"))
            else:
                self.root.after(0, lambda: self.export_btn.configure(state="disabled"))

            # Update UI on main thread
            self.root.after(0, self._update_emails_display)
            self.root.after(0, self._update_folders_display)

            # Update status - show loaded/total count
            loaded_count = len(self.emails)
            if hasattr(self, 'total_email_count') and self.total_email_count and self.total_email_count > loaded_count:
                status_text = f"{loaded_count}/{self.total_email_count} messages loaded"
            else:
                status_text = f"{loaded_count} messages"
            self.root.after(0, lambda: self.status_count.configure(text=status_text))
            self.root.after(0, lambda: self.status_info.configure(text="Ready"))

        except Exception as e:
            logger.error(f"Error loading folder emails: {e}")
            self.root.after(0, lambda e=e: self._show_error(f"Error loading emails: {str(e)}"))

    def _update_emails_display(self):
        """Update the email list display with enhanced visual design"""
        # Clear existing emails
        for widget in self.email_scrollable.winfo_children():
            widget.destroy()

        if not self.emails:
            no_emails = ctk.CTkLabel(
                self.email_scrollable, text="No emails in this folder",
                font=ctk.CTkFont(size=12)
            )
            no_emails.pack(pady=20)
            return

        # Add emails to list with enhanced design
        for email in self.emails:
            # Create email item frame with selection state
            is_selected = self.selected_email and self.selected_email.email_id == email.email_id
            bg_color = "#e3f2fd" if is_selected else None  # Light blue for selected

            email_frame = ctk.CTkFrame(
                self.email_scrollable,
                height=70,
                fg_color=bg_color if bg_color else "transparent"
            )
            email_frame.pack(fill="x", padx=2, pady=1)
            email_frame._email_obj = email  # Attach email object for read status updates

            # Configure grid
            email_frame.grid_columnconfigure(1, weight=1)
            email_frame.grid_rowconfigure(0, weight=1)

            # Avatar circle (left side)
            avatar_frame = ctk.CTkFrame(
                email_frame,
                width=40,
                height=40,
                corner_radius=20,
                fg_color=email.avatar_color
            )
            avatar_frame.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nw")
            avatar_frame.grid_propagate(False)

            # Avatar initials
            avatar_label = ctk.CTkLabel(
                avatar_frame,
                text=email.sender_initials,
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color="white"
            )
            avatar_label.place(relx=0.5, rely=0.5, anchor="center")

            # Email content (right side)
            content_frame = ctk.CTkFrame(email_frame, fg_color="transparent")
            content_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=5)

            # Subject line (bold if unread)
            subject = email.subject or "(No Subject)"
            subject_font = ctk.CTkFont(
                size=12,
                weight="bold" if not email.is_read else "normal"
            )
            subject_color = "#1565c0" if not email.is_read else None  # Blue for unread

            subject_label = ctk.CTkLabel(
                content_frame,
                text=subject[:60] + "..." if len(subject) > 60 else subject,
                font=subject_font,
                text_color=subject_color,
                anchor="w"
            )
            subject_label.pack(fill="x", pady=(0, 2))

            # Sender and date row
            sender_date_frame = ctk.CTkFrame(content_frame, fg_color="transparent")
            sender_date_frame.pack(fill="x")

            sender = email.sender or "Unknown Sender"
            sender_label = ctk.CTkLabel(
                sender_date_frame,
                text=sender,
                font=ctk.CTkFont(size=10),
                text_color="#666666",
                anchor="w"
            )
            sender_label.pack(side="left")

            date = email.sent_date.strftime("%Y-%m-%d %H:%M") if email.sent_date else "Unknown Date"
            date_label = ctk.CTkLabel(
                sender_date_frame,
                text=date,
                font=ctk.CTkFont(size=10),
                text_color="#666666",
                anchor="e"
            )
            date_label.pack(side="right")

            # Preview text (truncated body)
            preview_text = ""
            if email.body:
                # Clean up the body text for preview
                body_clean = email.body.replace('\n', ' ').replace('\r', ' ').strip()
                preview_text = body_clean[:80] + "..." if len(body_clean) > 80 else body_clean

            if preview_text:
                preview_label = ctk.CTkLabel(
                    content_frame,
                    text=preview_text,
                    font=ctk.CTkFont(size=9),
                    text_color="#888888",
                    anchor="w"
                )
                preview_label.pack(fill="x", pady=(2, 0))

            # Make entire frame clickable
            def make_clickable(widget, email_obj):
                widget.bind("<Button-1>", lambda: self._select_email(email_obj))
                # Make children clickable too
                for child in widget.winfo_children():
                    make_clickable(child, email_obj)

            make_clickable(email_frame, email)

    def _select_email(self, email: EmailMessage):
        """Select an email and show preview"""
        # Mark as read when selected
        if not email.is_read:
            email.is_read = True
            # Update only the visual indicator for this email instead of full refresh
            self._update_email_read_status(email)

        # Only update preview if different email selected
        if self.selected_email != email:
            self.selected_email = email
            self._update_email_preview()

    def _update_email_read_status(self, email: EmailMessage):
        """Update only the read status visual indicator for a specific email"""
        # Find and update the specific email widget instead of full refresh
        for child in self.email_scrollable.winfo_children():
            if hasattr(child, '_email_obj') and child._email_obj == email:
                # Update the visual styling for read/unread status
                if email.is_read:
                    child.configure(fg_color="#f8f9fa")  # Read emails have lighter background
                else:
                    child.configure(fg_color="#ffffff")  # Unread emails have white background
                break

    def _update_email_preview(self):
        """Update the email preview display with enhanced professional layout"""
        # Clear existing preview
        for widget in self.preview_scrollable.winfo_children():
            widget.destroy()

        if not self.selected_email:
            return

        email = self.selected_email

        # Main email card container
        email_card = ctk.CTkFrame(self.preview_scrollable, fg_color="#f8f9fa")
        email_card.pack(fill="both", expand=True, padx=5, pady=5)

        # Subject header (top of card)
        subject_frame = ctk.CTkFrame(email_card, fg_color="#ffffff", height=60)
        subject_frame.pack(fill="x", padx=2, pady=(2, 0))
        subject_frame.pack_propagate(False)

        subject_label = ctk.CTkLabel(
            subject_frame,
            text=email.subject or "(No Subject)",
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="w"
        )
        subject_label.pack(fill="x", padx=15, pady=10)

        # Header section with avatar and actions
        header_frame = ctk.CTkFrame(email_card, fg_color="#ffffff")
        header_frame.pack(fill="x", padx=2, pady=(0, 2))

        # Avatar and sender info
        sender_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        sender_frame.pack(fill="x", padx=15, pady=10)

        # Avatar circle
        avatar_frame = ctk.CTkFrame(
            sender_frame,
            width=50,
            height=50,
            corner_radius=25,
            fg_color=email.avatar_color
        )
        avatar_frame.pack(side="left", padx=(0, 10))
        avatar_frame.pack_propagate(False)

        # Avatar initials
        avatar_label = ctk.CTkLabel(
            avatar_frame,
            text=email.sender_initials,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="white"
        )
        avatar_label.place(relx=0.5, rely=0.5, anchor="center")

        # Sender details
        details_frame = ctk.CTkFrame(sender_frame, fg_color="transparent")
        details_frame.pack(side="left", fill="x", expand=True)

        # Full sender info
        full_sender = f"{email.sender or 'Unknown Sender'}"
        if email.sender:
            full_sender += f" <{email.sender}>"

        sender_label = ctk.CTkLabel(
            details_frame,
            text=full_sender,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w"
        )
        sender_label.pack(fill="x", pady=(0, 2))

        # Recipients
        if email.recipients:
            recipients_text = f"To: {', '.join(email.recipients[:3])}"
            if len(email.recipients) > 3:
                recipients_text += f" (+{len(email.recipients) - 3} more)"
            recipients_label = ctk.CTkLabel(
                details_frame,
                text=recipients_text,
                font=ctk.CTkFont(size=11),
                text_color="#666666",
                anchor="w"
            )
            recipients_label.pack(fill="x", pady=(0, 2))

        # Date
        date_str = email.sent_date.strftime("%B %d, %Y at %I:%M %p") if email.sent_date else "Unknown Date"
        date_label = ctk.CTkLabel(
            details_frame,
            text=date_str,
            font=ctk.CTkFont(size=11),
            text_color="#666666",
            anchor="w"
        )
        date_label.pack(fill="x")

        # Action buttons (Reply, Reply All, Forward)
        actions_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        actions_frame.pack(fill="x", padx=15, pady=(0, 10))

        # Create action buttons
        reply_btn = ctk.CTkButton(
            actions_frame,
            text="Reply",
            width=80,
            height=30,
            font=ctk.CTkFont(size=11)
        )
        reply_btn.pack(side="right", padx=(5, 0))

        reply_all_btn = ctk.CTkButton(
            actions_frame,
            text="Reply All",
            width=90,
            height=30,
            font=ctk.CTkFont(size=11)
        )
        reply_all_btn.pack(side="right", padx=(5, 0))

        forward_btn = ctk.CTkButton(
            actions_frame,
            text="Forward",
            width=90,
            height=30,
            font=ctk.CTkFont(size=11)
        )
        forward_btn.pack(side="right", padx=(5, 0))

        # Separator
        separator = ctk.CTkFrame(email_card, height=1, fg_color="#e9ecef")
        separator.pack(fill="x", padx=2, pady=2)

        # Email body content area
        body_frame = ctk.CTkFrame(email_card, fg_color="#ffffff")
        body_frame.pack(fill="both", expand=True, padx=2, pady=(0, 2))

        if email.body:
            # Create a text box for better text handling
            body_textbox = ctk.CTkTextbox(
                body_frame,
                font=ctk.CTkFont(size=11),
                wrap="word",
                fg_color="transparent",
                border_width=0
            )
            body_textbox.pack(fill="both", expand=True, padx=15, pady=15)
            body_textbox.insert("0.0", email.body)
            body_textbox.configure(state="disabled")  # Read-only
        else:
            no_body = ctk.CTkLabel(
                body_frame,
                text="(No message body)",
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color="#666666"
            )
            no_body.pack(pady=50)

        # Attachments section (if any)
        if hasattr(email, 'attachments') and email.attachments:
            self._show_attachments(email_card, email.attachments)
        elif hasattr(email, 'attachment_count') and email.attachment_count > 0:
            # Lazy-load metadata from backend when available
            try:
                if getattr(email, '_handle', None) and self.extractor and getattr(self.extractor, '_backend', None):
                    meta = self.extractor._backend.list_attachments(email._handle)  # type: ignore[arg-type]
                    email.attachments = [{"index": a.index, "filename": a.filename, "size": a.size} for a in meta]
                    if email.attachments:
                        self._show_attachments(email_card, email.attachments)
                    else:
                        self._show_attachments_basic(email_card, email.attachment_count)
                else:
                    self._show_attachments_basic(email_card, email.attachment_count)
            except Exception:
                self._show_attachments_basic(email_card, email.attachment_count)

    def _show_attachments(self, parent, attachments):
        """Show detailed attachment information with view/edit options."""
        # Attachments header
        attachments_header = ctk.CTkFrame(parent, fg_color="#e9ecef", height=35)
        attachments_header.pack(fill="x", padx=2, pady=(0, 2))
        attachments_header.pack_propagate(False)

        header_label = ctk.CTkLabel(
            attachments_header,
            text=f"{len(attachments)} Attachment{'s' if len(attachments) != 1 else ''}",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        header_label.pack(fill="x", padx=15, pady=5)

        # Attachments container
        attachments_container = ctk.CTkFrame(parent, fg_color="#ffffff")
        attachments_container.pack(fill="x", padx=2, pady=(0, 2))

        for attachment in attachments:
            self._create_attachment_item(attachments_container, attachment)

    def _show_attachments_basic(self, parent, count):
        """Show basic attachment information when detailed info isn't available."""
        # Attachments header
        attachments_header = ctk.CTkFrame(parent, fg_color="#e9ecef", height=35)
        attachments_header.pack(fill="x", padx=2, pady=(0, 2))
        attachments_header.pack_propagate(False)

        header_label = ctk.CTkLabel(
            attachments_header,
            text=f"{count} Attachment{'s' if count != 1 else ''}",
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        header_label.pack(fill="x", padx=15, pady=5)

        # Basic info container
        basic_container = ctk.CTkFrame(parent, fg_color="#ffffff")
        basic_container.pack(fill="x", padx=2, pady=(0, 2))

        info_label = ctk.CTkLabel(
            basic_container,
            text="Attachment details not available in preview.\nUse export to access full attachment data.",
            font=ctk.CTkFont(size=11),
            text_color="#666666"
        )
        info_label.pack(padx=15, pady=15)

    def _create_attachment_item(self, parent, attachment):
        """Create an individual attachment item with view/edit options."""
        # Attachment item frame
        item_frame = ctk.CTkFrame(parent, fg_color="#f8f9fa", height=50)
        item_frame.pack(fill="x", padx=10, pady=2)
        item_frame.pack_propagate(False)

        # Icon (based on file type)
        icon_label = ctk.CTkLabel(
            item_frame,
            text=self._get_attachment_icon(attachment.get('filename', '')),
            font=ctk.CTkFont(size=16)
        )
        icon_label.pack(side="left", padx=(10, 5))

        # File info
        info_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        info_frame.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # Filename
        filename = attachment.get('filename', 'Unknown File')
        filename_label = ctk.CTkLabel(
            info_frame,
            text=filename,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        filename_label.pack(fill="x", pady=(2, 0))

        # File size (if available)
        size_info = ""
        if 'size' in attachment:
            size_info = f"Size: {self._format_file_size(attachment['size'])}"

        size_label = ctk.CTkLabel(
            info_frame,
            text=size_info,
            font=ctk.CTkFont(size=10),
            text_color="#666666",
            anchor="w"
        )
        size_label.pack(fill="x", pady=(0, 2))

        # Action buttons
        actions_frame = ctk.CTkFrame(item_frame, fg_color="transparent")
        actions_frame.pack(side="right", padx=(0, 10))

        # View button
        view_btn = ctk.CTkButton(
            actions_frame,
            text="View",
            width=60,
            height=25,
            font=ctk.CTkFont(size=10),
            command=lambda: self._view_attachment(attachment)
        )
        view_btn.pack(side="right", padx=(5, 0))

        # Save button
        save_btn = ctk.CTkButton(
            actions_frame,
            text="Save",
            width=60,
            height=25,
            font=ctk.CTkFont(size=10),
            fg_color="#28a745",
            hover_color="#218838",
            command=lambda: self._save_attachment(attachment)
        )
        save_btn.pack(side="right", padx=(5, 0))

    def _get_attachment_icon(self, filename):
        """Get appropriate icon for file type."""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''

        icon_map = {
            'pdf': 'PDF',
            'doc': 'DOC', 'docx': 'DOCX',
            'xls': 'XLS', 'xlsx': 'XLSX',
            'ppt': 'PPT', 'pptx': 'PPTX',
            'txt': 'TXT',
            'jpg': 'IMG', 'jpeg': 'IMG', 'png': 'IMG', 'gif': 'IMG',
            'zip': 'ZIP', 'rar': 'RAR',
            'exe': 'EXE',
        }
        return icon_map.get(ext, 'FILE')

    def _format_file_size(self, size_bytes):
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"

        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1

        return f"{size_bytes:.1f} {size_names[i]}"

    def _view_attachment(self, attachment):
        """View attachment (opens with default application)."""
        filename = attachment.get('filename', '')
        if not filename:
            messagebox.showerror("Error", "Attachment filename not available.")
            return

        try:
            # Stream bytes to a temp file and open
            if not self.selected_email or not getattr(self.selected_email, '_handle', None):
                messagebox.showerror("Error", "Attachment handle not available.")
                return
            handle = self.selected_email._handle
            backend = self.extractor._backend if self.extractor else None
            if not backend or not handle:
                messagebox.showerror("Error", "Backend not available.")
                return
            idx = int(attachment.get('index', 0))
            data = backend.read_attachment_bytes(handle, idx)
            if not data:
                messagebox.showerror("Error", "Could not read attachment bytes.")
                return
            import os
            import subprocess
            import sys
            import tempfile
            suffix = ('.' + filename.split('.')[-1]) if '.' in filename else ''
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
                tf.write(data)
                temp_path = tf.name
            try:
                os.startfile(temp_path)  # type: ignore[attr-defined]
            except Exception:
                if sys.platform == 'darwin':
                    subprocess.call(['open', temp_path])
                else:
                    subprocess.call(['xdg-open', temp_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not view attachment: {str(e)}")

    def _save_attachment(self, attachment):
        """Save attachment to disk."""
        filename = attachment.get('filename', '')
        if not filename:
            messagebox.showerror("Error", "Attachment filename not available.")
            return

        # Ask for save location
        save_path = filedialog.asksaveasfilename(
            defaultextension="",
            initialfile=filename,
            title="Save Attachment As"
        )

        if save_path:
            try:
                if not self.selected_email or not getattr(self.selected_email, '_handle', None):
                    messagebox.showerror("Error", "Attachment handle not available.")
                    return
                handle = self.selected_email._handle
                backend = self.extractor._backend if self.extractor else None
                if not backend or not handle:
                    messagebox.showerror("Error", "Backend not available.")
                    return
                idx = int(attachment.get('index', 0))
                data = backend.read_attachment_bytes(handle, idx)
                if not data:
                    messagebox.showerror("Error", "Could not read attachment bytes.")
                    return
                with open(save_path, 'wb') as f:
                    f.write(data)
                messagebox.showinfo("Success", f"Attachment saved to:\n{save_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save attachment: {str(e)}")

    def _show_export_dialog(self):
        """Show export dialog with format selection and AI options."""
        if not self.emails:
            messagebox.showwarning("No Data", "No emails to export. Please load a PST file first.")
            return

        # Create export dialog
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Export Options")
        dialog.geometry("500x400")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        # Title
        title_label = ctk.CTkLabel(dialog, text="Export Email Data", font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=20)

        # Format selection
        format_frame = ctk.CTkFrame(dialog)
        format_frame.pack(fill="x", padx=20, pady=10)

        format_label = ctk.CTkLabel(format_frame, text="Export Format:", font=ctk.CTkFont(weight="bold"))
        format_label.pack(anchor="w", padx=10, pady=5)

        self.export_format = ctk.StringVar(value="json")
        json_radio = ctk.CTkRadioButton(format_frame, text="JSON", variable=self.export_format, value="json")
        json_radio.pack(anchor="w", padx=20, pady=2)
        csv_radio = ctk.CTkRadioButton(format_frame, text="CSV", variable=self.export_format, value="csv")
        csv_radio.pack(anchor="w", padx=20, pady=2)
        eml_radio = ctk.CTkRadioButton(format_frame, text="EML (Individual Files)", variable=self.export_format, value="eml")
        eml_radio.pack(anchor="w", padx=20, pady=2)
        mbox_radio = ctk.CTkRadioButton(format_frame, text="MBOX", variable=self.export_format, value="mbox")
        mbox_radio.pack(anchor="w", padx=20, pady=2)

        # AI Processing options
        ai_frame = ctk.CTkFrame(dialog)
        ai_frame.pack(fill="x", padx=20, pady=10)

        ai_label = ctk.CTkLabel(ai_frame, text="AI Processing:", font=ctk.CTkFont(weight="bold"))
        ai_label.pack(anchor="w", padx=10, pady=5)

        self.ai_sanitize_var = ctk.BooleanVar(value=False)

        sanitize_check = ctk.CTkCheckBox(ai_frame, text="Clean & Polish (PII removal + spelling & grammar)",
                                        variable=self.ai_sanitize_var)
        sanitize_check.pack(anchor="w", padx=20, pady=2)

        # Deduplication option (export-scoped)
        dedup_frame = ctk.CTkFrame(dialog)
        dedup_frame.pack(fill="x", padx=20, pady=0)
        self.dedup_var = ctk.BooleanVar(value=bool(getattr(self, "_settings", {}).get("deduplicate", False)))
        dedup_check = ctk.CTkCheckBox(dedup_frame, text="Deduplicate emails (hash-based)", variable=self.dedup_var)
        dedup_check.pack(anchor="w", padx=0, pady=2)

        # Buttons
        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)

        # Advanced settings button
        advanced_btn = ctk.CTkButton(
            button_frame,
            text="Advanced...",
            command=lambda: self._show_advanced_settings(dialog),
            width=100
        )
        advanced_btn.pack(side="left")

        cancel_btn = ctk.CTkButton(button_frame, text="Cancel", command=dialog.destroy, width=100)
        cancel_btn.pack(side="right", padx=10)

        export_btn = ctk.CTkButton(button_frame, text="Export", command=lambda: self._start_export(dialog),
                                  width=100, fg_color="#007acc", hover_color="#005999")
        export_btn.pack(side="right")

    def _show_advanced_settings(self, parent_dialog):
        """Show advanced settings dialog."""
        dialog = AdvancedSettingsDialog(parent_dialog)
        # Wait for dialog to close, then store options
        parent_dialog.wait_window(dialog.dialog)
        self.attachment_content_options = dialog.get_options()

    def _start_export(self, dialog):
        """Start the export process from the dialog."""
        dialog.destroy()
        export_type = self.export_format.get()

        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir:
            return

        # Effective AI flags: checkbox takes precedence, else use saved settings
        ai_checkbox = self.ai_sanitize_var.get()
        ai_sanitize = ai_checkbox or bool(self._settings.get("ai_sanitize", False))
        ai_polish = ai_checkbox or bool(self._settings.get("ai_polish", False))

        # Deduplication: dialog checkbox overrides persisted setting for this export
        if hasattr(self, "dedup_var"):
            self._settings["deduplicate"] = bool(self.dedup_var.get())

        # Export in background thread
        thread = threading.Thread(
            target=self._perform_export,
            args=(export_type, output_dir, ai_sanitize, ai_polish)
        )
        thread.daemon = True
        thread.start()

    def _perform_export(self, export_type: str, output_dir: str, ai_sanitize: bool = False, ai_polish: bool = False):
        """Perform the export operation"""
        try:
            self.root.after(0, lambda: self.progress_label.configure(text="Exporting...", text_color="#8B5CF6"))
            self.root.after(0, lambda: self.progress_bar.set_progress(0.1))
            self.root.after(0, lambda: self.progress_bar.set_status("processing"))
            self.root.after(0, lambda: self.status_indicator.configure(text_color="#8B5CF6"))
            self.root.after(0, lambda: self.export_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.open_btn.configure(state="disabled"))

            fmt = export_type.lower()
            formats = [fmt]
            attachments_needed = fmt in {"eml", "mbox"}

            # Map attachment settings if present
            attach_opts = self.attachment_content_options
            extract_attach_content = attach_opts is not None
            if isinstance(self._settings.get("attachments"), dict):
                a = self._settings.get("attachments") or {}
                try:
                    from pst_email_extractor.core.attachment_processor import AttachmentContentOptions
                    attach_opts = AttachmentContentOptions(
                        enable_ocr=bool(a.get("ocr", True)),
                        ocr_languages=list(a.get("langs", ["eng"])),
                        max_file_size_mb=int(a.get("max_mb", 50)),
                    ) if bool(a.get("extract", False)) else None
                    extract_attach_content = attach_opts is not None
                except Exception:
                    pass

            cfg = ExtractionConfig(
                pst_path=Path(self.current_pst_path or ""),
                output_dir=Path(output_dir),
                formats=formats,
                mode="extract",
                deduplicate=bool(self._settings.get("deduplicate", False)),
                extract_attachments=attachments_needed,
                attachments_dir=None,
                log_file=None,
                html_index=bool(self._settings.get("html_index", False)),
                ai_sanitize=ai_sanitize,
                ai_polish=ai_polish,
                ai_language=str(self._settings.get("ai_language", "en-US")),
                ai_neural_model_dir=(Path(self._settings["ai_neural_model_dir"]) if self._settings.get("ai_neural_model_dir") else None),
                extract_attachment_content=extract_attach_content,
                attachment_content_options=attach_opts,
                compress=bool(self._settings.get("compress", False)),
            )

            def _progress_cb(update, total: int | None = None, message: str | None = None):
                # Support object or tuple styles
                if isinstance(update, ProgressUpdate):
                    current = update.current
                    total_v = update.total
                    msg = update.message
                else:
                    current = int(update)
                    total_v = int(total or 0)
                    msg = message or ""

                def _ui():
                    if total_v and total_v > 0:
                        self.progress_bar.set_progress(max(0.0, min(1.0, current / total_v)))
                    self.progress_label.configure(text=msg)
                self.root.after(0, _ui)

            result = run_extraction(cfg, progress_callback=_progress_cb)

            # Complete
            self.root.after(0, lambda: self.progress_bar.set_progress(1.0))
            self.root.after(0, lambda: self.progress_bar.set_status("success"))
            self.root.after(0, lambda: self.progress_label.configure(text="Export complete", text_color="#10B981"))
            self.root.after(0, lambda: self.status_indicator.configure(text_color="#10B981"))
            def _notify():
                if result and result.exported_paths:
                    paths = "\n".join(str(p) for p in result.exported_paths)
                    messagebox.showinfo("Success", f"Exported files:\n{paths}")
            self.root.after(0, _notify)

            # Re-enable buttons and reset progress
            self.root.after(0, lambda: self.export_btn.configure(state="normal"))
            self.root.after(0, lambda: self.open_btn.configure(state="normal"))
            self.root.after(2000, lambda: self._reset_progress())

        except Exception as e:
            logger.error(f"Export error: {e}")
            self.root.after(0, lambda e=e: self._show_error(f"Export failed: {str(e)}"))
            self.root.after(0, self._hide_progress)

    def _show_settings(self):
        """Show settings dialog (AI/Export options)."""
        dialog = SettingsDialog(self.root, initial=self._settings)
        self.root.wait_window(dialog.dialog)
        updated = dialog.get_settings()
        if updated:
            self._settings.update(updated)

    def _show_error(self, message: str):
        """Show error message"""
        messagebox.showerror("Error", message)

    def _hide_progress(self):
        """Hide or reset progress indicators"""
        try:
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar.set_progress(0)
                self.progress_bar.set_status("idle")
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.configure(text="Ready", text_color="#8B949E")
            if hasattr(self, 'status_indicator') and self.status_indicator:
                self.status_indicator.configure(text_color="#6B7280")
        except (RuntimeError, AttributeError) as exc:
            # Widget may be destroyed due to shutdown/teardown; ignore but trace
            logger.debug(f"Progress widgets unavailable during hide: {exc}")

    def _reset_progress(self):
        """Reset progress indicators to ready state"""
        self.progress_bar.set_progress(0)
        self.progress_bar.set_status("idle")
        self.progress_label.configure(text="Ready", text_color="#8B949E")
        self.status_indicator.configure(text_color="#10B981")  # Keep green for ready state

    def _create_custom_icon(self):
        """No-op placeholder: we now always use project-root logo.ico and no animations."""
        return

    def _start_window_animation(self):
        """Disabled: animations removed to keep window fully opaque."""
        return

    def _animate_status_indicator(self):
        """Animate the status indicator with blinking effect"""
        if hasattr(self, 'status_indicator') and self.status_indicator:
            # Toggle visibility by changing opacity/color
            if self.status_blink_state:
                self.status_indicator.configure(text_color="#10B981")  # Green
            else:
                self.status_indicator.configure(text_color="#6B7280")  # Gray (invisible effect)

            self.status_blink_state = not self.status_blink_state

            # Schedule next animation frame (blink every 800ms)
            self.root.after(800, self._animate_status_indicator)

    def _reset_ui_state(self):
        """Reset UI to initial state after error"""
        self.open_btn.configure(state="normal")
        self.export_btn.configure(state="normal")
        self.status_info.configure(text="Ready")
        self._hide_progress()

    def run(self):
        """Start the GUI application"""
        self.root.mainloop()


class AdvancedSettingsDialog:
    """Advanced settings dialog for attachment content extraction."""

    def __init__(self, parent):
        self.parent = parent
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Advanced Export Settings")
        self.dialog.geometry("500x450")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # Initialize variables with defaults
        self.extract_content = ctk.BooleanVar(value=False)
        self.enable_ocr = ctk.BooleanVar(value=True)
        self.ocr_languages = ctk.StringVar(value="eng")
        self.max_size = ctk.IntVar(value=50)

        self._create_ui()
        self._center_dialog()

    def _create_ui(self):
        """Create the dialog UI components."""
        # Title
        title_label = ctk.CTkLabel(
            self.dialog,
            text="Attachment Content Extraction",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.pack(pady=20)

        # Main container
        container = ctk.CTkFrame(self.dialog)
        container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        # Enable content extraction
        enable_frame = ctk.CTkFrame(container)
        enable_frame.pack(fill="x", padx=20, pady=10)

        enable_label = ctk.CTkLabel(
            enable_frame,
            text="Extract text content from attachments:",
            font=ctk.CTkFont(weight="bold")
        )
        enable_label.pack(anchor="w", padx=10, pady=5)

        enable_check = ctk.CTkCheckBox(
            enable_frame,
            text="Enable automatic text extraction from attachments (PDF, DOCX, images via OCR, etc.)",
            variable=self.extract_content,
            command=self._toggle_content_options
        )
        enable_check.pack(anchor="w", padx=20, pady=5)

        # Content options frame
        self.options_frame = ctk.CTkFrame(container)
        self.options_frame.pack(fill="x", padx=20, pady=10)

        # OCR settings
        ocr_frame = ctk.CTkFrame(self.options_frame)
        ocr_frame.pack(fill="x", padx=10, pady=5)

        ocr_check = ctk.CTkCheckBox(
            ocr_frame,
            text="Enable OCR for images and scanned documents",
            variable=self.enable_ocr
        )
        ocr_check.pack(anchor="w", padx=10, pady=5)

        # OCR languages
        lang_label = ctk.CTkLabel(ocr_frame, text="OCR Languages:")
        lang_label.pack(anchor="w", padx=30, pady=(5, 0))

        lang_entry = ctk.CTkEntry(
            ocr_frame,
            textvariable=self.ocr_languages,
            placeholder_text="eng,spa,fra"
        )
        lang_entry.pack(fill="x", padx=30, pady=(0, 10))

        # Max file size
        size_frame = ctk.CTkFrame(self.options_frame)
        size_frame.pack(fill="x", padx=10, pady=5)

        size_label = ctk.CTkLabel(size_frame, text="Maximum attachment size to process:")
        size_label.pack(anchor="w", padx=10, pady=5)

        size_entry = ctk.CTkEntry(
            size_frame,
            textvariable=self.max_size,
            placeholder_text="50"
        )
        size_entry.pack(fill="x", padx=10, pady=(0, 5))

        size_mb_label = ctk.CTkLabel(size_frame, text="MB (attachments larger than this will be skipped)")
        size_mb_label.pack(anchor="w", padx=10, pady=(0, 10))

        # Supported formats info
        info_frame = ctk.CTkFrame(self.options_frame)
        info_frame.pack(fill="x", padx=10, pady=5)

        info_label = ctk.CTkLabel(
            info_frame,
            text="Supported formats:\n• PDF (text extraction, OCR fallback)\n• DOCX (native text extraction)\n• Images (JPEG, PNG, TIFF via OCR)\n• Plain text files\n• Embedded messages (MSG)",
            font=ctk.CTkFont(size=11)
        )
        info_label.pack(anchor="w", padx=10, pady=10)

        # Buttons
        button_frame = ctk.CTkFrame(self.dialog, fg_color="transparent")
        button_frame.pack(fill="x", padx=20, pady=20)

        cancel_btn = ctk.CTkButton(
            button_frame,
            text="Cancel",
            command=self.dialog.destroy,
            width=100
        )
        cancel_btn.pack(side="right", padx=(10, 0))

        ok_btn = ctk.CTkButton(
            button_frame,
            text="OK",
            command=self._apply_settings,
            width=100
        )
        ok_btn.pack(side="right")

        # Initial state
        self._toggle_content_options()

    def _toggle_content_options(self):
        """Enable/disable content options based on main checkbox."""
        state = "normal" if self.extract_content.get() else "disabled"
        for child in self.options_frame.winfo_children():
            self._set_widget_state(child, state)

    def _set_widget_state(self, widget, state):
        """Recursively set widget state."""
        with contextlib.suppress(builtins.BaseException):
            widget.configure(state=state)
        for child in widget.winfo_children():
            self._set_widget_state(child, state)

    def _center_dialog(self):
        """Center the dialog on the parent window."""
        self.dialog.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.dialog.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.dialog.winfo_height() // 2)
        self.dialog.geometry(f"+{x}+{y}")

    def _apply_settings(self):
        """Apply the settings and close dialog."""
        self.dialog.destroy()

    def get_options(self) -> AttachmentContentOptions | None:
        """Get the configured attachment content options."""
        if not self.extract_content.get():
            return None

        return AttachmentContentOptions(
            enable_ocr=self.enable_ocr.get(),
            ocr_languages=[lang.strip() for lang in self.ocr_languages.get().split(",") if lang.strip()],
            max_file_size_mb=self.max_size.get()
        )


def launch_gui():
    """Launch the GUI application"""
    try:
        app = PSTEmailExtractorGUI()
        app.run()
    except Exception as e:
        logger.error(f"GUI Error: {e}")
        messagebox.showerror("Application Error", f"Failed to start GUI: {str(e)}")


class SettingsDialog:
    """Settings dialog for AI and Export options."""

    def __init__(self, parent, initial: dict[str, Any] | None = None):
        self.parent = parent
        self.dialog = ctk.CTkToplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("600x420")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        initial = initial or {}

        # Container
        container = ctk.CTkFrame(self.dialog)
        container.pack(fill="both", expand=True, padx=12, pady=12)

        # AI section
        ai_frame = ctk.CTkFrame(container)
        ai_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(ai_frame, text="AI Processing", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=8, pady=6)
        self.ai_sanitize = ctk.BooleanVar(value=bool(initial.get("ai_sanitize", False)))
        self.ai_polish = ctk.BooleanVar(value=bool(initial.get("ai_polish", False)))
        self.ai_language = ctk.StringVar(value=str(initial.get("ai_language", "en-US")))
        self.ai_model_dir = ctk.StringVar(value=str(initial.get("ai_neural_model_dir") or ""))
        ctk.CTkCheckBox(ai_frame, text="Sanitize PII", variable=self.ai_sanitize).pack(anchor="w", padx=12)
        ctk.CTkCheckBox(ai_frame, text="Polish text (spell/grammar)", variable=self.ai_polish).pack(anchor="w", padx=12)
        lang_row = ctk.CTkFrame(ai_frame)
        lang_row.pack(fill="x", padx=12, pady=(6, 0))
        ctk.CTkLabel(lang_row, text="Language code:").pack(side="left")
        ctk.CTkEntry(lang_row, textvariable=self.ai_language, width=150, placeholder_text="en-US").pack(side="left", padx=(8, 0))
        model_row = ctk.CTkFrame(ai_frame)
        model_row.pack(fill="x", padx=12, pady=(6, 8))
        ctk.CTkLabel(model_row, text="Neural model dir (ONNX):").pack(anchor="w")
        ctk.CTkEntry(model_row, textvariable=self.ai_model_dir, placeholder_text="path to model dir").pack(fill="x")

        # Export section
        ex_frame = ctk.CTkFrame(container)
        ex_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(ex_frame, text="Export Options", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=8, pady=6)
        self.compress = ctk.BooleanVar(value=bool(initial.get("compress", False)))
        self.html_index = ctk.BooleanVar(value=bool(initial.get("html_index", False)))
        self.deduplicate = ctk.BooleanVar(value=bool(initial.get("deduplicate", False)))
        ctk.CTkCheckBox(ex_frame, text="Compress JSON/CSV (gzip)", variable=self.compress).pack(anchor="w", padx=12)
        ctk.CTkCheckBox(ex_frame, text="Generate HTML index", variable=self.html_index).pack(anchor="w", padx=12)
        ctk.CTkCheckBox(ex_frame, text="Deduplicate emails", variable=self.deduplicate).pack(anchor="w", padx=12, pady=(0, 6))

        # Attachments section (basic)
        at_frame = ctk.CTkFrame(container)
        at_frame.pack(fill="x", pady=(0, 10))
        ctk.CTkLabel(at_frame, text="Attachments", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=8, pady=6)
        self.attach_extract = ctk.BooleanVar(value=False)
        self.attach_ocr = ctk.BooleanVar(value=True)
        self.attach_langs = ctk.StringVar(value="eng")
        self.attach_max = ctk.IntVar(value=50)
        ctk.CTkCheckBox(at_frame, text="Extract text from attachments", variable=self.attach_extract).pack(anchor="w", padx=12)
        ctk.CTkCheckBox(at_frame, text="Enable OCR for images/PDFs", variable=self.attach_ocr).pack(anchor="w", padx=12)
        row_lang = ctk.CTkFrame(at_frame)
        row_lang.pack(fill="x", padx=12, pady=(6, 0))
        ctk.CTkLabel(row_lang, text="OCR languages (comma-separated)").pack(anchor="w")
        ctk.CTkEntry(row_lang, textvariable=self.attach_langs, placeholder_text="eng,spa,fra").pack(fill="x")
        row_max = ctk.CTkFrame(at_frame)
        row_max.pack(fill="x", padx=12, pady=(6, 0))
        ctk.CTkLabel(row_max, text="Max attachment size (MB)").pack(anchor="w")
        ctk.CTkEntry(row_max, textvariable=self.attach_max, placeholder_text="50").pack(fill="x")

        # Footer
        footer = ctk.CTkFrame(self.dialog)
        footer.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(footer, text="Apply", command=self._apply, width=100).pack(side="right", padx=(0, 8))
        ctk.CTkButton(footer, text="Close", command=self.dialog.destroy, width=100).pack(side="right")

        self._saved: dict[str, Any] | None = None

    def _apply(self) -> None:
        self._saved = {
            "ai_sanitize": self.ai_sanitize.get(),
            "ai_polish": self.ai_polish.get(),
            "ai_language": self.ai_language.get().strip() or "en-US",
            "ai_neural_model_dir": self.ai_model_dir.get().strip() or None,
            "compress": self.compress.get(),
            "html_index": self.html_index.get(),
            "deduplicate": self.deduplicate.get(),
            "attachments": {
                "extract": self.attach_extract.get(),
                "ocr": self.attach_ocr.get(),
                "langs": [lang.strip() for lang in self.attach_langs.get().split(",") if lang.strip()],
                "max_mb": self.attach_max.get(),
            },
        }
        self.dialog.destroy()

    def get_settings(self) -> dict[str, Any] | None:
        return self._saved