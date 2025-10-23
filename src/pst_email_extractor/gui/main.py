"""
PST Email Extractor GUI 
"""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
from pathlib import Path
import threading
from typing import Optional, Dict, List, Any
from datetime import datetime
from dataclasses import dataclass
import logging

from ..pst_parser import is_pypff_available
from ..core.services import run_extraction
from ..core.backends.pypff import PypffBackend
from ..core.models import ExtractionConfig, ProgressUpdate, MessageHandle
from ..core.attachment_processor import AttachmentContentOptions

logger = logging.getLogger(__name__)

# Pre-computed avatar color palette for performance
_AVATAR_COLORS = ['#dc3545', '#fd7e14', '#ffc107', '#28a745',
                  '#20c997', '#17a2b8', '#6f42c1', '#007bff']


@dataclass
class EmailMessage:
    """Simple email data model for GUI"""
    email_id: str
    subject: str
    sender: str
    recipients: List[str]
    sent_date: Optional[datetime]
    body: str
    is_read: bool = False
    attachment_count: int = 0
    attachments: list[dict[str, Any]] | None = None
    _handle: MessageHandle | None = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmailMessage':
        """Create EmailMessage from dictionary returned by pst_parser"""
        sent_date = None
        if data.get("Date_Sent"):
            try:
                sent_date = datetime.fromisoformat(data["Date_Sent"].replace('Z', '+00:00'))
            except:
                pass

        # Parse recipients from To, CC, BCC fields
        recipients = []
        for field in ["To", "CC", "BCC"]:
            if data.get(field):
                recipients.extend([r.strip() for r in str(data[field]).split(',') if r.strip()])

        return cls(
            email_id=data.get("Email_ID", ""),
            subject=data.get("Subject", ""),
            sender=data.get("From", ""),
            recipients=recipients,
            sent_date=sent_date,
            body=data.get("Body", ""),
            is_read=False
        )

    @property
    def sender_initials(self) -> str:
        """Get sender initials for avatar display"""
        if not self.sender:
            return "?"
        parts = [part.strip() for part in self.sender.split() if part.strip()]
        if not parts:
            return "?"
        # Take first letter of first and last name, or just first letter
        if len(parts) >= 2:
            return (parts[0][0] + parts[-1][0]).upper()
        else:
            return parts[0][0].upper()[:2]

    @property
    def avatar_color(self) -> str:
        """Get avatar background color based on sender initials"""
        if not self.sender_initials or self.sender_initials == "?":
            return "#6c757d"  # Gray

        # Optimized: direct character access and pre-computed color palette
        first_char = self.sender_initials[0]
        if 'A' <= first_char <= 'Z':
            index = (ord(first_char) - 65) % len(_AVATAR_COLORS)
            return _AVATAR_COLORS[index]
        elif 'a' <= first_char <= 'z':
            index = (ord(first_char) - 97) % len(_AVATAR_COLORS)
            return _AVATAR_COLORS[index]
        # For any other character (numbers, symbols), use hash for consistency
        index = hash(first_char) % len(_AVATAR_COLORS)
        return _AVATAR_COLORS[index]


@dataclass
class PSTFolder:
    """Simple folder data model for GUI"""
    name: str
    path: str
    email_count: int = 0
    unread_count: int = 0
    favorite: bool = False


class PSTExtractor:
    """Simplified PST extractor for GUI"""

    def __init__(self):
        self.pst_path: Optional[str] = None
        self.is_loaded = False
        self._backend: Optional[PypffBackend] = None

    def load_pst(self, file_path: str) -> bool:
        """Check if PST file can be loaded"""
        if not is_pypff_available():
            raise Exception("PyPFF library not available. Please install libpff-python-ratom.")

        if not os.path.exists(file_path):
            raise Exception(f"PST file not found: {file_path}")

        self.pst_path = file_path
        self.is_loaded = True
        self._backend = PypffBackend()
        self._backend.open(Path(file_path))
        return True

    def get_folders(self) -> List[PSTFolder]:
        """Get list of folders via backend."""
        if not self.is_loaded or not self._backend:
            return []
        folders: List[PSTFolder] = []
        try:
            for info in self._backend.list_folders():
                folders.append(PSTFolder(name=str(info.name), path=str(info.id), email_count=int(info.total_count), unread_count=int(info.unread_count)))
        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return []
        return folders


    def get_total_email_count(self) -> int:
        """Estimate total emails by summing folder counts."""
        if not self.is_loaded or not self._backend:
            return 0
        try:
            return sum(int(f.total_count) for f in self._backend.list_folders())
        except Exception:
            return 0

    def get_emails_from_folder(self, folder: PSTFolder) -> List[EmailMessage]:
        """Get emails from specified folder using backend."""
        if not self._backend:
            return []
        emails: List[EmailMessage] = []
        try:
            for row, handle in self._backend.iter_folder_messages(folder.path, start=0, limit=50):
                email = EmailMessage.from_dict(row)
                email.attachment_count = int(row.get("Attachment_Count", 0) or 0)
                email._handle = handle
                emails.append(email)
        except Exception as e:
            logger.error(f"Error loading folder emails: {e}")
            return []
        return emails


class PSTEmailExtractorGUI:
    """Main GUI application """

    def __init__(self):
        # Initialize CustomTkinter
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # Create main window
        self.root = ctk.CTk()
        self.root.title("PST Email Extractor")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 700)

        # Initialize variables
        self.current_pst_path: Optional[str] = None
        self.extractor: Optional[PSTExtractor] = None
        self.folders: List[PSTFolder] = []
        self.current_folder: Optional[PSTFolder] = None
        self.emails: List[EmailMessage] = []
        self.selected_email: Optional[EmailMessage] = None
        self.attachment_content_options: Optional[AttachmentContentOptions] = None

        # Create UI components
        self._setup_ui()

        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

    def _setup_ui(self):
        """Setup the main UI """
        # Create toolbar
        self._create_toolbar()

        # Create main content area
        self._create_main_content()

        # Create status bar
        self._create_status_bar()

    def _create_toolbar(self):
        """Create the top toolbar with file operations"""
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

        # Settings button - placed before progress widgets for better UX
        self.settings_btn = ctk.CTkButton(
            toolbar_frame, text="Settings",
            command=self._show_settings,
            width=100,
            state="disabled"
        )
        self.settings_btn.grid(row=0, column=3, padx=5, pady=5)

        # Progress bar (always visible)
        self.progress_bar = ctk.CTkProgressBar(toolbar_frame, width=300)
        self.progress_bar.grid(row=0, column=4, padx=5, pady=5)
        self.progress_bar.set(0)  # Start at 0

        # Progress label
        self.progress_label = ctk.CTkLabel(toolbar_frame, text="Ready")
        self.progress_label.grid(row=0, column=5, padx=5, pady=5)

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
        self.progress_bar.grid()
        self.progress_bar.set(0)
        self.progress_label.configure(text="Loading PST file...")

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
            self.root.after(0, lambda: self.progress_label.configure(text="Initializing extractor..."))
            self.root.after(0, lambda: self.progress_bar.set(0.1))

            # Create extractor
            self.extractor = PSTExtractor()

            # Load PST file
            self.root.after(0, lambda: self.progress_label.configure(text="Loading PST structure..."))
            self.root.after(0, lambda: self.progress_bar.set(0.3))

            success = self.extractor.load_pst(file_path)
            if not success:
                raise Exception("Failed to load PST file")

            # Get folders
            self.root.after(0, lambda: self.progress_label.configure(text="Reading folders..."))
            self.root.after(0, lambda: self.progress_bar.set(0.6))

            self.folders = self.extractor.get_folders()

            # Update UI on main thread
            self.root.after(0, self._update_folders_display)

            # Auto-select first folder if available to show emails
            if self.folders:
                self.root.after(0, lambda: self._select_folder(self.folders[0]))

            # Complete
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            self.root.after(0, lambda: self.progress_label.configure(text="Complete"))
            self.root.after(0, lambda: self.status_info.configure(text="Ready"))

            # Hide progress after a moment
            self.root.after(2000, self._hide_progress)

        except Exception as e:
            logger.error(f"Error processing PST file: {e}")
            self.root.after(0, lambda: self._show_error(f"Error loading PST file: {str(e)}"))
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

    def _create_folder_group(self, group_name: str, folders: List[PSTFolder]):
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
            print(f"Loading emails for folder: {folder.name} (path: {folder.path})")  # Debug
            self.emails = self.extractor.get_emails_from_folder(folder)
            print(f"Loaded {len(self.emails)} emails")  # Debug
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
            self.root.after(0, lambda: self._show_error(f"Error loading emails: {str(e)}"))

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
                widget.bind("<Button-1>", lambda e: self._select_email(email_obj))
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
            text=f"📎 {len(attachments)} Attachment{'s' if len(attachments) != 1 else ''}",
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
            text=f"📎 {count} Attachment{'s' if count != 1 else ''}",
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
            'pdf': '📄',
            'doc': '📝', 'docx': '📝',
            'xls': '📊', 'xlsx': '📊',
            'ppt': '📽️', 'pptx': '📽️',
            'txt': '📄',
            'jpg': '🖼️', 'jpeg': '🖼️', 'png': '🖼️', 'gif': '🖼️',
            'zip': '📦', 'rar': '📦',
            'exe': '⚙️',
        }
        return icon_map.get(ext, '📎')

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
            import tempfile, os, subprocess, sys
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

        # When sanitize is checked, apply both sanitization and polishing
        ai_sanitize = self.ai_sanitize_var.get()
        ai_polish = ai_sanitize  # Polish is now tied to sanitize

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
            self.root.after(0, lambda: self.progress_label.configure(text="Exporting..."))
            self.root.after(0, lambda: self.progress_bar.set(0.1))
            self.root.after(0, lambda: self.export_btn.configure(state="disabled"))
            self.root.after(0, lambda: self.open_btn.configure(state="disabled"))

            fmt = export_type.lower()
            formats = [fmt]
            attachments_needed = fmt in {"eml", "mbox"}

            cfg = ExtractionConfig(
                pst_path=Path(self.current_pst_path or ""),
                output_dir=Path(output_dir),
                formats=formats,
                mode="extract",
                deduplicate=False,
                extract_attachments=attachments_needed,
                attachments_dir=None,
                log_file=None,
                html_index=False,
                ai_sanitize=ai_sanitize,
                ai_polish=ai_polish,
                ai_language="en-US",
                ai_neural_model_dir=None,
                extract_attachment_content=self.attachment_content_options is not None,
                attachment_content_options=self.attachment_content_options,
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
                        self.progress_bar.set(max(0.0, min(1.0, current / total_v)))
                    self.progress_label.configure(text=msg)
                self.root.after(0, _ui)

            result = run_extraction(cfg, progress_callback=_progress_cb)

            # Complete
            self.root.after(0, lambda: self.progress_bar.set(1.0))
            self.root.after(0, lambda: self.progress_label.configure(text="Export complete"))
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
            self.root.after(0, lambda: self._show_error(f"Export failed: {str(e)}"))
            self.root.after(0, self._hide_progress)

    def _show_settings(self):
        """Show settings dialog (not implemented)"""
        messagebox.showinfo("Settings", "Settings dialog is not implemented yet.\n\nUse the command line interface for advanced configuration options.")

    def _show_error(self, message: str):
        """Show error message"""
        messagebox.showerror("Error", message)

    def _hide_progress(self):
        """Hide or reset progress indicators"""
        try:
            if hasattr(self, 'progress_bar') and self.progress_bar:
                self.progress_bar.set(0)
            if hasattr(self, 'progress_label') and self.progress_label:
                self.progress_label.configure(text="Ready")
        except (RuntimeError, AttributeError) as exc:
            # Widget may be destroyed due to shutdown/teardown; ignore but trace
            logger.debug(f"Progress widgets unavailable during hide: {exc}")

    def _reset_progress(self):
        """Reset progress indicators to ready state"""
        self.progress_bar.set(0)
        self.progress_label.configure(text="Ready")

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
        try:
            widget.configure(state=state)
        except:
            pass
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