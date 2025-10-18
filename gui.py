#!/usr/bin/env python3
"""
Modern PyQt6 GUI for PST Email Extractor.
Delivers a 2025-ready dark theme with polished visuals and responsive feedback.
"""

import sys
import os
from contextlib import ExitStack
from pathlib import Path
from typing import List
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QCheckBox, QProgressBar,
    QFileDialog, QMessageBox, QFrame, QStyleFactory
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QCoreApplication, QUrl
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon, QDesktopServices

# import core modules
try:
    from pst_tools import pst_parser, id_generator, exporters
except ImportError as e:
    print(f"Missing dependencies: {e}")
    sys.exit(1)


def resource_path(relative: str) -> Path:
    """
    Resolve resource paths for both development and bundled environments.
    """
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return (base_path / relative).resolve()


class ExtractionThread(QThread):
    """Background thread for PST extraction."""
    progress_update = pyqtSignal(str)
    progress_changed = pyqtSignal(int, int, str)
    error_occurred = pyqtSignal(str)
    extraction_complete = pyqtSignal(int, list)

    def __init__(self, pst_path, output_path, export_json, export_csv):
        super().__init__()
        self.pst_path = pst_path
        self.output_path = output_path
        self.export_json = export_json
        self.export_csv = export_csv

    def run(self):
        try:
            unique_id = id_generator.generate()
            self.progress_update.emit("Parsing PST file...")

            def progress_callback(current, total, status):
                self.progress_changed.emit(current, total, status)

            output_dir = Path(self.output_path).resolve()
            exported_paths: List[str] = []

            with ExitStack() as stack:
                csv_writer = stack.enter_context(
                    exporters.CSVStreamWriter(output_dir / f"pst_{unique_id}.csv")
                ) if self.export_csv else None
                json_writer = stack.enter_context(
                    exporters.JSONStreamWriter(output_dir / f"pst_{unique_id}.json")
                ) if self.export_json else None

                if csv_writer:
                    exported_paths.append(str(csv_writer.path))
                if json_writer:
                    exported_paths.append(str(json_writer.path))

                self.progress_update.emit("Streaming exports to disk...")

                email_count = 0
                for email in pst_parser.iter_emails(
                    self.pst_path,
                    progress_callback=progress_callback
                ):
                    email_count += 1
                    if csv_writer:
                        csv_writer.write(email)
                    if json_writer:
                        json_writer.write(email)

            if email_count == 0:
                self.error_occurred.emit("No emails found in PST file")
                return

            self.extraction_complete.emit(email_count, exported_paths)

        except Exception as e:
            self.error_occurred.emit(str(e))


class PSTExtractorGUI(QMainWindow):
    """Main GUI window for PST Extractor."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PST Email Extractor")
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(800, 600)
        icon_path = resource_path("logo.ico")
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            QApplication.instance().setWindowIcon(icon)
            self.setWindowIcon(icon)
        logs_dir = Path.home() / "PST Email Extractor" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = logs_dir / "pst_parser.log"
        pst_parser.configure_logging(self.log_path)
        self.setup_ui()
        self.extraction_thread = None
        self.is_processing = False

    def setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)

        header = QLabel("PST Email Extractor")
        header_font = QFont("Segoe UI", 26, QFont.Weight.Bold)
        header.setFont(header_font)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        subtitle = QLabel("Convert Outlook PST mailboxes into clean JSON or CSV in minutes.")
        subtitle.setObjectName("Subtitle")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)
        card_layout.setSpacing(18)
        layout.addWidget(card, 1)

        pst_layout = QHBoxLayout()
        pst_label = QLabel("Select PST File")
        pst_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        self.pst_entry = QLineEdit()
        self.pst_entry.setPlaceholderText("Choose a .pst file to extract")
        self.pst_entry.setReadOnly(True)
        self.pst_browse = QPushButton("Browse")
        self.pst_browse.setObjectName("SecondaryButton")
        self.pst_browse.clicked.connect(self.browse_pst)
        pst_layout.addWidget(pst_label)
        pst_layout.addWidget(self.pst_entry, 1)
        pst_layout.addWidget(self.pst_browse)
        card_layout.addLayout(pst_layout)

        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory")
        output_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        default_output = Path.home() / "Desktop"
        self.output_entry = QLineEdit(str(default_output if default_output.exists() else Path.home()))
        self.output_browse = QPushButton("Browse")
        self.output_browse.setObjectName("SecondaryButton")
        self.output_browse.clicked.connect(self.browse_output)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_entry, 1)
        output_layout.addWidget(self.output_browse)
        card_layout.addLayout(output_layout)

        format_layout = QHBoxLayout()
        format_label = QLabel("Export Formats")
        format_label.setFont(QFont("Segoe UI", 12, QFont.Weight.Medium))
        self.json_check = QCheckBox("JSON")
        self.json_check.setChecked(True)
        self.csv_check = QCheckBox("CSV")
        self.csv_check.setChecked(True)
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.json_check)
        format_layout.addWidget(self.csv_check)
        format_layout.addStretch()
        card_layout.addLayout(format_layout)

        self.progress_label = QLabel("Ready to extract emails")
        self.progress_label.setFont(QFont("Segoe UI", 11))
        card_layout.addWidget(self.progress_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        card_layout.addWidget(self.progress_bar)

        self.extract_button = QPushButton("Extract Emails")
        self.extract_button.setObjectName("PrimaryButton")
        self.extract_button.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        self.extract_button.setFixedHeight(52)
        self.extract_button.clicked.connect(self.start_extraction)
        card_layout.addWidget(self.extract_button)

        log_layout = QHBoxLayout()
        log_label = QLabel("Log file")
        log_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Medium))
        self.log_path_display = QLabel(str(self.log_path))
        self.log_path_display.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        open_logs_button = QPushButton("Open Log")
        open_logs_button.setObjectName("SecondaryButton")
        open_logs_button.clicked.connect(self.open_logs)
        log_layout.addWidget(log_label)
        log_layout.addWidget(self.log_path_display, 1)
        log_layout.addWidget(open_logs_button)
        card_layout.addLayout(log_layout)

        layout.addStretch(1)

        self.statusBar().showMessage(f"Ready — logs: {self.log_path}")
        status_link = QLabel(f"<a href=\"{self.log_path.as_uri()}\">Open log file</a>")
        status_link.setTextFormat(Qt.TextFormat.RichText)
        status_link.setOpenExternalLinks(True)
        status_link.setStyleSheet("color: #93C5FD;")
        self.statusBar().addPermanentWidget(status_link)

        self.apply_modern_style()

    def apply_modern_style(self):
        """Apply a polished dark theme aesthetic."""
        app = QApplication.instance()
        app.setStyle(QStyleFactory.create("Fusion"))

        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#0B1120"))
        palette.setColor(QPalette.ColorRole.WindowText, QColor("#E2E8F0"))
        palette.setColor(QPalette.ColorRole.Base, QColor("#0F172A"))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#111C2D"))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#1E293B"))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#F8FAFC"))
        palette.setColor(QPalette.ColorRole.Text, QColor("#E2E8F0"))
        palette.setColor(QPalette.ColorRole.Button, QColor("#1E293B"))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor("#F8FAFC"))
        palette.setColor(QPalette.ColorRole.Link, QColor("#60A5FA"))
        palette.setColor(QPalette.ColorRole.Highlight, QColor("#3B82F6"))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0B1120"))
        palette.setColor(QPalette.ColorRole.BrightText, QColor("#F43F5E"))
        app.setPalette(palette)

        app.setStyleSheet("""
            QWidget {
                background-color: #0B1120;
                color: #E2E8F0;
                font-family: 'Segoe UI';
            }
            QFrame#Card {
                background-color: #111B2E;
                border-radius: 22px;
                border: 1px solid #1F2A40;
            }
            QLabel#Subtitle {
                color: #94A3B8;
            }
            QLineEdit {
                background-color: #0F172A;
                border: 1px solid #1F2A3E;
                border-radius: 12px;
                padding: 10px 14px;
                color: #E2E8F0;
                selection-background-color: #3B82F6;
                selection-color: #F8FAFC;
            }
            QLineEdit:focus {
                border-color: #3B82F6;
                background-color: #15213B;
            }
            QCheckBox {
                font-size: 13px;
                color: #E2E8F0;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 1px solid #334155;
                background: #0F172A;
            }
            QCheckBox::indicator:checked {
                background-color: #3B82F6;
                border: 1px solid #60A5FA;
            }
            QPushButton {
                border-radius: 14px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 500;
            }
            QPushButton#PrimaryButton {
                background-color: #3B82F6;
                color: #F8FAFC;
                font-weight: 600;
                border: 1px solid #3B82F6;
            }
            QPushButton#PrimaryButton:hover {
                background-color: #2563EB;
            }
            QPushButton#PrimaryButton:disabled {
                background-color: #334155;
                color: #94A3B8;
            }
            QPushButton#SecondaryButton {
                background-color: #1E293B;
                color: #93C5FD;
                border: 1px solid #1F2A40;
            }
            QPushButton#SecondaryButton:hover {
                background-color: #233046;
            }
            QProgressBar {
                background-color: #111C2D;
                border: 1px solid #1F2A40;
                border-radius: 12px;
                text-align: center;
                height: 28px;
                color: #E2E8F0;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                            stop:0 #3B82F6, stop:1 #60A5FA);
                border-radius: 12px;
            }
            QStatusBar {
                background: #0F172A;
                border-top: 1px solid #1F2A40;
                color: #94A3B8;
                padding: 4px 12px;
            }
        """)

    def browse_pst(self):
        """Browse for PST file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Select PST File",
            "",
            "PST Files (*.pst);;All Files (*.*)"
        )
        if filename:
            self.pst_entry.setText(filename)
            self.statusBar().showMessage(f"PST file selected: {Path(filename).name}")

    def browse_output(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_entry.text()
        )
        if directory:
            self.output_entry.setText(directory)
            self.statusBar().showMessage(f"Output directory: {directory}")

    def open_logs(self):
        """Open the log directory in the system file explorer."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            try:
                self.log_path.touch()
            except Exception:
                pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_path.parent)))

    def start_extraction(self):
        """Start the extraction process."""
        if self.is_processing:
            QMessageBox.warning(self, "Warning", "Extraction already in progress")
            return

        pst_path = self.pst_entry.text()
        output_path = self.output_entry.text()

        if not pst_path or not os.path.isfile(pst_path):
            QMessageBox.critical(self, "Error", "Please select a valid PST file")
            return

        if not output_path or not os.path.isdir(output_path):
            QMessageBox.critical(self, "Error", "Please select a valid output directory")
            return

        if not self.json_check.isChecked() and not self.csv_check.isChecked():
            QMessageBox.critical(self, "Error", "Please select at least one export format")
            return

        self.is_processing = True
        self.extract_button.setEnabled(False)
        self.extract_button.setText("Processing...")
        self.progress_label.setText("Processing PST file...")
        self.progress_bar.setRange(0, 0)  # indeterminate mode
        self.statusBar().showMessage("Processing PST file...")

        self.extraction_thread = ExtractionThread(
            pst_path, output_path,
            self.json_check.isChecked(),
            self.csv_check.isChecked()
        )
        self.extraction_thread.progress_update.connect(self.update_progress)
        self.extraction_thread.progress_changed.connect(self.update_progress_metrics)
        self.extraction_thread.error_occurred.connect(self.show_error)
        self.extraction_thread.extraction_complete.connect(self.extraction_done)
        self.extraction_thread.start()

    def update_progress(self, message):
        """Update progress label."""
        self.progress_label.setText(message)
        self.statusBar().showMessage(message)

    def update_progress_metrics(self, current, total, status):
        """Update determinate progress bar and label."""
        if total:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(current)
            percentage = (current / total) * 100 if total else 0
            self.progress_bar.setFormat(f"{percentage:0.1f}%")
        else:
            self.progress_bar.setRange(0, 0)
        self.progress_label.setText(status)
        self.statusBar().showMessage(status)

    def show_error(self, message):
        """Show error message."""
        QMessageBox.critical(self, "Error", message)
        self.reset_ui()

    def extraction_done(self, email_count: int, exported_paths: List[str]):
        """Handle extraction completion."""
        output_directory = Path(exported_paths[0]).parent if exported_paths else Path(self.output_entry.text())
        files_list = "\n".join(exported_paths)
        success_msg = (
            f"Successfully extracted {email_count} emails!\n\n"
            f"Saved files:\n{files_list or 'No outputs generated'}"
        )
        QMessageBox.information(self, "Success", success_msg)
        self.statusBar().showMessage(f"Completed — exported {email_count} emails")

        # open output folder
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(output_directory)))
        except Exception:
            pass

        self.reset_ui()

    def reset_ui(self):
        """Reset UI after extraction."""
        self.is_processing = False
        self.extract_button.setEnabled(True)
        self.extract_button.setText("Extract Emails")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_label.setText("Ready to extract emails")
        self.statusBar().showMessage(f"Ready — logs: {self.log_path}")
        self.extraction_thread = None

    def closeEvent(self, event):
        """Handle window close."""
        if self.is_processing:
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Extraction is in progress. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = PSTExtractorGUI()
    window.show()
    sys.exit(app.exec())

