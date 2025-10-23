from __future__ import annotations

import mailbox
from pathlib import Path

import pytest

from pst_email_extractor import exporters


@pytest.fixture()
def sample_email(tmp_path: Path) -> tuple[dict, Path]:
    attachments_root = tmp_path / "attachments"
    attachments_root.mkdir()
    attachment = attachments_root / "hello.txt"
    attachment.write_text("hello world", encoding="utf-8")
    record = {
        "Email_ID": "abc123",
        "Subject": "Test Email",
        "From": "Sender <sender@example.com>",
        "Sender_Email": "sender@example.com",
        "To": "recipient@example.com",
        "CC": "",
        "BCC": "",
        "Reply_To": "",
        "Date_Received": "2025-01-01 10:00:00",
        "Date_Sent": "2025-01-01 09:59:00",
        "Body": "Hello body",
        "Attachment_Count": 1,
        "Attachment_Paths": [attachment.name],
        "Message_ID": "<message-id>",
        "Content_Type": "text/plain",
        "Email_Client": "TestClient",
        "Return_Path": "",
        "Client_Info": "",
        "Thread_ID": "thread-1",
        "Parent_ID": "",
        "References": [],
        "Received_Hops": ["mail.example.com"],
        "Full_Headers": "From: sender@example.com\nTo: recipient@example.com",
    }
    return record, attachments_root


def test_json_stream_writer(tmp_path: Path) -> None:
    data = {
        "1": {"Email_ID": "1", "Subject": "One"},
        "2": {"Email_ID": "2", "Subject": "Two"},
    }
    output = tmp_path / "out.json"
    exporters.export_to_json(output, data)
    content = output.read_text(encoding="utf-8")
    assert '"1"' in content and '"2"' in content


def test_csv_stream_writer(tmp_path: Path) -> None:
    data = {
        "1": {"Email_ID": "1", "Subject": "One"},
        "2": {"Email_ID": "2", "Subject": "Two"},
    }
    output = tmp_path / "out.csv"
    exporters.export_to_csv(output, data, fields=["Email_ID", "Subject"])
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "Email_ID,Subject"
    assert "1,One" in lines[1]


def test_eml_writer(sample_email: tuple[dict, Path], tmp_path: Path) -> None:
    record, attachments_root = sample_email
    with exporters.EMLWriter(tmp_path / "eml", attachments_root) as writer:
        path = writer.write(record)
    assert path.exists()
    content = path.read_text(encoding="utf-8", errors="ignore")
    assert "Subject: Test Email" in content
    assert "filename=\"hello.txt\"" in content


def test_mbox_writer(sample_email: tuple[dict, Path], tmp_path: Path) -> None:
    record, attachments_root = sample_email
    mbox_path = tmp_path / "messages.mbox"
    with exporters.MBOXWriter(mbox_path, attachments_root) as writer:
        writer.write(record)
    box = mailbox.mbox(mbox_path)
    try:
        assert len(box) == 1
        msg = box[0]
        assert "Test Email" in msg["Subject"]
    finally:
        box.close()


def test_generate_html_index(tmp_path: Path) -> None:
    html_path = exporters.generate_html_index(
        email_records=[
            {
                "subject": "Hello",
                "sender": "sender@example.com",
                "recipients": "recipient@example.com",
                "date": "2025-01-01",
                "attachments": [],
                "id": "1",
                "preview": "Body text",
            }
        ],
        output_path=tmp_path / "index.html",
    )
    assert html_path.exists()
    content = html_path.read_text(encoding="utf-8")
    assert "Email Extraction Index" in content
    assert "Hello" in content
