"""
HTML index generator for PST email data.

This module creates searchable HTML files that provide a web-based interface
for browsing extracted email data. The generated HTML includes client-side
search functionality and supports viewing attachments.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("pst_email_extractor.exporters.html")


def generate_html_index(
    records: Sequence[Mapping[str, Any]] | None = None,
    *,
    output_path: Path,
    attachments_dir: Path | None = None,
    email_records: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    """
    Generate a searchable HTML index of extracted emails.

    Creates a single HTML file with a searchable and filterable table of email
    records. Includes client-side JavaScript for instant filtering and proper
    handling of attachments and email previews.

    Args:
        records / email_records: Sequence of email record dictionaries
        output_path: Path where the HTML file will be written
        attachments_dir: Optional directory containing email attachments

    Returns:
        Path to the generated HTML file
    """
    logger.info("Generating HTML index to %s", output_path)

    data = records if records is not None else email_records
    if data is None:
        raise ValueError("No records provided for HTML index generation.")

    # Normalize key casing to support both canonical (TitleCase) and lowercase keys
    normalised: list[dict[str, Any]] = []
    for item in data:
        # Copy and add TitleCase aliases when lowercase provided
        d = dict(item)
        # Define mappings expected by the frontend
        mappings = {
            "subject": "Subject",
            "from": "From",
            "sender": "Sender_Email",
            "to": "To",
            "cc": "CC",
            "bcc": "BCC",
            "date": "Date_Received",
            "attachments": "Attachment_Paths",
            "id": "Email_ID",
            "preview": "Body",
        }
        for low, proper in mappings.items():
            if low in d and proper not in d:
                d[proper] = d[low]
        normalised.append(d)

    # Convert records to JSON for embedding in HTML
    records_json = json.dumps(normalised, ensure_ascii=False, default=str, indent=2)

    # Determine base URL for attachments
    attachments_base_url = ""
    if attachments_dir:
        try:
            attachments_base_url = attachments_dir.as_uri()
        except ValueError:
            # Fallback for relative paths
            attachments_base_url = f"./{attachments_dir.name}/"

    # Generate HTML with embedded CSS and JavaScript
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Email Extraction Index</title>
    <style>
        :root {{
            --bg-primary: #0d1628;
            --bg-secondary: #0b1322;
            --bg-tertiary: #111d34;
            --text-primary: #e2e8f0;
            --text-secondary: #64748b;
            --accent: #38bdf8;
            --border: #1f2a40;
            --success: #10b981;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px;
        }}

        h1 {{
            margin-bottom: 24px;
            color: var(--text-primary);
            font-size: 2em;
            font-weight: 600;
        }}

        .search-container {{
            margin-bottom: 24px;
        }}

        .search-input {{
            width: 100%;
            max-width: 600px;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background-color: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 16px;
            transition: border-color 0.2s ease;
        }}

        .search-input:focus {{
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.1);
        }}

        .stats {{
            margin-bottom: 16px;
            color: var(--text-secondary);
            font-size: 14px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            background-color: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}

        th, td {{
            text-align: left;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            vertical-align: top;
        }}

        th {{
            background-color: var(--bg-tertiary);
            font-weight: 600;
            color: var(--text-primary);
            position: sticky;
            top: 0;
            z-index: 10;
        }}

        tr {{
            transition: background-color 0.2s ease;
        }}

        tr:hover {{
            background-color: var(--bg-tertiary);
            cursor: pointer;
        }}

        .email-subject {{
            font-weight: 600;
            color: var(--accent);
            word-break: break-word;
        }}

        .email-meta {{
            font-size: 0.875em;
            color: var(--text-secondary);
            margin-top: 4px;
        }}

        .email-attachments {{
            font-size: 0.8em;
            color: var(--success);
            margin-top: 4px;
        }}

        .attachment-link {{
            color: var(--accent);
            text-decoration: none;
            margin-right: 8px;
            padding: 2px 6px;
            border-radius: 4px;
            background-color: rgba(56, 189, 248, 0.1);
            transition: background-color 0.2s ease;
        }}

        .attachment-link:hover {{
            background-color: rgba(56, 189, 248, 0.2);
            text-decoration: underline;
        }}

        .email-preview {{
            background-color: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 16px;
            margin-top: 8px;
            white-space: pre-wrap;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
            font-size: 14px;
            line-height: 1.5;
            display: none;
            max-height: 400px;
            overflow-y: auto;
        }}

        .no-results {{
            text-align: center;
            padding: 48px;
            color: var(--text-secondary);
            font-size: 18px;
        }}

        .loading {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid var(--border);
            border-radius: 50%;
            border-top-color: var(--accent);
            animation: spin 1s ease-in-out infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 16px;
            }}

            th, td {{
                padding: 8px 12px;
            }}

            .email-meta {{
                font-size: 0.8em;
            }}
        }}
    </style>
</head>
<body>
    <h1>Extracted Emails Index</h1>
    <div class="search-container">
        <input type="text" id="search-input" class="search-input" placeholder="Search by subject, sender, recipient, or date...">
    </div>
    <div id="stats" class="stats">Loading...</div>
    <table id="email-table">
        <thead>
            <tr>
                <th>Subject</th>
                <th>From</th>
                <th>To</th>
                <th>Date</th>
                <th>Attachments</th>
            </tr>
        </thead>
        <tbody id="email-list">
            <tr>
                <td colspan="5" style="text-align: center; padding: 48px;">
                    <div class="loading"></div>
                    Loading emails...
                </td>
            </tr>
        </tbody>
    </table>

    <script>
        const emails = {records_json};
        const emailList = document.getElementById('email-list');
        const searchInput = document.getElementById('search-input');
        const statsDiv = document.getElementById('stats');
        const attachmentsBaseUrl = "{attachments_base_url}";

        function formatDate(dateString) {{
            if (!dateString) return 'N/A';
            try {{
                const date = new Date(dateString);
                return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
            }} catch {{
                return dateString;
            }}
        }}

        function renderEmails(filterText = '') {{
            emailList.innerHTML = '';
            const filteredEmails = emails.filter(email => {{
                const searchString = `${{email.Subject || ''}} ${{email.From || ''}} ${{email.Sender_Email || ''}} ${{email.To || ''}} ${{email.Date_Received || ''}}`.toLowerCase();
                return searchString.includes(filterText.toLowerCase());
            }});

            if (filteredEmails.length === 0) {{
                emailList.innerHTML = '<tr><td colspan="5" class="no-results">No emails found matching your search.</td></tr>';
                return;
            }}

            filteredEmails.forEach((email, index) => {{
                const row = document.createElement('tr');
                row.dataset.index = index;

                const subject = email.Subject || '(No Subject)';
                const sender = email.From || email.Sender_Email || 'N/A';
                const recipients = [email.To, email.CC, email.BCC].filter(Boolean).join(', ') || 'N/A';
                const date = formatDate(email.Date_Received || email.Date_Sent);
                const hasAttachments = email.Attachment_Count > 0;
                const attachmentCount = email.Attachment_Count || 0;

                row.innerHTML = `
                    <td class="email-subject">${{subject}}</td>
                    <td>${{sender}}</td>
                    <td>${{recipients}}</td>
                    <td>${{date}}</td>
                    <td>
                        ${{hasAttachments ? `<span class="email-attachments">${{attachmentCount}} attachment${{attachmentCount !== 1 ? 's' : ''}}</span>` : ''}}
                    </td>
                `;

                // Add click handler for expanding preview
                row.addEventListener('click', () => {{
                    const existingPreview = row.nextElementSibling;
                    if (existingPreview && existingPreview.classList.contains('email-preview-row')) {{
                        existingPreview.remove();
                        return;
                    }}

                    // Remove any existing previews
                    document.querySelectorAll('.email-preview-row').forEach(preview => preview.remove());

                    const previewRow = document.createElement('tr');
                    previewRow.className = 'email-preview-row';
                    previewRow.innerHTML = `
                        <td colspan="5">
                            <div class="email-preview">
                                <strong>Body:</strong><br>
                                ${{email.Body || 'No body content available.'}}
                                ${{hasAttachments ? `<br><br><strong>Attachments:</strong><br>${{email.Attachment_Paths ? email.Attachment_Paths.map(path => `<a href="${{attachmentsBaseUrl}}${{path}}" class="attachment-link" target="_blank">${{path.split('/').pop()}}</a>`).join(', ') : 'None'}}` : ''}}
                            </div>
                        </td>
                    `;
                    row.after(previewRow);
                }});

                emailList.appendChild(row);
            }});

            // Update stats
            statsDiv.textContent = `Showing ${{filteredEmails.length}} of ${{emails.length}} emails`;
        }}

        // Initial render
        renderEmails();

        // Search functionality
        let searchTimeout;
        searchInput.addEventListener('input', (event) => {{
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {{
                renderEmails(event.target.value);
            }}, 300); // Debounce search
        }});

        // Clear search on Escape
        searchInput.addEventListener('keydown', (event) => {{
            if (event.key === 'Escape') {{
                searchInput.value = '';
                renderEmails();
            }}
        }});
    </script>
</body>
</html>"""

    # Ensure output directory exists and write file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("HTML index generated successfully: %s", output_path)
    return output_path
