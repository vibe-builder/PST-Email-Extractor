import csv
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

output_dir = Path('output')
if len(sys.argv) > 1:
    target_path = Path(sys.argv[1])
else:
    candidates = sorted(output_dir.glob('pst_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    target_path = candidates[0] if candidates else None

if not target_path or not target_path.exists():
    raise SystemExit("CSV file not found. Pass a path or run the extractor first.")

rows = []
with target_path.open('r', encoding='utf-8', newline='') as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        rows.append(row)
        if len(rows) == 25:
            break

for idx, row in enumerate(rows, start=1):
    print(f"Email {idx}:")
    print(f"  Subject: {row['Subject']}")
    print(f"  From: {row['From']}")
    print(f"  To: {row['To']}")
    print(f"  Date Sent: {row['Date_Sent']}")
    snippet = row['Body'][:200].replace('\r', ' ').replace('\n', ' ')
    print(f"  Body (first 200 chars): {snippet}")
    print()

print(f"Total sampled: {len(rows)} (from {target_path.name})")
