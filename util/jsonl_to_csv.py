"""Convert a JSONL file to CSV."""

import csv
import json
from pathlib import Path
from typing import List, Optional


def jsonl_to_csv(
    input_path: Path,
    output_path: Optional[Path] = None,
    columns: Optional[List[str]] = None,
) -> Path:
    if output_path is None:
        output_path = input_path.with_suffix(".csv")

    rows = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No records found in {input_path}")

    if columns:
        fieldnames = columns
    else:
        # Collect all keys in order of first appearance
        fieldnames = []
        seen: set = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    return output_path
