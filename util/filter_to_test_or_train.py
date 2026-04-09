"""
Filter a JSONL file to only the records whose task_id belongs to the
train or test split.
"""

import json
from pathlib import Path

from util.dataset import PATHS


def _load_split_ids(split: str) -> set:
    ids = set()
    for path in ([PATHS["train"], PATHS["test"]] if split == "all" else [PATHS[split]]):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(json.loads(line)["task_id"])
    return ids


def filter_to_split(
    source_path: Path,
    split: str,
    output_dir: Path,
) -> Path:
    """Filter source_path to records matching split (train|test), write to output_dir."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")

    split_ids = _load_split_ids(split)

    stem = source_path.stem
    output_path = output_dir / f"{stem}_{split}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    with open(source_path) as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("task_id") in split_ids:
                f_out.write(json.dumps(rec) + "\n")
                kept += 1

    return output_path
