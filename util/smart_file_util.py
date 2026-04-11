"""
smart_file_util.py - Canonical path resolution and file I/O for stateful outputs.

All paths for named-convention files (activations, SAE weights, MLP weights, etc.)
are constructed here so the naming format is defined in exactly one place.
"""

import csv
import json
from pathlib import Path
from typing import Optional

# ── Base directories ──────────────────────────────────────────────────────────

ACTIVATIONS_DIR = Path("activations")
SAE_OUTPUT_DIR  = Path("sae_output")
MLP_OUTPUT_DIR  = Path("mlp_output")

FLUSH_EVERY = 10


# ── Model slug ────────────────────────────────────────────────────────────────

def model_slug(model_name: str) -> str:
    """Filesystem-safe representation of a model name (replaces '/' with '_')."""
    return model_name.replace("/", "_")


# ── Stateful file path helpers ────────────────────────────────────────────────

def activations_path(split_id: int, model_name: str) -> Path:
    """Dense residual-stream activations for a split/model pair."""
    return ACTIVATIONS_DIR / f"activations_{split_id}_{model_slug(model_name)}.pt"


def sparse_features_path(split_id: int, model_name: str) -> Path:
    """SAE-encoded sparse feature vectors for a split/model pair."""
    return ACTIVATIONS_DIR / f"activations_{split_id}_{model_slug(model_name)}_sparse.pt"


def sae_weights_path(split_id: int, model_name: str) -> Path:
    """Trained SAE weights checkpoint directory (loaded via SAE.load_from_disk)."""
    return SAE_OUTPUT_DIR / f"sae_{split_id}_{model_slug(model_name)}_weights"


def sae_cfg_path(split_id: int, model_name: str) -> Path:
    """SAE architecture and hyperparameter config."""
    return SAE_OUTPUT_DIR / f"cfg_{split_id}_{model_slug(model_name)}.json"


def mlp_path(split_id: int, model_name: str) -> Path:
    """Trained MLP router weights."""
    return MLP_OUTPUT_DIR / f"mlp_{split_id}_{model_slug(model_name)}.pt"


def sae_checkpoint_path(model_key: str) -> Path:
    """SAELens live training checkpoint directory (intermediate, not final weights)."""
    return SAE_OUTPUT_DIR / model_key


# ── Generic JSONL / CSV I/O ───────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, record in enumerate(records):
            f.write(json.dumps(record) + "\n")
            if i % FLUSH_EVERY == 0:
                f.flush()


def export_csv(
    path: Path,
    records: list[dict],
    columns: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Write records to CSV. Returns the output path.

    If output_path is None, replaces the source extension with .csv.
    If columns is None, columns are inferred from key order of first appearance.
    """
    if output_path is None:
        output_path = Path(path).with_suffix(".csv")

    if not records:
        raise ValueError(f"No records to export from {path}")

    if columns:
        fieldnames = columns
    else:
        fieldnames = []
        seen: set = set()
        for row in records:
            for key in row:
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    return output_path


def filter_by_task_ids(records: list[dict], task_ids: set[str]) -> list[dict]:
    """Return only records whose task_id is in the provided set.

    Callers should get task_ids from task_split_dao.get_task_ids_for_split().
    """
    return [r for r in records if r.get("task_id") in task_ids]
