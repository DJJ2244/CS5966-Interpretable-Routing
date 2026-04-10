"""
calculate_threshold.py - Compute the RouteLLM routing threshold from toughness scores.

The threshold is the score below which problems are routed to the weak model.
Given a target fraction of calls to send to the strong model, we find the score
at the corresponding percentile of the toughness distribution.
"""

import json
from pathlib import Path
from typing import Optional


def calculate_threshold(
    toughness_path: Path,
    target_strong_rate: float = 0.5,
) -> float:
    """Compute the routing threshold from a toughness.jsonl file.

    Args:
        toughness_path:    Path to toughness.jsonl (records with task_id + score).
        target_strong_rate: Fraction of problems to route to the strong model (0–1).
                            Default 0.5 sends the hardest 50% to strong.

    Returns:
        Threshold float. Problems with score >= threshold are routed to strong.
    """
    scores = _load_scores(toughness_path)
    if not scores:
        raise ValueError(f"No scores found in {toughness_path}")

    scores_sorted = sorted(scores)
    cutoff_idx = int(len(scores_sorted) * (1.0 - target_strong_rate))
    cutoff_idx = max(0, min(cutoff_idx, len(scores_sorted) - 1))
    threshold = scores_sorted[cutoff_idx]

    strong_count = sum(1 for s in scores if s >= threshold)
    print(
        f"Threshold: {threshold:.5f}  "
        f"({strong_count}/{len(scores)} = {strong_count/len(scores):.1%} routed to strong)"
    )
    return threshold


def _load_scores(toughness_path: Path) -> list[float]:
    scores = []
    with open(toughness_path) as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(float(json.loads(line)["score"]))
    return scores


def save_threshold(path: Path, threshold: float) -> None:
    """Persist a computed threshold to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"route_llm_threshold": threshold}, f)
    print(f"Threshold saved to {path}")


def load_threshold(path: Path) -> Optional[float]:
    """Load a previously saved threshold, or None if the file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p) as f:
        return float(json.load(f)["route_llm_threshold"])
