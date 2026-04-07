"""
toughness.py - Score every dataset problem with the BERT router.
No model inference is performed.
"""

import json
from pathlib import Path

from route_llm_inference.router_client import client, ROUTER
from util.dataset import load


def record_toughness(split: str = "train", output_dir: str = "route_llm_results") -> None:
    """
    Run the router over the dataset and save per-problem difficulty scores.

    Args:
        split:      "train" or "test"
        output_dir: directory for output .jsonl file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "toughness.jsonl"

    router = client.routers[ROUTER]

    with open(output_path, "w") as out:
        for i, problem in enumerate(load(split=split)):
            score = router.calculate_strong_win_rate(problem.prompt)
            record = {"task_id": problem.task_id, "score": float(score)}
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i + 1}] {problem.task_id} -> {score:.4f}")

    print(f"\nDone. Results saved to {output_path}")
