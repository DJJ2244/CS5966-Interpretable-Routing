"""
toughness.py - Score dataset problems with the RouteLLM BERT router.
No model inference is performed — only difficulty scoring.
"""

import json
from pathlib import Path


ROUTER    = "bert"

def get_router_client(weak_model: str = "", strong_model: str = ""):
    """Return a RouteLLM Controller for the given models."""
    from routellm.controller import Controller
    return Controller(
        routers=[ROUTER],
        strong_model=f"openai/{strong_model}",
        weak_model=f"openai/{weak_model}",
    )


def record_toughness(
    split_id: int,
    is_test: bool = False,
    output_dir: str = "route_llm_results",
) -> None:
    """Run the router over the dataset and save per-problem difficulty scores.

    Args:
        split_id:   DB split id to score.
        is_test:    Whether to score the test partition (default: train).
        output_dir: Directory for the output toughness.jsonl file.
    """
    from daos import tasks_dao

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "toughness.jsonl"  # TODO: Move to DB store

    router_client = get_router_client()
    router = router_client.routers[ROUTER]

    tasks = tasks_dao.get_all_for_split(split_id, is_test=is_test)

    with open(output_path, "w") as out:
        for i, task in enumerate(tasks):
            score = router.calculate_strong_win_rate(task.prompt)
            record = {"task_id": task.id, "score": float(score)}
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i + 1}] {task.id} -> {score:.4f}")

    print(f"\nDone. Results saved to {output_path}")
