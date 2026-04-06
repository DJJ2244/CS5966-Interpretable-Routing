"""
dataset.py - Loads HumanEval-XL and prepares prompts for RouteLLM inference.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

PATHS = {
    "test":  Path("data/humaneval_xl_english_test.jsonl"),
    "train": Path("data/humaneval_xl_english_train.jsonl"),
}


@dataclass
class Problem:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test: str
    description: str
    programming_language: str = "python"


def count(split: str = "test") -> int:
    """Return the total number of problems in the dataset."""
    with open(PATHS[split]) as f:
        return sum(1 for _ in f)


def load(split: str = "test", limit: Optional[int] = None) -> Generator[Problem, None, None]:
    """Yield problems from the dataset, optionally capped at `limit`."""
    with open(PATHS[split]) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            yield Problem(
                task_id=rec["task_id"],
                prompt=rec["prompt"],
                entry_point=rec["entry_point"],
                canonical_solution=rec["canonical_solution"],
                test=rec["test"],
                description=rec.get("description", ""),
                programming_language=rec.get("programming_language", "python"),
            )


def as_message(problem: Problem) -> list[dict]:
    """Format a problem as a messages list ready to pass to the Controller."""
    return [{"role": "user", "content": problem.prompt}]
