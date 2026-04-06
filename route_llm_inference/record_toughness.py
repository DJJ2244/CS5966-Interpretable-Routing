"""
record_toughness.py - Score every dataset problem with the router and save results
to toughness.jsonl. No model inference is performed.

Each line of toughness.jsonl contains:
  {"task_id": ..., "score": ...}
"""

from dotenv import load_dotenv
load_dotenv()

import json
import dataset
from router_client import client, ROUTER

OUTPUT_PATH = "results/toughness.jsonl"

router = client.routers[ROUTER]

with open(OUTPUT_PATH, "w") as out:
    for i, problem in enumerate(dataset.load()):
        prompt = dataset.as_message(problem)[-1]["content"]
        score = router.calculate_strong_win_rate(prompt)
        record = {
            "task_id": problem.task_id,
            "score":   score,
        }
        out.write(json.dumps(record) + "\n")
        out.flush()
        print(f"[{i+1}] {problem.task_id} -> {score:.4f}")

print(f"\nDone. Results saved to {OUTPUT_PATH}")
