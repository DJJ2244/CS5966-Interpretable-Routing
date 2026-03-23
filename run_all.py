"""
run_all.py - Run the full dataset through RouteLLM and save results to results.jsonl.

Each line of results.jsonl contains:
  {"task_id": ..., "model": ..., "completion": ...}
"""

import json
from dotenv import load_dotenv
load_dotenv()

import dataset
from routellm.controller import Controller

WEAK_MODEL   = "ollama/llama3.2:1b"
STRONG_MODEL = "ollama/llama3"
ROUTER       = "bert"
THRESHOLD    = 0.11593
OUTPUT_PATH  = "results.jsonl"

client = Controller(
    routers=[ROUTER],
    strong_model=STRONG_MODEL,
    weak_model=WEAK_MODEL,
)

with open(OUTPUT_PATH, "w") as out:
    for i, problem in enumerate(dataset.load()):
        response = client.chat.completions.create(
            model=f"router-{ROUTER}-{THRESHOLD}",
            messages=dataset.as_message(problem),
        )
        record = {
            "task_id":    problem.task_id,
            "model":      response.model,
            "completion": response.choices[0].message.content,
        }
        out.write(json.dumps(record) + "\n")
        out.flush()
        print(f"[{i+1}] {problem.task_id} -> {response.model}")

print(f"\nDone. Results saved to {OUTPUT_PATH}")
