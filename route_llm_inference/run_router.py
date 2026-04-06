"""
run_router.py - Run the full dataset through RouteLLM and save results to results.jsonl.

Each line of results.jsonl contains:
  {"task_id": ..., "model": ..., "completion": ...}
"""

from dotenv import load_dotenv
load_dotenv()

import os
from routellm.controller import Controller
from inference import run_inference

WEAK_MODEL   = os.environ["WEAK_MODEL"]
STRONG_MODEL = os.environ["STRONG_MODEL"]
ROUTER       = "bert"
THRESHOLD    = 0.11593
OUTPUT_PATH  = "router_results.jsonl"

client = Controller(
    routers=[ROUTER],
    strong_model=STRONG_MODEL,
    weak_model=WEAK_MODEL,
)

run_inference(
    create_fn=client.chat.completions.create,
    model_str=f"router-{ROUTER}-{THRESHOLD}",
    output_path=OUTPUT_PATH,
)
