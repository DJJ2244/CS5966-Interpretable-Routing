"""
run_router.py - Run the full dataset through RouteLLM and save results to results.jsonl.

Each line of results.jsonl contains:
  {"task_id": ..., "model": ..., "completion": ...}
"""

from dotenv import load_dotenv
load_dotenv()

from router_client import client, ROUTER, THRESHOLD
from inference import run_inference

OUTPUT_PATH = "results/router_results.jsonl"

run_inference(
    create_fn=client.chat.completions.create,
    model_str=f"router-{ROUTER}-{THRESHOLD}",
    output_path=OUTPUT_PATH,
)
