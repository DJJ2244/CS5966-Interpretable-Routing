"""
run_router.py - Run the full dataset through RouteLLM and save results to results.jsonl.

Each line of results.jsonl contains:
  {"task_id": ..., "model": ..., "completion": ...}
"""

from dotenv import load_dotenv
load_dotenv()

from route_llm_inference.router_client import client, ROUTER, THRESHOLD
from route_llm_inference.inference import run_inference
from util.dataset import load

OUTPUT_PATH = "route_llm_results/router_results.jsonl"

run_inference(
    problems=load(),
    create_fn=client.chat.completions.create,
    model_str=f"router-{ROUTER}-{THRESHOLD}",
    output_path=OUTPUT_PATH,
)
