"""
routing.py - Run the full dataset through the RouteLLM router.
"""

from pathlib import Path

from route_llm_inference.router_client import client, ROUTER, THRESHOLD
from route_llm_inference.inference import run_inference
from util.dataset import load, count


def run_routing(split: str = "train", output_dir: str = "route_llm_results") -> None:
    """
    Route each problem to weak or strong model via the BERT router.

    Args:
        split:      "train" or "test"
        output_dir: directory for output .jsonl file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "router_results.jsonl"

    run_inference(
        problems=load(split=split),
        create_fn=client.completions.create,
        model_str=f"router-{ROUTER}-{THRESHOLD}",
        output_path=str(output_path),
        total=count(split=split),
    )
