"""
baseline.py - Run weak and/or strong model inference without routing.
"""

import os
from pathlib import Path
from openai import OpenAI

from route_llm_inference.inference import run_inference
from util.dataset import load, count


def run_baseline(model: str = "all", split: str = "train", output_dir: str = "route_llm_results") -> None:
    """
    Run baseline inference for the selected model(s).

    Args:
        model:      "weak", "strong", or "all"
        split:      "train" or "test"
        output_dir: directory for output .jsonl files
    """
    weak_name   = os.environ["WEAK_MODEL"].removeprefix("openai/")
    strong_name = os.environ["STRONG_MODEL"].removeprefix("openai/")
    client      = OpenAI(base_url=os.environ["OPENAI_API_BASE"].rstrip("/") + "/v1", api_key="dummy")
    total       = count(split=split)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    targets = []
    if model in ("weak", "all"):
        targets.append((weak_name,   Path(output_dir) / "results_weak.jsonl"))
    if model in ("strong", "all"):
        targets.append((strong_name, Path(output_dir) / "results_strong.jsonl"))

    if not targets:
        raise ValueError(f"Unknown model '{model}'. Choose weak, strong, or all.")

    for model_str, output_path in targets:
        print(f"\n=== Running {model_str} ===")
        run_inference(
            problems=load(split=split),
            create_fn=client.chat.completions.create,
            model_str=model_str,
            output_path=str(output_path),
            total=total,
        )
