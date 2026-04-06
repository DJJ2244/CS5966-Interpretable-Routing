"""
run_strong_and_weak.py - Run both the strong and weak models individually on all
dataset problems (no routing) and save results to separate .jsonl files.

Each line of output contains:
  {"task_id": ..., "model": ..., "completion": ...}
"""

from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI
from route_llm_inference.inference import run_inference
from util.dataset import load

# Strip the "ollama/" litellm prefix — Ollama's own API just uses the bare model name
WEAK_MODEL   = os.environ["WEAK_MODEL"].removeprefix("ollama/")
STRONG_MODEL = os.environ["STRONG_MODEL"].removeprefix("ollama/")

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

print("=== Running STRONG model ===")
run_inference(
    problems=load(),
    create_fn=client.chat.completions.create,
    model_str=STRONG_MODEL,
    output_path="route_llm_results/results_strong.jsonl",
)

print("\n=== Running WEAK model ===")
run_inference(
    problems=load(),
    create_fn=client.chat.completions.create,
    model_str=WEAK_MODEL,
    output_path="route_llm_results/results_weak.jsonl",
)
