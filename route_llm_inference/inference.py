"""
inference.py - Shared inference loop for running a model over the full dataset.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

FLUSH_EVERY = 10


def run_inference(problems, create_fn, model_str, output_path, total=None, max_workers=8):
    """
    Run inference on a dataset of problems and write results to output_path.

    Args:
        problems:    Iterable of problems (each with .task_id and .prompt).
        create_fn:   A callable with the signature of client.completions.create
                     (must accept `model`, `prompt`, `max_tokens`, `temperature` kwargs).
        model_str:   The model string passed to create_fn.
        output_path: Path to the output .jsonl file.
        total:       Total number of problems (for the progress bar percentage).
        max_workers: Number of concurrent inference requests.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def infer(problem):
        response = create_fn(
            model=model_str,
            prompt=problem.prompt,
            max_tokens=2048,
            temperature=0,
        )
        return {
            "task_id":    problem.task_id,
            "model":      response.model,
            "completion": response.choices[0].text,
        }

    with open(output_path, "w") as out:
        pbar = tqdm(total=total, desc=f"{model_str:<20}", unit="problem", leave=True)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(infer, problem): problem for problem in problems}
            for i, future in enumerate(as_completed(futures)):
                record = future.result()
                with lock:
                    out.write(json.dumps(record) + "\n")
                    if i % FLUSH_EVERY == 0:
                        out.flush()
                pbar.update(1)
        pbar.close()

    tqdm.write(f"Done. Results saved to {output_path}")
