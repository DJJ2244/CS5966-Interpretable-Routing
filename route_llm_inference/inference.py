"""
inference.py - Shared inference loop for running a model over the full dataset.
"""

import json
from pathlib import Path
from tqdm import tqdm

FLUSH_EVERY = 10


def run_inference(problems, create_fn, model_str, output_path, total=None):
    """
    Run inference on a dataset of problems and write results to output_path.

    Args:
        problems:    Iterable of problems (each with .task_id and .prompt).
        create_fn:   A callable with the signature of client.chat.completions.create
                     (must accept `model` and `messages` kwargs).
        model_str:   The model string passed to create_fn.
        output_path: Path to the output .jsonl file.
        total:       Total number of problems (for the progress bar percentage).
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out:
        pbar = tqdm(total=total, desc=f"{model_str:<20}", unit="problem", leave=True)
        for i, problem in enumerate(problems):
            response = create_fn(
                model=model_str,
                prompt=problem.prompt,
                max_tokens=4096,
            )
            record = {
                "task_id":    problem.task_id,
                "model":      response.model,
                "completion": response.choices[0].text,
            }
            out.write(json.dumps(record) + "\n")
            if i % FLUSH_EVERY == 0:
                out.flush()
            pbar.update(1)
        pbar.close()

    tqdm.write(f"Done. Results saved to {output_path}")
