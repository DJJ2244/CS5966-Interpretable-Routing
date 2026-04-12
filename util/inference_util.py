"""
inference_util.py - Inference loop, RouteLLM client, and OpenAI proxy client.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm

# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(
    problems,
    create_fn,
    model_str: str,
    model_name: str,
    total: Optional[int] = None,
    max_workers: int = 8,
) -> None:
    """Run inference on a dataset of problems and write results to the database.

    Args:
        problems:    Iterable of objects with .task_id and .prompt.
        create_fn:   Callable matching client.completions.create signature.
        model_str:   Model string passed to create_fn.
        model_name:  HuggingFace model name used to upsert into model_task_result.
        total:       Total problem count for progress bar.
        max_workers: Number of concurrent inference threads.
    """
    def infer(problem):
        start = time.monotonic()
        response = create_fn(
            model=model_str,
            prompt=problem.prompt,
            max_tokens=2048,
            temperature=0,
        )
        run_millis = int((time.monotonic() - start) * 1000)
        return {
            "task_id":    problem.task_id,
            "model":      response.model,
            "completion": response.choices[0].text,
            "run_millis": run_millis,
        }

    pbar = tqdm(total=total, desc=f"{model_str:<20}", unit="problem", leave=True)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(infer, problem): problem for problem in problems}
        for future in as_completed(futures):
            record = future.result()
            _write_to_db(record, model_name)
            pbar.update(1)
    pbar.close()


def _write_to_db(record: dict, model_name: str) -> None:
    from daos.model_task_result_dao import ModelTaskResult, upsert
    upsert(
        ModelTaskResult(
            task_id=record["task_id"],
            model_name=model_name,
            result=record["completion"],
            run_millis=record.get("run_millis"),
            extracted_code=None,
            passed=None,
        ),
    )


# ── OpenAI / litellm proxy client ────────────────────────────────────────────

def get_openai_client():
    """Return an OpenAI client pointed at the local litellm proxy."""
    from openai import OpenAI
    base = os.environ["OPENAI_API_BASE"].rstrip("/") + "/v1"
    return OpenAI(base_url=base, api_key="dummy")
