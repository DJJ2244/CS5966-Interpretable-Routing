"""
inference_util.py - Inference loop, RouteLLM client, and OpenAI proxy client.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from tqdm import tqdm

ROUTER    = "bert"
THRESHOLD = 0.11593

FLUSH_EVERY = 10


# ── Inference loop ────────────────────────────────────────────────────────────

def run_inference(
    problems,
    create_fn,
    model_str: str,
    output_path,
    conn=None,
    model_id: Optional[int] = None,
    total: Optional[int] = None,
    max_workers: int = 8,
) -> None:
    """Run inference on a dataset of problems and write results to output_path.

    Args:
        problems:    Iterable of objects with .task_id and .prompt.
        create_fn:   Callable matching client.completions.create signature.
        model_str:   Model string passed to create_fn.
        output_path: Path to the output .jsonl file.
        conn:        Optional DB connection. If provided along with model_id,
                     each result is also upserted into model_task_result.
        model_id:    DB model id, required when conn is provided.
        total:       Total problem count for progress bar.
        max_workers: Number of concurrent inference threads.
    """
    from util.smart_file_util import write_jsonl

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

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
                    if conn is not None and model_id is not None:
                        _write_to_db(conn, record, model_id)
                pbar.update(1)
        pbar.close()

    tqdm.write(f"Done. Results saved to {output_path}")


def _write_to_db(conn, record: dict, model_id: int) -> None:
    from daos.model_task_result_dao import ModelTaskResult, upsert
    upsert(
        conn,
        ModelTaskResult(
            task_id=record["task_id"],
            model_id=model_id,
            result=record["completion"],
            run_millis=record.get("run_millis"),
            extracted_code=None,
            passed=None,
        ),
    )


# ── RouteLLM router client ────────────────────────────────────────────────────

_router_client = None
_router_lock = threading.Lock()


def get_router_client():
    """Return the shared RouteLLM Controller, creating it lazily on first call."""
    global _router_client
    if _router_client is None:
        with _router_lock:
            if _router_client is None:
                from routellm.controller import Controller
                _router_client = Controller(
                    routers=[ROUTER],
                    strong_model=os.environ["STRONG_MODEL"],
                    weak_model=os.environ["WEAK_MODEL"],
                )
    return _router_client


# ── OpenAI / litellm proxy client ────────────────────────────────────────────

def get_openai_client():
    """Return an OpenAI client pointed at the local litellm proxy."""
    from openai import OpenAI
    base = os.environ["OPENAI_API_BASE"].rstrip("/") + "/v1"
    return OpenAI(base_url=base, api_key="dummy")
