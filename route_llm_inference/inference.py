"""
inference.py - Shared inference loop for running a model over the full dataset.
"""

import json

def run_inference(problems, create_fn, model_str, output_path):
    """
    Run inference on a dataset of problems and write results to output_path.

    Args:
        problems:   Iterable of problems (each with .task_id and a messages representation).
        create_fn:  A callable with the signature of client.chat.completions.create
                    (must accept `model` and `messages` kwargs).
        model_str:  The model string passed to create_fn.
        output_path: Path to the output .jsonl file.
    """
    with open(output_path, "w") as out:
        for i, problem in enumerate(problems):
            response = create_fn(
                model=model_str,
                messages=[{"role": "user", "content": problem.prompt}],
            )
            record = {
                "task_id":    problem.task_id,
                "model":      response.model,
                "completion": response.choices[0].message.content,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            print(f"[{i+1}] {problem.task_id} -> {response.model}")

    print(f"\nDone. Results saved to {output_path}")
