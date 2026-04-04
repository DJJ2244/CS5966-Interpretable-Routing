"""
extractActivation_mock.py - Smoke test for the CHPC cluster.

Runs the exact same pipeline as extractActivation.py but only on 2 problems
per model so you can verify GPU access, model loading, caching, and file I/O
before committing to the full dataset run.
"""

import torch
import os
import json
from pathlib import Path
from transformer_lens import HookedTransformer

# ── Config ────────────────────────────────────────────────
MODELS = {
    "weak":   "meta-llama/Llama-3.2-1B",
    "strong": "meta-llama/Meta-Llama-3-8B",
}

OUTPUT_DIR = "activations_mock"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MOCK_N = 2  # number of problems to run per model

# ── Load 2 problems from local JSONL ─────────────────────
DATA_PATH = Path("data/humaneval_xl_english.jsonl")

problems = []
with open(DATA_PATH) as f:
    for line in f:
        rec = json.loads(line.strip())
        if rec.get("programming_language", "python") == "python":
            problems.append((rec["task_id"], rec["prompt"]))
        if len(problems) >= MOCK_N:
            break

print(f"Mock run: {len(problems)} problems loaded from {DATA_PATH}")


# ── Extraction Function (identical logic to extractActivation.py) ──
def extract_activations(model_name, model_key, problems):
    print(f"\nLoading {model_key} model: {model_name}")

    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()

    num_layers   = model.cfg.n_layers
    middle_layer = num_layers // 2
    print(f"Model has {num_layers} layers — extracting layer {middle_layer}")

    all_vectors = []
    task_ids = []

    for i, (task_id, prompt) in enumerate(problems):
        try:
            tokens = model.to_tokens(prompt, truncate=True)

            with torch.no_grad():
                logits, cache = model.run_with_cache(
                    tokens,
                    names_filter=f"blocks.{middle_layer}.hook_resid_post"
                )

            residual = cache[f"blocks.{middle_layer}.hook_resid_post"]
            vector = residual.mean(dim=1).squeeze(0).cpu().float()
            all_vectors.append(vector)
            task_ids.append(task_id)

            del cache, logits
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Failed on {task_id}: {e}")
            all_vectors.append(torch.zeros(model.cfg.d_model))
            task_ids.append(task_id)

        print(f"  {i + 1}/{len(problems)} problems processed")

    activation_matrix = torch.stack(all_vectors)

    save_path = os.path.join(OUTPUT_DIR, f"{model_key}_activations_mock.pt")
    torch.save({"task_ids": task_ids, "activations": activation_matrix}, save_path)
    print(f"Saved {activation_matrix.shape} + {len(task_ids)} task IDs -> {save_path}")

    del model
    torch.cuda.empty_cache()

    return activation_matrix


# ── Run for Both Models ───────────────────────────────────
for model_key, model_name in MODELS.items():
    extract_activations(model_name, model_key, problems)

print("\nMock run complete. Files saved:")
for model_key in MODELS:
    path = f"{OUTPUT_DIR}/{model_key}_activations_mock.pt"
    data = torch.load(path)
    print(f"  {path}: activations={data['activations'].shape}, ids={data['task_ids']}")
