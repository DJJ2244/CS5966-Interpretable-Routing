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

OUTPUT_DIR  = "activations"
MODEL_KEY   = "weak"   # weak | strong
SPLIT_KEY   = "train"  # train | test
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH  = Path(f"data/humaneval_xl_english_{SPLIT_KEY}.jsonl")

# ── Load Dataset ──────────────────────────────────────────
problems = []
with open(DATA_PATH) as f:
    for line in f:
        rec = json.loads(line.strip())
        problems.append((rec["task_id"], rec["prompt"]))
print(f"Loaded {len(problems)} problems from {DATA_PATH}")


# ── Extraction Function ───────────────────────────────────
def extract_activations(model_name, model_key, problems):
    print(f"\nLoading {model_key} model: {model_name}")

    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        dtype=torch.float16,   # saves memory
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()

    # Find the middle layer
    num_layers   = model.cfg.n_layers
    middle_layer = num_layers // 2
    print(f"Model has {num_layers} layers — extracting layer {middle_layer}")

    all_vectors = []
    task_ids = []

    for i, (task_id, prompt) in enumerate(problems):
        try:
            # Tokenize — truncate long prompts so they fit in memory
            tokens = model.to_tokens(prompt, truncate=True)

            # Forward pass — only cache the one layer we need
            with torch.no_grad():
                logits, cache = model.run_with_cache(
                    tokens,
                    names_filter=f"blocks.{middle_layer}.hook_resid_post"
                )

            # Pull residual stream at middle layer
            # Shape: [1, seq_len, hidden_dim]
            residual = cache[f"blocks.{middle_layer}.hook_resid_post"]

            # Mean pool across tokens → one vector per problem
            # Shape: [hidden_dim]
            vector = residual.mean(dim=1).squeeze(0).cpu().float()
            all_vectors.append(vector)
            task_ids.append(task_id)

            # Clean up cache immediately to save memory
            del cache, logits
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Failed on {task_id}: {e}")
            # Append a zero vector so indices stay aligned
            all_vectors.append(torch.zeros(model.cfg.d_model))
            task_ids.append(task_id)

        if i % 20 == 0:
            print(f"  {i}/{len(problems)} problems processed")

    # Stack into one matrix — Shape: [num_problems, hidden_dim]
    activation_matrix = torch.stack(all_vectors)

    # Save both the tensor and ordered task IDs together
    save_path = os.path.join(OUTPUT_DIR, f"activations_{SPLIT_KEY}_{model_key}.pt")
    torch.save({"task_ids": task_ids, "activations": activation_matrix}, save_path)
    print(f"Saved {activation_matrix.shape} + {len(task_ids)} task IDs → {save_path}")

    # Free model from memory before loading the next one
    del model
    torch.cuda.empty_cache()

    return activation_matrix


# ── Run for configured model and split ───────────────────
extract_activations(MODELS[MODEL_KEY], MODEL_KEY, problems)

print("\nDone. Files saved:")
path = f"{OUTPUT_DIR}/activations_{SPLIT_KEY}_{MODEL_KEY}.pt"
data = torch.load(path)
print(f"  {path}: activations={data['activations'].shape}, ids={len(data['task_ids'])}")
