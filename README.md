# CS5966-Interpretable-Routing

## Setup

```bash
python -m venv .venv
# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

> For GPU acceleration: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` (adjust CUDA version as needed)

Models are downloaded from HuggingFace automatically on first use and cached in `~/.cache/huggingface/`. No re-download on subsequent runs.

Both models are gated and require a free HuggingFace account:

1. Accept the license for each model:
   - huggingface.co/meta-llama/Llama-3.2-1B
   - huggingface.co/meta-llama/Meta-Llama-3-8B
2. Authenticate locally:
   ```bash
   python experiment.py login
   ```

## Run

All experiment commands go through `experiment.py`. Start a session with `up`, run what you need, then `down` when done.

```bash
# Start inference servers (downloads models on first run — takes a few minutes)
python experiment.py up

# Session commands
python experiment.py inference --model weak     # weak model only
python experiment.py inference --model strong   # strong model only
python experiment.py inference --model all      # both (default)
python experiment.py route                      # RouteLLM router inference
python experiment.py toughness                  # BERT router scoring (no servers needed)
python experiment.py test --results route_llm_results/results_weak.jsonl  # Docker only, no servers needed

# Run the full pipeline in one shot
python experiment.py run-all

# Stop servers
python experiment.py down
```

All commands accept `--split train|test` (default: `train`) and `--output-dir` (default: `route_llm_results/`).

```bash
python experiment.py inference --model all --split test --output-dir my_results/
```

### Server options

```bash
python experiment.py up --weak-gpu 0 --strong-gpu 1   # default: separate GPUs
python experiment.py up --single-gpu                   # both models on GPU 0 (~19 GB VRAM)
python experiment.py status                            # check what's running
```

Server logs are written to `logs/servers/` for debugging.

### Output formats

**Inference** (`inference`, `route`) — one JSON object per line:
```json
{"task_id": "HumanEval/0", "model": "meta-llama/Llama-3.2-1B", "completion": "..."}
```

**Toughness** — one JSON object per line:
```json
{"task_id": "HumanEval/0", "score": 0.312}
```

## Models

Both pipelines use the same weights and precision:

| Role   | Model                          |
|--------|-------------------------------|
| Weak   | `meta-llama/Llama-3.2-1B`     |
| Strong | `meta-llama/Meta-Llama-3-8B`  |

Inference runs via vLLM (FP16, no quantization) behind a litellm proxy. This matches the activation extraction pipeline exactly — same HuggingFace weights, same tokenizer, same precision.

## Project Structure

```
experiment.py                    ← sole entry point (CLI)

servers/
  manager.py                     ← up/down/status server lifecycle

route_llm_inference/
  baseline.py                    ← weak/strong model inference
  routing.py                     ← RouteLLM router inference
  toughness.py                   ← BERT router scoring
  router_client.py               ← shared Controller instance
  inference.py                   ← shared inference loop

util/
  dataset.py                     ← HumanEval-XL loader
  split.py                       ← stratified train/test split

testing/
  runner.py                      ← Docker-based multi-language evaluation
```

## Approach

The baseline router is RouteLLM's built-in `bert` router. The plan is to replace it with a custom router using our own MLP classification head trained on model activations. RouteLLM's router interface only requires one method — `calculate_strong_win_rate(prompt) -> float` — so the internals can be swapped freely.

## Evaluation

Inference and evaluation are split into two stages:

**Stage 1 — Inference (cluster or local):** Run the dataset through the models. Results are saved to JSONL files under `route_llm_results/`.

**Stage 2 — Evaluation (local):** Run each completion against test cases via Docker. No GPUs or LLMs needed — fast, pure CPU. Multi-language execution is handled here (see `testing/`).

This separation means expensive inference can run on a Slurm cluster without needing Docker or language runtimes installed there.

## Data Split

```bash
python experiment.py split    # regenerate stratified 80/20 train/test split by language
```
