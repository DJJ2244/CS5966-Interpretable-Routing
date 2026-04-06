# CS5966-Interpretable-Routing

## Setup

```bash
python -m venv .venv
# ACTIVATE SESSION ----------
# Mac/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
# INSTALL DEPENDENCIES ----------
pip install -r requirements.txt
```

> For GPU acceleration on torch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130` (adjust CUDA version as needed)

Make sure [Ollama](https://ollama.com) is installed and running, then pull the models:

```bash
ollama pull llama3.2:1b
ollama pull llama3:8b
```

## Run

Scripts are run from the project root with Ollama running locally.

```bash
python record_toughness.py      # score all problems (no LLM inference)
python run_strong_and_weak.py   # run strong and weak models individually
python run_router.py            # run full router inference
```

Results are written to `route_llm_results/`.

### Output formats

**Inference scripts** (`run_router.py`, `run_strong_and_weak.py`) — one JSON object per line:
```json
{"task_id": "HumanEval/0", "model": "ollama/llama3:8b", "completion": "..."}
```

**Toughness script** (`record_toughness.py`) — one JSON object per line:
```json
{"task_id": "HumanEval/0", "score": 0.312}
```

### Inference modules (`route_llm_inference/`)

| File | Purpose |
|------|---------|
| `router_client.py` | Shared RouteLLM `Controller` instance. Single source of truth for router configuration (`ROUTER`, `THRESHOLD`, model names). Guarantees toughness scores are consistent across scripts. |
| `inference.py` | Shared inference loop — iterates a dataset, calls a model, writes `.jsonl` results. Accepts any OpenAI-compatible `create` callable. |

## Approach

The baseline router is RouteLLM's built-in `bert` router. The plan is to replace it with a custom router that runs the same bert method with our own MLP classification head, which is modified for our work. RouteLLM's router interface only requires one method — `calculate_strong_win_rate(prompt) -> float` — so the internals can be swapped freely without touching the rest of the framework.

## Evaluation

Inference and evaluation are intentionally split into two stages:

**Stage 1 — Inference (cluster or local):** Run the dataset through RouteLLM. Each problem is routed to either the strong or weak model, which generates a completion. Results are saved to JSONL files under `route_llm_results/`.

**Stage 2 — Evaluation (local):** Load the results file and run each completion against the problem's test cases. This is pure CPU logic — no GPUs, no LLMs — so it runs fast locally. Multi-language execution is handled here (see `testing/`).

This separation means the expensive inference can run on a Slurm cluster without needing Docker or language runtimes installed there.

## Docker (Local Evaluation)

See [`testing/`](testing/) for a fully containerized local setup. This is optional — only needed if you want to evaluate completions in languages other than Python without installing runtimes locally.
