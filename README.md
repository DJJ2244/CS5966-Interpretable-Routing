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

note: use pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 or similar for gpu acceleration on torch

Make sure [Ollama](https://ollama.com) is installed and running, then pull the models:

```bash
ollama pull llama3.2:1b
ollama pull llama3
```

## Run

Run a single problem (quick sanity check):

```bash
python main.py
```

Run the full dataset and save results to `results.jsonl`:

```bash
python run_all.py
```

Each line of `results.jsonl` is a JSON object:

```json
{"task_id": "...", "model": "...", "completion": "..."}
```

Progress is flushed after each problem, so the file is safe to inspect mid-run or resume after an interruption.

## Approach

The baseline router is RouteLLM's built-in `bert` router. The plan is to replace it with a custom router that runs the same bert method with our own MLP classification head, which is modified for our work. RouteLLM's router interface only requires one method — `calculate_strong_win_rate(prompt) -> float` — so the internals can be swapped freely without touching the rest of the framework.

## Evaluation

Inference and evaluation are intentionally split into two stages:

**Stage 1 — Inference (cluster or local):** Run the dataset through RouteLLM. Each problem is routed to either the strong or weak model, which generates a completion. Results are saved to a JSONL file (task_id, chosen model, completion).

**Stage 2 — Evaluation (local):** Load the results file and run each completion against the problem's test cases. This is pure CPU logic — no GPUs, no LLMs — so it runs fast locally. Multi-language execution is handled here (see `testing/`).

This separation means the expensive inference can run on a Slurm cluster without needing Docker or language runtimes installed there.

## Docker (Local Evaluation)

See [`testing/`](testing/) for a fully containerized local setup. This is optional — only needed if you want to evaluate completions in languages other than Python without installing runtimes locally.
