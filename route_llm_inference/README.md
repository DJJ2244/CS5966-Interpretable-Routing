# route_llm_inference

Runs inference on the HumanEval dataset using RouteLLM and individual models via a local Ollama server.

## Files

| File | Purpose |
|------|---------|
| `router_client.py` | Shared RouteLLM `Controller` instance. Single source of truth for router configuration (`ROUTER`, `THRESHOLD`, model names). Import from here to guarantee scores are consistent across scripts. |
| `inference.py` | Shared inference loop — iterates the dataset, calls a model, writes `.jsonl` results. Accepts any OpenAI-compatible `create` callable. |
| `run_router.py` | Runs all problems through the BERT router. Each problem is routed to the strong or weak model based on the threshold. Outputs `results/router_results.jsonl`. |
| `run_strong_and_weak.py` | Runs all problems through both models individually with no routing. Outputs `results/results_strong.jsonl` and `results/results_weak.jsonl`. |
| `record_toughness.py` | Scores every problem with the router and records the raw toughness score — no model inference is performed. Outputs `results/toughness.jsonl`. |

## Output format

**Inference scripts** (`run_router.py`, `run_strong_and_weak.py`) — one JSON object per line:
```json
{"task_id": "HumanEval/0", "model": "ollama/llama3:8b", "completion": "..."}
```

**Toughness script** (`record_toughness.py`) — one JSON object per line:
```json
{"task_id": "HumanEval/0", "score": 0.312}
```

## Running

All scripts must be run from the `route_llm_inference/` directory with Ollama running locally.

**Toughness scores only** (no Ollama models needed):
```bash
python record_toughness.py
```

**Strong and weak model baselines** (no routing):
```bash
python run_strong_and_weak.py
```

**Full router inference:**
```bash
python run_router.py
```

Results are written to the `results/` subdirectory.
