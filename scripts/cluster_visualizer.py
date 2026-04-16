#!/usr/bin/env python3
"""
cluster_visualizer.py — SAE sparse feature clustering and interactive HTML viz.

Pipeline
--------
  1. Load sparse SAE feature vectors
  2. Filter dead features  (zero-activation columns removed)
  3. TruncatedSVD to --svd-components dims
  4. K-means clustering   (default k=12)
  5. UMAP 2-D projection
  6. Overlay routing decisions (weak=green / strong=red) from JSONL
  7. [Optional] Neuronpedia cross-reference
  8. Label clusters with local meta-llama/Meta-Llama-3-8B via 🤗 transformers
  9. Generate self-contained interactive HTML

Usage examples
--------------
# By split-id + model name:
    python scripts/cluster_visualizer.py \\
        --split-id 1 --model-name meta-llama/Llama-3.2-1B --k 12

# Direct path override:
    python scripts/cluster_visualizer.py \\
        --features-path activations/activations_1_meta-llama_Llama-3.2-1B_sparse.pt

# Skip LLM labelling:
    python scripts/cluster_visualizer.py --features-path ... --no-llm

# With routing decisions overlay:
    python scripts/cluster_visualizer.py \\
        --split-id 1 --model-name meta-llama/Llama-3.2-1B \\
        --routing-decisions routing_decisions.jsonl
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# Project root so we can import util/ from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so $STRONG_MODEL, $HF_HOME, etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import util.smart_file_util as sfu


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster SAE sparse features and build an interactive HTML visualisation."
    )

    # ── feature source (one of two ways) ──────────────────────────────────
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--features-path", type=Path, metavar="PATH",
        help="Direct path to a sparse-features .pt file.",
    )
    src.add_argument(
        "--split-id", type=int, default=None,
        help="Split ID (used with --model-name to resolve path via smart_file_util).",
    )
    p.add_argument(
        "--model-name", type=str, default=None,
        help="Full model name, e.g. 'meta-llama/Llama-3.2-1B' (used with --split-id).",
    )

    # ── clustering & reduction ────────────────────────────────────────────
    p.add_argument("--k", type=int, default=12, metavar="K",
                   help="Number of clusters (default: 12).")
    p.add_argument("--svd-components", type=int, default=128,
                   help="TruncatedSVD dimensionality before clustering (default: 128).")
    p.add_argument("--top-k-features", type=int, default=20,
                   help="Top features per cluster shown to the LLM (default: 20).")
    p.add_argument("--umap-neighbors", type=int, default=30,
                   help="UMAP n_neighbors (default: 30).")
    p.add_argument("--umap-min-dist", type=float, default=0.3,
                   help="UMAP min_dist (default: 0.3).")
    p.add_argument("--umap-spread", type=float, default=1.5,
                   help="UMAP spread (default: 1.5).")

    # ── Neuronpedia (optional) ─────────────────────────────────────────────
    p.add_argument("--neuronpedia-model-id", type=str, default=None, metavar="MODEL_ID",
                   help="Neuronpedia model ID for feature lookup (e.g. 'gpt2-small').")
    p.add_argument("--neuronpedia-layer", type=str, default=None, metavar="LAYER",
                   help="Neuronpedia layer string (e.g. '7-res-jb').")

    # ── routing overlay ───────────────────────────────────────────────────
    p.add_argument(
        "--routing-decisions", type=Path, default=None,
        help="Path to routing_decisions.jsonl for weak/strong colour overlay. "
             "Defaults to <project_root>/routing_decisions.jsonl if it exists.",
    )

    # ── LLM labelling ─────────────────────────────────────────────────────
    p.add_argument("--no-llm", action="store_true",
                   help="Skip LLM labelling (use generic 'Cluster N' labels).")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Load LLM in 4-bit quantization (requires bitsandbytes). "
                        "Cuts VRAM to ~4 GB for 7B models — useful on small GPUs.")
    p.add_argument(
        "--llm-model", type=str,
        default=os.environ.get("STRONG_MODEL", "meta-llama/Meta-Llama-3-8B"),
        help="HuggingFace model ID for cluster labelling "
             "(default: $STRONG_MODEL or meta-llama/Meta-Llama-3-8B).",
    )
    p.add_argument("--samples-per-cluster", type=int, default=5,
                   help="Task prompt snippets per cluster shown to the LLM (default: 5).")

    # ── output ────────────────────────────────────────────────────────────
    p.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "visuals" / "cluster_explorer.html",
        help="Output HTML path (default: visuals/cluster_explorer.html).",
    )
    p.add_argument(
        "--db-path", type=Path,
        default=PROJECT_ROOT / "data" / "routing.db",
        help="SQLite DB for task prompts/descriptions (default: data/routing.db).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def resolve_features_path(args: argparse.Namespace) -> Path:
    if args.features_path:
        return args.features_path
    if args.split_id is not None and args.model_name:
        return PROJECT_ROOT / sfu.sparse_features_path(args.split_id, args.model_name)
    raise SystemExit(
        "Error: supply --features-path OR both --split-id and --model-name."
    )


def load_features_file(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load .pt file → (float32 numpy array [N, d], list of task_ids)."""
    print(f"Loading features from {path} …")
    raw = torch.load(str(path), weights_only=False, map_location="cpu")
    if isinstance(raw, dict):
        tensor = raw["features"] if "features" in raw else raw["activations"]
        task_ids = list(raw.get("task_ids", [f"task_{i}" for i in range(tensor.shape[0])]))
    else:
        tensor = raw
        task_ids = [f"task_{i}" for i in range(tensor.shape[0])]
    return tensor.float().numpy(), task_ids


def load_task_prompts(db_path: Path) -> dict[str, str]:
    """Return {task_id: description} from the tasks table.

    Uses the natural-language description field, which contains only the
    algorithmic concept without language boilerplate.  Falls back to the
    natural_language column, then to stripping the docstring from prompt.
    """
    if not db_path.exists():
        return {}
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, description, natural_language, prompt FROM tasks"
        ).fetchall()
        conn.close()
        result = {}
        for r in rows:
            desc = (r["description"] or r["natural_language"] or "").strip()
            if not desc:
                # fallback: extract first triple-quoted string from prompt
                prompt = r["prompt"] or ""
                m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
                if m:
                    desc = m.group(1).strip().split("\n")[0][:200]
                else:
                    desc = prompt.strip().split("\n")[0][:120]
            result[r["id"]] = desc[:200]
        return result
    except Exception as exc:
        print(f"Warning: could not load task prompts ({exc})")
        return {}


def load_routing_decisions(path: Path | None) -> dict[str, str]:
    """Return {task_id: 'weak'|'strong'} from a routing decisions JSONL."""
    # Try the default path if none supplied
    if path is None:
        default = PROJECT_ROOT / "routing_decisions.jsonl"
        if default.exists():
            path = default
        else:
            return {}
    if not path.exists():
        print(f"Warning: routing decisions file not found: {path}")
        return {}
    decisions: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                tid = obj.get("task_id")
                route = obj.get("route")
                if tid and route in ("weak", "strong"):
                    decisions[tid] = route
            except json.JSONDecodeError:
                pass
    print(f"  Loaded {len(decisions)} routing decisions from {path}")
    return decisions


# ---------------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------------

def filter_dead_features(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove SAE feature dimensions that are zero across all samples.

    Returns (filtered_features [N, d_live], live_mask [d_sae]).
    """
    live_mask = features.max(axis=0) > 0
    n_dead = int((~live_mask).sum())
    features_live = features[:, live_mask]
    print(f"  Dead feature filter: {n_dead} dead / {features.shape[1]} total → "
          f"{features_live.shape[1]} live features")
    return features_live, live_mask


def reduce_svd(features: np.ndarray, n_components: int) -> tuple[np.ndarray, TruncatedSVD]:
    """TruncatedSVD on sparse-ish features (no centering needed)."""
    n_comp = min(n_components, features.shape[1] - 1, features.shape[0] - 1)
    print(f"  TruncatedSVD: {features.shape[1]} → {n_comp} components …")
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    reduced = svd.fit_transform(features)
    explained = svd.explained_variance_ratio_.sum()
    print(f"  Explained variance: {explained:.1%}")
    return reduced, svd


def project_umap(
    features: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    n_neighbors: int,
    min_dist: float,
    spread: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """UMAP 2-D projection; centroid = mean of per-cluster projected points."""
    try:
        import umap
    except ImportError:
        raise SystemExit(
            "Error: umap-learn not installed. Run: pip install umap-learn"
        )
    print(f"  UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, spread={spread}) …")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        spread=spread,
        random_state=42,
        metric="cosine",
        low_memory=True,
    )
    pts2d = reducer.fit_transform(features)
    centroids_2d = np.array([
        pts2d[cluster_labels == i].mean(axis=0) for i in range(k)
    ])
    return pts2d, centroids_2d


# ---------------------------------------------------------------------------
# Neuronpedia
# ---------------------------------------------------------------------------

def query_neuronpedia(
    model_id: str,
    layer: str,
    feature_indices: list[int],
    top_k: int = 10,
) -> dict[int, str]:
    try:
        import requests
    except ImportError:
        print("Warning: 'requests' not installed — skipping Neuronpedia lookup.")
        return {}

    descriptions: dict[int, str] = {}
    targets = feature_indices[:top_k]
    print(f"Querying Neuronpedia ({model_id}/{layer}) for {len(targets)} features …")
    for idx in targets:
        url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{idx}"
        try:
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                desc = (
                    data.get("description")
                    or (data.get("explanations") or [{}])[0].get("description", "")
                    or data.get("label", "")
                )
                if desc:
                    descriptions[int(idx)] = str(desc)[:120]
        except Exception:
            pass
        time.sleep(0.1)

    print(f"  Got {len(descriptions)} Neuronpedia descriptions.")
    return descriptions


# ---------------------------------------------------------------------------
# LLM cluster labelling — local HuggingFace model (no API)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_LLM = (
    "You are an expert at analysing clusters of coding problems. "
    "Given representative coding tasks, identify the underlying algorithmic or "
    "programming concept they share. "
    "The label must name a concrete programming concept such as "
    "'dynamic programming', 'string manipulation', 'binary search', "
    "'tree traversal', 'graph algorithms', 'sorting', 'recursion', etc. "
    "Do NOT reference the visualization method, clustering, feature vectors, "
    "sparse activations, SAE, UMAP, dimensions, or any machine learning terminology. "
    "Do NOT use generic labels like '2D clustering', 'feature mapping', "
    "'data points', or anything that describes the analysis process rather than "
    "the actual programming problems. "
    "Respond with ONLY a JSON object — no preamble, no markdown fences:\n"
    '{"label": "<2-5 word programming concept>", "description": "<one sentence about the coding tasks>"}'
)

_LABEL_PROMPT_TEMPLATE = """\
Top activated SAE feature dimensions for cluster {cid}:
{feature_lines}

Representative coding tasks in this cluster:
{sample_lines}

What algorithmic or programming concept do these coding problems share? \
Examples of good labels: "dynamic programming", "string manipulation", \
"binary search", "tree traversal", "graph algorithms", "sorting", "recursion". \
Do NOT use labels like "2D clustering", "feature vectors", "sparse activations", \
or any term from machine learning or data visualization — label the coding concept only. \
Respond with ONLY: {{"label": "...", "description": "..."}}"""

# Completion-style prompt for base (non-instruct) models
_COMPLETION_PROMPT_TEMPLATE = """\
### Sparse autoencoder feature cluster analysis
### Top activated SAE feature dimensions for cluster {cid}:
{feature_lines}
### Representative coding tasks in this cluster:
{sample_lines}
### Short cluster label (2-5 words, algorithmic concept only): """


def _is_instruct(model_name: str) -> bool:
    return "instruct" in model_name.lower() or "chat" in model_name.lower()


def load_llm(model_name: str, load_in_4bit: bool = False):
    """Load HuggingFace causal LM.

    Pass load_in_4bit=True to quantize to ~4 GB VRAM (requires bitsandbytes).
    Without it the 7B model needs ~15 GB — only load on a node with enough VRAM
    or it will page to CPU and stall.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        print(f"  HF_HOME={hf_home}")

    print(f"Loading tokenizer: {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"device_map": "auto"}
    if load_in_4bit:
        print("  4-bit quantization enabled (bitsandbytes)")
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    print(f"Loading model: {model_name} …")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    print(f"  Model loaded — device map: {getattr(model, 'hf_device_map', 'single device')}")
    return model, tokenizer


def label_cluster_llm(
    model,
    tokenizer,
    cluster_id: int,
    top_indices: np.ndarray,
    top_activations: np.ndarray,
    neuronpedia: dict[int, str],
    task_samples: list[str],
    model_name: str = "",
) -> dict[str, str]:
    """Label one cluster using the local LLM.

    Instruct/chat models (Qwen-Instruct, Llama-Instruct, etc.) use the
    tokenizer's chat template for proper formatting.  Base models fall back
    to plain completion prompting.
    """
    feat_lines = []
    for idx, act in zip(top_indices[:15].tolist(), top_activations[:15].tolist()):
        line = f"  Feature {idx}  (mean_act={act:.4f})"
        if idx in neuronpedia:
            line += f"  →  {neuronpedia[idx]}"
        feat_lines.append(line)

    sample_lines = "\n".join(
        f"  • {s}" for s in task_samples[:5]
    ) if task_samples else "  (no task samples)"

    user_content = _LABEL_PROMPT_TEMPLATE.format(
        cid=cluster_id,
        feature_lines="\n".join(feat_lines),
        sample_lines=sample_lines,
    )

    if _is_instruct(model_name) and hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT_LLM},
            {"role": "user",   "content": user_content},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = _COMPLETION_PROMPT_TEMPLATE.format(
            cid=cluster_id,
            feature_lines="\n".join(feat_lines),
            sample_lines=sample_lines,
        )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

    # Build stop-token list: include model EOS + any chat end tokens (e.g. Qwen's <|im_end|>)
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    for tok in ("<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id:
            stop_ids.add(tid)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            eos_token_id=list(stop_ids),
            pad_token_id=next(iter(stop_ids)),
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Try to parse JSON first (instruct models follow the format reliably)
    m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            lbl = parsed.get("label", "").strip()[:60]
            desc = parsed.get("description", "").strip()[:200]
            if lbl:
                return {"label": lbl, "description": desc}
        except json.JSONDecodeError:
            pass

    # Fallback: take first non-empty line
    label = next((ln.strip() for ln in raw.splitlines() if ln.strip()), raw)
    label = re.sub(r'[\"\'`{}]', "", label).strip()[:60]
    return {"label": label or f"Cluster {cluster_id}", "description": raw[:200]}


# ---------------------------------------------------------------------------
# Visualisation JSON + HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAE Feature Cluster Explorer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
     background:#0d1117;color:#c9d1d9;height:100vh;overflow:hidden}
#app{display:grid;grid-template-columns:300px 1fr;grid-template-rows:48px 1fr;height:100vh}
#hdr{grid-column:1/-1;background:#161b22;border-bottom:1px solid #30363d;
     display:flex;align-items:center;gap:14px;padding:0 18px}
#hdr h1{font-size:15px;font-weight:600;color:#f0f6fc;white-space:nowrap}
#hdr .sub{font-size:12px;color:#8b949e}
/* routing legend in header */
.rt-badge{display:inline-flex;align-items:center;gap:5px;font-size:12px;
          padding:2px 9px;border-radius:10px;font-weight:500;margin-left:6px}
.rt-weak  {background:#14532d;color:#4ade80}
.rt-strong{background:#450a0a;color:#f87171}
.rt-none  {background:#1c2128;color:#8b949e}
#side{background:#161b22;border-right:1px solid #30363d;overflow-y:auto;
      display:flex;flex-direction:column;gap:14px;padding:14px}
#plot-wrap{position:relative;overflow:hidden}
.sec{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.07em;
     color:#8b949e;margin-bottom:6px}
#search{width:100%;background:#21262d;border:1px solid #30363d;border-radius:6px;
        color:#c9d1d9;padding:6px 10px;font-size:13px;outline:none}
#search:focus{border-color:#58a6ff}
/* cluster list items */
.cl-item{display:flex;align-items:center;gap:8px;padding:7px 9px;border-radius:6px;
         cursor:pointer;transition:background .12s;border:1px solid transparent}
.cl-item:hover{background:#21262d}
.cl-item.active{background:#21262d;border-color:#30363d}
.cl-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.cl-lbl{font-size:13px;font-weight:500;flex:1;white-space:nowrap;overflow:hidden;
        text-overflow:ellipsis}
.cl-sz{font-size:11px;color:#8b949e;flex-shrink:0}
/* routing mini-bars on cluster items */
.cl-rt{display:flex;height:3px;border-radius:2px;overflow:hidden;width:40px;flex-shrink:0}
.cl-rt-w{background:#22c55e}
.cl-rt-s{background:#ef4444}
/* info panel */
#info{background:#1c2128;border:1px solid #30363d;border-radius:8px;padding:13px;
      display:none;flex-direction:column;gap:8px}
#info.show{display:flex}
#inf-title{font-size:14px;font-weight:600;color:#f0f6fc}
#inf-desc{font-size:12px;color:#8b949e;line-height:1.55}
#inf-rt{font-size:12px;display:flex;gap:8px;flex-wrap:wrap}
.feat-row{display:flex;align-items:center;gap:7px;padding:2px 0}
.feat-id{font-family:monospace;font-size:11px;color:#79c0ff;width:72px;flex-shrink:0}
.feat-bg{flex:1;height:4px;background:#30363d;border-radius:2px}
.feat-bar{height:100%;border-radius:2px}
.feat-txt{font-size:11px;color:#8b949e;max-width:140px;white-space:nowrap;
          overflow:hidden;text-overflow:ellipsis}
#stats{font-size:12px;color:#8b949e;border-top:1px solid #30363d;padding-top:10px}
.st-row{display:flex;justify-content:space-between;padding:2px 0}
.st-val{color:#c9d1d9}
</style>
</head>
<body>
<div id="app">
  <div id="hdr">
    <div><h1>SAE Feature Cluster Explorer</h1></div>
    <div class="sub" id="hdr-sub"></div>
    <div style="margin-left:auto;display:flex;align-items:center">
      <span class="rt-badge rt-weak">&#9679; Weak</span>
      <span class="rt-badge rt-strong">&#9679; Strong</span>
      <span class="rt-badge rt-none">&#9679; No data</span>
    </div>
  </div>
  <div id="side">
    <div>
      <div class="sec">Search</div>
      <input id="search" type="text" placeholder="Filter by task ID or keyword…">
    </div>
    <div>
      <div class="sec">Clusters</div>
      <div id="cl-list"></div>
    </div>
    <div id="info">
      <div id="inf-title"></div>
      <div id="inf-desc"></div>
      <div id="inf-rt"></div>
      <div id="inf-feats"></div>
    </div>
    <div id="stats">
      <div class="sec">Stats</div>
      <div class="st-row"><span>Total points</span><span class="st-val" id="st-total">—</span></div>
      <div class="st-row"><span>Visible</span><span class="st-val" id="st-visible">—</span></div>
      <div class="st-row"><span>Active cluster</span><span class="st-val" id="st-active">—</span></div>
      <div class="st-row"><span>Routed weak</span><span class="st-val" id="st-weak">—</span></div>
      <div class="st-row"><span>Routed strong</span><span class="st-val" id="st-strong">—</span></div>
    </div>
  </div>
  <div id="plot-wrap">
    <div id="plot" style="width:100%;height:100%"></div>
  </div>
</div>

<!-- embedded data -->
<script type="application/json" id="viz-data">
__VIZ_DATA__
</script>

<script>
'use strict';
const DATA = JSON.parse(document.getElementById('viz-data').textContent);
const K = DATA.centroids.length;

// cluster colour palette
const PAL = [
  '#58a6ff','#3fb950','#f78166','#d2a8ff','#ffa657','#79c0ff',
  '#56d364','#ff7b72','#bc8cff','#ff9a3c','#1f6feb','#238636',
  '#c0392b','#7c3aed','#92400e','#0ea5e9','#16a34a','#dc2626',
  '#9333ea','#ca8a04',
];
function clCol(id){ return PAL[id % PAL.length]; }

// routing colours
const ROUTE_COL = { weak:'#22c55e', strong:'#ef4444', null:'#4b5563' };
function routeCol(r){ return ROUTE_COL[r] || ROUTE_COL[null]; }

function hexToRgba(hex, alpha){
  const r=parseInt(hex.slice(1,3),16);
  const g=parseInt(hex.slice(3,5),16);
  const b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// trace layout: [0..K-1] hull fills, [K..2K-1] point scatters,
//               [2K] centroid labels (text only), [2K+1/2K+2] routing legend
const HULL_TRACES = Array.from({length:K},(_,i)=>i);
const PT_TRACES   = Array.from({length:K},(_,i)=>K+i);

// global state
let selectedCluster = null;

// ── pre-compute per-cluster arrays ────────────────────────────────────────
const byCluster = {};
for(let i=0;i<K;i++) byCluster[i]={xs:[],ys:[],ids:[],texts:[],colors:[],routes:[]};
let totalWeak=0, totalStrong=0;
for(const pt of DATA.points){
  const c=pt.c;
  const lbl=DATA.centroids[c].label;
  byCluster[c].xs.push(pt.x);
  byCluster[c].ys.push(pt.y);
  byCluster[c].ids.push(pt.id);
  byCluster[c].routes.push(pt.r||null);
  byCluster[c].colors.push(routeCol(pt.r||null));
  const routeTag = pt.r==='weak'
    ? '<span style="color:#4ade80">&#11044; weak</span>'
    : pt.r==='strong'
      ? '<span style="color:#f87171">&#11044; strong</span>'
      : '<span style="color:#6b7280">&#11044; —</span>';
  const snippet = pt.d
    ? `<span style="color:#8b949e;font-size:11px">${pt.d}</span>`
    : '';
  byCluster[c].texts.push(
    `<b>${pt.id}</b>  ${routeTag}<br>`+
    `<i style="color:#8b949e;font-size:11px">cluster: ${lbl}</i>`+
    (snippet?`<br>${snippet}`:'')
  );
  if(pt.r==='weak') totalWeak++;
  else if(pt.r==='strong') totalStrong++;
}

// ── build Plotly traces ───────────────────────────────────────────────────
function buildTraces(){
  const traces=[];

  // ── [0..K-1] convex hull fills ────────────────────────────────────────
  for(let i=0;i<K;i++){
    const c=DATA.centroids[i];
    const hull=c.hull||[];
    const rtLine=(c.n_weak||c.n_strong)
      ? `<span style="color:#4ade80">${c.n_weak||0} weak</span> / `
        +`<span style="color:#f87171">${c.n_strong||0} strong</span><br>`
      : '';
    const hov=`<b>${c.label}</b><br>${c.desc||''}<br>${rtLine}`
             +`<i style="color:#8b949e">n = ${c.n}</i>`;
    traces.push({
      type:'scatter', mode:'lines',
      name:c.label,
      x: hull.length ? hull.map(v=>v[0]) : [],
      y: hull.length ? hull.map(v=>v[1]) : [],
      fill:'toself',
      fillcolor: hexToRgba(clCol(i), 0.20),
      line:{color: hexToRgba(clCol(i), 0.70), width:1.5},
      customdata: Array(hull.length).fill(i),
      hovertemplate: hov+'<extra></extra>',
      legendgroup:`c${i}`, showlegend:false,
    });
  }

  // ── [K..2K-1] per-cluster scatter — routing-colour per point ──────────
  for(let i=0;i<K;i++){
    const b=byCluster[i];
    traces.push({
      type:'scatter', mode:'markers',
      name:DATA.centroids[i].label,
      x:b.xs, y:b.ys,
      text:b.texts, customdata:b.ids,
      hovertemplate:'%{text}<extra></extra>',
      marker:{
        size:5,
        color:b.colors,
        opacity:0.80,
        symbol:'circle',
        line:{width:0},
      },
      legendgroup:`c${i}`, showlegend:false,
    });
  }

  // ── [2K] centroid labels — always-visible text, no markers ───────────
  const cx=[],cy=[],ctxt=[];
  for(const c of DATA.centroids){
    cx.push(c.x); cy.push(c.y);
    ctxt.push(c.label);
  }
  traces.push({
    type:'scatter', mode:'text',
    name:'Labels',
    x:cx, y:cy,
    text:ctxt,
    textposition:'middle center',
    textfont:{size:12, color:'#f0f6fc',
              family:'-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif'},
    hoverinfo:'skip',
    showlegend:false,
    cliponaxis:false,
  });

  // ── routing legend entries (dummy traces) ─────────────────────────────
  traces.push({
    type:'scatter', mode:'markers', name:'Weak',
    x:[null], y:[null], showlegend:true,
    marker:{color:'#22c55e',size:9,symbol:'circle'},
    legendgroup:'routing',
  });
  traces.push({
    type:'scatter', mode:'markers', name:'Strong',
    x:[null], y:[null], showlegend:true,
    marker:{color:'#ef4444',size:9,symbol:'circle'},
    legendgroup:'routing',
  });

  return traces;
}

const LAYOUT={
  paper_bgcolor:'#0d1117', plot_bgcolor:'#161b22',
  font:{color:'#8b949e',family:'-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif'},
  xaxis:{title:{text:'UMAP 1',font:{size:11}},gridcolor:'#21262d',zerolinecolor:'#30363d',tickfont:{size:10}},
  yaxis:{title:{text:'UMAP 2',font:{size:11}},gridcolor:'#21262d',zerolinecolor:'#30363d',tickfont:{size:10}},
  legend:{bgcolor:'#161b22',bordercolor:'#30363d',borderwidth:1,font:{size:11},x:1.01,y:1},
  margin:{l:50,r:120,t:28,b:48},
  hovermode:'closest',
  hoverlabel:{bgcolor:'#1c2128',bordercolor:'#58a6ff',font:{color:'#c9d1d9',size:12}},
};

// ── sidebar cluster list ──────────────────────────────────────────────────
function renderClusterList(){
  const list=document.getElementById('cl-list');
  list.innerHTML='';
  for(const c of DATA.centroids){
    const total=c.n||1;
    const wPct=Math.round(((c.n_weak||0)/total)*100);
    const sPct=Math.round(((c.n_strong||0)/total)*100);
    const div=document.createElement('div');
    div.className='cl-item'; div.id=`ci${c.id}`;
    div.innerHTML=
      `<div class="cl-dot" style="background:${clCol(c.id)}"></div>`+
      `<div class="cl-lbl">${c.label}</div>`+
      `<div class="cl-rt">`+
        `<div class="cl-rt-w" style="width:${wPct}%"></div>`+
        `<div class="cl-rt-s" style="width:${sPct}%"></div>`+
      `</div>`+
      `<div class="cl-sz">${c.n}</div>`;
    div.onclick=()=>toggleCluster(c.id);
    list.appendChild(div);
  }
}

// ── info panel ────────────────────────────────────────────────────────────
function showInfo(id){
  const c=DATA.centroids[id];
  document.getElementById('info').classList.add('show');
  document.getElementById('inf-title').textContent=c.label;
  document.getElementById('inf-desc').textContent=c.desc||'';
  const nW=c.n_weak||0, nS=c.n_strong||0, nU=(c.n||0)-nW-nS;
  document.getElementById('inf-rt').innerHTML=
    `<span class="rt-badge rt-weak">${nW} weak</span>`+
    `<span class="rt-badge rt-strong">${nS} strong</span>`+
    (nU>0?`<span class="rt-badge rt-none">${nU} n/a</span>`:'');
  const maxA=Math.max(...(c.feats||[]).map(f=>f.a||0),1e-9);
  const html=(c.feats||[]).slice(0,10).map(f=>{
    const pct=Math.round((f.a||0)/maxA*100);
    return `<div class="feat-row">`+
      `<div class="feat-id">feat ${f.i}</div>`+
      `<div class="feat-bg"><div class="feat-bar" style="width:${pct}%;background:${clCol(id)}"></div></div>`+
      (f.desc?`<div class="feat-txt" title="${f.desc}">${f.desc}</div>`:'')+
      `</div>`;
  }).join('');
  document.getElementById('inf-feats').innerHTML=html||
    '<div style="color:#8b949e;font-size:12px">No feature data</div>';
  document.getElementById('st-active').textContent=`${c.label} (${c.n})`;
}

function hideInfo(){
  document.getElementById('info').classList.remove('show');
  document.getElementById('st-active').textContent='—';
}

// ── cluster toggle ────────────────────────────────────────────────────────
function toggleCluster(id){
  if(selectedCluster===id){
    selectedCluster=null;
    Plotly.restyle('plot',{'opacity':0.20},HULL_TRACES);
    Plotly.restyle('plot',{'marker.opacity':0.80},PT_TRACES);
    document.querySelectorAll('.cl-item').forEach(el=>el.classList.remove('active'));
    hideInfo(); updateVisible(DATA.points.length);
  } else {
    selectedCluster=id;
    const hullOps=HULL_TRACES.map(i=>i===id?0.35:0.04);
    const ptOps=PT_TRACES.map(i=>(i-K)===id?0.90:0.10);
    Plotly.restyle('plot',{'opacity':hullOps},HULL_TRACES);
    Plotly.restyle('plot',{'marker.opacity':ptOps},PT_TRACES);
    document.querySelectorAll('.cl-item').forEach(el=>el.classList.remove('active'));
    document.getElementById(`ci${id}`)?.classList.add('active');
    showInfo(id); updateVisible(DATA.centroids[id].n);
  }
}

function updateVisible(n){ document.getElementById('st-visible').textContent=n; }

// ── search ────────────────────────────────────────────────────────────────
document.getElementById('search').addEventListener('input',function(){
  const q=this.value.trim().toLowerCase();
  if(!q){
    Plotly.restyle('plot',{'opacity':0.20},HULL_TRACES);
    Plotly.restyle('plot',{'marker.opacity':0.80},PT_TRACES);
    updateVisible(DATA.points.length); return;
  }
  const hits=new Set();
  for(const pt of DATA.points){
    if(pt.id.toLowerCase().includes(q)||(pt.d&&pt.d.toLowerCase().includes(q)))
      hits.add(pt.c);
  }
  Plotly.restyle('plot',{'opacity':HULL_TRACES.map(i=>hits.has(i)?0.30:0.04)},HULL_TRACES);
  Plotly.restyle('plot',{'marker.opacity':PT_TRACES.map(i=>hits.has(i-K)?0.85:0.08)},PT_TRACES);
  updateVisible([...hits].reduce((s,c)=>s+DATA.centroids[c].n,0));
});

// ── init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded',function(){
  Plotly.newPlot('plot', buildTraces(), LAYOUT, {
    responsive:true, displayModeBar:true,
    modeBarButtonsToRemove:['select2d','lasso2d'],
    displaylogo:false,
  });

  // centroid click
  document.getElementById('plot').on('plotly_click',function(data){
    if(!data||!data.points||!data.points[0]) return;
    const pt=data.points[0];
    // hull traces are 0..K-1; clicking inside a hull fires with that curveNumber
    if(pt.curveNumber < K) toggleCluster(pt.curveNumber);
  });

  renderClusterList();

  const hasRouting = DATA.points.some(p=>p.r);
  document.getElementById('hdr-sub').textContent=
    `${K} clusters · UMAP · ${DATA.points.length} points`+
    (hasRouting?' · routing overlay':'');
  document.getElementById('st-total').textContent=DATA.points.length;
  document.getElementById('st-weak').textContent=totalWeak||'—';
  document.getElementById('st-strong').textContent=totalStrong||'—';
  updateVisible(DATA.points.length);
});
</script>
</body>
</html>
"""


def build_viz_json(
    points_2d: np.ndarray,
    cluster_labels: np.ndarray,
    centroids_2d: np.ndarray,
    cluster_info: list[dict],
    task_ids: list[str],
    task_prompts: dict[str, str],
    routing: dict[str, str],
) -> str:
    """Serialise everything the HTML needs into a compact JSON string."""
    points = []
    for i, (x, y) in enumerate(points_2d):
        tid = task_ids[i] if i < len(task_ids) else f"pt_{i}"
        route = routing.get(tid)
        points.append({
            "x": round(float(x), 5),
            "y": round(float(y), 5),
            "c": int(cluster_labels[i]),
            "id": tid,
            "d": (task_prompts.get(tid, "")[:100]),
            **({"r": route} if route else {}),
        })

    try:
        from scipy.spatial import ConvexHull as _ConvexHull
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    centroids = []
    for info in cluster_info:
        cid = info["cluster_id"]
        cx, cy = centroids_2d[cid]

        # count routing decisions
        cluster_tids = [task_ids[i] for i in range(len(task_ids)) if cluster_labels[i] == cid]
        n_weak = sum(1 for tid in cluster_tids if routing.get(tid) == "weak")
        n_strong = sum(1 for tid in cluster_tids if routing.get(tid) == "strong")

        # convex hull vertices (closed polygon)
        cluster_pts = points_2d[cluster_labels == cid]
        hull_verts: list[list[float]] = []
        if _has_scipy and len(cluster_pts) >= 3:
            try:
                hull = _ConvexHull(cluster_pts)
                verts = cluster_pts[hull.vertices]
                verts_closed = np.vstack([verts, verts[0]])
                hull_verts = [
                    [round(float(v[0]), 4), round(float(v[1]), 4)]
                    for v in verts_closed
                ]
            except Exception:
                pass

        centroids.append({
            "x": round(float(cx), 5),
            "y": round(float(cy), 5),
            "id": cid,
            "label": info.get("label", f"Cluster {cid}"),
            "desc": info.get("description", ""),
            "n": info["size"],
            "n_weak": n_weak,
            "n_strong": n_strong,
            "hull": hull_verts,
            "feats": [
                {
                    "i": int(idx),
                    "a": round(float(act), 5),
                    **({"desc": info["neuronpedia"].get(int(idx), "")}
                       if info.get("neuronpedia", {}).get(int(idx)) else {}),
                }
                for idx, act in zip(
                    info["top_indices"][:10].tolist(),
                    info["top_activations"][:10].tolist(),
                )
            ],
        })

    payload = {
        "points": points,
        "centroids": centroids,
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def generate_html(viz_json: str) -> str:
    return _HTML_TEMPLATE.replace("__VIZ_DATA__", viz_json)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── load features ─────────────────────────────────────────────────────
    feat_path = resolve_features_path(args)
    if not feat_path.exists():
        raise SystemExit(
            f"Error: features file not found: {feat_path}\n"
            "Run `python cli.py sae extract` first."
        )
    features, task_ids = load_features_file(feat_path)
    n_samples, d_feat = features.shape
    print(f"  {n_samples} samples × {d_feat} features")

    # ── task prompts from DB ──────────────────────────────────────────────
    task_prompts = load_task_prompts(args.db_path)
    print(f"  {len(task_prompts)} task prompts loaded from DB")

    # ── routing decisions ─────────────────────────────────────────────────
    routing = load_routing_decisions(args.routing_decisions)

    # ── filter dead features ──────────────────────────────────────────────
    print("\nFiltering dead features …")
    features_live, _ = filter_dead_features(features)

    # ── SVD ───────────────────────────────────────────────────────────────
    print("\nReducing with TruncatedSVD …")
    features_svd, _ = reduce_svd(features_live, args.svd_components)

    # ── k-means ───────────────────────────────────────────────────────────
    print(f"\nRunning k-means (k={args.k}) …")
    km = KMeans(n_clusters=args.k, random_state=42, n_init=10, max_iter=300, verbose=0)
    labels = km.fit_predict(features_svd)
    for i in range(args.k):
        print(f"  Cluster {i:2d}: {int((labels == i).sum()):4d} samples")

    # ── UMAP ──────────────────────────────────────────────────────────────
    print("\nProjecting to 2-D via UMAP …")
    pts2d, centroids_2d = project_umap(
        features_svd, labels, args.k, args.umap_neighbors, args.umap_min_dist, args.umap_spread
    )

    # ── top features per cluster (on original live features) ──────────────
    print("\nComputing top features per cluster …")
    cluster_info = []
    for cid in range(args.k):
        mask = labels == cid
        mean_act = features_live[mask].mean(axis=0)
        top_idx = np.argsort(mean_act)[::-1][:args.top_k_features]
        cluster_info.append({
            "cluster_id": cid,
            "size": int(mask.sum()),
            "top_indices": top_idx,
            "top_activations": mean_act[top_idx],
            "label": f"Cluster {cid}",
            "description": "",
            "neuronpedia": {},
        })

    # ── Neuronpedia ───────────────────────────────────────────────────────
    if args.neuronpedia_model_id and args.neuronpedia_layer:
        all_feat_ids = sorted({
            int(idx)
            for info in cluster_info
            for idx in info["top_indices"][:10].tolist()
        })
        np_descs = query_neuronpedia(
            args.neuronpedia_model_id,
            args.neuronpedia_layer,
            all_feat_ids,
            top_k=min(len(all_feat_ids), 60),
        )
        for info in cluster_info:
            info["neuronpedia"] = {
                int(idx): np_descs[int(idx)]
                for idx in info["top_indices"][:10].tolist()
                if int(idx) in np_descs
            }

    # ── LLM labelling ─────────────────────────────────────────────────────
    if not args.no_llm:
        try:
            llm_model, llm_tokenizer = load_llm(args.llm_model, load_in_4bit=args.load_in_4bit)
            print(f"\nLabelling {args.k} clusters with {args.llm_model} …")

            for info in cluster_info:
                cid = info["cluster_id"]
                cluster_tids = [
                    task_ids[i] for i in range(n_samples) if labels[i] == cid
                ]
                samples = [
                    task_prompts[tid]
                    for tid in cluster_tids[: args.samples_per_cluster]
                    if task_prompts.get(tid)
                ]

                print(f"  Cluster {cid} …", end=" ", flush=True)
                result = label_cluster_llm(
                    model=llm_model,
                    tokenizer=llm_tokenizer,
                    cluster_id=cid,
                    top_indices=info["top_indices"],
                    top_activations=info["top_activations"],
                    neuronpedia=info.get("neuronpedia", {}),
                    task_samples=samples,
                )
                info["label"] = result.get("label", f"Cluster {cid}")
                info["description"] = result.get("description", "")
                print(f"→ \"{info['label']}\"")

            # free GPU memory
            del llm_model, llm_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except ImportError:
            print("Warning: 'transformers' not installed — skipping LLM labelling.")
            print("  Install with: pip install transformers accelerate")
        except Exception as exc:
            print(f"Warning: LLM labelling failed ({exc}). Continuing with generic labels.")

    # ── build HTML ────────────────────────────────────────────────────────
    print("\nGenerating HTML …")
    viz_json = build_viz_json(
        points_2d=pts2d,
        cluster_labels=labels,
        centroids_2d=centroids_2d,
        cluster_info=cluster_info,
        task_ids=task_ids,
        task_prompts=task_prompts,
        routing=routing,
    )
    html = generate_html(viz_json)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")

    size_kb = args.output.stat().st_size / 1024
    print(f"\nSaved → {args.output}  ({size_kb:.0f} KB)")
    print(f"Open:   file://{args.output.resolve()}")


if __name__ == "__main__":
    main()
