"""
router.py - End-to-end MLP router for a single coding prompt.

Loads the weak LLM, SAE, and MLP once, then routes any prompt in one call.

Usage:
  python router.py --prompt "Write a function that reverses a string."
  python router.py --file my_problem.txt
  python router.py --batch problems.jsonl   # jsonl with {"task_id", "prompt"} per line

Config (edit or override via CLI):
  --model     HuggingFace model ID for the weak model  (default: meta-llama/Llama-3.2-1B)
  --sae       Path to SAE checkpoint directory          (default: sae_output/weak/iucek2q3/final_5001216)
  --mlp       Path to trained MLP weights               (default: mlp_output/weak_mlp.pt)
  --layer     Residual stream layer to extract from     (default: middle layer)
  --output    Path to write batch results               (default: routing_output.jsonl)
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from pathlib import Path

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_SAE   = "sae_output/weak/iucek2q3/final_5001216"
DEFAULT_MLP   = "mlp_output/weak_mlp.pt"
HIDDEN_DIM    = 256

device = "cuda" if torch.cuda.is_available() else "cpu"


# ── MLP definition (must match trainMLP.py) ───────────────
class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Router class ──────────────────────────────────────────
class Router:
    def __init__(self, model_name: str, sae_path: str, mlp_path: str, layer: int = None):
        from transformer_lens import HookedTransformer
        from sae_lens import SAE

        print("Loading weak LLM ...")
        self.lm = HookedTransformer.from_pretrained(
            model_name,
            center_writing_weights=False,
            center_unembed=False,
            fold_ln=False,
            dtype=torch.float16,
            device=device,
        )
        self.lm.eval()

        self.layer = layer if layer is not None else self.lm.cfg.n_layers // 2
        print(f"  Extracting residual stream at layer {self.layer}/{self.lm.cfg.n_layers}")

        print("Loading SAE ...")
        self.sae = SAE.load_from_disk(sae_path, device=device)
        self.sae.eval()
        d_sae = self.sae.cfg.d_sae
        print(f"  SAE: d_in={self.sae.cfg.d_in}, d_sae={d_sae}")

        print("Loading MLP ...")
        self.mlp = MLP(d_in=d_sae, hidden=HIDDEN_DIM).to(device)
        self.mlp.load_state_dict(torch.load(mlp_path, map_location=device))
        self.mlp.eval()

        print("Router ready.\n")

    def route(self, prompt: str) -> dict:
        """Route a single prompt. Returns dict with decision and confidence."""
        # Step 1: residual stream activation — early exit after target layer
        tokens = self.lm.to_tokens(prompt, truncate=True)

        class _EarlyStop(Exception):
            pass

        captured = {}

        def _hook(value, hook=None):
            captured["act"] = value
            raise _EarlyStop()

        with torch.no_grad():
            try:
                self.lm.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{self.layer}.hook_resid_post", _hook)],
                )
            except _EarlyStop:
                pass

        activation = captured["act"].mean(dim=1).squeeze(0).float().cpu()  # [d_model]

        # Free LLM from GPU so SAE fits
        self.lm.to("cpu")
        torch.cuda.empty_cache()

        # Step 2: SAE sparse features (on GPU)
        with torch.no_grad():
            features = self.sae.encode(activation.to(device).unsqueeze(0))  # [1, d_sae]

        # Step 3: MLP routing decision
        with torch.no_grad():
            logit = self.mlp(features).item()

        # Move LLM back to GPU for next call
        self.lm.to(device)
        torch.cuda.empty_cache()

        return {
            "routed_to":  "weak" if logit > 0 else "strong",
            "logit":      round(logit, 4),
            "confidence": round(torch.sigmoid(torch.tensor(logit)).item(), 4),
        }

    def route_batch(self, problems: list[dict]) -> list[dict]:
        """Route a list of {"task_id", "prompt"} dicts."""
        results = []
        for i, p in enumerate(problems):
            decision = self.route(p["prompt"])
            decision["task_id"] = p["task_id"]
            results.append(decision)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(problems)} routed")
        return results


# ── CLI ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Route a coding prompt to weak or strong model.")
    parser.add_argument("--prompt", type=str,        help="Prompt string to route")
    parser.add_argument("--file",   type=Path,       help="Text file containing the prompt")
    parser.add_argument("--batch",  type=Path,       help="JSONL file with {task_id, prompt} per line")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--sae",    default=DEFAULT_SAE)
    parser.add_argument("--mlp",    default=DEFAULT_MLP)
    parser.add_argument("--layer",  type=int,        default=None)
    parser.add_argument("--output", default="routing_output.jsonl")
    args = parser.parse_args()

    if not any([args.prompt, args.file, args.batch]):
        parser.error("Provide one of --prompt, --file, or --batch")

    router = Router(model_name=args.model, sae_path=args.sae, mlp_path=args.mlp, layer=args.layer)

    if args.prompt:
        result = router.route(args.prompt)
        print(f"Decision : {result['routed_to'].upper()}")
        print(f"Logit    : {result['logit']}")
        print(f"Confidence (weak): {result['confidence']:.1%}")
        print(f"\nFull result: {json.dumps(result, indent=2)}")

    elif args.file:
        prompt = args.file.read_text()
        result = router.route(prompt)
        print(f"Decision : {result['routed_to'].upper()}")
        print(f"Logit    : {result['logit']}")
        print(f"Confidence (weak): {result['confidence']:.1%}")

    elif args.batch:
        problems = []
        with open(args.batch) as f:
            for line in f:
                problems.append(json.loads(line))

        print(f"Routing {len(problems)} problems ...")
        results = router.route_batch(problems)

        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        to_weak   = sum(1 for r in results if r["routed_to"] == "weak")
        to_strong = len(results) - to_weak
        print(f"\nWrote {len(results)} decisions → {args.output}")
        print(f"  Weak:   {to_weak} ({to_weak/len(results):.1%})")
        print(f"  Strong: {to_strong} ({to_strong/len(results):.1%})")
        print(f"\nFirst result: {json.dumps(results[0], indent=2)}")


if __name__ == "__main__":
    main()
