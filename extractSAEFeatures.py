"""
extractSAEFeatures.py - Pass saved residual stream activations through the
trained SAE encoder to produce sparse feature vectors.

Input : activations/weak_activations.pt  (shape: [N, d_model])
Output: activations/weak_sae_features.pt (shape: [N, d_sae])
"""

import os
import torch
import argparse

SAE_CHECKPOINT = "sae_output/weak/3ca7qrr9/final_5001216"
ACTIVATIONS_IN = "activations/weak_activations.pt"
FEATURES_OUT   = "activations/weak_sae_features.pt"


def extract_sae_features(sae_checkpoint: str, activations_path: str, output_path: str):
    from sae_lens import SAE

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load SAE ───────────────────────────────────────────
    print(f"Loading SAE from {sae_checkpoint} ...")
    sae = SAE.load_from_disk(sae_checkpoint, device=device)
    sae.eval()
    print(f"  SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    # ── Load activations ───────────────────────────────────
    print(f"\nLoading activations from {activations_path} ...")
    data        = torch.load(activations_path, map_location=device)
    activations = data["activations"].to(device)   # [N, d_model]
    task_ids    = data["task_ids"]
    print(f"  {activations.shape[0]} problems, d_model={activations.shape[1]}")

    # ── Encode through SAE ─────────────────────────────────
    print("\nEncoding through SAE ...")
    with torch.no_grad():
        feature_acts = sae.encode(activations)     # [N, d_sae]

    print(f"  Feature matrix: {feature_acts.shape}")
    print(f"  Avg active features per problem (L0): {(feature_acts > 0).float().sum(dim=1).mean():.1f}")
    print(f"  Dead features: {(feature_acts > 0).any(dim=0).logical_not().sum().item()} / {feature_acts.shape[1]}")

    # ── Save ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({"task_ids": task_ids, "features": feature_acts.cpu()}, output_path)
    print(f"\nSaved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae",         default=SAE_CHECKPOINT, help="Path to SAE final checkpoint")
    parser.add_argument("--activations", default=ACTIVATIONS_IN, help="Path to activations .pt file")
    parser.add_argument("--output",      default=FEATURES_OUT,   help="Path to save feature vectors")
    args = parser.parse_args()

    extract_sae_features(args.sae, args.activations, args.output)
