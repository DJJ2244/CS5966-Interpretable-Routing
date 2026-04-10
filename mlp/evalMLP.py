"""
evalMLP.py - Evaluate the trained MLP classifier on the test split.

Input:
  activations/weak_sae_features_test.pt          — sparse features [N, 32768]
  testing_results_weak_test.jsonl                — labels (passed: true/false)
  mlp_output/weak_mlp.pt                         — trained MLP weights

Output: classification report printed to stdout
"""

import json
import torch
import torch.nn as nn

MODEL_KEY     = "weak"   # weak | strong
SPLIT_KEY     = "test"   # train | test

FEATURES_PATH = f"activations/activations_{SPLIT_KEY}_{MODEL_KEY}_sparse.pt"
LABELS_PATH   = f"route_llm_results/testing_results_{MODEL_KEY}_{SPLIT_KEY}.jsonl"
MODEL_PATH    = f"mlp_output/mlp_train_{MODEL_KEY}.pt"
HIDDEN_DIM    = 256

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def evaluate():
    # ── Load features ──────────────────────────────────────
    feat_data = torch.load(FEATURES_PATH, map_location="cpu")
    feat_ids  = feat_data["task_ids"]
    feat_mat  = feat_data["features"]

    # ── Load labels ────────────────────────────────────────
    labels_map = {}
    with open(LABELS_PATH) as f:
        for line in f:
            rec = json.loads(line)
            labels_map[rec["task_id"]] = int(rec["passed"])

    # ── Align ──────────────────────────────────────────────
    X, y = [], []
    for i, tid in enumerate(feat_ids):
        if tid in labels_map:
            X.append(feat_mat[i])
            y.append(labels_map[tid])

    X = torch.stack(X).to(device)
    y = torch.tensor(y, dtype=torch.float32)

    print(f"Test set: {len(y)} samples")
    print(f"  Positive (weak passed): {int(y.sum())}")
    print(f"  Negative (weak failed): {int((y == 0).sum())}")

    # ── Load model ─────────────────────────────────────────
    model = MLP(d_in=X.shape[1], hidden=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # ── Inference ──────────────────────────────────────────
    with torch.no_grad():
        logits = model(X)
        preds  = (logits > 0).float().cpu()

    y = y.cpu()

    # ── Metrics ────────────────────────────────────────────
    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()

    acc       = (tp + tn) / len(y)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n── Results ──────────────────────────────")
    print(f"  Accuracy  : {acc:.3f}")
    print(f"  Precision : {precision:.3f}  (of problems routed to weak, how many it actually passed)")
    print(f"  Recall    : {recall:.3f}  (of problems weak can pass, how many we correctly route to it)")
    print(f"  F1        : {f1:.3f}")
    print(f"\n── Confusion Matrix ─────────────────────")
    print(f"               Predicted weak  Predicted strong")
    print(f"  Actual pass  {tp:14d}  {fn:16d}")
    print(f"  Actual fail  {fp:14d}  {tn:16d}")


if __name__ == "__main__":
    evaluate()
