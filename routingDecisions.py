"""
routingDecisions.py - Run the MLP router on the test split and write per-problem decisions.

Input:
  activations/weak_sae_features_test.pt   — sparse features [N, 32768]
  testing_results_weak_test.jsonl         — labels (passed: true/false)
  mlp_output/weak_mlp.pt                  — trained MLP weights

Output:
  routing_decisions.jsonl  — one record per problem:
    { task_id, routed_to, weak_passed, correct }
"""

import json
import torch
import torch.nn as nn

FEATURES_PATH   = "activations/weak_sae_features_test.pt"
LABELS_PATH     = "testing_results_weak_test.jsonl"
MODEL_PATH      = "mlp_output/weak_mlp.pt"
OUTPUT_PATH     = "routing_decisions.jsonl"
HIDDEN_DIM      = 256

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


def run():
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
    task_ids, X, y = [], [], []
    for i, tid in enumerate(feat_ids):
        if tid in labels_map:
            task_ids.append(tid)
            X.append(feat_mat[i])
            y.append(labels_map[tid])

    X = torch.stack(X).to(device)
    y = torch.tensor(y, dtype=torch.float32)

    # ── Load model ─────────────────────────────────────────
    model = MLP(d_in=X.shape[1], hidden=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # ── Inference ──────────────────────────────────────────
    with torch.no_grad():
        logits = model(X)
        preds  = (logits > 0).float().cpu()

    # ── Write decisions ────────────────────────────────────
    records = []
    for tid, pred, actual in zip(task_ids, preds.tolist(), y.tolist()):
        routed_to   = "weak" if pred == 1.0 else "strong"
        weak_passed = bool(actual)
        # correct = routed to weak and it can pass, OR routed to strong (always a fallback)
        correct     = (routed_to == "weak" and weak_passed) or (routed_to == "strong")
        records.append({
            "task_id":     tid,
            "routed_to":   routed_to,
            "weak_passed": weak_passed,
            "correct":     correct,
        })

    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    # ── Summary ────────────────────────────────────────────
    n           = len(records)
    to_weak     = sum(1 for r in records if r["routed_to"] == "weak")
    to_strong   = sum(1 for r in records if r["routed_to"] == "strong")
    correct     = sum(1 for r in records if r["correct"])
    bad_routes  = sum(1 for r in records if r["routed_to"] == "weak" and not r["weak_passed"])

    print(f"Wrote {n} decisions → {OUTPUT_PATH}")
    print(f"  Routed to weak:   {to_weak} ({to_weak/n:.1%})")
    print(f"  Routed to strong: {to_strong} ({to_strong/n:.1%})")
    print(f"  Correct routes:   {correct} ({correct/n:.1%})")
    print(f"  Bad routes (sent to weak but failed): {bad_routes} ({bad_routes/n:.1%})")


if __name__ == "__main__":
    run()
