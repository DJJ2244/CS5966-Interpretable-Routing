"""
trainMLP.py - Train a simple MLP classifier on SAE sparse feature vectors.

Input:
  activations/weak_sae_features.pt       — sparse features [N, 32768]
  route_llm_results/testing_results_weak_train.jsonl — labels (passed: true/false)

Output:
  mlp_output/weak_mlp.pt  — trained MLP weights
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────
MODEL_KEY      = "weak"   # weak | strong
SPLIT_KEY      = "train"  # train | test

FEATURES_PATH  = f"activations/activations_{SPLIT_KEY}_{MODEL_KEY}_sparse.pt"
LABELS_PATH    = f"route_llm_results/testing_results_{MODEL_KEY}_{SPLIT_KEY}.jsonl"
OUTPUT_DIR     = "mlp_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM     = 256
LR             = 1e-3
EPOCHS         = 50
BATCH_SIZE     = 64
SEED           = 42

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ── Dataset ───────────────────────────────────────────────
class SAEDataset(Dataset):
    def __init__(self, features_path, labels_path):
        feat_data  = torch.load(features_path, map_location="cpu")
        feat_ids   = feat_data["task_ids"]
        feat_mat   = feat_data["features"]           # [N, d_sae]

        labels_map = {}
        with open(labels_path) as f:
            for line in f:
                rec = json.loads(line)
                labels_map[rec["task_id"]] = int(rec["passed"])

        # Align by task_id
        self.X, self.y, self.ids = [], [], []
        missing = 0
        for i, tid in enumerate(feat_ids):
            if tid in labels_map:
                self.X.append(feat_mat[i])
                self.y.append(labels_map[tid])
                self.ids.append(tid)
            else:
                missing += 1

        self.X = torch.stack(self.X)
        self.y = torch.tensor(self.y, dtype=torch.float32)

        print(f"Dataset: {len(self.X)} samples ({missing} skipped — no label match)")
        print(f"  Positive (weak passed): {self.y.sum().int().item()}")
        print(f"  Negative (weak failed): {(self.y == 0).sum().int().item()}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Model ─────────────────────────────────────────────────
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


# ── Train ─────────────────────────────────────────────────
def train():
    dataset    = SAEDataset(FEATURES_PATH, LABELS_PATH)
    loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model      = MLP(d_in=dataset.X.shape[1], hidden=HIDDEN_DIM).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)

    # Weighted loss to handle class imbalance
    n_pos      = dataset.y.sum().item()
    n_neg      = len(dataset.y) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"\nTraining MLP (d_in={dataset.X.shape[1]}, hidden={HIDDEN_DIM}) for {EPOCHS} epochs ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            # Quick train accuracy
            model.eval()
            with torch.no_grad():
                logits = model(dataset.X.to(device))
                preds  = (logits > 0).float().cpu()
                acc    = (preds == dataset.y).float().mean().item()
            print(f"  Epoch {epoch:3d} | loss: {total_loss/len(loader):.4f} | train acc: {acc:.3f}")

    save_path = os.path.join(OUTPUT_DIR, f"mlp_{SPLIT_KEY}_{MODEL_KEY}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"\nSaved → {save_path}")


if __name__ == "__main__":
    train()
