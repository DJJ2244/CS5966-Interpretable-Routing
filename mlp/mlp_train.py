"""
mlp/mlp_train.py - Train the MLP router on SAE sparse feature vectors.

Labels are read from the model_task_result DB table.
Features are loaded from the canonical activations path.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mlp.model import MLP, HIDDEN_DIM
from util import tensor_util
from util.smart_file_util import mlp_path

LR         = 1e-3
EPOCHS     = 50
BATCH_SIZE = 64
SEED       = 42


class SAEDataset(Dataset):
    def __init__(self, split_id: int, model_id: int):
        from daos import model_task_result_dao

        features_dict = tensor_util.load_features(split_id, model_id)
        labels        = model_task_result_dao.get_all_for_model_split(
            model_id, split_id, is_test=False
        )

        self.X, self.y, self.ids = tensor_util.align_features_with_labels(
            features_dict, labels
        )

        print(f"Dataset: {len(self.X)} samples")
        print(f"  Positive (weak passed): {int(self.y.sum())}")
        print(f"  Negative (weak failed): {int((self.y == 0).sum())}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_mlp(split_id: int, model_id: int) -> None:
    """Train an MLP router for the given split/model pair.

    Args:
        split_id: DB split id (selects the training partition).
        model_id: DB model id (selects which model's features + results to use).
    """
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset   = SAEDataset(split_id, model_id)
    loader    = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model     = MLP(d_in=dataset.X.shape[1], hidden=HIDDEN_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
            model.eval()
            with torch.no_grad():
                logits = model(dataset.X.to(device))
                preds  = (logits > 0).float().cpu()
                acc    = (preds == dataset.y).float().mean().item()
            print(f"  Epoch {epoch:3d} | loss: {total_loss/len(loader):.4f} | train acc: {acc:.3f}")

    save_path = mlp_path(split_id, model_id)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    print(f"\nSaved → {save_path}")
