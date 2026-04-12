"""
mlp/mlp_train.py - Train the MLP router on SAE sparse feature vectors.

Labels are read from the model_task_result DB table.
Features are loaded from the canonical activations path.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from mlp.model import MLP, HIDDEN_DIM
from util import tensor_util
from util.smart_file_util import mlp_path

LR         = 1e-3
EPOCHS     = 50
BATCH_SIZE = 64
SEED       = 42
VAL_SPLIT  = 0.2   # 20% of training data held out for grid search validation

GRID = {
    "lr":         [1e-2, 1e-3, 1e-4],
    "hidden_dim": [64, 128, 256, 512],
    "epochs":     [25, 50, 100],
}


class SAEDataset(Dataset):
    def __init__(self, split_id: int, model_name: str):
        from daos import model_task_result_dao

        features_dict = tensor_util.load_features(split_id, model_name)
        labels        = model_task_result_dao.get_all_for_model_split(
            model_name, split_id, is_test=False
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


def _run_training(dataset, d_in, lr, epochs, batch_size, hidden_dim, device):
    #Run one full training pass. Returns the trained model.
    n_pos      = dataset.y.sum().item() if hasattr(dataset, "y") else sum(y for _, y in dataset)
    n_neg      = len(dataset) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(device)

    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model     = MLP(d_in=d_in, hidden=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            optimizer.step()

    return model


def _evaluate(model, X, y, device):
    #Evaluate model on X, y tensors. Returns accuracy
    model.eval()
    with torch.no_grad():
        preds = (model(X.to(device)) > 0).float().cpu()
    return (preds == y).float().mean().item()


def grid_search_mlp(dataset, device) -> dict:
    """Single holdout grid search over GRID. Returns the best hyperparams dict.

    Splits the training dataset 80/20, trains each hyperparameter combination
    on the 80% portion, evaluates on the held-out 20%, and returns the
    combination with the highest validation accuracy.
    """
    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_subset, val_subset = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Pull val tensors out once so we don't re-index every evaluation
    val_indices = val_subset.indices
    val_X = dataset.X[val_indices]
    val_y = dataset.y[val_indices]

    combos = [
        {"lr": lr, "hidden_dim": hd, "epochs": ep}
        for lr in GRID["lr"]
        for hd in GRID["hidden_dim"]
        for ep in GRID["epochs"]
    ]
    print(f"\nGrid search: {len(combos)} combinations  (holdout val size: {n_val})")

    best_params = None
    best_acc    = -1.0

    for params in combos:
        model = _run_training(
            dataset    = train_subset,
            d_in       = dataset.X.shape[1],
            lr         = params["lr"],
            epochs     = params["epochs"],
            batch_size = BATCH_SIZE,
            hidden_dim = params["hidden_dim"],
            device     = device,
        )
        acc = _evaluate(model, val_X, val_y, device)
        print(f"  lr={params['lr']:.0e}  hidden={params['hidden_dim']:3d}  epochs={params['epochs']:3d}  → val acc: {acc:.3f}")

        if acc > best_acc:
            best_acc    = acc
            best_params = params

    print(f"\nBest params: {best_params}  (val acc: {best_acc:.3f})")
    return best_params


def train_mlp(split_id: int, model_name: str, grid_search: bool = False) -> None:
    """Train an MLP router for the given split/model pair.

    Args:
        split_id:    DB split id (selects the training partition).
        model_name:  HuggingFace model name (selects which model's features + results to use).
        grid_search: If True, run holdout grid search to find best hyperparams before final training.
    """
    save_path = mlp_path(split_id, model_name)
    if save_path.exists() and not grid_search:
        print(f"MLP weights already exist at {save_path}, skipping.")
        return

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = SAEDataset(split_id, model_name)

    if grid_search:
        best       = grid_search_mlp(dataset, device)
        lr         = best["lr"]
        hidden_dim = best["hidden_dim"]
        epochs     = best["epochs"]
    else:
        lr         = LR
        hidden_dim = HIDDEN_DIM
        epochs     = EPOCHS

    print(f"\nTraining final MLP (d_in={dataset.X.shape[1]}, hidden={hidden_dim}) for {epochs} epochs ...")
    model = _run_training(
        dataset    = dataset,
        d_in       = dataset.X.shape[1],
        lr         = lr,
        epochs     = epochs,
        batch_size = BATCH_SIZE,
        hidden_dim = hidden_dim,
        device     = device,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(save_path))
    print(f"\nSaved → {save_path}")
