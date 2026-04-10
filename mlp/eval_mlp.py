"""
mlp/eval_mlp.py - Evaluate the trained MLP router on the test split.

Labels are read from the model_task_result DB table.
Features and weights are loaded from the canonical paths.
"""

import torch

from mlp.model import MLP, HIDDEN_DIM
from util import tensor_util
from util.smart_file_util import mlp_path


def evaluate_mlp(split_id: int, model_id: int, conn) -> None:
    """Evaluate the MLP router for the given split/model pair.

    Args:
        split_id: DB split id (selects the test partition).
        model_id: DB model id.
        conn:     Open DB connection.
    """
    from daos import model_task_result_dao

    device = "cuda" if torch.cuda.is_available() else "cpu"

    features_dict = tensor_util.load_features(split_id, model_id)
    labels        = model_task_result_dao.get_all_for_model_split(
        conn, model_id, split_id, is_test=True
    )
    X, y, _ = tensor_util.align_features_with_labels(features_dict, labels)
    X = X.to(device)

    print(f"Test set: {len(y)} samples")
    print(f"  Positive (weak passed): {int(y.sum())}")
    print(f"  Negative (weak failed): {int((y == 0).sum())}")

    weights = mlp_path(split_id, model_id)
    model = MLP(d_in=X.shape[1], hidden=HIDDEN_DIM).to(device)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(X)
        preds  = (logits > 0).float().cpu()

    y = y.cpu()

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
