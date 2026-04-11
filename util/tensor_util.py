"""
tensor_util.py - Shared tensor I/O and feature alignment utilities.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from util import smart_file_util

if TYPE_CHECKING:
    from daos.model_task_result_dao import ModelTaskResult


def load_activations(split_id: int, model_name: str) -> dict:
    """Load dense activations. Returns {"task_ids": list[str], "activations": Tensor}."""
    path = smart_file_util.activations_path(split_id, model_name)
    return torch.load(str(path), weights_only=False)


def save_activations(split_id: int, model_name: str, task_ids: list[str], tensor: Tensor) -> None:
    """Save dense activations with their task_id index."""
    path = smart_file_util.activations_path(split_id, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task_ids": task_ids, "activations": tensor}, str(path))


def load_features(split_id: int, model_name: str) -> dict:
    """Load sparse SAE feature file. Returns {"task_ids": list[str], "features": Tensor}."""
    path = smart_file_util.sparse_features_path(split_id, model_name)
    return torch.load(str(path), weights_only=False)


def save_features(split_id: int, model_name: str, task_ids: list[str], tensor: Tensor) -> None:
    """Save sparse SAE features with their task_id index."""
    path = smart_file_util.sparse_features_path(split_id, model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task_ids": task_ids, "features": tensor}, str(path))


def align_features_with_labels(
    features_dict: dict,
    labels: list,
) -> tuple[Tensor, Tensor, list[str]]:
    """Align SAE feature vectors with pass/fail labels by task_id.

    Args:
        features_dict: Output of load_features() — {"task_ids": [...], "features": Tensor[N, d]}
        labels:        List of ModelTaskResult objects (must have .task_id and .passed attributes)

    Returns:
        (X, y, task_ids) where X is the aligned feature matrix, y is the label vector,
        and task_ids is the list of task ids in the same order.
    """
    label_map: dict[str, int] = {}
    for r in labels:
        if r.passed is not None:
            label_map[r.task_id] = 1 if r.passed else 0

    feature_index: dict[str, int] = {
        tid: i for i, tid in enumerate(features_dict["task_ids"])
    }
    features = features_dict["features"]

    xs, ys, ids = [], [], []
    missing = 0
    for task_id, label in label_map.items():
        idx = feature_index.get(task_id)
        if idx is None:
            missing += 1
            continue
        xs.append(features[idx])
        ys.append(label)
        ids.append(task_id)

    if missing:
        print(f"align_features_with_labels: {missing} labels had no matching feature vector.")

    X = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.float32)
    return X, y, ids
