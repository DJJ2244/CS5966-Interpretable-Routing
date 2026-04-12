"""
sae/extract_spv.py - Extract sparse feature vectors (SPVs) from model activations.

Pipeline:
  1. extract_activations() — run transformer_lens to get middle-layer residual stream
  2. extract_sparse_features() — encode dense activations through SAE → sparse vectors
  3. run() — chains both steps

Output paths follow the smart_file_util convention:
  activations/activations_<split_id>_<model_slug>.pt         (dense)
  activations/activations_<split_id>_<model_slug>_sparse.pt  (sparse)
"""

import torch
from pathlib import Path

from util.model_util import MODELS


def extract_activations(
    model_name: str,
    split_id: int,
    is_test: bool,
) -> Path:
    """Extract middle-layer residual stream activations for all tasks in a split.

    Args:
        model_name: HuggingFace model identifier.
        split_id:   DB split id (used for output path naming and task loading).
        is_test:    Whether to extract from the test partition.

    Returns:
        Path to the saved activations .pt file.
    """
    from transformer_lens import HookedTransformer
    from daos import tasks_dao
    from util.smart_file_util import activations_path
    from util import tensor_util

    out_path = activations_path(split_id, model_name)
    if out_path.exists():
        print(f"Activations already exist at {out_path}, skipping.")
        return out_path

    tasks = tasks_dao.get_all_for_split(split_id, is_test=is_test)

    problems = [(t.id, t.prompt) for t in tasks]
    print(f"Loaded {len(problems)} tasks from split {split_id} ({'test' if is_test else 'train'})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {model_name}")
    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        dtype=torch.float16,
        device=device,
    )
    model.eval()

    num_layers   = model.cfg.n_layers
    middle_layer = num_layers // 2
    print(f"  {num_layers} layers — extracting layer {middle_layer}")

    class _EarlyStop(Exception):
        pass

    all_vectors = []
    task_ids    = []

    for i, (task_id, prompt) in enumerate(problems):
        try:
            tokens   = model.to_tokens(prompt, truncate=True)
            captured = {}

            def _hook(value, hook=None):  # noqa: ARG001
                captured["act"] = value
                raise _EarlyStop()

            with torch.no_grad():
                try:
                    model.run_with_hooks(
                        tokens,
                        fwd_hooks=[(f"blocks.{middle_layer}.hook_resid_post", _hook)],
                    )
                except _EarlyStop:
                    pass

            vector = captured["act"].mean(dim=1).squeeze(0).float().cpu()
            all_vectors.append(vector)
            task_ids.append(task_id)
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Failed on {task_id}: {e}")
            all_vectors.append(torch.zeros(model.cfg.d_model))
            task_ids.append(task_id)

    activation_matrix = torch.stack(all_vectors)
    del model
    torch.cuda.empty_cache()

    out_path = activations_path(split_id, model_name)
    tensor_util.save_activations(split_id, model_name, task_ids, activation_matrix)
    print(f"Saved {activation_matrix.shape} → {out_path}")
    return out_path


def extract_sparse_features(
    model_name: str,
    split_id: int,
    sae_path: str = None,
) -> Path:
    """Encode dense activations through a trained SAE to produce sparse feature vectors.

    Args:
        model_name: HuggingFace model identifier.
        split_id:   DB split id.
        sae_path:   Path to SAE checkpoint directory. Defaults to the canonical
                    smart_file_util.sae_weights_path(split_id, model_name).

    Returns:
        Path to the saved sparse features .pt file.
    """
    from sae_lens import SAE
    from util.smart_file_util import sae_weights_path, sparse_features_path
    from util import tensor_util

    out_path = sparse_features_path(split_id, model_name)
    if out_path.exists():
        print(f"Sparse features already exist at {out_path}, skipping.")
        return out_path

    if sae_path is None:
        sae_path = str(sae_weights_path(split_id, model_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading SAE from {sae_path} ...")
    sae = SAE.load_from_disk(sae_path, device=device)
    sae.eval()
    print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")

    data        = tensor_util.load_activations(split_id, model_name)
    activations = data["activations"].to(device)
    task_ids    = data["task_ids"]
    print(f"  {activations.shape[0]} problems, d_model={activations.shape[1]}")

    print("Encoding through SAE ...")
    with torch.no_grad():
        feature_acts = sae.encode(activations)

    l0   = (feature_acts > 0).float().sum(dim=1).mean()
    dead = (feature_acts > 0).any(dim=0).logical_not().sum().item()
    print(f"  Feature matrix: {feature_acts.shape}")
    print(f"  Avg L0: {l0:.1f}   Dead features: {dead}/{feature_acts.shape[1]}")

    tensor_util.save_features(split_id, model_name, task_ids, feature_acts.cpu())
    out_path = sparse_features_path(split_id, model_name)
    print(f"Saved → {out_path}")
    return out_path


def run(
    model_key: str,
    split_id: int,
    is_test: bool,
    sae_path: str = None,
) -> None:
    """Full pipeline: extract activations then encode through SAE."""
    model_name = MODELS[model_key]
    extract_activations(model_name=model_name, split_id=split_id, is_test=is_test)
    extract_sparse_features(model_name=model_name, split_id=split_id, sae_path=sae_path)
