"""
sae/train_sae.py - Train a Sparse Autoencoder (SAE) using SAELens on the middle
residual stream of the weak or strong Llama model.

Output paths follow the smart_file_util convention:
  sae_output/sae_<split_id>_<model_slug>_weights/   (SAELens checkpoint dir)
  sae_output/cfg_<split_id>_<model_slug>.json        (standalone config copy)
"""

import argparse
import glob
import os
import shutil

from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, StandardTrainingSAEConfig
from util.smart_file_util import sae_weights_path, sae_cfg_path, sae_checkpoint_path

os.environ["WANDB_MODE"] = "disabled"

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "humaneval_train")


def train_sae(model_name: str, hook_name: str, d_model: int, split_id: int) -> None:
    """Train a SAE for the given model.

    Args:
        model_name: HuggingFace model ID (e.g. "meta-llama/Llama-3.2-1B")
        hook_name:  TransformerLens hook point (e.g. "blocks.8.hook_resid_post")
        d_model:    Model hidden dimension
        split_id:   DB split id (used for output naming).
    """
    weights_dst = sae_weights_path(split_id, model_name)
    if weights_dst.exists():
        print(f"SAE weights already exist at {weights_dst}, skipping.")
        return

    sae_cfg = StandardTrainingSAEConfig(
        d_in           = d_model,
        d_sae          = d_model * 16,
        dtype          = "float32",
        device         = "cuda",
        l1_coefficient = 5e-2,
    )

    #point of optmization is maybe tokenize before training
    runner_cfg = LanguageModelSAERunnerConfig(
        sae                       = sae_cfg,
        model_name                = model_name,
        hook_name                 = hook_name,
        dataset_path              = DATA_PATH,
        is_dataset_tokenized      = False,
        dataset_trust_remote_code = False,
        prepend_bos               = True,
        context_size              = 512,
        n_batches_in_buffer       = 64,
        training_tokens           = 5_000_000,
        train_batch_size_tokens   = 4096,
        lr                        = 2e-4,
        lr_scheduler_name         = "cosineannealing",
        lr_warm_up_steps          = 500,
        device                    = "cuda",
        dtype                     = "float32",
        seed                      = 42,
        n_checkpoints             = 5,
        checkpoint_path           = str(sae_checkpoint_path(model_name)),
        save_final_checkpoint     = True,
    )

    print(f"\nTraining SAE on {model_name}")
    print(f"  Hookpoint : {hook_name}")
    print(f"  d_model   : {d_model}  →  d_sae: {d_model * 16}")

    runner = LanguageModelSAETrainingRunner(runner_cfg)
    runner.run()

    run_dirs = sorted(
        glob.glob(str(sae_checkpoint_path(model_name) / "*" / "final_*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not run_dirs:
        print(f"\nDone. SAE saved to {sae_checkpoint_path(model_name)}")
        return

    src     = run_dirs[0]
    cfg_dst = sae_cfg_path(split_id, model_name)

    shutil.copytree(src, weights_dst)
    shutil.copy2(os.path.join(src, "cfg.json"), cfg_dst)

    print(f"\nDone.")
    print(f"  Weights → {weights_dst}")
    print(f"  Config  → {cfg_dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="HuggingFace model ID")
    parser.add_argument("--hook-name",  required=True, help="TransformerLens hook point")
    parser.add_argument("--d-model",    type=int, required=True, help="Model hidden dimension")
    parser.add_argument("--split-id",   type=int, required=True)
    args = parser.parse_args()
    train_sae(model_name=args.model_name, hook_name=args.hook_name, d_model=args.d_model, split_id=args.split_id)
