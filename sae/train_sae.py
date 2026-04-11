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
from util.model_util import WEAK_MODEL, STRONG_MODEL
from util.smart_file_util import sae_weights_path, sae_cfg_path, sae_checkpoint_path

os.environ["WANDB_MODE"] = "disabled"

SAE_CONFIGS = {
    "weak": {
        "model_name": WEAK_MODEL,
        "hook_name":  "blocks.8.hook_resid_post",   # middle of 16 layers
        "d_model":    2048,
    },
    "strong": {
        "model_name": STRONG_MODEL,
        "hook_name":  "blocks.16.hook_resid_post",  # middle of 32 layers
        "d_model":    4096,
    },
}

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "humaneval_train")


def train_sae(model_key: str, split_id: int) -> None:
    """Train a SAE for the given model key.

    Args:
        model_key: "weak" or "strong"
        split_id:  DB split id (used for output naming).
    """
    cfg_args   = SAE_CONFIGS[model_key]
    model_name = cfg_args["model_name"]
    d_model    = cfg_args["d_model"]

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
        hook_name                 = cfg_args["hook_name"],
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
        checkpoint_path           = str(sae_checkpoint_path(model_key)),
        save_final_checkpoint     = True,
    )

    print(f"\nTraining SAE on {model_key} model ({model_name})")
    print(f"  Hookpoint : {cfg_args['hook_name']}")
    print(f"  d_model   : {d_model}  →  d_sae: {d_model * 16}")

    runner = LanguageModelSAETrainingRunner(runner_cfg)
    runner.run()

    run_dirs = sorted(
        glob.glob(str(sae_checkpoint_path(model_key) / "*" / "final_*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not run_dirs:
        print(f"\nDone. SAE saved to {sae_checkpoint_path(model_key)}")
        return

    src         = run_dirs[0]
    weights_dst = sae_weights_path(split_id, model_name)
    cfg_dst     = sae_cfg_path(split_id, model_name)

    if os.path.exists(weights_dst):
        shutil.rmtree(weights_dst)
    shutil.copytree(src, weights_dst)
    shutil.copy2(os.path.join(src, "cfg.json"), cfg_dst)

    print(f"\nDone.")
    print(f"  Weights → {weights_dst}")
    print(f"  Config  → {cfg_dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["weak", "strong"], required=True)
    parser.add_argument("--split-id", type=int, required=True)
    args = parser.parse_args()
    train_sae(args.model, split_id=args.split_id)
