"""
sae/train_sae.py - Train a Sparse Autoencoder (SAE) using SAELens on the middle
residual stream of the weak or strong Llama model.

Output paths follow the smart_file_util convention:
  sae_output/sae_<split_id>_<model_id>_weights/   (SAELens checkpoint dir)
  sae_output/cfg_<split_id>_<model_id>.json        (standalone config copy)
"""

import argparse
import glob
import os
import shutil

from sae_lens import LanguageModelSAERunnerConfig, LanguageModelSAETrainingRunner, StandardTrainingSAEConfig

os.environ["WANDB_MODE"] = "disabled"

MODELS = {
    "weak": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "hook_name":  "blocks.8.hook_resid_post",   # middle of 16 layers
        "d_model":    2048,
    },
    "strong": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "hook_name":  "blocks.16.hook_resid_post",  # middle of 32 layers
        "d_model":    4096,
    },
}

DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "humaneval_train")
OUTPUT_DIR = "sae_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_sae(model_key: str, split_id: int = None, model_id: int = None) -> None:
    """Train a SAE for the given model key.

    Args:
        model_key: "weak" or "strong"
        split_id:  DB split id (used for output naming). If None, uses model_key string.
        model_id:  DB model id (used for output naming). If None, uses model_key string.
    """
    from util.smart_file_util import sae_weights_path, sae_cfg_path

    cfg_args = MODELS[model_key]
    d_model  = cfg_args["d_model"]

    sae_cfg = StandardTrainingSAEConfig(
        d_in           = d_model,
        d_sae          = d_model * 16,
        dtype          = "float32",
        device         = "cuda",
        l1_coefficient = 5e-2,
    )

    runner_cfg = LanguageModelSAERunnerConfig(
        sae                       = sae_cfg,
        model_name                = cfg_args["model_name"],
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
        checkpoint_path           = os.path.join(OUTPUT_DIR, model_key),
        save_final_checkpoint     = True,
    )

    print(f"\nTraining SAE on {model_key} model ({cfg_args['model_name']})")
    print(f"  Hookpoint : {cfg_args['hook_name']}")
    print(f"  d_model   : {d_model}  →  d_sae: {d_model * 16}")

    runner = LanguageModelSAETrainingRunner(runner_cfg)
    runner.run()

    # Copy to canonical named paths
    run_dirs = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, model_key, "*", "final_*")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not run_dirs:
        print(f"\nDone. SAE saved to {os.path.join(OUTPUT_DIR, model_key)}")
        return

    src = run_dirs[0]

    if split_id is not None and model_id is not None:
        weights_dst = str(sae_weights_path(split_id, model_id))
        cfg_dst     = str(sae_cfg_path(split_id, model_id))
    else:
        # Fallback: use string-keyed names for backward compatibility
        weights_dst = os.path.join(OUTPUT_DIR, f"sae_train_{model_key}_weights")
        cfg_dst     = os.path.join(OUTPUT_DIR, f"cfg_train_{model_key}.json")

    if os.path.exists(weights_dst):
        shutil.rmtree(weights_dst)
    shutil.copytree(src, weights_dst)
    shutil.copy2(os.path.join(src, "cfg.json"), cfg_dst)

    print(f"\nDone.")
    print(f"  Weights → {weights_dst}")
    print(f"  Config  → {cfg_dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["weak", "strong"], required=True)
    parser.add_argument("--split-id", type=int, default=None)
    parser.add_argument("--model-id", type=int, default=None)
    args = parser.parse_args()
    train_sae(args.model, split_id=args.split_id, model_id=args.model_id)
