#!/bin/bash
#SBATCH --job-name=train-sae
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --account=cs6966
#SBATCH --qos=soc-gpu-class-grn

set -e

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

export TRANSFORMERS_OFFLINE=1
unset HF_DATASETS_OFFLINE
export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

# Usage: sbatch run_train_sae.sh [split_id]
# MODEL_NAME="meta-llama/Meta-Llama-3-8B"; HOOK_NAME="blocks.16.hook_resid_post"; D_MODEL=4096  # strong
MODEL_NAME="meta-llama/Llama-3.2-1B"; HOOK_NAME="blocks.8.hook_resid_post"; D_MODEL=2048        # weak
SPLIT_ID=${1:-1}

echo "Training SAE: model=$MODEL_NAME hook=$HOOK_NAME d_model=$D_MODEL split=$SPLIT_ID"
python cli.py sae train \
    --model-name "$MODEL_NAME" \
    --hook-name  "$HOOK_NAME" \
    --d-model    "$D_MODEL" \
    --split-id   "$SPLIT_ID"

echo "Done."
