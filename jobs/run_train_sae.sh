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
source .env
source .venv/bin/activate

export TRANSFORMERS_OFFLINE=1
unset HF_DATASETS_OFFLINE
export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface
export WANDB_MODE=disabled
export WANDB_DIR=/scratch/general/vast/$USER/wandb
mkdir -p $WANDB_DIR

# Usage: sbatch run_train_sae.sh [split_id]
MODEL_NAME=${MODEL_NAME:-$WEAK_MODEL}
HOOK_NAME=${HOOK_NAME:-$WEAK_HOOK_NAME}
D_MODEL=${D_MODEL:-$WEAK_D_MODEL}
SPLIT_ID=${1:-1}

echo "Training SAE: model=$MODEL_NAME hook=$HOOK_NAME d_model=$D_MODEL split=$SPLIT_ID"
python cli.py sae train \
    --model-name "$MODEL_NAME" \
    --hook-name  "$HOOK_NAME" \
    --d-model    "$D_MODEL" \
    --split-id   "$SPLIT_ID"

echo "Done."
