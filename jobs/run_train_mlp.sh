#!/bin/bash
#SBATCH --job-name=train-mlp
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
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

export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

# Usage: sbatch run_train_mlp.sh [split_id]
MODEL_NAME=${MODEL_NAME:-$WEAK_MODEL}
SPLIT_ID=${1:-1}

echo "Training MLP router: model=$MODEL_NAME split=$SPLIT_ID"
python cli.py mlp train \
    --model-name "$MODEL_NAME" \
    --split-id   "$SPLIT_ID"

echo "Done."
