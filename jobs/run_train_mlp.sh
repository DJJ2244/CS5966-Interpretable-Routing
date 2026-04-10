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
source .venv/bin/activate

export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

# Usage: sbatch run_train_mlp.sh [split_id] [model_id]
SPLIT_ID=${1:-1}
MODEL_ID=${2:-1}

echo "Training MLP router: split_id=$SPLIT_ID model_id=$MODEL_ID"
python cli.py mlp train --split-id "$SPLIT_ID" --model-id "$MODEL_ID"

echo "Done."
