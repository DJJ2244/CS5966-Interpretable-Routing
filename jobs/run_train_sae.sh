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

# Usage: sbatch run_train_sae.sh [model_key] [split_id] [model_id]
MODEL_KEY=${1:-weak}
SPLIT_ID=${2:-}
MODEL_ID=${3:-}

echo "Training SAE on model=$MODEL_KEY ..."
if [ -n "$SPLIT_ID" ] && [ -n "$MODEL_ID" ]; then
    python cli.py sae train --model "$MODEL_KEY" --split-id "$SPLIT_ID" --model-id "$MODEL_ID"
else
    python cli.py sae train --model "$MODEL_KEY"
fi

echo "Done."
