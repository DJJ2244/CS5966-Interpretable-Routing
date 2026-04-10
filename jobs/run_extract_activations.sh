#!/bin/bash
#SBATCH --job-name=interp-extract
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
export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

# Usage: sbatch run_extract_activations.sh [model_key] [split_id] [model_id]
MODEL_KEY=${1:-weak}
SPLIT_ID=${2:-1}
MODEL_ID=${3:-1}

echo "Extracting activations: model=$MODEL_KEY split=$SPLIT_ID model_id=$MODEL_ID"
python cli.py sae extract \
    --model-key "$MODEL_KEY" \
    --split-id "$SPLIT_ID" \
    --model-id "$MODEL_ID"

echo "Done."
