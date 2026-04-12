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

# Usage: sbatch run_extract_activations.sh [split_id]
# MODEL_NAME="meta-llama/Meta-Llama-3-8B"  # strong
MODEL_NAME="meta-llama/Llama-3.2-1B"        # weak
SPLIT_ID=${1:-1}

echo "Extracting activations: model=$MODEL_NAME split=$SPLIT_ID"
python cli.py sae extract \
    --model-name "$MODEL_NAME" \
    --split-id   "$SPLIT_ID"

echo "Done."
