#!/bin/bash
#SBATCH --job-name=interp-test
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --account=cs6966
#SBATCH --qos=granite-guest
#SBATCH --partition=granite-guest

set -e

module load python/3.13.5

cd $SLURM_SUBMIT_DIR
source .env
source .venv/bin/activate
mkdir -p logs/slurm

export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

# Usage: sbatch run_test_model_code.sh [split_id]
MODEL_NAME=${MODEL_NAME:-$WEAK_MODEL}
SPLIT_ID=${1:-1}

echo "Running tests: model=$MODEL_NAME split=$SPLIT_ID"
python cli.py test run --model-name "$MODEL_NAME" --split-id "$SPLIT_ID"

echo "Done."
