#!/bin/bash
#SBATCH --job-name=interp-inference
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest
#SBATCH --partition=granite-gpu-guest
#SBATCH --gres=gpu:2

set -e

module load cuda
module load python/3.13.5

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
mkdir -p logs/slurm

cleanup() {
    echo "Shutting down servers..."
    python cli.py server down 2>/dev/null || true
}
trap cleanup EXIT

# MODEL_NAME_STRONG="meta-llama/Meta-Llama-3-8B"  # strong
WEAK_MODEL="meta-llama/Llama-3.2-1B"
STRONG_MODEL="meta-llama/Meta-Llama-3-8B"
SPLIT_ID=${1:-1}

echo "Starting servers..."
python cli.py server up --weak-gpu 0 --strong-gpu 1 --detach

echo "Running inference (weak + strong): split=$SPLIT_ID"
python cli.py inference run \
    --weak-model   "$WEAK_MODEL" \
    --strong-model "$STRONG_MODEL" \
    --model all \
    --split-id "$SPLIT_ID"

echo "Done."
