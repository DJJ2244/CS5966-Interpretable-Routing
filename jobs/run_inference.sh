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

WEAK_MODEL="meta-llama/Llama-3.2-1B"
STRONG_MODEL="meta-llama/Meta-Llama-3-8B"
SPLIT_ID=${1:-1}

echo "Starting servers..."
python cli.py server up \
    --model "$WEAK_MODEL:0" \
    --model "$STRONG_MODEL:1" \
    --detach

echo "Running inference (weak): split=$SPLIT_ID"
python cli.py inference run --model-name "$WEAK_MODEL" --split-id "$SPLIT_ID" &
WEAK_PID=$!

echo "Running inference (strong): split=$SPLIT_ID"
python cli.py inference run --model-name "$STRONG_MODEL" --split-id "$SPLIT_ID" &
STRONG_PID=$!

wait $WEAK_PID
wait $STRONG_PID

echo "Done."
