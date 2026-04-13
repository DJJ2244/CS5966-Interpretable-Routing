#!/bin/bash
#SBATCH --job-name=interp-inference
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --account=cs6966
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --gres=gpu:2

set -e

module load python/3.13.5

cd $SLURM_SUBMIT_DIR
source .env
source .venv/bin/activate
mkdir -p logs/slurm

export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

cleanup() {
    echo "Shutting down servers..."
    python cli.py server down 2>/dev/null || true
}
trap cleanup EXIT

SPLIT_ID=${1:-1}

echo "Starting servers..."
python cli.py server up \
    --model "$WEAK_MODEL:0" \
    --model "$STRONG_MODEL:1" \
    --detach

echo "Running inference (weak, train): split=$SPLIT_ID"
python cli.py inference run --model-name "$WEAK_MODEL" --split-id "$SPLIT_ID" &
WEAK_TRAIN_PID=$!

echo "Running inference (strong, train): split=$SPLIT_ID"
python cli.py inference run --model-name "$STRONG_MODEL" --split-id "$SPLIT_ID" &
STRONG_TRAIN_PID=$!

wait $WEAK_TRAIN_PID
wait $STRONG_TRAIN_PID

echo "Running inference (weak, test): split=$SPLIT_ID"
python cli.py inference run --model-name "$WEAK_MODEL" --split-id "$SPLIT_ID" --test &
WEAK_TEST_PID=$!

echo "Running inference (strong, test): split=$SPLIT_ID"
python cli.py inference run --model-name "$STRONG_MODEL" --split-id "$SPLIT_ID" --test &
STRONG_TEST_PID=$!

wait $WEAK_TEST_PID
wait $STRONG_TEST_PID

echo "Done."
