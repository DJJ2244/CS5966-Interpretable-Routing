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

echo "Starting servers..."
python cli.py server up --weak-gpu 0 --strong-gpu 1 --detach

echo "Running inference (weak + strong, all splits)..."
python cli.py inference run --model all --split-id 1

echo "Done."
