#!/bin/bash
#SBATCH --job-name=interp-threshold
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest
#SBATCH --partition=granite-gpu-guest

set -e

module load python/3.13.5

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
mkdir -p logs/slurm

TOUGHNESS_PATH=${1:-route_llm_results/toughness.jsonl}
TARGET_STRONG_RATE=${2:-0.5}

echo "Calculating RouteLLM threshold from $TOUGHNESS_PATH ..."
python cli.py route-llm calculate-threshold \
    --toughness-path "$TOUGHNESS_PATH" \
    --target-strong-rate "$TARGET_STRONG_RATE"

echo "Done."
