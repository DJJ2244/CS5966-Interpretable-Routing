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

SPLIT_ID=${1:-1}
WEAK_MODEL=${2:-""}
STRONG_MODEL=${3:-""}

echo "Calculating RouteLLM threshold for split $SPLIT_ID ..."
python cli.py route-llm calculate-threshold \
    --split-id "$SPLIT_ID" \
    --weak-model "$WEAK_MODEL" \
    --strong-model "$STRONG_MODEL"

echo "Done."
