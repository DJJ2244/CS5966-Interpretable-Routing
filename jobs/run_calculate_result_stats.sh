#!/bin/bash
#SBATCH --job-name=interp-stats
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

export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

echo "Calculating result stats..."
python cli.py stats calculate \
    --sae-router routing_decisions.jsonl \
    --route-llm  route_llm_decisions.jsonl \
    --output     visuals/routing_stats.png

echo "Done."
