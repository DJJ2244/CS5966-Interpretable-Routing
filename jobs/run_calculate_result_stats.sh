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

# Usage: sbatch run_calculate_result_stats.sh [split_id]
SPLIT_ID=${1:-1}

echo "Calculating result stats: split_id=$SPLIT_ID"
python cli.py stats calculate --split-id "$SPLIT_ID"

echo "Done."
