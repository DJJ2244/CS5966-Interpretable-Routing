#!/bin/bash
#SBATCH --job-name=interp-init-db
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --account=cs6966
#SBATCH --qos=soc-gpu-class-grn
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --gres=gpu:1

set -e

module load python/3.13.5

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate
mkdir -p logs/slurm

export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

echo "Initializing database..."
echo "y" | python cli.py db init

echo "Done."
