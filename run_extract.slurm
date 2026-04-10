#!/bin/bash
#SBATCH --job-name=extract-act
#SBATCH --output=slurm-extract-%j.out
#SBATCH --error=slurm-extract-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:gh200:1

#SBATCH --partition=soc-gpu-gh200-class-grn
#SBATCH --account=cs6966

source /uufs/chpc.utah.edu/common/home/u0760641/extractActivations/CS5966-Interpretable-Routing/venv_sae/bin/activate
cd /uufs/chpc.utah.edu/common/home/u0760641/extractActivations/CS5966-Interpretable-Routing

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

python extractActivation.py
