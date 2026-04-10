#!/bin/bash
#SBATCH --job-name=llm-demo
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtxpr6000bl:1

#SBATCH --partition=soc-gpu-class-grn
#SBATCH --account=cs6966
#SBATCH --qos=soc-gpu-class-grn

source /uufs/chpc.utah.edu/common/home/u0760641/extractActivations/CS5966-Interpretable-Routing/venv_sae/bin/activate
cd /uufs/chpc.utah.edu/common/home/u0760641/extractActivations/CS5966-Interpretable-Routing

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

python extractActivation.py