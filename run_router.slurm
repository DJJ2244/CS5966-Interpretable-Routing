#!/bin/bash
#SBATCH --job-name=router-run
#SBATCH --output=slurm-router-%j.out
#SBATCH --error=slurm-router-%j.err
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

export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

# Usage: sbatch run_router.slurm --prompt "your prompt here"
#        sbatch run_router.slurm --batch data/humaneval_xl_english_test.jsonl
python router.py "$@"