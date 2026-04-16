#!/bin/bash
#SBATCH --job-name=cluster-viz
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --gres=gpu:rtxpr6000bl:1
#SBATCH --partition=soc-gpu-class-grn
#SBATCH --account=cs6966
#SBATCH --qos=soc-gpu-class-grn

set -e

cd $SLURM_SUBMIT_DIR
source .env
source .venv/bin/activate

export HF_HOME=/scratch/general/vast/$USER/.cache/huggingface

# Usage: sbatch jobs/run_cluster_visualizer.sh [split_id] [k]
SPLIT_ID=${1:-1}
K=${2:-12}
LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}

echo "Cluster visualizer: model=$WEAK_MODEL split=$SPLIT_ID k=$K llm=$LLM_MODEL"
python scripts/cluster_visualizer.py \
    --split-id      "$SPLIT_ID" \
    --model-name    "$WEAK_MODEL" \
    --k             "$K" \
    --llm-model     "$LLM_MODEL"

echo "Done. Output: visuals/cluster_explorer.html"
