#!/bin/bash
#SBATCH --job-name=interp-test
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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

# Usage: sbatch run_test_model_code.sh <results_path> <model_name>
# Example: sbatch run_test_model_code.sh route_llm_results/results_weak.jsonl weak
RESULTS_PATH=${1:-route_llm_results/results_weak.jsonl}
MODEL=${2:-weak}

echo "Running tests on $RESULTS_PATH for model '$MODEL'..."
python cli.py test run --results "$RESULTS_PATH" --model "$MODEL"

echo "Done."
