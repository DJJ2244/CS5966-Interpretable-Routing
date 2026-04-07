#!/bin/bash
#SBATCH --job-name=interp-routing
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-freecycle
#SBATCH --partition=granite-gpu
#SBATCH --gres=gpu:2

set -e

# ── Environment ───────────────────────────────────────────────────────────────
module load cuda
module load python/3.13.5

# Point HuggingFace cache to scratch so model weights persist between jobs
# Models cached to default HF location (~/.cache/huggingface)

cd $SLURM_SUBMIT_DIR
source .venv/bin/activate

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo "Shutting down servers..."
    python experiment.py down 2>/dev/null || true
}
trap cleanup EXIT

# ── Servers ───────────────────────────────────────────────────────────────────
# Blocks until both servers are healthy, then returns
mkdir -p logs/slurm
echo "Starting servers..."
python experiment.py up --weak-gpu 0 --strong-gpu 0 --detach

# ── Experiment ────────────────────────────────────────────────────────────────
echo "Running inference..."
python experiment.py inference --model all --split all

echo "Running routing..."
python experiment.py route --split all

echo "Running toughness..."
python experiment.py toughness --split all

# ── Activations ───────────────────────────────────────────────────────────────
# Stop servers first to free VRAM before TransformerLens loads full models
python experiment.py down
python _activations_extraction/extractActivation.py

echo "Done."
