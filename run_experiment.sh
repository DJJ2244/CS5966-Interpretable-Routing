#!/bin/bash
#SBATCH --job-name=interp-routing
#SBATCH --account=cs6966
#SBATCH --qos=granite-gpu-guest
#SBATCH --partition=granite-gpu-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100:1        # needs 20+ GB VRAM for both models
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err

set -e

# ── Environment ───────────────────────────────────────────────────────────────
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
