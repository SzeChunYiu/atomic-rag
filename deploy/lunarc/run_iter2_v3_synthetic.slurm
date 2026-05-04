#!/bin/bash
#SBATCH --job-name=acsrag-iter2-v3
#SBATCH --account=lu2026-2-51
#SBATCH --partition=lu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"

module purge
module load GCC/12.3.0 Python/3.11.3
set +u
source "${VENV}/bin/activate"
set -u

cd "${PROJECT_DIR}"
mkdir -p logs

echo "[acsrag] node = $(hostname); cpus = $(nproc)"
echo "[acsrag] selector = anti_kt v3 (n_jets=-2 atomic-unit greedy)"

PYTHONPATH=. python scripts/synthetic_irc_v3_compare.py \
  --n-topics 120 --n-distractors 240 --n-seeds 5 \
  --chunk-min 50 --chunk-max 300 --chunk-steps 20 \
  --out runs/synthetic_irc_iter2_v3

echo "[acsrag] iter2 v3 done"
