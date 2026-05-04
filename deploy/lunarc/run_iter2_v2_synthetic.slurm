#!/bin/bash
#SBATCH --job-name=acsrag-iter2-v2
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

echo "[acsrag] node     = $(hostname)"
echo "[acsrag] cpus     = $(nproc)"
echo "[acsrag] selector = anti_kt v2 (n_jets=-1, member SNR sort) — editable install verified"

PYTHONPATH=. python scripts/synthetic_irc_iter2.py \
  --n-topics 120 \
  --n-distractors 240 \
  --n-seeds 5 \
  --chunk-steps 20 \
  --chunk-min 50 \
  --chunk-max 300 \
  --selectors greedy anti_kt mmr \
  --out runs/synthetic_irc_iter2_v2

echo "[acsrag] iter2 v2 done"
