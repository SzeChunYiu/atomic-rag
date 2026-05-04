#!/bin/bash
#SBATCH --job-name=acsrag-d04-crowding
#SBATCH --account=lu2026-2-51
#SBATCH --partition=lu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase-1 evidence-crowding sweep (D04 refinement).
# Pure CPU: HashEmbedder + oracle metric, no LLM in loop.
# 6 (n_distractors) x 3 (similarity) x 4 (token_budget) = 72 cells x 50 queries
# x 2 systems = 7,200 oracle evaluations. Fits comfortably on lu48.
#
# Stage only — submit with `sbatch deploy/lunarc/run_d04_crowding.slurm`
# from the project dir on LUNARC.

set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"

module purge
module load GCC/12.3.0 Python/3.11.3
set +u
source "${VENV}/bin/activate"
set -u

cd "${PROJECT_DIR}"

OUT_DIR=runs/d04_crowding/phase1_$(date +%Y%m%d_%H%M%S)
mkdir -p "${OUT_DIR}"

echo "[d04-crowding] node=$(hostname -s) out=${OUT_DIR}"

python3 -m astro_cs_rag.cli.run_crowding_sweep \
    --grid phase1 \
    --n_queries 50 \
    --systems atom_dense chunk_dense \
    --seed_base 1000 \
    --out_dir "${OUT_DIR}"

echo "[d04-crowding] done; results under ${OUT_DIR}"
