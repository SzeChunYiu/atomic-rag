#!/bin/bash
#SBATCH --job-name=acsrag-hybrid-recall
#SBATCH --account=lu2026-2-51
#SBATCH --partition=lu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"
export HF_HOME=/projects/hep/fs10/shared/nnbar/billy/hf_cache
module purge
module load GCC/12.3.0 Python/3.11.3
set +u
source "${VENV}/bin/activate"
set -u
cd "${PROJECT_DIR}"

INDEX_DIR=runs/realgen_terse/hotpotqa_1k_cs384_greedy_qwen7b/index_bundle
ATOMS_DIR=${INDEX_DIR}/atoms

mkdir -p runs/realgen_hybrid_recall
python3 scripts/hybrid_recall_bench.py \
    --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
    --queries data/hotpotqa_1k/queries.jsonl \
    --out_dir runs/realgen_hybrid_recall

echo "[hybrid-recall] done"
