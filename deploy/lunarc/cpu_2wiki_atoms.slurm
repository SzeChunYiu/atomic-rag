#!/bin/bash
#SBATCH --job-name=acsrag-2wiki-atoms
#SBATCH --account=lu2026-2-51
#SBATCH --partition=lu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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

INDEX_DIR=runs/2wiki/2wiki_1k_cs384_greedy_qwen7b/index_bundle
ATOMS_DIR=${INDEX_DIR}/atoms

if [ -f "${ATOMS_DIR}/atoms.jsonl" ]; then
    echo "atoms already exist; skipping"
    exit 0
fi

# Build 2wiki atom catalog (CPU; BGE-M3 will run on CPU since no GPU
# requested). ~30k atoms expected, ~1-2h on CPU.
python3 scripts/build_atoms.py "${INDEX_DIR}" --out_dir "${ATOMS_DIR}" --batch_size 32

# Also build BM25 for 2wiki atoms
python3 scripts/build_atom_bm25.py --atoms_dir "${ATOMS_DIR}" --out_dir "${ATOMS_DIR}"

echo "[2wiki-atoms] done"
