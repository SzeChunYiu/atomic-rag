#!/bin/bash
#SBATCH --job-name=acsrag-rebuild-atoms
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"
export HF_HOME=/projects/hep/fs10/shared/nnbar/billy/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
module purge
module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
set +u
source "${VENV}/bin/activate"
set -u
cd "${PROJECT_DIR}"

INDEX_DIR=runs/realgen_terse/hotpotqa_1k_cs384_greedy_qwen7b/index_bundle
ATOMS_DIR=${INDEX_DIR}/atoms

echo "[rebuild-atoms] node = $(hostname -s)"
echo "[rebuild-atoms] Force-rebuilding atoms with claim_type_conf field"

# Force rebuild (remove existing atoms to pick up new claim_type_conf field)
rm -rf "${ATOMS_DIR}"
mkdir -p "${ATOMS_DIR}"
python3 scripts/build_atoms.py "${INDEX_DIR}" --out_dir "${ATOMS_DIR}" --batch_size 64

echo "[rebuild-atoms] Done. Atom type distribution:"
cat "${ATOMS_DIR}/atom_meta.json"
