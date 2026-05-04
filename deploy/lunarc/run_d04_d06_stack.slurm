#!/bin/bash
#SBATCH --job-name=acsrag-d04d06
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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

# Atom catalog must already exist (D04 build does it)
if [ ! -f "${ATOMS_DIR}/atoms.jsonl" ]; then
    echo "ERR atoms not built yet"; exit 1
fi

mkdir -p runs/realgen_d06
# Three configs: terse+submodular, cot+submodular, both with typed lambda
for prompt in citation cot; do
    out=runs/realgen_d06/d04_d06_${prompt}
    rm -rf "${out}"
    if [ "${prompt}" = "cot" ]; then
        max_tokens=200
    else
        max_tokens=32
    fi
    echo "==== running D04+D06 prompt=${prompt} ===="
    python3 scripts/atom_submodular_bench.py \
        --atoms_dir "${ATOMS_DIR}" \
        --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl \
        --out_dir "${out}" \
        --lambda_type 0.05 \
        --top_k 50 \
        --token_budget 1024 \
        --max_new_tokens "${max_tokens}" \
        --prompt_style "${prompt}"
done
echo "[d04d06] done"
