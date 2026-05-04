#!/bin/bash
#SBATCH --job-name=acsrag-multiscale
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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
mkdir -p runs/realgen_multiscale

# Sweep top_k_doc (coarse-scale aggressiveness)
for kd in 10 30 50 100; do
    out=runs/realgen_multiscale/kd_${kd}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    python3 scripts/atom_multiscale_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --top_k_doc "${kd}" --top_k_chunk 80 --top_k_atom 50 \
        --entity_weight_chunk 0.1 --entity_weight_atom 0.1 \
        --beta 2.0 --lambda_f 0.5 --token_budget 1024 \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
done

# Sweep entity weight at chunk scale (RG kernel-flow strength)
for ec in 0.0 0.05 0.1 0.2 0.5; do
    out=runs/realgen_multiscale/ec_${ec}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    python3 scripts/atom_multiscale_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --top_k_doc 30 --top_k_chunk 80 --top_k_atom 50 \
        --entity_weight_chunk "${ec}" --entity_weight_atom 0.1 \
        --beta 2.0 --lambda_f 0.5 --token_budget 1024 \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
done

# Best multi-scale + CoT (full stack candidate)
out=runs/realgen_multiscale/best_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_multiscale_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --top_k_doc 30 --top_k_chunk 80 --top_k_atom 50 \
        --entity_weight_chunk 0.1 --entity_weight_atom 0.1 \
        --beta 2.0 --lambda_f 0.5 --token_budget 1024 \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

echo "[multiscale] done"
