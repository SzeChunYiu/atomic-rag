#!/bin/bash
#SBATCH --job-name=acsrag-prf
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
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
mkdir -p runs/realgen_prf

# PRF alpha sweep with score-greedy selector (isolate PRF effect)
for alpha in 0.5 0.7 0.9; do
    out=runs/realgen_prf/greedy_alpha_${alpha}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    python3 scripts/atom_prf_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --prf_alpha "${alpha}" --prf_top_m 10 --prf_intent_weight 0.2 \
        --selector greedy \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
done

# PRF top_m sweep
for tm in 5 10 20; do
    out=runs/realgen_prf/greedy_topm_${tm}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    python3 scripts/atom_prf_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --prf_alpha 0.7 --prf_top_m "${tm}" --prf_intent_weight 0.2 \
        --selector greedy \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
done

# Best PRF + MaxEnt + CoT (full stack candidate for paper)
out=runs/realgen_prf/best_full_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    python3 scripts/atom_prf_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --prf_alpha 0.7 --prf_top_m 10 --prf_intent_weight 0.2 \
        --selector maxent --beta 2.0 --lambda_f 0.5 --mu 0.001 \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

echo "[prf] done"
