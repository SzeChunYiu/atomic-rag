#!/bin/bash
#SBATCH --job-name=acsrag-d04d06-research
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
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

mkdir -p runs/realgen_d06_research

# Equivalence check: fast batched version with same params as existing
# baseline; confirms batching is lossless before trusting the rest.
out=runs/realgen_d06_research/equiv_check
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    echo "==== equivalence check (fast vs slow) ===="
    python3 scripts/atom_submodular_bench_fast.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
fi

# Research direction 1: doc-diversity bonus sweep (matroid soft constraint)
for ddb in 0.0 0.02 0.05 0.1 0.2; do
    out=runs/realgen_d06_research/ddb_${ddb}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    echo "==== doc_diversity_bonus=${ddb} ===="
    python3 scripts/atom_submodular_bench_fast.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --doc_diversity_bonus "${ddb}" \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
done

# Research direction 1b: hard cap atoms-per-doc (matroid hard constraint)
for cap in 0 2 3 5 8; do
    out=runs/realgen_d06_research/cap_${cap}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    echo "==== max_atoms_per_doc=${cap} ===="
    python3 scripts/atom_submodular_bench_fast.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --max_atoms_per_doc "${cap}" \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
done

# Research direction 3: adaptive top_k for ANY-intent queries
out=runs/realgen_d06_research/adaptive_topk
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    echo "==== adaptive_top_k ===="
    python3 scripts/atom_submodular_bench_fast.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --adaptive_top_k \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
fi

# Combined best (will be tuned post-hoc once individual sweeps land)
out=runs/realgen_d06_research/combined_best
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    echo "==== combined: ddb=0.1 + cap=3 + adaptive ===="
    python3 scripts/atom_submodular_bench_fast.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --doc_diversity_bonus 0.1 --max_atoms_per_doc 3 --adaptive_top_k \
        --max_new_tokens 32 --prompt_style citation --gen_batch_size 8
fi

# CoT variant of combined-best
out=runs/realgen_d06_research/combined_best_cot
if [ ! -f "${out}/metrics.json" ]; then
    rm -rf "${out}"
    echo "==== combined + CoT prompt ===="
    python3 scripts/atom_submodular_bench_fast.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --doc_diversity_bonus 0.1 --max_atoms_per_doc 3 --adaptive_top_k \
        --max_new_tokens 200 --prompt_style cot --gen_batch_size 4
fi

echo "[d04d06-research] done"
