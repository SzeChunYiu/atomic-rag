#!/bin/bash
#SBATCH --job-name=acsrag-d04d06-sweep
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
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

if [ ! -f "${ATOMS_DIR}/atoms.jsonl" ]; then
    echo "ERR atoms not built yet"; exit 1
fi

mkdir -p runs/realgen_d06_sweep

# Sweep 1: lambda_type (typed-bonus weight)
for lt in 0.0 0.025 0.05 0.1 0.2 0.5; do
    out=runs/realgen_d06_sweep/lambda_${lt}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    echo "==== lambda_type=${lt} ===="
    python3 scripts/atom_submodular_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type "${lt}" --top_k 50 --token_budget 1024 \
        --max_new_tokens 32 --prompt_style citation
done

# Sweep 2: token_budget (saturation curve)
for tb in 512 1024 1536 2048; do
    out=runs/realgen_d06_sweep/budget_${tb}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    echo "==== token_budget=${tb} ===="
    python3 scripts/atom_submodular_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget "${tb}" \
        --max_new_tokens 32 --prompt_style citation
done

# Sweep 3: top_k (candidate-pool size)
for tk in 25 50 100 200; do
    out=runs/realgen_d06_sweep/topk_${tk}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    echo "==== top_k=${tk} ===="
    python3 scripts/atom_submodular_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k "${tk}" --token_budget 1024 \
        --max_new_tokens 32 --prompt_style citation
done

# Sweep 4: score_floor (drop low-similarity atoms before selection)
for sf in 0.0 0.1 0.2 0.3; do
    out=runs/realgen_d06_sweep/floor_${sf}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    echo "==== score_floor=${sf} ===="
    python3 scripts/atom_submodular_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --score_floor "${sf}" --max_new_tokens 32 --prompt_style citation
done

# Sweep 5: score_bonus (score-vs-coverage tradeoff in submodular gain)
for sb in 0.01 0.05 0.1 0.2 0.5; do
    out=runs/realgen_d06_sweep/bonus_${sb}
    [ -f "${out}/metrics.json" ] && { echo "skip ${out}"; continue; }
    rm -rf "${out}"
    echo "==== score_bonus=${sb} ===="
    python3 scripts/atom_submodular_bench.py \
        --atoms_dir "${ATOMS_DIR}" --index_dir "${INDEX_DIR}" \
        --queries data/hotpotqa_1k/queries.jsonl --out_dir "${out}" \
        --lambda_type 0.05 --top_k 50 --token_budget 1024 \
        --score_bonus "${sb}" --max_new_tokens 32 --prompt_style citation
done

echo "[d04d06-sweep] done"
