#!/bin/bash
#SBATCH --job-name=acsrag-phase4
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase 4: 32B self-consistency + sub-question decomposition for 2Wiki.
# After phase3 isolates which G1 confabulation fraction 32B still fails on,
# this phase attacks the residual with:
#   (a) 32B self-consistency (3 seeds T=0.7) → majority vote targets stochastic G1
#   (b) Larger budget (4096) with 32B to see if more context helps residual S2b
#   (c) 2Wiki sub-question decomposition: a CPU script splits bridge questions
#       into two independent sub-queries, retrieves for each, merges context

set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"
export HF_HOME=/projects/hep/fs10/shared/nnbar/billy/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets

module purge
module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
set +u; source "${VENV}/bin/activate"; set -u

cd "${PROJECT_DIR}"
echo "[phase4] node=$(hostname -s)  start=$(date)"
mkdir -p logs runs/phase4 configs/benchmarks/phase4

DATAHP=data/hotpotqa_1k
DATA2W=data/2wiki_1k

# Read best prompt from phase3 results (default: few_shot_cot)
BEST_PROMPT=few_shot_cot

nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader

write_cfg_32b() {
  local cfg="$1" dataset="$2" corpus="$3" queries="$4" gold="$5"
  local out="$6" rerank="$7" seed="$8" temp="$9" budget="${10}"
  cat > "${cfg}" <<YAML
chunk_overlap: 48
chunk_size: 384
dataset: ${dataset}
detector:
  background_mode: tail
  window: 10
embedding:
  batch_size: 16
  model_name: BAAI/bge-m3
  use_hash_embedder: false
generator:
  enabled: true
  provider: transformers
  model_name: Qwen/Qwen2.5-32B-Instruct
  hf_dtype: float16
  temperature: ${temp}
  max_tokens: 200
  prompt_style: ${BEST_PROMPT}
metrics:
  ks: [1, 3, 5, 10]
paths:
  corpus_path: ${corpus}
  gold_path: ${gold}
  output_dir: ${out}
  queries_path: ${queries}
retriever:
  candidate_top_n: 100
  mode: dense
seed: ${seed}
selector:
  mmr_lambda: 0.7
  mode: greedy
  token_budget: ${budget}
side_indices: {}
reranker:
  enabled: ${rerank}
  model_name: BAAI/bge-reranker-v2-m3
  top_n_in: 100
  top_n_out: 20
  batch_size: 32
YAML
}

run() { echo "==== [$(date +%H:%M)] $2 ===="; rag-run-benchmark --config "$1" && echo "[OK] $2" || echo "[FAIL] $2"; }

# ── 1. 32B self-consistency (HotpotQA) — 3 seeds at T=0.7 ────────────────────
for seed in 1 2 3; do
  write_cfg_32b configs/benchmarks/phase4/hp_32b_sc${seed}.yaml \
    "hotpotqa_1k_32b_sc${seed}_${BEST_PROMPT}" \
    "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
    "runs/phase4/hp_32b_sc${seed}" "true" "${seed}" "0.7" "2048"
  run configs/benchmarks/phase4/hp_32b_sc${seed}.yaml "hotpot 32B SC seed=${seed}"
done

# ── 2. 32B self-consistency (2Wiki) — 3 seeds at T=0.7 ───────────────────────
for seed in 1 2 3; do
  write_cfg_32b configs/benchmarks/phase4/2w_32b_sc${seed}.yaml \
    "2wiki_1k_32b_sc${seed}_${BEST_PROMPT}" \
    "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
    "runs/phase4/2w_32b_sc${seed}" "true" "${seed}" "0.7" "2048"
  run configs/benchmarks/phase4/2w_32b_sc${seed}.yaml "2wiki 32B SC seed=${seed}"
done

# ── 3. Ensemble self-consistency results ──────────────────────────────────────
echo "==== [$(date +%H:%M)] ensemble HP self-consistency ===="
python3 scripts/ensemble_answers.py \
  --queries "${DATAHP}/queries.jsonl" \
  --runs \
    $(ls -d runs/phase4/hp_32b_sc1/*/ 2>/dev/null | grep -v index_bundle | head -1) \
    $(ls -d runs/phase4/hp_32b_sc2/*/ 2>/dev/null | grep -v index_bundle | head -1) \
    $(ls -d runs/phase4/hp_32b_sc3/*/ 2>/dev/null | grep -v index_bundle | head -1) \
  --out-dir runs/phase4/hp_32b_selfcon3 2>&1 || echo "[WARN] HP ensemble failed"

echo "==== [$(date +%H:%M)] ensemble 2W self-consistency ===="
python3 scripts/ensemble_answers.py \
  --queries "${DATA2W}/queries.jsonl" \
  --runs \
    $(ls -d runs/phase4/2w_32b_sc1/*/ 2>/dev/null | grep -v index_bundle | head -1) \
    $(ls -d runs/phase4/2w_32b_sc2/*/ 2>/dev/null | grep -v index_bundle | head -1) \
    $(ls -d runs/phase4/2w_32b_sc3/*/ 2>/dev/null | grep -v index_bundle | head -1) \
  --out-dir runs/phase4/2w_32b_selfcon3 2>&1 || echo "[WARN] 2W ensemble failed"

# ── 4. Rescore + decompose ────────────────────────────────────────────────────
echo "==== [$(date +%H:%M)] rescoring ===="
python3 scripts/rescore_em_f1.py --runs-root runs/phase4 --root . 2>&1

for label in hp_32b_sc1 2w_32b_sc1; do
  run_dir=runs/phase4/${label}
  rd=$(ls -d "${run_dir}"/*/ 2>/dev/null | grep -v index_bundle | head -1)
  if [ -z "${rd}" ]; then continue; fi
  dataset=$(echo "${label}" | grep -q '2w_' && echo "2wiki_1k" || echo "hotpotqa_1k")
  echo "==== decomposing ${label} ===="
  python3 scripts/decompose_failures.py \
    --run-dir "${rd}" \
    --queries "data/${dataset}/queries.jsonl" \
    --chunks  "${rd}/../index_bundle/chunks.jsonl" \
    --out-csv "runs/analysis/${label}_phase4_decomp.csv" 2>&1 || true
done

# ── 5. Full results table ─────────────────────────────────────────────────────
python3 scripts/summarize_results.py \
  --runs-root runs \
  --out runs/analysis/all_results_summary.md 2>&1

echo "[phase4] DONE at $(date)"
