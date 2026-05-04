#!/bin/bash
#SBATCH --job-name=acsrag-phase3
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase 3: larger model + best prompting strategy from phase 2.
# Qwen 2.5-32B fits on A100 80GB in float16 (~64GB).
# Targets G1 confabulation (17.8% HotpotQA, 22.8% 2Wiki) — the hardest
# remaining bottleneck, requiring genuine model reasoning capability.

set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"
export HF_HOME=/projects/hep/fs10/shared/nnbar/billy/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets

module purge
module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
set +u; source "${VENV}/bin/activate"; set -u

cd "${PROJECT_DIR}"
echo "[phase3] node=$(hostname -s)  start=$(date)"
mkdir -p logs runs/phase3 configs/benchmarks/phase3

DATAHP=data/hotpotqa_1k
DATA2W=data/2wiki_1k

# Best prompt from phase2 — we expect few_shot_cot with budget=2048 to win.
# If rescore shows otherwise, edit BEST_PROMPT below.
BEST_PROMPT=few_shot_cot

# ── Check available VRAM ───────────────────────────────────────────────────────
nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader
python3 -c "import torch; print('CUDA:', torch.cuda.get_device_name(0)); \
  print('VRAM:', torch.cuda.get_device_properties(0).total_memory//1024**3, 'GB')"

write_cfg_32b() {
  local cfg="$1" dataset="$2" corpus="$3" queries="$4" gold="$5" out="$6" rerank="$7"
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
  temperature: 0.0
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
seed: 0
selector:
  mmr_lambda: 0.7
  mode: greedy
  token_budget: 2048
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

# ── 1. Qwen 2.5-32B on HotpotQA (greedy + rerank) ────────────────────────────
write_cfg_32b configs/benchmarks/phase3/hp_32b_greedy.yaml \
  "hotpotqa_1k_32b_greedy_${BEST_PROMPT}" \
  "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
  "runs/phase3/hp_32b_greedy" "false"
run configs/benchmarks/phase3/hp_32b_greedy.yaml "hotpot 32B greedy"

write_cfg_32b configs/benchmarks/phase3/hp_32b_rerank.yaml \
  "hotpotqa_1k_32b_rerank_${BEST_PROMPT}" \
  "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
  "runs/phase3/hp_32b_rerank" "true"
run configs/benchmarks/phase3/hp_32b_rerank.yaml "hotpot 32B rerank"

# ── 2. Qwen 2.5-32B on 2Wiki ──────────────────────────────────────────────────
write_cfg_32b configs/benchmarks/phase3/2w_32b_greedy.yaml \
  "2wiki_1k_32b_greedy_${BEST_PROMPT}" \
  "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
  "runs/phase3/2w_32b_greedy" "false"
run configs/benchmarks/phase3/2w_32b_greedy.yaml "2wiki 32B greedy"

write_cfg_32b configs/benchmarks/phase3/2w_32b_rerank.yaml \
  "2wiki_1k_32b_rerank_${BEST_PROMPT}" \
  "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
  "runs/phase3/2w_32b_rerank" "true"
run configs/benchmarks/phase3/2w_32b_rerank.yaml "2wiki 32B rerank"

# ── 3. Rescore + decompose ────────────────────────────────────────────────────
echo "==== [$(date +%H:%M)] rescoring ===="
python3 scripts/rescore_em_f1.py --runs-root runs/phase3 --root . 2>&1

for label in hp_32b_rerank 2w_32b_rerank; do
  run_dir=runs/phase3/${label}
  rd=$(ls -d "${run_dir}"/*/ 2>/dev/null | grep -v index_bundle | head -1)
  if [ -z "${rd}" ]; then continue; fi
  dataset=$(echo "${label}" | grep -q '2w_' && echo "2wiki_1k" || echo "hotpotqa_1k")
  echo "==== decomposing ${label} ===="
  python3 scripts/decompose_failures.py \
    --run-dir "${rd}" \
    --queries "data/${dataset}/queries.jsonl" \
    --chunks  "${rd}/../index_bundle/chunks.jsonl" \
    --out-csv "runs/analysis/${label}_decomp.csv" 2>&1 || true
done

# ── 4. Full results table ─────────────────────────────────────────────────────
python3 scripts/summarize_results.py \
  --runs-root runs \
  --out runs/analysis/all_results_summary.md 2>&1

echo "[phase3] DONE at $(date)"
