#!/bin/bash
#SBATCH --job-name=acsrag-breakthrough
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
echo "[breakthrough] node = $(hostname -s)  time = $(date)"

mkdir -p logs configs/benchmarks/breakthrough runs/breakthrough

DATA2W=data/2wiki_1k
DATAHP=data/hotpotqa_1k

# ── Helper: write a benchmark config ──────────────────────────────────────────
write_cfg() {
  local cfg="$1" dataset="$2" corpus="$3" queries="$4" gold="$5"
  local out_dir="$6" mode="$7" prompt="$8" rerank="$9"
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
  model_name: Qwen/Qwen2.5-7B-Instruct
  temperature: 0.0
  max_tokens: 200
  prompt_style: ${prompt}
metrics:
  ks: [1, 3, 5, 10]
paths:
  corpus_path: ${corpus}
  gold_path: ${gold}
  output_dir: ${out_dir}
  queries_path: ${queries}
retriever:
  candidate_top_n: 50
  mode: dense
seed: 0
selector:
  mmr_lambda: 0.7
  mode: ${mode}
  token_budget: 1024
  anti_kt_R: 1.0
  anti_kt_n_jets: -2
  partner_score_gate_alpha: 0.7
  partner_use_median_floor: true
side_indices: {}
reranker:
  enabled: ${rerank}
  model_name: BAAI/bge-reranker-v2-m3
  top_n_in: 50
  top_n_out: 20
  batch_size: 32
YAML
}

run() {
  local cfg="$1" label="$2"
  echo "==== [$(date +%H:%M)] ${label} ===="
  rag-run-benchmark --config "${cfg}" && echo "[OK] ${label}" || echo "[FAIL] ${label}"
}

# ── Experiment 1: 2Wiki + CoT (greedy) ────────────────────────────────────────
write_cfg \
  configs/benchmarks/breakthrough/2wiki_greedy_cot.yaml \
  "2wiki_1k_cs384_greedy_cot" \
  "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
  "runs/breakthrough/2wiki_greedy_cot" \
  "greedy" "cot" "false"
run configs/benchmarks/breakthrough/2wiki_greedy_cot.yaml "2wiki greedy+CoT"

# ── Experiment 2: 2Wiki + CoT + rerank ────────────────────────────────────────
write_cfg \
  configs/benchmarks/breakthrough/2wiki_rerank_cot.yaml \
  "2wiki_1k_cs384_rerank_cot" \
  "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
  "runs/breakthrough/2wiki_rerank_cot" \
  "greedy" "cot" "true"
run configs/benchmarks/breakthrough/2wiki_rerank_cot.yaml "2wiki rerank+CoT"

# ── Experiment 3: HotpotQA + v4 + CoT ────────────────────────────────────────
write_cfg \
  configs/benchmarks/breakthrough/hotpot_v4_cot.yaml \
  "hotpotqa_1k_cs384_v4_cot" \
  "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
  "runs/breakthrough/hotpot_v4_cot" \
  "anti_kt" "cot" "false"
run configs/benchmarks/breakthrough/hotpot_v4_cot.yaml "hotpot v4+CoT"

# ── Experiment 4: HotpotQA + v4 + rerank + CoT ───────────────────────────────
write_cfg \
  configs/benchmarks/breakthrough/hotpot_v4_rerank_cot.yaml \
  "hotpotqa_1k_cs384_v4_rerank_cot" \
  "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
  "runs/breakthrough/hotpot_v4_rerank_cot" \
  "anti_kt" "cot" "true"
run configs/benchmarks/breakthrough/hotpot_v4_rerank_cot.yaml "hotpot v4+rerank+CoT"

# ── Experiment 5: 2Wiki + v4 + CoT ───────────────────────────────────────────
write_cfg \
  configs/benchmarks/breakthrough/2wiki_v4_cot.yaml \
  "2wiki_1k_cs384_v4_cot" \
  "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
  "runs/breakthrough/2wiki_v4_cot" \
  "anti_kt" "cot" "false"
run configs/benchmarks/breakthrough/2wiki_v4_cot.yaml "2wiki v4+CoT"

# ── Experiment 6: 2-step iterative retrieval for 2Wiki ───────────────────────
echo "==== [$(date +%H:%M)] 2wiki iterative retrieval (2-step) ===="
python3 scripts/iterative_retrieval_bench.py \
  --queries   "${DATA2W}/queries.jsonl" \
  --corpus    "${DATA2W}/corpus.jsonl" \
  --index-dir runs/2wiki/2wiki_1k_cs384_greedy_qwen7b/index_bundle \
  --out-dir   runs/breakthrough/2wiki_iterative \
  --model     Qwen/Qwen2.5-7B-Instruct \
  --top-k     50 \
  --token-budget 1024 \
  --max-steps 2 \
  && echo "[OK] 2wiki iterative" || echo "[FAIL] 2wiki iterative"

# ── Re-score all new runs ─────────────────────────────────────────────────────
echo "==== [$(date +%H:%M)] rescoring ===="
python3 scripts/rescore_em_f1.py --runs-root runs/breakthrough --root . 2>&1

echo "[breakthrough] ALL DONE at $(date)"
