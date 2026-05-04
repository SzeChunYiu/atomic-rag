#!/bin/bash
#SBATCH --job-name=acsrag-phase2
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase 2: larger retrieval pool + self-consistency + few-shot CoT.
# Targets identified from CPU failure decomposition:
#   S2a (16.6% 2Wiki): answer chunk outside top-50 → fix: top_k=100, budget=2048
#   G3 abstention:     few-shot CoT should kill remaining "I don't know" cases
#   G1 confabulation:  self-consistency (5 samples, majority vote) chips away at this

set -euo pipefail

PROJECT_DIR="/projects/hep/fs10/shared/nnbar/billy/RAG"
VENV="/projects/hep/fs10/shared/nnbar/billy/packages/acsrag_venv"
export HF_HOME=/projects/hep/fs10/shared/nnbar/billy/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets

module purge
module load GCC/12.3.0 Python/3.11.3 CUDA/12.1.1
set +u; source "${VENV}/bin/activate"; set -u

cd "${PROJECT_DIR}"
echo "[phase2] node=$(hostname -s)  start=$(date)"
mkdir -p logs runs/phase2 configs/benchmarks/phase2

DATAHP=data/hotpotqa_1k
DATA2W=data/2wiki_1k

write_cfg() {
  local cfg="$1" dataset="$2" corpus="$3" queries="$4" gold="$5"
  local out="$6" mode="$7" prompt="$8" rerank="$9" topk="${10}" budget="${11}"
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
  output_dir: ${out}
  queries_path: ${queries}
retriever:
  candidate_top_n: ${topk}
  mode: dense
seed: 0
selector:
  mmr_lambda: 0.7
  mode: ${mode}
  token_budget: ${budget}
  anti_kt_R: 1.0
  anti_kt_n_jets: -2
  partner_score_gate_alpha: 0.7
  partner_use_median_floor: true
side_indices: {}
reranker:
  enabled: ${rerank}
  model_name: BAAI/bge-reranker-v2-m3
  top_n_in: ${topk}
  top_n_out: 20
  batch_size: 32
YAML
}

run() { echo "==== [$(date +%H:%M)] $2 ===="; rag-run-benchmark --config "$1" && echo "[OK] $2" || echo "[FAIL] $2"; }

# ── 1. Budget=2048 + top-100 on HotpotQA ──────────────────────────────────────
write_cfg configs/benchmarks/phase2/hp_greedy_cot_b2048.yaml \
  "hotpotqa_1k_greedy_cot_b2048" \
  "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
  "runs/phase2/hp_greedy_cot_b2048" \
  "greedy" "cot" "false" "100" "2048"
run configs/benchmarks/phase2/hp_greedy_cot_b2048.yaml "hotpot greedy+CoT budget=2048 top100"

write_cfg configs/benchmarks/phase2/hp_rerank_cot_b2048.yaml \
  "hotpotqa_1k_rerank_cot_b2048" \
  "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
  "runs/phase2/hp_rerank_cot_b2048" \
  "greedy" "cot" "true" "100" "2048"
run configs/benchmarks/phase2/hp_rerank_cot_b2048.yaml "hotpot rerank+CoT budget=2048 top100"

# ── 2. Budget=2048 + top-100 on 2Wiki ─────────────────────────────────────────
write_cfg configs/benchmarks/phase2/2w_greedy_cot_b2048.yaml \
  "2wiki_1k_greedy_cot_b2048" \
  "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
  "runs/phase2/2w_greedy_cot_b2048" \
  "greedy" "cot" "false" "100" "2048"
run configs/benchmarks/phase2/2w_greedy_cot_b2048.yaml "2wiki greedy+CoT budget=2048 top100"

write_cfg configs/benchmarks/phase2/2w_rerank_cot_b2048.yaml \
  "2wiki_1k_rerank_cot_b2048" \
  "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
  "runs/phase2/2w_rerank_cot_b2048" \
  "greedy" "cot" "true" "100" "2048"
run configs/benchmarks/phase2/2w_rerank_cot_b2048.yaml "2wiki rerank+CoT budget=2048 top100"

# ── 3. Few-shot CoT on HotpotQA ───────────────────────────────────────────────
write_cfg configs/benchmarks/phase2/hp_greedy_fewshot.yaml \
  "hotpotqa_1k_greedy_fewshot_cot" \
  "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
  "runs/phase2/hp_greedy_fewshot" \
  "greedy" "few_shot_cot" "false" "100" "2048"
run configs/benchmarks/phase2/hp_greedy_fewshot.yaml "hotpot greedy+few-shot-CoT"

write_cfg configs/benchmarks/phase2/hp_rerank_fewshot.yaml \
  "hotpotqa_1k_rerank_fewshot_cot" \
  "${DATAHP}/corpus.jsonl" "${DATAHP}/queries.jsonl" "${DATAHP}/gold.jsonl" \
  "runs/phase2/hp_rerank_fewshot" \
  "greedy" "few_shot_cot" "true" "100" "2048"
run configs/benchmarks/phase2/hp_rerank_fewshot.yaml "hotpot rerank+few-shot-CoT"

# ── 4. Few-shot CoT on 2Wiki ───────────────────────────────────────────────────
write_cfg configs/benchmarks/phase2/2w_rerank_fewshot.yaml \
  "2wiki_1k_rerank_fewshot_cot" \
  "${DATA2W}/corpus.jsonl" "${DATA2W}/queries.jsonl" "${DATA2W}/gold.jsonl" \
  "runs/phase2/2w_rerank_fewshot" \
  "greedy" "few_shot_cot" "true" "100" "2048"
run configs/benchmarks/phase2/2w_rerank_fewshot.yaml "2wiki rerank+few-shot-CoT"

# ── 5. Self-consistency on best config (rerank+CoT, 5 samples at T=0.7) ───────
echo "==== [$(date +%H:%M)] self-consistency (5 seeds x HotpotQA rerank+CoT) ===="
for seed in 1 2 3 4 5; do
  cfg=configs/benchmarks/phase2/hp_rerank_cot_sc${seed}.yaml
  cat > "${cfg}" <<YAML
chunk_overlap: 48
chunk_size: 384
dataset: hotpotqa_1k_rerank_cot_sc${seed}
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
  temperature: 0.7
  max_tokens: 200
  prompt_style: cot
metrics:
  ks: [1, 3, 5, 10]
paths:
  corpus_path: ${DATAHP}/corpus.jsonl
  gold_path: ${DATAHP}/gold.jsonl
  output_dir: runs/phase2/hp_rerank_cot_sc${seed}
  queries_path: ${DATAHP}/queries.jsonl
retriever:
  candidate_top_n: 100
  mode: dense
seed: ${seed}
selector:
  mmr_lambda: 0.7
  mode: greedy
  token_budget: 2048
side_indices: {}
reranker:
  enabled: true
  model_name: BAAI/bge-reranker-v2-m3
  top_n_in: 100
  top_n_out: 20
  batch_size: 32
YAML
  run "${cfg}" "hotpot rerank+CoT self-consistency seed=${seed}"
done

# ── 6. Rescore all phase2 runs ─────────────────────────────────────────────────
echo "==== [$(date +%H:%M)] rescoring all phase2 runs ===="
python3 scripts/rescore_em_f1.py --runs-root runs/phase2 --root . 2>&1

# ── 7. Ensemble self-consistency samples ──────────────────────────────────────
echo "==== [$(date +%H:%M)] self-consistency ensemble ===="
python3 scripts/ensemble_answers.py \
  --queries "${DATAHP}/queries.jsonl" \
  --runs \
    runs/phase2/hp_rerank_cot_sc1/* \
    runs/phase2/hp_rerank_cot_sc2/* \
    runs/phase2/hp_rerank_cot_sc3/* \
    runs/phase2/hp_rerank_cot_sc4/* \
    runs/phase2/hp_rerank_cot_sc5/* \
  --out-dir runs/phase2/hp_rerank_cot_selfcon5 2>&1 || echo "ensemble failed"

# ── 8. Fine-grained decompose on best new runs ─────────────────────────────────
for run_dir in \
  runs/phase2/hp_rerank_cot_b2048 \
  runs/phase2/2w_rerank_cot_b2048 \
  runs/phase2/hp_rerank_fewshot \
  runs/phase2/2w_rerank_fewshot; do
  rd=$(ls -d "${run_dir}"/*/ 2>/dev/null | grep -v index_bundle | head -1)
  if [ -z "${rd}" ]; then continue; fi
  dataset=$(echo "${run_dir}" | grep -q '2w_' && echo "2wiki_1k" || echo "hotpotqa_1k")
  label=$(basename "${run_dir}")
  echo "==== decomposing ${label} ===="
  python3 scripts/decompose_failures.py \
    --run-dir "${rd}" \
    --queries "data/${dataset}/queries.jsonl" \
    --chunks  "${rd}/../index_bundle/chunks.jsonl" \
    --out-csv "runs/analysis/${label}_decomp.csv" 2>&1 || true
done

# ── 9. Generate phase2 summary report ─────────────────────────────────────────
echo "==== [$(date +%H:%M)] phase2 summary ===="
python3 scripts/summarize_results.py \
  --runs-root runs/phase2 \
  --baseline-dir runs/realgen_cot/hotpotqa_1k_cs384_greedy_rerank_qwen7b \
  --out runs/analysis/phase2_summary.md 2>&1 || true

echo "[phase2] DONE at $(date)"
