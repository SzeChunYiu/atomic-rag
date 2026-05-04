#!/bin/bash
#SBATCH --job-name=acsrag-2wiki
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

echo "[acsrag] node     = $(hostname -s)"

# Stage 1 — Prepare 2WikiMultihopQA subset (1000 queries).
DATA=data/2wiki_1k
if [ ! -f "${DATA}/queries.jsonl" ]; then
  rag-prepare-data --dataset 2wikimultihop --split validation --n 1000 --seed 0 --out "${DATA}"
fi
ls -la "${DATA}/" 2>/dev/null

# Stage 2 — Build configs: greedy / v4 (alpha=0.7) / greedy+rerank with terse prompt + Qwen 7B.
mkdir -p configs/benchmarks/2wiki runs/2wiki
for variant in greedy v4a0_7 greedy_rerank; do
  cfg=configs/benchmarks/2wiki/2wiki_1k_cs384_${variant}_qwen7b.yaml
  cat > "${cfg}" <<YAML
chunk_overlap: 48
chunk_size: 384
dataset: 2wiki_1k_cs384_${variant}_terse
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
metrics:
  ks: [1, 3, 5, 10]
paths:
  corpus_path: ${DATA}/corpus.jsonl
  gold_path: ${DATA}/gold.jsonl
  output_dir: runs/2wiki/2wiki_1k_cs384_${variant}_qwen7b
  queries_path: ${DATA}/queries.jsonl
retriever:
  candidate_top_n: 50
  mode: dense
seed: 0
selector:
YAML
  if [ "${variant}" = "greedy" ]; then
    cat >> "${cfg}" <<YAML
  mmr_lambda: 0.7
  mode: greedy
  token_budget: 1024
side_indices: {}
reranker:
  enabled: false
YAML
  elif [ "${variant}" = "v4a0_7" ]; then
    cat >> "${cfg}" <<YAML
  anti_kt_R: 1.0
  anti_kt_n_jets: -2
  partner_score_gate_alpha: 0.7
  partner_use_median_floor: true
  mmr_lambda: 0.7
  mode: anti_kt
  token_budget: 1024
side_indices: {}
reranker:
  enabled: false
YAML
  else
    cat >> "${cfg}" <<YAML
  mmr_lambda: 0.7
  mode: greedy
  token_budget: 1024
side_indices: {}
reranker:
  enabled: true
  model_name: BAAI/bge-reranker-v2-m3
  top_n_in: 50
  top_n_out: 20
  batch_size: 32
YAML
  fi
  echo "  wrote ${cfg}"
done

# Stage 3 — Run benchmarks
for variant in greedy v4a0_7 greedy_rerank; do
  cfg=configs/benchmarks/2wiki/2wiki_1k_cs384_${variant}_qwen7b.yaml
  echo "==== running ${cfg} ===="
  rag-run-benchmark --config "${cfg}" || echo "FAILED: ${cfg}"
done

echo "[acsrag] 2wiki sweep done"
