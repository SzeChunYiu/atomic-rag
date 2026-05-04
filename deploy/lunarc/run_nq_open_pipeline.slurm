#!/bin/bash
#SBATCH --job-name=acsrag-nq-open
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

echo "[acsrag] node     = $(hostname -s)"
echo "[acsrag] gpu      = $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Stage 1 — Prepare NQ-open subset (1000 queries).
echo "==== Stage 1: prepare NQ-open data ===="
NQ_DATA=data/nq_open_1k
if [ ! -f "${NQ_DATA}/queries.jsonl" ]; then
  rag-prepare-data --dataset nq_open --split validation --n 1000 --seed 0 --out "${NQ_DATA}" || { echo "FAIL prep"; exit 1; }
else
  echo "  data exists at ${NQ_DATA}, skipping"
fi
ls -la "${NQ_DATA}/" 2>/dev/null

# Stage 2 — Build configs on-the-fly: greedy / v4 / MMR with terse prompt + Qwen 7B.
echo "==== Stage 2: write configs ===="
mkdir -p configs/benchmarks/nq_open runs/nq_open
for sel in greedy v4a0_7 mmr; do
  cfg=configs/benchmarks/nq_open/nq_open_1k_cs384_${sel}_qwen7b.yaml
  cat > "${cfg}" <<YAML
chunk_overlap: 48
chunk_size: 384
dataset: nq_open_1k_cs384_${sel}
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
  corpus_path: ${NQ_DATA}/corpus.jsonl
  gold_path: ${NQ_DATA}/gold.jsonl
  output_dir: runs/nq_open/nq_open_1k_cs384_${sel}_qwen7b
  queries_path: ${NQ_DATA}/queries.jsonl
reranker:
  enabled: false
retriever:
  candidate_top_n: 50
  mode: dense
seed: 0
selector:
YAML
  if [ "${sel}" = "greedy" ]; then
    cat >> "${cfg}" <<YAML
  mmr_lambda: 0.7
  mode: greedy
  token_budget: 1024
side_indices: {}
YAML
  elif [ "${sel}" = "v4a0_7" ]; then
    cat >> "${cfg}" <<YAML
  anti_kt_R: 1.0
  anti_kt_n_jets: -2
  partner_score_gate_alpha: 0.7
  partner_use_median_floor: true
  mmr_lambda: 0.7
  mode: anti_kt
  token_budget: 1024
side_indices: {}
YAML
  else
    cat >> "${cfg}" <<YAML
  mmr_lambda: 0.7
  mode: mmr
  token_budget: 1024
side_indices: {}
YAML
  fi
  echo "  wrote ${cfg}"
done

# Stage 3 — Run benchmarks
echo "==== Stage 3: NQ-open benchmark sweep ===="
for sel in greedy v4a0_7 mmr; do
  cfg=configs/benchmarks/nq_open/nq_open_1k_cs384_${sel}_qwen7b.yaml
  echo "==== running ${cfg} ===="
  rag-run-benchmark --config "${cfg}" || echo "FAILED: ${cfg}"
done

echo "[acsrag] NQ-open sweep done"
