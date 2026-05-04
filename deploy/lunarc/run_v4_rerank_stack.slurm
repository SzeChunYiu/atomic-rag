#!/bin/bash
#SBATCH --job-name=acsrag-v4-rerank
#SBATCH --account=lu2026-2-51
#SBATCH --partition=gpua100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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

# Stack test: v4 anti-kT selector ON TOP of cross-encoder rerank.
# Hypothesis: rerank wins at retriever stage (+7.2pp recall@5), v4 wins at
# selection stage (+1.4pp cit_acc on bridge queries). If orthogonal they stack.
mkdir -p runs/realgen_terse
base=hotpotqa_1k_cs384_v4_rerank_qwen7b
override=runs/realgen_terse/${base}
rm -rf "${override}"
mkdir -p "${override}"

cfg="${override}/cfg.yaml"
cat > "${cfg}" <<'YAML'
chunk_overlap: 48
chunk_size: 384
dataset: hotpotqa_1k_cs384_v4_rerank_qwen7b_terse
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
  corpus_path: data/hotpotqa_1k/corpus.jsonl
  gold_path: data/hotpotqa_1k/gold.jsonl
  output_dir: PLACEHOLDER
  queries_path: data/hotpotqa_1k/queries.jsonl
reranker:
  enabled: true
  model_name: BAAI/bge-reranker-v2-m3
  top_n_in: 50
  top_n_out: 20
  batch_size: 32
retriever:
  candidate_top_n: 50
  mode: dense
seed: 0
selector:
  anti_kt_R: 1.0
  anti_kt_n_jets: -2
  partner_score_gate_alpha: 0.7
  partner_use_median_floor: true
  mmr_lambda: 0.7
  mode: anti_kt
  token_budget: 1024
side_indices: {}
YAML
sed -i "s|PLACEHOLDER|${override}|" "${cfg}"

echo "==== running ${cfg} ===="
rag-run-benchmark --config "${cfg}"

echo "[acsrag] v4-rerank stack done"
