#!/bin/bash
#SBATCH --job-name=acsrag-sc
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

echo "[acsrag] node = $(hostname -s)"

# Self-consistency CoT on greedy+rerank+CoT (best-known config)
# 5 samples × 1000 queries × ~2.5s/sample = ~3.5 hours expected.
RD=$(ls -d runs/realgen_cot/hotpotqa_1k_cs384_greedy_rerank_qwen7b/*/ | grep -v index_bundle | head -1)
echo "Run dir: ${RD}"
python3 scripts/self_consistency_cot.py "${RD}" --n_samples 5 --temperature 0.7

# Recompute metrics on the SC output
python3 -c "
import json, re, string
from collections import Counter
from pathlib import Path

QUERIES = Path('data/hotpotqa_1k/queries.jsonl')
gold_docs = {}; gold_ans = {}
for line in open(QUERIES):
    q = json.loads(line)
    gold_docs[q['query_id']] = set(q['gold_doc_ids'])
    a = q.get('metadata', {}).get('answer')
    gold_ans[q['query_id']] = [a] if isinstance(a, str) else (a or [])

PUNC = re.compile(rf'[{re.escape(string.punctuation)}]')
ART = re.compile(r'\b(a|an|the)\b')
WS = re.compile(r'\s+')
def norm(s):
    s = PUNC.sub(' ', s.lower())
    s = ART.sub(' ', s)
    return WS.sub(' ', s).strip()
def f1(p, refs):
    if not refs: return 0.0
    pt = norm(p).split()
    if not pt: return 0.0
    best = 0.0
    for r in refs:
        rt = norm(r).split()
        if not rt: continue
        common = Counter(pt) & Counter(rt)
        n = sum(common.values())
        if n == 0: continue
        prec = n/len(pt); rec = n/len(rt)
        best = max(best, 2*prec*rec/(prec+rec))
    return best

rd = Path('${RD}')
chunk_doc = {}
for line in open(rd.parent / 'index_bundle' / 'chunks.jsonl'):
    c = json.loads(line); chunk_doc[c['chunk_id']] = c['doc_id']

n=0; sf=0.0; em=0.0; cit=0.0
for line in open(rd / 'generated_answers_sc.jsonl'):
    r = json.loads(line)
    qid = r['query_id']; n += 1
    sf += f1(r['answer_text'], gold_ans.get(qid, []))
    em += float(any(norm(r['answer_text']) == norm(a) for a in gold_ans.get(qid, [])))
    cited = r['cited_chunk_ids']
    if cited:
        cit += sum(1 for c in cited if chunk_doc.get(c) in gold_docs.get(qid, set())) / len(cited)
print(f'n={n}  F1={sf/n:.4f}  EM={em/n:.4f}  cit_acc={cit/n:.4f}')
"

echo "[acsrag] self-consistency done"
