"""Paired bootstrap on per-query metrics for all terse-prompt selector pairs.

Computes Δ, P(Δ>0), 95% CI for citation_accuracy and answer F1.
"""
import json
import glob
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/projects/hep/fs10/shared/nnbar/billy/RAG/runs/realgen_terse")
QUERIES = Path("/projects/hep/fs10/shared/nnbar/billy/RAG/data/hotpotqa_1k/queries.jsonl")

gold_docs: dict[str, set] = {}
gold_ans: dict[str, list[str]] = {}
for line in open(QUERIES):
    q = json.loads(line)
    gold_docs[q["query_id"]] = set(q["gold_doc_ids"])
    a = q.get("metadata", {}).get("answer")
    gold_ans[q["query_id"]] = [a] if isinstance(a, str) else (a or [])

c2d: dict[str, str] = {}
for p in glob.glob(str(ROOT / "*/index_bundle/chunks.jsonl")):
    for line in open(p):
        c = json.loads(line)
        c2d[c["chunk_id"]] = c["doc_id"]


def _norm(s: str) -> list[str]:
    s = re.sub(r"\[E\d+\]", " ", s)  # strip evidence markers
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    return [tok for tok in s.split() if tok not in {"a", "an", "the"}]


def _f1(pred: str, gold: list[str]) -> float:
    if not gold:
        return 0.0
    p = _norm(pred)
    if not p:
        return 0.0
    best = 0.0
    for g in gold:
        ng = _norm(g)
        if not ng:
            continue
        common = Counter(p) & Counter(ng)
        match = sum(common.values())
        if match == 0:
            continue
        precision = match / len(p)
        recall = match / len(ng)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def per_query(name: str) -> tuple[dict, dict]:
    subs = list(ROOT.glob(f"{name}/*/generated_answers.jsonl"))
    if not subs:
        return {}, {}
    cit, f1 = {}, {}
    for line in open(subs[0]):
        r = json.loads(line)
        cited = r.get("cited_chunk_ids", [])
        gd = gold_docs.get(r["query_id"], set())
        cit[r["query_id"]] = (
            sum(1 for cid in cited if c2d.get(cid) in gd) / len(cited) if cited else 0.0
        )
        f1[r["query_id"]] = _f1(r.get("answer_text") or r.get("answer", ""), gold_ans.get(r["query_id"], []))
    return cit, f1


def boot(diffs: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(diffs), (n_boot, len(diffs)))
    means = diffs[resamples].mean(axis=1)
    return float(diffs.mean()), float((means > 0).mean()), (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


SELECTORS = [
    "hotpotqa_1k_cs384_greedy_qwen7b",
    "hotpotqa_1k_cs384_v4a0_7_qwen7b",
    "hotpotqa_1k_cs384_mmr_qwen7b",
    "hotpotqa_1k_cs384_greedy_rerank_qwen7b",
    "hotpotqa_1k_cs384_v4_rerank_qwen7b",
    "hotpotqa_1k_cs384_clean_rag_qwen7b",
]

short = {
    SELECTORS[0]: "greedy",
    SELECTORS[1]: "v4_a0.7",
    SELECTORS[2]: "mmr",
    SELECTORS[3]: "greedy+rrk",
    SELECTORS[4]: "v4+rrk",
    SELECTORS[5]: "clean_rag",
}

data = {s: per_query(s) for s in SELECTORS}

print(f"{'comparison':<28} | {'metric':<8} | {'delta':>8} | {'P(>0)':>6} | {'CI95':>22}")
print("-" * 86)

baseline = SELECTORS[0]
for s in SELECTORS[1:]:
    common = sorted(set(data[baseline][0]) & set(data[s][0]))
    if not common:
        continue
    for label, idx in [("cit_acc", 0), ("F1", 1)]:
        diffs = np.array([data[s][idx][q] - data[baseline][idx][q] for q in common])
        mean, p_pos, ci = boot(diffs)
        cmp = f"{short[s]} - {short[baseline]}"
        print(f"{cmp:<28} | {label:<8} | {mean:>+.4f} | {p_pos:>6.3f} | [{ci[0]:>+.4f}, {ci[1]:>+.4f}]")

print("\nHead-to-head: v4_a0.7 vs greedy+rrk:")
common = sorted(set(data[SELECTORS[1]][0]) & set(data[SELECTORS[3]][0]))
for label, idx in [("cit_acc", 0), ("F1", 1)]:
    diffs = np.array([data[SELECTORS[3]][idx][q] - data[SELECTORS[1]][idx][q] for q in common])
    mean, p_pos, ci = boot(diffs)
    print(f"  greedy+rrk - v4_a0.7  {label}: Δ={mean:+.4f}  P(rrk>v4)={p_pos:.3f}  CI95=[{ci[0]:+.4f}, {ci[1]:+.4f}]")
