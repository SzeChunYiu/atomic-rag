"""Paired bootstrap on per-query citation_accuracy for terse-prompt sweep."""
import json
import glob
from pathlib import Path

import numpy as np

ROOT = Path("/projects/hep/fs10/shared/nnbar/billy/RAG/runs/realgen_terse")
QUERIES = Path("/projects/hep/fs10/shared/nnbar/billy/RAG/data/hotpotqa_1k/queries.jsonl")

gold = {}
for line in open(QUERIES):
    q = json.loads(line)
    gold[q["query_id"]] = set(q["gold_doc_ids"])

c2d = {}
for p in glob.glob(str(ROOT / "*/index_bundle/chunks.jsonl")):
    for line in open(p):
        c = json.loads(line)
        c2d[c["chunk_id"]] = c["doc_id"]


def per_query_cit(name: str) -> dict:
    subs = list((ROOT / name).glob("*/generated_answers.jsonl"))
    if not subs:
        return {}
    out = {}
    for line in open(subs[0]):
        r = json.loads(line)
        cited = r.get("cited_chunk_ids", [])
        if not cited:
            out[r["query_id"]] = 0.0
            continue
        gd = gold.get(r["query_id"], set())
        hits = sum(1 for cid in cited if c2d.get(cid) in gd)
        out[r["query_id"]] = hits / len(cited)
    return out


g = per_query_cit("hotpotqa_1k_cs384_greedy_qwen7b")
v = per_query_cit("hotpotqa_1k_cs384_v4a0_7_qwen7b")
m = per_query_cit("hotpotqa_1k_cs384_mmr_qwen7b")

print(f"loaded: greedy={len(g)} v4={len(v)} mmr={len(m)}")
for cmp, label in [(v, "v4_a07"), (m, "mmr")]:
    common = sorted(set(g) & set(cmp))
    diffs = np.array([cmp[q] - g[q] for q in common])
    rng = np.random.default_rng(0)
    boot = np.array([diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(10000)])
    print(f"{label} vs greedy: n={len(diffs)} mean={diffs.mean():+.4f} P(>0)={float((boot > 0).mean()):.3f} CI95=[{np.percentile(boot, 2.5):+.4f}, {np.percentile(boot, 97.5):+.4f}]")
    print(f"  wins={int((diffs > 0).sum())} losses={int((diffs < 0).sum())} ties={int((diffs == 0).sum())}")
