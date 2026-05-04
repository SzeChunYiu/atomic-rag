"""Generate one YAML config per (dataset × retriever × selector) combination.

Run:  python scripts/build_experiment_matrix.py
Output: configs/benchmarks/matrix/<dataset>__<retr>__<selector>.yaml
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import yaml

DATASETS = {
    "tiny": {
        "corpus_path": "data/tiny/corpus.jsonl",
        "queries_path": "data/tiny/queries.jsonl",
        "gold_path": "data/tiny/gold.jsonl",
        "use_hash_embedder": True,
        "splade_use_hash": True,
        "late_interaction_use_hash": True,
    },
    "hotpotqa_1k": {
        "corpus_path": "data/hotpotqa_1k/corpus.jsonl",
        "queries_path": "data/hotpotqa_1k/queries.jsonl",
        "gold_path": "data/hotpotqa_1k/gold.jsonl",
        "use_hash_embedder": False,
        "splade_use_hash": False,
        "late_interaction_use_hash": False,
    },
    "nq_open_1k": {
        "corpus_path": "data/nq_open_1k/corpus.jsonl",
        "queries_path": "data/nq_open_1k/queries.jsonl",
        "gold_path": "data/nq_open_1k/gold.jsonl",
        "use_hash_embedder": False,
        "splade_use_hash": False,
        "late_interaction_use_hash": False,
    },
}

RETRIEVERS = ["bm25", "dense", "fusion_rrf", "hierarchical", "splade", "late_interaction"]
SELECTORS = ["greedy", "anti_kt", "mmr"]


def build_one(
    dataset: str,
    retriever: str,
    selector: str,
    *,
    out_dir: Path,
) -> Path:
    d = DATASETS[dataset]
    side_indices: dict[str, object] = {}
    if retriever == "hierarchical":
        side_indices = {"hierarchical": True, "hierarchical_branching": 6}
    elif retriever == "splade":
        side_indices = {"splade": True, "splade_use_hash": d["splade_use_hash"]}
    elif retriever == "late_interaction":
        side_indices = {"late_interaction": True, "late_interaction_use_hash": d["late_interaction_use_hash"]}

    name = f"{dataset}__{retriever}__{selector}"
    cfg = {
        "dataset": name,
        "seed": 0,
        "paths": {
            "corpus_path": d["corpus_path"],
            "queries_path": d["queries_path"],
            "gold_path": d["gold_path"],
            "output_dir": f"runs/matrix/{name}",
        },
        "chunk_size": 512 if dataset != "tiny" else 120,
        "chunk_overlap": 64 if dataset != "tiny" else 20,
        "embedding": {
            "use_hash_embedder": d["use_hash_embedder"],
            "model_name": "BAAI/bge-m3",
            "batch_size": 16,
        },
        "side_indices": side_indices,
        "retriever": {
            "candidate_top_n": 50,
            "mode": retriever,
        },
        "reranker": {"enabled": False},
        "detector": {"window": 10, "background_mode": "tail"},
        "selector": {
            "token_budget": 1024 if dataset != "tiny" else 256,
            "mode": selector,
            "anti_kt_R": 1.0,
            "anti_kt_n_jets": 1,
            "mmr_lambda": 0.7,
        },
        "generator": {
            "enabled": True,
            "provider": "ollama" if dataset != "tiny" else "stub",
            "model_name": "llama3.1:8b-instruct-q4_K_M",
            "temperature": 0.0,
            "seed": 0,
            "max_tokens": 256,
            "prompt_style": "citation",
        },
        "metrics": {"ks": [1, 3, 5, 10]},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("configs/benchmarks/matrix"))
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS.keys()))
    args = ap.parse_args()
    paths: list[Path] = []
    for ds, retr, sel in product(args.datasets, RETRIEVERS, SELECTORS):
        paths.append(build_one(ds, retr, sel, out_dir=args.out))
    print(f"wrote {len(paths)} configs to {args.out}")


if __name__ == "__main__":
    main()
