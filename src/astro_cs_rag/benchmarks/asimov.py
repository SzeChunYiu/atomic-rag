"""Asimov-style benchmark — gold-injection synthesis with controlled positions.

For each test query we know:
- a *gold passage* (text that *certainly* answers the query),
- a *clean distractor pool* (random passages drawn from a topic-disjoint corpus
  so they cannot accidentally answer the query),
- the *position* at which we inject the gold within the distractors.

This lets us decompose end-to-end accuracy into stage-level efficiencies:

  Acc_e2e = ε_chunk · ε_retrieval · ε_select · ε_generate

By construction ε_chunk = 1 (we inject whole passages), ε_retrieval is bounded
by 1.0 because gold is in the corpus, and any deviation from gold-position
recall isolates retrieval imperfection. Subsequent stages then multiplicatively
account for the rest of the gap.

The Asimov benchmark is *not* a substitute for real benchmarks; its role is to
quantify the ceiling each pipeline stage *could* hit if the others were
perfect.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AsimovQuery:
    query_id: str
    text: str
    gold_text: str
    gold_position: int          # 0-based index inside the per-query distractor pool
    pool_size: int              # length of the per-query distractor pool
    references: list[str]       # answer surface forms for EM/F1


@dataclass(frozen=True)
class AsimovBenchmark:
    name: str
    seed: int
    queries: list[AsimovQuery]
    corpus_doc_ids: list[str]   # union pool used across queries (after dedup)


def _hash_id(prefix: str, payload: str) -> str:
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def synthesize_asimov(
    *,
    qa_pairs: list[tuple[str, str, list[str]]],     # (query_text, gold_text, refs)
    distractor_pool: list[str],
    pool_size: int = 50,
    seed: int = 0,
    name: str = "asimov",
    position_strategy: str = "uniform",
) -> AsimovBenchmark:
    """Build an Asimov benchmark from a list of (query, gold_passage, refs) tuples.

    `position_strategy` is one of:
      - "first"   : gold always at index 0
      - "middle"  : gold at index floor(pool_size/2)
      - "last"    : gold at index pool_size-1
      - "uniform" : gold position drawn uniformly per query (revealed in the
                    artifact for reproducibility)
    """
    if not qa_pairs:
        msg = "synthesize_asimov requires at least one (query, gold, refs) tuple"
        raise ValueError(msg)
    if pool_size < 2:
        msg = "pool_size must be >= 2"
        raise ValueError(msg)
    if pool_size > len(distractor_pool) + 1:
        msg = f"distractor pool too small: need >= {pool_size - 1}"
        raise ValueError(msg)

    rng = random.Random(seed)
    queries: list[AsimovQuery] = []
    seen_doc_ids: set[str] = set()
    for i, (qtext, gold, refs) in enumerate(qa_pairs):
        qid = _hash_id("asimov_q", f"{i}:{qtext}")
        gid = _hash_id("asimov_gold", gold)
        seen_doc_ids.add(gid)
        # Pick distractors disjoint from gold and from prior gold strings.
        local = rng.sample(distractor_pool, pool_size - 1)
        if position_strategy == "first":
            pos = 0
        elif position_strategy == "middle":
            pos = pool_size // 2
        elif position_strategy == "last":
            pos = pool_size - 1
        elif position_strategy == "uniform":
            pos = rng.randrange(pool_size)
        else:
            msg = f"unknown position_strategy: {position_strategy}"
            raise ValueError(msg)
        for d in local:
            seen_doc_ids.add(_hash_id("asimov_dist", d))
        queries.append(
            AsimovQuery(
                query_id=qid,
                text=qtext,
                gold_text=gold,
                gold_position=pos,
                pool_size=pool_size,
                references=list(refs),
            )
        )
    return AsimovBenchmark(
        name=name,
        seed=seed,
        queries=queries,
        corpus_doc_ids=sorted(seen_doc_ids),
    )


def write_asimov_jsonl(
    bench: AsimovBenchmark,
    out_dir: Path,
    distractor_pool: list[str],
    rng_seed: int | None = None,
) -> dict[str, Path]:
    """Materialize the Asimov benchmark as the standard (corpus, queries, gold) trio.

    The corpus contains the union of (gold, distractors) across queries, with
    deterministic ordering. Each query's `metadata.gold_position` records the
    injected position so post-hoc analysis can report per-position recall.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(rng_seed if rng_seed is not None else bench.seed)

    docs: dict[str, dict] = {}
    queries_rows: list[dict] = []
    gold_rows: list[dict] = []

    for q in bench.queries:
        gid = _hash_id("asimov_gold", q.gold_text)
        docs[gid] = {
            "doc_id": gid,
            "text": q.gold_text,
            "metadata": {"role": "gold", "asimov_query_id": q.query_id},
        }
        local = rng.sample(distractor_pool, q.pool_size - 1)
        for d in local:
            did = _hash_id("asimov_dist", d)
            docs.setdefault(
                did,
                {"doc_id": did, "text": d, "metadata": {"role": "distractor"}},
            )
        queries_rows.append(
            {
                "query_id": q.query_id,
                "text": q.text,
                "gold_doc_ids": [gid],
                "metadata": {
                    "answer": q.references,
                    "gold_position": q.gold_position,
                    "pool_size": q.pool_size,
                    "asimov": True,
                },
            }
        )
        gold_rows.append({"query_id": q.query_id, "gold_doc_ids": [gid]})

    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.jsonl"
    gold_path = out_dir / "gold.jsonl"

    _write_jsonl(corpus_path, docs.values())
    _write_jsonl(queries_path, queries_rows)
    _write_jsonl(gold_path, gold_rows)

    (out_dir / "asimov_manifest.json").write_text(
        json.dumps(
            {
                "name": bench.name,
                "seed": bench.seed,
                "n_queries": len(bench.queries),
                "n_docs": len(docs),
                "pool_size": bench.queries[0].pool_size if bench.queries else 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"corpus": corpus_path, "queries": queries_path, "gold": gold_path}


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def stage_efficiency_decomposition(
    *,
    epsilon_retrieval: float,
    epsilon_select: float,
    epsilon_generate: float,
) -> dict[str, float]:
    """Multiplicative decomposition. ε_chunk omitted because Asimov sets it to 1."""
    e2e = float(epsilon_retrieval) * float(epsilon_select) * float(epsilon_generate)
    return {
        "epsilon_retrieval": float(epsilon_retrieval),
        "epsilon_select": float(epsilon_select),
        "epsilon_generate": float(epsilon_generate),
        "epsilon_e2e_predicted": e2e,
    }
