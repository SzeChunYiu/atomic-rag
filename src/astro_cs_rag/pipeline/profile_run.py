"""Per-query score-shape profiles + archetype histogram.

Run after retrieve_run (or rerank_run). Produces:
- query_shapes.jsonl  : ScoreShape per query
- archetype_summary.json : archetype histogram + mean retrieval metrics per archetype
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.diagnostics.calorimetry import ScoreShape, score_shape
from astro_cs_rag.pipeline.retrieve_run import load_candidates_jsonl, rankings_from_candidates


def profile_run(run_dir: Path, gold_path: Path | None = None, ks: tuple[int, ...] = (1, 5, 10)) -> Path:
    candidates = load_candidates_jsonl(run_dir / "candidates.jsonl")
    by_q: dict[str, list[float]] = defaultdict(list)
    for c in candidates:
        by_q[c.query_id].append(c.raw_score)

    rankings = rankings_from_candidates(candidates)

    chunk_to_doc = _load_chunk_to_doc_map_from_run(run_dir)
    gold = _load_gold_map(gold_path) if gold_path else {}

    rows: list[dict] = []
    by_arch: dict[str, dict] = defaultdict(lambda: {"n": 0, **{f"recall@{k}": 0.0 for k in ks}})
    for qid, scores in by_q.items():
        shape: ScoreShape = score_shape(scores)
        out_row: dict[str, object] = {"query_id": qid, **asdict(shape)}
        rows.append(out_row)
        bucket = by_arch[shape.archetype]
        bucket["n"] = int(bucket["n"]) + 1
        gold_docs = set(gold.get(qid, []))
        if gold_docs and rankings.get(qid):
            ranked = rankings[qid]
            for k in ks:
                hit_docs = {chunk_to_doc.get(c) for c in ranked[:k] if chunk_to_doc.get(c) in gold_docs}
                rec = (len(hit_docs & gold_docs) / max(1, len(gold_docs))) if gold_docs else 0.0
                bucket[f"recall@{k}"] = float(bucket[f"recall@{k}"]) + rec

    summary: dict[str, object] = {"archetypes": {}}
    for arch, agg in by_arch.items():
        n = int(agg["n"]) or 1
        summary_archetype: dict[str, float] = {"n": float(agg["n"])}
        for k in ks:
            summary_archetype[f"recall@{k}_mean"] = float(agg[f"recall@{k}"]) / n
        summary["archetypes"][arch] = summary_archetype  # type: ignore[index]
    summary["query_count"] = float(len(by_q))

    writer = ArtifactWriter.attach(run_dir)
    writer.write_jsonl("query_shapes.jsonl", rows)
    writer.write_json("archetype_summary.json", summary)
    return run_dir


def _load_chunk_to_doc_map_from_run(run_dir: Path) -> dict[str, str]:
    """Recover chunk→doc by parsing the chunk_id format `<doc_id>::<part>`.

    We don't always have access to the index_dir from inside a run, so we
    use the chunk_id naming convention. Robust because chunking is the only
    component that mints chunk_ids.
    """
    candidates = load_candidates_jsonl(run_dir / "candidates.jsonl")
    out: dict[str, str] = {}
    for c in candidates:
        if "::" in c.chunk_id:
            out[c.chunk_id] = c.chunk_id.split("::", 1)[0]
    return out


def _load_gold_map(gold_path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in gold_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out[str(row["query_id"])] = [str(x) for x in row.get("gold_doc_ids") or []]
    return out
