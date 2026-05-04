"""Mine failure modes from completed runs (see implementation/16_error_analysis.md)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from astro_cs_rag.atoms.schemas import Query
from astro_cs_rag.cli.helpers import chunk_to_doc_map
from astro_cs_rag.indexing.io import load_chunks_jsonl
from astro_cs_rag.pipeline.retrieve_run import load_candidates_jsonl, rankings_from_candidates


def _first_relevant_rank(
    ranked: list[str],
    chunk_to_doc: dict[str, str],
    gold_docs: set[str],
    k: int,
) -> int | None:
    for i, cid in enumerate(ranked[:k], start=1):
        did = chunk_to_doc.get(cid)
        if did is not None and did in gold_docs:
            return i
    return None


def analyze_run_errors(
    *,
    run_dir: Path,
    index_dir: Path,
    queries: list[Query],
    baseline_run_dir: Path | None = None,
    analysis_k: int = 20,
) -> dict[str, Any]:
    chunks = load_chunks_jsonl(index_dir / "chunks.jsonl")
    c2d = chunk_to_doc_map(chunks)

    candidates = load_candidates_jsonl(run_dir / "candidates.jsonl")
    rankings = rankings_from_candidates(candidates)

    base_rankings: dict[str, list[str]] | None = None
    if baseline_run_dir is not None:
        bc = load_candidates_jsonl(baseline_run_dir / "candidates.jsonl")
        base_rankings = rankings_from_candidates(bc)

    selected_path = run_dir / "selected_context.jsonl"
    selected_by_q: dict[str, list[str]] = {}
    if selected_path.is_file():
        for line in selected_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = str(row.get("query_id", ""))
            cid = str(row.get("chunk_id", ""))
            selected_by_q.setdefault(qid, []).append(cid)

    cases: list[dict[str, Any]] = []
    for q in queries:
        if not q.gold_doc_ids:
            continue
        gold = set(q.gold_doc_ids)
        ranked = rankings.get(q.query_id, [])
        fr = _first_relevant_rank(ranked, c2d, gold, analysis_k)

        if fr is None:
            cases.append(
                {
                    "query_id": q.query_id,
                    "kind": "gold_miss_in_candidates",
                    "detail": f"No gold-linked chunk in top-{analysis_k} candidates.",
                }
            )
            continue

        if base_rankings is not None:
            bf = _first_relevant_rank(
                base_rankings.get(q.query_id, []), c2d, gold, analysis_k
            )
            if bf is not None and bf < fr:
                cases.append(
                    {
                        "query_id": q.query_id,
                        "kind": "baseline_beats_method",
                        "detail": f"Baseline first-hit rank {bf} vs method {fr}.",
                    }
                )

        sels = selected_by_q.get(q.query_id, [])
        if sels and not any(c2d.get(cid) in gold for cid in sels):
            cases.append(
                {
                    "query_id": q.query_id,
                    "kind": "irrelevant_selection",
                    "detail": "Selected chunks map outside gold documents.",
                }
            )

    counts = Counter(str(c["kind"]) for c in cases)
    cluster_lines = ["# Failure clusters", "", "| Kind | Count |", "|------|-------|"]
    for kind, n in sorted(counts.items()):
        cluster_lines.append(f"| {kind} | {n} |")
    clusters_md = "\n".join(cluster_lines) + "\n"

    root_md = "\n".join(
        [
            "# Root cause hypotheses",
            "",
            "- If `baseline_beats_method` dominates: revisit fusion / dense vs lexical balance.",
            "- If `gold_miss_in_candidates` dominates: expand candidate pool or indexing.",
            "- If `irrelevant_selection` dominates: detector SNR threshold or greedy budget/overlap.",
            "",
        ]
    )

    return {
        "error_cases": cases,
        "failure_clusters_md": clusters_md,
        "root_cause_report_md": root_md,
    }
