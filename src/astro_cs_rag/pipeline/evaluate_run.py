"""Compute retrieval metrics from a completed run."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

_CITE_RE = re.compile(r"\[E\d+\]")
_FINAL_ANS_RE = re.compile(r"Final answer:\s*([^\n]+)", re.IGNORECASE)


def _extract_answer(text: str) -> str:
    """Strip citations and extract 'Final answer:' portion if present."""
    m = _FINAL_ANS_RE.search(text)
    s = m.group(1) if m else text
    return _CITE_RE.sub("", s).strip()

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.config.schema import EvaluateConfig
from astro_cs_rag.data.loaders import load_gold_jsonl, load_queries_jsonl, merge_gold_into_queries
from astro_cs_rag.evaluation.answer_metrics import (
    aggregate_answer_metrics,
    citation_accuracy,
    exact_match,
    token_f1,
)
from astro_cs_rag.evaluation.metrics import evaluate_ranked_queries
from astro_cs_rag.pipeline.retrieve_run import load_candidates_jsonl, rankings_from_candidates
from astro_cs_rag.reporting.reports import write_summary_report
from astro_cs_rag.cli.helpers import chunk_to_doc_map
from astro_cs_rag.indexing.io import load_chunks_jsonl


def evaluate_run(cfg: EvaluateConfig) -> Path:
    qp = cfg.paths.queries_path
    if qp is None:
        msg = "queries_path missing"
        raise ValueError(msg)
    queries = load_queries_jsonl(qp)
    if cfg.paths.gold_path is not None:
        gold = load_gold_jsonl(cfg.paths.gold_path)
        queries = merge_gold_into_queries(queries, gold)

    candidates = load_candidates_jsonl(cfg.run_dir / "candidates.jsonl")
    rankings = rankings_from_candidates(candidates)

    chunks = load_chunks_jsonl(cfg.index_dir / "chunks.jsonl")
    c2d = chunk_to_doc_map(chunks)

    t0 = time.perf_counter()
    metrics = evaluate_ranked_queries(queries, rankings, c2d, cfg.metrics.ks)
    elapsed = time.perf_counter() - t0

    writer = ArtifactWriter.attach(cfg.run_dir)

    timing_path = cfg.run_dir / "timing.json"
    merged_timing: dict[str, float] = {}
    if timing_path.exists():
        raw = json.loads(timing_path.read_text(encoding="utf-8"))
        merged_timing = {k: float(v) for k, v in raw.items()}
    merged_timing["evaluate_seconds"] = elapsed
    writer.write_json("timing.json", merged_timing)

    metrics_out: dict[str, float] = dict(metrics)
    metrics_out["evaluate_seconds"] = elapsed
    if "retrieve_seconds" in merged_timing:
        metrics_out["retrieve_seconds"] = merged_timing["retrieve_seconds"]

    if cfg.score_answers:
        ans_path = cfg.run_dir / "generated_answers.jsonl"
        if ans_path.is_file():
            chunk_text_lookup = {c.chunk_id: c.text for c in chunks}
            answer_metrics = _score_answers(
                ans_path,
                queries,
                c2d,
                run_dir=cfg.run_dir,
                chunk_text_lookup=chunk_text_lookup,
            )
            metrics_out.update(answer_metrics)
    writer.write_metrics(metrics_out)

    repro = [
        "python -m pip install -e '.[dev]'",
        "rag-run-benchmark --config configs/benchmark.yaml",
        f"rag-evaluate --config configs/eval.yaml  # set run_dir to {cfg.run_dir.name}",
    ]
    write_summary_report(
        cfg.run_dir / "report.md",
        title=f"Evaluation {cfg.dataset}",
        metrics=metrics_out,
        notes=[
            "Doc-level recall / chunk precision / NDCG need gold_doc_ids or gold file.",
            "Answer EM/F1 require generated_answers.jsonl + queries.metadata.answer.",
        ],
        reproduction_commands=repro,
    )
    return cfg.run_dir


def _score_answers(ans_path, queries, chunk_to_doc, *, run_dir=None, chunk_text_lookup=None) -> dict[str, float]:
    import json
    from collections.abc import Mapping

    from astro_cs_rag.evaluation.conservation import conservation_residuals

    if not isinstance(chunk_to_doc, Mapping):
        chunk_to_doc = dict(chunk_to_doc)

    qmap = {q.query_id: q for q in queries}
    rows: list[dict] = []
    cons_rows: list[dict] = []
    for line in ans_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        ans = json.loads(line)
        q = qmap.get(ans["query_id"])
        if q is None:
            continue
        refs = q.metadata.get("answer") or []
        if isinstance(refs, str):
            refs = [refs]
        refs = [r for r in refs if r]
        pred_clean = _extract_answer(ans["answer_text"])
        em = exact_match(pred_clean, refs)
        f1 = token_f1(pred_clean, refs)
        cite = citation_accuracy(
            ans.get("cited_chunk_ids") or [],
            chunk_to_doc,
            q.gold_doc_ids,
        )
        rows.append({"em": em, "f1": f1, "cite_acc": cite})
        if chunk_text_lookup is not None:
            evid_texts = [
                chunk_text_lookup.get(cid, "") for cid in ans.get("selected_chunk_ids") or []
            ]
            cr = conservation_residuals(ans["answer_text"], evid_texts)
            cons_rows.append(
                {
                    "query_id": ans["query_id"],
                    "R_entity": cr.R_entity,
                    "R_numeric": cr.R_numeric,
                    "R_temporal": cr.R_temporal,
                    "faithfulness": cr.faithfulness,
                }
            )
    out = aggregate_answer_metrics(rows)
    if cons_rows:
        out["conservation_R_entity_mean"] = sum(r["R_entity"] for r in cons_rows) / len(cons_rows)
        out["conservation_R_numeric_mean"] = sum(r["R_numeric"] for r in cons_rows) / len(cons_rows)
        out["conservation_R_temporal_mean"] = sum(r["R_temporal"] for r in cons_rows) / len(cons_rows)
        out["conservation_faithfulness_mean"] = sum(r["faithfulness"] for r in cons_rows) / len(cons_rows)
        if run_dir is not None:
            (run_dir / "conservation_residuals.jsonl").write_text(
                "\n".join(json.dumps(r) for r in cons_rows) + "\n", encoding="utf-8"
            )
    return out
