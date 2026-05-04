"""IRC-robustness empirical test for the anti-kT selector (Paper Theorem 1).

Two sub-experiments:

  CHUNK-SIZE SWEEP (collinear safety):
    For chunk_size in [60, 90, 120, 150, 180, 240]:
      build index, retrieve, select with each selector ∈ {greedy, mmr, anti_kt}
      record recall@1_doc and answer_em on the tiny corpus.
    Compute std/range of metrics across chunk_size for each selector.
    Pass condition: anti_kt std < min(greedy.std, mmr.std).

  DISTRACTOR-INJECTION SWEEP (infrared safety):
    Fix chunk_size; inject N synthetic distractor docs (N ∈ {0, 5, 10, 25, 50}).
    Same metric tracking.
    Pass condition: anti_kt's leading-jet membership is a superset of its
    no-distractor leading-jet membership across N (preservation of hard atoms).

Output: runs/irc_robustness/{chunk_sweep.csv, ir_sweep.csv, results.md}.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import yaml

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import BenchmarkConfig
from astro_cs_rag.pipeline.benchmark import benchmark_run

REPO = Path(__file__).resolve().parents[1]
TINY = REPO / "data" / "tiny"


def base_cfg(out: Path, *, chunk_size: int, mode: str, corpus: Path | None = None) -> dict:
    return {
        "dataset": f"irc_{chunk_size}_{mode}",
        "seed": 0,
        "paths": {
            "corpus_path": str(corpus or (TINY / "corpus.jsonl")),
            "queries_path": str(TINY / "queries.jsonl"),
            "gold_path": str(TINY / "gold.jsonl"),
            "output_dir": str(out),
        },
        "chunk_size": int(chunk_size),
        "chunk_overlap": min(20, max(1, chunk_size // 6)),
        "embedding": {"use_hash_embedder": True},
        "retriever": {"candidate_top_n": 20, "mode": "fusion_rrf"},
        "reranker": {"enabled": False},
        "detector": {"window": 5, "background_mode": "tail"},
        "selector": {
            "token_budget": 256,
            "mode": mode,
            "anti_kt_R": 1.0,
            "anti_kt_n_jets": 1,
            "mmr_lambda": 0.7,
        },
        "generator": {"enabled": True, "provider": "stub"},
        "metrics": {"ks": [1, 3, 5]},
    }


def _run(cfg_dict: dict, tmp: Path) -> dict:
    cfg_path = tmp / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
    cfg = load_yaml(cfg_path, BenchmarkConfig)
    run_dir = benchmark_run(cfg)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {
        "recall@1": float(metrics.get("recall@1_doc_mean", 0.0)),
        "recall@5": float(metrics.get("recall@5_doc_mean", 0.0)),
        "answer_em": float(metrics.get("answer_em_mean", 0.0)),
        "answer_f1": float(metrics.get("answer_f1_mean", 0.0)),
        "cite_acc": float(metrics.get("citation_accuracy_mean", 0.0)),
        "run_dir": str(run_dir),
    }


def chunk_sweep(out_root: Path) -> list[dict]:
    rows: list[dict] = []
    chunk_sizes = [60, 90, 120, 150, 180, 240]
    selectors = ["greedy", "mmr", "anti_kt"]
    for cs in chunk_sizes:
        for sel in selectors:
            run_out = out_root / "chunk_sweep" / f"cs{cs}_{sel}"
            run_out.mkdir(parents=True, exist_ok=True)
            tmp = run_out / "tmp"
            tmp.mkdir(exist_ok=True)
            cfg = base_cfg(run_out, chunk_size=cs, mode=sel)
            res = _run(cfg, tmp)
            row = {"chunk_size": cs, "selector": sel, **{k: v for k, v in res.items() if k != "run_dir"}}
            rows.append(row)
    return rows


def _build_corpus_with_distractors(n_distractors: int, out_path: Path) -> Path:
    base = (TINY / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    rows = list(base)
    for i in range(n_distractors):
        rows.append(json.dumps({
            "doc_id": f"distractor_{i:04d}",
            "text": f"This is irrelevant filler passage number {i} about an unrelated topic.",
            "metadata": {"role": "distractor"},
        }))
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out_path


def ir_sweep(out_root: Path) -> list[dict]:
    rows: list[dict] = []
    n_list = [0, 5, 10, 25, 50]
    selectors = ["greedy", "mmr", "anti_kt"]
    for n in n_list:
        corpus_path = out_root / "ir_sweep" / f"corpus_n{n}.jsonl"
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        _build_corpus_with_distractors(n, corpus_path)
        for sel in selectors:
            run_out = out_root / "ir_sweep" / f"n{n}_{sel}"
            run_out.mkdir(parents=True, exist_ok=True)
            tmp = run_out / "tmp"
            tmp.mkdir(exist_ok=True)
            cfg = base_cfg(run_out, chunk_size=120, mode=sel, corpus=corpus_path)
            res = _run(cfg, tmp)
            rows.append({"n_distractors": n, "selector": sel, **{k: v for k, v in res.items() if k != "run_dir"}})
    return rows


def _selector_stats(rows: list[dict], group_key: str, metric: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sel in {r["selector"] for r in rows}:
        sel_rows = [r for r in rows if r["selector"] == sel]
        sel_rows.sort(key=lambda r: r[group_key])
        vals = [r[metric] for r in sel_rows]
        out[sel] = {
            "values": vals,
            "mean": statistics.mean(vals) if vals else 0.0,
            "stdev": statistics.pstdev(vals) if len(vals) >= 2 else 0.0,
            "range": max(vals) - min(vals) if vals else 0.0,
        }
    return out


def write_results(out_root: Path, chunk_rows: list[dict], ir_rows: list[dict]) -> None:
    import csv

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "chunk_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(chunk_rows[0].keys()))
        w.writeheader()
        w.writerows(chunk_rows)
    with (out_root / "ir_sweep.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ir_rows[0].keys()))
        w.writeheader()
        w.writerows(ir_rows)

    chunk_stats = _selector_stats(chunk_rows, "chunk_size", "recall@1")
    ir_stats = _selector_stats(ir_rows, "n_distractors", "recall@1")

    md = ["# IRC-robustness empirical results", ""]
    md.append("## Chunk-size sweep (collinear safety, recall@1 vs chunk_size)")
    md.append("")
    md.append("| selector | mean | stdev | range | values |")
    md.append("|---|---|---|---|---|")
    for sel, st in sorted(chunk_stats.items()):
        md.append(f"| {sel} | {st['mean']:.3f} | {st['stdev']:.3f} | {st['range']:.3f} | {st['values']} |")
    md.append("")
    anti = chunk_stats.get("anti_kt", {})
    others = {k: v for k, v in chunk_stats.items() if k != "anti_kt"}
    if anti and others:
        min_other = min(v["stdev"] for v in others.values())
        verdict = "PASS" if anti.get("stdev", 1) < min_other else "FAIL"
        md.append(f"**Theorem-1 chunk-perturbation pass condition (anti_kt.stdev < min(other.stdev)): `{verdict}`**")
    md.append("")
    md.append("## Distractor-injection sweep (infrared safety, recall@1 vs n_distractors)")
    md.append("")
    md.append("| selector | mean | stdev | range | values |")
    md.append("|---|---|---|---|---|")
    for sel, st in sorted(ir_stats.items()):
        md.append(f"| {sel} | {st['mean']:.3f} | {st['stdev']:.3f} | {st['range']:.3f} | {st['values']} |")
    anti = ir_stats.get("anti_kt", {})
    others = {k: v for k, v in ir_stats.items() if k != "anti_kt"}
    if anti and others:
        min_other = min(v["stdev"] for v in others.values())
        verdict = "PASS" if anti.get("stdev", 1) < min_other else "FAIL"
        md.append("")
        md.append(f"**Theorem-1 IR pass condition (anti_kt.stdev < min(other.stdev)): `{verdict}`**")
    (out_root / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs/irc_robustness")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    chunk_rows = chunk_sweep(args.out)
    ir_rows = ir_sweep(args.out)
    write_results(args.out, chunk_rows, ir_rows)
    print(f"wrote {args.out}/results.md")


if __name__ == "__main__":
    main()
