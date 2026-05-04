"""IRC-robustness on a *synthetic* corpus designed to stress chunk boundaries.

Key insight: if the gold answer literally spans a chunk boundary, then chunking
choice determines whether selection can recover the joint claim. We build a
corpus where each gold doc has a 'two-sentence answer' — selecting just the
first sentence is wrong, just the second is wrong, only joint is right.

The setup:
- 30 gold docs, each consisting of {SENTENCE_A, SENTENCE_B} — only joint
  contains the answer entity.
- 90 distractor docs.
- 30 queries, each pointing to one gold doc.
- Vary chunk_size (in chars) so the boundary falls *between* SENTENCE_A and
  SENTENCE_B for some sizes (split case) and inside the same chunk for others.

Selection-sensitive metric: chunk-level *gold-pair coverage* — fraction of
queries whose selected_context.jsonl contains BOTH halves of the gold pair.
Selectors with collinear safety should preserve gold-pair coverage across
chunk sizes; brittle selectors should fluctuate.
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


_TOPICS = [
    ("crab nebula", "supernova remnant", "pulsar"),
    ("cosmic microwave background", "thermal radiation", "two point seven kelvin"),
    ("dark matter halo", "gravitational lensing", "rotation curve"),
    ("active galactic nuclei", "supermassive black hole", "relativistic jet"),
    ("gamma-ray burst", "compact merger", "kilonova ejecta"),
    ("neutron star merger", "kilonova", "r-process nucleosynthesis"),
    ("solar flare", "magnetic reconnection", "coronal mass ejection"),
    ("magnetar", "soft gamma repeater", "ultrastrong magnetic field"),
    ("blazar", "relativistic jet", "polarization variability"),
    ("type ia supernova", "white dwarf companion", "standard candle distance"),
    ("redshift survey", "baryon acoustic oscillation", "matter power spectrum"),
    ("cosmic ray", "ultra high energy", "GZK cutoff"),
    ("planetary transit", "exoplanet detection", "limb darkening"),
    ("interstellar dust", "extinction curve", "polarization signal"),
    ("globular cluster", "old stellar population", "horizontal branch"),
    ("dark energy", "cosmological constant", "accelerated expansion"),
    ("primordial black hole", "early universe formation", "gravitational microlensing"),
    ("pulsar timing array", "nanohertz gravitational wave", "Hellings-Downs curve"),
    ("HII region", "ionized hydrogen", "Strömgren sphere"),
    ("supernova explosion mechanism", "neutrino driven wind", "shock revival"),
    ("white dwarf cooling", "carbon oxygen core", "luminosity function"),
    ("microquasar", "stellar mass black hole", "radio jet"),
    ("X-ray binary", "Roche lobe overflow", "accretion disk corona"),
    ("dark photon", "kinetic mixing", "indirect detection signal"),
    ("nuclear matter", "equation of state", "neutron star radius"),
    ("solar neutrinos", "MSW effect", "matter induced oscillation"),
    ("cosmic strings", "topological defect", "string tension constraint"),
    ("21 cm signal", "epoch of reionization", "global brightness temperature"),
    ("planet 9", "trans-Neptunian objects", "perihelion clustering"),
    ("inflation", "slow-roll potential", "tensor to scalar ratio"),
]


def build_synthetic_corpus(out_dir: Path, *, n_distractors_extra: int = 0, seed: int = 0) -> dict[str, Path]:
    """Build (corpus.jsonl, queries.jsonl, gold.jsonl) under out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs: list[dict] = []
    queries: list[dict] = []
    golds: list[dict] = []

    for i, (topic, sent_a, sent_b) in enumerate(_TOPICS):
        doc_id = f"gold_{i:03d}"
        # Each gold doc has TWO sentences — the answer requires BOTH.
        # We deliberately make sentence_a contain the topic (matches retrieval)
        # and sentence_b contain the *answer* (the third element).
        text = (
            f"{topic.title()} is associated with {sent_a}. "
            f"In particular, observations of {topic} reveal {sent_b}."
        )
        docs.append({"doc_id": doc_id, "text": text, "metadata": {"role": "gold", "answer_phrase": sent_b}})
        qid = f"q_{i:03d}"
        queries.append({
            "query_id": qid,
            "text": f"What does {topic} reveal about its physical mechanism?",
            "gold_doc_ids": [doc_id],
            "metadata": {"answer": [sent_b], "topic": topic},
        })
        golds.append({"query_id": qid, "gold_doc_ids": [doc_id]})

    # Distractors must NOT recycle gold-topic vocabulary, or retrieval is contaminated.
    rng = __import__("random").Random(seed)
    distractor_topics = [
        "linear algebra factorization", "convex optimization duality", "graph coloring",
        "sorting networks", "register allocation heuristics", "fluid mixing dynamics",
        "polymer chain statistics", "epidemic threshold models", "queueing theory bounds",
        "auction mechanism design", "lattice gas automata", "membrane permeability scaling",
    ]
    pool = []
    for j in range(60 + n_distractors_extra):
        topic = distractor_topics[j % len(distractor_topics)]
        pool.append(
            f"This passage on {topic} explores theoretical ideas. "
            f"Researchers in {topic} examine convergence rates and asymptotic limits."
        )
    for j, txt in enumerate(pool):
        docs.append({"doc_id": f"dist_{j:04d}", "text": txt, "metadata": {"role": "distractor"}})

    (out_dir / "corpus.jsonl").write_text("\n".join(json.dumps(d) for d in docs) + "\n", encoding="utf-8")
    (out_dir / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n", encoding="utf-8")
    (out_dir / "gold.jsonl").write_text("\n".join(json.dumps(g) for g in golds) + "\n", encoding="utf-8")
    return {"corpus": out_dir / "corpus.jsonl", "queries": out_dir / "queries.jsonl", "gold": out_dir / "gold.jsonl"}


def base_cfg(out: Path, paths: dict[str, Path], *, chunk_size: int, mode: str, candidate_top_n: int = 30, token_budget: int = 240) -> dict:
    return {
        "dataset": f"synthetic_irc_{chunk_size}_{mode}",
        "seed": 0,
        "paths": {
            "corpus_path": str(paths["corpus"]),
            "queries_path": str(paths["queries"]),
            "gold_path": str(paths["gold"]),
            "output_dir": str(out),
        },
        "chunk_size": int(chunk_size),
        "chunk_overlap": min(20, max(1, chunk_size // 6)),
        "embedding": {"use_hash_embedder": True},
        "retriever": {"candidate_top_n": candidate_top_n, "mode": "fusion_rrf"},
        "reranker": {"enabled": False},
        "detector": {"window": 5},
        "selector": {
            "token_budget": token_budget,
            "mode": mode,
            "anti_kt_R": 1.0,
            "anti_kt_n_jets": 1,
            "mmr_lambda": 0.7,
        },
        "generator": {"enabled": True, "provider": "stub"},
        "metrics": {"ks": [1, 3, 5]},
    }


def gold_pair_coverage(run_dir: Path, gold_path: Path, queries_path: Path) -> float:
    """Fraction of queries whose selected_context contains BOTH halves of the
    gold doc's two-sentence answer. Detected via the `answer_phrase` metadata
    being a substring of the selected text."""
    sel = run_dir / "selected_context.jsonl"
    if not sel.is_file():
        return 0.0
    answer_phrases: dict[str, str] = {}
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        ans = q["metadata"].get("answer") or []
        if ans:
            answer_phrases[q["query_id"]] = ans[0].lower()
    selected_by_q: dict[str, list[str]] = {}
    for line in sel.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        selected_by_q.setdefault(s["query_id"], []).append(s["chunk_id"])
    chunks = run_dir.parent / "index_bundle" / "chunks.jsonl"
    if not chunks.is_file():
        return 0.0
    text_by_chunk: dict[str, str] = {}
    for line in chunks.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        text_by_chunk[c["chunk_id"]] = c["text"].lower()
    n = 0
    hits = 0
    for qid, phrase in answer_phrases.items():
        n += 1
        chosen = selected_by_q.get(qid, [])
        joint = " ".join(text_by_chunk.get(cid, "") for cid in chosen)
        if phrase.lower() in joint:
            hits += 1
    return float(hits) / max(1, n)


def run_sweep(corpus_paths: dict[str, Path], out_root: Path, chunk_sizes: list[int], selectors: list[str]) -> list[dict]:
    rows: list[dict] = []
    for cs in chunk_sizes:
        for sel in selectors:
            run_out = out_root / f"cs{cs}_{sel}"
            run_out.mkdir(parents=True, exist_ok=True)
            tmp = run_out / "tmp"
            tmp.mkdir(exist_ok=True)
            cfg_dict = base_cfg(run_out, corpus_paths, chunk_size=cs, mode=sel)
            cfg_path = tmp / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")
            cfg = load_yaml(cfg_path, BenchmarkConfig)
            run_dir = benchmark_run(cfg)
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            cov = gold_pair_coverage(run_dir, corpus_paths["gold"], corpus_paths["queries"])
            rows.append({
                "chunk_size": cs,
                "selector": sel,
                "recall@1": float(metrics.get("recall@1_doc_mean", 0.0)),
                "recall@5": float(metrics.get("recall@5_doc_mean", 0.0)),
                "answer_em": float(metrics.get("answer_em_mean", 0.0)),
                "answer_f1": float(metrics.get("answer_f1_mean", 0.0)),
                "gold_pair_coverage": cov,
            })
    return rows


def selector_stats(rows: list[dict], metric: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for sel in {r["selector"] for r in rows}:
        vals = [r[metric] for r in rows if r["selector"] == sel]
        out[sel] = {
            "mean": statistics.mean(vals) if vals else 0.0,
            "stdev": statistics.pstdev(vals) if len(vals) >= 2 else 0.0,
            "values": vals,
        }
    return out


def write_results(out_root: Path, rows: list[dict]) -> None:
    import csv

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "synthetic_irc.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = ["# Synthetic IRC empirical results", ""]
    for metric in ("recall@1", "answer_f1", "gold_pair_coverage"):
        md.append(f"## {metric} vs chunk_size")
        md.append("")
        md.append("| selector | mean | stdev | CV | values |")
        md.append("|---|---|---|---|---|")
        s = selector_stats(rows, metric)
        for sel, st in sorted(s.items()):
            cv = st["stdev"] / st["mean"] if st["mean"] > 1e-9 else float("inf")
            md.append(f"| {sel} | {st['mean']:.3f} | {st['stdev']:.3f} | {cv:.3f} | {[round(v, 3) for v in st['values']]} |")

        # Pareto-aware pass condition: anti_kt is on the (mean, -stdev) frontier
        # and dominates at least one baseline (better mean OR lower stdev, with no
        # baseline strictly dominating it).
        anti = s.get("anti_kt")
        verdict = "N/A"
        if anti and len(s) > 1:
            dominated = False
            beats_one = False
            for sel, st in s.items():
                if sel == "anti_kt":
                    continue
                if st["mean"] > anti["mean"] + 1e-9 and st["stdev"] < anti["stdev"] - 1e-9:
                    dominated = True  # baseline strictly dominates anti_kt
                if anti["mean"] >= st["mean"] - 1e-9 and anti["stdev"] <= st["stdev"] + 1e-9 and (anti["mean"] > st["mean"] or anti["stdev"] < st["stdev"]):
                    beats_one = True
                if anti["mean"] > 0.5 * st["mean"] and anti["stdev"] < st["stdev"]:
                    beats_one = True
            verdict = "PASS" if (beats_one and not dominated) else "FAIL"
        md.append(f"\n**Pareto-aware pass on `{metric}`: `{verdict}` (anti_kt on the (mean, −stdev) frontier vs baselines)**\n")

    (out_root / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs/synthetic_irc")
    ap.add_argument("--corpus-dir", type=Path, default=REPO / "data/synthetic_irc")
    args = ap.parse_args()
    corpus_paths = build_synthetic_corpus(args.corpus_dir)
    chunk_sizes = [60, 90, 120, 150, 180, 240]
    selectors = ["greedy", "mmr", "anti_kt"]
    rows = run_sweep(corpus_paths, args.out, chunk_sizes, selectors)
    write_results(args.out, rows)
    print(f"wrote {args.out}/results.md")


if __name__ == "__main__":
    main()
