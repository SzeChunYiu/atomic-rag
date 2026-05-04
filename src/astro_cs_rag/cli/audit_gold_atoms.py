"""CLI: classify per-query failure layers using a gold-atom audit.

Usage:
    python -m astro_cs_rag.cli.audit_gold_atoms \\
        --atoms_dir runs/<run>/index_bundle/atoms \\
        --gold data/<dataset>/gold.jsonl \\
        --candidates runs/<run>/candidates.jsonl \\
        --selected runs/<run>/selected.jsonl \\
        --out runs/<run>/gold_atom_audit.jsonl

Candidates and selected files are JSONL with rows
    {"query_id": str, "atom_ids": list[str]}
optional `scores` (atom_id -> float) in the candidates row.

Generation correctness is optional via --generation_jsonl with rows
    {"query_id": str, "correct": bool}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from astro_cs_rag.diagnostics.gold_atom_audit import (
    QueryArtifacts,
    aggregate,
    audit_query,
    load_atoms_jsonl,
    load_gold_jsonl,
    write_audit_jsonl,
)


def _load_jsonl_by_query(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["query_id"]] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms_dir", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--generation_jsonl", type=Path, default=None)
    ap.add_argument("--detector_threshold", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    atoms = load_atoms_jsonl(args.atoms_dir / "atoms.jsonl")
    gold = load_gold_jsonl(args.gold)
    cand_by_q = _load_jsonl_by_query(args.candidates)
    sel_by_q = _load_jsonl_by_query(args.selected)
    gen_by_q = (
        _load_jsonl_by_query(args.generation_jsonl)
        if args.generation_jsonl
        else {}
    )

    rows = []
    for g in gold:
        cand_row = cand_by_q.get(g.query_id, {})
        sel_row = sel_by_q.get(g.query_id, {})
        gen_row = gen_by_q.get(g.query_id, {})
        artifacts = QueryArtifacts(
            query_id=g.query_id,
            candidate_atom_ids=list(cand_row.get("atom_ids", [])),
            selected_atom_ids=list(sel_row.get("atom_ids", [])),
            detector_scores=dict(cand_row.get("scores", {})),
            detector_threshold=args.detector_threshold,
            generation_correct=gen_row.get("correct"),
        )
        rows.append(audit_query(atoms, g, artifacts))

    write_audit_jsonl(rows, args.out)
    summary = aggregate(rows)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {args.out} ({len(rows)} rows)")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
