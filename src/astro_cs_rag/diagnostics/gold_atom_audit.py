"""Gold-atom audit — classify per-query failures into layers.

Failure layers (in upstream order):

    corpus           — gold doc absent or no answer alias appears anywhere in corpus.
    atom_extraction  — gold doc retrieved into atoms.jsonl, but no atom contains
                       any answer alias span.
    retrieval        — at least one gold atom exists, but none enters the
                       atom-level top-k candidate list.
    detection        — gold atom is in candidates but its detector score
                       (raw cosine or detectability) is below selection threshold.
    selection        — gold atom has reasonable score but is not selected
                       under token budget (set / submodular layer dropped it).
    generation       — gold atom selected, final answer wrong/unsupported.

If `answer_aliases` are missing for a query the audit downgrades to
*doc-level* triage: only `corpus` vs `retrieval` vs `selection` are
reported, and atom_extraction/detection are folded into
`detection_or_extraction`. This keeps the diagnostic runnable on
existing gold.jsonl files that lack answer text.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class AtomRecord:
    atom_id: str
    chunk_id: str
    doc_id: str
    text: str
    claim_type: str = "ANY"


@dataclass
class GoldRecord:
    query_id: str
    gold_doc_ids: list[str]
    answer_aliases: list[str] = field(default_factory=list)


@dataclass
class QueryArtifacts:
    """Per-query inputs to the auditor — IDs only, no embeddings."""

    query_id: str
    candidate_atom_ids: list[str]   # ranked, top-k atom IDs
    selected_atom_ids: list[str]    # post-selection atom IDs
    detector_scores: dict[str, float]  # atom_id -> score (any monotonic detector)
    detector_threshold: float = 0.0
    generation_correct: bool | None = None


@dataclass
class AuditRow:
    query_id: str
    gold_doc_ids: list[str]
    answer_aliases: list[str]
    gold_atoms_found: list[str]
    gold_atoms_missing_in_corpus: bool
    gold_atom_in_candidates: list[str]
    gold_atom_selected: list[str]
    failure_layer: str
    notes: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _alias_in(text: str, aliases: Iterable[str]) -> bool:
    if not aliases:
        return False
    t = text.lower()
    return any(a and a.lower() in t for a in aliases)


def find_gold_atoms(
    atoms: Sequence[AtomRecord],
    gold: GoldRecord,
) -> tuple[list[str], bool]:
    """Locate gold atoms by (doc_id ∩ alias-substring).

    Returns (gold_atom_ids, alias_seen_anywhere).
    `alias_seen_anywhere` is true if any atom in the corpus contains an
    alias — used to separate `corpus` from `atom_extraction` failures.
    Without aliases the second value is reported as None upstream.
    """
    gold_doc = set(gold.gold_doc_ids)
    in_doc = [a for a in atoms if a.doc_id in gold_doc]
    if not gold.answer_aliases:
        return [a.atom_id for a in in_doc], True
    matched = [a.atom_id for a in in_doc if _alias_in(a.text, gold.answer_aliases)]
    seen_anywhere = any(_alias_in(a.text, gold.answer_aliases) for a in atoms)
    return matched, seen_anywhere


def classify(
    gold: GoldRecord,
    artifacts: QueryArtifacts,
    gold_atom_ids: list[str],
    alias_seen_anywhere: bool,
) -> tuple[str, dict]:
    """Return (failure_layer, notes). 'none' if generation_correct is True."""
    has_aliases = bool(gold.answer_aliases)
    notes: dict = {"has_aliases": has_aliases}

    if artifacts.generation_correct is True:
        return "none", notes

    if not gold_atom_ids:
        if has_aliases and not alias_seen_anywhere:
            return "corpus", notes
        if has_aliases:
            return "atom_extraction", notes
        return "corpus_or_retrieval", {**notes, "downgraded": True}

    cand = set(artifacts.candidate_atom_ids)
    in_cand = [g for g in gold_atom_ids if g in cand]
    if not in_cand:
        return "retrieval", notes

    sel = set(artifacts.selected_atom_ids)
    in_sel = [g for g in in_cand if g in sel]
    if not in_sel:
        scores = [artifacts.detector_scores.get(g, 0.0) for g in in_cand]
        if max(scores) < artifacts.detector_threshold:
            layer = "detection" if has_aliases else "detection_or_extraction"
            return layer, {**notes, "max_gold_score": float(max(scores))}
        return "selection", {**notes, "max_gold_score": float(max(scores))}

    if artifacts.generation_correct is False:
        return "generation", notes
    return "none", notes


def audit_query(
    atoms: Sequence[AtomRecord],
    gold: GoldRecord,
    artifacts: QueryArtifacts,
) -> AuditRow:
    gold_atom_ids, alias_seen_anywhere = find_gold_atoms(atoms, gold)
    layer, notes = classify(gold, artifacts, gold_atom_ids, alias_seen_anywhere)
    cand = set(artifacts.candidate_atom_ids)
    sel = set(artifacts.selected_atom_ids)
    return AuditRow(
        query_id=gold.query_id,
        gold_doc_ids=list(gold.gold_doc_ids),
        answer_aliases=list(gold.answer_aliases),
        gold_atoms_found=list(gold_atom_ids),
        gold_atoms_missing_in_corpus=(
            bool(gold.answer_aliases) and not alias_seen_anywhere
        ),
        gold_atom_in_candidates=[g for g in gold_atom_ids if g in cand],
        gold_atom_selected=[g for g in gold_atom_ids if g in sel],
        failure_layer=layer,
        notes=notes,
    )


def aggregate(rows: Iterable[AuditRow]) -> dict:
    rows = list(rows)
    if not rows:
        return {"n_queries": 0}
    n = len(rows)
    layers: dict[str, int] = {}
    for r in rows:
        layers[r.failure_layer] = layers.get(r.failure_layer, 0) + 1
    has_gold_atom = sum(1 for r in rows if r.gold_atoms_found)
    in_cand = sum(1 for r in rows if r.gold_atom_in_candidates)
    in_sel = sum(1 for r in rows if r.gold_atom_selected)
    all_in_sel = sum(
        1
        for r in rows
        if r.gold_atoms_found
        and len(r.gold_atom_selected) == len(r.gold_atoms_found)
    )
    return {
        "n_queries": n,
        "gold_atom_presence_rate": has_gold_atom / n,
        "gold_atom_in_candidates_rate": in_cand / n,
        "gold_atom_selected_rate": in_sel / n,
        "all_gold_atoms_selected_rate": all_in_sel / n,
        "failure_layer_distribution": layers,
        "any_alias_present": any(r.answer_aliases for r in rows),
    }


def write_audit_jsonl(rows: Iterable[AuditRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(r.to_json() + "\n")


def load_atoms_jsonl(path: Path) -> list[AtomRecord]:
    out: list[AtomRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(
                AtomRecord(
                    atom_id=d["atom_id"],
                    chunk_id=d["chunk_id"],
                    doc_id=d["doc_id"],
                    text=d.get("text", ""),
                    claim_type=d.get("claim_type", "ANY"),
                )
            )
    return out


def load_gold_jsonl(path: Path) -> list[GoldRecord]:
    out: list[GoldRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out.append(
                GoldRecord(
                    query_id=d["query_id"],
                    gold_doc_ids=list(d.get("gold_doc_ids", [])),
                    answer_aliases=list(d.get("answer_aliases", [])),
                )
            )
    return out
