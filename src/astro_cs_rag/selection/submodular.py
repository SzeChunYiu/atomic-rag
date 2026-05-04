"""Submodular set-cover selection over typed evidence atoms.

Implements greedy submodular maximization with diminishing-return
coverage utility. Designed to plug into D04 atom-level retrieval as a
drop-in replacement for greedy budget-fill.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


_WORD = re.compile(r"\b[A-Za-z][A-Za-z\-']{2,}\b")
_PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b")


@dataclass(frozen=True)
class Facet:
    type_tag: str
    entity: str  # lowercase normalized
    weight: float = 1.0  # soft query-type confidence; 1.0 = hard match

    def key(self) -> str:
        return f"{self.type_tag}::{self.entity}"


def query_facets(
    query: str,
    intent_type: str,
    intent_conf: dict[str, float] | None = None,
    type_conf_threshold: float = 0.20,
) -> list[Facet]:
    """Extract candidate facets from the query.

    When intent_conf is provided, emits one weighted type-facet per type
    whose confidence exceeds type_conf_threshold — recovering signal from
    queries that hard query_intent() labels "ANY".
    """
    proper = set(m.group().lower() for m in _PROPER.finditer(query))
    if not proper:
        proper = set(w.lower() for w in _WORD.findall(query)
                     if len(w) > 3 and w.lower() not in
                     {"what", "when", "where", "which", "whose", "whom",
                      "name", "year", "date", "time", "place"})
    facets: list[Facet] = []
    if intent_conf is not None:
        for tp, conf in intent_conf.items():
            if tp in {"ANY", "WHAT_OBJ"}:
                continue
            if conf >= type_conf_threshold:
                facets.append(Facet(type_tag=tp, entity="<answer>", weight=conf))
    elif intent_type and intent_type != "ANY":
        facets.append(Facet(type_tag=intent_type, entity="<answer>"))
    for ent in proper:
        facets.append(Facet(type_tag="ANY", entity=ent))
    return facets


def atom_covers_facet(
    atom_text: str,
    atom_type: str,
    facet: Facet,
    atom_type_conf: dict[str, float] | None = None,
) -> float:
    """Coverage in [0, 1] with optional soft type confidence.

    facet.weight (query-side confidence) is applied by the caller so that
    coverage gains are scaled per facet — this function returns the raw
    atom-side coverage only.
    """
    if facet.entity == "<answer>":
        if atom_type_conf is not None:
            return 0.7 * atom_type_conf.get(facet.type_tag, 0.0)
        return 0.7 if atom_type == facet.type_tag else 0.0
    # entity facet
    if facet.entity not in atom_text.lower():
        return 0.0
    if facet.type_tag == "ANY":
        return 1.0
    if atom_type_conf is not None:
        return atom_type_conf.get(facet.type_tag, 0.0)
    return 1.0 if atom_type == facet.type_tag else 0.0


def submodular_select(
    *,
    atoms: list[dict],  # each {text, claim_type, atom_id, score, doc_id}
    facets: list[Facet],
    token_budget: int = 1024,
    score_floor: float = 0.0,
    score_bonus: float = 0.05,
    doc_diversity_bonus: float = 0.0,
    max_atoms_per_doc: int = 0,  # 0 = no cap
) -> list[dict]:
    """Greedy submodular maximization with diminishing-return coverage.

    With doc_diversity_bonus > 0, marginal gain is augmented for atoms
    whose doc_id is not yet represented in S (partition-matroid soft
    constraint). With max_atoms_per_doc > 0, hard cap per doc.
    """
    if not atoms or not facets:
        return atoms[:0]

    selected: list[dict] = []
    cov_so_far: dict[str, float] = {f.key(): 0.0 for f in facets}
    docs_seen: dict[str, int] = {}
    budget_used = 0

    while True:
        best_atom = None
        best_gain = 0.0
        best_idx = -1
        for i, a in enumerate(atoms):
            if a in selected:
                continue
            if a.get("score", 0.0) < score_floor:
                continue
            tok = a.get("token_count") or max(1, len(a["text"].split()))
            if budget_used + tok > token_budget:
                continue
            doc_id = a.get("doc_id", "")
            if max_atoms_per_doc > 0 and docs_seen.get(doc_id, 0) >= max_atoms_per_doc:
                continue
            gain = 0.0
            for f in facets:
                cov = atom_covers_facet(a["text"], a["claim_type"], f, a.get("claim_type_conf"))
                if cov <= 0:
                    continue
                old = cov_so_far[f.key()]
                new = old + (1 - old) * cov
                gain += f.weight * (new - old)
            sb = a.get("score", 0.0) * score_bonus
            div_bonus = doc_diversity_bonus if doc_id not in docs_seen else 0.0
            gain_per_tok = (gain + sb + div_bonus) / tok
            if gain_per_tok > best_gain:
                best_gain = gain_per_tok
                best_atom = a
                best_idx = i
        if best_atom is None or best_gain <= 0:
            break
        selected.append(best_atom)
        budget_used += max(1, len(best_atom["text"].split()))
        d = best_atom.get("doc_id", "")
        docs_seen[d] = docs_seen.get(d, 0) + 1
        for f in facets:
            cov = atom_covers_facet(best_atom["text"], best_atom["claim_type"], f,
                                    best_atom.get("claim_type_conf"))
            if cov > 0:
                old = cov_so_far[f.key()]
                cov_so_far[f.key()] = old + (1 - old) * cov

    # If budget allows more, fill with top-score atoms not yet selected
    sel_ids = {a.get("atom_id") for a in selected}
    for a in sorted([x for x in atoms if x.get("atom_id") not in sel_ids],
                    key=lambda x: -x.get("score", 0.0)):
        tok = max(1, len(a["text"].split()))
        if budget_used + tok > token_budget:
            continue
        selected.append(a)
        budget_used += tok

    return selected
