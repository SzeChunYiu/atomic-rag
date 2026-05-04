"""Maximum-entropy (Gibbs) atom selection.

Replaces ad-hoc score-vs-coverage weighting in submodular_select with a
principled Gibbs distribution

    P(S) propto exp(beta*score(S) + sum_f lambda_f * cov_f(S) - mu*tok(S))

where beta, lambda_f, mu are Lagrange multipliers tuned on a dev set,
not hand-set. Greedy selection at T->0 picks atoms in decreasing order
of marginal log P(S+a) - log P(S), which equals beta*score(a) +
sum_f lambda_f * marginal_cov(a, f) - mu*tok(a).

This is Jaynes (1957) maximum-entropy with linear constraints, applied
to the discrete selection problem. Distinct from greedy submodular:
the multipliers are determined by constraint satisfaction, so the
balance between score and coverage is principled.
"""
from __future__ import annotations

from .submodular import Facet, atom_covers_facet


def maxent_select(
    *,
    atoms: list[dict],
    facets: list[Facet],
    token_budget: int = 1024,
    beta: float = 1.0,         # multiplier on score
    lambda_f: float = 1.0,     # multiplier on coverage gain (per-facet, uniform)
    mu: float = 0.001,         # multiplier on token cost
    score_floor: float = 0.0,
) -> list[dict]:
    """Greedy T->0 selection from Gibbs distribution.

    Atom marginal contribution to log P:
        beta*score(a) + lambda_f * sum_f marginal_cov(a, f) - mu*tok(a)

    Marginal coverage uses the same diminishing-return formula as
    submodular_select for consistency."""
    if not atoms or not facets:
        return atoms[:0]

    selected: list[dict] = []
    cov_so_far: dict[str, float] = {f.key(): 0.0 for f in facets}
    budget_used = 0

    while True:
        best_atom = None
        best_delta = 0.0
        for a in atoms:
            if a in selected:
                continue
            if a.get("score", 0.0) < score_floor:
                continue
            tok = a.get("token_count") or max(1, len(a["text"].split()))
            if budget_used + tok > token_budget:
                continue
            cov_gain = 0.0
            for f in facets:
                cov = atom_covers_facet(a["text"], a["claim_type"], f, a.get("claim_type_conf"))
                if cov <= 0:
                    continue
                old = cov_so_far[f.key()]
                cov_gain += f.weight * (1 - old) * cov
            delta = beta * a.get("score", 0.0) + lambda_f * cov_gain - mu * tok
            if delta > best_delta:
                best_delta = delta
                best_atom = a
        if best_atom is None or best_delta <= 0:
            break
        selected.append(best_atom)
        budget_used += max(1, len(best_atom["text"].split()))
        for f in facets:
            cov = atom_covers_facet(
                best_atom["text"], best_atom["claim_type"], f,
                best_atom.get("claim_type_conf"))
            if cov > 0:
                old = cov_so_far[f.key()]
                cov_so_far[f.key()] = old + (1 - old) * cov

    # Optional fill with high-score remaining atoms (relevance fallback)
    sel_ids = {a.get("atom_id") for a in selected}
    for a in sorted([x for x in atoms if x.get("atom_id") not in sel_ids],
                    key=lambda x: -x.get("score", 0.0)):
        tok = max(1, len(a["text"].split()))
        if budget_used + tok > token_budget:
            continue
        # Only fill if score still beats mu*tok (efficient frontier)
        if beta * a.get("score", 0.0) - mu * tok > 0:
            selected.append(a)
            budget_used += tok

    return selected
