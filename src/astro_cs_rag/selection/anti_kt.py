"""Anti-$k_T$ jet clustering of evidence atoms — IRC-safe selection.

Adapts the anti-$k_T$ algorithm of Cacciari, Salam, Soyez (JHEP 04:063, 2008)
to retrieval. Distance metric:

    d_ij  = min(s_i^{-2}, s_j^{-2}) * Δ_ij^2 / R^2
    d_iB  = s_i^{-2}

where:
    s_i  = relevance score of atom i (must be > 0; we shift if needed),
    Δ_ij = semantic distance in [0, 2] (1 - cosine_similarity),
    R    = jet radius (tunable; default 1.0 in cosine units).

At each step:
    1. find min(d_ij, d_iB) over all surviving atoms.
    2. if it is d_ij, merge i and j into a single proto-jet whose relevance is
       the sum of constituent relevances (analog of summed pT) and whose
       direction is the relevance-weighted mean of constituent embeddings,
       re-normalized.
    3. if it is d_iB, atom i becomes a final jet and is removed from the pool.

Two key invariances inherited from the algorithm (formalized in IRC theorem):
- Collinear safety: splitting a chunk into two co-located halves of relevance
  s/2 produces the same jets as keeping the chunk at relevance s.
- Infrared safety: adding atoms of arbitrarily small relevance does not change
  the leading jet.

These properties are why anti-$k_T$ outperforms MMR / DPP / greedy under
chunk-boundary perturbations and noisy candidate pools.

Complexity: O(N^2) per merge × N merges = O(N^3) — acceptable for N ≤ 200.
For larger N use the FastJet-style neighborhood reduction (TODO).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class JetAtom:
    atom_id: str
    relevance: float          # s_i
    embedding: np.ndarray     # (d,) L2-normalized
    member_atom_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class JetClusterResult:
    final_jets: list[JetAtom]              # ordered by descending relevance
    leading_atoms: list[str]               # member ids of the leading jet
    history: list[tuple[str, float]]       # ("merge:i+j", d_ij) | ("emit:i", d_iB)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _delta(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, 1.0 - _cosine_similarity(a, b))


def _shift_to_positive(atoms: list[JetAtom]) -> None:
    # Clip to a small positive floor rather than additive shift.
    # Additive shift preserves absolute score differences but violates the 1/s²
    # nonlinearity: shifting [0.1, -0.1] by +0.101 gives [0.201, 0.001] whose
    # beam-distance ratio is (0.001)^-2 / (0.201)^-2 ≈ 40000x, breaking IRC safety.
    # A floor clip preserves high-score ordering while bounding the distance range.
    for a in atoms:
        if a.relevance <= 1e-9:
            a.relevance = 1e-3


def _pair_distance(a: JetAtom, b: JetAtom, R: float) -> float:
    inv2 = min(1.0 / (a.relevance ** 2), 1.0 / (b.relevance ** 2))
    return inv2 * (_delta(a.embedding, b.embedding) ** 2) / max(R, 1e-9) ** 2


def _beam_distance(a: JetAtom) -> float:
    return 1.0 / (a.relevance ** 2)


def cluster_anti_kt(
    atom_ids: Sequence[str],
    relevances: Sequence[float],
    embeddings: np.ndarray,
    *,
    R: float = 1.0,
) -> JetClusterResult:
    """Run anti-$k_T$ clustering. Returns sorted final jets + history trace."""
    if len(atom_ids) != len(relevances) or len(atom_ids) != embeddings.shape[0]:
        msg = "atom_ids / relevances / embeddings must align"
        raise ValueError(msg)
    if not atom_ids:
        return JetClusterResult(final_jets=[], leading_atoms=[], history=[])

    pool: list[JetAtom] = []
    for cid, s, e in zip(atom_ids, relevances, embeddings, strict=True):
        v = e.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9)
        pool.append(
            JetAtom(
                atom_id=str(cid),
                relevance=float(s),
                embedding=v,
                member_atom_ids=[str(cid)],
            )
        )
    _shift_to_positive(pool)

    finalized: list[JetAtom] = []
    history: list[tuple[str, float]] = []

    # Vectorized clustering: maintain (relevances, embeddings) arrays and a
    # mask of live indices. d_ij computed via a single matmul per merge update.
    n0 = len(pool)
    relev = np.array([p.relevance for p in pool], dtype=np.float32)
    embs = np.stack([p.embedding for p in pool], axis=0).astype(np.float32)
    R2 = max(R, 1e-9) ** 2
    inv2 = 1.0 / (relev * relev + 1e-12)
    # Cosine similarity matrix (NxN); pair distance Δ_ij = max(0, 1 - cos).
    cos = embs @ embs.T
    delta = np.maximum(0.0, 1.0 - cos)
    # d_ij = min(inv2_i, inv2_j) * Δ²/R²
    pair_inv = np.minimum(inv2[:, None], inv2[None, :])
    d_pair = pair_inv * (delta * delta) / R2
    np.fill_diagonal(d_pair, np.inf)
    d_beam = inv2.copy()
    alive = np.ones(n0, dtype=bool)

    while alive.any():
        live_idx = np.flatnonzero(alive)
        # min over pair distances among live × live
        sub = d_pair[np.ix_(live_idx, live_idx)]
        # find global min between best d_pair and best d_beam
        flat_best = np.argmin(sub)
        i_loc, j_loc = divmod(int(flat_best), len(live_idx))
        best_pair_d = float(sub[i_loc, j_loc])
        beam_sub = d_beam[live_idx]
        i_beam_loc = int(np.argmin(beam_sub))
        best_beam_d = float(beam_sub[i_beam_loc])

        if best_beam_d <= best_pair_d:
            # Emit
            i_global = int(live_idx[i_beam_loc])
            atom = pool[i_global]
            finalized.append(JetAtom(
                atom_id=atom.atom_id, relevance=atom.relevance,
                embedding=atom.embedding, member_atom_ids=list(atom.member_atom_ids),
            ))
            history.append((f"emit:{atom.atom_id}", float(best_beam_d)))
            alive[i_global] = False
        else:
            i_global = int(live_idx[i_loc])
            j_global = int(live_idx[j_loc])
            a = pool[i_global]
            b = pool[j_global]
            new_rel = a.relevance + b.relevance
            mean = a.relevance * a.embedding + b.relevance * b.embedding
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            merged = JetAtom(
                atom_id=f"{a.atom_id}+{b.atom_id}",
                relevance=float(new_rel),
                embedding=mean.astype(np.float32),
                member_atom_ids=[*a.member_atom_ids, *b.member_atom_ids],
            )
            history.append((f"merge:{a.atom_id}+{b.atom_id}", float(best_pair_d)))
            # Mark j dead, replace i in-place with merged. Update column i of distance arrays.
            pool[i_global] = merged
            alive[j_global] = False
            relev[i_global] = new_rel
            embs[i_global] = merged.embedding
            new_inv2 = 1.0 / (new_rel * new_rel + 1e-12)
            inv2[i_global] = new_inv2
            d_beam[i_global] = new_inv2
            # Recompute distances FROM new i to all live atoms.
            cos_i = embs @ merged.embedding
            delta_i = np.maximum(0.0, 1.0 - cos_i)
            pair_inv_i = np.minimum(inv2, new_inv2)
            new_d = pair_inv_i * (delta_i * delta_i) / R2
            new_d[i_global] = np.inf
            d_pair[i_global, :] = new_d
            d_pair[:, i_global] = new_d
            # j is dead; mask its row/col with inf.
            d_pair[j_global, :] = np.inf
            d_pair[:, j_global] = np.inf
            d_beam[j_global] = np.inf

    finalized.sort(key=lambda x: -x.relevance)
    leading = list(finalized[0].member_atom_ids) if finalized else []
    return JetClusterResult(final_jets=finalized, leading_atoms=leading, history=history)


def select_evidence_via_jets(
    atom_ids: Sequence[str],
    relevances: Sequence[float],
    embeddings: np.ndarray,
    *,
    R: float = 1.0,
    n_jets: int = 1,
) -> list[str]:
    """Convenience wrapper: return the union of member ids in the top `n_jets`."""
    res = cluster_anti_kt(atom_ids, relevances, embeddings, R=R)
    keep: list[str] = []
    for j in res.final_jets[:n_jets]:
        keep.extend(j.member_atom_ids)
    seen: set[str] = set()
    out: list[str] = []
    for cid in keep:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)
    return out
