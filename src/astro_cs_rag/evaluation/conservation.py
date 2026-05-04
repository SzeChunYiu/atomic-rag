"""Conservation-law faithfulness residuals for RAG answers.

In particle physics, every reconstructed event must satisfy energy-momentum
conservation; violations indicate missing particles or detector failure.
We define three computable residuals on (selected_evidence, answer):

  R_entity   = | { entities in answer } \\ { entities in evidence } | / |answer entities|
  R_numeric  = mean | answer_number - closest_evidence_number | / |answer_number|
  R_temporal = fraction of (date_a, date_b) pairs in the answer whose ordering
               disagrees with the evidence

Each residual is bounded in [0, 1]; 0 = perfect conservation. They serve as:
- a *faithfulness score* (1 - mean_R),
- an *abstention trigger* (mean_R > τ_abstain),
- a *re-retrieval trigger* (R_entity high but R_numeric low → missing evidence
  for some entity, retrieve more for it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_ENTITY = re.compile(r"\b[A-Z][a-zA-Z0-9'\-]*(?:\s+[A-Z][a-zA-Z0-9'\-]*){0,4}\b")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_DATE = re.compile(r"\b(?:\d{4}-\d{2}-\d{2}|\d{4})\b")


@dataclass(frozen=True)
class ConservationResiduals:
    R_entity: float
    R_numeric: float
    R_temporal: float

    @property
    def mean(self) -> float:
        return (self.R_entity + self.R_numeric + self.R_temporal) / 3.0

    @property
    def faithfulness(self) -> float:
        return max(0.0, 1.0 - self.mean)


def _entities(text: str) -> set[str]:
    return {m.group(0) for m in _ENTITY.finditer(text)}


def _numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUMBER.finditer(text):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def _dates(text: str) -> list[str]:
    return [m.group(0) for m in _DATE.finditer(text)]


def entity_residual(answer: str, evidence: str) -> float:
    a = _entities(answer)
    e = _entities(evidence)
    if not a:
        return 0.0
    return len(a - e) / len(a)


def numeric_residual(answer: str, evidence: str, *, tol_rel: float = 0.05) -> float:
    a_nums = _numbers(answer)
    e_nums = _numbers(evidence)
    if not a_nums:
        return 0.0
    if not e_nums:
        return 1.0
    misses = 0
    for x in a_nums:
        denom = max(abs(x), 1e-9)
        gap = min(abs(x - y) / denom for y in e_nums)
        if gap > tol_rel:
            misses += 1
    return misses / len(a_nums)


def temporal_residual(answer: str, evidence: str) -> float:
    a_dates = _dates(answer)
    e_dates = _dates(evidence)
    if len(a_dates) < 2:
        return 0.0
    pairs = [(a_dates[i], a_dates[j]) for i in range(len(a_dates)) for j in range(i + 1, len(a_dates))]
    if not pairs:
        return 0.0
    bad = 0
    for x, y in pairs:
        a_order = x <= y
        # Look for (x, y) co-occurrence in evidence in same order.
        e_text_xy = evidence.find(x) <= evidence.find(y) if x in evidence and y in evidence else None
        if e_text_xy is None:
            continue
        if a_order != e_text_xy:
            bad += 1
    return bad / len(pairs)


def conservation_residuals(answer: str, evidence_texts: list[str]) -> ConservationResiduals:
    evidence_blob = "\n".join(evidence_texts)
    return ConservationResiduals(
        R_entity=entity_residual(answer, evidence_blob),
        R_numeric=numeric_residual(answer, evidence_blob),
        R_temporal=temporal_residual(answer, evidence_blob),
    )
