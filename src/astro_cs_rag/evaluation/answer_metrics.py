"""Answer-quality metrics — EM, token F1, citation accuracy.

EM and F1 follow the SQuAD/Rajpurkar normalization (lowercase, drop articles,
strip punctuation, collapse whitespace) so numbers are comparable to published
literature. Citation accuracy = fraction of cited chunks that map to a gold doc.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping


_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.UNICODE)
_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_WS = re.compile(r"\s+")


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = _PUNC.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _tokens(s: str) -> list[str]:
    return normalize_answer(s).split()


def exact_match(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(r) for r in references))


def token_f1(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    pt = _tokens(prediction)
    if not pt:
        return 0.0
    best = 0.0
    for r in references:
        rt = _tokens(r)
        if not rt:
            continue
        common = Counter(pt) & Counter(rt)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pt)
        recall = n_common / len(rt)
        f = 2 * precision * recall / (precision + recall)
        best = max(best, f)
    return best


def citation_accuracy(
    cited_chunk_ids: list[str],
    chunk_to_doc: Mapping[str, str],
    gold_doc_ids: list[str],
) -> float:
    if not cited_chunk_ids:
        return 0.0
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    hits = sum(1 for cid in cited_chunk_ids if chunk_to_doc.get(cid) in gold)
    return hits / len(cited_chunk_ids)


def aggregate_answer_metrics(
    rows: list[dict],
) -> dict[str, float]:
    """rows: each has em, f1, cite_acc. Return mean fields."""
    if not rows:
        return {
            "answer_em_mean": 0.0,
            "answer_f1_mean": 0.0,
            "citation_accuracy_mean": 0.0,
            "answer_count": 0.0,
        }
    n = float(len(rows))
    return {
        "answer_em_mean": sum(r["em"] for r in rows) / n,
        "answer_f1_mean": sum(r["f1"] for r in rows) / n,
        "citation_accuracy_mean": sum(r["cite_acc"] for r in rows) / n,
        "answer_count": n,
    }
