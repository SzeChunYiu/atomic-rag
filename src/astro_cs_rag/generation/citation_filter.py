"""B3 citation filter: drop hallucinated [E_i] citations.

Mechanism (from citation_hallucination_finding.md):
  55% of citation failures are hallucinated IDs — the LLM emits [E_i]
  tags that refer to atoms with zero content-token overlap with the
  answer. A trivial overlap filter recovers +4-12pp cit_acc at zero cost.

Usage:
    from astro_cs_rag.generation.citation_filter import filter_citations

    cited = filter_citations(answer_text, raw_cited_ids, atom_text_by_id)
"""
from __future__ import annotations

import re

_STOP = frozenset(
    "a an the of in on at to for from with by is are was were be been "
    "being and or but if then so than that this these those it its "
    "as which who whom whose what when where why how".split()
)
_TOK = re.compile(r"\b(?:[a-zA-Z][a-zA-Z\-']{2,}|\d+)\b")


def _content_tokens(s: str) -> frozenset[str]:
    return frozenset(t.lower() for t in _TOK.findall(s) if t.lower() not in _STOP)


def filter_citations(
    answer_text: str,
    cited_ids: list[str],
    evidence_texts: dict[str, str],
    min_overlap: int = 1,
) -> list[str]:
    """Return only cited_ids whose atom text shares >= min_overlap content
    tokens with answer_text. Preserves order; drops hallucinated citations.

    Pass min_overlap=0 to keep all (disables filter). Default=1 drops
    zero-overlap citations only — the cheapest safe threshold.
    """
    ans_tokens = _content_tokens(answer_text)
    if not ans_tokens or min_overlap <= 0:
        return cited_ids
    return [
        cid for cid in cited_ids
        if len(ans_tokens & _content_tokens(evidence_texts.get(cid, ""))) >= min_overlap
    ]
