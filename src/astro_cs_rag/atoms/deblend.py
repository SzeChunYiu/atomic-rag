"""Sentence-level evidence atom extraction with claim-type tags.

Splits each chunk into sentences and tags each with a claim-type:
WHO, WHEN, WHERE, WHAT_NUM, WHAT_OBJ, ANY. Each atom carries its own
embedding so retrieval can match query intent to atom type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"\'\d])")
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_MONTH = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
    re.I,
)
_LOCATION_HINT = re.compile(
    r"\b(City|Country|State|Province|Republic|Kingdom|Federation|"
    r"County|District|Region|Town|Village|Borough)\b",
    re.I,
)
_LARGE_NUM = re.compile(r"\b\d{3,}\b|\b\d+(\.\d+)?\s?(km|kg|m|cm|%|miles|years)\b", re.I)
_PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b")  # 2-4 capitalized tokens

# A small list of country-like suffixes for WHERE detection.
_COUNTRY_LIST = {
    "France", "Germany", "Italy", "Spain", "Russia", "China", "Japan",
    "India", "Brazil", "Argentina", "Canada", "Mexico", "Australia",
    "Sweden", "Norway", "Finland", "Denmark", "Poland", "Greece",
    "Portugal", "Netherlands", "Belgium", "Austria", "Switzerland",
    "Turkey", "Egypt", "Iran", "Iraq", "Israel", "Pakistan", "Bangladesh",
    "Indonesia", "Vietnam", "Thailand", "Malaysia", "Philippines", "Korea",
    "Ukraine", "Romania", "Hungary", "Czechia", "Slovakia", "Bulgaria",
    "Ireland", "Iceland", "Mongolia", "Kazakhstan", "Uzbekistan",
}


_CLAIM_TYPES = ["WHEN", "WHERE", "WHAT_NUM", "WHO", "WHAT_OBJ"]


@dataclass(frozen=True)
class EvidenceAtomV2:
    atom_id: str
    chunk_id: str
    doc_id: str
    text: str
    claim_type: str
    span_start: int
    span_end: int
    claim_type_conf: dict = field(default_factory=dict)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences with character spans (start, end)."""
    out: list[tuple[str, int, int]] = []
    cursor = 0
    pieces = _SENT_SPLIT.split(text)
    for p in pieces:
        p_strip = p.strip()
        if not p_strip:
            cursor += len(p)
            continue
        # find piece in original text starting from cursor for span tracking
        idx = text.find(p_strip, cursor)
        if idx < 0:
            idx = cursor
        end = idx + len(p_strip)
        out.append((p_strip, idx, end))
        cursor = end
    return out


def type_tag_conf(text: str) -> dict[str, float]:
    """Soft claim-type distribution over _CLAIM_TYPES, summing to 1.0.

    Each signal adds independent evidence; WHAT_OBJ holds a base prior.
    Preserves multi-type signal that the old hard tagger discarded.
    """
    t = text.strip()
    if not t:
        u = 1.0 / len(_CLAIM_TYPES)
        return {tp: u for tp in _CLAIM_TYPES}
    raw: dict[str, float] = {tp: 0.0 for tp in _CLAIM_TYPES}
    raw["WHAT_OBJ"] = 0.15
    if _YEAR.search(t) or _MONTH.search(t):
        raw["WHEN"] += 0.80
    if _LOCATION_HINT.search(t) or any(c in t for c in _COUNTRY_LIST):
        raw["WHERE"] += 0.70
    if _LARGE_NUM.search(t):
        raw["WHAT_NUM"] += 0.55
    if _PROPER.search(t):
        raw["WHO"] += 0.40
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def type_tag(text: str) -> str:
    """Hard claim-type label — argmax of type_tag_conf."""
    conf = type_tag_conf(text)
    return max(conf, key=conf.__getitem__)


def query_intent_conf(text: str) -> dict[str, float]:
    """Soft query answer-type distribution, summing to 1.0.

    Strong keyword matches give high confidence; weaker/implicit signals
    give partial confidence. Residual mass stays on ANY.
    """
    t = text.lower()
    raw: dict[str, float] = {tp: 0.0 for tp in _CLAIM_TYPES}
    raw["ANY"] = 0.10  # base residual

    # Strong signals
    if (any(k in t for k in (" when ", "what year", "what date", "in what year",
                              "in which year", "what time"))
            or t.startswith("when ")):
        raw["WHEN"] += 0.90
    elif any(k in t for k in ("founded in", "born in", "died in", "established in",
                               "since ", "until ", "during ")):
        raw["WHEN"] += 0.55

    if (any(k in t for k in (" who ", "whose ", "by whom", "what is the name",
                              "who is", "which person", "which scientist",
                              "which author", "which director"))
            or t.startswith("who ")):
        raw["WHO"] += 0.85
    elif any(k in t for k in ("inventor", "founder", "author", "creator",
                               "wrote ", "discovered ")):
        raw["WHO"] += 0.45

    if (any(k in t for k in (" where ", "what country", "what city", "what state",
                              "located in", "headquarters", "capital of",
                              "based in", "native to", "come from"))
            or t.startswith("where ")):
        raw["WHERE"] += 0.85
    elif any(k in t for k in ("country", "city", "region", "province",
                               "continent", "ocean", "river", "lake")):
        raw["WHERE"] += 0.40

    if any(k in t for k in ("how many", "how much", "what is the number",
                             "how old", "how long", "how far", "how tall",
                             "what is the population", "how large")):
        raw["WHAT_NUM"] += 0.85

    # Residual WHAT_OBJ prior for all other queries
    raw["WHAT_OBJ"] = max(raw["WHAT_OBJ"], 0.05)

    # Ensure "ANY" type is in the output for backward compat
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def query_intent(text: str) -> str:
    """Hard answer-type label — argmax of query_intent_conf (excludes ANY)."""
    conf = query_intent_conf(text)
    typed = {k: v for k, v in conf.items() if k != "ANY"}
    best_type = max(typed, key=typed.__getitem__)
    # Return "ANY" only if the top typed signal is weak (dominated by ANY residual)
    if conf["ANY"] >= typed[best_type]:
        return "ANY"
    return best_type


def deblend_chunk(
    *,
    chunk_id: str,
    doc_id: str,
    chunk_text: str,
) -> list[EvidenceAtomV2]:
    """Split one chunk into typed atoms (no embedding here)."""
    out: list[EvidenceAtomV2] = []
    for i, (sent, start, end) in enumerate(split_sentences(chunk_text)):
        if len(sent) < 4:
            continue
        out.append(
            EvidenceAtomV2(
                atom_id=f"{chunk_id}::s{i}",
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=sent,
                claim_type=type_tag(sent),
                claim_type_conf=type_tag_conf(sent),
                span_start=start,
                span_end=end,
            )
        )
    return out
