"""Chunk → sentence → claim atom decomposition.

Pure-Python regex fallback (no spaCy dep) so CI runs anywhere; spaCy plug-in
slot is left at `_extract_with_spacy` if/when we add it later.

The decomposer is deliberately *over-eager*: it favors recall of candidate
atoms, leaving the eventual selection (Phase 2 onwards) to the downstream
selector. This matches the project's atomic discipline — see one atom, decide
explicitly to keep or drop.
"""

from __future__ import annotations

import hashlib
import re

from astro_cs_rag.atoms.schemas import Chunk, ClaimAtom


_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# Capitalized-multi-word entities (Crab Nebula, NASA, Dark Energy Survey).
_ENTITY = re.compile(r"\b(?:[A-Z][a-zA-Z0-9'\-]*)(?:\s+[A-Z][a-zA-Z0-9'\-]*){0,4}\b")

# Numbers including units, scientific notation, percentages.
_NUMBER = re.compile(
    r"(?<![A-Za-z])"
    r"(?:[+-]?\d{1,3}(?:[,]\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"(?:\s?[%a-zA-Zµ°][a-zA-Z/^\-²³]*)?"
)

_DATE = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}|\d{4})\b"
)

_STOP_ENTITIES = {
    "The",
    "A",
    "An",
    "It",
    "They",
    "These",
    "This",
    "That",
    "There",
    "When",
    "Where",
    "Who",
    "What",
    "Why",
    "How",
}


def _atom_id(chunk_id: str, sent_index: int, span_start: int) -> str:
    h = hashlib.sha1(f"{chunk_id}:{sent_index}:{span_start}".encode()).hexdigest()[:10]
    return f"a_{h}"


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    if not text.strip():
        return []
    out: list[tuple[int, int, str]] = []
    start = 0
    pieces = _SENT_BOUNDARY.split(text)
    cursor = 0
    for piece in pieces:
        if not piece.strip():
            cursor += len(piece)
            continue
        s = text.find(piece, cursor)
        if s == -1:
            s = cursor
        e = s + len(piece)
        out.append((s, e, piece.strip()))
        cursor = e
    if not out:
        out.append((0, len(text), text.strip()))
    _ = start
    return out


def _extract_entities(text: str) -> list[str]:
    found = _ENTITY.findall(text)
    out: list[str] = []
    seen: set[str] = set()
    for raw in found:
        e = raw.strip()
        if not e or e in _STOP_ENTITIES:
            continue
        if e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def _extract_numbers(text: str) -> list[str]:
    return [m.group(0).strip() for m in _NUMBER.finditer(text)]


def _extract_dates(text: str) -> list[str]:
    return [m.group(0) for m in _DATE.finditer(text)]


def decompose_chunk(chunk: Chunk) -> list[ClaimAtom]:
    sentences = _split_sentences(chunk.text)
    atoms: list[ClaimAtom] = []
    for i, (rel_start, rel_end, sent_text) in enumerate(sentences):
        if not sent_text:
            continue
        ents = _extract_entities(sent_text)
        nums = _extract_numbers(sent_text)
        dates = _extract_dates(sent_text)
        atom = ClaimAtom(
            atom_id=_atom_id(chunk.chunk_id, i, rel_start),
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            sent_index=i,
            span_start=chunk.start_char + rel_start,
            span_end=chunk.start_char + rel_end,
            text=sent_text,
            entities=ents,
            numbers=nums,
            dates=dates,
            token_count=len(sent_text.split()),
            metadata={"backend": "regex_v1"},
        )
        atoms.append(atom)
    return atoms


def decompose_chunks(chunks: list[Chunk]) -> list[ClaimAtom]:
    out: list[ClaimAtom] = []
    for c in chunks:
        out.extend(decompose_chunk(c))
    return out
