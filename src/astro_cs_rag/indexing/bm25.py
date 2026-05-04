"""Lexical BM25 over chunked corpus."""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from astro_cs_rag.atoms.schemas import Chunk


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.lower()) if t]


class BM25Index:
    def __init__(self, chunk_ids: list[str], tokenized: list[list[str]]) -> None:
        if len(chunk_ids) != len(tokenized):
            msg = "chunk_ids and tokenized length mismatch"
            raise ValueError(msg)
        self.chunk_ids = chunk_ids
        self.tokenized = tokenized
        self._bm25 = BM25Okapi(tokenized)

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> BM25Index:
        ids = [c.chunk_id for c in chunks]
        tok = [_tokenize(c.text) for c in chunks]
        if any(not t for t in tok):
            msg = "BM25 cannot index empty token lists — drop empty chunks upstream"
            raise ValueError(msg)
        return cls(ids, tok)

    def scores(self, query: str) -> dict[str, float]:
        q_tok = _tokenize(query)
        if not q_tok:
            return {}
        raw = self._bm25.get_scores(q_tok)
        return {self.chunk_ids[i]: float(raw[i]) for i in range(len(self.chunk_ids))}

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {"chunk_ids": self.chunk_ids, "tokens": self.tokenized}
        (directory / "bm25_tokens.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        raw = json.loads((directory / "bm25_tokens.json").read_text(encoding="utf-8"))
        return cls(list(raw["chunk_ids"]), [list(x) for x in raw["tokens"]])
