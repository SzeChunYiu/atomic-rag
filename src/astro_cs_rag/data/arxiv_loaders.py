"""Materialize physics arXiv abstract subsets (astro-ph, hep-ph) as our JSONL trio.

Network IO is deliberately encapsulated; offline runs use a cached snapshot.

We pull abstract metadata from the arXiv OAI-PMH endpoint (no API key needed)
in chunks of 1000. Subsetting and seed-based shuffling happens locally.

Question generation is *not* done in this module — it is a separate step
(`qa_synth.py`) that consumes the materialized corpus and emits a
queries/gold pair.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

OAI_BASE = "http://export.arxiv.org/oai2"
OAI_NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "arxiv": "http://arxiv.org/OAI/arXiv/"}


@dataclass(frozen=True)
class ArxivSubset:
    name: str
    n_docs: int
    out_dir: Path

    @property
    def corpus_path(self) -> Path:
        return self.out_dir / "corpus.jsonl"


def _hash_id(prefix: str, payload: str) -> str:
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


def _fetch_oai_chunk(set_spec: str, resumption: str | None) -> tuple[list[dict], str | None]:
    """Fetch one OAI-PMH page; return (rows, resumption_token)."""
    if resumption:
        params = {"verb": "ListRecords", "resumptionToken": resumption}
    else:
        params = {
            "verb": "ListRecords",
            "metadataPrefix": "arXiv",
            "set": set_spec,
        }
    url = f"{OAI_BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "astro-cs-rag/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read()
    root = ET.fromstring(body)
    rows: list[dict] = []
    for record in root.iter("{http://www.openarchives.org/OAI/2.0/}record"):
        meta = record.find(".//{http://arxiv.org/OAI/arXiv/}arXiv")
        if meta is None:
            continue
        ident = (meta.findtext("{http://arxiv.org/OAI/arXiv/}id") or "").strip()
        title = (meta.findtext("{http://arxiv.org/OAI/arXiv/}title") or "").strip().replace("\n", " ")
        abstract = (meta.findtext("{http://arxiv.org/OAI/arXiv/}abstract") or "").strip().replace("\n", " ")
        categories = (meta.findtext("{http://arxiv.org/OAI/arXiv/}categories") or "").strip()
        if not abstract:
            continue
        rows.append(
            {
                "arxiv_id": ident,
                "title": title,
                "abstract": abstract,
                "categories": categories,
            }
        )
    token_el = root.find(".//{http://www.openarchives.org/OAI/2.0/}resumptionToken")
    token = token_el.text.strip() if token_el is not None and token_el.text else None
    return rows, token


def fetch_arxiv_subset(
    *,
    out_dir: Path,
    set_spec: str = "physics:astro-ph",
    n_docs: int = 5000,
    seed: int = 0,
    sleep_s: float = 1.0,
) -> ArxivSubset:
    """Pull up to `n_docs` abstracts from the given OAI set into corpus.jsonl.

    Note: arXiv asks clients to throttle to <=1 req/s; we sleep `sleep_s`
    between requests by default.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    token: str | None = None
    while len(rows) < n_docs:
        chunk, token = _fetch_oai_chunk(set_spec, token)
        if not chunk:
            break
        rows.extend(chunk)
        if not token:
            break
        time.sleep(sleep_s)
    rows = rows[:n_docs]
    import random

    rng = random.Random(seed)
    rng.shuffle(rows)

    corpus_path = out_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as f:
        for r in rows:
            doc_id = _hash_id("arx", r["arxiv_id"])
            text = f"{r['title']}.\n{r['abstract']}"
            f.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "text": text,
                        "metadata": {
                            "arxiv_id": r["arxiv_id"],
                            "title": r["title"],
                            "categories": r["categories"],
                            "set_spec": set_spec,
                        },
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")

    sub = ArxivSubset(name=set_spec, n_docs=len(rows), out_dir=out_dir)
    (out_dir / "subset_manifest.json").write_text(
        json.dumps(
            {
                "set_spec": set_spec,
                "n_docs": len(rows),
                "seed": seed,
                "source": "arxiv_oai",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return sub


def load_corpus_for_qa(corpus_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows
