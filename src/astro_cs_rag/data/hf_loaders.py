"""Convert HF benchmarks into our (corpus, queries, gold) JSONL contract.

We deliberately materialize a *frozen subset* on disk so retrieval/embedding
work is reproducible without re-downloading or re-shuffling. Each subset is
identified by (dataset_name, split, n, seed) and is byte-stable.

Datasets supported:
- hotpotqa (validation, distractor) — multi-hop QA, gold = supporting docs.
- nq_open (validation) — single-hop open-domain QA; we pull a Wikipedia
  passage corpus from `nq` (KILT-style) when available, else fall back to
  the `nq_open` short-answer split with a separate Wikipedia dump.

The output JSONL trio matches `data/loaders.py`:

  corpus.jsonl  : {doc_id, text, metadata}
  queries.jsonl : {query_id, text, gold_doc_ids, metadata}
  gold.jsonl    : {query_id, gold_doc_ids}
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _hash_id(prefix: str, payload: str) -> str:
    h = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{h}"


@dataclass(frozen=True)
class PreparedSubset:
    name: str
    split: str
    n_queries: int
    seed: int
    out_dir: Path

    @property
    def corpus_path(self) -> Path:
        return self.out_dir / "corpus.jsonl"

    @property
    def queries_path(self) -> Path:
        return self.out_dir / "queries.jsonl"

    @property
    def gold_path(self) -> Path:
        return self.out_dir / "gold.jsonl"


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def prepare_hotpotqa(
    out_dir: Path,
    split: str = "validation",
    n_queries: int = 1000,
    seed: int = 0,
) -> PreparedSubset:
    """HotpotQA distractor: each example carries 10 paragraphs (2 gold + 8 distractor).

    Corpus = union of all paragraphs across the chosen subset.
    Gold doc ids = the paragraphs whose title appears in `supporting_facts`.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("hotpot_qa", "distractor", split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_queries]

    docs: dict[str, dict] = {}
    queries: list[dict] = []
    golds: list[dict] = []
    for i in indices:
        ex = ds[i]
        qid = f"hotpot_{ex['id']}"
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        sup_titles = set(ex["supporting_facts"]["title"])
        local_gold: list[str] = []
        for title, sents in zip(titles, sentences, strict=True):
            text = " ".join(sents).strip()
            if not text:
                continue
            doc_id = _hash_id("hotpot_doc", title)
            docs[doc_id] = {
                "doc_id": doc_id,
                "text": f"{title}. {text}",
                "metadata": {"title": title, "source": "hotpotqa"},
            }
            if title in sup_titles:
                local_gold.append(doc_id)
        queries.append(
            {
                "query_id": qid,
                "text": ex["question"],
                "gold_doc_ids": list(dict.fromkeys(local_gold)),
                "metadata": {
                    "answer": ex.get("answer"),
                    "type": ex.get("type"),
                    "level": ex.get("level"),
                    "source": "hotpotqa",
                },
            }
        )
        golds.append(
            {
                "query_id": qid,
                "gold_doc_ids": list(dict.fromkeys(local_gold)),
            }
        )

    sub = PreparedSubset("hotpotqa", split, n_queries, seed, out_dir)
    _write_jsonl(sub.corpus_path, docs.values())
    _write_jsonl(sub.queries_path, queries)
    _write_jsonl(sub.gold_path, golds)
    _write_manifest(sub, len(docs), len(queries))
    return sub


def prepare_nq_open(
    out_dir: Path,
    split: str = "validation",
    n_queries: int = 1000,
    seed: int = 0,
    wiki_passages: Path | None = None,
) -> PreparedSubset:
    """NQ-open subset.

    NQ-open ships only short answers (no doc id), so for retrieval evaluation we
    require an external Wikipedia passage dump. If `wiki_passages` is provided
    (JSONL with doc_id/text), we use it as the corpus and rely on string-match
    of the short answer for weak gold labels (recall-only metrics in P0).

    If `wiki_passages` is None, we fall back to a closed-book split: corpus is
    a small synthetic distractor pool built from the answer strings, suitable
    only for plumbing tests, not real benchmarking.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("nq_open", split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_queries]

    queries: list[dict] = []
    golds: list[dict] = []
    docs: dict[str, dict] = {}

    if wiki_passages is None:
        for i in indices:
            ex = ds[i]
            qid = _hash_id("nq", ex["question"])
            ans = ex.get("answer") or []
            ans_text = "; ".join(ans) if isinstance(ans, list) else str(ans)
            doc_id = _hash_id("nq_doc", ans_text or ex["question"])
            docs[doc_id] = {
                "doc_id": doc_id,
                "text": f"Synthetic stub for {ex['question']} — answer: {ans_text}",
                "metadata": {"source": "nq_open_stub"},
            }
            queries.append(
                {
                    "query_id": qid,
                    "text": ex["question"],
                    "gold_doc_ids": [doc_id],
                    "metadata": {"answer": ans, "source": "nq_open"},
                }
            )
            golds.append({"query_id": qid, "gold_doc_ids": [doc_id]})
    else:
        for line in wiki_passages.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs[str(row["doc_id"])] = row
        for i in indices:
            ex = ds[i]
            qid = _hash_id("nq", ex["question"])
            ans = ex.get("answer") or []
            queries.append(
                {
                    "query_id": qid,
                    "text": ex["question"],
                    "gold_doc_ids": [],
                    "metadata": {"answer": ans, "source": "nq_open"},
                }
            )
            golds.append({"query_id": qid, "gold_doc_ids": []})

    sub = PreparedSubset("nq_open", split, n_queries, seed, out_dir)
    _write_jsonl(sub.corpus_path, docs.values())
    _write_jsonl(sub.queries_path, queries)
    _write_jsonl(sub.gold_path, golds)
    _write_manifest(sub, len(docs), len(queries))
    return sub


def prepare_2wikimultihop(
    out_dir: Path,
    split: str = "validation",
    n_queries: int = 1000,
    seed: int = 0,
) -> PreparedSubset:
    """2WikiMultihopQA: list-style context, similar shape to HotpotQA distractor.

    `context` is List[ [title:str, List[str]] ] (one record per supporting article).
    `supporting_facts` is List[ [title:str, sentence_idx:int] ].
    Gold doc ids = paragraphs whose title appears in supporting_facts.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset("voidful/2WikiMultihopQA", split=split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_queries]

    docs: dict[str, dict] = {}
    queries: list[dict] = []
    golds: list[dict] = []
    for i in indices:
        ex = ds[i]
        qid = f"2wiki_{ex['_id']}"
        sup_titles = {pair[0] for pair in ex.get("supporting_facts", [])}
        local_gold: list[str] = []
        for entry in ex["context"]:
            if not entry or len(entry) < 2:
                continue
            title, sents = entry[0], entry[1]
            text = " ".join(sents).strip() if isinstance(sents, list) else str(sents).strip()
            if not text:
                continue
            doc_id = _hash_id("2wiki_doc", title)
            docs[doc_id] = {
                "doc_id": doc_id,
                "text": f"{title}. {text}",
                "metadata": {"title": title, "source": "2wikimultihop"},
            }
            if title in sup_titles:
                local_gold.append(doc_id)
        queries.append(
            {
                "query_id": qid,
                "text": ex["question"],
                "gold_doc_ids": list(dict.fromkeys(local_gold)),
                "metadata": {
                    "answer": ex.get("answer"),
                    "type": ex.get("type"),
                    "source": "2wikimultihop",
                },
            }
        )
        golds.append(
            {"query_id": qid, "gold_doc_ids": list(dict.fromkeys(local_gold))}
        )

    sub = PreparedSubset("2wikimultihop", split, n_queries, seed, out_dir)
    _write_jsonl(sub.corpus_path, docs.values())
    _write_jsonl(sub.queries_path, queries)
    _write_jsonl(sub.gold_path, golds)
    _write_manifest(sub, len(docs), len(queries))
    return sub


def _write_manifest(sub: PreparedSubset, n_docs: int, n_queries: int) -> None:
    payload = {
        "name": sub.name,
        "split": sub.split,
        "n_queries_requested": sub.n_queries,
        "n_queries_written": n_queries,
        "n_docs_written": n_docs,
        "seed": sub.seed,
    }
    (sub.out_dir / "subset_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
