"""Load corpus / queries / gold labels — no retrieval or embedding here."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from astro_cs_rag.atoms.schemas import Document, Query


class GoldRow(BaseModel):
    query_id: str
    gold_doc_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class DatasetBundle:
    documents: list[Document]
    queries: list[Query]


def _normalize_doc_id(raw: str) -> str:
    return raw.strip()


def load_corpus_jsonl(path: Path) -> list[Document]:
    docs: list[Document] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        doc_id = _normalize_doc_id(str(row["doc_id"]))
        text = str(row["text"])
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict):
            msg = f"{path}: metadata must be object for doc {doc_id}"
            raise ValueError(msg)
        docs.append(Document(doc_id=doc_id, text=text, metadata=meta))
    return docs


def load_queries_jsonl(path: Path) -> list[Query]:
    queries: list[Query] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        qid = str(row["query_id"]).strip()
        text = str(row["text"])
        gold = row.get("gold_doc_ids") or []
        if gold and not isinstance(gold, list):
            msg = f"{path}: gold_doc_ids must be a list for query {qid}"
            raise ValueError(msg)
        gold_ids = [_normalize_doc_id(str(x)) for x in gold]
        meta = row.get("metadata") or {}
        if not isinstance(meta, dict):
            msg = f"{path}: metadata must be object for query {qid}"
            raise ValueError(msg)
        queries.append(
            Query(query_id=qid, text=text, gold_doc_ids=gold_ids, metadata=meta)
        )
    return queries


def load_gold_jsonl(path: Path) -> dict[str, list[str]]:
    gold_map: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = GoldRow.model_validate(json.loads(line))
        gold_map[row.query_id] = list(row.gold_doc_ids)
    return gold_map


def merge_gold_into_queries(
    queries: list[Query], gold_map: dict[str, list[str]]
) -> list[Query]:
    out: list[Query] = []
    for q in queries:
        extra = gold_map.get(q.query_id)
        if extra:
            merged_ids = list(dict.fromkeys([*q.gold_doc_ids, *extra]))
            out.append(q.model_copy(update={"gold_doc_ids": merged_ids}))
        else:
            out.append(q)
    return out


def write_dataset_manifest(path: Path, bundle: DatasetBundle, name: str) -> None:
    payload = {
        "dataset_name": name,
        "corpus_doc_count": len(bundle.documents),
        "query_count": len(bundle.queries),
        "doc_ids": [d.doc_id for d in bundle.documents],
        "query_ids": [q.query_id for q in bundle.queries],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
