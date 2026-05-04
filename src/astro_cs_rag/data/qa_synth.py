"""LLM-driven QA synthesis from a corpus of physics abstracts.

Pipeline:
  1. For each abstract, ask the generator to produce N short factoid questions
     and the *exact span* of the abstract that answers each.
  2. Verify span-grounding: the answer span must occur as a contiguous substring
     of the abstract; otherwise reject.
  3. Optionally validate by re-asking the generator: given (question, abstract),
     does it produce an answer that contains the gold span?

This is the standard QA-from-passage recipe used in Natural Questions and
QuAC pipelines, adapted for short abstracts.

Quality is the user's responsibility: generated QAs require a manual sweep
before they enter a final paper benchmark. We keep the size small (~150 per
corpus) so manual review is cheap.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from astro_cs_rag.config.schema import GeneratorSettings
from astro_cs_rag.generation.generator import build_generator


_QA_SYSTEM = (
    "You are a physics PhD writing benchmark questions for a retrieval system.\n"
    "From the abstract, write up to 3 short factoid questions whose answer is\n"
    "a single word or noun-phrase appearing verbatim in the abstract.\n"
    "Output strict JSON: a list of {\"question\": ..., \"answer_span\": ...}.\n"
    "Do not invent facts not in the abstract. If you cannot find one, return []."
)


@dataclass(frozen=True)
class QAItem:
    query_id: str
    text: str
    gold_doc_id: str
    answer: str


def _build_user_prompt(abstract: str) -> str:
    return f"Abstract:\n{abstract.strip()}\n\nReturn JSON list now:"


def _extract_first_json_array(text: str) -> list[dict] | None:
    m = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        out = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(out, list):
        return None
    cleaned: list[dict] = []
    for item in out:
        if isinstance(item, dict) and "question" in item and "answer_span" in item:
            cleaned.append(
                {
                    "question": str(item["question"]).strip(),
                    "answer_span": str(item["answer_span"]).strip(),
                }
            )
    return cleaned


def synthesize_qa(
    corpus_rows: list[dict],
    *,
    generator_settings: GeneratorSettings,
    target_n: int = 150,
    max_per_doc: int = 3,
    seed: int = 0,
) -> list[QAItem]:
    if not corpus_rows:
        return []
    gen = build_generator(generator_settings)

    import random

    rng = random.Random(seed)
    docs = list(corpus_rows)
    rng.shuffle(docs)

    out: list[QAItem] = []
    for doc in docs:
        if len(out) >= target_n:
            break
        abstract = str(doc.get("text") or "").strip()
        doc_id = str(doc["doc_id"])
        if len(abstract) < 80:
            continue
        prompt = _build_user_prompt(abstract)
        ans = gen.answer(query_id=doc_id, query_text=prompt, evidence=[])
        items = _extract_first_json_array(ans.answer_text)
        if not items:
            continue
        kept = 0
        for it in items[:max_per_doc]:
            q = it["question"]
            span = it["answer_span"]
            if not q or not span:
                continue
            if span.lower() not in abstract.lower():
                continue
            qid = f"qa_{len(out):05d}"
            out.append(
                QAItem(
                    query_id=qid,
                    text=q,
                    gold_doc_id=doc_id,
                    answer=span,
                )
            )
            kept += 1
            if len(out) >= target_n:
                break
        _ = kept
    return out


def write_qa_jsonl(items: list[QAItem], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    queries_path = out_dir / "queries.jsonl"
    gold_path = out_dir / "gold.jsonl"
    with queries_path.open("w", encoding="utf-8") as fq, gold_path.open("w", encoding="utf-8") as fg:
        for it in items:
            fq.write(
                json.dumps(
                    {
                        "query_id": it.query_id,
                        "text": it.text,
                        "gold_doc_ids": [it.gold_doc_id],
                        "metadata": {"answer": [it.answer], "source": "qa_synth"},
                    },
                    ensure_ascii=False,
                )
            )
            fq.write("\n")
            fg.write(
                json.dumps({"query_id": it.query_id, "gold_doc_ids": [it.gold_doc_id]})
            )
            fg.write("\n")
    return {"queries": queries_path, "gold": gold_path}
