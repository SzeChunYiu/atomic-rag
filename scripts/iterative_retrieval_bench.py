"""2-step iterative retrieval benchmark for multi-hop QA (IRCoT-lite).

Step 1: Retrieve top-k for original query -> generate CoT intermediate answer.
Step 2: Extract intermediate claim from CoT -> re-retrieve -> augment context.
Step 3: Generate final answer with combined context.

Targets 2Wiki bridge-coverage gap: only 47.8% of 2Wiki queries have both
gold docs in single-pass selection. Intermediate reasoning drives a second
retrieval to recover the missing hop.

Usage:
    python scripts/iterative_retrieval_bench.py \
        --queries data/2wiki_1k/queries.jsonl \
        --corpus  data/2wiki_1k/corpus.jsonl \
        --index-dir runs/2wiki/index_bundle \
        --out-dir runs/breakthrough/2wiki_iterative \
        --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_CITE_RE = re.compile(r"\[E\d+\]")
_FINAL_ANS_RE = re.compile(r"Final answer:\s*(.+)", re.IGNORECASE | re.DOTALL)
_INTER_RE = re.compile(r"Intermediate:\s*(.+?)(?:\n|Final answer:|$)", re.IGNORECASE)
_REASONING_RE = re.compile(r"Reasoning:\s*(.+?)(?:Final answer:|$)", re.IGNORECASE | re.DOTALL)
_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")

STEP1_SYSTEM = (
    "Answer concisely with citations. Show your reasoning and give a final answer.\n"
    "Format:\nReasoning: <one or two steps>\nIntermediate: <key fact needed for the next step>\n"
    "Final answer: <short answer, 1-5 words>"
)

STEP2_SYSTEM = (
    "Answer concisely with citations. Use all provided evidence.\n"
    "Format:\nReasoning: <one or two steps>\nFinal answer: <short answer, 1-5 words>"
)

USER_TMPL = "Question: {query}\n\nEvidence:\n{evidence}"


def norm(s: str) -> str:
    s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s)
    return _WS.sub(" ", s).strip()


def extract_final(text: str) -> str:
    m = _FINAL_ANS_RE.search(text)
    return norm(_CITE_RE.sub("", m.group(1) if m else text).strip())


def extract_intermediate(text: str) -> str:
    m = _INTER_RE.search(text)
    if m:
        return _CITE_RE.sub("", m.group(1)).strip()
    m2 = _REASONING_RE.search(text)
    if m2:
        return _CITE_RE.sub("", m2.group(1)).strip()[:200]
    return _CITE_RE.sub("", text).strip()[:200]


def token_f1(pred: str, refs: list[str]) -> float:
    pt = norm(pred).split()
    if not pt:
        return 0.0
    best = 0.0
    for r in refs:
        rt = norm(r).split()
        if not rt:
            continue
        common = Counter(pt) & Counter(rt)
        nc = sum(common.values())
        if nc == 0:
            continue
        p, rec = nc / len(pt), nc / len(rt)
        best = max(best, 2 * p * rec / (p + rec))
    return best


def exact_match(pred: str, refs: list[str]) -> float:
    p = norm(pred)
    return float(any(p == norm(r) for r in refs))


def evidence_block(chunks: list[tuple[str, str]], offset: int = 0) -> str:
    lines = []
    for i, (_cid, text) in enumerate(chunks, start=offset + 1):
        lines.append(f"[E{i}] {text.strip()}")
    return "\n".join(lines)


def load_queries(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def select_by_tokens(chunks_list: list[tuple[str, str]], budget: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    tokens_used = 0
    for cid, text in chunks_list:
        toks = len(text.split())
        if tokens_used + toks > budget:
            continue
        out.append((cid, text))
        tokens_used += toks
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--index-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--token-budget", type=int, default=1024)
    ap.add_argument("--step2-top-k", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--n", type=int, default=0, help="0=all")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import json as _json
    from astro_cs_rag.indexing.io import load_chunks_jsonl
    from astro_cs_rag.indexing.dense import DenseIndex
    from astro_cs_rag.indexing.embedders import SentenceEmbedder

    chunks = load_chunks_jsonl(args.index_dir / "chunks.jsonl")
    chunk_text = {c.chunk_id: c.text for c in chunks}
    chunk_doc = {c.chunk_id: c.doc_id for c in chunks}

    dense_idx = DenseIndex.load(args.index_dir)

    # Use the same embedding model that built the index to ensure compatibility.
    _meta_path = args.index_dir / "index_meta.json"
    _embed_model = "BAAI/bge-m3"  # safe fallback
    if _meta_path.is_file():
        _meta = _json.loads(_meta_path.read_text(encoding="utf-8"))
        _embed_model = _meta.get("embedding_model", _embed_model)
    print(f"[iterative] loading embedder model: {_embed_model}")
    embedder = SentenceEmbedder(_embed_model, batch_size=1)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("[iterative] loading generator model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    gen_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="auto"
    )
    gen_model.eval()

    queries = load_queries(args.queries)
    if args.n > 0:
        queries = queries[:args.n]

    def embed_query(text: str) -> np.ndarray:
        return embedder.encode([text])[0]

    def retrieve(q_emb: np.ndarray, top_k: int, exclude: set[str] | None = None) -> list[tuple[str, str]]:
        n_fetch = top_k + (len(exclude) if exclude else 0)
        hits = dense_idx.topk(q_emb, n_fetch)
        results = []
        for cid, _score in hits:
            if exclude and cid in exclude:
                continue
            results.append((cid, chunk_text.get(cid, "")))
            if len(results) >= top_k:
                break
        return results

    def generate(system: str, user: str, max_new_tokens: int = 200) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(gen_model.device)
        with torch.no_grad():
            out = gen_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    results = []
    em_sum = f1_sum = 0.0
    bridge_covered_list: list[bool] = []
    t0 = time.perf_counter()

    print(f"[iterative] {len(queries)} queries, max_steps={args.max_steps}")

    for i, q in enumerate(queries):
        qid = q["query_id"]
        query_text = q["text"]
        gold_doc_ids = set(q.get("gold_doc_ids") or [])
        gold_ans = q.get("metadata", {}).get("answer") or ""
        refs = [gold_ans] if gold_ans else []

        # Step 1
        q_emb = embed_query(query_text)
        step1_pool = retrieve(q_emb, args.top_k)
        step1_sel = select_by_tokens(step1_pool[:20], args.token_budget // 2)

        evid1 = evidence_block(step1_sel, offset=0)
        step1_out = generate(STEP1_SYSTEM, USER_TMPL.format(query=query_text, evidence=evid1))

        if args.max_steps < 2:
            final_text = step1_out
            step2_sel: list[tuple[str, str]] = []
        else:
            # Step 2: use intermediate reasoning as follow-up query
            intermediate = extract_intermediate(step1_out)
            step2_query = f"{query_text} {intermediate}"
            q_emb2 = embed_query(step2_query)
            used_ids = {cid for cid, _ in step1_sel}
            step2_pool = retrieve(q_emb2, args.step2_top_k, exclude=used_ids)
            step2_sel = select_by_tokens(step2_pool, args.token_budget // 2)

            all_sel = step1_sel + step2_sel
            all_sel = select_by_tokens(all_sel, args.token_budget)
            evid_final = evidence_block(all_sel, offset=0)
            final_text = generate(
                STEP2_SYSTEM, USER_TMPL.format(query=query_text, evidence=evid_final)
            )

        pred = extract_final(final_text)
        em = exact_match(pred, refs)
        f1 = token_f1(pred, refs)
        em_sum += em
        f1_sum += f1

        all_cids = list({cid for cid, _ in (step1_sel + step2_sel)})
        sel_docs = {chunk_doc.get(c, "") for c in all_cids}
        if len(gold_doc_ids) >= 2:
            covered = gold_doc_ids.issubset(sel_docs)
            bridge_covered_list.append(covered)

        results.append({
            "query_id": qid,
            "answer_text": pred,
            "step1_out": step1_out,
            "final_out": final_text,
            "em": em,
            "f1": f1,
            "selected_chunk_ids": all_cids,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{i+1}/{len(queries)}] EM={em_sum/(i+1):.3f}  "
                f"F1={f1_sum/(i+1):.3f}  elapsed={elapsed:.0f}s"
            )

    n = len(results)
    metrics = {
        "answer_em_mean": em_sum / n,
        "answer_f1_mean": f1_sum / n,
        "answer_count": float(n),
        "bridge_coverage": (
            sum(bridge_covered_list) / len(bridge_covered_list)
            if bridge_covered_list else 0.0
        ),
    }

    out_ans = args.out_dir / "generated_answers.jsonl"
    out_ans.write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")

    out_met = args.out_dir / "metrics.json"
    out_met.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== FINAL RESULTS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"Total time: {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
