"""Multi-scale (RG-style) retrieval -> MaxEnt selection -> Qwen 7B.

Tests B2 method: 3-scale retrieval with scale-dependent kernel.
- Scale D: doc-centroid cosine (smooth, topic)
- Scale C: chunk cosine + entity overlap
- Scale A: atom cosine + entity + claim-type
"""
from __future__ import annotations
import argparse, json, re, string, time
from pathlib import Path
from collections import Counter
import numpy as np

import sys
sys.path.insert(0, "/projects/hep/fs10/shared/nnbar/billy/RAG/src")
sys.path.insert(0, "/Users/billy/Desktop/projects/AI_engineering/RAG/src")
from astro_cs_rag.atoms.deblend import query_intent, query_intent_conf
from astro_cs_rag.cli.helpers import embedder_from_meta, load_index_bundle
from astro_cs_rag.config.schema import EmbeddingSettings
from astro_cs_rag.generation.prompts import assemble
from astro_cs_rag.generation.citation_filter import filter_citations
from astro_cs_rag.selection.submodular import query_facets
from astro_cs_rag.selection.maxent import maxent_select
from astro_cs_rag.retrieval.multiscale import (
    build_doc_centroids, multiscale_retrieve_atoms)


_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_EVID = re.compile(r"\[E\d+\]")
def normalize(s):
    s = _EVID.sub(" ", s); s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s); return _WS.sub(" ", s).strip()
def tokf1(p, refs):
    if not refs: return 0.0
    pt = normalize(p).split()
    if not pt: return 0.0
    best = 0.0
    for r in refs:
        rt = normalize(r).split()
        if not rt: continue
        common = Counter(pt) & Counter(rt)
        n = sum(common.values())
        if n == 0: continue
        prec = n/len(pt); rec = n/len(rt)
        best = max(best, 2*prec*rec/(prec+rec))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms_dir", type=Path, required=True)
    ap.add_argument("--index_dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--top_k_doc", type=int, default=30)
    ap.add_argument("--top_k_chunk", type=int, default=80)
    ap.add_argument("--top_k_atom", type=int, default=50)
    ap.add_argument("--entity_weight_chunk", type=float, default=0.1)
    ap.add_argument("--entity_weight_atom", type=float, default=0.1)
    ap.add_argument("--typed_weight_atom", type=float, default=0.05)
    ap.add_argument("--token_budget", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--prompt_style", default="citation")
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--lambda_f", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.001)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    atoms = [json.loads(l) for l in open(args.atoms_dir / "atoms.jsonl")]
    atom_embs = np.load(args.atoms_dir / "atom_embs.npy")
    atom_embs = atom_embs / (np.linalg.norm(atom_embs, axis=1, keepdims=True) + 1e-9)

    chunks_path = args.index_dir / "chunks.jsonl"
    chunk_meta = [json.loads(l) for l in open(chunks_path)]
    chunk_embs = np.load(args.index_dir / "embeddings.npy")
    chunk_embs = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)
    print(f"loaded {len(chunk_meta)} chunks, {len(atoms)} atoms")

    cents, doc_ids, _ = build_doc_centroids(
        chunk_embs, [c["doc_id"] for c in chunk_meta])
    print(f"built {len(doc_ids)} doc centroids")

    _, _, _, meta = load_index_bundle(args.index_dir)
    embedder = embedder_from_meta(meta, EmbeddingSettings())
    queries = [json.loads(l) for l in open(args.queries)]
    q_embs = embedder.encode([q["text"] for q in queries]).astype(np.float32)
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)

    selected_per_q = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        intent = query_intent(q["text"])
        intent_conf = query_intent_conf(q["text"])
        atom_idx = multiscale_retrieve_atoms(
            query_emb=q_embs[qi], query_text=q["text"],
            atom_embs=atom_embs, atom_meta=atoms,
            chunk_embs=chunk_embs, chunk_meta=chunk_meta,
            doc_centroids=cents, doc_ids=doc_ids,
            top_k_doc=args.top_k_doc, top_k_chunk=args.top_k_chunk,
            top_k_atom=args.top_k_atom,
            entity_weight_chunk=args.entity_weight_chunk,
            entity_weight_atom=args.entity_weight_atom,
            typed_weight_atom=args.typed_weight_atom,
            intent_type=intent,
            intent_conf=intent_conf)
        cands = []
        for i in atom_idx:
            a = atoms[i]
            cands.append({
                "atom_id": a["atom_id"], "chunk_id": a["chunk_id"],
                "doc_id": a["doc_id"], "text": a["text"],
                "claim_type": a["claim_type"],
                "score": float(atom_embs[i] @ q_embs[qi]),
            })
        sel = maxent_select(
            atoms=cands, facets=query_facets(q["text"], intent, intent_conf=intent_conf),
            token_budget=args.token_budget,
            beta=args.beta, lambda_f=args.lambda_f, mu=args.mu)
        selected_per_q.append(sel)
    print(f"Multi-scale + MaxEnt selected in {time.time()-t0:.1f}s")

    with open(args.out_dir / "selected_atoms.jsonl", "w") as f:
        for q, sel in zip(queries, selected_per_q, strict=True):
            for s in sel:
                f.write(json.dumps({"query_id": q["query_id"], **s}) + "\n")

    print("Loading Qwen 2.5-7B-Instruct ...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", padding_side="left")
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
        device_map="cuda")
    model.train(False)

    gold_docs = {q["query_id"]: set(q["gold_doc_ids"]) for q in queries}
    gold_ans = {q["query_id"]: ([q["metadata"]["answer"]]
                if isinstance(q["metadata"].get("answer"), str)
                else (q["metadata"].get("answer") or []))
                for q in queries}

    prompts_text = []
    for q, sel in zip(queries, selected_per_q, strict=True):
        evidence = [(s["atom_id"], s["text"]) for s in sel]
        p = assemble(query=q["text"], evidence=evidence, style=args.prompt_style)
        chat = [{"role": "system", "content": p.system},
                {"role": "user", "content": p.user}]
        prompts_text.append(tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True))

    out_gens = []
    t1 = time.time()
    B = args.gen_batch_size
    for i in range(0, len(prompts_text), B):
        batch = prompts_text[i:i+B]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=8192).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False,
                                 max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
        for j, prompt_ids in enumerate(enc["input_ids"]):
            new_ids = out[j, prompt_ids.shape[0]:]
            out_gens.append(tok.decode(new_ids, skip_special_tokens=True).strip())
        if (i + B) % (B * 12) == 0 or i + B >= len(prompts_text):
            done = min(i + B, len(prompts_text))
            print(f"  generated {done}/{len(prompts_text)} elapsed={time.time()-t1:.1f}s")

    out_rows = []
    cit_sum = f1_sum = em_sum = 0.0; n = 0
    bridge_recall = []  # 1.0 if both gold docs appear in candidates
    for (q, sel), gen in zip(zip(queries, selected_per_q, strict=True),
                              out_gens, strict=True):
        cited = []
        for m in re.finditer(r"\[E(\d+)\]", gen):
            i = int(m.group(1)) - 1
            if 0 <= i < len(sel): cited.append(sel[i]["atom_id"])
        cited = filter_citations(gen, cited, {s["atom_id"]: s["text"] for s in sel})
        gd = gold_docs.get(q["query_id"], set())
        if cited:
            n_hit = sum(1 for aid in cited
                        if any(s["atom_id"] == aid and s["doc_id"] in gd
                               for s in sel))
            cit_sum += n_hit / len(cited)
        ans = gen
        if args.prompt_style == "cot":
            mfa = list(re.finditer(r"final\s*answer\s*:\s*", gen,
                                   flags=re.IGNORECASE))
            if mfa: ans = gen[mfa[-1].end():].split("\n", 1)[0]
        f1 = tokf1(ans, gold_ans.get(q["query_id"], []))
        em = float(any(normalize(ans) == normalize(g)
                   for g in gold_ans.get(q["query_id"], [])))
        # Bridge recall = both gold docs in candidate pool
        cand_docs = {s["doc_id"] for s in sel}
        if gd:
            br = float(gd.issubset(cand_docs))
            bridge_recall.append(br)
        f1_sum += f1; em_sum += em; n += 1
        out_rows.append({"query_id": q["query_id"], "answer_text_full": gen,
                         "answer_text": ans, "cited_atom_ids": cited,
                         "f1": f1, "em": em})

    with open(args.out_dir / "generated_answers.jsonl", "w") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {
        "n": n, "F1": f1_sum/max(1, n), "EM": em_sum/max(1, n),
        "cit_acc": cit_sum/max(1, n),
        "bridge_recall_in_pool": (sum(bridge_recall) / max(1, len(bridge_recall))),
        "selector": "multiscale+maxent",
        "beta": args.beta, "lambda_f": args.lambda_f, "mu": args.mu,
        "top_k_doc": args.top_k_doc, "top_k_chunk": args.top_k_chunk,
        "top_k_atom": args.top_k_atom,
        "prompt_style": args.prompt_style, "elapsed_sec": time.time()-t0,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nF1={metrics['F1']:.4f} EM={metrics['EM']:.4f} "
          f"cit={metrics['cit_acc']:.4f} bridge={metrics['bridge_recall_in_pool']:.4f}")


if __name__ == "__main__":
    main()
