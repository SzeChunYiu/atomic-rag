"""End-to-end D04 benchmark: atom-level retrieval + greedy budget select + Qwen 7B terse.

Skips the existing chunk-level retrieve/atom-detect/select pipeline.
Directly:
  1. Read pre-built atoms (from build_atoms.py)
  2. For each query: encode, score each atom, optional typed bonus
  3. Greedy select top atoms within token budget
  4. Generate with Qwen 7B terse (citation prompt)
  5. Evaluate
"""
from __future__ import annotations
import argparse, json, re, string, time
from pathlib import Path
from collections import Counter

import numpy as np

import sys
sys.path.insert(0, "/projects/hep/fs10/shared/nnbar/billy/RAG/src")
sys.path.insert(0, "/Users/billy/Desktop/projects/AI_engineering/RAG/src")
from astro_cs_rag.atoms.deblend import query_intent
from astro_cs_rag.cli.helpers import embedder_from_meta, load_index_bundle
from astro_cs_rag.config.schema import EmbeddingSettings
from astro_cs_rag.generation.prompts import assemble


_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")
_EVID = re.compile(r"\[E\d+\]")
def normalize(s):
    s = _EVID.sub(" ", s)
    s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s)
    return _WS.sub(" ", s).strip()
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
    ap.add_argument("--atoms_dir", type=Path, required=True,
                    help="Directory with atoms.jsonl and atom_embs.npy")
    ap.add_argument("--index_dir", type=Path, required=True,
                    help="Original index_bundle (for embedder meta)")
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--lambda_type", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--token_budget", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load atoms
    atoms = []
    for line in open(args.atoms_dir / "atoms.jsonl"):
        atoms.append(json.loads(line))
    atom_embs = np.load(args.atoms_dir / "atom_embs.npy")
    print(f"Loaded {len(atoms)} atoms, embs shape {atom_embs.shape}")

    # Load embedder
    _, _, _, meta = load_index_bundle(args.index_dir)
    embedder = embedder_from_meta(meta, EmbeddingSettings())

    # Load queries
    queries = []
    for line in open(args.queries):
        queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries")

    # Encode queries
    q_texts = [q["text"] for q in queries]
    q_embs = embedder.encode(q_texts).astype(np.float32)
    q_norms = np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9
    q_embs = q_embs / q_norms
    print(f"Encoded queries, shape {q_embs.shape}")

    atom_types = np.array([a["claim_type"] for a in atoms])

    # Per-query: score atoms, take top-K, fill budget
    selected_per_q: list[list[dict]] = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        qe = q_embs[qi]
        s = atom_embs @ qe  # (N_atoms,)
        intent = query_intent(q["text"])
        if args.lambda_type > 0 and intent != "ANY":
            type_match = atom_types == intent
            s = s + args.lambda_type * type_match.astype(np.float32)
        # Take top-K atoms
        topk_idx = np.argpartition(-s, min(args.top_k, len(s) - 1))[:args.top_k]
        topk_idx = topk_idx[np.argsort(-s[topk_idx])]
        # Fill token budget (count by approximate tokens = words * 1.3)
        sel = []
        budget = 0
        for i in topk_idx:
            atom = atoms[int(i)]
            tok = max(1, int(len(atom["text"].split()) * 1.3))
            if budget + tok > args.token_budget:
                continue
            sel.append({
                "atom_id": atom["atom_id"],
                "chunk_id": atom["chunk_id"],
                "doc_id": atom["doc_id"],
                "text": atom["text"],
                "claim_type": atom["claim_type"],
                "score": float(s[int(i)]),
            })
            budget += tok
        selected_per_q.append(sel)
    print(f"Selected atoms in {time.time()-t0:.1f}s")

    # Save selected_context
    with open(args.out_dir / "selected_atoms.jsonl", "w") as f:
        for q, sel in zip(queries, selected_per_q, strict=True):
            for s in sel:
                f.write(json.dumps({"query_id": q["query_id"], **s}) + "\n")

    # Generation with Qwen 7B
    print("Loading Qwen 2.5-7B-Instruct ...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    model.train(False)

    # Prep gold for metrics
    gold_docs = {q["query_id"]: set(q["gold_doc_ids"]) for q in queries}
    gold_ans = {q["query_id"]: ([q["metadata"]["answer"]]
                                 if isinstance(q["metadata"].get("answer"), str)
                                 else (q["metadata"].get("answer") or []))
                 for q in queries}

    out_rows = []
    cit_sum = 0.0; f1_sum = 0.0; em_sum = 0.0; n_eval = 0
    t1 = time.time()
    for qi, (q, sel) in enumerate(zip(queries, selected_per_q, strict=True)):
        evidence = [(s["atom_id"], s["text"]) for s in sel]
        prompt = assemble(query=q["text"], evidence=evidence, style="citation")
        chat = [{"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=8192).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, do_sample=False, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = out[0, enc["input_ids"].shape[1]:]
        gen = tok.decode(new, skip_special_tokens=True).strip()

        # Citation parsing
        cited_atoms = []
        for m in re.finditer(r"\[E(\d+)\]", gen):
            i = int(m.group(1)) - 1
            if 0 <= i < len(sel):
                cited_atoms.append(sel[i]["atom_id"])
        # Citation accuracy at doc level
        cited_docs = {sel[i]["doc_id"]
                      for m in re.finditer(r"\[E(\d+)\]", gen)
                      for i in [int(m.group(1)) - 1]
                      if 0 <= i < len(sel)}
        gd = gold_docs.get(q["query_id"], set())
        if cited_atoms:
            n_hit = sum(1 for aid in cited_atoms
                        if any(s["atom_id"] == aid and s["doc_id"] in gd for s in sel))
            cit_sum += n_hit / len(cited_atoms)
        f1 = tokf1(gen, gold_ans.get(q["query_id"], []))
        em = float(any(normalize(gen) == normalize(g) for g in gold_ans.get(q["query_id"], [])))
        f1_sum += f1
        em_sum += em
        n_eval += 1
        out_rows.append({
            "query_id": q["query_id"],
            "answer_text": gen,
            "cited_atom_ids": cited_atoms,
            "f1": f1,
            "em": em,
        })
        if (qi + 1) % 100 == 0:
            print(f"  generated {qi+1}/{len(queries)}  elapsed={time.time()-t1:.1f}s")

    with open(args.out_dir / "generated_answers.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {
        "n": n_eval,
        "F1": f1_sum / max(1, n_eval),
        "EM": em_sum / max(1, n_eval),
        "cit_acc": cit_sum / max(1, n_eval),
        "lambda_type": args.lambda_type,
        "top_k": args.top_k,
        "token_budget": args.token_budget,
        "elapsed_sec": time.time() - t0,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults: F1={metrics['F1']:.4f}  EM={metrics['EM']:.4f}  cit={metrics['cit_acc']:.4f}")


if __name__ == "__main__":
    main()
