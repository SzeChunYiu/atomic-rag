"""D04 atoms + PRF query expansion + greedy/MaxEnt selection + Qwen 7B.

PRF (Lavrenko-Croft 2001 RM3) adapted for atomic RAG with intent-typed
re-weighting. Tests whether classical IR query expansion fixes the
bridge-recall failure (10.6% queries miss 2nd gold doc)."""
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
from astro_cs_rag.retrieval.prf import prf_expand_query


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
    ap.add_argument("--lambda_type", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--token_budget", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--prompt_style", default="citation")
    # PRF
    ap.add_argument("--prf_top_m", type=int, default=10)
    ap.add_argument("--prf_alpha", type=float, default=0.7)
    ap.add_argument("--prf_intent_weight", type=float, default=0.2)
    # Selector: maxent (default) or score-greedy fallback
    ap.add_argument("--selector", default="maxent",
                    choices=["maxent", "greedy"])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--lambda_f", type=float, default=0.5)
    ap.add_argument("--mu", type=float, default=0.001)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    atoms = [json.loads(l) for l in open(args.atoms_dir / "atoms.jsonl")]
    atom_embs = np.load(args.atoms_dir / "atom_embs.npy")
    norms = np.linalg.norm(atom_embs, axis=1, keepdims=True) + 1e-9
    atom_embs_norm = atom_embs / norms

    _, _, _, meta = load_index_bundle(args.index_dir)
    embedder = embedder_from_meta(meta, EmbeddingSettings())
    queries = [json.loads(l) for l in open(args.queries)]
    q_embs = embedder.encode([q["text"] for q in queries]).astype(np.float32)
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)
    atom_types = np.array([a["claim_type"] for a in atoms])

    selected_per_q = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        intent = query_intent(q["text"])
        intent_conf = query_intent_conf(q["text"])
        # PRF query expansion (Lavrenko-Croft 2001 + intent-typed weighting)
        qe = prf_expand_query(
            q_embs[qi], atom_embs_norm, atom_types,
            top_m=args.prf_top_m, alpha=args.prf_alpha,
            intent_type=intent, intent_weight=args.prf_intent_weight)
        s = atom_embs_norm @ qe
        if args.lambda_type > 0 and intent != "ANY":
            s = s + args.lambda_type * (atom_types == intent).astype(np.float32)
        topk_idx = np.argpartition(-s, min(args.top_k, len(s) - 1))[:args.top_k]
        topk_idx = topk_idx[np.argsort(-s[topk_idx])]
        cands = [{
            "atom_id": atoms[int(i)]["atom_id"],
            "chunk_id": atoms[int(i)]["chunk_id"],
            "doc_id": atoms[int(i)]["doc_id"],
            "text": atoms[int(i)]["text"],
            "claim_type": atoms[int(i)]["claim_type"],
            "score": float(s[int(i)]),
        } for i in topk_idx]
        if args.selector == "maxent":
            sel = maxent_select(
                atoms=cands, facets=query_facets(q["text"], intent, intent_conf=intent_conf),
                token_budget=args.token_budget,
                beta=args.beta, lambda_f=args.lambda_f, mu=args.mu)
        else:
            # Score-greedy budget-fill (D04-style)
            sel = []; used = 0
            for c in cands:
                tok = max(1, len(c["text"].split()))
                if used + tok > args.token_budget: continue
                sel.append(c); used += tok
        selected_per_q.append(sel)
    print(f"Selected ({args.selector}+PRF) in {time.time()-t0:.1f}s")

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
            mfa = list(re.finditer(r"final\s*answer\s*:\s*", gen, flags=re.IGNORECASE))
            if mfa: ans = gen[mfa[-1].end():].split("\n", 1)[0]
        f1 = tokf1(ans, gold_ans.get(q["query_id"], []))
        em = float(any(normalize(ans) == normalize(g)
                   for g in gold_ans.get(q["query_id"], [])))
        f1_sum += f1; em_sum += em; n += 1
        out_rows.append({"query_id": q["query_id"], "answer_text_full": gen,
                         "answer_text": ans, "cited_atom_ids": cited,
                         "f1": f1, "em": em})

    with open(args.out_dir / "generated_answers.jsonl", "w") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {
        "n": n, "F1": f1_sum/max(1, n), "EM": em_sum/max(1, n),
        "cit_acc": cit_sum/max(1, n),
        "selector": args.selector, "prf_top_m": args.prf_top_m,
        "prf_alpha": args.prf_alpha,
        "prf_intent_weight": args.prf_intent_weight,
        "lambda_type": args.lambda_type, "prompt_style": args.prompt_style,
        "elapsed_sec": time.time()-t0,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nF1={metrics['F1']:.4f} EM={metrics['EM']:.4f} cit={metrics['cit_acc']:.4f}")


if __name__ == "__main__":
    main()
