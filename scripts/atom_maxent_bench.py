"""D04 atoms + MaxEnt selection + Qwen 7B (batched).

MaxEnt principled balance of score vs coverage with Lagrange multipliers
tunable per-experiment. Diagnoses the submodular regression by sweeping
(beta, lambda_f) — when beta dominates, behaves like score-greedy; when
lambda_f dominates, behaves like submodular set-cover."""
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
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--lambda_f", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=0.001)
    ap.add_argument("--score_floor", type=float, default=0.0)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    # Hybrid retrieval (RRF or linear) using BM25 alongside dense
    ap.add_argument("--hybrid", default="dense",
                    choices=["dense", "rrf", "linear"])
    ap.add_argument("--hybrid_alpha", type=float, default=0.7,
                    help="dense weight in linear fusion (0..1)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    atoms = [json.loads(l) for l in open(args.atoms_dir / "atoms.jsonl")]
    atom_embs = np.load(args.atoms_dir / "atom_embs.npy")
    atom_embs = atom_embs / (np.linalg.norm(atom_embs, axis=1, keepdims=True) + 1e-9)

    _, _, _, meta = load_index_bundle(args.index_dir)
    embedder = embedder_from_meta(meta, EmbeddingSettings())

    queries = [json.loads(l) for l in open(args.queries)]
    q_embs = embedder.encode([q["text"] for q in queries]).astype(np.float32)
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)

    atom_types = np.array([a["claim_type"] for a in atoms])

    # Hybrid retrieval setup
    bm25 = bm25_idf = bm25_vocab = None
    if args.hybrid != "dense":
        from scipy import sparse as sp
        bm25 = sp.load_npz(args.atoms_dir / "atom_bm25.npz")
        bm25_idf = np.load(args.atoms_dir / "atom_bm25_idf.npy")
        bm25_vocab = json.loads((args.atoms_dir / "atom_bm25_vocab.json").read_text())
        _TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{1,}\b")
        _STOP = set(("a an the of in on at to for from with by is are was were be "
                     "been being and or but if then so than that this these those "
                     "it its as i you he she we they me him her us them my your "
                     "his their our what when where why how which who whom whose"
                     ).split())
        def _tok_q(text):
            return [t.lower() for t in _TOK.findall(text)
                    if t.lower() not in _STOP]
        def _bm25_score(text):
            cols = [bm25_vocab[t] for t in _tok_q(text) if t in bm25_vocab]
            if not cols:
                return np.zeros(bm25.shape[0], dtype=np.float32)
            return np.asarray(bm25[:, cols].sum(axis=1)).flatten()
        print(f"hybrid={args.hybrid} alpha={args.hybrid_alpha} "
              f"(BM25 vocab={len(bm25_vocab)})")

    selected_per_q = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        qe = q_embs[qi]
        s_dense = atom_embs @ qe
        if args.hybrid == "dense":
            s = s_dense
        else:
            s_bm25 = _bm25_score(q["text"])
            if args.hybrid == "linear":
                d_n = (s_dense - s_dense.mean()) / (s_dense.std() + 1e-9)
                b_n = (s_bm25 - s_bm25.mean()) / (s_bm25.std() + 1e-9)
                s = args.hybrid_alpha * d_n + (1 - args.hybrid_alpha) * b_n
            else:  # rrf
                d_ranks = np.argsort(-s_dense).argsort()
                b_ranks = np.argsort(-s_bm25).argsort()
                s = 1.0 / (60 + d_ranks) + 1.0 / (60 + b_ranks)
        intent = query_intent(q["text"])
        intent_conf = query_intent_conf(q["text"])
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
        sel = maxent_select(
            atoms=cands, facets=query_facets(q["text"], intent, intent_conf=intent_conf),
            token_budget=args.token_budget,
            beta=args.beta, lambda_f=args.lambda_f, mu=args.mu,
            score_floor=args.score_floor)
        selected_per_q.append(sel)
    print(f"MaxEnt selected in {time.time()-t0:.1f}s")

    with open(args.out_dir / "selected_atoms.jsonl", "w") as f:
        for q, sel in zip(queries, selected_per_q, strict=True):
            for s in sel:
                f.write(json.dumps({"query_id": q["query_id"], **s}) + "\n")

    print("Loading Qwen 2.5-7B-Instruct ...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
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
            print(f"  generated {done}/{len(prompts_text)} "
                  f"elapsed={time.time()-t1:.1f}s")

    out_rows = []
    cit_sum = f1_sum = em_sum = 0.0; n = 0
    for (q, sel), gen in zip(zip(queries, selected_per_q, strict=True),
                              out_gens, strict=True):
        cited = []
        for m in re.finditer(r"\[E(\d+)\]", gen):
            i = int(m.group(1)) - 1
            if 0 <= i < len(sel):
                cited.append(sel[i]["atom_id"])
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
            if mfa:
                ans = gen[mfa[-1].end():].split("\n", 1)[0]
        f1 = tokf1(ans, gold_ans.get(q["query_id"], []))
        em = float(any(normalize(ans) == normalize(g)
                   for g in gold_ans.get(q["query_id"], [])))
        f1_sum += f1; em_sum += em; n += 1
        out_rows.append({"query_id": q["query_id"], "answer_text_full": gen,
                         "answer_text": ans, "cited_atom_ids": cited,
                         "f1": f1, "em": em})

    with open(args.out_dir / "generated_answers.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {
        "n": n, "F1": f1_sum/max(1, n), "EM": em_sum/max(1, n),
        "cit_acc": cit_sum/max(1, n),
        "selector": "maxent", "beta": args.beta, "lambda_f": args.lambda_f,
        "mu": args.mu, "lambda_type": args.lambda_type,
        "prompt_style": args.prompt_style,
        "elapsed_sec": time.time()-t0,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nF1={metrics['F1']:.4f} EM={metrics['EM']:.4f} "
          f"cit={metrics['cit_acc']:.4f}")


if __name__ == "__main__":
    main()
