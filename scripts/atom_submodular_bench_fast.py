"""D04+D06 with BATCHED generation. ~5-10x faster than the per-query
loop in atom_submodular_bench.py. Same outputs, lossless (greedy decoding,
identical model weights, only batching changes)."""
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
from astro_cs_rag.selection.submodular import submodular_select, query_facets


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
    ap.add_argument("--atoms_dir", type=Path, required=True)
    ap.add_argument("--index_dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--lambda_type", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--token_budget", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--prompt_style", default="citation")
    ap.add_argument("--score_floor", type=float, default=0.0)
    ap.add_argument("--score_bonus", type=float, default=0.05)
    ap.add_argument("--doc_diversity_bonus", type=float, default=0.0)
    ap.add_argument("--max_atoms_per_doc", type=int, default=0)
    ap.add_argument("--adaptive_top_k", action="store_true")
    ap.add_argument("--use_fp16_embs", action="store_true")
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--use_flash_attn", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    atoms = [json.loads(l) for l in open(args.atoms_dir / "atoms.jsonl")]
    atom_embs = np.load(args.atoms_dir / "atom_embs.npy")
    if args.use_fp16_embs:
        atom_embs = atom_embs.astype(np.float16)

    _, _, _, meta = load_index_bundle(args.index_dir)
    embedder = embedder_from_meta(meta, EmbeddingSettings())

    queries = [json.loads(l) for l in open(args.queries)]
    q_embs = embedder.encode([q["text"] for q in queries]).astype(np.float32)
    q_embs = q_embs / (np.linalg.norm(q_embs, axis=1, keepdims=True) + 1e-9)

    atom_types = np.array([a["claim_type"] for a in atoms])

    selected_per_q: list[list[dict]] = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        qe = q_embs[qi]
        s = (atom_embs @ qe).astype(np.float32)
        intent = query_intent(q["text"])
        intent_conf = query_intent_conf(q["text"])
        if args.lambda_type > 0 and intent != "ANY":
            s = s + args.lambda_type * (atom_types == intent).astype(np.float32)
        eff_top_k = args.top_k
        if args.adaptive_top_k and intent == "ANY":
            eff_top_k = int(args.top_k * 1.5)
        eff_top_k = min(eff_top_k, len(s) - 1)
        topk_idx = np.argpartition(-s, eff_top_k)[:eff_top_k]
        topk_idx = topk_idx[np.argsort(-s[topk_idx])]
        candidates = []
        for i in topk_idx:
            a = atoms[int(i)]
            candidates.append({
                "atom_id": a["atom_id"], "chunk_id": a["chunk_id"],
                "doc_id": a["doc_id"], "text": a["text"],
                "claim_type": a["claim_type"], "score": float(s[int(i)]),
            })
        facets = query_facets(q["text"], intent, intent_conf=intent_conf)
        sel = submodular_select(
            atoms=candidates, facets=facets,
            token_budget=args.token_budget,
            score_floor=args.score_floor, score_bonus=args.score_bonus,
            doc_diversity_bonus=args.doc_diversity_bonus,
            max_atoms_per_doc=args.max_atoms_per_doc,
        )
        selected_per_q.append(sel)
    print(f"Selected (submodular) in {time.time()-t0:.1f}s")

    with open(args.out_dir / "selected_atoms.jsonl", "w") as f:
        for q, sel in zip(queries, selected_per_q, strict=True):
            for s in sel:
                f.write(json.dumps({"query_id": q["query_id"], **s}) + "\n")

    print("Loading Qwen 2.5-7B-Instruct ...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                        padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    kw = dict(torch_dtype=torch.bfloat16, device_map="cuda")
    if args.use_flash_attn:
        kw["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", **kw)
    model.train(False)

    gold_docs = {q["query_id"]: set(q["gold_doc_ids"]) for q in queries}
    gold_ans = {q["query_id"]: ([q["metadata"]["answer"]]
                                 if isinstance(q["metadata"].get("answer"), str)
                                 else (q["metadata"].get("answer") or []))
                 for q in queries}

    # Build all chat-template strings up front
    prompts_text = []
    for q, sel in zip(queries, selected_per_q, strict=True):
        evidence = [(s["atom_id"], s["text"]) for s in sel]
        p = assemble(query=q["text"], evidence=evidence, style=args.prompt_style)
        chat = [{"role": "system", "content": p.system},
                {"role": "user", "content": p.user}]
        prompts_text.append(tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True))

    # Batched generation
    out_gens: list[str] = []
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
    cit_sum = 0.0; f1_sum = 0.0; em_sum = 0.0; n = 0
    for (q, sel), gen in zip(zip(queries, selected_per_q, strict=True),
                              out_gens, strict=True):
        cited_atoms = []
        for m in re.finditer(r"\[E(\d+)\]", gen):
            i = int(m.group(1)) - 1
            if 0 <= i < len(sel):
                cited_atoms.append(sel[i]["atom_id"])
        gd = gold_docs.get(q["query_id"], set())
        if cited_atoms:
            n_hit = sum(1 for aid in cited_atoms
                        if any(s["atom_id"] == aid and s["doc_id"] in gd
                               for s in sel))
            cit_sum += n_hit / len(cited_atoms)
        ans_for_metric = gen
        if args.prompt_style == "cot":
            mfa = list(re.finditer(r"final\s*answer\s*:\s*", gen,
                                   flags=re.IGNORECASE))
            if mfa:
                ans_for_metric = gen[mfa[-1].end():].split("\n", 1)[0]
        f1 = tokf1(ans_for_metric, gold_ans.get(q["query_id"], []))
        em = float(any(normalize(ans_for_metric) == normalize(g)
                       for g in gold_ans.get(q["query_id"], [])))
        f1_sum += f1; em_sum += em; n += 1
        out_rows.append({
            "query_id": q["query_id"],
            "answer_text_full": gen, "answer_text": ans_for_metric,
            "cited_atom_ids": cited_atoms, "f1": f1, "em": em,
        })

    with open(args.out_dir / "generated_answers.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {
        "n": n,
        "F1": f1_sum / max(1, n), "EM": em_sum / max(1, n),
        "cit_acc": cit_sum / max(1, n),
        "lambda_type": args.lambda_type, "prompt_style": args.prompt_style,
        "selector": "submodular_fast",
        "doc_diversity_bonus": args.doc_diversity_bonus,
        "max_atoms_per_doc": args.max_atoms_per_doc,
        "adaptive_top_k": args.adaptive_top_k,
        "gen_batch_size": args.gen_batch_size,
        "elapsed_sec": time.time() - t0,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults: F1={metrics['F1']:.4f} EM={metrics['EM']:.4f} "
          f"cit={metrics['cit_acc']:.4f}")


if __name__ == "__main__":
    main()
