"""B4 method: prompt-ordering replicas to mitigate LLM position bias.

Diagnosis: 26% of queries have gold answer text in the selected atom
pool, but F1 < 0.5 — the LLM has the answer but doesn't extract it.
Documented mechanism: position bias (Liu et al. 2023, "Lost in the
Middle"). Answer atoms buried mid-context get low attention.

Method: K replicas, each with a different permutation of the [E_i]
atoms in the prompt. Position-bias errors are uncorrelated across
orderings; majority-vote averages them out.

Distinct from self-consistency CoT (perturbs decoding sampling). For
greedy T=0 decoding on Qwen 7B, SC is null because greedy is
deterministic. Ordering replicas perturb the layer where Qwen's bias
actually lives.

Falsifier: predicted to specifically help the F1- cit+ gold+ text+
bucket (139 queries). Will report bucket-conditional gains.
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


def extract_answer(gen: str, prompt_style: str) -> str:
    if prompt_style != "cot":
        return gen
    mfa = list(re.finditer(r"final\s*answer\s*:\s*", gen, flags=re.IGNORECASE))
    if mfa:
        return gen[mfa[-1].end():].split("\n", 1)[0]
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms_dir", type=Path, required=True)
    ap.add_argument("--index_dir", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--lambda_type", type=float, default=0.05)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--token_budget", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--prompt_style", default="cot")
    ap.add_argument("--n_replicas", type=int, default=3)
    ap.add_argument("--gen_batch_size", type=int, default=4)
    ap.add_argument("--selector", default="maxent",
                    choices=["maxent", "greedy"])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--lambda_f", type=float, default=0.5)
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

    selected_per_q = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        intent = query_intent(q["text"])
        intent_conf = query_intent_conf(q["text"])
        s = atom_embs @ q_embs[qi]
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
                beta=args.beta, lambda_f=args.lambda_f, mu=0.001)
        else:
            sel = []; used = 0
            for c in cands:
                tok = max(1, len(c["text"].split()))
                if used + tok > args.token_budget: continue
                sel.append(c); used += tok
        selected_per_q.append(sel)
    print(f"Selected in {time.time()-t0:.1f}s")

    with open(args.out_dir / "selected_atoms.jsonl", "w") as f:
        for q, sel in zip(queries, selected_per_q, strict=True):
            for s in sel:
                f.write(json.dumps({"query_id": q["query_id"], **s}) + "\n")

    # Build K orderings of selected atoms per query
    rng_seeds = list(range(args.n_replicas))
    print(f"Loading Qwen 2.5-7B-Instruct ({args.n_replicas} replicas)...")
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

    # Replicas: per query, K different orderings
    replica_answers: list[list[str]] = [[] for _ in queries]
    replica_full: list[list[str]] = [[] for _ in queries]
    replica_perms: list[list[list[int]]] = [[] for _ in queries]

    for r_i, seed in enumerate(rng_seeds):
        rng = np.random.RandomState(seed)
        prompts_text = []
        perms_this = []
        for qi, (q, sel) in enumerate(zip(queries, selected_per_q, strict=True)):
            n = len(sel)
            if seed == 0:
                perm = list(range(n))            # identity ordering
            else:
                perm = list(rng.permutation(n))
            perms_this.append(perm)
            sel_perm = [sel[j] for j in perm]
            evidence = [(s["atom_id"], s["text"]) for s in sel_perm]
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
                print(f"  replica {r_i+1}/{args.n_replicas} {done}/{len(prompts_text)} "
                      f"elapsed={time.time()-t1:.1f}s")

        for qi, gen in enumerate(out_gens):
            replica_answers[qi].append(extract_answer(gen, args.prompt_style))
            replica_full[qi].append(gen)
            replica_perms[qi].append(perms_this[qi])

    # Aggregate via majority vote on normalized answers
    out_rows = []
    f1_sum = em_sum = cit_sum = 0.0; n = 0
    f1_replica0 = em_replica0 = 0.0
    for qi, (q, sel) in enumerate(zip(queries, selected_per_q, strict=True)):
        ans_norms = [normalize(a) for a in replica_answers[qi]]
        cnt = Counter(ans_norms)
        # Pick majority normalized; tiebreak = replica 0
        majority_norm, _ = cnt.most_common(1)[0]
        if majority_norm == ans_norms[0] or cnt[majority_norm] == 1:
            chosen_idx = 0
        else:
            chosen_idx = next(i for i, a in enumerate(ans_norms)
                              if a == majority_norm)
        chosen_ans = replica_answers[qi][chosen_idx]
        chosen_full = replica_full[qi][chosen_idx]
        chosen_perm = replica_perms[qi][chosen_idx]

        # Cited atoms in original (non-permuted) sel indexing
        cited = []
        for m in re.finditer(r"\[E(\d+)\]", chosen_full):
            i = int(m.group(1)) - 1
            if 0 <= i < len(sel):
                # The [E_i] in chosen_full refers to the PERMUTED order
                # so map back via chosen_perm
                if i < len(chosen_perm):
                    cited.append(sel[chosen_perm[i]]["atom_id"])
        cited = filter_citations(chosen_full, cited, {s["atom_id"]: s["text"] for s in sel})
        gd = gold_docs.get(q["query_id"], set())
        if cited:
            atom_doc = {s["atom_id"]: s["doc_id"] for s in sel}
            n_hit = sum(1 for aid in cited if atom_doc.get(aid) in gd)
            cit_sum += n_hit / len(cited)

        f1 = tokf1(chosen_ans, gold_ans.get(q["query_id"], []))
        em = float(any(normalize(chosen_ans) == normalize(g)
                   for g in gold_ans.get(q["query_id"], [])))
        f1_sum += f1; em_sum += em; n += 1

        # Replica-0 baseline (no replicas, single ordering)
        f1_r0 = tokf1(replica_answers[qi][0], gold_ans.get(q["query_id"], []))
        em_r0 = float(any(normalize(replica_answers[qi][0]) == normalize(g)
                      for g in gold_ans.get(q["query_id"], [])))
        f1_replica0 += f1_r0; em_replica0 += em_r0

        out_rows.append({
            "query_id": q["query_id"],
            "answers_replicas": replica_answers[qi],
            "chosen_idx": chosen_idx,
            "answer_text": chosen_ans, "answer_text_full": chosen_full,
            "cited_atom_ids": cited, "f1": f1, "em": em,
            "f1_replica0": f1_r0, "em_replica0": em_r0,
        })

    with open(args.out_dir / "generated_answers.jsonl", "w") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    metrics = {
        "n": n, "F1": f1_sum/max(1, n), "EM": em_sum/max(1, n),
        "cit_acc": cit_sum/max(1, n),
        "F1_replica0": f1_replica0/max(1, n),
        "EM_replica0": em_replica0/max(1, n),
        "delta_F1": f1_sum/max(1, n) - f1_replica0/max(1, n),
        "n_replicas": args.n_replicas, "selector": args.selector,
        "elapsed_sec": time.time()-t0,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nReplica-0:  F1={metrics['F1_replica0']:.4f} EM={metrics['EM_replica0']:.4f}")
    print(f"Majority:   F1={metrics['F1']:.4f} EM={metrics['EM']:.4f} "
          f"cit={metrics['cit_acc']:.4f}")
    print(f"Δ F1 (majority - replica0): {metrics['delta_F1']:+.4f}")


if __name__ == "__main__":
    main()
