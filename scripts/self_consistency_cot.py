"""Self-consistency over CoT chains.

Sample N CoT answers per query at temperature T, extract the final answer
from each, vote majority. Output a `generated_answers_sc.jsonl` with the
voted final answer + citations from the most-voted sample.

Run after a CoT-prompted run; uses the same model loaded into the SAME
process for efficiency.
"""
from __future__ import annotations
import argparse, json, re, string
from pathlib import Path
from collections import Counter

import torch
import yaml

CIT_RE = re.compile(r"\[E(\d+)\]")
PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
ART = re.compile(r"\b(a|an|the)\b")
WS = re.compile(r"\s+")


def normalize(s):
    s = PUNC.sub(" ", s.lower())
    s = ART.sub(" ", s)
    return WS.sub(" ", s).strip()


def extract_final(text):
    m = list(re.finditer(r"final\s*answer\s*:\s*", text, flags=re.IGNORECASE))
    if not m:
        line = next((ln for ln in reversed(text.splitlines()) if ln.strip()), text)
    else:
        line = text[m[-1].end():]
        line = line.split("\n", 1)[0]
    cits = [int(x) for x in CIT_RE.findall(line)]
    final_only = CIT_RE.sub(" ", line).strip()
    return final_only, cits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="run dir from a CoT-prompted run")
    ap.add_argument("--n_samples", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    args = ap.parse_args()

    rd = args.run_dir
    cfg = yaml.safe_load(open(rd / "config.yaml"))
    qpath = Path(cfg["paths"]["queries_path"])
    if not qpath.is_absolute():
        qpath = Path("/projects/hep/fs10/shared/nnbar/billy/RAG") / qpath
    model_id = "Qwen/Qwen2.5-7B-Instruct"  # hard-coded; matches benchmark.

    # Re-build prompts from selected_context for each query
    sel = {}
    for line in open(rd / "selected_context.jsonl"):
        r = json.loads(line)
        sel.setdefault(r["query_id"], []).append(r["chunk_id"])

    chunks_text = {}
    for line in open(rd.parent / "index_bundle" / "chunks.jsonl"):
        c = json.loads(line)
        chunks_text[c["chunk_id"]] = c["text"]

    queries = {}
    for line in open(qpath):
        q = json.loads(line)
        queries[q["query_id"]] = q["text"]

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.train(False)

    import sys
    sys.path.insert(0, "/projects/hep/fs10/shared/nnbar/billy/RAG/src")
    from astro_cs_rag.generation.prompts import assemble

    out_rows = []
    for qid in queries:
        if qid not in sel:
            continue
        cids = sel[qid]
        evidence = [(c, chunks_text.get(c, "")) for c in cids]
        prompt = assemble(query=queries[qid], evidence=evidence, style="cot")
        chat = [{"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user}]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=8192).to("cuda")

        # Sample N times
        finals = []
        sample_data = []
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=args.n_samples,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        for seq in out:
            new = seq[enc["input_ids"].shape[1]:]
            txt = tok.decode(new, skip_special_tokens=True).strip()
            final, cits = extract_final(txt)
            finals.append(final)
            sample_data.append({"text": txt, "final": final, "cits": cits})

        # Majority vote on normalized final
        norms = [normalize(f) for f in finals]
        cnt = Counter(norms)
        winner_norm = cnt.most_common(1)[0][0]
        # Pick the first sample matching winner
        winner = next((s for s, n in zip(sample_data, norms) if n == winner_norm), sample_data[0])

        cited_ids = []
        seen = set()
        for i in winner["cits"]:
            if 1 <= i <= len(cids):
                cid = cids[i-1]
                if cid not in seen:
                    cited_ids.append(cid); seen.add(cid)

        out_rows.append({
            "query_id": qid,
            "answer_text": winner["final"],
            "answer_text_full": winner["text"],
            "cited_chunk_ids": cited_ids,
            "selected_chunk_ids": cids,
            "n_samples": args.n_samples,
            "temperature": args.temperature,
            "samples": [s["final"] for s in sample_data],
            "vote_count": cnt.most_common(1)[0][1],
        })
        if len(out_rows) % 100 == 0:
            print(f"  {len(out_rows)} queries done")

    with open(rd / "generated_answers_sc.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(out_rows)} rows to {rd}/generated_answers_sc.jsonl")


if __name__ == "__main__":
    main()
