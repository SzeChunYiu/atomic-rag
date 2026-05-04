"""Stage diagnostic: generation only.

Takes pre-selected evidence (selected_context.jsonl from a completed LUNARC run)
and runs the generator locally. Verifies:
  - _hf_model fix (no ValidationError on model=None)
  - CoT answer extraction (_FINAL_ANS_RE without DOTALL)
  - Prompt style output (plain / cot / few_shot_cot)

Usage:
  python scripts/test_generator.py \
    --selected selected_context.jsonl \
    --queries queries.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --prompt-style cot --n 50 --load-in-4bit
"""
from __future__ import annotations
import argparse, json, re, sys, time
from collections import defaultdict
from pathlib import Path

_TOK = re.compile(r"\b[a-zA-Z][a-zA-Z\-']{2,}\b")
_STOP = set("a an the of in on at to for from with by is are was were be been "
            "being and or but if then so than that this these those it its as "
            "which who whom whose what when where why how".split())
_FINAL_RE = re.compile(r"Final answer:\s*([^\n]+)", re.IGNORECASE)


def _tok(s: str) -> set[str]:
    return {t.lower() for t in _TOK.findall(s) if t.lower() not in _STOP}


def token_f1(pred: str, gold: str) -> float:
    p, g = _tok(pred), _tok(gold)
    if not p or not g:
        return float(pred.strip().lower() == gold.strip().lower())
    common = p & g
    if not common:
        return 0.0
    pr, rc = len(common) / len(p), len(common) / len(g)
    return 2 * pr * rc / (pr + rc)


def extract_final(text: str, style: str) -> str:
    if style in ("cot", "few_shot_cot"):
        m = _FINAL_RE.search(text)
        return m.group(1).strip() if m else text.strip()
    return text.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--prompt-style", default="cot",
                    choices=["plain", "citation", "cot", "few_shot_cot"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("smoke_gen.jsonl"))
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from astro_cs_rag.generation.prompts import assemble

    queries = {q["query_id"]: q for q in
               (json.loads(l) for l in open(args.queries))}

    sel_by_q: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.selected):
        s = json.loads(line)
        sel_by_q[s["query_id"]].append(s)

    qids = list(sel_by_q)[:args.n]
    print(f"Queries: {len(qids)}  model: {args.model}  style: {args.prompt_style}  4bit: {args.load_in_4bit}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    kw: dict = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
    if args.load_in_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

    lm = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    lm.eval()
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded. GPU allocated: {allocated:.1f} GB\n")

    f1s, lats, extr_fails = [], [], 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(args.out, "w")

    for idx, qid in enumerate(qids):
        q = queries.get(qid)
        if q is None:
            continue
        sel = sel_by_q[qid]
        evidence = [(s.get("chunk_id") or s.get("atom_id", ""), s["text"]) for s in sel]

        prompt = assemble(query=q.get("query") or q.get("text", ""), evidence=evidence, style=args.prompt_style)
        msgs = [{"role": "system", "content": prompt.system},
                {"role": "user",   "content": prompt.user}]
        try:
            inp = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            inp = f"{prompt.system}\n\n{prompt.user}\n"

        enc = tokenizer(inp, return_tensors="pt", truncation=True, max_length=6144)
        ids = enc["input_ids"].to(lm.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            out = lm.generate(ids, max_new_tokens=256, do_sample=False,
                              pad_token_id=tokenizer.pad_token_id)
        lat = time.perf_counter() - t0
        lats.append(lat)

        raw = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        answer = extract_final(raw, args.prompt_style)

        if args.prompt_style in ("cot", "few_shot_cot") and not _FINAL_RE.search(raw):
            extr_fails += 1

        gold = q.get("metadata", {}).get("answer") or ""
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        score = token_f1(answer, str(gold))
        f1s.append(score)

        row = {"query_id": qid, "answer": answer, "f1": score,
               "latency": lat, "raw": raw[:400]}
        out_fh.write(json.dumps(row) + "\n")
        print(f"  [{idx+1:3d}/{len(qids)}] F1={score:.3f}  lat={lat:.1f}s  {answer[:70]}")

    out_fh.close()
    n = len(f1s)
    print(f"\n=== Summary ({n} queries) ===")
    print(f"  Mean F1:          {sum(f1s)/n:.4f}")
    print(f"  F1 >= 0.5:        {sum(s>=0.5 for s in f1s)/n:.1%}")
    print(f"  Extraction fails: {extr_fails} ({extr_fails/n:.1%})")
    print(f"  Mean latency:     {sum(lats)/n:.1f}s/query")
    print(f"  Saved to:         {args.out}")


if __name__ == "__main__":
    main()
