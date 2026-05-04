"""Pre-generate paraphrases for a queries.jsonl file, cache to disk.

Usage:
    python scripts/cache_paraphrases.py \
        --queries data/hotpotqa_1k/queries.jsonl \
        --out data/hotpotqa_1k/paraphrases_qwen7b_n4.jsonl \
        --n 4

Output schema (one JSON per line):
    {"query_id": "...", "paraphrases": ["original", "alt1", "alt2", "alt3"]}

Re-runs are idempotent: existing query_ids in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from astro_cs_rag.config.schema import GeneratorSettings  # noqa: E402
from astro_cs_rag.generation.paraphrase import generate_paraphrases  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=4, help="paraphrase count INCLUDING the original")
    ap.add_argument("--provider", choices=["transformers", "ollama"], default="transformers")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    settings = GeneratorSettings(
        enabled=True,
        provider=args.provider,
        model_name=args.model,
        hf_model_id=args.model,
        temperature=args.temperature,
        max_tokens=256,
        prompt_style="plain",
    )

    done: set[str] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if not line.strip():
                continue
            try:
                done.add(json.loads(line)["query_id"])
            except Exception:
                continue
        print(f"resume: {len(done)} queries already cached")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    f = args.out.open("a", encoding="utf-8")
    n_new = 0
    for line in args.queries.read_text().splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        qid = q["query_id"]
        if qid in done:
            continue
        text = q["text"]
        try:
            paras = generate_paraphrases(text, n=args.n, settings=settings)
        except Exception as e:
            print(f"  fail {qid}: {e}", file=sys.stderr)
            paras = [text]
        rec = {"query_id": qid, "paraphrases": paras}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        n_new += 1
        if n_new % 25 == 0:
            print(f"  cached {n_new}/{len(done) + n_new}")
    f.close()
    print(f"done: wrote {n_new} new entries to {args.out}")


if __name__ == "__main__":
    main()
