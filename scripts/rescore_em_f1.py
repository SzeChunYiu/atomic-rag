"""Re-score EM and F1 for all existing runs using the fixed citation-stripping normalizer.

Reads generated_answers.jsonl + queries.jsonl for each run and rewrites
answer_em_mean / answer_f1_mean in metrics.json without re-running generation.

Usage:
    python scripts/rescore_em_f1.py --runs-root runs/
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

_CITE_RE = re.compile(r"\[E\d+\]")
_FINAL_ANS_RE = re.compile(r"Final answer:\s*([^\n]+)", re.IGNORECASE)
_PUNC = re.compile(rf"[{re.escape(string.punctuation)}]")
_ART = re.compile(r"\b(a|an|the)\b")
_WS = re.compile(r"\s+")

DATASET_QUERY_PATHS = {
    "hotpotqa_1k": "data/hotpotqa_1k/queries.jsonl",
    "2wiki_1k": "data/2wiki_1k/queries.jsonl",
    "nq_open_1k": "data/nq_open_1k/queries.jsonl",
}


def _norm(s: str) -> str:
    s = _PUNC.sub(" ", s.lower())
    s = _ART.sub(" ", s)
    return _WS.sub(" ", s).strip()


def _extract(s: str) -> str:
    m = _FINAL_ANS_RE.search(s)
    return _norm(_CITE_RE.sub("", m.group(1) if m else s).strip())


def _norm_clean(s: str) -> str:
    return _extract(s)


def _f1(pred: str, refs: list[str]) -> float:
    pt = _norm_clean(pred).split()
    if not pt:
        return 0.0
    best = 0.0
    for r in refs:
        rt = _norm(r).split()
        if not rt:
            continue
        common = Counter(pt) & Counter(rt)
        nc = sum(common.values())
        if nc == 0:
            continue
        p = nc / len(pt)
        rec = nc / len(rt)
        best = max(best, 2 * p * rec / (p + rec))
    return best


def _em(pred: str, refs: list[str]) -> float:
    pc = _norm_clean(pred)
    return float(any(pc == _norm(r) for r in refs))


def load_query_gold(queries_path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        ans = r.get("metadata", {}).get("answer", "")
        if isinstance(ans, str):
            ans = [ans] if ans else []
        out[r["query_id"]] = ans
    return out


def rescore_run(run_dir: Path, gold_map: dict[str, list[str]]) -> dict[str, float] | None:
    ans_path = run_dir / "generated_answers.jsonl"
    if not ans_path.is_file():
        return None
    em_sum = f1_sum = 0.0
    n = 0
    for line in ans_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        refs = gold_map.get(row["query_id"], [])
        pred = row.get("answer_text", "")
        em_sum += _em(pred, refs)
        f1_sum += _f1(pred, refs)
        n += 1
    if n == 0:
        return None
    return {"answer_em_mean": em_sum / n, "answer_f1_mean": f1_sum / n, "answer_count": float(n)}


def find_dataset(run_dir: Path, root: Path) -> str | None:
    cfg = run_dir / "config.yaml"
    if cfg.is_file():
        text = cfg.read_text(encoding="utf-8")
        for ds in DATASET_QUERY_PATHS:
            if ds in text:
                return ds
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = root / runs_root

    gold_cache: dict[str, dict[str, list[str]]] = {}

    updated = skipped = 0
    for ans_file in sorted(runs_root.rglob("generated_answers.jsonl")):
        run_dir = ans_file.parent
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            skipped += 1
            continue

        ds = find_dataset(run_dir, root)
        if ds is None:
            skipped += 1
            continue

        if ds not in gold_cache:
            qp = root / DATASET_QUERY_PATHS[ds]
            if not qp.is_file():
                skipped += 1
                continue
            gold_cache[ds] = load_query_gold(qp)

        new_scores = rescore_run(run_dir, gold_cache[ds])
        if new_scores is None:
            skipped += 1
            continue

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        old_em = metrics.get("answer_em_mean", "—")
        old_f1 = metrics.get("answer_f1_mean", "—")
        metrics.update(new_scores)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(
            f"{run_dir.relative_to(runs_root)}  "
            f"EM: {old_em:.4f}→{new_scores['answer_em_mean']:.4f}  "
            f"F1: {old_f1:.4f}→{new_scores['answer_f1_mean']:.4f}"
        )
        updated += 1

    print(f"\nDone. Updated={updated}  Skipped={skipped}")


if __name__ == "__main__":
    main()
