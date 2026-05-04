"""Read all metrics.json under runs/realgen_d06_sweep/ and emit
saturation/optimum curves per swept parameter.

Output: change_log/d04d06_sweep_curves.md with a table per sweep axis."""
from __future__ import annotations
import json, re
from pathlib import Path

ROOT = Path("runs/realgen_d06_sweep")
PAT = re.compile(r"^(lambda|budget|topk|floor|bonus)_(.+)$")
LABEL = {"lambda": "lambda_type", "budget": "token_budget",
         "topk": "top_k", "floor": "score_floor", "bonus": "score_bonus"}


def main():
    rows = {}
    for d in sorted(ROOT.iterdir()):
        m = PAT.match(d.name)
        if not m: continue
        axis, val = m.group(1), m.group(2)
        mfile = d / "metrics.json"
        if not mfile.exists(): continue
        m_obj = json.loads(mfile.read_text())
        rows.setdefault(axis, []).append({
            "value": float(val) if "." in val or axis in ("lambda", "floor", "bonus")
                     else int(val),
            "F1": m_obj.get("F1", 0.0),
            "EM": m_obj.get("EM", 0.0),
            "cit": m_obj.get("cit_acc", 0.0),
            "n": m_obj.get("n", 0),
        })
    out = ["# D04+D06 hyperparameter sweep curves\n"]
    for axis, items in rows.items():
        items.sort(key=lambda x: x["value"])
        if not items: continue
        best = max(items, key=lambda x: x["F1"])
        out.append(f"\n## {LABEL[axis]} (best F1={best['F1']:.4f} at "
                   f"{LABEL[axis]}={best['value']})\n")
        out.append("| value | F1 | EM | cit_acc | n |")
        out.append("|---|---|---|---|---|")
        for it in items:
            mk = " **best**" if it is best else ""
            out.append(f"| {it['value']}{mk} | {it['F1']:.4f} | {it['EM']:.4f} "
                       f"| {it['cit']:.4f} | {it['n']} |")
    text = "\n".join(out) + "\n"
    Path("change_log/d04d06_sweep_curves.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
