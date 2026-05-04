"""Generate hotpotqa_1k chunk-size × selector configs for the linchpin sweep.

Tests whether the synthetic IRC advantage transfers to real data, by sweeping
chunk_size and showing anti-$k_T$ outperforms greedy where boundary effects
matter (small chunks) and matches at large chunks (no regression).
"""
from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "configs/benchmarks/matrix/hotpotqa_1k__dense__greedy.yaml"
OUT_DIR = REPO / "configs/benchmarks/chunksize_sweep"


def main() -> None:
    base = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_sizes = [64, 128, 256, 384, 512, 768]
    selectors = ["greedy", "anti_kt", "mmr"]
    paths = []
    for cs in chunk_sizes:
        for sel in selectors:
            cfg = yaml.safe_load(yaml.safe_dump(base))
            tag = f"hotpotqa_1k_cs{cs}_{sel}"
            cfg["dataset"] = tag
            cfg["chunk_size"] = int(cs)
            cfg["chunk_overlap"] = max(8, cs // 8)
            cfg["paths"]["output_dir"] = f"runs/chunksize_sweep/{tag}"
            cfg["selector"]["mode"] = sel
            cfg["selector"]["token_budget"] = 1024
            cfg["selector"]["anti_kt_n_jets"] = -1  # v2 default
            cfg["generator"] = {"enabled": True, "provider": "stub", "model_name": "stub", "temperature": 0.0}
            out = OUT_DIR / f"{tag}.yaml"
            out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            paths.append(out.relative_to(REPO))

    manifest = OUT_DIR / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in paths) + "\n", encoding="utf-8")
    print(f"wrote {len(paths)} configs to {OUT_DIR}")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
