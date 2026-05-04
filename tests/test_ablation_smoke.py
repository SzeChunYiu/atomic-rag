from __future__ import annotations

from pathlib import Path

import yaml

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import AblationConfig
from astro_cs_rag.pipeline.ablation_run import ablation_run


def test_ablation_runs_two_variants(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    base_path = repo / "configs" / "benchmark.yaml"
    cfg_path = tmp_path / "abl.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "base_config_path": str(base_path),
                "output_dir": str(tmp_path / "abl_out"),
                "variants": [
                    {"name": "v1", "overrides": {"retriever": {"candidate_top_n": 8}}},
                    {"name": "v2", "overrides": {"selector": {"token_budget": 200}}},
                ],
            }
        ),
        encoding="utf-8",
    )
    root = ablation_run(load_yaml(cfg_path, AblationConfig))
    assert (root / "ablation_results.jsonl").is_file()
    assert (root / "ablation_report.md").is_file()
    text = (root / "ablation_results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(text) == 2
