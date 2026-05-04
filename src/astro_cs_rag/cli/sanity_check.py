"""CLI: rag-sanity-check --config configs/sanity.yaml"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import SanityConfig
from astro_cs_rag.diagnostics.sanity import load_run_checks, sanity_report_payload


def main() -> None:
    typer.run(_run)


def _run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
    ),
) -> None:
    cfg = load_yaml(config, SanityConfig)
    payload = load_run_checks(cfg.run_dir)
    out_path = cfg.run_dir / "sanity_checks.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    typer.echo(sanity_report_payload(payload))


if __name__ == "__main__":
    main()
