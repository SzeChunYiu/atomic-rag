"""CLI: rag-build-index --config configs/index.yaml"""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import IndexConfig
from astro_cs_rag.pipeline.index_build import build_index_bundle


def main() -> None:
    typer.run(_run)


def _run(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        readable=True,
        help="YAML index configuration",
    ),
) -> None:
    cfg = load_yaml(config, IndexConfig)
    out = build_index_bundle(cfg)
    typer.echo(str(out.resolve()))


if __name__ == "__main__":
    main()
