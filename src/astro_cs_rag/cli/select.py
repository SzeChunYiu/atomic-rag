"""CLI: rag-select --config configs/select.yaml"""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import SelectConfig
from astro_cs_rag.pipeline.select_run import select_run


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
    cfg = load_yaml(config, SelectConfig)
    out = select_run(cfg)
    typer.echo(str(out.resolve()))


if __name__ == "__main__":
    main()
