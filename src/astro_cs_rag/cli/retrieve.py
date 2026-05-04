"""CLI: rag-retrieve --config configs/retrieve.yaml"""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import RetrieveConfig
from astro_cs_rag.pipeline.retrieve_run import retrieve_run


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
    cfg = load_yaml(config, RetrieveConfig)
    run_dir = retrieve_run(cfg)
    typer.echo(str(run_dir.resolve()))


if __name__ == "__main__":
    main()
