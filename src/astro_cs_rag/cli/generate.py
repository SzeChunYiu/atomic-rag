"""CLI: rag-generate — produce answers for a completed select run."""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import GenerateConfig
from astro_cs_rag.pipeline.generate_run import generate_run


def main() -> None:
    typer.run(_run)


def _run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
) -> None:
    cfg = load_yaml(config, GenerateConfig)
    out = generate_run(cfg)
    typer.echo(str(out.resolve()))


if __name__ == "__main__":
    main()
