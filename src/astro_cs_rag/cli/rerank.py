"""CLI: rag-rerank — apply a cross-encoder reranker to an existing retrieve run."""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import RerankConfig
from astro_cs_rag.pipeline.rerank_run import rerank_run


def main() -> None:
    typer.run(_run)


def _run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
) -> None:
    cfg = load_yaml(config, RerankConfig)
    out_dir = rerank_run(cfg)
    typer.echo(str(out_dir.resolve()))


if __name__ == "__main__":
    main()
