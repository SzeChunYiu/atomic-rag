"""CLI: rag-error-analysis --config configs/error_analysis.yaml"""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import ErrorAnalysisConfig
from astro_cs_rag.pipeline.error_analysis_run import error_analysis_run


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
    cfg = load_yaml(config, ErrorAnalysisConfig)
    out = error_analysis_run(cfg)
    typer.echo(str(out.resolve()))


if __name__ == "__main__":
    main()
