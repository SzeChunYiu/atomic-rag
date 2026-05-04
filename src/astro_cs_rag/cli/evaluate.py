"""CLI: rag-evaluate --config configs/eval.yaml"""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import EvaluateConfig
from astro_cs_rag.pipeline.evaluate_run import evaluate_run


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
    cfg = load_yaml(config, EvaluateConfig)
    out = evaluate_run(cfg)
    typer.echo(str(out.resolve()))


if __name__ == "__main__":
    main()
