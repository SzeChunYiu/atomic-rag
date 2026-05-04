"""CLI: rag-detect --config configs/detect.yaml"""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.loader import load_yaml
from astro_cs_rag.config.schema import DetectConfig
from astro_cs_rag.pipeline.detect_run import detect_run


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
    cfg = load_yaml(config, DetectConfig)
    out = detect_run(cfg)
    typer.echo(str(out.resolve()))


if __name__ == "__main__":
    main()
