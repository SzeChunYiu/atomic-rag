"""CLI: rag-decompose — write claim_atoms.jsonl next to a built index."""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.pipeline.decompose_run import decompose_run


def main() -> None:
    typer.run(_run)


def _run(
    index_dir: Path = typer.Option(..., "--index-dir", exists=True, file_okay=False),
) -> None:
    out = decompose_run(index_dir)
    typer.echo(str(out.resolve()))


if __name__ == "__main__":
    main()
