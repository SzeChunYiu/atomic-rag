"""CLI: rag-profile — calorimetric query archetype profile over a run."""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.pipeline.profile_run import profile_run


def main() -> None:
    typer.run(_run)


def _run(
    run_dir: Path = typer.Option(..., "--run-dir", exists=True, file_okay=False),
    gold_path: Path | None = typer.Option(None, "--gold-path", exists=True, dir_okay=False),
    ks: list[int] = typer.Option([1, 5, 10], "--k"),
) -> None:
    out = profile_run(run_dir, gold_path=gold_path, ks=tuple(ks))
    typer.echo(str(out.resolve()))


if __name__ == "__main__":
    main()
