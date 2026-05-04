"""CLI: rag-prepare-data — materialize a frozen subset of an HF benchmark."""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.data.hf_loaders import (
    prepare_2wikimultihop,
    prepare_hotpotqa,
    prepare_nq_open,
)


def main() -> None:
    typer.run(_run)


def _run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="hotpotqa | nq_open | 2wikimultihop"),
    split: str = typer.Option("validation", "--split", "-s"),
    n: int = typer.Option(1000, "--n", "-n"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(..., "--out", "-o"),
    wiki_passages: Path | None = typer.Option(
        None,
        "--wiki-passages",
        help="optional JSONL of Wikipedia passages for nq_open real corpus",
    ),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    if dataset == "hotpotqa":
        sub = prepare_hotpotqa(out_dir=out, split=split, n_queries=n, seed=seed)
    elif dataset == "nq_open":
        sub = prepare_nq_open(
            out_dir=out,
            split=split,
            n_queries=n,
            seed=seed,
            wiki_passages=wiki_passages,
        )
    elif dataset == "2wikimultihop":
        sub = prepare_2wikimultihop(
            out_dir=out, split=split, n_queries=n, seed=seed
        )
    else:
        raise typer.BadParameter(f"unknown dataset {dataset}")
    typer.echo(str(sub.out_dir.resolve()))


if __name__ == "__main__":
    main()
