"""CLI: rag-build-asimov — synthesize an Asimov benchmark from a QA + distractors file.

Input:
- qa_jsonl  : lines of {query, gold_text, answer} (answer can be a string or list).
- distractors_jsonl : lines with a `text` field; passages disjoint from any gold.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from astro_cs_rag.benchmarks.asimov import synthesize_asimov, write_asimov_jsonl


def main() -> None:
    typer.run(_run)


def _run(
    qa: Path = typer.Option(..., "--qa", exists=True, dir_okay=False),
    distractors: Path = typer.Option(..., "--distractors", exists=True, dir_okay=False),
    out: Path = typer.Option(..., "--out"),
    pool_size: int = typer.Option(50, "--pool-size"),
    seed: int = typer.Option(0, "--seed"),
    name: str = typer.Option("asimov", "--name"),
    position: str = typer.Option("uniform", "--position",
                                 help="first|middle|last|uniform"),
) -> None:
    qa_pairs = []
    for line in qa.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ans = row.get("answer") or []
        if isinstance(ans, str):
            ans = [ans]
        qa_pairs.append((str(row["query"]), str(row["gold_text"]), [str(a) for a in ans]))

    pool = []
    for line in distractors.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        pool.append(str(row["text"]))

    bench = synthesize_asimov(
        qa_pairs=qa_pairs,
        distractor_pool=pool,
        pool_size=pool_size,
        seed=seed,
        name=name,
        position_strategy=position,
    )
    paths = write_asimov_jsonl(bench, out, pool)
    typer.echo(json.dumps({k: str(v) for k, v in paths.items()}))


if __name__ == "__main__":
    main()
