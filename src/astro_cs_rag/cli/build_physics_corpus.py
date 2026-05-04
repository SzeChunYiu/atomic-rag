"""CLI: rag-build-physics-corpus — fetch arXiv subset + (optional) QA synthesis."""

from __future__ import annotations

from pathlib import Path

import typer

from astro_cs_rag.config.schema import GeneratorSettings
from astro_cs_rag.data.arxiv_loaders import fetch_arxiv_subset, load_corpus_for_qa
from astro_cs_rag.data.qa_synth import synthesize_qa, write_qa_jsonl


def main() -> None:
    typer.run(_run)


def _run(
    out: Path = typer.Option(..., "--out"),
    set_spec: str = typer.Option("physics:astro-ph", "--set", help="OAI set spec"),
    n_docs: int = typer.Option(5000, "--n-docs"),
    seed: int = typer.Option(0, "--seed"),
    do_qa: bool = typer.Option(True, "--qa/--no-qa"),
    qa_target_n: int = typer.Option(150, "--qa-n"),
    ollama_model: str = typer.Option("llama3.1:8b-instruct-q4_K_M", "--ollama-model"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url"),
) -> None:
    sub = fetch_arxiv_subset(out_dir=out, set_spec=set_spec, n_docs=n_docs, seed=seed)
    typer.echo(f"corpus: {sub.corpus_path} (n={sub.n_docs})")
    if not do_qa:
        return
    rows = load_corpus_for_qa(sub.corpus_path)
    settings = GeneratorSettings(
        enabled=True,
        provider="ollama",
        model_name=ollama_model,
        base_url=ollama_url,
        temperature=0.0,
        seed=seed,
        max_tokens=512,
        prompt_style="plain",
    )
    items = synthesize_qa(
        rows, generator_settings=settings, target_n=qa_target_n, seed=seed
    )
    paths = write_qa_jsonl(items, out)
    typer.echo(f"qa: {paths['queries']} (n={len(items)})")


if __name__ == "__main__":
    main()
