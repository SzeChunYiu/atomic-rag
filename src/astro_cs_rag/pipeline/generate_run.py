"""Generate answers from selected evidence; write generated_answers.jsonl."""

from __future__ import annotations

import time
from pathlib import Path

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.atoms.schemas import GeneratedAnswer
from astro_cs_rag.config.schema import GenerateConfig
from astro_cs_rag.data.loaders import load_queries_jsonl
from astro_cs_rag.generation.generator import OllamaGenerator, build_generator
from astro_cs_rag.generation.ollama_client import OllamaClient
from astro_cs_rag.indexing.io import load_chunks_jsonl


def generate_run(cfg: GenerateConfig) -> Path:
    if not cfg.generator.enabled:
        return cfg.run_dir
    chunks = load_chunks_jsonl(cfg.index_dir / "chunks.jsonl")
    text_by_chunk = {c.chunk_id: c.text for c in chunks}
    queries = {q.query_id: q.text for q in load_queries_jsonl(cfg.queries_path)}

    selected_path = cfg.run_dir / "selected_context.jsonl"
    if not selected_path.is_file():
        msg = f"missing selected_context.jsonl in {cfg.run_dir}"
        raise FileNotFoundError(msg)

    by_q: dict[str, list[tuple[str, str]]] = {}
    for line in selected_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        import json

        row = json.loads(line)
        qid = str(row["query_id"])
        cid = str(row["chunk_id"])
        text = text_by_chunk.get(cid, "")
        by_q.setdefault(qid, []).append((cid, text))

    gen = build_generator(cfg.generator)
    writer = ArtifactWriter.attach(cfg.run_dir)

    answers: list[GeneratedAnswer] = []
    t0 = time.perf_counter()
    for qid, qt in queries.items():
        evidence = by_q.get(qid, [])
        answers.append(gen.answer(query_id=qid, query_text=qt, evidence=evidence))
    total = time.perf_counter() - t0

    writer.write_jsonl("generated_answers.jsonl", answers)
    show_meta = {}
    if isinstance(gen, OllamaGenerator):
        client: OllamaClient = gen.client
        show_meta = client.show(cfg.generator.model_name)
    writer.write_json(
        "generation_meta.json",
        {
            "provider": gen.provider,
            "model": gen.model,
            "n_queries": len(answers),
            "wall_seconds": total,
            "ollama_show": show_meta,
        },
    )
    return cfg.run_dir
