"""Materialize claim atoms over a chunk index — written next to chunks.jsonl."""

from __future__ import annotations

from pathlib import Path

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.atoms.decomposer import decompose_chunks
from astro_cs_rag.indexing.io import load_chunks_jsonl


def decompose_run(index_dir: Path) -> Path:
    chunks = load_chunks_jsonl(index_dir / "chunks.jsonl")
    atoms = decompose_chunks(chunks)
    out = ArtifactWriter.attach(index_dir)
    out.write_jsonl("claim_atoms.jsonl", atoms)
    out.write_json(
        "decompose_meta.json",
        {
            "atom_count": len(atoms),
            "chunk_count": len(chunks),
            "atoms_per_chunk_mean": (len(atoms) / len(chunks)) if chunks else 0.0,
            "backend": "regex_v1",
        },
    )
    return index_dir
