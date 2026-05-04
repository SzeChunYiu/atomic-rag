"""SNR detection pass over retrieval candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from astro_cs_rag.artifacts.writer import ArtifactWriter
from astro_cs_rag.atoms.schemas import EvidenceAtom
from astro_cs_rag.config.schema import DetectConfig
from astro_cs_rag.detection.aperture import aperture_snr
from astro_cs_rag.detection.snr import detect_evidence
from astro_cs_rag.indexing.dense import DenseIndex
from astro_cs_rag.indexing.io import load_chunks_jsonl
from astro_cs_rag.pipeline.retrieve_run import load_candidates_jsonl


def detect_run(cfg: DetectConfig) -> Path:
    candidates = load_candidates_jsonl(cfg.run_dir / "candidates.jsonl")
    by_q: dict[str, list] = defaultdict(list)
    for c in candidates:
        by_q[c.query_id].append(c)

    chunk_texts: dict[str, str] | None = None
    if cfg.detector.length_normalize_snr:
        if cfg.index_dir is None:
            msg = "DetectConfig.index_dir required when length_normalize_snr is true"
            raise ValueError(msg)
        chs = load_chunks_jsonl(cfg.index_dir / "chunks.jsonl")
        chunk_texts = {c.chunk_id: c.text for c in chs}

    embeddings_by_id: dict = {}
    if cfg.detector.background_mode == "aperture" and cfg.index_dir is not None:
        try:
            dense = DenseIndex.load(cfg.index_dir)
            for cid, vec in zip(dense.chunk_ids, dense.embeddings, strict=True):
                embeddings_by_id[cid] = vec
        except FileNotFoundError:
            pass

    atoms: list[EvidenceAtom] = []
    for qid, parts in by_q.items():
        scores = {p.chunk_id: p.raw_score for p in parts}
        if cfg.detector.background_mode == "aperture" and embeddings_by_id:
            cids = [p.chunk_id for p in parts if p.chunk_id in embeddings_by_id]
            atoms.extend(
                aperture_snr(
                    query_id=qid,
                    candidate_chunk_ids=cids,
                    candidate_scores={c: scores[c] for c in cids},
                    embeddings_by_id=embeddings_by_id,
                    radius_in=cfg.detector.aperture_radius_in,
                    radius_out=cfg.detector.aperture_radius_out,
                    snr_threshold=cfg.detector.threshold,
                )
            )
        else:
            atoms.extend(
                detect_evidence(
                    qid,
                    scores,
                    window=cfg.detector.window,
                    snr_threshold=cfg.detector.threshold,
                    background_mode=cfg.detector.background_mode,  # type: ignore[arg-type]
                    chunk_texts=chunk_texts,
                    length_normalize_snr=cfg.detector.length_normalize_snr,
                )
            )

    writer = ArtifactWriter.attach(cfg.run_dir)
    writer.write_jsonl("evidence_atoms.jsonl", atoms)
    return cfg.run_dir


def load_evidence_atoms_jsonl(path: Path) -> list[EvidenceAtom]:
    rows: list[EvidenceAtom] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(EvidenceAtom.model_validate_json(line))
    return rows
