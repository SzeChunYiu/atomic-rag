"""Typed pipeline records — no raw dicts across module boundaries."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Query(BaseModel):
    query_id: str
    text: str
    gold_doc_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    query_id: str
    chunk_id: str
    raw_score: float
    retriever: str
    rank: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceAtom(BaseModel):
    query_id: str
    chunk_id: str
    raw_score: float
    bg_mean: float
    bg_std: float
    snr: float
    detector_rank: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimAtom(BaseModel):
    """A minimal evidential unit extracted from a chunk.

    A ClaimAtom is the substrate on which all P3+ methods operate (anti-kT
    clustering, conservation-law balance, lock-in coherent rerank). It is
    intentionally narrower than EvidenceAtom (which lives at chunk level).
    """

    atom_id: str
    chunk_id: str
    doc_id: str
    sent_index: int
    span_start: int
    span_end: int
    text: str
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GeneratedAnswer(BaseModel):
    query_id: str
    answer_text: str
    cited_chunk_ids: list[str] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)
    prompt_tokens_estimate: int = 0
    completion_tokens_estimate: int = 0
    latency_seconds: float = 0.0
    provider: str = "stub"
    model: str = "none"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricRecord(BaseModel):
    name: str
    value: float
    extra: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    run_id: str
    dataset_name: str
    corpus_doc_count: int
    query_count: int
    chunk_count: int
    seed: int
    paths: dict[str, str] = Field(default_factory=dict)
