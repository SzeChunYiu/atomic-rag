"""Pydantic schemas for the evidence-crowding benchmark."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DistractorClass = Literal["entity_overlap", "type_overlap", "semantic", "noise"]
SimilarityBin = Literal["low", "medium", "high"]
EntityOverlapBin = Literal["none", "partial", "high"]
ChunkMixing = Literal["gold_isolated", "gold_with_distractors", "bridge_buried"]


class CrowdingAtom(BaseModel):
    atom_id: str
    chunk_id: str
    doc_id: str
    text: str
    claim_type: str
    is_gold: bool = False
    role: str = ""  # hop1 / hop2 / answer / distractor / noise
    distractor_class: DistractorClass | None = None
    entities: list[str] = Field(default_factory=list)


class CrowdingChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    atom_ids: list[str]


class CrowdingQuery(BaseModel):
    query_id: str
    text: str
    answer: str
    answer_aliases: list[str]
    gold_atom_ids: list[str]
    gold_doc_ids: list[str]
    hop_count: int
    template: str


class CrowdingCell(BaseModel):
    """One sweep cell — a corpus + queries + control variables."""

    cell_id: str
    n_distractors_per_gold: int
    semantic_similarity: SimilarityBin
    entity_overlap: EntityOverlapBin
    answer_type_overlap: bool
    chunk_size: int
    chunk_mixing: ChunkMixing
    hop_count: int
    token_budget: int
    seed: int


class CrowdingDataset(BaseModel):
    cell: CrowdingCell
    atoms: list[CrowdingAtom]
    chunks: list[CrowdingChunk]
    queries: list[CrowdingQuery]


class CrowdingResult(BaseModel):
    """One row of `crowding_sweep_results.jsonl`."""

    system_name: str
    cell_id: str
    query_id: str
    n_distractors_per_gold: int
    semantic_similarity: SimilarityBin
    entity_overlap: EntityOverlapBin
    chunk_size: int
    hop_count: int
    token_budget: int
    gold_doc_recall_at_k: float = 0.0
    gold_atom_recall_at_k: float = 0.0
    gold_atom_selected: float = 0.0
    support_chain_complete: bool = False
    answer_oracle_success: bool = False
    selected_tokens: int = 0
    latency_ms: float = 0.0
