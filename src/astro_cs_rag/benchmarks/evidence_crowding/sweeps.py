"""Sweep grid for the evidence-crowding benchmark.

Phase-1 grid (matches `crowding_benchmark.md`):
  hop_count=2, chunk_size=384, chunk_mixing=bridge_buried,
  vary n_distractors_per_gold ∈ {0,2,5,10,20,50},
       semantic_similarity ∈ {low,medium,high},
       token_budget ∈ {128,256,512,1024}.

Each cell holds 50 queries by default. The generator is deterministic
under the cell seed; cell seeds are derived from the grid coordinates.
"""

from __future__ import annotations

from collections.abc import Iterator

from .schema import CrowdingCell


def phase1_grid(seed_base: int = 1000) -> Iterator[CrowdingCell]:
    n_distractors_levels = [0, 2, 5, 10, 20, 50]
    sim_levels = ["low", "medium", "high"]
    budgets = [128, 256, 512, 1024]
    idx = 0
    for nd in n_distractors_levels:
        for sim in sim_levels:
            for tb in budgets:
                cell_id = f"p1_nd{nd}_{sim}_tb{tb}"
                yield CrowdingCell(
                    cell_id=cell_id,
                    n_distractors_per_gold=nd,
                    semantic_similarity=sim,  # type: ignore[arg-type]
                    entity_overlap="partial",
                    answer_type_overlap=True,
                    chunk_size=384,
                    chunk_mixing="bridge_buried",
                    hop_count=2,
                    token_budget=tb,
                    seed=seed_base + idx,
                )
                idx += 1


def custom_grid(
    n_distractors: list[int],
    similarities: list[str],
    token_budgets: list[int],
    chunk_size: int = 384,
    chunk_mixing: str = "bridge_buried",
    hop_count: int = 2,
    seed_base: int = 1000,
    tag: str = "custom",
) -> Iterator[CrowdingCell]:
    """Parametric sweep grid — used by the self-driving study loop."""
    idx = 0
    for nd in n_distractors:
        for sim in similarities:
            for tb in token_budgets:
                yield CrowdingCell(
                    cell_id=f"{tag}_nd{nd}_{sim}_tb{tb}",
                    n_distractors_per_gold=nd,
                    semantic_similarity=sim,  # type: ignore[arg-type]
                    entity_overlap="partial",
                    answer_type_overlap=True,
                    chunk_size=chunk_size,
                    chunk_mixing=chunk_mixing,  # type: ignore[arg-type]
                    hop_count=hop_count,
                    token_budget=tb,
                    seed=seed_base + idx,
                )
                idx += 1


def smoke_grid(seed_base: int = 100) -> Iterator[CrowdingCell]:
    """Tiny grid for laptop tests — 4 cells, varies one axis only."""
    for i, nd in enumerate([0, 5, 20]):
        yield CrowdingCell(
            cell_id=f"smoke_nd{nd}",
            n_distractors_per_gold=nd,
            semantic_similarity="medium",
            entity_overlap="partial",
            answer_type_overlap=True,
            chunk_size=384,
            chunk_mixing="bridge_buried",
            hop_count=2,
            token_budget=512,
            seed=seed_base + i,
        )
